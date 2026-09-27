"""
Сборка признаков для детекции ботов (уровень cookie_id).

Главная функция — build_features(events, meta): одинаково применяется к train и test
и возвращает по одной строке на каждую куку из meta (в том же порядке).

Все признаки считаются только по событиям внутри окна наблюдения
window_start_ts <= event_ts < window_end_ts, то есть доступны на момент окончания окна.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- константы

DATE_COLS = ['cookie_created_at', 'window_start_ts', 'window_end_ts']

SESSION_GAP_S = 30 * 60      # пауза длиннее 30 минут начинает новую сессию
FAST_DT_S = 10               # «быстрый» интервал между действиями, сек
METRONOME_TOL = 0.2          # интервал считается «как медиана», если отличается от неё не более чем на 20%
NIGHT_HOURS = (0, 6)         # ночь: 0:00–5:59

PLATFORM_MAP = {'web': 'web', 'desktop': 'web',
                'android': 'android',
                'ios': 'ios', 'iphone': 'ios'}

# типы событий, которые используем; captcha_shown сознательно исключена:
# в train она встречается только после окна (реакция антибота = утечка таргета)
EVENT_NAMES = ['search_results_view', 'item_view', 'photo_swipe', 'seller_page_view',
               'contact_phone_show', 'contact_chat_open', 'contact_message_sent',
               'favorite_add', 'login']
CONTACT_EVENTS = ['contact_phone_show', 'contact_chat_open', 'contact_message_sent']

# грубые семейства клиента; конкретные строки UA и версии не используем —
# в EDA они оказались ненадёжны (боты маскируются, «ботовые» UA бывают у людей)
UA_FAMILIES = ['app', 'mobile_web', 'desktop_browser', 'headless', 'http_lib']
PLATFORMS = ['web', 'android', 'ios']


# ---------------------------------------------------------------- загрузка

def load_data(data_dir: str = 'data') -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Читает train, test и events с разбором дат."""
    train = pd.read_csv(f'{data_dir}/train.csv', parse_dates=DATE_COLS)
    test = pd.read_csv(f'{data_dir}/test.csv', parse_dates=DATE_COLS)
    events = pd.read_csv(f'{data_dir}/events.csv.gz', parse_dates=['event_ts'])
    return train, test, events


# ---------------------------------------------------------------- чистка событий

def ua_family(ua: str) -> str:
    """Грубое семейство клиента по строке User-Agent. Порядок проверок важен."""
    if not isinstance(ua, str):
        return 'unknown'
    if 'Headless' in ua:
        return 'headless'
    if ua.startswith('Avito/'):
        return 'app'
    if ua.startswith('Mozilla/'):
        if 'Android' in ua or 'iPhone' in ua:
            return 'mobile_web'
        return 'desktop_browser'
    return 'http_lib'          # curl, python-requests, Scrapy, Go-http-client, ...


def clean_events(events: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """
    Оставляет события куки из meta внутри их окна, удаляет дубли и сортирует.
    Добавляет служебные колонки: plat, ua_fam, dt, is_new_session, session_id, hour.
    """
    ev = events.drop_duplicates()                                  # полные дубли — технический шум
    ev = ev.merge(meta[['cookie_id', 'window_start_ts', 'window_end_ts']], on='cookie_id', how='inner')
    ev = ev[(ev.event_ts >= ev.window_start_ts) & (ev.event_ts < ev.window_end_ts)]
    ev = ev[ev.event_name.isin(EVENT_NAMES)]                       # без капчи
    ev = ev.sort_values(['cookie_id', 'event_ts', 'eid'], kind='mergesort').reset_index(drop=True)

    ev['plat'] = ev.platform.str.lower().map(PLATFORM_MAP)
    ev['ua_fam'] = ev.user_agent.map(ua_family)

    # интервал до предыдущего события той же куки (у первого события — NaN)
    ev['dt'] = ev.groupby('cookie_id').event_ts.diff().dt.total_seconds()

    # сессии: новая начинается с первого события куки или после паузы > SESSION_GAP_S
    ev['is_new_session'] = ev.dt.isna() | (ev.dt > SESSION_GAP_S)
    ev['session_id'] = ev.groupby('cookie_id').is_new_session.cumsum()

    ev['hour'] = ev.event_ts.dt.hour
    return ev


# ---------------------------------------------------------------- группы признаков
# каждая функция возвращает DataFrame с индексом cookie_id

def _entropy(counts: pd.DataFrame) -> pd.Series:
    """Энтропия Шеннона по строкам таблицы счётчиков (в натах)."""
    p = counts.div(counts.sum(axis=1), axis=0)
    return -(p * np.log(p.where(p > 0))).sum(axis=1)


def feat_cookie_age(ev: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Возраст куки: на начало окна и на момент первого события в окне."""
    m = meta.set_index('cookie_id')
    first_ts = ev.groupby('cookie_id').event_ts.min().reindex(m.index)
    f = pd.DataFrame(index=m.index)
    age_h = (m.window_start_ts - m.cookie_created_at).dt.total_seconds() / 3600
    f['log_age_h'] = np.log1p(age_h.clip(lower=0))
    f['is_fresh_3d'] = (age_h < 72).astype(int)
    f['log_created_to_first_h'] = np.log1p(((first_ts - m.cookie_created_at).dt.total_seconds() / 3600).clip(lower=0))
    f['first_event_hour_in_window'] = (first_ts - m.window_start_ts).dt.total_seconds() / 3600
    return f


def feat_volume_sessions(ev: pd.DataFrame) -> pd.DataFrame:
    """Объём активности и её разбиение на сессии."""
    g = ev.groupby('cookie_id')
    f = pd.DataFrame({'n_events': g.size(),
                      'n_sessions': g.session_id.max()})
    f['active_span_h'] = (g.event_ts.max() - g.event_ts.min()).dt.total_seconds() / 3600

    s = ev.groupby(['cookie_id', 'session_id']).agg(n=('event_ts', 'size'),
                                                     start=('event_ts', 'min'),
                                                     end=('event_ts', 'max'))
    s['dur_s'] = (s.end - s.start).dt.total_seconds()
    sg = s.groupby('cookie_id')
    f['events_per_session_mean'] = sg.n.mean()
    f['events_per_session_max'] = sg.n.max()
    f['log_session_dur_mean'] = np.log1p(sg.dur_s.mean())
    f['log_session_dur_max'] = np.log1p(sg.dur_s.max())
    f['max_session_share'] = f.events_per_session_max / f.n_events   # насколько активность сосредоточена в одной сессии
    return f


def feat_timing(ev: pd.DataFrame) -> pd.DataFrame:
    """
    Темп и ритм действий. Интервалы берём только внутри сессий,
    чтобы паузы между сессиями не смешивались с темпом действий.
    """
    d = ev.loc[ev.dt.notna() & ~ev.is_new_session, ['cookie_id', 'dt']]
    g = d.groupby('cookie_id').dt

    f = pd.DataFrame({'n_dt': g.size()})
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        f[f'log_dt_q{int(q * 100)}'] = np.log1p(g.quantile(q))
    mean, std = g.mean(), g.std()
    f['dt_cv'] = std / mean.replace(0, np.nan)
    f['dt_cv_norm'] = f.dt_cv / np.sqrt(f.n_dt)      # CV ограничен sqrt(n) — нормируем на эту границу

    d = d.assign(med=d.cookie_id.map(g.median()))
    d['fast'] = d.dt < FAST_DT_S
    d['zero'] = d.dt == 0
    d['near_med'] = (d.dt - d.med).abs() <= METRONOME_TOL * d.med   # «метроном»
    dg = d.groupby('cookie_id')
    f['share_fast'] = dg.fast.mean()
    f['share_zero'] = dg.zero.mean()
    f['share_near_median'] = dg.near_med.mean()

    # самая длинная пауза в окне (включая паузы между сессиями)
    f['log_max_gap_s'] = np.log1p(ev.groupby('cookie_id').dt.max())
    return f


def feat_hours(ev: pd.DataFrame) -> pd.DataFrame:
    """Суточный профиль активности куки."""
    counts = pd.crosstab(ev.cookie_id, ev.hour)
    f = pd.DataFrame(index=counts.index)
    lo, hi = NIGHT_HOURS
    f['night_share'] = ev.hour.between(lo, hi - 1).groupby(ev.cookie_id).mean()
    f['n_active_hours'] = (counts > 0).sum(axis=1)
    f['hour_entropy'] = _entropy(counts)
    return f


def feat_event_mix(ev: pd.DataFrame) -> pd.DataFrame:
    """Состав действий: доли типов событий и простые переходы."""
    counts = pd.crosstab(ev.cookie_id, ev.event_name).reindex(columns=EVENT_NAMES, fill_value=0)
    n = counts.sum(axis=1)
    f = counts.div(n, axis=0).add_prefix('share_')
    f['n_contacts'] = counts[CONTACT_EVENTS].sum(axis=1)
    f['has_contact'] = (f.n_contacts > 0).astype(int)
    f['has_login'] = (counts['login'] > 0).astype(int)
    f['has_favorite'] = (counts['favorite_add'] > 0).astype(int)
    f['event_type_entropy'] = _entropy(counts)
    # глубина просмотра объявления: сколько фото листают на один просмотр
    f['photos_per_item_view'] = counts['photo_swipe'] / counts['item_view'].replace(0, np.nan)

    # переходы: из выдачи в объявление и повторы одного и того же типа подряд
    prev = ev.groupby('cookie_id').event_name.shift()
    tr = pd.DataFrame({'cookie_id': ev.cookie_id,
                       'search_to_item': (prev == 'search_results_view') & (ev.event_name == 'item_view'),
                       'same_as_prev': prev == ev.event_name,
                       'has_prev': prev.notna()})
    tg = tr[tr.has_prev].groupby('cookie_id')
    f['share_search_to_item'] = tg.search_to_item.mean()
    f['share_same_as_prev'] = tg.same_as_prev.mean()
    return f


def feat_diversity(ev: pd.DataFrame) -> pd.DataFrame:
    """Разнообразие того, что смотрит кука, и поведение в поиске."""
    g = ev.groupby('cookie_id')
    n = g.size()
    f = pd.DataFrame(index=n.index)
    for col, name in [('item_id', 'items'), ('item_category', 'cats'),
                      ('item_location', 'locs'), ('search_query', 'queries')]:
        nu = g[col].nunique()
        f[f'n_{name}'] = nu
        f[f'{name}_per_event'] = nu / n

    # повторные просмотры: доля событий с объявлением, где объявление уже встречалось
    it = ev[ev.item_id.notna()]
    f['item_repeat_share'] = it.duplicated(['cookie_id', 'item_id']).groupby(it.cookie_id).mean()
    f['pro_seller_share'] = (it.seller_type == 'pro').groupby(it.cookie_id).mean()

    se = ev[ev.event_name == 'search_results_view']
    sg = se.groupby('cookie_id').search_page
    f['search_page_max'] = sg.max()
    f['search_page_mean'] = sg.mean()
    f['search_deep_share'] = (se.search_page > 3).groupby(se.cookie_id).mean()   # доля «глубоких» страниц выдачи
    return f


def feat_cursor(ev: pd.DataFrame) -> pd.DataFrame:
    """Курсор. Существует только на web, поэтому для мобильных куки признаки = NaN."""
    web = ev[ev.plat == 'web']
    f = pd.DataFrame(index=web.cookie_id.unique())
    f['pointer_share'] = web.pointer_x.notna().groupby(web.cookie_id).mean()

    p = web[web.pointer_x.notna()]
    pg = p.groupby('cookie_id')
    f['pointer_x_std'] = pg.pointer_x.std()
    f['pointer_y_std'] = pg.pointer_y.std()
    f['pointer_unique_share'] = p.duplicated(['cookie_id', 'pointer_x', 'pointer_y']).groupby(p.cookie_id).mean().rsub(1)
    f['pointer_origin_share'] = ((p.pointer_x == 0) & (p.pointer_y == 0)).groupby(p.cookie_id).mean()
    return f


def feat_client(ev: pd.DataFrame) -> pd.DataFrame:
    """Клиент: платформа и грубое семейство UA первого события, число разных UA."""
    g = ev.groupby('cookie_id')        # ev уже отсортирован по времени, 'first' = первое событие
    f = pd.DataFrame({'plat': g.plat.first(),
                      'ua_fam': g.ua_fam.first(),
                      'n_user_agents': g.user_agent.nunique()})
    f['plat'] = pd.Categorical(f.plat, categories=PLATFORMS)
    f['ua_fam'] = pd.Categorical(f.ua_fam, categories=UA_FAMILIES)
    return f


# ---------------------------------------------------------------- сборка

def build_features(events: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """
    Признаки для всех куки из meta. Возвращает DataFrame: cookie_id + признаки,
    строки в порядке meta. У куки без событий в окне поведенческие признаки = NaN, n_events = 0.
    """
    ev = clean_events(events, meta)
    parts = [feat_cookie_age(ev, meta),
             feat_volume_sessions(ev),
             feat_timing(ev),
             feat_hours(ev),
             feat_event_mix(ev),
             feat_diversity(ev),
             feat_cursor(ev),
             feat_client(ev)]
    X = pd.concat(parts, axis=1).reindex(meta.cookie_id)
    X['n_events'] = X.n_events.fillna(0)
    X.index.name = 'cookie_id'
    return X.reset_index()


def feature_columns(X: pd.DataFrame) -> list[str]:
    """Все колонки, кроме идентификатора."""
    return [c for c in X.columns if c != 'cookie_id']


if __name__ == '__main__':
    train, test, events = load_data()
    Xtr = build_features(events, train)
    Xte = build_features(events, test)
    print('train:', Xtr.shape, ' test:', Xte.shape)
    print(Xtr.head())
