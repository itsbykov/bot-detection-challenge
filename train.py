"""
Обучение, валидация и сабмит.

    python train.py                  # CV по дням + holdout + обучение на всём train + submission.csv
    python train.py --no-submit      # только CV и holdout (для экспериментов с признаками)
    python train.py --n-seeds 2      # быстрая версия
    python train.py --drop-group cursor --no-submit   # выключить ещё одну группу (см. features.FEATURE_GROUPS)
    python train.py --all-groups     # все 8 групп признаков, включая исключённые по проведенной абляции

Модель — LightGBM, усреднение предсказаний по нескольким сидам
(снижает дисперсию и почти убирает одинаковые score, которые метрика склеивает в группы).
Метрика — официальная precision_at_recall из metric.py.

=====================================================================================
ПРОТОКОЛ ВАЛИДАЦИИ И ПОЧЕМУ ОН ТАКОЙ
=====================================================================================

Используем две оценки с разными ролями.

1. CV С ГРУППИРОВКОЙ ПО ДНЯМ — инструмент для решений (брать ли признак, какие параметры).

   14 дней train делятся на 7 групп по 2 подряд идущих дня. Каждая группа по очереди
   становится валидацией, модель учится на остальных 12 днях. В итоге у каждой куки есть
   out-of-fold (OOF) предсказание модели, которая эту куку не видела, и метрика считается
   один раз по всем ~11 тыс. куки и ~900 ботам.

       фолд 0: обучение [08.04–19.04]              → валидация [06.04–07.04]
       фолд 1: обучение [06.04–07.04, 10.04–19.04] → валидация [08.04–09.04]
       ...
       фолд 6: обучение [06.04–17.04]              → валидация [18.04–19.04]

   Зачем: один holdout (неделя, ~400 ботов) слишком шумный. При precision около 0.7
   и ~400 отмеченных куки стандартная ошибка precision около sqrt(0.7*0.3/400) ≈ 0.023,
   то есть ±2.3 п.п. В разборе ошибок признаки, давшие +4 п.п. на holdout,
   на CV прироста не дали — это был шум. OOF по всему train снижает ошибку
   примерно до ±1.5 п.п. и усредняет разные периоды.

   Почему допустимо, что модель в части фолдов учится на более поздних днях,
   чем валидирует:

   a) Нет утечки через объект. Куки в train уникальны, каждая относится ровно к одному
      суточному окну. Одна и та же кука не может оказаться и в обучении, и в валидации.

   b) Нет утечки через признаки. Все признаки куки считаются только по её собственным
      событиям внутри её окна (features.py). Никаких агрегатов по всей выборке
      (target encoding, частоты по всем кукам и т.п.), которые переносили бы информацию
      из будущих дней в прошлые, нет.

   c) Нет утечки через «соседей». Группировка по дням целиком: куки одного дня
      (общие всплески активности конкретного сервиса, общие поисковые запросы дня)
      не делятся между обучением и валидацией.

   d) Данные стационарны во времени (EDA): доля ботов по дням колеблется в пределах
      статистического шума (σ ≈ 1 п.п. при ~800 куки в день), тренда нет; распределение
      семейств UA и платформ в train и test совпадает; строк UA, которых нет в train,
      в test нет. Если зависимость «признаки → таргет» не меняется во времени,
      то порядок дней для оценки качества не важен, и K-fold даёт несмещённую оценку.

    Ограничение: стационарность проверена по распределению признаков (adversarial validation:
    train против test ROC-AUC 0.50, боты недели 1 против ботов недели 2 — 0.52), но не по связи
    признаков с таргетом: если сервисы меняют тактику так, что то же поведение начинает означать
    другое, K-fold будет оптимистичен. Намёк на это есть: последние фолды (16–19.04) хуже
    остальных. Поэтому нужна вторая оценка.

2. HOLDOUT ВПЕРЁД ВО ВРЕМЕНИ.

   Обучение на 06.04–12.04, валидация на 13.04–19.04. Та же форма, что у реального теста
   (целая неделя, строго позже обучения). Если улучшение есть на CV, но пропадает здесь —
   значит, оно не переносится во времени, и такое изменение не принимаем.

Правило: изменение принимаем, если оно не ухудшает ни OOF-метрику CV, ни holdout,
и улучшает хотя бы одну из них больше, чем на уровень шума.
=====================================================================================
"""

from __future__ import annotations

import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from features import build_features, load_data
from metric import precision_at_recall, recall_at_fpr

# ---------------------------------------------------------------- настройки

VALID_START = '2026-04-13'          # holdout вперёд во времени
DAYS_PER_FOLD = 2                   # CV
N_SEEDS = 5

# Группы признаков, которые НЕ идут в модель (по результатам абляции).
# По отдельности и все вместе они не меняют качество в пределах шума:
#   все 65 признаков : CV P@R 0.7702, PR-AUC 0.7923 | holdout P@R 0.6925, PR-AUC 0.7771
#   без этих 4 групп : CV P@R 0.7836, PR-AUC 0.7910 | holdout P@R 0.7097, PR-AUC 0.7749
# Их сигнал (возраст куки, объём, суточный профиль, тип клиента) уже покрыт группами
# cursor / diversity / timing / event_mix. Убираем ради простоты и интерпретируемости.
# Сами функции оставляем в features.py, вернуть группы можно флагом --all-groups.
EXCLUDED_GROUPS = ['cookie_age', 'volume_sessions', 'hours', 'client']

LGB_PARAMS = dict(
    objective='binary',
    n_estimators=400,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    deterministic=True,             # требуемая воспроизводимость между запусками
    force_row_wise=True,
    verbose=-1,
)


# ---------------------------------------------------------------- модель

def fit_predict(X_fit: pd.DataFrame, y_fit: np.ndarray, X_pred: pd.DataFrame,
                seeds: list[int]) -> tuple[np.ndarray, pd.Series]:
    # обучает по модели на каждый сид, возвращает средний score и средний gain.
    preds, imps = [], []
    for seed in seeds:
        model = lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed)
        model.fit(X_fit, y_fit)
        preds.append(model.predict_proba(X_pred)[:, 1])
        imps.append(pd.Series(model.booster_.feature_importance('gain'), index=X_fit.columns))
    return np.mean(preds, axis=0), pd.concat(imps, axis=1).mean(axis=1)


def report(y: np.ndarray, score: np.ndarray, title: str) -> None:
    # официальная метрика + диагностика.
    print(f'--- {title}')
    print(f'  P@R>=0.7      : {precision_at_recall(y, score):.4f}')
    print(f'  PR-AUC        : {average_precision_score(y, score):.4f}')
    print(f'  ROC-AUC       : {roc_auc_score(y, score):.4f}')
    print(f'  Recall@FPR=1% : {recall_at_fpr(y, score, 0.01):.4f}')
    print(f'  константа     : {y.mean():.4f}  (доля ботов, n={len(y)}, ботов={int(y.sum())})')


def day_folds(meta: pd.DataFrame, days_per_fold: int = DAYS_PER_FOLD) -> np.ndarray:
    # номер фолда для каждой куки.
    day = meta.window_start_ts.dt.normalize()
    day_idx = np.searchsorted(np.sort(day.unique()), day)
    return day_idx // days_per_fold


def run_cv(X: pd.DataFrame, y: np.ndarray, meta: pd.DataFrame, cols: list[str],
           seeds: list[int], verbose: bool = True) -> np.ndarray:
    # CV с группировкой по дням - OOF-предсказания + метрика по каждому фолду и по всем сразу.
    folds = day_folds(meta)
    oof = np.zeros(len(y))
    if verbose:
        print(f'--- CV по дням: {folds.max() + 1} фолдов по {DAYS_PER_FOLD} дня')
    for k in np.unique(folds):
        val = folds == k
        oof[val], _ = fit_predict(X.loc[~val, cols], y[~val], X.loc[val, cols], seeds)
        if verbose:
            d = meta.window_start_ts[val]
            print(f'  фолд {k}: {d.min():%d.%m}–{d.max():%d.%m}  ботов {int(y[val].sum()):3d}  '
                  f'P@R>=0.7 = {precision_at_recall(y[val], oof[val]):.4f}')
    if verbose:
        report(y, oof, 'CV: OOF по всему train')
    return oof


def select_columns(groups: dict[str, list[str]], drop: list[str] | None = None) -> list[str]:
    # колонки всех групп, кроме перечисленных в drop.
    drop = drop or []
    unknown = set(drop) - set(groups)
    if unknown:
        raise ValueError(f'неизвестные группы: {sorted(unknown)}; есть: {list(groups)}')
    return [c for name, cc in groups.items() if name not in drop for c in cc]


# ---------------------------------------------------------------- main

def main(submit: bool = True, n_seeds: int = N_SEEDS, drop_groups: list[str] | None = None) -> None:
    # drop_groups — какие группы признаков не использовать (по умолчанию EXCLUDED_GROUPS).
    t0 = time.time()
    seeds = list(range(n_seeds))
    train, test, events = load_data()
    X_tr, groups = build_features(events, train, return_groups=True)
    X_te = build_features(events, test)
    cols = select_columns(groups, drop_groups)
    y = train.target.values
    print(f'используются группы: {[g for g in groups if g not in (drop_groups or [])]}')
    print(f'признаки собраны: {len(cols)} шт., train {X_tr.shape}, test {X_te.shape}, {time.time() - t0:.1f} с\n')

    # CV
    run_cv(X_tr, y, train, cols, seeds)

    # holdout
    is_valid = (train.window_start_ts >= VALID_START).values
    p_valid, imp = fit_predict(X_tr.loc[~is_valid, cols], y[~is_valid], X_tr.loc[is_valid, cols], seeds)
    print()
    report(y[is_valid], p_valid, f'holdout {VALID_START}+ (обучение только на более ранних днях)')

    print(f'\nтоп-20 признаков по gain (holdout-модель, среднее по {len(seeds)} сидам):')
    print((imp / imp.sum()).sort_values(ascending=False).head(20).round(3).to_string())

    if not submit:
        print(f'\nготово за {time.time() - t0:.1f} с')
        return

    # train → test
    p_test, _ = fit_predict(X_tr[cols], y, X_te[cols], seeds)
    sub = pd.DataFrame({'cookie_id': X_te.cookie_id, 'score': p_test})

    # проверки формата. Куки из test, по одной строке, score в [0, 1].
    assert len(sub) == len(test) and sub.cookie_id.is_unique
    assert set(sub.cookie_id) == set(test.cookie_id)
    assert sub.score.between(0, 1).all() and sub.score.notna().all()

    sub.to_csv('submission.csv', index=False)
    print(f'\nsubmission.csv сохранён: {sub.shape}, уникальных score: {sub.score.nunique()}, '
          f'всего {time.time() - t0:.1f} с')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-submit', action='store_true', help='только CV и holdout, без сабмита')
    parser.add_argument('--n-seeds', type=int, default=N_SEEDS, help='число сидов для усреднения')
    parser.add_argument('--drop-group', action='append', default=[],
                        help='выключить ещё одну группу признаков (можно несколько раз), например --drop-group cursor')
    parser.add_argument('--all-groups', action='store_true',
                        help='использовать все группы, включая EXCLUDED_GROUPS')
    args = parser.parse_args()
    base = [] if args.all_groups else EXCLUDED_GROUPS
    main(submit=not args.no_submit, n_seeds=args.n_seeds, drop_groups=base + args.drop_group)
