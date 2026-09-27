"""
Обучение и сабмит.

    python train.py                 # валидация на holdout + обучение на всём train + submission.csv
    python train.py --no-submit     # только валидация

Схема:
  1. Holdout — последняя неделя train (окна с VALID_START): та же форма, что у теста
     (целая неделя, тест лежит позже трейна по времени).
  2. Модель — LightGBM, усреднение по нескольким сидам (снижает дисперсию и число одинаковых score).
  3. Финал — те же параметры, обучение на всём train, предсказание test.

Метрика — официальная precision_at_recall из metric.py.
"""

from __future__ import annotations

import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from features import build_features, feature_columns, load_data
from metric import precision_at_recall, recall_at_fpr

# ---------------------------------------------------------------- настройки

VALID_START = '2026-04-13'          # holdout: окна 13.04–19.04
SEEDS = [0, 1, 2, 3, 4]

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
    deterministic=True,             # воспроизводимость между запусками
    force_row_wise=True,
    verbose=-1,
)


# ---------------------------------------------------------------- модель

def fit_predict(X_fit: pd.DataFrame, y_fit: np.ndarray, X_pred: pd.DataFrame,
                seeds: list[int] = SEEDS) -> tuple[np.ndarray, pd.Series]:
    """Обучает по модели на каждый сид, возвращает средний score и среднюю важность (gain)."""
    preds, imps = [], []
    for seed in seeds:
        model = lgb.LGBMClassifier(**LGB_PARAMS, random_state=seed)
        model.fit(X_fit, y_fit)
        preds.append(model.predict_proba(X_pred)[:, 1])
        imps.append(pd.Series(model.booster_.feature_importance('gain'), index=X_fit.columns))
    return np.mean(preds, axis=0), pd.concat(imps, axis=1).mean(axis=1)


def report(y: np.ndarray, score: np.ndarray, title: str) -> None:
    """Официальная метрика + диагностика."""
    print(f'--- {title}')
    print(f'  P@R>=0.7      : {precision_at_recall(y, score):.4f}')
    print(f'  PR-AUC        : {average_precision_score(y, score):.4f}')
    print(f'  ROC-AUC       : {roc_auc_score(y, score):.4f}')
    print(f'  Recall@FPR=1% : {recall_at_fpr(y, score, 0.01):.4f}')
    print(f'  константа     : {y.mean():.4f}  (доля ботов, n={len(y)}, ботов={int(y.sum())})')


# ---------------------------------------------------------------- main

def main(submit: bool = True) -> None:
    t0 = time.time()
    train, test, events = load_data()
    X_tr = build_features(events, train)
    X_te = build_features(events, test)
    cols = feature_columns(X_tr)
    y = train.target.values
    print(f'признаки собраны: {len(cols)} шт., train {X_tr.shape}, test {X_te.shape}, {time.time() - t0:.1f} с')

    # ---- валидация
    is_valid = (train.window_start_ts >= VALID_START).values
    p_valid, imp = fit_predict(X_tr.loc[~is_valid, cols], y[~is_valid], X_tr.loc[is_valid, cols])
    report(y[is_valid], p_valid, f'holdout {VALID_START}+ (усреднение по {len(SEEDS)} сидам)')

    print('\nтоп-20 признаков по gain:')
    print((imp / imp.sum()).sort_values(ascending=False).head(20).round(3).to_string())

    if not submit:
        return

    # ---- финал: весь train → test
    p_test, _ = fit_predict(X_tr[cols], y, X_te[cols])
    sub = pd.DataFrame({'cookie_id': X_te.cookie_id, 'score': p_test})

    # проверки формата: ровно куки из test, по одной строке, score в [0, 1]
    assert len(sub) == len(test) and sub.cookie_id.is_unique
    assert set(sub.cookie_id) == set(test.cookie_id)
    assert sub.score.between(0, 1).all() and sub.score.notna().all()

    sub.to_csv('submission.csv', index=False)
    print(f'\nsubmission.csv сохранён: {sub.shape}, уникальных score: {sub.score.nunique()}, '
          f'всего {time.time() - t0:.1f} с')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-submit', action='store_true', help='только валидация, без сабмита')
    args = parser.parse_args()
    main(submit=not args.no_submit)
