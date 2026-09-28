"""
Абляция групп признаков: по очереди выключаем каждую группу и смотрим, как меняется качество.

    python ablation.py               # 2 сида на модель
    python ablation.py --n-seeds 5   # точнее, но дольше

Для каждой конфигурации считаются:
  - OOF по CV с группировкой по дням (основа для решения),
  - holdout вперёд во времени.

Как читать delta = метрика_без_группы − метрика_со_всеми:
  delta < 0  → без группы хуже, группа полезна;
  delta ≈ 0  → группа не даёт ничего сверх остальных;
  delta > 0  → без группы лучше, группа шумит.
Шум OOF P@R≈±1.5 п.п., holdout ±2.3 п.п.; PR-AUC стабильнее и помогает отличить сигнал от шума.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from features import build_features, load_data
from metric import precision_at_recall
from train import VALID_START, fit_predict, run_cv, select_columns


def evaluate(X: pd.DataFrame, y: np.ndarray, meta: pd.DataFrame, cols: list[str],
             seeds: list[int]) -> dict[str, float]:
    """OOF-метрики CV + holdout-метрики для заданного набора колонок."""
    oof = run_cv(X, y, meta, cols, seeds, verbose=False)
    hold = (meta.window_start_ts >= VALID_START).values
    p_hold, _ = fit_predict(X.loc[~hold, cols], y[~hold], X.loc[hold, cols], seeds)
    return {'n_feat': len(cols),
            'cv_p@r': precision_at_recall(y, oof),
            'cv_prauc': average_precision_score(y, oof),
            'hold_p@r': precision_at_recall(y[hold], p_hold),
            'hold_prauc': average_precision_score(y[hold], p_hold)}


def main(n_seeds: int) -> None:
    t0 = time.time()
    seeds = list(range(n_seeds))
    train, _, events = load_data()
    X, groups = build_features(events, train, return_groups=True)
    y = train.target.values

    rows = {'ALL': evaluate(X, y, train, select_columns(groups), seeds)}
    print(f"ALL: CV P@R {rows['ALL']['cv_p@r']:.4f}  holdout P@R {rows['ALL']['hold_p@r']:.4f}")
    for name in groups:
        rows[f'- {name}'] = evaluate(X, y, train, select_columns(groups, [name]), seeds)
        r = rows[f'- {name}']
        print(f"без {name:16s} ({len(groups[name]):2d} призн.): CV P@R {r['cv_p@r']:.4f}  "
              f"holdout P@R {r['hold_p@r']:.4f}   [{time.time() - t0:.0f} с]")

    res = pd.DataFrame(rows).T
    base = res.loc['ALL']
    for m in ['cv_p@r', 'cv_prauc', 'hold_p@r', 'hold_prauc']:
        res[f'd_{m}'] = res[m] - base[m]
    res['n_feat'] = res.n_feat.astype(int)

    cols_show = ['n_feat', 'cv_p@r', 'd_cv_p@r', 'd_cv_prauc', 'hold_p@r', 'd_hold_p@r', 'd_hold_prauc']
    print('\n=== итог (сортировка по d_cv_p@r: сверху самые полезные группы) ===')
    print(res[cols_show].sort_values('d_cv_p@r').round(4).to_string())
    res.to_csv('ablation_results.csv')
    print(f'\nсохранено в ablation_results.csv, всего {time.time() - t0:.0f} с')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-seeds', type=int, default=2)
    main(parser.parse_args().n_seeds)
