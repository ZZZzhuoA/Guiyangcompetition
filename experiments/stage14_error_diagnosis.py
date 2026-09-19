from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_utils import load_competition_data
from src.feature_engineering import build_engineered_features
from src.ranking_models import RandomForestUtility
from src.stage2_eval import group_folds, interception_priority_utility, scenario_keys


def evaluate_errors() -> tuple[list[dict[str, object]], dict[str, object]]:
    bundle = load_competition_data("Data")
    utility = interception_priority_utility(bundle.train_metric1, bundle.train_metric2, 0.05)
    engineered_x, _ = build_engineered_features(bundle.train_x)
    keys = scenario_keys(bundle.train_x)
    splits = group_folds(keys)

    rows: list[dict[str, object]] = []
    for fold_id, (train_idx, valid_idx) in enumerate(splits, start=1):
        model = RandomForestUtility(n_trees=25, max_depth=6, min_samples_leaf=8)
        model.fit(engineered_x[train_idx], bundle.train_y[train_idx], utility[train_idx])

        by_key: dict[tuple[int, ...], list[int]] = {}
        for idx in valid_idx:
            by_key.setdefault(keys[idx], []).append(int(idx))

        for idxs in by_key.values():
            candidate_strategies = np.array(sorted(set(int(bundle.train_y[i]) for i in idxs)), dtype=int)
            if len(candidate_strategies) < 2:
                continue
            candidate_x = np.repeat(engineered_x[idxs[0]].reshape(1, -1), len(candidate_strategies), axis=0)
            scores = model.predict(candidate_x, candidate_strategies)

            true_by_strategy = []
            for s in candidate_strategies:
                s_idxs = [i for i in idxs if int(bundle.train_y[i]) == int(s)]
                true_by_strategy.append(
                    {
                        "strategy": int(s),
                        "utility": float(np.mean(utility[s_idxs])),
                        "metric1": float(np.mean(bundle.train_metric1[s_idxs])),
                        "metric2": float(np.mean(bundle.train_metric2[s_idxs])),
                        "count": int(len(s_idxs)),
                    }
                )
            true_utility = np.array([v["utility"] for v in true_by_strategy], dtype=float)
            true_m1 = np.array([v["metric1"] for v in true_by_strategy], dtype=float)
            true_m2 = np.array([v["metric2"] for v in true_by_strategy], dtype=float)

            pred_pos = int(np.argmax(scores))
            best_pos = int(np.argmax(true_utility))
            sorted_utility = np.sort(true_utility)[::-1]
            second_utility = float(sorted_utility[1]) if len(sorted_utility) > 1 else float(sorted_utility[0])
            rows.append(
                {
                    "fold": fold_id,
                    "scenario_key_hash": hash(idxs[0]),
                    "candidate_strategy_count": int(len(candidate_strategies)),
                    "candidate_strategies": ";".join(str(int(s)) for s in candidate_strategies),
                    "pred_strategy": int(candidate_strategies[pred_pos]),
                    "best_strategy": int(candidate_strategies[best_pos]),
                    "top1": int(pred_pos == best_pos),
                    "pred_rank": int(1 + np.sum(true_utility > true_utility[pred_pos])),
                    "best_utility": float(true_utility[best_pos]),
                    "pred_utility": float(true_utility[pred_pos]),
                    "utility_regret": float(true_utility[best_pos] - true_utility[pred_pos]),
                    "best_metric1": float(true_m1[best_pos]),
                    "pred_metric1": float(true_m1[pred_pos]),
                    "metric1_regret": float(true_m1[best_pos] - true_m1[pred_pos]),
                    "best_metric2": float(true_m2[best_pos]),
                    "pred_metric2": float(true_m2[pred_pos]),
                    "metric2_regret": float(true_m2[best_pos] - true_m2[pred_pos]),
                    "best_minus_second_utility": float(true_utility[best_pos] - second_utility),
                    "score_gap_pred_best": float(scores[pred_pos] - scores[best_pos]),
                    "complete_four_strategies": int(len(candidate_strategies) == 4),
                }
            )

    arr_top1 = np.array([r["top1"] for r in rows], dtype=float)
    metric1_regret = np.array([r["metric1_regret"] for r in rows], dtype=float)
    utility_regret = np.array([r["utility_regret"] for r in rows], dtype=float)
    complete = np.array([r["complete_four_strategies"] for r in rows], dtype=bool)
    cand_count = np.array([r["candidate_strategy_count"] for r in rows], dtype=int)
    close_gap = np.array([r["best_minus_second_utility"] for r in rows], dtype=float)

    summary = {
        "evaluated_scenarios": int(len(rows)),
        "top1_match": float(np.mean(arr_top1)),
        "mean_metric1_regret": float(np.mean(metric1_regret)),
        "median_metric1_regret": float(np.median(metric1_regret)),
        "p90_metric1_regret": float(np.quantile(metric1_regret, 0.9)),
        "mean_utility_regret": float(np.mean(utility_regret)),
        "complete_four_strategy_scenarios": int(np.sum(complete)),
        "complete_four_strategy_top1": float(np.mean(arr_top1[complete])) if np.any(complete) else None,
        "partial_strategy_scenarios": int(np.sum(~complete)),
        "partial_strategy_top1": float(np.mean(arr_top1[~complete])) if np.any(~complete) else None,
        "candidate_count_distribution": {str(k): int(v) for k, v in zip(*np.unique(cand_count, return_counts=True))},
        "close_best_second_gap_under_0_02": int(np.sum(close_gap < 0.02)),
        "close_gap_under_0_02_top1": float(np.mean(arr_top1[close_gap < 0.02])) if np.any(close_gap < 0.02) else None,
        "large_metric1_regret_over_0_05": int(np.sum(metric1_regret > 0.05)),
        "negative_metric2_tradeoff_cases": int(np.sum(np.array([r["metric2_regret"] for r in rows], dtype=float) < 0)),
    }
    return rows, summary


def main() -> None:
    output_dir = Path("outputs") / "stage14_error_diagnosis"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = evaluate_errors()

    fieldnames = list(rows[0].keys()) if rows else []
    with (output_dir / "cv_error_cases.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
