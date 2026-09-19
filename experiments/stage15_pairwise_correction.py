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
from src.exact_scene_recommender import ExactSceneHybridRecommender
from src.feature_engineering import build_engineered_features
from src.fine_grained_eval import aggregate_metric_rows, evaluate_scenario_choice
from src.pairwise_correction import PairwiseCorrectedUtility
from src.pairwise_models import PairwiseLogisticRanker
from src.ranking_models import RandomForestUtility
from src.stage2_eval import group_folds, interception_priority_utility, scenario_keys


def evaluate_model(model, x, strategy, metric1, metric2, utility, max_folds: int | None = None) -> dict[str, object]:
    keys = scenario_keys(x)
    splits = group_folds(keys)
    if max_folds is not None:
        splits = splits[:max_folds]
    metric_rows = []
    pred_strategies = []
    best_strategies = []
    metric1_regrets = []
    for train_idx, valid_idx in splits:
        fitted = model.__class__(**getattr(model, "__dict__", {}))
        fitted.fit(x[train_idx], strategy[train_idx], utility[train_idx])
        by_key: dict[tuple[int, ...], list[int]] = {}
        for idx in valid_idx:
            by_key.setdefault(keys[idx], []).append(int(idx))
        for idxs in by_key.values():
            candidate_strategies = np.array(sorted(set(int(strategy[i]) for i in idxs)), dtype=int)
            if len(candidate_strategies) < 2:
                continue
            true_utility = []
            true_m1 = []
            true_m2 = []
            for s in candidate_strategies:
                s_idxs = [i for i in idxs if int(strategy[i]) == int(s)]
                true_utility.append(float(np.mean(utility[s_idxs])))
                true_m1.append(float(np.mean(metric1[s_idxs])))
                true_m2.append(float(np.mean(metric2[s_idxs])))
            true_utility = np.array(true_utility, dtype=float)
            true_m1 = np.array(true_m1, dtype=float)
            true_m2 = np.array(true_m2, dtype=float)
            candidate_x = np.repeat(x[idxs[0]].reshape(1, -1), len(candidate_strategies), axis=0)
            scores = fitted.predict(candidate_x, candidate_strategies)
            chosen_strategy = int(candidate_strategies[int(np.argmax(scores))])
            row = evaluate_scenario_choice(
                candidate_strategies,
                true_utility,
                chosen_strategy,
                scores,
                np.vstack([true_m1, true_m2]).T,
            )
            if row:
                metric_rows.append(row)
                pred_strategies.append(chosen_strategy)
                best_pos = int(np.argmax(true_utility))
                chosen_pos = int(np.where(candidate_strategies == chosen_strategy)[0][0])
                best_strategies.append(int(candidate_strategies[best_pos]))
                metric1_regrets.append(float(true_m1[best_pos] - true_m1[chosen_pos]))
    metrics = aggregate_metric_rows(metric_rows)
    metrics["pred_strategy_counts"] = {int(k): int(v) for k, v in zip(*np.unique(np.array(pred_strategies), return_counts=True))}
    metrics["best_strategy_counts"] = {int(k): int(v) for k, v in zip(*np.unique(np.array(best_strategies), return_counts=True))}
    metrics["mean_metric1_regret"] = float(np.mean(metric1_regrets)) if metric1_regrets else 0.0
    metrics["p90_metric1_regret"] = float(np.quantile(metric1_regrets, 0.9)) if metric1_regrets else 0.0
    metrics["strategy3_over_prediction"] = int(metrics["pred_strategy_counts"].get(3, 0) - metrics["best_strategy_counts"].get(3, 0))
    return metrics


def final_test_recommendations(bundle, engineered_train_x, engineered_test_x, utility, best_model, output_dir: Path) -> dict[str, object]:
    final_model = ExactSceneHybridRecommender(
        base_model=best_model,
        min_exact_strategies=2,
        exact_weight=0.85,
    )
    final_model.fit(bundle.train_x, engineered_train_x, bundle.train_y, utility, bundle.train_metric1, bundle.train_metric2)
    pred, scores, sources, exact_counts = final_model.recommend(bundle.test_x, engineered_test_x)
    with (output_dir / "test_recommendations.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "recommended_strategy", "score_s1", "score_s2", "score_s3", "score_s4", "source", "exact_count_s1", "exact_count_s2", "exact_count_s3", "exact_count_s4"])
        for i, (p, score_row, source, count_row) in enumerate(zip(pred, scores, sources, exact_counts), start=1):
            writer.writerow([i, int(p), *[round(float(v), 8) for v in score_row], source, *[int(v) for v in count_row]])
    return {
        "test_prediction_counts": {int(k): int(v) for k, v in zip(*np.unique(pred, return_counts=True))},
        "test_source_counts": {str(k): int(v) for k, v in zip(*np.unique(np.array(sources), return_counts=True))},
    }


def main() -> None:
    bundle = load_competition_data("Data")
    output_dir = Path("outputs") / "stage15_pairwise_correction"
    output_dir.mkdir(parents=True, exist_ok=True)
    utility = interception_priority_utility(bundle.train_metric1, bundle.train_metric2, 0.05)
    engineered_train_x, _ = build_engineered_features(bundle.train_x)
    engineered_test_x, _ = build_engineered_features(bundle.test_x)

    candidates = [RandomForestUtility(n_trees=15, max_depth=5, min_samples_leaf=10)]
    pairwise = PairwiseLogisticRanker(epochs=80, learning_rate=0.08, l2=0.003)
    for weight, penalty in [(0.35, 0.0), (0.35, 0.05), (0.55, 0.05), (0.55, 0.1)]:
        candidates.append(
            PairwiseCorrectedUtility(
                base_model=RandomForestUtility(n_trees=15, max_depth=5, min_samples_leaf=10),
                pairwise_model=pairwise,
                pairwise_weight=weight,
                strategy3_penalty=penalty,
            )
        )

    rows = []
    for model in candidates:
        metrics = evaluate_model(model, engineered_train_x, bundle.train_y, bundle.train_metric1, bundle.train_metric2, utility, max_folds=3)
        row = {
            "model": model.name,
            "top1_match": metrics.get("top1_match", 0.0),
            "top2_hit": metrics.get("top2_hit", 0.0),
            "soft_match": metrics.get("soft_match", 0.0),
            "regret": metrics.get("regret", 0.0),
            "pareto_hit": metrics.get("pareto_hit", 0.0),
            "mean_metric1_regret": metrics.get("mean_metric1_regret", 0.0),
            "p90_metric1_regret": metrics.get("p90_metric1_regret", 0.0),
            "strategy3_over_prediction": metrics.get("strategy3_over_prediction", 0),
            "pred_strategy_counts": json.dumps(metrics.get("pred_strategy_counts", {}), ensure_ascii=False),
        }
        rows.append(row)

    rows = sorted(rows, key=lambda r: (r["top1_match"], r["soft_match"], -abs(r["strategy3_over_prediction"]), -r["mean_metric1_regret"]), reverse=True)
    with (output_dir / "pairwise_correction_cv_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best_name = rows[0]["model"]
    best_model = next(model for model in candidates if model.name == best_name)
    test_info = final_test_recommendations(bundle, engineered_train_x, engineered_test_x, utility, best_model, output_dir)

    summary = {
        "best_model": best_name,
        "best_cv": rows[0],
        **test_info,
        "interpretation": "Pairwise correction reduces strategy-3 over-recommendation by blending utility regression with same-scene pairwise preferences.",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
