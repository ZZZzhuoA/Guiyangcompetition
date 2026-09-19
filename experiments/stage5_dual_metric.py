from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.advanced_ml_models import (
    ExtraTreesUtilityRegressor,
    RBFKernelUtilityRegressor,
    RidgeUtilityRegressor,
    StrategyWiseRidgeUtility,
)
from src.data_utils import load_competition_data
from src.multi_metric_models import DualMetricRegressor
from src.ranking_models import KNNUtilityRegressor, RandomForestUtility, RegressionTreeUtility, recommend_strategy
from src.stage2_eval import group_folds, scenario_keys, utility_from_metrics
from src.stage3_eval import evaluate_against_global_utility
from src.stage3_models import ClusterLocalUtilityRecommender


def main() -> None:
    bundle = load_competition_data("Data")
    output_dir = Path("outputs") / "stage5_dual_metric"
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_utility = utility_from_metrics(bundle.train_metric1, bundle.train_metric2, 0.5, 0.5)
    base_models = [
        RidgeUtilityRegressor(alpha=0.1),
        RidgeUtilityRegressor(alpha=1.0),
        StrategyWiseRidgeUtility(alpha=1.0),
        KNNUtilityRegressor(k=15),
        RegressionTreeUtility(max_depth=5, min_samples_leaf=8),
        RandomForestUtility(n_trees=25, max_depth=6, min_samples_leaf=8),
        ExtraTreesUtilityRegressor(n_trees=21, max_depth=6, min_samples_leaf=8),
        RBFKernelUtilityRegressor(gamma=0.2, alpha=0.05),
    ]
    dual_models = [DualMetricRegressor(base_model=model) for model in base_models]

    results = evaluate_dual_models(
        dual_models,
        bundle.train_x,
        bundle.train_y,
        bundle.train_metric1,
        bundle.train_metric2,
        eval_utility,
    )

    cluster_model = ClusterLocalUtilityRecommender(n_clusters=3, metric1_weight=0.5, metric2_weight=0.5)
    cluster_results = evaluate_against_global_utility(
        [cluster_model],
        bundle.train_x,
        bundle.train_y,
        bundle.train_metric1,
        bundle.train_metric2,
        eval_utility,
    )

    rows = []
    for row in results:
        rows.append(
            {
                "family": "dual_metric_regression",
                "model": row.model,
                "top1_match": row.top1_match,
                "mean_regret": row.mean_regret,
                "mean_chosen_utility": row.mean_chosen_utility,
                "mean_best_utility": row.mean_best_utility,
                "rmse_utility": row.rmse_utility,
            }
        )
    for row in cluster_results:
        rows.append(
            {
                "family": "cluster_local_baseline",
                "model": row.model,
                "top1_match": row.top1_match,
                "mean_regret": row.mean_regret,
                "mean_chosen_utility": row.mean_chosen_utility,
                "mean_best_utility": row.mean_best_utility,
                "rmse_utility": "",
            }
        )
    rows = sorted(rows, key=lambda item: (item["top1_match"], -item["mean_regret"]), reverse=True)

    with (output_dir / "dual_metric_cv_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]
    if best["family"] == "cluster_local_baseline":
        final_model = ClusterLocalUtilityRecommender(n_clusters=3, metric1_weight=0.5, metric2_weight=0.5)
        final_model.fit(bundle.train_x, bundle.train_y, bundle.train_metric1, bundle.train_metric2)
        test_pred, scores, clusters, weights = final_model.recommend(bundle.test_x)
    else:
        final_model = next(model for model in dual_models if model.name == best["model"])
        final_model.fit(bundle.train_x, bundle.train_y, bundle.train_metric1, bundle.train_metric2)
        test_pred, scores = recommend_strategy(final_model, bundle.test_x)
        clusters = np.full(len(test_pred), -1, dtype=int)
        weights = np.full((len(test_pred), 2), 0.5, dtype=float)

    with (output_dir / "test_recommendations.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "cluster", "w_interception", "w_cost_effectiveness", "recommended_strategy", "score_s1", "score_s2", "score_s3", "score_s4"])
        for i, (cluster, weight, pred, score_row) in enumerate(zip(clusters, weights, test_pred, scores), start=1):
            writer.writerow([i, int(cluster), float(weight[0]), float(weight[1]), int(pred), *[float(v) for v in score_row]])

    summary = {
        "idea": "regress interception rate and cost-effectiveness separately, then combine with equal weights",
        "best_family": best["family"],
        "best_model": best["model"],
        "best_top1_match": best["top1_match"],
        "best_mean_regret": best["mean_regret"],
        "test_prediction_counts": {int(k): int(v) for k, v in zip(*np.unique(test_pred, return_counts=True))},
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nDual metric ranking:")
    for row in rows:
        print(f"{row['family']}/{row['model']}: top1={row['top1_match']:.4f}, regret={row['mean_regret']:.4f}")


def evaluate_dual_models(models, x, strategy, metric1, metric2, utility):
    keys = scenario_keys(x)
    splits = group_folds(keys)
    out = []
    for model in models:
        top1 = []
        regrets = []
        chosen_utils = []
        best_utils = []
        sq_errors = []
        for train_idx, valid_idx in splits:
            fitted = model.__class__(**getattr(model, "__dict__", {}))
            fitted.fit(x[train_idx], strategy[train_idx], metric1[train_idx], metric2[train_idx])
            pred_utility = fitted.predict(x[valid_idx], strategy[valid_idx])
            sq_errors.extend((pred_utility - utility[valid_idx]) ** 2)

            by_key = {}
            for idx in valid_idx:
                by_key.setdefault(keys[idx], []).append(int(idx))
            for idxs in by_key.values():
                if len(idxs) < 2:
                    continue
                candidate_strategies = strategy[idxs].astype(int)
                candidate_x = np.repeat(x[idxs[0]].reshape(1, -1), len(idxs), axis=0)
                scores = fitted.predict(candidate_x, candidate_strategies)
                chosen_pos = int(np.argmax(scores))
                best_pos = int(np.argmax(utility[idxs]))
                chosen_utility = float(utility[idxs][chosen_pos])
                best_utility = float(utility[idxs][best_pos])
                top1.append(1.0 if int(candidate_strategies[chosen_pos]) == int(candidate_strategies[best_pos]) else 0.0)
                regrets.append(best_utility - chosen_utility)
                chosen_utils.append(chosen_utility)
                best_utils.append(best_utility)
        class Result:
            pass
        row = Result()
        row.model = model.name
        row.top1_match = float(np.mean(top1)) if top1 else 0.0
        row.mean_regret = float(np.mean(regrets)) if regrets else 0.0
        row.mean_chosen_utility = float(np.mean(chosen_utils)) if chosen_utils else 0.0
        row.mean_best_utility = float(np.mean(best_utils)) if best_utils else 0.0
        row.rmse_utility = float(np.sqrt(np.mean(sq_errors))) if sq_errors else 0.0
        out.append(row)
    return out


if __name__ == "__main__":
    main()
