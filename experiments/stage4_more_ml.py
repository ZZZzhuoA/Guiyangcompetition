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
    GradientBoostedStumpsUtility,
    RBFKernelUtilityRegressor,
    RidgeUtilityRegressor,
    StrategyWiseRidgeUtility,
)
from src.data_utils import load_competition_data
from src.ranking_models import (
    KNNUtilityRegressor,
    MeanUtilityByStrategy,
    RandomForestUtility,
    RegressionTreeUtility,
    recommend_strategy,
)
from src.stage2_eval import evaluate_rankers, utility_from_metrics
from src.stage3_eval import evaluate_against_global_utility
from src.stage3_models import ClusterLocalUtilityRecommender


def main() -> None:
    bundle = load_competition_data("Data")
    output_dir = Path("outputs") / "stage4_more_ml"
    output_dir.mkdir(parents=True, exist_ok=True)

    utility = utility_from_metrics(bundle.train_metric1, bundle.train_metric2, 0.5, 0.5)

    utility_models = [
        MeanUtilityByStrategy(),
        KNNUtilityRegressor(k=7),
        KNNUtilityRegressor(k=15),
        RegressionTreeUtility(max_depth=5, min_samples_leaf=8),
        RandomForestUtility(n_trees=25, max_depth=6, min_samples_leaf=8),
        RidgeUtilityRegressor(alpha=0.1),
        RidgeUtilityRegressor(alpha=1.0),
        StrategyWiseRidgeUtility(alpha=0.1),
        StrategyWiseRidgeUtility(alpha=1.0),
        RBFKernelUtilityRegressor(gamma=0.2, alpha=0.05),
        ExtraTreesUtilityRegressor(n_trees=21, max_depth=6, min_samples_leaf=8),
        ExtraTreesUtilityRegressor(n_trees=31, max_depth=7, min_samples_leaf=6),
        GradientBoostedStumpsUtility(n_estimators=40, learning_rate=0.06),
        GradientBoostedStumpsUtility(n_estimators=80, learning_rate=0.04),
    ]

    ranking_results = evaluate_rankers(utility_models, bundle.train_x, bundle.train_y, utility, 0.5, 0.5)

    cluster_models = [
        ClusterLocalUtilityRecommender(n_clusters=3, metric1_weight=0.5, metric2_weight=0.5),
        ClusterLocalUtilityRecommender(n_clusters=4, metric1_weight=0.5, metric2_weight=0.5),
        ClusterLocalUtilityRecommender(n_clusters=6, metric1_weight=0.5, metric2_weight=0.5),
    ]
    cluster_results = evaluate_against_global_utility(
        cluster_models,
        bundle.train_x,
        bundle.train_y,
        bundle.train_metric1,
        bundle.train_metric2,
        utility,
    )

    rows = []
    for row in ranking_results:
        rows.append(
            {
                "family": "utility_regression",
                "model": row.model,
                "top1_match": row.top1_match,
                "mean_regret": row.mean_regret,
                "mean_chosen_utility": row.mean_chosen_utility,
                "mean_best_utility": row.mean_best_utility,
                "rmse_utility": row.rmse_utility,
                "recommendation_entropy": "",
                "used_strategy_count": "",
            }
        )
    for row in cluster_results:
        rows.append(
            {
                "family": "cluster_local",
                "model": row.model,
                "top1_match": row.top1_match,
                "mean_regret": row.mean_regret,
                "mean_chosen_utility": row.mean_chosen_utility,
                "mean_best_utility": row.mean_best_utility,
                "rmse_utility": "",
                "recommendation_entropy": row.recommendation_entropy,
                "used_strategy_count": row.used_strategy_count,
            }
        )

    rows = sorted(rows, key=lambda item: (item["top1_match"], -item["mean_regret"]), reverse=True)
    with (output_dir / "more_ml_cv_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]
    if best["family"] == "cluster_local":
        model = next(m for m in cluster_models if m.name == best["model"])
        model.fit(bundle.train_x, bundle.train_y, bundle.train_metric1, bundle.train_metric2)
        test_pred, scores, clusters, weights = model.recommend(bundle.test_x)
    else:
        model = next(m for m in utility_models if m.name == best["model"])
        model.fit(bundle.train_x, bundle.train_y, utility)
        test_pred, scores = recommend_strategy(model, bundle.test_x)
        clusters = np.full(len(test_pred), -1, dtype=int)
        weights = np.full((len(test_pred), 2), 0.5, dtype=float)

    with (output_dir / "test_recommendations.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "cluster", "w_interception", "w_cost_effectiveness", "recommended_strategy", "score_s1", "score_s2", "score_s3", "score_s4"])
        for i, (cluster, weight, pred, score_row) in enumerate(zip(clusters, weights, test_pred, scores), start=1):
            writer.writerow([i, int(cluster), float(weight[0]), float(weight[1]), int(pred), *[float(v) for v in score_row]])

    summary = {
        "metric_weight_rule": "equal weights: 0.5 interception rate + 0.5 cost-effectiveness",
        "method_count": len(rows),
        "best_family": best["family"],
        "best_model": best["model"],
        "best_top1_match": best["top1_match"],
        "best_mean_regret": best["mean_regret"],
        "test_prediction_counts": {int(k): int(v) for k, v in zip(*np.unique(test_pred, return_counts=True))},
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nMore ML ranking:")
    for row in rows:
        print(
            f"{row['family']}/{row['model']}: top1={row['top1_match']:.4f}, "
            f"regret={row['mean_regret']:.4f}, chosen={row['mean_chosen_utility']:.4f}"
        )


if __name__ == "__main__":
    main()
