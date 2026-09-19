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
from src.stage2_eval import utility_from_metrics
from src.stage3_eval import evaluate_against_global_utility
from src.stage3_models import ClusterLocalUtilityRecommender


def main() -> None:
    bundle = load_competition_data("Data")
    output_dir = Path("outputs") / "stage3"
    output_dir.mkdir(parents=True, exist_ok=True)

    models = []
    for k in [2, 3, 4, 5, 6, 8]:
        models.append(ClusterLocalUtilityRecommender(n_clusters=k, metric1_weight=0.5, metric2_weight=0.5))
        models.append(ClusterLocalUtilityRecommender(n_clusters=k, metric1_weight=0.3, metric2_weight=0.7))
        models.append(ClusterLocalUtilityRecommender(n_clusters=k, adaptive_weights=True))

    eval_utility = utility_from_metrics(bundle.train_metric1, bundle.train_metric2, 0.5, 0.5)
    results = evaluate_against_global_utility(
        models,
        bundle.train_x,
        bundle.train_y,
        bundle.train_metric1,
        bundle.train_metric2,
        eval_utility,
    )
    results = sorted(
        results,
        key=lambda item: (item.top1_match, -item.mean_regret, item.recommendation_entropy),
        reverse=True,
    )

    with (output_dir / "cluster_cv_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].__dict__.keys()))
        writer.writeheader()
        for row in results:
            writer.writerow(row.__dict__)

    best = results[0]
    best_model = next(model for model in models if model.name == best.model)
    best_model.fit(bundle.train_x, bundle.train_y, bundle.train_metric1, bundle.train_metric2)
    pred, scores, clusters, weights = best_model.recommend(bundle.test_x)

    with (output_dir / "test_recommendations.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_id",
                "cluster",
                "adaptive_w_interception",
                "adaptive_w_cost_effectiveness",
                "recommended_strategy",
                "score_s1",
                "score_s2",
                "score_s3",
                "score_s4",
            ]
        )
        for i, (cluster, weight, recommendation, score_row) in enumerate(zip(clusters, weights, pred, scores), start=1):
            writer.writerow([i, int(cluster), float(weight[0]), float(weight[1]), int(recommendation), *[float(v) for v in score_row]])

    cluster_summary = {}
    for cluster in sorted(set(clusters.astype(int).tolist())):
        mask = clusters == cluster
        cluster_summary[int(cluster)] = {
            "test_rows": int(np.sum(mask)),
            "recommended_counts": {int(k): int(v) for k, v in zip(*np.unique(pred[mask], return_counts=True))},
            "weight_interception": float(np.mean(weights[mask, 0])),
            "weight_cost_effectiveness": float(np.mean(weights[mask, 1])),
        }

    summary = {
        "innovation": "scenario clustering + local utility estimation under equal metric weights; adaptive weights are retained only as ablation comparison",
        "evaluation_utility": "0.5 * normalized interception rate + 0.5 * normalized cost-effectiveness",
        "best_model": best.model,
        "best_top1_match": best.top1_match,
        "best_mean_regret": best.mean_regret,
        "best_recommendation_entropy": best.recommendation_entropy,
        "best_used_strategy_count": best.used_strategy_count,
        "test_prediction_counts": {int(k): int(v) for k, v in zip(*np.unique(pred, return_counts=True))},
        "test_cluster_summary": cluster_summary,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nCluster CV:")
    for row in results:
        print(
            f"{row.model}: top1={row.top1_match:.4f}, regret={row.mean_regret:.4f}, "
            f"chosen={row.mean_chosen_utility:.4f}, best={row.mean_best_utility:.4f}, "
            f"entropy={row.recommendation_entropy:.4f}, strategies={row.used_strategy_count}"
        )


if __name__ == "__main__":
    main()
