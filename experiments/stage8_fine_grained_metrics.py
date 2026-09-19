from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.advanced_ml_models import RBFKernelUtilityRegressor, RidgeUtilityRegressor
from src.data_utils import load_competition_data
from src.fine_grained_eval import evaluate_recommender_fine_grained
from src.multi_metric_models import DualMetricRegressor
from src.pairwise_models import PairwiseLogisticRanker
from src.pareto_models import ParetoStrategyRecommender
from src.ranking_models import MeanUtilityByStrategy, RandomForestUtility
from src.stage2_eval import utility_from_metrics
from src.stage3_models import ClusterLocalUtilityRecommender


def main() -> None:
    bundle = load_competition_data("Data")
    output_dir = Path("outputs") / "stage8_fine_grained"
    output_dir.mkdir(parents=True, exist_ok=True)
    utility = utility_from_metrics(bundle.train_metric1, bundle.train_metric2, 0.5, 0.5)

    models = [
        ("utility", MeanUtilityByStrategy()),
        ("utility", RidgeUtilityRegressor(alpha=1.0)),
        ("utility", RBFKernelUtilityRegressor(gamma=0.2, alpha=0.05)),
        ("utility", RandomForestUtility(n_trees=25, max_depth=6, min_samples_leaf=8)),
        ("cluster", ClusterLocalUtilityRecommender(n_clusters=3, metric1_weight=0.5, metric2_weight=0.5)),
        ("pairwise", PairwiseLogisticRanker(epochs=800, learning_rate=0.05, l2=0.003)),
        ("pareto", ParetoStrategyRecommender(DualMetricRegressor(RandomForestUtility(n_trees=25, max_depth=6, min_samples_leaf=8)), tie_breaker="product")),
        ("pareto", ParetoStrategyRecommender(DualMetricRegressor(RidgeUtilityRegressor(alpha=0.1)), tie_breaker="equal_weight")),
    ]

    rows = []
    for fit_mode, model in models:
        metrics = evaluate_recommender_fine_grained(
            model,
            fit_mode,
            bundle.train_x,
            bundle.train_y,
            bundle.train_metric1,
            bundle.train_metric2,
            utility,
        )
        metrics["family"] = fit_mode
        metrics["model"] = model.name
        rows.append(metrics)

    rows = sorted(rows, key=lambda row: (row.get("soft_match", 0), row.get("ndcg", 0), -row.get("regret", 1)), reverse=True)
    fieldnames = ["family", "model", "top1_match", "top2_hit", "chosen_rank", "rank_percentile", "soft_match", "regret", "relative_regret", "ndcg", "mrr", "pareto_hit", "chosen_utility", "best_utility"]
    with (output_dir / "fine_grained_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    summary = {
        "idea": "replace binary top1 match with soft/ranking metrics",
        "best_by_soft_match": rows[0]["model"],
        "best_soft_match": rows[0].get("soft_match"),
        "best_ndcg": max(row.get("ndcg", 0) for row in rows),
        "lowest_regret": min(row.get("regret", 999) for row in rows),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nFine-grained metrics:")
    for row in rows:
        print(
            f"{row['family']}/{row['model']}: top1={row.get('top1_match',0):.4f}, "
            f"top2={row.get('top2_hit',0):.4f}, soft={row.get('soft_match',0):.4f}, "
            f"rank_pct={row.get('rank_percentile',0):.4f}, ndcg={row.get('ndcg',0):.4f}, "
            f"regret={row.get('regret',0):.4f}, pareto={row.get('pareto_hit',0):.4f}"
        )


if __name__ == "__main__":
    main()
