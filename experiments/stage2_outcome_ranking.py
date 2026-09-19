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
from src.ranking_models import (
    KNNUtilityRegressor,
    MeanUtilityByStrategy,
    RandomForestUtility,
    RegressionTreeUtility,
    recommend_strategy,
)
from src.stage2_eval import evaluate_rankers, utility_from_metrics


def main() -> None:
    bundle = load_competition_data("Data")
    strategies = bundle.train_y.astype(int)

    models = [
        MeanUtilityByStrategy(),
        KNNUtilityRegressor(k=7),
        KNNUtilityRegressor(k=15),
        RegressionTreeUtility(max_depth=5, min_samples_leaf=8),
        RegressionTreeUtility(max_depth=8, min_samples_leaf=5),
        RandomForestUtility(n_trees=25, max_depth=6, min_samples_leaf=8),
    ]

    weights = [(1.0, 0.0), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7)]
    output_dir = Path("outputs") / "stage2"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for w1, w2 in weights:
        utility = utility_from_metrics(bundle.train_metric1, bundle.train_metric2, w1, w2)
        all_results.extend(evaluate_rankers(models, bundle.train_x, strategies, utility, w1, w2))

    all_results = sorted(
        all_results,
        key=lambda item: (item.top1_match, -item.mean_regret, -item.rmse_utility),
        reverse=True,
    )

    with (output_dir / "ranking_cv_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_results[0].__dict__.keys()))
        writer.writeheader()
        for row in all_results:
            writer.writerow(row.__dict__)

    mandated_weight_results = [
        row for row in all_results if abs(row.weight_metric1 - 0.5) < 1e-12 and abs(row.weight_metric2 - 0.5) < 1e-12
    ]
    best = sorted(
        mandated_weight_results,
        key=lambda item: (item.top1_match, -item.mean_regret, -item.rmse_utility),
        reverse=True,
    )[0]
    best_model = next(model for model in models if model.name == best.model)
    best_utility = utility_from_metrics(bundle.train_metric1, bundle.train_metric2, best.weight_metric1, best.weight_metric2)
    best_model.fit(bundle.train_x, strategies, best_utility)
    test_pred, test_scores = recommend_strategy(best_model, bundle.test_x)

    with (output_dir / "test_recommendations.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "recommended_strategy", "score_s1", "score_s2", "score_s3", "score_s4"])
        for i, (pred, scores) in enumerate(zip(test_pred, test_scores), start=1):
            writer.writerow([i, int(pred), *[float(v) for v in scores]])

    summary = {
        "metric_direction": "metric1 interception rate higher is better; metric2 cost-effectiveness higher is better",
        "metric_weight_rule": "metric1 and metric2 use equal weights: 0.5 / 0.5",
        "best_model": best.model,
        "best_weight_metric1": best.weight_metric1,
        "best_weight_metric2": best.weight_metric2,
        "best_top1_match": best.top1_match,
        "best_mean_regret": best.mean_regret,
        "best_rmse_utility": best.rmse_utility,
        "test_prediction_counts": {
            int(k): int(v) for k, v in zip(*np.unique(test_pred, return_counts=True))
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nRanking CV:")
    for row in all_results:
        print(
            f"{row.model}, w=({row.weight_metric1:.1f},{row.weight_metric2:.1f}): "
            f"top1={row.top1_match:.4f}, regret={row.mean_regret:.4f}, "
            f"chosen={row.mean_chosen_utility:.4f}, best={row.mean_best_utility:.4f}, "
            f"rmse={row.rmse_utility:.4f}"
        )


if __name__ == "__main__":
    main()
