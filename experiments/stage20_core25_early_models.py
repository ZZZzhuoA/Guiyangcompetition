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
from src.feature_engineering import build_engineered_features
from src.feature_selection import CORE_ENGINEERED_FEATURES, select_features
from src.fine_grained_eval import evaluate_recommender_fine_grained
from src.multi_metric_models import DualMetricRegressor
from src.pairwise_models import PairwiseLogisticRanker
from src.pareto_models import ParetoStrategyRecommender
from src.ranking_models import MeanUtilityByStrategy, RandomForestUtility
from src.stage2_eval import interception_priority_utility
from src.stage3_models import ClusterLocalUtilityRecommender


def representative_models():
    return [
        ("utility", MeanUtilityByStrategy()),
        ("utility", RidgeUtilityRegressor(alpha=1.0)),
        ("utility", RBFKernelUtilityRegressor(gamma=0.2, alpha=0.05)),
        ("utility", RandomForestUtility(n_trees=25, max_depth=6, min_samples_leaf=8)),
        ("cluster", ClusterLocalUtilityRecommender(n_clusters=3, metric1_weight=1.0, metric2_weight=0.05)),
        ("pairwise", PairwiseLogisticRanker(epochs=800, learning_rate=0.05, l2=0.003)),
        (
            "pareto",
            ParetoStrategyRecommender(
                DualMetricRegressor(RandomForestUtility(n_trees=25, max_depth=6, min_samples_leaf=8)),
                tie_breaker="product",
            ),
        ),
    ]


def run_suite(label: str, x, bundle, utility):
    rows = []
    for fit_mode, model in representative_models():
        metrics = evaluate_recommender_fine_grained(
            model,
            fit_mode,
            x,
            bundle.train_y,
            bundle.train_metric1,
            bundle.train_metric2,
            utility,
        )
        rows.append(
            {
                "feature_set": label,
                "family": fit_mode,
                "model": model.name,
                "top1_match": metrics.get("top1_match", 0.0),
                "top2_hit": metrics.get("top2_hit", 0.0),
                "soft_match": metrics.get("soft_match", 0.0),
                "regret": metrics.get("regret", 0.0),
                "pareto_hit": metrics.get("pareto_hit", 0.0),
                "mean_metric1_regret": metrics.get("mean_metric1_regret", ""),
                "p90_metric1_regret": metrics.get("p90_metric1_regret", ""),
                "ndcg": metrics.get("ndcg", 0.0),
                "mrr": metrics.get("mrr", 0.0),
            }
        )
    return rows


def main() -> None:
    bundle = load_competition_data("Data")
    output_dir = Path("outputs") / "stage20_core25_early_models"
    output_dir.mkdir(parents=True, exist_ok=True)

    utility = interception_priority_utility(bundle.train_metric1, bundle.train_metric2, 0.05)
    full_train_x, feature_names = build_engineered_features(bundle.train_x)
    core25_train_x, core25_names = select_features(full_train_x, feature_names, CORE_ENGINEERED_FEATURES)

    rows = []
    rows.extend(run_suite("raw42", bundle.train_x, bundle, utility))
    rows.extend(run_suite("raw42_plus_core25", core25_train_x, bundle, utility))
    rows = sorted(rows, key=lambda r: (r["feature_set"], r["soft_match"], r["top2_hit"], r["top1_match"]), reverse=True)

    fieldnames = [
        "feature_set",
        "family",
        "model",
        "top1_match",
        "top2_hit",
        "soft_match",
        "regret",
        "pareto_hit",
        "mean_metric1_regret",
        "p90_metric1_regret",
        "ndcg",
        "mrr",
    ]
    with (output_dir / "core25_early_model_cv_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fieldnames} for row in rows])

    by_model = {}
    for row in rows:
        by_model.setdefault(row["model"], {})[row["feature_set"]] = row
    deltas = []
    for model, parts in by_model.items():
        raw = parts.get("raw42")
        core = parts.get("raw42_plus_core25")
        if not raw or not core:
            continue
        deltas.append(
            {
                "model": model,
                "delta_top1": core["top1_match"] - raw["top1_match"],
                "delta_top2": core["top2_hit"] - raw["top2_hit"],
                "delta_soft_match": core["soft_match"] - raw["soft_match"],
                "delta_regret": core["regret"] - raw["regret"],
                "delta_pareto_hit": core["pareto_hit"] - raw["pareto_hit"],
            }
        )

    summary = {
        "objective": "interception_priority_utility",
        "core25_feature_count": len(core25_names),
        "core25_engineered_feature_count": len(core25_names) - bundle.train_x.shape[1],
        "best_by_soft_match": max(rows, key=lambda r: r["soft_match"]),
        "deltas_core25_minus_raw42": deltas,
        "selected_features": core25_names,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
