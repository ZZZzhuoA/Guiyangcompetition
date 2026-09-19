from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.advanced_ml_models import RBFKernelUtilityRegressor, RidgeUtilityRegressor
from src.data_utils import load_competition_data
from src.feature_engineering import build_engineered_features
from src.fine_grained_eval import evaluate_recommender_fine_grained
from src.fusion_models import FusionStrategyRecommender
from src.ranking_models import RandomForestUtility
from src.stage2_eval import utility_from_metrics
from src.stage3_models import ClusterLocalUtilityRecommender


def run_suite(label: str, x: np.ndarray, bundle, utility: np.ndarray) -> list[dict[str, object]]:
    candidates = [
        ("utility", RidgeUtilityRegressor(alpha=1.0)),
        ("utility", RBFKernelUtilityRegressor(gamma=0.2, alpha=0.05)),
        ("utility", RandomForestUtility(n_trees=25, max_depth=6, min_samples_leaf=8)),
        ("cluster", ClusterLocalUtilityRecommender(n_clusters=3, metric1_weight=0.5, metric2_weight=0.5)),
    ]
    rows = []
    for fit_mode, model in candidates:
        metrics = evaluate_recommender_fine_grained(
            model,
            fit_mode,
            x,
            bundle.train_y,
            bundle.train_metric1,
            bundle.train_metric2,
            utility,
        )
        metrics["feature_set"] = label
        metrics["family"] = fit_mode
        metrics["model"] = model.name
        rows.append(metrics)
    return rows


def main() -> None:
    bundle = load_competition_data("Data")
    output_dir = Path("outputs") / "stage11_feature_engineering"
    output_dir.mkdir(parents=True, exist_ok=True)
    utility = utility_from_metrics(bundle.train_metric1, bundle.train_metric2, 0.5, 0.5)

    engineered_train_x, feature_names = build_engineered_features(bundle.train_x)
    engineered_test_x, _ = build_engineered_features(bundle.test_x)

    rows = []
    rows.extend(run_suite("raw", bundle.train_x, bundle, utility))
    rows.extend(run_suite("engineered", engineered_train_x, bundle, utility))
    rows = sorted(rows, key=lambda row: (row.get("soft_match", 0), -row.get("regret", 999)), reverse=True)

    fieldnames = [
        "feature_set",
        "family",
        "model",
        "top1_match",
        "top2_hit",
        "soft_match",
        "regret",
        "relative_regret",
        "rank_percentile",
        "ndcg",
        "mrr",
        "pareto_hit",
    ]
    with (output_dir / "feature_engineering_cv_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    best = rows[0]
    # Use the best engineered/raw non-fusion model to create a recommendation table.
    if best["feature_set"] == "engineered":
        final_x = engineered_train_x
        final_test_x = engineered_test_x
    else:
        final_x = bundle.train_x
        final_test_x = bundle.test_x
    if best["family"] == "cluster":
        model = ClusterLocalUtilityRecommender(n_clusters=3, metric1_weight=0.5, metric2_weight=0.5)
        model.fit(final_x, bundle.train_y, bundle.train_metric1, bundle.train_metric2)
        pred, scores, clusters, weights = model.recommend(final_test_x)
    elif "rbf_kernel" in best["model"]:
        model = RBFKernelUtilityRegressor(gamma=0.2, alpha=0.05)
        model.fit(final_x, bundle.train_y, utility)
        pred, scores = recommend_from_regressor(model, final_test_x)
        clusters = np.full(len(pred), -1, dtype=int)
    elif "ridge" in best["model"]:
        model = RidgeUtilityRegressor(alpha=1.0)
        model.fit(final_x, bundle.train_y, utility)
        pred, scores = recommend_from_regressor(model, final_test_x)
        clusters = np.full(len(pred), -1, dtype=int)
    else:
        model = RandomForestUtility(n_trees=25, max_depth=6, min_samples_leaf=8)
        model.fit(final_x, bundle.train_y, utility)
        pred, scores = recommend_from_regressor(model, final_test_x)
        clusters = np.full(len(pred), -1, dtype=int)

    with (output_dir / "test_recommendations.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "recommended_strategy", "score_s1", "score_s2", "score_s3", "score_s4", "cluster"])
        for i, (p, score_row, cluster) in enumerate(zip(pred, scores, clusters), start=1):
            writer.writerow([i, int(p), *[float(v) for v in score_row], int(cluster)])

    summary = {
        "engineered_feature_count": int(engineered_train_x.shape[1] - bundle.train_x.shape[1]),
        "total_feature_count": int(engineered_train_x.shape[1]),
        "best_feature_set": best["feature_set"],
        "best_model": best["model"],
        "best_soft_match": best.get("soft_match"),
        "best_top1_match": best.get("top1_match"),
        "best_regret": best.get("regret"),
        "test_prediction_counts": {int(k): int(v) for k, v in zip(*np.unique(pred, return_counts=True))},
        "engineered_features": feature_names[42:],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nFeature engineering ranking:")
    for row in rows:
        print(
            f"{row['feature_set']}/{row['family']}/{row['model']}: "
            f"soft={row.get('soft_match', 0):.4f}, top1={row.get('top1_match', 0):.4f}, "
            f"regret={row.get('regret', 0):.4f}, pareto={row.get('pareto_hit', 0):.4f}"
        )


def recommend_from_regressor(model, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    strategies = np.array([1, 2, 3, 4], dtype=int)
    preds = []
    scores = []
    for row in x:
        candidate_x = np.repeat(row.reshape(1, -1), len(strategies), axis=0)
        score = model.predict(candidate_x, strategies)
        scores.append(score)
        preds.append(int(strategies[np.argmax(score)]))
    return np.array(preds, dtype=int), np.vstack(scores)


if __name__ == "__main__":
    main()
