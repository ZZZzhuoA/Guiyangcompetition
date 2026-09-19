from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stage15_pairwise_correction import evaluate_model
from src.advanced_ml_models import RBFKernelUtilityRegressor, RidgeUtilityRegressor
from src.data_utils import load_competition_data
from src.feature_engineering import build_engineered_features
from src.feature_selection import CORE_ENGINEERED_FEATURES, select_features
from src.pairwise_correction import PairwiseCorrectedUtility
from src.pairwise_models import PairwiseLogisticRanker
from src.ranking_models import KNNUtilityRegressor, RandomForestUtility, recommend_strategy
from src.stage2_eval import interception_priority_utility
from src.voting_ensemble import VotingUtilityEnsemble


def base_models() -> list[object]:
    return [
        RidgeUtilityRegressor(alpha=1.0),
        RBFKernelUtilityRegressor(gamma=0.2, alpha=0.05),
        KNNUtilityRegressor(k=15),
        RandomForestUtility(n_trees=15, max_depth=5, min_samples_leaf=10),
        PairwiseCorrectedUtility(
            base_model=RandomForestUtility(n_trees=15, max_depth=5, min_samples_leaf=10),
            pairwise_model=PairwiseLogisticRanker(epochs=80, learning_rate=0.08, l2=0.003),
            pairwise_weight=0.35,
            strategy3_penalty=0.05,
        ),
    ]


def main() -> None:
    bundle = load_competition_data("Data")
    output_dir = Path("outputs") / "stage18_voting_ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)

    utility = interception_priority_utility(bundle.train_metric1, bundle.train_metric2, 0.05)
    full_train_x, feature_names = build_engineered_features(bundle.train_x)
    full_test_x, _ = build_engineered_features(bundle.test_x)
    train_x, selected_names = select_features(full_train_x, feature_names, CORE_ENGINEERED_FEATURES)
    test_x, _ = select_features(full_test_x, feature_names, CORE_ENGINEERED_FEATURES)

    models = base_models()
    candidates = []
    candidates.extend(models)
    candidates.extend(
        [
            VotingUtilityEnsemble(models=base_models(), mode="soft"),
            VotingUtilityEnsemble(models=base_models(), mode="hard"),
            VotingUtilityEnsemble(models=base_models(), mode="rank"),
        ]
    )

    rows = []
    for model in candidates:
        metrics = evaluate_model(model, train_x, bundle.train_y, bundle.train_metric1, bundle.train_metric2, utility, max_folds=None)
        rows.append(
            {
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
        )

    rows = sorted(rows, key=lambda r: (r["soft_match"], r["top1_match"], r["top2_hit"], -r["regret"]), reverse=True)
    with (output_dir / "voting_ensemble_cv_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best_name = rows[0]["model"]
    best_model = next(model for model in candidates if model.name == best_name)
    best_model.fit(train_x, bundle.train_y, utility)
    pred, scores = recommend_strategy(best_model, test_x)
    with (output_dir / "test_recommendations.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "recommended_strategy", "score_s1", "score_s2", "score_s3", "score_s4"])
        for i, (p, score_row) in enumerate(zip(pred, scores), start=1):
            writer.writerow([i, int(p), *[round(float(v), 8) for v in score_row]])

    summary = {
        "best_model": best_name,
        "best_cv": rows[0],
        "feature_set": "raw42_plus_core25",
        "selected_feature_count": int(train_x.shape[1]),
        "selected_engineered_feature_count": int(train_x.shape[1] - bundle.train_x.shape[1]),
        "selected_features": selected_names,
        "test_prediction_counts": {int(k): int(v) for k, v in zip(*np.unique(pred, return_counts=True))},
        "voting_members": [model.name for model in base_models()],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
