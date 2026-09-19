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
from src.data_utils import load_competition_data
from src.feature_engineering import build_engineered_features
from src.feature_selection import CORE_ENGINEERED_FEATURES, select_features
from src.pairwise_correction import PairwiseCorrectedUtility
from src.pairwise_models import PairwiseLogisticRanker
from src.ranking_models import RandomForestUtility, recommend_strategy
from src.stage2_eval import interception_priority_utility


def make_model() -> PairwiseCorrectedUtility:
    return PairwiseCorrectedUtility(
        base_model=RandomForestUtility(n_trees=15, max_depth=5, min_samples_leaf=10),
        pairwise_model=PairwiseLogisticRanker(epochs=80, learning_rate=0.08, l2=0.003),
        pairwise_weight=0.35,
        strategy3_penalty=0.05,
    )


def main() -> None:
    bundle = load_competition_data("Data")
    output_dir = Path("outputs") / "stage17_model_only_generalization"
    output_dir.mkdir(parents=True, exist_ok=True)

    utility = interception_priority_utility(bundle.train_metric1, bundle.train_metric2, 0.05)
    full_train_x, feature_names = build_engineered_features(bundle.train_x)
    full_test_x, _ = build_engineered_features(bundle.test_x)
    train_x, selected_names = select_features(full_train_x, feature_names, CORE_ENGINEERED_FEATURES)
    test_x, _ = select_features(full_test_x, feature_names, CORE_ENGINEERED_FEATURES)

    cv_metrics = evaluate_model(make_model(), train_x, bundle.train_y, bundle.train_metric1, bundle.train_metric2, utility, max_folds=None)

    model = make_model()
    model.fit(train_x, bundle.train_y, utility)
    pred, scores = recommend_strategy(model, test_x)

    with (output_dir / "test_recommendations.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "recommended_strategy", "score_s1", "score_s2", "score_s3", "score_s4"])
        for i, (p, score_row) in enumerate(zip(pred, scores), start=1):
            writer.writerow([i, int(p), *[round(float(v), 8) for v in score_row]])

    summary = {
        "method": "model_only_interception_priority_core25_pairwise_corrected",
        "uses_exact_scene_lookup": False,
        "feature_set": "raw42_plus_core25",
        "total_feature_count": int(train_x.shape[1]),
        "engineered_feature_count": int(train_x.shape[1] - bundle.train_x.shape[1]),
        "selected_features": selected_names,
        "cv_metrics": cv_metrics,
        "test_prediction_counts": {int(k): int(v) for k, v in zip(*np.unique(pred, return_counts=True))},
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
