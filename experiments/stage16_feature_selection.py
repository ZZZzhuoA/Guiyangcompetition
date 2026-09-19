from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.stage15_pairwise_correction import evaluate_model, final_test_recommendations
from src.data_utils import load_competition_data
from src.feature_engineering import build_engineered_features
from src.feature_selection import (
    COMPACT_ENGINEERED_FEATURES,
    CORE_ENGINEERED_FEATURES,
    MATCH_ENGINEERED_FEATURES,
    RESOURCE_ENGINEERED_FEATURES,
    THREAT_ENGINEERED_FEATURES,
    select_features,
)
from src.pairwise_correction import PairwiseCorrectedUtility
from src.pairwise_models import PairwiseLogisticRanker
from src.ranking_models import RandomForestUtility
from src.stage2_eval import interception_priority_utility


def make_model() -> PairwiseCorrectedUtility:
    return PairwiseCorrectedUtility(
        base_model=RandomForestUtility(n_trees=15, max_depth=5, min_samples_leaf=10),
        pairwise_model=PairwiseLogisticRanker(epochs=80, learning_rate=0.08, l2=0.003),
        pairwise_weight=0.35,
        strategy3_penalty=0.05,
    )


def flatten_counts(counts: dict[int, int]) -> str:
    return json.dumps({int(k): int(v) for k, v in counts.items()}, ensure_ascii=False)


def main() -> None:
    bundle = load_competition_data("Data")
    output_dir = Path("outputs") / "stage16_feature_selection"
    output_dir.mkdir(parents=True, exist_ok=True)

    utility = interception_priority_utility(bundle.train_metric1, bundle.train_metric2, 0.05)
    full_train_x, feature_names = build_engineered_features(bundle.train_x)
    full_test_x, _ = build_engineered_features(bundle.test_x)

    feature_sets = {
        "raw42": [],
        "raw42_plus_threat9": THREAT_ENGINEERED_FEATURES,
        "raw42_plus_resource12": RESOURCE_ENGINEERED_FEATURES,
        "raw42_plus_match10": MATCH_ENGINEERED_FEATURES,
        "raw42_plus_compact15": COMPACT_ENGINEERED_FEATURES,
        "raw42_plus_core25": CORE_ENGINEERED_FEATURES,
        "raw42_plus_all59": feature_names[42:],
    }

    rows = []
    selected_matrices = {}
    for label, selected in feature_sets.items():
        if selected:
            train_x, names = select_features(full_train_x, feature_names, selected)
            test_x, _ = select_features(full_test_x, feature_names, selected)
        else:
            train_x, names = bundle.train_x, feature_names[:42]
            test_x = bundle.test_x
        selected_matrices[label] = (train_x, test_x, names)
        metrics = evaluate_model(make_model(), train_x, bundle.train_y, bundle.train_metric1, bundle.train_metric2, utility, max_folds=None)
        rows.append(
            {
                "feature_set": label,
                "total_feature_count": len(names),
                "engineered_feature_count": max(0, len(names) - 42),
                "top1_match": metrics.get("top1_match", 0.0),
                "top2_hit": metrics.get("top2_hit", 0.0),
                "soft_match": metrics.get("soft_match", 0.0),
                "regret": metrics.get("regret", 0.0),
                "pareto_hit": metrics.get("pareto_hit", 0.0),
                "mean_metric1_regret": metrics.get("mean_metric1_regret", 0.0),
                "p90_metric1_regret": metrics.get("p90_metric1_regret", 0.0),
                "strategy3_over_prediction": metrics.get("strategy3_over_prediction", 0),
                "pred_strategy_counts": flatten_counts(metrics.get("pred_strategy_counts", {})),
                "selected_features": ";".join(selected),
            }
        )

    rows = sorted(rows, key=lambda r: (r["soft_match"], r["top2_hit"], r["top1_match"], -r["regret"]), reverse=True)
    with (output_dir / "feature_selection_cv_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]
    best_train_x, best_test_x, best_names = selected_matrices[best["feature_set"]]
    test_info = final_test_recommendations(bundle, best_train_x, best_test_x, utility, make_model(), output_dir)

    summary = {
        "best_feature_set": best["feature_set"],
        "best_cv": best,
        "selected_feature_count": int(best["total_feature_count"]),
        "selected_engineered_feature_count": int(best["engineered_feature_count"]),
        "selected_feature_names": best_names,
        **test_info,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
