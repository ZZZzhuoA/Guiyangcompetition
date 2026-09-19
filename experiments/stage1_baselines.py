from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_utils import describe_bundle, load_competition_data
from src.evaluation import cross_validate
from src.models import (
    DecisionTreeStrategy,
    KNNStrategy,
    MajorityStrategy,
    OutcomeKNNRecommender,
    RandomForestStrategy,
    StrategyPrototype,
)


def main() -> None:
    bundle = load_competition_data("Data")
    models = [
        MajorityStrategy(),
        StrategyPrototype(metric="median"),
        StrategyPrototype(metric="mean"),
        KNNStrategy(k=3, distance_weighted=False),
        KNNStrategy(k=7, distance_weighted=True),
        KNNStrategy(k=15, distance_weighted=True),
        DecisionTreeStrategy(max_depth=4, min_samples_leaf=12),
        DecisionTreeStrategy(max_depth=7, min_samples_leaf=6),
        RandomForestStrategy(n_trees=25, max_depth=6, min_samples_leaf=8),
        OutcomeKNNRecommender(k=5, metric2_higher_is_better=True),
        OutcomeKNNRecommender(k=9, metric2_higher_is_better=True),
        OutcomeKNNRecommender(k=9, metric2_higher_is_better=False),
    ]

    output_dir = Path("outputs") / "stage1"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = cross_validate(models, bundle.train_x, bundle.train_y, bundle.train_metric1, bundle.train_metric2)
    results = sorted(results, key=lambda item: (item.accuracy_mean, item.macro_f1_mean), reverse=True)

    with (output_dir / "cv_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "accuracy_mean", "accuracy_std", "macro_f1_mean", "macro_f1_std"],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row.__dict__)

    best_name = results[0].model
    best_model = next(model for model in models if model.name == best_name)
    best_model.fit(bundle.train_x, bundle.train_y, bundle.train_metric1, bundle.train_metric2)
    test_pred = best_model.predict(bundle.test_x)

    with (output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "recommended_strategy"])
        for i, pred in enumerate(test_pred, start=1):
            writer.writerow([i, int(pred)])

    summary = describe_bundle(bundle)
    feature_keys = [tuple(row.astype(int).tolist()) for row in bundle.train_x]
    duplicate_groups = {}
    for idx, key in enumerate(feature_keys):
        duplicate_groups.setdefault(key, []).append(idx)
    ambiguous = [
        idxs for idxs in duplicate_groups.values() if len(set(int(bundle.train_y[i]) for i in idxs)) > 1
    ]
    summary["unique_train_scenarios"] = len(duplicate_groups)
    summary["ambiguous_scenarios"] = len(ambiguous)
    summary["rows_in_ambiguous_scenarios"] = int(sum(len(idxs) for idxs in ambiguous))
    summary["best_model"] = best_name
    summary["best_cv_accuracy"] = results[0].accuracy_mean
    summary["best_cv_macro_f1"] = results[0].macro_f1_mean
    summary["test_prediction_counts"] = {
        int(k): int(v) for k, v in zip(*np.unique(test_pred, return_counts=True))
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nCV ranking:")
    for row in results:
        print(
            f"{row.model}: acc={row.accuracy_mean:.4f}+/-{row.accuracy_std:.4f}, "
            f"macro_f1={row.macro_f1_mean:.4f}+/-{row.macro_f1_std:.4f}"
        )


if __name__ == "__main__":
    main()
