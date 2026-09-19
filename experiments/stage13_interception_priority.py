from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_utils import load_competition_data
from src.exact_scene_recommender import ExactSceneHybridRecommender
from src.feature_engineering import build_engineered_features
from src.fine_grained_eval import aggregate_metric_rows, evaluate_recommender_fine_grained, evaluate_scenario_choice
from src.ranking_models import RandomForestUtility
from src.stage2_eval import interception_priority_utility, scenario_keys


SECONDARY_WEIGHT = 0.05


def exact_scene_consistency(
    x: np.ndarray,
    strategy: np.ndarray,
    metric1: np.ndarray,
    metric2: np.ndarray,
    utility: np.ndarray,
) -> dict[str, float]:
    keys = scenario_keys(x)
    by_key: dict[tuple[int, ...], list[int]] = {}
    for idx, key in enumerate(keys):
        by_key.setdefault(key, []).append(idx)

    metric_rows = []
    covered_groups = 0
    complete_strategy_groups = 0
    for idxs in by_key.values():
        candidate_strategies = np.array(sorted(set(int(strategy[i]) for i in idxs)), dtype=int)
        if len(candidate_strategies) < 2:
            continue
        covered_groups += 1
        if len(candidate_strategies) == 4:
            complete_strategy_groups += 1

        true_utility = []
        true_m1 = []
        true_m2 = []
        for s in candidate_strategies:
            s_idxs = [i for i in idxs if int(strategy[i]) == int(s)]
            true_utility.append(float(np.mean(utility[s_idxs])))
            true_m1.append(float(np.mean(metric1[s_idxs])))
            true_m2.append(float(np.mean(metric2[s_idxs])))
        true_utility = np.array(true_utility, dtype=float)
        objectives = np.vstack([true_m1, true_m2]).T

        chosen_strategy = int(candidate_strategies[np.argmax(true_utility)])
        row = evaluate_scenario_choice(candidate_strategies, true_utility, chosen_strategy, true_utility, objectives)
        if row:
            metric_rows.append(row)

    result = aggregate_metric_rows(metric_rows)
    result["covered_train_scenarios"] = float(covered_groups)
    result["complete_strategy_train_scenarios"] = float(complete_strategy_groups)
    return result


def build_exact_match_table(bundle, utility: np.ndarray, output_dir: Path) -> dict[str, object]:
    train_map: dict[tuple[int, ...], list[int]] = {}
    for idx, key in enumerate(scenario_keys(bundle.train_x)):
        train_map.setdefault(key, []).append(idx)

    rows = []
    matched_ids = set()
    for j, key in enumerate(scenario_keys(bundle.test_x)):
        idxs = train_map.get(key, [])
        if not idxs:
            continue
        matched_ids.add(j + 1)
        strategies = sorted(set(int(bundle.train_y[i]) for i in idxs))
        scored = sorted(idxs, key=lambda i: (float(bundle.train_metric1[i]), float(bundle.train_metric2[i])), reverse=True)
        priority_scored = sorted(idxs, key=lambda i: float(utility[i]), reverse=True)
        best_idx = priority_scored[0]
        best_lex_idx = scored[0]
        rows.append(
            {
                "test_sample_id": j + 1,
                "train_match_count": len(idxs),
                "train_rows_1based": ";".join(str(i + 1) for i in idxs),
                "strategies_seen": ";".join(str(s) for s in strategies),
                "best_priority_strategy": int(bundle.train_y[best_idx]),
                "best_priority_metric1": round(float(bundle.train_metric1[best_idx]), 6),
                "best_priority_metric2": round(float(bundle.train_metric2[best_idx]), 6),
                "best_priority_score": round(float(utility[best_idx]), 6),
                "best_lexicographic_strategy": int(bundle.train_y[best_lex_idx]),
                "best_lexicographic_metric1": round(float(bundle.train_metric1[best_lex_idx]), 6),
                "best_lexicographic_metric2": round(float(bundle.train_metric2[best_lex_idx]), 6),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "test_exact_feature_matches_in_train.csv", index=False, encoding="utf-8-sig")
    return {
        "matched_test_rows": int(len(table)),
        "unmatched_test_rows": int(len(bundle.test_x) - len(table)),
        "matched_test_sample_ids": [int(v) for v in table["test_sample_id"].tolist()],
        "unmatched_test_sample_ids": [int(i + 1) for i in range(len(bundle.test_x)) if (i + 1) not in matched_ids],
        "match_count_distribution": {str(k): int(v) for k, v in table["train_match_count"].value_counts().sort_index().items()},
    }


def main() -> None:
    bundle = load_competition_data("Data")
    output_dir = Path("outputs") / "stage13_interception_priority"
    output_dir.mkdir(parents=True, exist_ok=True)

    utility = interception_priority_utility(bundle.train_metric1, bundle.train_metric2, SECONDARY_WEIGHT)
    engineered_train_x, feature_names = build_engineered_features(bundle.train_x)
    engineered_test_x, _ = build_engineered_features(bundle.test_x)

    base_model = RandomForestUtility(n_trees=25, max_depth=6, min_samples_leaf=8)
    base_metrics = evaluate_recommender_fine_grained(
        base_model,
        "utility",
        engineered_train_x,
        bundle.train_y,
        bundle.train_metric1,
        bundle.train_metric2,
        utility,
    )
    base_metrics["model"] = "interception_priority_engineered_reg_forest_t25_d6"
    base_metrics["evaluation"] = "group_cv_unseen_scene"

    exact_metrics = exact_scene_consistency(bundle.train_x, bundle.train_y, bundle.train_metric1, bundle.train_metric2, utility)
    exact_metrics["model"] = "interception_priority_exact_scene_table"
    exact_metrics["evaluation"] = "seen_scene_consistency"

    coverage = build_exact_match_table(bundle, utility, output_dir)

    final_model = ExactSceneHybridRecommender(
        base_model=RandomForestUtility(n_trees=25, max_depth=6, min_samples_leaf=8),
        min_exact_strategies=2,
        exact_weight=0.85,
    )
    final_model.fit(
        bundle.train_x,
        engineered_train_x,
        bundle.train_y,
        utility,
        bundle.train_metric1,
        bundle.train_metric2,
    )
    pred, scores, sources, exact_counts = final_model.recommend(bundle.test_x, engineered_test_x)

    with (output_dir / "test_recommendations.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_id",
                "recommended_strategy",
                "priority_score_s1",
                "priority_score_s2",
                "priority_score_s3",
                "priority_score_s4",
                "source",
                "exact_count_s1",
                "exact_count_s2",
                "exact_count_s3",
                "exact_count_s4",
            ]
        )
        for i, (p, score_row, source, count_row) in enumerate(zip(pred, scores, sources, exact_counts), start=1):
            writer.writerow([i, int(p), *[round(float(v), 8) for v in score_row], source, *[int(v) for v in count_row]])

    rows = [base_metrics, exact_metrics]
    fieldnames = sorted(set().union(*(row.keys() for row in rows)))
    with (output_dir / "stage13_cv_and_consistency.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    source_counts = {source: int(count) for source, count in zip(*np.unique(np.array(sources), return_counts=True))}
    prediction_counts = {int(k): int(v) for k, v in zip(*np.unique(pred, return_counts=True))}
    summary = {
        "method": "interception_priority_exact_scene_hybrid_with_engineered_rf_fallback",
        "objective_rule": "maximize interception rate first; use cost-effectiveness as secondary tie-breaker",
        "secondary_metric_weight": SECONDARY_WEIGHT,
        "feature_definition": "42 original full-scene features plus 59 engineered features for fallback model",
        "base_group_cv": base_metrics,
        "seen_scene_consistency": exact_metrics,
        "test_exact_coverage": coverage,
        "test_source_counts": source_counts,
        "test_prediction_counts": prediction_counts,
        "engineered_feature_count": int(engineered_train_x.shape[1] - bundle.train_x.shape[1]),
        "total_fallback_feature_count": int(engineered_train_x.shape[1]),
        "engineered_features": feature_names[42:],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
