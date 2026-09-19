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
from src.pairwise_models import ClusterPairwisePreference, PairwiseLogisticRanker
from src.stage2_eval import group_folds, scenario_keys, utility_from_metrics
from src.stage3_eval import evaluate_against_global_utility
from src.stage3_models import ClusterLocalUtilityRecommender


def evaluate_pairwise(models, x, strategy, utility):
    keys = scenario_keys(x)
    splits = group_folds(keys)
    rows = []
    for model in models:
        top1 = []
        regrets = []
        chosen_utils = []
        best_utils = []
        recs = []
        for train_idx, valid_idx in splits:
            fitted = model.__class__(**getattr(model, "__dict__", {}))
            fitted.fit(x[train_idx], strategy[train_idx], utility[train_idx])
            by_key = {}
            for idx in valid_idx:
                by_key.setdefault(keys[idx], []).append(int(idx))
            for idxs in by_key.values():
                if len(idxs) < 2:
                    continue
                candidate_strategies = strategy[idxs].astype(int)
                candidate_x = np.repeat(x[idxs[0]].reshape(1, -1), len(candidate_strategies), axis=0)
                if isinstance(fitted, ClusterPairwisePreference):
                    pred, scores, _ = fitted.recommend(candidate_x, candidate_strategies)
                    chosen_strategy = int(candidate_strategies[np.argmax(scores[0])])
                else:
                    pred, scores = fitted.recommend(candidate_x[:1], candidate_strategies)
                    chosen_strategy = int(pred[0])
                best_pos = int(np.argmax(utility[idxs]))
                best_strategy = int(candidate_strategies[best_pos])
                chosen_matches = np.where(candidate_strategies == chosen_strategy)[0]
                if len(chosen_matches) == 0:
                    continue
                chosen_pos = int(chosen_matches[0])
                recs.append(chosen_strategy)
                top1.append(1.0 if chosen_strategy == best_strategy else 0.0)
                chosen_utility = float(utility[idxs][chosen_pos])
                best_utility = float(utility[idxs][best_pos])
                regrets.append(best_utility - chosen_utility)
                chosen_utils.append(chosen_utility)
                best_utils.append(best_utility)
        rows.append(
            {
                "family": "pairwise",
                "model": model.name,
                "top1_match": float(np.mean(top1)) if top1 else 0.0,
                "mean_regret": float(np.mean(regrets)) if regrets else 0.0,
                "mean_chosen_utility": float(np.mean(chosen_utils)) if chosen_utils else 0.0,
                "mean_best_utility": float(np.mean(best_utils)) if best_utils else 0.0,
                "used_strategy_count": int(len(set(recs))),
            }
        )
    return rows


def main() -> None:
    bundle = load_competition_data("Data")
    output_dir = Path("outputs") / "stage6_pairwise"
    output_dir.mkdir(parents=True, exist_ok=True)
    utility = utility_from_metrics(bundle.train_metric1, bundle.train_metric2, 0.5, 0.5)

    pairwise_models = [
        PairwiseLogisticRanker(epochs=400, learning_rate=0.06, l2=0.001),
        PairwiseLogisticRanker(epochs=800, learning_rate=0.05, l2=0.003),
        ClusterPairwisePreference(n_clusters=3, shrinkage=4.0),
        ClusterPairwisePreference(n_clusters=4, shrinkage=4.0),
        ClusterPairwisePreference(n_clusters=6, shrinkage=6.0),
    ]
    rows = evaluate_pairwise(pairwise_models, bundle.train_x, bundle.train_y, utility)

    cluster_baseline = ClusterLocalUtilityRecommender(n_clusters=3, metric1_weight=0.5, metric2_weight=0.5)
    baseline = evaluate_against_global_utility(
        [cluster_baseline],
        bundle.train_x,
        bundle.train_y,
        bundle.train_metric1,
        bundle.train_metric2,
        utility,
    )[0]
    rows.append(
        {
            "family": "cluster_local_baseline",
            "model": baseline.model,
            "top1_match": baseline.top1_match,
            "mean_regret": baseline.mean_regret,
            "mean_chosen_utility": baseline.mean_chosen_utility,
            "mean_best_utility": baseline.mean_best_utility,
            "used_strategy_count": baseline.used_strategy_count,
        }
    )

    rows = sorted(rows, key=lambda item: (item["top1_match"], -item["mean_regret"]), reverse=True)
    with (output_dir / "pairwise_cv_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]
    if best["family"] == "cluster_local_baseline":
        final_model = ClusterLocalUtilityRecommender(n_clusters=3, metric1_weight=0.5, metric2_weight=0.5)
        final_model.fit(bundle.train_x, bundle.train_y, bundle.train_metric1, bundle.train_metric2)
        test_pred, scores, clusters, weights = final_model.recommend(bundle.test_x)
    else:
        final_model = next(model for model in pairwise_models if model.name == best["model"])
        final_model.fit(bundle.train_x, bundle.train_y, utility)
        if isinstance(final_model, ClusterPairwisePreference):
            test_pred, scores, clusters = final_model.recommend(bundle.test_x)
        else:
            test_pred, scores = final_model.recommend(bundle.test_x)
            clusters = np.full(len(test_pred), -1, dtype=int)
        weights = np.full((len(test_pred), 2), 0.5, dtype=float)

    with (output_dir / "test_recommendations.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "cluster", "w_interception", "w_cost_effectiveness", "recommended_strategy", "score_s1", "score_s2", "score_s3", "score_s4"])
        for i, (cluster, weight, pred, score_row) in enumerate(zip(clusters, weights, test_pred, scores), start=1):
            writer.writerow([i, int(cluster), float(weight[0]), float(weight[1]), int(pred), *[float(v) for v in score_row]])

    summary = {
        "idea": "pairwise strategy ranking optimizes whether one strategy beats another in the same scenario",
        "best_family": best["family"],
        "best_model": best["model"],
        "best_top1_match": best["top1_match"],
        "best_mean_regret": best["mean_regret"],
        "test_prediction_counts": {int(k): int(v) for k, v in zip(*np.unique(test_pred, return_counts=True))},
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nPairwise ranking:")
    for row in rows:
        print(f"{row['family']}/{row['model']}: top1={row['top1_match']:.4f}, regret={row['mean_regret']:.4f}, strategies={row['used_strategy_count']}")


if __name__ == "__main__":
    main()
