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
from src.multi_metric_models import DualMetricRegressor
from src.pareto_models import ParetoStrategyRecommender, pareto_mask_maximize
from src.ranking_models import KNNUtilityRegressor, RandomForestUtility
from src.stage2_eval import group_folds, scenario_keys, utility_from_metrics
from src.stage3_eval import evaluate_against_global_utility
from src.stage3_models import ClusterLocalUtilityRecommender


def evaluate_pareto(models, x, strategy, metric1, metric2):
    keys = scenario_keys(x)
    splits = group_folds(keys)
    eval_utility = utility_from_metrics(metric1, metric2, 0.5, 0.5)
    rows = []
    for model in models:
        top1 = []
        regret = []
        pareto_hit = []
        chosen_m1 = []
        chosen_m2 = []
        best_m1 = []
        best_m2 = []
        for train_idx, valid_idx in splits:
            fitted = model.__class__(**getattr(model, "__dict__", {}))
            fitted.fit(x[train_idx], strategy[train_idx], metric1[train_idx], metric2[train_idx])
            by_key = {}
            for idx in valid_idx:
                by_key.setdefault(keys[idx], []).append(int(idx))
            for idxs in by_key.values():
                if len(idxs) < 2:
                    continue
                candidate_strategies = strategy[idxs].astype(int)
                candidate_x = np.repeat(x[idxs[0]].reshape(1, -1), len(candidate_strategies), axis=0)
                pred, _, _, _ = fitted.recommend(candidate_x[:1], candidate_strategies)
                chosen_strategy = int(pred[0])
                chosen_matches = np.where(candidate_strategies == chosen_strategy)[0]
                if len(chosen_matches) == 0:
                    continue
                chosen_pos = int(chosen_matches[0])
                objectives = np.vstack([metric1[idxs], metric2[idxs]]).T
                pareto = pareto_mask_maximize(objectives)
                best_pos = int(np.argmax(eval_utility[idxs]))
                top1.append(1.0 if chosen_strategy == int(candidate_strategies[best_pos]) else 0.0)
                pareto_hit.append(1.0 if pareto[chosen_pos] else 0.0)
                regret.append(float(eval_utility[idxs][best_pos] - eval_utility[idxs][chosen_pos]))
                chosen_m1.append(float(metric1[idxs][chosen_pos]))
                chosen_m2.append(float(metric2[idxs][chosen_pos]))
                best_m1.append(float(np.max(metric1[idxs])))
                best_m2.append(float(np.max(metric2[idxs])))
        rows.append(
            {
                "family": "pareto_multiobjective",
                "model": model.name,
                "top1_match_equal_score": float(np.mean(top1)) if top1 else 0.0,
                "pareto_hit_rate": float(np.mean(pareto_hit)) if pareto_hit else 0.0,
                "mean_regret_equal_score": float(np.mean(regret)) if regret else 0.0,
                "mean_chosen_metric1": float(np.mean(chosen_m1)) if chosen_m1 else 0.0,
                "mean_chosen_metric2": float(np.mean(chosen_m2)) if chosen_m2 else 0.0,
                "mean_best_available_metric1": float(np.mean(best_m1)) if best_m1 else 0.0,
                "mean_best_available_metric2": float(np.mean(best_m2)) if best_m2 else 0.0,
            }
        )
    return rows


def main() -> None:
    bundle = load_competition_data("Data")
    output_dir = Path("outputs") / "stage7_pareto"
    output_dir.mkdir(parents=True, exist_ok=True)

    dual_bases = [
        RidgeUtilityRegressor(alpha=0.1),
        RidgeUtilityRegressor(alpha=1.0),
        RBFKernelUtilityRegressor(gamma=0.2, alpha=0.05),
        KNNUtilityRegressor(k=15),
        RandomForestUtility(n_trees=25, max_depth=6, min_samples_leaf=8),
    ]
    models = []
    for base in dual_bases:
        for tie_breaker in ["equal_weight", "product", "minimax"]:
            models.append(ParetoStrategyRecommender(DualMetricRegressor(base), tie_breaker=tie_breaker))

    rows = evaluate_pareto(models, bundle.train_x, bundle.train_y, bundle.train_metric1, bundle.train_metric2)

    eval_utility = utility_from_metrics(bundle.train_metric1, bundle.train_metric2, 0.5, 0.5)
    baseline = evaluate_against_global_utility(
        [ClusterLocalUtilityRecommender(n_clusters=3, metric1_weight=0.5, metric2_weight=0.5)],
        bundle.train_x,
        bundle.train_y,
        bundle.train_metric1,
        bundle.train_metric2,
        eval_utility,
    )[0]
    rows.append(
        {
            "family": "cluster_local_baseline",
            "model": baseline.model,
            "top1_match_equal_score": baseline.top1_match,
            "pareto_hit_rate": "",
            "mean_regret_equal_score": baseline.mean_regret,
            "mean_chosen_metric1": "",
            "mean_chosen_metric2": "",
            "mean_best_available_metric1": "",
            "mean_best_available_metric2": "",
        }
    )

    rows = sorted(
        rows,
        key=lambda item: (
            item["pareto_hit_rate"] if item["pareto_hit_rate"] != "" else -1,
            item["top1_match_equal_score"],
            -item["mean_regret_equal_score"],
        ),
        reverse=True,
    )

    with (output_dir / "pareto_cv_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]
    final_model = next(model for model in models if model.name == best["model"])
    final_model.fit(bundle.train_x, bundle.train_y, bundle.train_metric1, bundle.train_metric2)
    test_pred, utility_scores, pred_m1, pred_m2 = final_model.recommend(bundle.test_x)

    with (output_dir / "test_recommendations.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id",
            "recommended_strategy",
            "utility_s1",
            "utility_s2",
            "utility_s3",
            "utility_s4",
            "pred_interception_s1",
            "pred_interception_s2",
            "pred_interception_s3",
            "pred_interception_s4",
            "pred_cost_effect_s1",
            "pred_cost_effect_s2",
            "pred_cost_effect_s3",
            "pred_cost_effect_s4",
        ])
        for i, (pred, u, m1, m2) in enumerate(zip(test_pred, utility_scores, pred_m1, pred_m2), start=1):
            writer.writerow([i, int(pred), *[float(v) for v in u], *[float(v) for v in m1], *[float(v) for v in m2]])

    summary = {
        "idea": "multi-objective recommendation: maximize interception rate and cost-effectiveness; Pareto filtering before tie-breaking",
        "best_model": best["model"],
        "best_pareto_hit_rate": best["pareto_hit_rate"],
        "best_top1_match_equal_score": best["top1_match_equal_score"],
        "best_mean_regret_equal_score": best["mean_regret_equal_score"],
        "test_prediction_counts": {int(k): int(v) for k, v in zip(*np.unique(test_pred, return_counts=True))},
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nPareto multi-objective:")
    for row in rows[:12]:
        print(
            f"{row['family']}/{row['model']}: pareto={row['pareto_hit_rate']}, "
            f"top1={row['top1_match_equal_score']:.4f}, regret={row['mean_regret_equal_score']:.4f}"
        )


if __name__ == "__main__":
    main()
