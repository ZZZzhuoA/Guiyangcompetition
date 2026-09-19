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
from src.dual_view_models import DualViewClusterRecommender
from src.fine_grained_eval import evaluate_recommender_fine_grained
from src.stage2_eval import group_folds, scenario_keys, utility_from_metrics
from src.stage3_eval import evaluate_against_global_utility
from src.stage3_models import ClusterLocalUtilityRecommender


def evaluate_dual_view(models, x, strategy, metric1, metric2, utility):
    keys = scenario_keys(x)
    splits = group_folds(keys)
    rows = []
    for model in models:
        metric_rows = []
        recs = []
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
                pred, scores, _, _, _, _ = fitted.recommend(candidate_x[:1], candidate_strategies)
                chosen_strategy = int(pred[0])
                recs.append(chosen_strategy)
                true_utility = utility[idxs]
                best_utility = float(np.max(true_utility))
                worst_utility = float(np.min(true_utility))
                chosen_pos = int(np.where(candidate_strategies == chosen_strategy)[0][0])
                chosen_utility = float(true_utility[chosen_pos])
                rank = 1 + int(np.sum(true_utility > chosen_utility))
                denom = best_utility - worst_utility
                metric_rows.append(
                    {
                        "top1_match": 1.0 if abs(chosen_utility - best_utility) < 1e-12 else 0.0,
                        "top2_hit": 1.0 if chosen_utility >= np.sort(true_utility)[::-1][min(1, len(true_utility) - 1)] - 1e-12 else 0.0,
                        "soft_match": 1.0 if abs(denom) < 1e-12 else (chosen_utility - worst_utility) / denom,
                        "regret": best_utility - chosen_utility,
                        "rank_percentile": 1.0 if len(true_utility) == 1 else 1.0 - (rank - 1) / (len(true_utility) - 1),
                    }
                )
        rows.append(
            {
                "family": "dual_view_cluster",
                "model": model.name,
                "top1_match": float(np.mean([r["top1_match"] for r in metric_rows])) if metric_rows else 0.0,
                "top2_hit": float(np.mean([r["top2_hit"] for r in metric_rows])) if metric_rows else 0.0,
                "soft_match": float(np.mean([r["soft_match"] for r in metric_rows])) if metric_rows else 0.0,
                "mean_regret": float(np.mean([r["regret"] for r in metric_rows])) if metric_rows else 0.0,
                "rank_percentile": float(np.mean([r["rank_percentile"] for r in metric_rows])) if metric_rows else 0.0,
                "used_strategy_count": int(len(set(recs))),
            }
        )
    return rows


def main() -> None:
    bundle = load_competition_data("Data")
    output_dir = Path("outputs") / "stage9_dual_view"
    output_dir.mkdir(parents=True, exist_ok=True)
    utility = utility_from_metrics(bundle.train_metric1, bundle.train_metric2, 0.5, 0.5)

    models = []
    for rc in [2, 3, 4]:
        for tc in [2, 3, 4, 5]:
            for tie in ["equal_weight", "product", "minimax"]:
                models.append(DualViewClusterRecommender(resource_clusters=rc, threat_clusters=tc, tie_breaker=tie))

    rows = evaluate_dual_view(models, bundle.train_x, bundle.train_y, bundle.train_metric1, bundle.train_metric2, utility)

    baseline = evaluate_against_global_utility(
        [ClusterLocalUtilityRecommender(n_clusters=3, metric1_weight=0.5, metric2_weight=0.5)],
        bundle.train_x,
        bundle.train_y,
        bundle.train_metric1,
        bundle.train_metric2,
        utility,
    )[0]
    rows.append(
        {
            "family": "single_view_cluster_baseline",
            "model": baseline.model,
            "top1_match": baseline.top1_match,
            "top2_hit": "",
            "soft_match": "",
            "mean_regret": baseline.mean_regret,
            "rank_percentile": "",
            "used_strategy_count": baseline.used_strategy_count,
        }
    )
    rows = sorted(rows, key=lambda row: (row["soft_match"] if row["soft_match"] != "" else -1, row["top1_match"], -row["mean_regret"]), reverse=True)

    with (output_dir / "dual_view_cv_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    best = rows[0]
    final_model = next(model for model in models if model.name == best["model"])
    final_model.fit(bundle.train_x, bundle.train_y, bundle.train_metric1, bundle.train_metric2)
    pred, utility_scores, pred_m1, pred_m2, pareto_flags, combos = final_model.recommend(bundle.test_x)

    with (output_dir / "test_recommendations.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id",
            "resource_cluster",
            "threat_cluster",
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
            "pareto_s1",
            "pareto_s2",
            "pareto_s3",
            "pareto_s4",
        ])
        for i, (combo, rec, u, m1, m2, pflag) in enumerate(zip(combos, pred, utility_scores, pred_m1, pred_m2, pareto_flags), start=1):
            writer.writerow([i, int(combo[0]), int(combo[1]), int(rec), *[float(v) for v in u], *[float(v) for v in m1], *[float(v) for v in m2], *[int(v) for v in pflag]])

    combo_summary = {}
    for combo in sorted(set(map(tuple, combos.astype(int).tolist()))):
        mask = np.all(combos == np.array(combo), axis=1)
        combo_summary[f"resource_{combo[0]}_threat_{combo[1]}"] = {
            "test_rows": int(np.sum(mask)),
            "recommended_counts": {int(k): int(v) for k, v in zip(*np.unique(pred[mask], return_counts=True))},
        }

    summary = {
        "innovation": "dual-view clustering: resource state clusters and threat situation clusters form interpretable combined scenarios",
        "best_model": best["model"],
        "best_soft_match": best["soft_match"],
        "best_top1_match": best["top1_match"],
        "best_mean_regret": best["mean_regret"],
        "test_prediction_counts": {int(k): int(v) for k, v in zip(*np.unique(pred, return_counts=True))},
        "test_combo_summary": combo_summary,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nDual-view clustering:")
    for row in rows[:12]:
        print(
            f"{row['family']}/{row['model']}: top1={row['top1_match']}, "
            f"top2={row['top2_hit']}, soft={row['soft_match']}, "
            f"regret={row['mean_regret']}, rank_pct={row['rank_percentile']}"
        )


if __name__ == "__main__":
    main()
