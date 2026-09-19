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
from src.fine_grained_eval import evaluate_scenario_choice
from src.fusion_models import FusionStrategyRecommender
from src.pairwise_models import PairwiseLogisticRanker
from src.pareto_models import ParetoStrategyRecommender
from src.multi_metric_models import DualMetricRegressor
from src.ranking_models import RandomForestUtility
from src.stage2_eval import group_folds, scenario_keys, utility_from_metrics
from src.stage3_models import ClusterLocalUtilityRecommender
from src.dual_view_models import DualViewClusterRecommender


def aggregate(rows):
    keys = sorted(set().union(*[r.keys() for r in rows])) if rows else []
    return {k: float(np.mean([r[k] for r in rows if k in r])) for k in keys}


def evaluate_fusion_model(model, x, strategy, metric1, metric2, utility):
    keys = scenario_keys(x)
    splits = group_folds(keys)
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
            _, all_scores, _ = fitted.recommend(candidate_x[:1])
            score_map = {int(s): float(all_scores[0, pos]) for pos, s in enumerate([1, 2, 3, 4])}
            pred_scores = np.array([score_map.get(int(s), -np.inf) for s in candidate_strategies])
            chosen = int(candidate_strategies[int(np.argmax(pred_scores))])
            recs.append(chosen)
            objectives = np.vstack([metric1[idxs], metric2[idxs]]).T
            row = evaluate_scenario_choice(candidate_strategies, utility[idxs], chosen, pred_scores, objectives)
            if row:
                metric_rows.append(row)
    out = aggregate(metric_rows)
    out["used_strategy_count"] = int(len(set(recs)))
    return out


def main() -> None:
    bundle = load_competition_data("Data")
    output_dir = Path("outputs") / "stage10_fusion"
    output_dir.mkdir(parents=True, exist_ok=True)
    utility = utility_from_metrics(bundle.train_metric1, bundle.train_metric2, 0.5, 0.5)

    fusion_models = [
        FusionStrategyRecommender(0.35, 0.25, 0.20, 0.20, 0.15),
        FusionStrategyRecommender(0.45, 0.25, 0.15, 0.15, 0.10),
        FusionStrategyRecommender(0.25, 0.35, 0.20, 0.20, 0.15),
        FusionStrategyRecommender(0.25, 0.25, 0.30, 0.20, 0.20),
        FusionStrategyRecommender(0.30, 0.30, 0.20, 0.20, 0.05),
    ]

    rows = []
    for model in fusion_models:
        m = evaluate_fusion_model(model, bundle.train_x, bundle.train_y, bundle.train_metric1, bundle.train_metric2, utility)
        m["family"] = "fusion"
        m["model"] = model.name
        rows.append(m)

    representative = [
        ("cluster", ClusterLocalUtilityRecommender(n_clusters=3, metric1_weight=0.5, metric2_weight=0.5)),
        ("pairwise", PairwiseLogisticRanker(epochs=800, learning_rate=0.05, l2=0.003)),
        ("pareto", ParetoStrategyRecommender(DualMetricRegressor(RandomForestUtility(n_trees=25, max_depth=6, min_samples_leaf=8)), tie_breaker="product")),
        ("dual_view", DualViewClusterRecommender(resource_clusters=2, threat_clusters=2, tie_breaker="equal_weight")),
    ]
    for family, model in representative:
        if family == "cluster":
            from src.fine_grained_eval import evaluate_recommender_fine_grained
            m = evaluate_recommender_fine_grained(model, "cluster", bundle.train_x, bundle.train_y, bundle.train_metric1, bundle.train_metric2, utility)
        elif family == "pairwise":
            from src.fine_grained_eval import evaluate_recommender_fine_grained
            m = evaluate_recommender_fine_grained(model, "pairwise", bundle.train_x, bundle.train_y, bundle.train_metric1, bundle.train_metric2, utility)
        elif family == "pareto":
            from src.fine_grained_eval import evaluate_recommender_fine_grained
            m = evaluate_recommender_fine_grained(model, "pareto", bundle.train_x, bundle.train_y, bundle.train_metric1, bundle.train_metric2, utility)
        else:
            # Keep dual-view result from its native stage metrics shape.
            from experiments.stage9_dual_view_clustering import evaluate_dual_view
            dv = evaluate_dual_view([model], bundle.train_x, bundle.train_y, bundle.train_metric1, bundle.train_metric2, utility)[0]
            m = {
                "top1_match": dv["top1_match"],
                "top2_hit": dv["top2_hit"],
                "soft_match": dv["soft_match"],
                "regret": dv["mean_regret"],
                "rank_percentile": dv["rank_percentile"],
                "used_strategy_count": dv["used_strategy_count"],
            }
        m["family"] = family
        m["model"] = model.name
        rows.append(m)

    rows = sorted(rows, key=lambda r: (r.get("soft_match", 0), r.get("top1_match", 0), -r.get("regret", 999)), reverse=True)
    fieldnames = ["family", "model", "top1_match", "top2_hit", "soft_match", "regret", "relative_regret", "rank_percentile", "ndcg", "mrr", "pareto_hit", "used_strategy_count"]
    with (output_dir / "fusion_cv_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    best = rows[0]
    final_model = next((m for m in fusion_models if m.name == best["model"]), None)
    if final_model is None:
        final_model = fusion_models[0]
    final_model.fit(bundle.train_x, bundle.train_y, bundle.train_metric1, bundle.train_metric2)
    pred, fused_scores, parts = final_model.recommend(bundle.test_x)

    with (output_dir / "test_recommendations.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "recommended_strategy", "fused_s1", "fused_s2", "fused_s3", "fused_s4", "cluster_s1", "cluster_s2", "cluster_s3", "cluster_s4", "pairwise_s1", "pairwise_s2", "pairwise_s3", "pairwise_s4", "pareto_flag_s1", "pareto_flag_s2", "pareto_flag_s3", "pareto_flag_s4"])
        for i, (p, fs, cs, ps, pf) in enumerate(zip(pred, fused_scores, parts["cluster"], parts["pairwise"], parts["pareto_flags"]), start=1):
            writer.writerow([i, int(p), *[float(v) for v in fs], *[float(v) for v in cs], *[float(v) for v in ps], *[int(v) for v in pf]])

    summary = {
        "innovation": "score-level ensemble of scenario-local utility, pairwise ranking, Pareto dominance, and dual-view clustering",
        "best_model": best["model"],
        "best_family": best["family"],
        "best_soft_match": best.get("soft_match"),
        "best_top1_match": best.get("top1_match"),
        "best_regret": best.get("regret"),
        "test_prediction_counts": {int(k): int(v) for k, v in zip(*np.unique(pred, return_counts=True))},
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nFusion ranking:")
    for row in rows:
        print(
            f"{row['family']}/{row['model']}: soft={row.get('soft_match','')}, "
            f"top1={row.get('top1_match','')}, regret={row.get('regret','')}, "
            f"pareto={row.get('pareto_hit','')}"
        )


if __name__ == "__main__":
    main()
