from __future__ import annotations

import numpy as np

from src.pareto_models import pareto_mask_maximize
from src.stage2_eval import group_folds, scenario_keys


def ndcg_at_all(true_utility: np.ndarray, pred_scores: np.ndarray) -> float:
    order = np.argsort(pred_scores)[::-1]
    ideal = np.argsort(true_utility)[::-1]
    gains = true_utility[order]
    ideal_gains = true_utility[ideal]
    discounts = 1.0 / np.log2(np.arange(2, len(true_utility) + 2))
    dcg = float(np.sum(gains * discounts))
    idcg = float(np.sum(ideal_gains * discounts))
    return dcg / idcg if abs(idcg) > 1e-12 else 0.0


def evaluate_scenario_choice(candidate_strategies: np.ndarray, true_utility: np.ndarray, chosen_strategy: int, pred_scores: np.ndarray | None = None, objectives: np.ndarray | None = None) -> dict[str, float]:
    best_utility = float(np.max(true_utility))
    worst_utility = float(np.min(true_utility))
    best_positions = np.where(np.abs(true_utility - best_utility) < 1e-12)[0]
    chosen_positions = np.where(candidate_strategies.astype(int) == int(chosen_strategy))[0]
    if len(chosen_positions) == 0:
        return {}
    chosen_pos = int(chosen_positions[0])
    chosen_utility = float(true_utility[chosen_pos])
    rank = 1 + int(np.sum(true_utility > chosen_utility))
    denom = best_utility - worst_utility
    soft_match = 1.0 if abs(denom) < 1e-12 else (chosen_utility - worst_utility) / denom
    top2_threshold = np.sort(true_utility)[::-1][min(1, len(true_utility) - 1)]
    row = {
        "top1_match": 1.0 if chosen_pos in best_positions else 0.0,
        "top2_hit": 1.0 if chosen_utility >= float(top2_threshold) - 1e-12 else 0.0,
        "chosen_rank": float(rank),
        "rank_percentile": 1.0 if len(true_utility) == 1 else 1.0 - (rank - 1) / (len(true_utility) - 1),
        "soft_match": float(soft_match),
        "regret": best_utility - chosen_utility,
        "relative_regret": 0.0 if abs(denom) < 1e-12 else (best_utility - chosen_utility) / denom,
        "chosen_utility": chosen_utility,
        "best_utility": best_utility,
    }
    if pred_scores is not None and len(pred_scores) == len(true_utility):
        row["ndcg"] = ndcg_at_all(true_utility, pred_scores)
        ordered = np.argsort(pred_scores)[::-1]
        best_rank_pred = 1 + int(np.where(ordered == int(best_positions[0]))[0][0])
        row["mrr"] = 1.0 / best_rank_pred
    if objectives is not None:
        pareto = pareto_mask_maximize(objectives)
        row["pareto_hit"] = 1.0 if pareto[chosen_pos] else 0.0
    return row


def aggregate_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(set().union(*[row.keys() for row in rows]))
    return {key: float(np.mean([row[key] for row in rows if key in row])) for key in keys}


def evaluate_recommender_fine_grained(model: object, fit_mode: str, x: np.ndarray, strategy: np.ndarray, metric1: np.ndarray, metric2: np.ndarray, utility: np.ndarray) -> dict[str, float]:
    keys = scenario_keys(x)
    splits = group_folds(keys)
    metric_rows = []
    for train_idx, valid_idx in splits:
        fitted = model.__class__(**getattr(model, "__dict__", {}))
        if fit_mode == "cluster":
            fitted.fit(x[train_idx], strategy[train_idx], metric1[train_idx], metric2[train_idx])
        elif fit_mode == "pairwise":
            fitted.fit(x[train_idx], strategy[train_idx], utility[train_idx])
        elif fit_mode == "pareto":
            fitted.fit(x[train_idx], strategy[train_idx], metric1[train_idx], metric2[train_idx])
        else:
            fitted.fit(x[train_idx], strategy[train_idx], utility[train_idx])

        by_key: dict[tuple[int, ...], list[int]] = {}
        for idx in valid_idx:
            by_key.setdefault(keys[idx], []).append(int(idx))
        for idxs in by_key.values():
            if len(idxs) < 2:
                continue
            candidate_strategies = strategy[idxs].astype(int)
            candidate_x = np.repeat(x[idxs[0]].reshape(1, -1), len(candidate_strategies), axis=0)
            if fit_mode == "cluster":
                _, score_matrix, _, _ = fitted.recommend(candidate_x[:1])
                all_strategies = np.array([1, 2, 3, 4], dtype=int)
                score_map = {int(s): float(score_matrix[0, pos]) for pos, s in enumerate(all_strategies)}
                pred_scores = np.array([score_map.get(int(s), -np.inf) for s in candidate_strategies])
                chosen_strategy = int(candidate_strategies[int(np.argmax(pred_scores))])
            elif fit_mode == "pairwise":
                pred, score_matrix = fitted.recommend(candidate_x[:1], candidate_strategies)
                chosen_strategy = int(pred[0])
                pred_scores = score_matrix[0]
            elif fit_mode == "pareto":
                pred, utility_scores, _, _ = fitted.recommend(candidate_x[:1], candidate_strategies)
                chosen_strategy = int(pred[0])
                pred_scores = utility_scores[0]
            else:
                pred_scores = fitted.predict(candidate_x, candidate_strategies)
                chosen_strategy = int(candidate_strategies[int(np.argmax(pred_scores))])
            objectives = np.vstack([metric1[idxs], metric2[idxs]]).T
            row = evaluate_scenario_choice(candidate_strategies, utility[idxs], chosen_strategy, pred_scores, objectives)
            if row:
                metric_rows.append(row)
    return aggregate_metric_rows(metric_rows)
