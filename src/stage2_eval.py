from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RankingResult:
    model: str
    weight_metric1: float
    weight_metric2: float
    top1_match: float
    mean_regret: float
    mean_chosen_utility: float
    mean_best_utility: float
    rmse_utility: float


def normalize_metric(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if abs(hi - lo) < 1e-12:
        return np.zeros_like(values, dtype=float), lo, hi
    return (values - lo) / (hi - lo), lo, hi


def utility_from_metrics(metric1: np.ndarray, metric2: np.ndarray, w1: float, w2: float) -> np.ndarray:
    m1, _, _ = normalize_metric(metric1.astype(float))
    m2, _, _ = normalize_metric(metric2.astype(float))
    return w1 * m1 + w2 * m2


def interception_priority_utility(metric1: np.ndarray, metric2: np.ndarray, secondary_weight: float = 0.05) -> np.ndarray:
    """Score strategies with interception rate as the primary objective.

    The cost-effectiveness term is intentionally small, so it only changes
    ranking when interception-rate scores are close.
    """
    m1, _, _ = normalize_metric(metric1.astype(float))
    m2, _, _ = normalize_metric(metric2.astype(float))
    return m1 + secondary_weight * m2


def scenario_keys(x: np.ndarray) -> list[tuple[int, ...]]:
    return [tuple(row.astype(int).tolist()) for row in x]


def group_folds(keys: list[tuple[int, ...]], n_splits: int = 5, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    unique_keys = np.array(sorted(set(keys)), dtype=object)
    rng.shuffle(unique_keys)
    folds = [[] for _ in range(n_splits)]
    for i, key in enumerate(unique_keys):
        folds[i % n_splits].append(tuple(key))
    splits = []
    key_array = np.array(keys, dtype=object)
    for fold_keys in folds:
        valid_mask = np.array([tuple(key) in set(fold_keys) for key in key_array])
        valid_idx = np.where(valid_mask)[0]
        train_idx = np.where(~valid_mask)[0]
        splits.append((train_idx, valid_idx))
    return splits


def evaluate_rankers(
    models: list[object],
    x: np.ndarray,
    strategy: np.ndarray,
    utility: np.ndarray,
    weight_metric1: float,
    weight_metric2: float,
) -> list[RankingResult]:
    keys = scenario_keys(x)
    splits = group_folds(keys)
    results = []
    for model in models:
        top1 = []
        regrets = []
        chosen_utils = []
        best_utils = []
        sq_errors = []
        for train_idx, valid_idx in splits:
            fitted = model.__class__(**getattr(model, "__dict__", {}))
            fitted.fit(x[train_idx], strategy[train_idx], utility[train_idx])
            pred_utility = fitted.predict(x[valid_idx], strategy[valid_idx])
            sq_errors.extend((pred_utility - utility[valid_idx]) ** 2)

            by_key: dict[tuple[int, ...], list[int]] = {}
            for idx in valid_idx:
                by_key.setdefault(keys[idx], []).append(int(idx))

            for idxs in by_key.values():
                if len(idxs) < 2:
                    continue
                candidate_strategies = strategy[idxs].astype(int)
                candidate_x = np.repeat(x[idxs[0]].reshape(1, -1), len(idxs), axis=0)
                scores = fitted.predict(candidate_x, candidate_strategies)
                chosen_pos = int(np.argmax(scores))
                best_pos = int(np.argmax(utility[idxs]))
                chosen_utility = float(utility[idxs][chosen_pos])
                best_utility = float(utility[idxs][best_pos])
                top1.append(1.0 if candidate_strategies[chosen_pos] == candidate_strategies[best_pos] else 0.0)
                regrets.append(best_utility - chosen_utility)
                chosen_utils.append(chosen_utility)
                best_utils.append(best_utility)

        results.append(
            RankingResult(
                model=model.name,
                weight_metric1=weight_metric1,
                weight_metric2=weight_metric2,
                top1_match=float(np.mean(top1)) if top1 else 0.0,
                mean_regret=float(np.mean(regrets)) if regrets else 0.0,
                mean_chosen_utility=float(np.mean(chosen_utils)) if chosen_utils else 0.0,
                mean_best_utility=float(np.mean(best_utils)) if best_utils else 0.0,
                rmse_utility=float(np.sqrt(np.mean(sq_errors))) if sq_errors else 0.0,
            )
        )
    return results
