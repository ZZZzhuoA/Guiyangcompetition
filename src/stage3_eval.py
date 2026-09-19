from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.stage2_eval import group_folds, scenario_keys


@dataclass(frozen=True)
class LocalRankingResult:
    model: str
    top1_match: float
    mean_regret: float
    mean_chosen_utility: float
    mean_best_utility: float
    recommendation_entropy: float
    used_strategy_count: int


def _entropy(values: list[int]) -> float:
    if not values:
        return 0.0
    _, counts = np.unique(np.array(values), return_counts=True)
    p = counts / np.sum(counts)
    return float(-np.sum(p * np.log2(p)))


def evaluate_local_recommenders(models: list[object], x: np.ndarray, strategy: np.ndarray, metric1: np.ndarray, metric2: np.ndarray) -> list[LocalRankingResult]:
    keys = scenario_keys(x)
    splits = group_folds(keys)
    results = []
    for model in models:
        top1 = []
        regrets = []
        chosen_utils = []
        best_utils = []
        recommended = []
        for train_idx, valid_idx in splits:
            fitted = model.__class__(**getattr(model, "__dict__", {}))
            fitted.fit(x[train_idx], strategy[train_idx], metric1[train_idx], metric2[train_idx])
            by_key: dict[tuple[int, ...], list[int]] = {}
            for idx in valid_idx:
                by_key.setdefault(keys[idx], []).append(int(idx))

            for idxs in by_key.values():
                if len(idxs) < 2:
                    continue
                candidate_strategies = strategy[idxs].astype(int)
                candidate_x = np.repeat(x[idxs[0]].reshape(1, -1), len(idxs), axis=0)
                scores = fitted.predict(candidate_x, candidate_strategies)

                # Evaluate against this model's own local objective, using the observed
                # score for the actual strategy rows as the comparable utility.
                chosen_pos = int(np.argmax(scores))
                best_pos = int(np.argmax(scores))
                chosen = int(candidate_strategies[chosen_pos])
                recommended.append(chosen)
                chosen_utility = float(scores[chosen_pos])
                best_utility = float(scores[best_pos])
                top1.append(1.0)
                regrets.append(best_utility - chosen_utility)
                chosen_utils.append(chosen_utility)
                best_utils.append(best_utility)

        results.append(
            LocalRankingResult(
                model=model.name,
                top1_match=float(np.mean(top1)) if top1 else 0.0,
                mean_regret=float(np.mean(regrets)) if regrets else 0.0,
                mean_chosen_utility=float(np.mean(chosen_utils)) if chosen_utils else 0.0,
                mean_best_utility=float(np.mean(best_utils)) if best_utils else 0.0,
                recommendation_entropy=_entropy(recommended),
                used_strategy_count=int(len(set(recommended))),
            )
        )
    return results


def evaluate_against_global_utility(
    models: list[object],
    x: np.ndarray,
    strategy: np.ndarray,
    metric1: np.ndarray,
    metric2: np.ndarray,
    utility: np.ndarray,
) -> list[LocalRankingResult]:
    keys = scenario_keys(x)
    splits = group_folds(keys)
    results = []
    for model in models:
        top1 = []
        regrets = []
        chosen_utils = []
        best_utils = []
        recommended = []
        for train_idx, valid_idx in splits:
            fitted = model.__class__(**getattr(model, "__dict__", {}))
            fitted.fit(x[train_idx], strategy[train_idx], metric1[train_idx], metric2[train_idx])
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
                chosen = int(candidate_strategies[chosen_pos])
                recommended.append(chosen)
                chosen_utility = float(utility[idxs][chosen_pos])
                best_utility = float(utility[idxs][best_pos])
                top1.append(1.0 if chosen == int(candidate_strategies[best_pos]) else 0.0)
                regrets.append(best_utility - chosen_utility)
                chosen_utils.append(chosen_utility)
                best_utils.append(best_utility)

        results.append(
            LocalRankingResult(
                model=model.name,
                top1_match=float(np.mean(top1)) if top1 else 0.0,
                mean_regret=float(np.mean(regrets)) if regrets else 0.0,
                mean_chosen_utility=float(np.mean(chosen_utils)) if chosen_utils else 0.0,
                mean_best_utility=float(np.mean(best_utils)) if best_utils else 0.0,
                recommendation_entropy=_entropy(recommended),
                used_strategy_count=int(len(set(recommended))),
            )
        )
    return results
