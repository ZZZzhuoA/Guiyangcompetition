from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.ranking_models import recommend_strategy
from src.stage2_eval import scenario_keys


@dataclass
class ExactSceneStats:
    counts: np.ndarray
    utility: np.ndarray
    metric1: np.ndarray
    metric2: np.ndarray


class ExactSceneHybridRecommender:
    """Recommend from exact full-scene evidence, falling back to a fitted model."""

    name = "exact_scene_hybrid"

    def __init__(self, base_model: object, min_exact_strategies: int = 2, exact_weight: float = 0.85):
        self.base_model = base_model
        self.min_exact_strategies = min_exact_strategies
        self.exact_weight = exact_weight

    def fit(
        self,
        raw_x: np.ndarray,
        model_x: np.ndarray,
        strategy: np.ndarray,
        utility: np.ndarray,
        metric1: np.ndarray,
        metric2: np.ndarray,
    ) -> "ExactSceneHybridRecommender":
        self.base_model.fit(model_x, strategy, utility)
        self.global_strategy_mean_ = np.array(
            [np.mean(utility[strategy.astype(int) == s]) for s in [1, 2, 3, 4]],
            dtype=float,
        )
        buckets: dict[tuple[int, ...], list[int]] = {}
        for idx, key in enumerate(scenario_keys(raw_x)):
            buckets.setdefault(key, []).append(idx)

        self.scene_stats_: dict[tuple[int, ...], ExactSceneStats] = {}
        for key, idxs in buckets.items():
            counts = np.zeros(4, dtype=int)
            util = np.full(4, np.nan, dtype=float)
            m1 = np.full(4, np.nan, dtype=float)
            m2 = np.full(4, np.nan, dtype=float)
            idx_array = np.array(idxs, dtype=int)
            for s in [1, 2, 3, 4]:
                mask = strategy[idx_array].astype(int) == s
                if np.any(mask):
                    chosen = idx_array[mask]
                    counts[s - 1] = len(chosen)
                    util[s - 1] = float(np.mean(utility[chosen]))
                    m1[s - 1] = float(np.mean(metric1[chosen]))
                    m2[s - 1] = float(np.mean(metric2[chosen]))
            self.scene_stats_[key] = ExactSceneStats(counts=counts, utility=util, metric1=m1, metric2=m2)
        return self

    def recommend(self, raw_x: np.ndarray, model_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
        base_pred, base_scores = recommend_strategy(self.base_model, model_x)
        strategies = np.array([1, 2, 3, 4], dtype=int)
        final_scores = []
        sources = []
        exact_counts = []

        for row_idx, key in enumerate(scenario_keys(raw_x)):
            stats = self.scene_stats_.get(key)
            score = base_scores[row_idx].astype(float).copy()
            source = "model_fallback"
            count = np.zeros(4, dtype=int)
            if stats is not None:
                count = stats.counts.copy()
                seen = stats.counts > 0
                seen_strategy_count = int(np.sum(seen))
                if seen_strategy_count >= self.min_exact_strategies:
                    exact_score = np.where(seen, stats.utility, self.global_strategy_mean_)
                    if seen_strategy_count == 4:
                        score = exact_score
                        source = "exact_full"
                    else:
                        score = self.exact_weight * exact_score + (1.0 - self.exact_weight) * score
                        source = "exact_partial_blend"
                else:
                    source = "exact_low_evidence_fallback"
            final_scores.append(score)
            sources.append(source)
            exact_counts.append(count)

        scores = np.vstack(final_scores)
        pred = strategies[np.argmax(scores, axis=1)]
        return pred.astype(int), scores, sources, np.vstack(exact_counts)
