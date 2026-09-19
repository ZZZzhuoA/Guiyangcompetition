from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.pairwise_models import PairwiseLogisticRanker
from src.ranking_models import RandomForestUtility


def _minmax_row(values: np.ndarray) -> np.ndarray:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if abs(hi - lo) < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


@dataclass
class PairwiseCorrectedUtility:
    base_model: object | None = None
    pairwise_model: object | None = None
    pairwise_weight: float = 0.35
    strategy3_penalty: float = 0.0

    @property
    def name(self) -> str:
        return f"pairwise_corrected_w{self.pairwise_weight:g}_p3{self.strategy3_penalty:g}"

    def fit(self, x: np.ndarray, strategy: np.ndarray, utility: np.ndarray) -> "PairwiseCorrectedUtility":
        if self.base_model is None:
            self.base_model_ = RandomForestUtility(n_trees=25, max_depth=6, min_samples_leaf=8)
        else:
            self.base_model_ = self.base_model
        if self.pairwise_model is None:
            self.pairwise_model_ = PairwiseLogisticRanker(epochs=800, learning_rate=0.05, l2=0.003)
        else:
            self.pairwise_model_ = self.pairwise_model
        self.base_model_.fit(x, strategy, utility)
        self.pairwise_model_.fit(x, strategy, utility)
        return self

    def predict(self, x: np.ndarray, strategy: np.ndarray) -> np.ndarray:
        base_scores = self.base_model_.predict(x, strategy)
        out = np.zeros(len(strategy), dtype=float)
        groups: dict[tuple[float, ...], list[int]] = {}
        for idx, row in enumerate(x):
            groups.setdefault(tuple(np.round(row.astype(float), 10).tolist()), []).append(idx)

        for idxs in groups.values():
            idx_array = np.array(idxs, dtype=int)
            group_strategies = strategy[idx_array].astype(int)
            unique_strategies, first_pos = np.unique(group_strategies, return_index=True)
            group_x = x[idx_array[first_pos[:1]]]
            _, pairwise_scores = self.pairwise_model_.recommend(group_x, unique_strategies)
            pair_map = {int(s): float(v) for s, v in zip(unique_strategies, pairwise_scores[0])}
            pair_scores = np.array([pair_map[int(s)] for s in group_strategies], dtype=float)

            base_norm = _minmax_row(base_scores[idx_array])
            pair_norm = _minmax_row(pair_scores)
            corrected = (1.0 - self.pairwise_weight) * base_norm + self.pairwise_weight * pair_norm
            corrected = corrected - self.strategy3_penalty * (group_strategies == 3).astype(float)
            out[idx_array] = corrected
        return out
