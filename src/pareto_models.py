from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def pareto_mask_maximize(values: np.ndarray) -> np.ndarray:
    mask = np.ones(values.shape[0], dtype=bool)
    for i in range(values.shape[0]):
        if not mask[i]:
            continue
        dominated = np.all(values >= values[i], axis=1) & np.any(values > values[i], axis=1)
        dominated[i] = False
        if np.any(dominated):
            mask[i] = False
    return mask


@dataclass
class ParetoStrategyRecommender:
    dual_metric_model: object
    tie_breaker: str = "equal_weight"

    @property
    def name(self) -> str:
        return f"pareto_{self.dual_metric_model.name}_{self.tie_breaker}"

    def fit(self, x: np.ndarray, strategy: np.ndarray, metric1: np.ndarray, metric2: np.ndarray) -> "ParetoStrategyRecommender":
        self.dual_metric_model.fit(x, strategy, metric1, metric2)
        self.m1_min_, self.m1_max_ = float(np.min(metric1)), float(np.max(metric1))
        self.m2_min_, self.m2_max_ = float(np.min(metric2)), float(np.max(metric2))
        return self

    def _norm(self, values: np.ndarray, lo: float, hi: float) -> np.ndarray:
        scale = hi - lo if abs(hi - lo) > 1e-12 else 1.0
        return (values - lo) / scale

    def recommend(self, x: np.ndarray, strategies: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if strategies is None:
            strategies = np.array([1, 2, 3, 4], dtype=int)
        recommendations = []
        utility_rows = []
        m1_rows = []
        m2_rows = []
        for row in x:
            candidate_x = np.repeat(row.reshape(1, -1), len(strategies), axis=0)
            pred_m1, pred_m2 = self.dual_metric_model.predict_metrics(candidate_x, strategies)
            objectives = np.vstack([pred_m1, pred_m2]).T
            keep = pareto_mask_maximize(objectives)
            m1n = self._norm(pred_m1, self.m1_min_, self.m1_max_)
            m2n = self._norm(pred_m2, self.m2_min_, self.m2_max_)
            if self.tie_breaker == "product":
                utility = np.maximum(m1n, 0.0) * np.maximum(m2n, 0.0)
            elif self.tie_breaker == "minimax":
                utility = np.minimum(m1n, m2n)
            else:
                utility = 0.5 * m1n + 0.5 * m2n
            masked_utility = np.where(keep, utility, -np.inf)
            best = int(np.argmax(masked_utility))
            recommendations.append(int(strategies[best]))
            utility_rows.append(utility)
            m1_rows.append(pred_m1)
            m2_rows.append(pred_m2)
        return np.array(recommendations, dtype=int), np.vstack(utility_rows), np.vstack(m1_rows), np.vstack(m2_rows)
