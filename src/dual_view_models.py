from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.pareto_models import pareto_mask_maximize
from src.stage3_models import SimpleKMeans, metric_normalizer, minmax_fit, minmax_transform


@dataclass
class DualViewClusterRecommender:
    resource_clusters: int = 3
    threat_clusters: int = 3
    shrinkage: float = 8.0
    tie_breaker: str = "equal_weight"
    seed: int = 42

    @property
    def name(self) -> str:
        return f"dual_view_rc{self.resource_clusters}_tc{self.threat_clusters}_{self.tie_breaker}"

    def fit(self, x: np.ndarray, strategy: np.ndarray, metric1: np.ndarray, metric2: np.ndarray) -> "DualViewClusterRecommender":
        x = x.astype(float)
        self.strategies_ = np.array([1, 2, 3, 4], dtype=int)
        self.resource_slice_ = slice(0, 18)
        self.threat_slice_ = slice(18, 42)

        resource_x = x[:, self.resource_slice_]
        threat_x = x[:, self.threat_slice_]
        self.resource_lo_, self.resource_scale_ = minmax_fit(resource_x)
        self.threat_lo_, self.threat_scale_ = minmax_fit(threat_x)
        resource_norm = minmax_transform(resource_x, self.resource_lo_, self.resource_scale_)
        threat_norm = minmax_transform(threat_x, self.threat_lo_, self.threat_scale_)

        self.resource_clusterer_ = SimpleKMeans(self.resource_clusters, seed=self.seed).fit(resource_norm)
        self.threat_clusterer_ = SimpleKMeans(self.threat_clusters, seed=self.seed + 17).fit(threat_norm)
        resource_labels = self.resource_clusterer_.predict(resource_norm)
        threat_labels = self.threat_clusterer_.predict(threat_norm)

        m1_norm, m2_norm, self.metric_norm_ = metric_normalizer(metric1.astype(float), metric2.astype(float))
        strategy = strategy.astype(int)
        self.global_m1_ = {s: float(np.mean(m1_norm[strategy == s])) for s in self.strategies_}
        self.global_m2_ = {s: float(np.mean(m2_norm[strategy == s])) for s in self.strategies_}

        self.local_m1_: dict[tuple[int, int, int], float] = {}
        self.local_m2_: dict[tuple[int, int, int], float] = {}
        self.combo_sizes_: dict[tuple[int, int], int] = {}

        for r in range(self.resource_clusters):
            for t in range(self.threat_clusters):
                combo_mask = (resource_labels == r) & (threat_labels == t)
                self.combo_sizes_[(r, t)] = int(np.sum(combo_mask))
                for s in self.strategies_:
                    mask = combo_mask & (strategy == s)
                    n = int(np.sum(mask))
                    if n == 0:
                        m1_value = self.global_m1_[int(s)]
                        m2_value = self.global_m2_[int(s)]
                    else:
                        alpha = n / (n + self.shrinkage)
                        m1_value = alpha * float(np.mean(m1_norm[mask])) + (1.0 - alpha) * self.global_m1_[int(s)]
                        m2_value = alpha * float(np.mean(m2_norm[mask])) + (1.0 - alpha) * self.global_m2_[int(s)]
                    self.local_m1_[(r, t, int(s))] = m1_value
                    self.local_m2_[(r, t, int(s))] = m2_value
        return self

    def _labels(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = x.astype(float)
        resource = minmax_transform(x[:, self.resource_slice_], self.resource_lo_, self.resource_scale_)
        threat = minmax_transform(x[:, self.threat_slice_], self.threat_lo_, self.threat_scale_)
        return self.resource_clusterer_.predict(resource), self.threat_clusterer_.predict(threat)

    def _tie_utility(self, m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
        if self.tie_breaker == "product":
            return np.maximum(m1, 0.0) * np.maximum(m2, 0.0)
        if self.tie_breaker == "minimax":
            return np.minimum(m1, m2)
        return 0.5 * m1 + 0.5 * m2

    def recommend(self, x: np.ndarray, strategies: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if strategies is None:
            strategies = self.strategies_
        resource_labels, threat_labels = self._labels(x)
        predictions = []
        utility_rows = []
        m1_rows = []
        m2_rows = []
        pareto_rows = []
        for r, t in zip(resource_labels, threat_labels):
            pred_m1 = np.array([self.local_m1_[(int(r), int(t), int(s))] for s in strategies], dtype=float)
            pred_m2 = np.array([self.local_m2_[(int(r), int(t), int(s))] for s in strategies], dtype=float)
            objectives = np.vstack([pred_m1, pred_m2]).T
            pareto = pareto_mask_maximize(objectives)
            utility = self._tie_utility(pred_m1, pred_m2)
            masked = np.where(pareto, utility, -np.inf)
            best = int(np.argmax(masked))
            predictions.append(int(strategies[best]))
            utility_rows.append(utility)
            m1_rows.append(pred_m1)
            m2_rows.append(pred_m2)
            pareto_rows.append(pareto.astype(int))
        combos = np.vstack([resource_labels, threat_labels]).T
        return (
            np.array(predictions, dtype=int),
            np.vstack(utility_rows),
            np.vstack(m1_rows),
            np.vstack(m2_rows),
            np.vstack(pareto_rows),
            combos,
        )

    def predict(self, x: np.ndarray, strategy: np.ndarray) -> np.ndarray:
        resource_labels, threat_labels = self._labels(x)
        out = []
        for r, t, s in zip(resource_labels, threat_labels, strategy.astype(int)):
            m1 = self.local_m1_[(int(r), int(t), int(s))]
            m2 = self.local_m2_[(int(r), int(t), int(s))]
            out.append(float(self._tie_utility(np.array([m1]), np.array([m2]))[0]))
        return np.array(out, dtype=float)
