from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def minmax_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo = np.min(x, axis=0)
    hi = np.max(x, axis=0)
    scale = np.where(np.abs(hi - lo) < 1e-12, 1.0, hi - lo)
    return lo, scale


def minmax_transform(x: np.ndarray, lo: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (x - lo) / scale


def metric_normalizer(metric1: np.ndarray, metric2: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float, float]]:
    m1_min, m1_max = float(np.min(metric1)), float(np.max(metric1))
    m2_min, m2_max = float(np.min(metric2)), float(np.max(metric2))
    m1_scale = m1_max - m1_min if abs(m1_max - m1_min) > 1e-12 else 1.0
    m2_scale = m2_max - m2_min if abs(m2_max - m2_min) > 1e-12 else 1.0
    m1 = (metric1 - m1_min) / m1_scale
    m2 = (metric2 - m2_min) / m2_scale
    return m1, m2, (m1_min, m1_scale, m2_min, m2_scale)


@dataclass
class SimpleKMeans:
    n_clusters: int = 4
    max_iter: int = 80
    seed: int = 42

    def fit(self, x: np.ndarray) -> "SimpleKMeans":
        rng = np.random.default_rng(self.seed)
        if len(x) < self.n_clusters:
            raise ValueError("n_clusters cannot exceed sample count")
        first = int(rng.integers(0, len(x)))
        centers = [x[first]]
        while len(centers) < self.n_clusters:
            d2 = np.min(((x[:, None, :] - np.vstack(centers)[None, :, :]) ** 2).sum(axis=2), axis=1)
            if float(np.sum(d2)) <= 1e-12:
                candidates = rng.choice(len(x), size=self.n_clusters, replace=False)
                centers = [x[i] for i in candidates]
                break
            probs = d2 / np.sum(d2)
            centers.append(x[int(rng.choice(len(x), p=probs))])
        self.centers_ = np.vstack(centers).astype(float)

        for _ in range(self.max_iter):
            labels = self.predict(x)
            new_centers = []
            for cluster in range(self.n_clusters):
                part = x[labels == cluster]
                if len(part) == 0:
                    new_centers.append(x[int(rng.integers(0, len(x)))])
                else:
                    new_centers.append(np.mean(part, axis=0))
            new_centers = np.vstack(new_centers)
            if np.allclose(new_centers, self.centers_):
                break
            self.centers_ = new_centers
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        d2 = ((x[:, None, :] - self.centers_[None, :, :]) ** 2).sum(axis=2)
        return np.argmin(d2, axis=1)


@dataclass
class ClusterLocalUtilityRecommender:
    n_clusters: int = 4
    shrinkage: float = 8.0
    metric1_weight: float = 0.5
    metric2_weight: float = 0.5
    adaptive_weights: bool = False
    seed: int = 42

    @property
    def name(self) -> str:
        mode = "adaptive" if self.adaptive_weights else f"w{self.metric1_weight:.1f}_{self.metric2_weight:.1f}"
        return f"cluster_local_k{self.n_clusters}_{mode}"

    def fit(self, x: np.ndarray, strategy: np.ndarray, metric1: np.ndarray, metric2: np.ndarray) -> "ClusterLocalUtilityRecommender":
        self.strategies_ = np.array([1, 2, 3, 4], dtype=int)
        self.x_lo_, self.x_scale_ = minmax_fit(x.astype(float))
        x_norm = minmax_transform(x.astype(float), self.x_lo_, self.x_scale_)
        self.clusterer_ = SimpleKMeans(n_clusters=self.n_clusters, seed=self.seed).fit(x_norm)
        labels = self.clusterer_.predict(x_norm)

        m1_norm, m2_norm, self.metric_norm_ = metric_normalizer(metric1.astype(float), metric2.astype(float))
        self.global_m1_ = {s: float(np.mean(m1_norm[strategy == s])) for s in self.strategies_}
        self.global_m2_ = {s: float(np.mean(m2_norm[strategy == s])) for s in self.strategies_}
        self.cluster_m1_: dict[tuple[int, int], float] = {}
        self.cluster_m2_: dict[tuple[int, int], float] = {}
        self.cluster_weights_: dict[int, tuple[float, float]] = {}
        self.cluster_sizes_: dict[int, int] = {}

        for cluster in range(self.n_clusters):
            cluster_mask = labels == cluster
            self.cluster_sizes_[cluster] = int(np.sum(cluster_mask))
            local_m1_by_strategy = []
            local_m2_by_strategy = []
            for s in self.strategies_:
                mask = cluster_mask & (strategy == s)
                n = int(np.sum(mask))
                if n == 0:
                    m1_value = self.global_m1_[int(s)]
                    m2_value = self.global_m2_[int(s)]
                else:
                    alpha = n / (n + self.shrinkage)
                    m1_value = alpha * float(np.mean(m1_norm[mask])) + (1.0 - alpha) * self.global_m1_[int(s)]
                    m2_value = alpha * float(np.mean(m2_norm[mask])) + (1.0 - alpha) * self.global_m2_[int(s)]
                self.cluster_m1_[(cluster, int(s))] = m1_value
                self.cluster_m2_[(cluster, int(s))] = m2_value
                local_m1_by_strategy.append(m1_value)
                local_m2_by_strategy.append(m2_value)

            if self.adaptive_weights:
                spread1 = float(np.max(local_m1_by_strategy) - np.min(local_m1_by_strategy))
                spread2 = float(np.max(local_m2_by_strategy) - np.min(local_m2_by_strategy))
                total = spread1 + spread2
                if total <= 1e-12:
                    w1, w2 = 0.5, 0.5
                else:
                    w1, w2 = spread1 / total, spread2 / total
                self.cluster_weights_[cluster] = (w1, w2)
            else:
                self.cluster_weights_[cluster] = (self.metric1_weight, self.metric2_weight)
        return self

    def _cluster_for(self, x: np.ndarray) -> np.ndarray:
        x_norm = minmax_transform(x.astype(float), self.x_lo_, self.x_scale_)
        return self.clusterer_.predict(x_norm)

    def score_candidates(self, x: np.ndarray, candidates: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if candidates is None:
            candidates = self.strategies_
        labels = self._cluster_for(x)
        all_scores = []
        weights = []
        for cluster in labels:
            w1, w2 = self.cluster_weights_[int(cluster)]
            weights.append([w1, w2])
            scores = []
            for s in candidates:
                m1 = self.cluster_m1_[(int(cluster), int(s))]
                m2 = self.cluster_m2_[(int(cluster), int(s))]
                scores.append(w1 * m1 + w2 * m2)
            all_scores.append(scores)
        return labels, np.vstack(all_scores), np.array(weights, dtype=float)

    def recommend(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        labels, scores, weights = self.score_candidates(x, self.strategies_)
        pred = self.strategies_[np.argmax(scores, axis=1)]
        return pred.astype(int), scores, labels, weights

    def predict(self, x: np.ndarray, strategy: np.ndarray) -> np.ndarray:
        labels = self._cluster_for(x)
        out = []
        for cluster, s in zip(labels, strategy.astype(int)):
            w1, w2 = self.cluster_weights_[int(cluster)]
            m1 = self.cluster_m1_[(int(cluster), int(s))]
            m2 = self.cluster_m2_[(int(cluster), int(s))]
            out.append(w1 * m1 + w2 * m2)
        return np.array(out, dtype=float)
