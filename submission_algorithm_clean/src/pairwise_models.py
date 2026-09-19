from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.stage2_eval import scenario_keys
from src.stage3_models import SimpleKMeans, minmax_fit, minmax_transform


def _one_hot_strategy(strategy: int) -> np.ndarray:
    out = np.zeros(4, dtype=float)
    if 1 <= int(strategy) <= 4:
        out[int(strategy) - 1] = 1.0
    return out


@dataclass
class PairwiseLogisticRanker:
    epochs: int = 600
    learning_rate: float = 0.08
    l2: float = 0.001

    @property
    def name(self) -> str:
        return f"pairwise_logistic_e{self.epochs}_lr{self.learning_rate:g}"

    def fit(self, x: np.ndarray, strategy: np.ndarray, utility: np.ndarray) -> "PairwiseLogisticRanker":
        self.x_lo_, self.x_scale_ = minmax_fit(x.astype(float))
        x_norm = minmax_transform(x.astype(float), self.x_lo_, self.x_scale_)
        pairs_x, pairs_y = self._make_pairs(x_norm, strategy.astype(int), utility.astype(float))
        self.feature_mean_ = np.mean(pairs_x, axis=0)
        self.feature_std_ = np.std(pairs_x, axis=0)
        self.feature_std_ = np.where(self.feature_std_ < 1e-12, 1.0, self.feature_std_)
        z = (pairs_x - self.feature_mean_) / self.feature_std_
        z = np.hstack([np.ones((z.shape[0], 1)), z])
        self.coef_ = np.zeros(z.shape[1], dtype=float)
        for _ in range(self.epochs):
            logits = np.clip(z @ self.coef_, -30.0, 30.0)
            pred = 1.0 / (1.0 + np.exp(-logits))
            grad = z.T @ (pred - pairs_y) / len(pairs_y)
            grad[1:] += self.l2 * self.coef_[1:]
            self.coef_ -= self.learning_rate * grad
        return self

    def _pair_features(self, row: np.ndarray, strategy_a: int, strategy_b: int) -> np.ndarray:
        a = _one_hot_strategy(strategy_a)
        b = _one_hot_strategy(strategy_b)
        diff = a - b
        interactions = np.concatenate([row * diff[i] for i in range(4)])
        return np.concatenate([row, a, b, diff, interactions])

    def _make_pairs(self, x_norm: np.ndarray, strategy: np.ndarray, utility: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        keys = scenario_keys(x_norm)
        groups: dict[tuple[int, ...], list[int]] = {}
        # Use exact original discrete values after normalization is not possible here,
        # so group by rounded normalized feature vector.
        for idx, row in enumerate(x_norm):
            key = tuple(np.round(row, 8).tolist())
            groups.setdefault(key, []).append(idx)

        features = []
        labels = []
        for idxs in groups.values():
            for pos_i in range(len(idxs)):
                for pos_j in range(pos_i + 1, len(idxs)):
                    i, j = idxs[pos_i], idxs[pos_j]
                    if strategy[i] == strategy[j] or abs(utility[i] - utility[j]) < 1e-12:
                        continue
                    row = x_norm[i]
                    label_ij = 1.0 if utility[i] > utility[j] else 0.0
                    features.append(self._pair_features(row, int(strategy[i]), int(strategy[j])))
                    labels.append(label_ij)
                    features.append(self._pair_features(row, int(strategy[j]), int(strategy[i])))
                    labels.append(1.0 - label_ij)
        if not features:
            raise ValueError("No pairwise training samples were generated")
        return np.vstack(features), np.array(labels, dtype=float)

    def _prob_a_beats_b(self, row_norm: np.ndarray, strategy_a: int, strategy_b: int) -> float:
        raw = self._pair_features(row_norm, strategy_a, strategy_b)
        z = (raw - self.feature_mean_) / self.feature_std_
        z = np.concatenate([[1.0], z])
        logit = float(np.clip(z @ self.coef_, -30.0, 30.0))
        return float(1.0 / (1.0 + np.exp(-logit)))

    def recommend(self, x: np.ndarray, strategies: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        if strategies is None:
            strategies = np.array([1, 2, 3, 4], dtype=int)
        x_norm = minmax_transform(x.astype(float), self.x_lo_, self.x_scale_)
        recommendations = []
        score_rows = []
        for row in x_norm:
            scores = []
            for s in strategies:
                score = 0.0
                for other in strategies:
                    if int(s) == int(other):
                        continue
                    score += self._prob_a_beats_b(row, int(s), int(other))
                scores.append(score)
            score_row = np.array(scores, dtype=float)
            score_rows.append(score_row)
            recommendations.append(int(strategies[np.argmax(score_row)]))
        return np.array(recommendations, dtype=int), np.vstack(score_rows)


@dataclass
class ClusterPairwisePreference:
    n_clusters: int = 3
    shrinkage: float = 6.0
    seed: int = 42

    @property
    def name(self) -> str:
        return f"cluster_pairwise_k{self.n_clusters}"

    def fit(self, x: np.ndarray, strategy: np.ndarray, utility: np.ndarray) -> "ClusterPairwisePreference":
        self.strategies_ = np.array([1, 2, 3, 4], dtype=int)
        self.x_lo_, self.x_scale_ = minmax_fit(x.astype(float))
        x_norm = minmax_transform(x.astype(float), self.x_lo_, self.x_scale_)
        self.clusterer_ = SimpleKMeans(n_clusters=self.n_clusters, seed=self.seed).fit(x_norm)
        labels = self.clusterer_.predict(x_norm)

        all_indices = np.arange(len(x))
        self.global_wins_ = self._pair_counts_from_indices(x_norm, strategy.astype(int), utility.astype(float), all_indices)
        self.local_wins_: dict[int, dict[tuple[int, int], tuple[float, float]]] = {}
        for cluster in range(self.n_clusters):
            indices = np.where(labels == cluster)[0]
            self.local_wins_[cluster] = self._pair_counts_from_indices(x_norm, strategy.astype(int), utility.astype(float), indices)
        return self

    def _pair_counts_from_indices(self, x_norm: np.ndarray, strategy: np.ndarray, utility: np.ndarray, indices: np.ndarray) -> dict[tuple[int, int], tuple[float, float]]:
        counts = {(a, b): [0.0, 0.0] for a in [1, 2, 3, 4] for b in [1, 2, 3, 4] if a != b}
        groups: dict[tuple[float, ...], list[int]] = {}
        for idx in indices:
            groups.setdefault(tuple(np.round(x_norm[idx], 8).tolist()), []).append(int(idx))
        for idxs in groups.values():
            for pos_i in range(len(idxs)):
                for pos_j in range(pos_i + 1, len(idxs)):
                    i, j = idxs[pos_i], idxs[pos_j]
                    if strategy[i] == strategy[j] or abs(utility[i] - utility[j]) < 1e-12:
                        continue
                    a, b = int(strategy[i]), int(strategy[j])
                    if utility[i] > utility[j]:
                        counts[(a, b)][0] += 1.0
                        counts[(b, a)][1] += 1.0
                    else:
                        counts[(a, b)][1] += 1.0
                        counts[(b, a)][0] += 1.0
        return {key: (value[0], value[1]) for key, value in counts.items()}

    def _prob(self, cluster: int, a: int, b: int) -> float:
        local_win, local_loss = self.local_wins_.get(cluster, {}).get((a, b), (0.0, 0.0))
        global_win, global_loss = self.global_wins_.get((a, b), (0.0, 0.0))
        local_total = local_win + local_loss
        global_prob = (global_win + 1.0) / (global_win + global_loss + 2.0)
        if local_total <= 0:
            return global_prob
        local_prob = local_win / local_total
        alpha = local_total / (local_total + self.shrinkage)
        return float(alpha * local_prob + (1.0 - alpha) * global_prob)

    def recommend(self, x: np.ndarray, strategies: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if strategies is None:
            strategies = self.strategies_
        x_norm = minmax_transform(x.astype(float), self.x_lo_, self.x_scale_)
        clusters = self.clusterer_.predict(x_norm)
        recommendations = []
        score_rows = []
        for cluster in clusters:
            scores = []
            for s in strategies:
                scores.append(sum(self._prob(int(cluster), int(s), int(other)) for other in strategies if int(other) != int(s)))
            score_row = np.array(scores, dtype=float)
            score_rows.append(score_row)
            recommendations.append(int(strategies[np.argmax(score_row)]))
        return np.array(recommendations, dtype=int), np.vstack(score_rows), clusters
