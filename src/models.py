from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def normalized_distance(a: np.ndarray, b: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    scale = np.where(ranges == 0, 1.0, ranges)
    return np.mean(np.abs((b - a) / scale), axis=1)


class MajorityStrategy:
    name = "majority_strategy"

    def fit(self, x: np.ndarray, y: np.ndarray, metric1: np.ndarray, metric2: np.ndarray) -> "MajorityStrategy":
        values, counts = np.unique(y, return_counts=True)
        self.strategy_ = int(values[np.argmax(counts)])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(x.shape[0], self.strategy_, dtype=int)


@dataclass
class KNNStrategy:
    k: int = 7
    distance_weighted: bool = True

    @property
    def name(self) -> str:
        suffix = "weighted" if self.distance_weighted else "vote"
        return f"knn_{suffix}_k{self.k}"

    def fit(self, x: np.ndarray, y: np.ndarray, metric1: np.ndarray, metric2: np.ndarray) -> "KNNStrategy":
        self.x_ = x.astype(float)
        self.y_ = y.astype(int)
        self.ranges_ = np.ptp(self.x_, axis=0)
        self.strategies_ = np.unique(self.y_)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        preds = []
        for row in x.astype(float):
            d = normalized_distance(row, self.x_, self.ranges_)
            idx = np.argsort(d)[: self.k]
            scores = {int(strategy): 0.0 for strategy in self.strategies_}
            for i in idx:
                weight = 1.0 / (d[i] + 1e-9) if self.distance_weighted else 1.0
                scores[int(self.y_[i])] += weight
            preds.append(max(scores.items(), key=lambda item: (item[1], -item[0]))[0])
        return np.array(preds, dtype=int)


@dataclass
class StrategyPrototype:
    metric: str = "median"

    @property
    def name(self) -> str:
        return f"strategy_prototype_{self.metric}"

    def fit(self, x: np.ndarray, y: np.ndarray, metric1: np.ndarray, metric2: np.ndarray) -> "StrategyPrototype":
        self.x_ = x.astype(float)
        self.y_ = y.astype(int)
        self.ranges_ = np.ptp(self.x_, axis=0)
        self.strategies_ = np.unique(self.y_)
        centers = []
        for strategy in self.strategies_:
            part = self.x_[self.y_ == strategy]
            if self.metric == "mean":
                centers.append(np.mean(part, axis=0))
            else:
                centers.append(np.median(part, axis=0))
        self.centers_ = np.vstack(centers)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        preds = []
        for row in x.astype(float):
            d = normalized_distance(row, self.centers_, self.ranges_)
            preds.append(int(self.strategies_[np.argmin(d)]))
        return np.array(preds, dtype=int)


@dataclass
class OutcomeKNNRecommender:
    k: int = 9
    metric1_weight: float = 0.7
    metric2_weight: float = 0.3
    metric2_higher_is_better: bool = True

    @property
    def name(self) -> str:
        direction = "m2_high" if self.metric2_higher_is_better else "m2_low"
        return f"outcome_knn_k{self.k}_{direction}"

    def fit(self, x: np.ndarray, y: np.ndarray, metric1: np.ndarray, metric2: np.ndarray) -> "OutcomeKNNRecommender":
        self.x_ = x.astype(float)
        self.y_ = y.astype(int)
        self.metric1_ = metric1.astype(float)
        self.metric2_ = metric2.astype(float)
        self.ranges_ = np.ptp(self.x_, axis=0)
        self.strategies_ = np.unique(self.y_)
        self.m1_min_, self.m1_max_ = float(np.min(metric1)), float(np.max(metric1))
        self.m2_min_, self.m2_max_ = float(np.min(metric2)), float(np.max(metric2))
        return self

    def _norm(self, values: np.ndarray, lo: float, hi: float) -> np.ndarray:
        if abs(hi - lo) < 1e-12:
            return np.zeros_like(values, dtype=float)
        return (values - lo) / (hi - lo)

    def _utility(self, m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
        m1n = self._norm(m1, self.m1_min_, self.m1_max_)
        m2n = self._norm(m2, self.m2_min_, self.m2_max_)
        if not self.metric2_higher_is_better:
            m2n = 1.0 - m2n
        return self.metric1_weight * m1n + self.metric2_weight * m2n

    def predict_with_scores(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        preds = []
        all_scores = []
        fallback = float(np.mean(self._utility(self.metric1_, self.metric2_)))
        for row in x.astype(float):
            strategy_scores = []
            for strategy in self.strategies_:
                mask = self.y_ == strategy
                candidate_x = self.x_[mask]
                d = normalized_distance(row, candidate_x, self.ranges_)
                idx = np.argsort(d)[: min(self.k, len(d))]
                weights = 1.0 / (d[idx] + 1e-6)
                utilities = self._utility(self.metric1_[mask][idx], self.metric2_[mask][idx])
                score = float(np.average(utilities, weights=weights)) if len(idx) else fallback
                strategy_scores.append(score)
            scores = np.array(strategy_scores, dtype=float)
            all_scores.append(scores)
            preds.append(int(self.strategies_[np.argmax(scores)]))
        return np.array(preds, dtype=int), np.vstack(all_scores)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.predict_with_scores(x)[0]


@dataclass
class DecisionTreeStrategy:
    max_depth: int = 6
    min_samples_leaf: int = 8

    @property
    def name(self) -> str:
        return f"decision_tree_d{self.max_depth}_leaf{self.min_samples_leaf}"

    def fit(self, x: np.ndarray, y: np.ndarray, metric1: np.ndarray, metric2: np.ndarray) -> "DecisionTreeStrategy":
        self.classes_ = np.unique(y.astype(int))
        self.default_ = self._majority(y)
        self.tree_ = self._build(x.astype(float), y.astype(int), depth=0)
        return self

    def _majority(self, y: np.ndarray) -> int:
        values, counts = np.unique(y, return_counts=True)
        return int(values[np.argmax(counts)])

    def _gini(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return float(1.0 - np.sum(p * p))

    def _best_split(self, x: np.ndarray, y: np.ndarray) -> tuple[int | None, float | None, float]:
        parent = self._gini(y)
        best_feature = None
        best_threshold = None
        best_gain = 0.0
        for feature in range(x.shape[1]):
            values = np.unique(x[:, feature])
            if len(values) <= 1:
                continue
            thresholds = (values[:-1] + values[1:]) / 2.0
            for threshold in thresholds:
                left = x[:, feature] <= threshold
                right = ~left
                if np.sum(left) < self.min_samples_leaf or np.sum(right) < self.min_samples_leaf:
                    continue
                weighted = (np.sum(left) * self._gini(y[left]) + np.sum(right) * self._gini(y[right])) / len(y)
                gain = parent - weighted
                if gain > best_gain:
                    best_feature = feature
                    best_threshold = float(threshold)
                    best_gain = float(gain)
        return best_feature, best_threshold, best_gain

    def _build(self, x: np.ndarray, y: np.ndarray, depth: int) -> dict[str, object]:
        prediction = self._majority(y)
        if depth >= self.max_depth or len(np.unique(y)) == 1 or len(y) < 2 * self.min_samples_leaf:
            return {"prediction": prediction}
        feature, threshold, gain = self._best_split(x, y)
        if feature is None or threshold is None or gain <= 1e-12:
            return {"prediction": prediction}
        left = x[:, feature] <= threshold
        return {
            "prediction": prediction,
            "feature": feature,
            "threshold": threshold,
            "left": self._build(x[left], y[left], depth + 1),
            "right": self._build(x[~left], y[~left], depth + 1),
        }

    def _predict_one(self, row: np.ndarray) -> int:
        node = self.tree_
        while "feature" in node:
            if row[int(node["feature"])] <= float(node["threshold"]):
                node = node["left"]
            else:
                node = node["right"]
        return int(node.get("prediction", self.default_))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.array([self._predict_one(row) for row in x.astype(float)], dtype=int)


@dataclass
class RandomForestStrategy:
    n_trees: int = 31
    max_depth: int = 7
    min_samples_leaf: int = 5
    feature_fraction: float = 0.7
    seed: int = 42

    @property
    def name(self) -> str:
        return f"random_forest_t{self.n_trees}_d{self.max_depth}"

    def fit(self, x: np.ndarray, y: np.ndarray, metric1: np.ndarray, metric2: np.ndarray) -> "RandomForestStrategy":
        rng = np.random.default_rng(self.seed)
        self.classes_ = np.unique(y.astype(int))
        self.trees_: list[tuple[np.ndarray, DecisionTreeStrategy]] = []
        n_features = x.shape[1]
        feature_count = max(1, int(round(n_features * self.feature_fraction)))
        for _ in range(self.n_trees):
            row_idx = rng.integers(0, len(y), size=len(y))
            feature_idx = np.sort(rng.choice(n_features, size=feature_count, replace=False))
            tree = DecisionTreeStrategy(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
            tree.fit(x[row_idx][:, feature_idx], y[row_idx], metric1[row_idx], metric2[row_idx])
            self.trees_.append((feature_idx, tree))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        votes = []
        for feature_idx, tree in self.trees_:
            votes.append(tree.predict(x[:, feature_idx]))
        vote_matrix = np.vstack(votes).T
        preds = []
        for row in vote_matrix:
            scores = {int(label): int(np.sum(row == label)) for label in self.classes_}
            preds.append(max(scores.items(), key=lambda item: (item[1], -item[0]))[0])
        return np.array(preds, dtype=int)
