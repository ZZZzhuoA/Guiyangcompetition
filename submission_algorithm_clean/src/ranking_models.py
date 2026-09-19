from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.models import normalized_distance


def append_strategy_feature(x: np.ndarray, strategy: np.ndarray) -> np.ndarray:
    strategy = strategy.astype(float).reshape(-1, 1)
    one_hot = np.zeros((len(strategy), 4), dtype=float)
    for i, value in enumerate(strategy.astype(int).ravel()):
        if 1 <= value <= 4:
            one_hot[i, value - 1] = 1.0
    return np.hstack([x.astype(float), strategy, one_hot])


class MeanUtilityByStrategy:
    name = "mean_utility_by_strategy"

    def fit(self, x: np.ndarray, strategy: np.ndarray, utility: np.ndarray) -> "MeanUtilityByStrategy":
        self.global_mean_ = float(np.mean(utility))
        self.strategy_mean_ = {}
        for value in np.unique(strategy.astype(int)):
            self.strategy_mean_[int(value)] = float(np.mean(utility[strategy == value]))
        return self

    def predict(self, x: np.ndarray, strategy: np.ndarray) -> np.ndarray:
        return np.array([self.strategy_mean_.get(int(s), self.global_mean_) for s in strategy], dtype=float)


@dataclass
class KNNUtilityRegressor:
    k: int = 15

    @property
    def name(self) -> str:
        return f"knn_utility_k{self.k}"

    def fit(self, x: np.ndarray, strategy: np.ndarray, utility: np.ndarray) -> "KNNUtilityRegressor":
        self.z_ = append_strategy_feature(x, strategy)
        self.utility_ = utility.astype(float)
        self.ranges_ = np.ptp(self.z_, axis=0)
        return self

    def predict(self, x: np.ndarray, strategy: np.ndarray) -> np.ndarray:
        z = append_strategy_feature(x, strategy)
        preds = []
        for row in z:
            d = normalized_distance(row, self.z_, self.ranges_)
            idx = np.argsort(d)[: min(self.k, len(d))]
            weights = 1.0 / (d[idx] + 1e-6)
            preds.append(float(np.average(self.utility_[idx], weights=weights)))
        return np.array(preds, dtype=float)


@dataclass
class RegressionTreeUtility:
    max_depth: int = 6
    min_samples_leaf: int = 8

    @property
    def name(self) -> str:
        return f"reg_tree_d{self.max_depth}_leaf{self.min_samples_leaf}"

    def fit(self, x: np.ndarray, strategy: np.ndarray, utility: np.ndarray) -> "RegressionTreeUtility":
        self.z_ = append_strategy_feature(x, strategy)
        self.y_ = utility.astype(float)
        self.default_ = float(np.mean(self.y_))
        self.tree_ = self._build(self.z_, self.y_, depth=0)
        return self

    def _loss(self, y: np.ndarray) -> float:
        if len(y) == 0:
            return 0.0
        return float(np.var(y) * len(y))

    def _best_split(self, z: np.ndarray, y: np.ndarray) -> tuple[int | None, float | None, float]:
        parent = self._loss(y)
        best_feature = None
        best_threshold = None
        best_gain = 0.0
        for feature in range(z.shape[1]):
            values = np.unique(z[:, feature])
            if len(values) <= 1:
                continue
            thresholds = (values[:-1] + values[1:]) / 2.0
            for threshold in thresholds:
                left = z[:, feature] <= threshold
                right = ~left
                if np.sum(left) < self.min_samples_leaf or np.sum(right) < self.min_samples_leaf:
                    continue
                gain = parent - self._loss(y[left]) - self._loss(y[right])
                if gain > best_gain:
                    best_feature = feature
                    best_threshold = float(threshold)
                    best_gain = float(gain)
        return best_feature, best_threshold, best_gain

    def _build(self, z: np.ndarray, y: np.ndarray, depth: int) -> dict[str, object]:
        prediction = float(np.mean(y))
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf:
            return {"prediction": prediction}
        feature, threshold, gain = self._best_split(z, y)
        if feature is None or threshold is None or gain <= 1e-12:
            return {"prediction": prediction}
        left = z[:, feature] <= threshold
        return {
            "prediction": prediction,
            "feature": feature,
            "threshold": threshold,
            "left": self._build(z[left], y[left], depth + 1),
            "right": self._build(z[~left], y[~left], depth + 1),
        }

    def _predict_one(self, row: np.ndarray) -> float:
        node = self.tree_
        while "feature" in node:
            if row[int(node["feature"])] <= float(node["threshold"]):
                node = node["left"]
            else:
                node = node["right"]
        return float(node.get("prediction", self.default_))

    def predict(self, x: np.ndarray, strategy: np.ndarray) -> np.ndarray:
        z = append_strategy_feature(x, strategy)
        return np.array([self._predict_one(row) for row in z], dtype=float)


@dataclass
class RandomForestUtility:
    n_trees: int = 31
    max_depth: int = 7
    min_samples_leaf: int = 5
    feature_fraction: float = 0.75
    seed: int = 42

    @property
    def name(self) -> str:
        return f"reg_forest_t{self.n_trees}_d{self.max_depth}"

    def fit(self, x: np.ndarray, strategy: np.ndarray, utility: np.ndarray) -> "RandomForestUtility":
        rng = np.random.default_rng(self.seed)
        z = append_strategy_feature(x, strategy)
        self.trees_: list[tuple[np.ndarray, RegressionTreeUtility]] = []
        feature_count = max(1, int(round(z.shape[1] * self.feature_fraction)))
        for _ in range(self.n_trees):
            row_idx = rng.integers(0, len(utility), size=len(utility))
            feature_idx = np.sort(rng.choice(z.shape[1], size=feature_count, replace=False))
            tree = RegressionTreeUtility(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
            tree.z_ = z[row_idx][:, feature_idx]
            tree.y_ = utility[row_idx].astype(float)
            tree.default_ = float(np.mean(tree.y_))
            tree.tree_ = tree._build(tree.z_, tree.y_, depth=0)
            self.trees_.append((feature_idx, tree))
        return self

    def predict(self, x: np.ndarray, strategy: np.ndarray) -> np.ndarray:
        z = append_strategy_feature(x, strategy)
        preds = []
        for feature_idx, tree in self.trees_:
            preds.append(np.array([tree._predict_one(row) for row in z[:, feature_idx]], dtype=float))
        return np.mean(np.vstack(preds), axis=0)


def recommend_strategy(model: object, x: np.ndarray, strategies: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if strategies is None:
        strategies = np.array([1, 2, 3, 4], dtype=int)
    recommendations = []
    score_rows = []
    for row in x:
        candidate_x = np.repeat(row.reshape(1, -1), len(strategies), axis=0)
        scores = model.predict(candidate_x, strategies)
        score_rows.append(scores)
        recommendations.append(int(strategies[np.argmax(scores)]))
    return np.array(recommendations, dtype=int), np.vstack(score_rows)
