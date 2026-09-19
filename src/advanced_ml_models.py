from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.ranking_models import append_strategy_feature


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean, std


def _standardize_transform(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def _ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    z = np.hstack([np.ones((x.shape[0], 1)), x])
    eye = np.eye(z.shape[1])
    eye[0, 0] = 0.0
    return np.linalg.pinv(z.T @ z + alpha * eye) @ z.T @ y


def _ridge_predict(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    z = np.hstack([np.ones((x.shape[0], 1)), x])
    return z @ coef


@dataclass
class RidgeUtilityRegressor:
    alpha: float = 1.0

    @property
    def name(self) -> str:
        return f"ridge_utility_a{self.alpha:g}"

    def fit(self, x: np.ndarray, strategy: np.ndarray, utility: np.ndarray) -> "RidgeUtilityRegressor":
        z = append_strategy_feature(x, strategy)
        self.mean_, self.std_ = _standardize_fit(z)
        zn = _standardize_transform(z, self.mean_, self.std_)
        self.coef_ = _ridge_fit(zn, utility.astype(float), self.alpha)
        return self

    def predict(self, x: np.ndarray, strategy: np.ndarray) -> np.ndarray:
        z = append_strategy_feature(x, strategy)
        zn = _standardize_transform(z, self.mean_, self.std_)
        return _ridge_predict(zn, self.coef_)


@dataclass
class StrategyWiseRidgeUtility:
    alpha: float = 1.0

    @property
    def name(self) -> str:
        return f"strategy_wise_ridge_a{self.alpha:g}"

    def fit(self, x: np.ndarray, strategy: np.ndarray, utility: np.ndarray) -> "StrategyWiseRidgeUtility":
        self.global_mean_ = float(np.mean(utility))
        self.models_: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for s in [1, 2, 3, 4]:
            mask = strategy.astype(int) == s
            if np.sum(mask) < 8:
                continue
            mean, std = _standardize_fit(x[mask].astype(float))
            xn = _standardize_transform(x[mask].astype(float), mean, std)
            coef = _ridge_fit(xn, utility[mask].astype(float), self.alpha)
            self.models_[s] = (mean, std, coef)
        return self

    def predict(self, x: np.ndarray, strategy: np.ndarray) -> np.ndarray:
        out = []
        for row, s in zip(x.astype(float), strategy.astype(int)):
            model = self.models_.get(int(s))
            if model is None:
                out.append(self.global_mean_)
            else:
                mean, std, coef = model
                out.append(float(_ridge_predict(_standardize_transform(row.reshape(1, -1), mean, std), coef)[0]))
        return np.array(out, dtype=float)


@dataclass
class RBFKernelUtilityRegressor:
    gamma: float = 0.5
    alpha: float = 0.01

    @property
    def name(self) -> str:
        return f"rbf_kernel_g{self.gamma:g}_a{self.alpha:g}"

    def fit(self, x: np.ndarray, strategy: np.ndarray, utility: np.ndarray) -> "RBFKernelUtilityRegressor":
        z = append_strategy_feature(x, strategy)
        self.mean_, self.std_ = _standardize_fit(z)
        self.z_ = _standardize_transform(z, self.mean_, self.std_)
        d2 = ((self.z_[:, None, :] - self.z_[None, :, :]) ** 2).mean(axis=2)
        k = np.exp(-self.gamma * d2)
        self.dual_ = np.linalg.solve(k + self.alpha * np.eye(len(k)), utility.astype(float))
        return self

    def predict(self, x: np.ndarray, strategy: np.ndarray) -> np.ndarray:
        z = append_strategy_feature(x, strategy)
        zn = _standardize_transform(z, self.mean_, self.std_)
        d2 = ((zn[:, None, :] - self.z_[None, :, :]) ** 2).mean(axis=2)
        k = np.exp(-self.gamma * d2)
        return k @ self.dual_


@dataclass
class ExtraTreesUtilityRegressor:
    n_trees: int = 41
    max_depth: int = 8
    min_samples_leaf: int = 5
    feature_fraction: float = 0.7
    n_thresholds: int = 8
    seed: int = 42

    @property
    def name(self) -> str:
        return f"extra_trees_t{self.n_trees}_d{self.max_depth}"

    def fit(self, x: np.ndarray, strategy: np.ndarray, utility: np.ndarray) -> "ExtraTreesUtilityRegressor":
        rng = np.random.default_rng(self.seed)
        z = append_strategy_feature(x, strategy)
        self.trees_ = []
        for _ in range(self.n_trees):
            row_idx = rng.integers(0, len(utility), size=len(utility))
            tree = self._build(z[row_idx], utility[row_idx].astype(float), 0, rng)
            self.trees_.append(tree)
        return self

    def _loss(self, y: np.ndarray) -> float:
        return float(np.var(y) * len(y)) if len(y) else 0.0

    def _build(self, z: np.ndarray, y: np.ndarray, depth: int, rng: np.random.Generator) -> dict[str, object]:
        pred = float(np.mean(y))
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_leaf:
            return {"prediction": pred}
        n_features = max(1, int(round(z.shape[1] * self.feature_fraction)))
        features = rng.choice(z.shape[1], size=n_features, replace=False)
        parent = self._loss(y)
        best = (None, None, 0.0)
        for feature in features:
            lo, hi = float(np.min(z[:, feature])), float(np.max(z[:, feature]))
            if abs(hi - lo) < 1e-12:
                continue
            for threshold in rng.uniform(lo, hi, size=self.n_thresholds):
                left = z[:, feature] <= threshold
                right = ~left
                if np.sum(left) < self.min_samples_leaf or np.sum(right) < self.min_samples_leaf:
                    continue
                gain = parent - self._loss(y[left]) - self._loss(y[right])
                if gain > best[2]:
                    best = (int(feature), float(threshold), float(gain))
        feature, threshold, gain = best
        if feature is None or threshold is None or gain <= 1e-12:
            return {"prediction": pred}
        left = z[:, feature] <= threshold
        return {
            "prediction": pred,
            "feature": feature,
            "threshold": threshold,
            "left": self._build(z[left], y[left], depth + 1, rng),
            "right": self._build(z[~left], y[~left], depth + 1, rng),
        }

    def _predict_tree(self, tree: dict[str, object], row: np.ndarray) -> float:
        node = tree
        while "feature" in node:
            node = node["left"] if row[int(node["feature"])] <= float(node["threshold"]) else node["right"]
        return float(node["prediction"])

    def predict(self, x: np.ndarray, strategy: np.ndarray) -> np.ndarray:
        z = append_strategy_feature(x, strategy)
        preds = []
        for tree in self.trees_:
            preds.append(np.array([self._predict_tree(tree, row) for row in z], dtype=float))
        return np.mean(np.vstack(preds), axis=0)


@dataclass
class GradientBoostedStumpsUtility:
    n_estimators: int = 80
    learning_rate: float = 0.05
    min_samples_leaf: int = 8

    @property
    def name(self) -> str:
        return f"gb_stumps_n{self.n_estimators}_lr{self.learning_rate:g}"

    def fit(self, x: np.ndarray, strategy: np.ndarray, utility: np.ndarray) -> "GradientBoostedStumpsUtility":
        self.z_ = append_strategy_feature(x, strategy)
        self.init_ = float(np.mean(utility))
        pred = np.full(len(utility), self.init_, dtype=float)
        self.stumps_: list[tuple[int, float, float, float]] = []
        for _ in range(self.n_estimators):
            residual = utility.astype(float) - pred
            stump = self._best_stump(self.z_, residual)
            if stump is None:
                break
            feature, threshold, left_value, right_value = stump
            update = np.where(self.z_[:, feature] <= threshold, left_value, right_value)
            pred += self.learning_rate * update
            self.stumps_.append(stump)
        return self

    def _best_stump(self, z: np.ndarray, residual: np.ndarray) -> tuple[int, float, float, float] | None:
        best = None
        best_loss = float("inf")
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
                lv = float(np.mean(residual[left]))
                rv = float(np.mean(residual[right]))
                loss = float(np.sum((residual[left] - lv) ** 2) + np.sum((residual[right] - rv) ** 2))
                if loss < best_loss:
                    best_loss = loss
                    best = (int(feature), float(threshold), lv, rv)
        return best

    def predict(self, x: np.ndarray, strategy: np.ndarray) -> np.ndarray:
        z = append_strategy_feature(x, strategy)
        pred = np.full(z.shape[0], self.init_, dtype=float)
        for feature, threshold, left_value, right_value in self.stumps_:
            pred += self.learning_rate * np.where(z[:, feature] <= threshold, left_value, right_value)
        return pred
