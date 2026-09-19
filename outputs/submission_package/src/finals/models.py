"""三策略收益模型：随机森林打分 + pairwise 纠偏。仅依赖 numpy。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.finals.schema import STRATEGIES


def append_strategy_feature(x: np.ndarray, strategy: np.ndarray, n_strategies: int = 3) -> np.ndarray:
    strategy = strategy.astype(int).reshape(-1)
    one_hot = np.zeros((len(strategy), n_strategies), dtype=float)
    for i, value in enumerate(strategy):
        if 1 <= value <= n_strategies:
            one_hot[i, value - 1] = 1.0
    return np.hstack([x.astype(float), strategy.astype(float).reshape(-1, 1), one_hot])


def _loss(y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    return float(np.var(y) * len(y))


def _best_split(z: np.ndarray, y: np.ndarray, min_leaf: int) -> tuple[int | None, float | None]:
    parent = _loss(y)
    best_gain = 0.0
    best_j, best_thr = None, None
    for j in range(z.shape[1]):
        order = np.argsort(z[:, j])
        col = z[order, j]
        yy = y[order]
        for i in range(min_leaf, len(y) - min_leaf + 1, max(1, len(y) // 24)):
            if col[i] == col[i - 1]:
                continue
            left, right = yy[:i], yy[i:]
            gain = parent - _loss(left) - _loss(right)
            if gain > best_gain:
                best_gain = gain
                best_j = j
                best_thr = 0.5 * (col[i] + col[i - 1])
    return best_j, best_thr


def _build_tree(z: np.ndarray, y: np.ndarray, depth: int, max_depth: int, min_leaf: int):
    if depth >= max_depth or len(y) < 2 * min_leaf or float(np.var(y)) < 1e-12:
        return {"leaf": True, "value": float(np.mean(y))}
    j, thr = _best_split(z, y, min_leaf)
    if j is None:
        return {"leaf": True, "value": float(np.mean(y))}
    mask = z[:, j] <= thr
    if mask.sum() < min_leaf or (~mask).sum() < min_leaf:
        return {"leaf": True, "value": float(np.mean(y))}
    return {
        "leaf": False,
        "j": int(j),
        "thr": float(thr),
        "left": _build_tree(z[mask], y[mask], depth + 1, max_depth, min_leaf),
        "right": _build_tree(z[~mask], y[~mask], depth + 1, max_depth, min_leaf),
    }


def _predict_tree(node: dict, row: np.ndarray) -> float:
    while not node["leaf"]:
        node = node["left"] if row[node["j"]] <= node["thr"] else node["right"]
    return float(node["value"])


@dataclass
class RandomForestUtility:
    n_trees: int = 12
    max_depth: int = 4
    min_samples_leaf: int = 6
    feature_fraction: float = 0.75
    seed: int = 42

    def fit(self, x: np.ndarray, strategy: np.ndarray, utility: np.ndarray, sample_weight: np.ndarray | None = None) -> "RandomForestUtility":
        rng = np.random.default_rng(self.seed)
        z = append_strategy_feature(x, strategy)
        y = utility.astype(float)
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        probs = sample_weight / sample_weight.sum()
        self.trees_ = []
        feature_count = max(1, int(round(z.shape[1] * self.feature_fraction)))
        for _ in range(self.n_trees):
            row_idx = rng.choice(len(y), size=len(y), replace=True, p=probs)
            feat_idx = np.sort(rng.choice(z.shape[1], size=feature_count, replace=False))
            tree = _build_tree(z[row_idx][:, feat_idx], y[row_idx], 0, self.max_depth, self.min_samples_leaf)
            self.trees_.append((feat_idx, tree))
        return self

    def predict(self, x: np.ndarray, strategy: np.ndarray) -> np.ndarray:
        z = append_strategy_feature(x, strategy)
        preds = [np.array([_predict_tree(tree, row[feat]) for row in z], dtype=float) for feat, tree in self.trees_]
        return np.mean(np.vstack(preds), axis=0)


class PairwiseLogistic:
    def __init__(self, epochs: int = 200, learning_rate: float = 0.08, l2: float = 0.003):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.l2 = l2

    def _pair_feat(self, row: np.ndarray, a: int, b: int) -> np.ndarray:
        one_a = np.zeros(3)
        one_b = np.zeros(3)
        one_a[a - 1] = 1.0
        one_b[b - 1] = 1.0
        diff = one_a - one_b
        return np.concatenate([row, one_a, one_b, diff, row * diff[0], row * diff[1], row * diff[2]])

    def fit(self, x: np.ndarray, strategy: np.ndarray, utility: np.ndarray, group_id: np.ndarray) -> "PairwiseLogistic":
        lo = x.min(axis=0)
        scale = np.where(x.max(axis=0) - lo < 1e-12, 1.0, x.max(axis=0) - lo)
        self.lo_, self.scale_ = lo, scale
        xn = (x - lo) / scale
        feats, labels = [], []
        groups: dict[str, list[int]] = {}
        for i, gid in enumerate(group_id):
            groups.setdefault(str(gid), []).append(i)
        for idxs in groups.values():
            for i in range(len(idxs)):
                for j in range(i + 1, len(idxs)):
                    a, b = idxs[i], idxs[j]
                    if int(strategy[a]) == int(strategy[b]) or abs(utility[a] - utility[b]) < 1e-12:
                        continue
                    feats.append(self._pair_feat(xn[a], int(strategy[a]), int(strategy[b])))
                    labels.append(1.0 if utility[a] > utility[b] else 0.0)
                    feats.append(self._pair_feat(xn[b], int(strategy[b]), int(strategy[a])))
                    labels.append(1.0 if utility[b] > utility[a] else 0.0)
        if not feats:
            for idxs in groups.values():
                for i in range(len(idxs)):
                    for j in range(i + 1, len(idxs)):
                        a, b = idxs[i], idxs[j]
                        if int(strategy[a]) == int(strategy[b]):
                            continue
                        feats.append(self._pair_feat(xn[a], int(strategy[a]), int(strategy[b])))
                        labels.append(1.0 if utility[a] >= utility[b] else 0.0)
        if not feats:
            self.coef_ = np.zeros(1)
            self.dim_ = 1
            return self
        z = np.vstack(feats)
        y = np.array(labels, dtype=float)
        z = np.hstack([np.ones((len(z), 1)), z])
        coef = np.zeros(z.shape[1])
        for _ in range(self.epochs):
            logits = np.clip(z @ coef, -30, 30)
            pred = 1.0 / (1.0 + np.exp(-logits))
            grad = z.T @ (pred - y) / len(y)
            grad[1:] += self.l2 * coef[1:]
            coef -= self.learning_rate * grad
        self.coef_ = coef
        self.dim_ = z.shape[1]
        return self

    def score_pair(self, row: np.ndarray, a: int, b: int) -> float:
        xn = (row - self.lo_) / self.scale_
        feat = np.concatenate([[1.0], self._pair_feat(xn, a, b)])
        if len(feat) != len(self.coef_):
            return 0.0
        logit = float(np.clip(feat @ self.coef_, -30, 30))
        return 1.0 / (1.0 + np.exp(-logit))

    def scores(self, row: np.ndarray, strategies=STRATEGIES) -> np.ndarray:
        out = []
        for s in strategies:
            wins = [self.score_pair(row, s, o) for o in strategies if o != s]
            out.append(float(np.mean(wins)) if wins else 0.0)
        return np.array(out, dtype=float)


def _minmax(values: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(values)), float(np.max(values))
    if abs(hi - lo) < 1e-12:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


@dataclass
class StrategyScorer:
    pairwise_weight: float = 0.35

    def fit(self, x, strategy, utility, group_id, sample_weight=None) -> "StrategyScorer":
        self.base_ = RandomForestUtility()
        self.pair_ = PairwiseLogistic()
        self.base_.fit(x, strategy, utility, sample_weight=sample_weight)
        self.pair_.fit(x, strategy, utility, group_id)
        return self

    def predict(self, x: np.ndarray, strategy: np.ndarray) -> np.ndarray:
        base = self.base_.predict(x, strategy)
        out = np.zeros(len(strategy), dtype=float)
        for i, row in enumerate(x):
            pair_scores = self.pair_.scores(row)
            pair_value = float(pair_scores[int(strategy[i]) - 1])
            out[i] = pair_value
        # mix after per-row min-max across the three strategies in recommend()
        self._last_base_ = base
        return 0.65 * _minmax(base) + 0.35 * _minmax(out)

    def recommend(self, x: np.ndarray, strategies=STRATEGIES) -> tuple[np.ndarray, np.ndarray]:
        recs, score_rows = [], []
        strategies = np.array(list(strategies), dtype=int)
        for row in x:
            cand_x = np.repeat(row.reshape(1, -1), len(strategies), axis=0)
            base = self.base_.predict(cand_x, strategies)
            pair = self.pair_.scores(row, strategies)
            scores = (1.0 - self.pairwise_weight) * _minmax(base) + self.pairwise_weight * _minmax(pair)
            score_rows.append(scores)
            recs.append(int(strategies[int(np.argmax(scores))]))
        return np.array(recs, dtype=int), np.vstack(score_rows)


def kind_weights(kind: np.ndarray) -> np.ndarray:
    mapping = {
        "t0": 3.0,
        "fork": 2.0,
        "slice": 2.0,
        "onpolicy": 0.3,
        "matched": 1.0,
        "observed_positive": 2.0,
        "observed_hard_negative": 1.0,
        "observed_idle": 0.2,
    }
    return np.array([mapping.get(str(k), 1.0) for k in kind], dtype=float)
