"""可替换的动作价值 / 策略模型。统一 recommend(x) -> (LJCL, scores[3])。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.finals.features import rule_from_features
from src.finals.models import PairwiseLogistic, RandomForestUtility, StrategyScorer, _minmax
from src.finals.schema import STRATEGIES


def _groups(group_id: np.ndarray) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for i, gid in enumerate(group_id):
        grouped.setdefault(str(gid), []).append(i)
    return grouped


def scores_to_recommend(score_rows: np.ndarray, strategies=STRATEGIES) -> tuple[np.ndarray, np.ndarray]:
    strategies = np.array(list(strategies), dtype=int)
    recs = np.array([int(strategies[int(np.argmax(row))]) for row in score_rows], dtype=int)
    return recs, np.asarray(score_rows, dtype=float)


class ConstantScorer:
    def __init__(self, action: int = 1):
        self.action = int(action)
        self.name = f"constant_{action}"

    def fit(self, x, strategy, utility, group_id, sample_weight=None):
        return self

    def recommend(self, x: np.ndarray, strategies=STRATEGIES) -> tuple[np.ndarray, np.ndarray]:
        scores = np.zeros((len(x), len(list(strategies))), dtype=float)
        scores[:, self.action - 1] = 1.0
        return np.full(len(x), self.action, dtype=int), scores


class RuleScorer:
    name = "rule"

    def fit(self, x, strategy, utility, group_id, sample_weight=None):
        return self

    def recommend(self, x: np.ndarray, strategies=STRATEGIES) -> tuple[np.ndarray, np.ndarray]:
        recs = np.array([rule_from_features(row) for row in x], dtype=int)
        scores = np.zeros((len(x), len(list(strategies))), dtype=float)
        for i, action in enumerate(recs):
            scores[i, int(action) - 1] = 1.0
        return recs, scores


class ForestScorer:
    name = "rf_utility"

    def __init__(self, **kwargs):
        self.base = RandomForestUtility(**kwargs)

    def fit(self, x, strategy, utility, group_id, sample_weight=None):
        self.base.fit(x, strategy, utility, sample_weight=sample_weight)
        return self

    def recommend(self, x: np.ndarray, strategies=STRATEGIES) -> tuple[np.ndarray, np.ndarray]:
        strategies = np.array(list(strategies), dtype=int)
        rows = []
        for row in x:
            cand = np.repeat(row.reshape(1, -1), len(strategies), axis=0)
            rows.append(self.base.predict(cand, strategies))
        return scores_to_recommend(np.vstack(rows), strategies)


class PairScorer:
    name = "pairwise"

    def __init__(self):
        self.pair = PairwiseLogistic()

    def fit(self, x, strategy, utility, group_id, sample_weight=None):
        self.pair.fit(x, strategy, utility, group_id)
        return self

    def recommend(self, x: np.ndarray, strategies=STRATEGIES) -> tuple[np.ndarray, np.ndarray]:
        rows = np.vstack([self.pair.scores(row, strategies) for row in x])
        return scores_to_recommend(rows, strategies)


@dataclass
class RidgeQ:
    l2: float = 1.0
    name: str = "ridge_q"

    def fit(self, x, strategy, utility, group_id, sample_weight=None):
        from src.finals.models import append_strategy_feature

        z = append_strategy_feature(x, strategy)
        y = utility.astype(float)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        w = w / max(w.mean(), 1e-12)
        zw = z * w.reshape(-1, 1)
        xtx = zw.T @ z + self.l2 * np.eye(z.shape[1])
        xty = zw.T @ y
        self.coef_ = np.linalg.solve(xtx, xty)
        return self

    def recommend(self, x: np.ndarray, strategies=STRATEGIES) -> tuple[np.ndarray, np.ndarray]:
        from src.finals.models import append_strategy_feature

        strategies = np.array(list(strategies), dtype=int)
        rows = []
        for row in x:
            cand = np.repeat(row.reshape(1, -1), len(strategies), axis=0)
            z = append_strategy_feature(cand, strategies)
            rows.append(z @ self.coef_)
        return scores_to_recommend(np.vstack(rows), strategies)


@dataclass
class MLPQ:
    hidden: int = 32
    epochs: int = 250
    lr: float = 0.05
    l2: float = 0.001
    seed: int = 0
    name: str = "mlp_q"

    def fit(self, x, strategy, utility, group_id, sample_weight=None):
        rng = np.random.default_rng(self.seed)
        grouped = _groups(group_id)
        xs, ys, ws = [], [], []
        weight = np.ones(len(utility)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        for idxs in grouped.values():
            q = np.full(3, np.nan)
            for i in idxs:
                q[int(strategy[i]) - 1] = float(utility[i])
            if np.any(np.isnan(q)):
                continue
            xs.append(x[idxs[0]])
            ys.append(q)
            ws.append(float(np.mean(weight[idxs])))
        if not xs:
            self.W1 = np.zeros((x.shape[1], self.hidden))
            self.b1 = np.zeros(self.hidden)
            self.W2 = np.zeros((self.hidden, 3))
            self.b2 = np.zeros(3)
            self.lo_ = np.zeros(x.shape[1])
            self.scale_ = np.ones(x.shape[1])
            return self
        X = np.vstack(xs)
        lo, scale = X.min(axis=0), np.where(X.max(axis=0) - X.min(axis=0) < 1e-12, 1.0, X.max(axis=0) - X.min(axis=0))
        self.lo_, self.scale_ = lo, scale
        Xn = (X - lo) / scale
        Y = np.vstack(ys)
        W = np.array(ws, dtype=float).reshape(-1, 1)
        W = W / max(W.mean(), 1e-12)
        d = Xn.shape[1]
        self.W1 = rng.normal(0, 0.15, size=(d, self.hidden))
        self.b1 = np.zeros(self.hidden)
        self.W2 = rng.normal(0, 0.15, size=(self.hidden, 3))
        self.b2 = np.zeros(3)
        for _ in range(self.epochs):
            h = np.maximum(0.0, Xn @ self.W1 + self.b1)
            pred = h @ self.W2 + self.b2
            err = (pred - Y) * W
            dW2 = h.T @ err / len(Xn) + self.l2 * self.W2
            db2 = err.mean(axis=0)
            dh = (err @ self.W2.T) * (h > 0)
            dW1 = Xn.T @ dh / len(Xn) + self.l2 * self.W1
            db1 = dh.mean(axis=0)
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
        return self

    def recommend(self, x: np.ndarray, strategies=STRATEGIES) -> tuple[np.ndarray, np.ndarray]:
        xn = (x - self.lo_) / self.scale_
        h = np.maximum(0.0, xn @ self.W1 + self.b1)
        scores = h @ self.W2 + self.b2
        return scores_to_recommend(scores, strategies)


class SoftmaxPolicy:
    """用对照样本的 oracle 最优策略做分类，不是整局 S_LJCL。"""

    name = "oracle_clf"
    epochs = 250
    lr = 0.08
    l2 = 0.004

    def fit(self, x, strategy, utility, group_id, sample_weight=None):
        grouped = _groups(group_id)
        xs, ys, ws = [], [], []
        weight = np.ones(len(utility)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        for idxs in grouped.values():
            values = np.asarray([utility[i] for i in idxs], dtype=float)
            # 并列组没有分类标签；argmax 会无条件把第一行（通常是策略 1）
            # 当作正确答案，导致分类器人为塌缩到策略 1。
            if float(np.max(values) - np.min(values)) <= 1e-12:
                continue
            best = idxs[int(np.argmax(values))]
            xs.append(x[idxs[0]])
            ys.append(int(strategy[best]) - 1)
            ws.append(float(np.mean(weight[idxs])))
        if not xs:
            self.coef_ = np.zeros((x.shape[1] + 1, 3))
            self.lo_ = np.zeros(x.shape[1])
            self.scale_ = np.ones(x.shape[1])
            return self
        X = np.vstack(xs)
        self.lo_ = X.min(axis=0)
        self.scale_ = np.where(X.max(axis=0) - self.lo_ < 1e-12, 1.0, X.max(axis=0) - self.lo_)
        Xn = np.hstack([np.ones((len(X), 1)), (X - self.lo_) / self.scale_])
        y = np.array(ys, dtype=int)
        w = np.array(ws, dtype=float)
        w = w / max(w.mean(), 1e-12)
        coef = np.zeros((Xn.shape[1], 3))
        for _ in range(self.epochs):
            logits = np.clip(Xn @ coef, -30, 30)
            logits = logits - logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            prob = exp / exp.sum(axis=1, keepdims=True)
            onehot = np.zeros_like(prob)
            onehot[np.arange(len(y)), y] = 1.0
            grad = Xn.T @ ((prob - onehot) * w.reshape(-1, 1)) / len(y)
            grad[1:] += self.l2 * coef[1:]
            coef -= self.lr * grad
        self.coef_ = coef
        return self

    def recommend(self, x: np.ndarray, strategies=STRATEGIES) -> tuple[np.ndarray, np.ndarray]:
        xn = np.hstack([np.ones((len(x), 1)), (x - self.lo_) / self.scale_])
        logits = np.clip(xn @ self.coef_, -30, 30)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        scores = exp / exp.sum(axis=1, keepdims=True)
        return scores_to_recommend(scores, strategies)


class BlendScorer:
    """森林 + pairwise，当前默认提交结构。"""

    name = "rf_pair"

    def __init__(self, pairwise_weight: float = 0.35):
        self.inner = StrategyScorer(pairwise_weight=pairwise_weight)

    def fit(self, x, strategy, utility, group_id, sample_weight=None):
        self.inner.fit(x, strategy, utility, group_id, sample_weight=sample_weight)
        return self

    def recommend(self, x: np.ndarray, strategies=STRATEGIES) -> tuple[np.ndarray, np.ndarray]:
        return self.inner.recommend(x, strategies=strategies)


# 保留给 pickle 旧模型：StrategyScorer 本身已有 recommend
_ = _minmax
