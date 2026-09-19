from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.advanced_ml_models import RBFKernelUtilityRegressor, RidgeUtilityRegressor
from src.dual_view_models import DualViewClusterRecommender
from src.multi_metric_models import DualMetricRegressor
from src.pairwise_models import PairwiseLogisticRanker
from src.pareto_models import ParetoStrategyRecommender, pareto_mask_maximize
from src.ranking_models import RandomForestUtility
from src.stage2_eval import utility_from_metrics
from src.stage3_models import ClusterLocalUtilityRecommender


def _row_minmax(scores: np.ndarray) -> np.ndarray:
    scores = scores.astype(float)
    lo = np.min(scores, axis=1, keepdims=True)
    hi = np.max(scores, axis=1, keepdims=True)
    scale = np.where(np.abs(hi - lo) < 1e-12, 1.0, hi - lo)
    return (scores - lo) / scale


@dataclass
class FusionStrategyRecommender:
    w_cluster: float = 0.35
    w_pairwise: float = 0.25
    w_pareto: float = 0.20
    w_dual_view: float = 0.20
    pareto_bonus: float = 0.15

    @property
    def name(self) -> str:
        return (
            f"fusion_c{self.w_cluster:g}_p{self.w_pairwise:g}_"
            f"pa{self.w_pareto:g}_dv{self.w_dual_view:g}_b{self.pareto_bonus:g}"
        )

    def fit(self, x: np.ndarray, strategy: np.ndarray, metric1: np.ndarray, metric2: np.ndarray) -> "FusionStrategyRecommender":
        utility = utility_from_metrics(metric1, metric2, 0.5, 0.5)
        self.strategies_ = np.array([1, 2, 3, 4], dtype=int)

        self.cluster_model_ = ClusterLocalUtilityRecommender(n_clusters=3, metric1_weight=0.5, metric2_weight=0.5)
        self.cluster_model_.fit(x, strategy, metric1, metric2)

        self.pairwise_model_ = PairwiseLogisticRanker(epochs=800, learning_rate=0.05, l2=0.003)
        self.pairwise_model_.fit(x, strategy, utility)

        pareto_base = DualMetricRegressor(RandomForestUtility(n_trees=25, max_depth=6, min_samples_leaf=8))
        self.pareto_model_ = ParetoStrategyRecommender(pareto_base, tie_breaker="product")
        self.pareto_model_.fit(x, strategy, metric1, metric2)

        self.dual_view_model_ = DualViewClusterRecommender(resource_clusters=2, threat_clusters=2, tie_breaker="equal_weight")
        self.dual_view_model_.fit(x, strategy, metric1, metric2)
        return self

    def _score_all(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        _, cluster_scores, _, _ = self.cluster_model_.recommend(x)
        _, pairwise_scores = self.pairwise_model_.recommend(x)
        _, pareto_scores, pareto_m1, pareto_m2 = self.pareto_model_.recommend(x)
        _, dual_view_scores, _, _, _, _ = self.dual_view_model_.recommend(x)

        pareto_flags = []
        for m1, m2 in zip(pareto_m1, pareto_m2):
            pareto_flags.append(pareto_mask_maximize(np.vstack([m1, m2]).T).astype(float))
        pareto_flags = np.vstack(pareto_flags)

        normalized = {
            "cluster": _row_minmax(cluster_scores),
            "pairwise": _row_minmax(pairwise_scores),
            "pareto": _row_minmax(pareto_scores),
            "dual_view": _row_minmax(dual_view_scores),
            "pareto_flags": pareto_flags,
        }
        fused = (
            self.w_cluster * normalized["cluster"]
            + self.w_pairwise * normalized["pairwise"]
            + self.w_pareto * normalized["pareto"]
            + self.w_dual_view * normalized["dual_view"]
            + self.pareto_bonus * normalized["pareto_flags"]
        )
        return fused, normalized

    def recommend(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        fused, parts = self._score_all(x)
        pred = self.strategies_[np.argmax(fused, axis=1)]
        return pred.astype(int), fused, parts
