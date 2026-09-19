from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DualMetricRegressor:
    base_model: object
    name_prefix: str = "dual_metric"

    @property
    def name(self) -> str:
        return f"{self.name_prefix}_{self.base_model.name}"

    def fit(self, x: np.ndarray, strategy: np.ndarray, metric1: np.ndarray, metric2: np.ndarray) -> "DualMetricRegressor":
        self.metric1_model_ = self.base_model.__class__(**getattr(self.base_model, "__dict__", {}))
        self.metric2_model_ = self.base_model.__class__(**getattr(self.base_model, "__dict__", {}))
        self.metric1_model_.fit(x, strategy, metric1.astype(float))
        self.metric2_model_.fit(x, strategy, metric2.astype(float))
        self.m1_min_, self.m1_max_ = float(np.min(metric1)), float(np.max(metric1))
        self.m2_min_, self.m2_max_ = float(np.min(metric2)), float(np.max(metric2))
        return self

    def _normalize(self, values: np.ndarray, lo: float, hi: float) -> np.ndarray:
        scale = hi - lo if abs(hi - lo) > 1e-12 else 1.0
        return (values - lo) / scale

    def predict_metrics(self, x: np.ndarray, strategy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m1 = self.metric1_model_.predict(x, strategy)
        m2 = self.metric2_model_.predict(x, strategy)
        return m1, m2

    def predict(self, x: np.ndarray, strategy: np.ndarray) -> np.ndarray:
        m1, m2 = self.predict_metrics(x, strategy)
        return 0.5 * self._normalize(m1, self.m1_min_, self.m1_max_) + 0.5 * self._normalize(m2, self.m2_min_, self.m2_max_)
