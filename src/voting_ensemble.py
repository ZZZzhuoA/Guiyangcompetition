from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _row_minmax(values: np.ndarray) -> np.ndarray:
    lo = float(np.min(values))
    hi = float(np.max(values))
    if abs(hi - lo) < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - lo) / (hi - lo)


@dataclass
class VotingUtilityEnsemble:
    models: list[object]
    mode: str = "soft"

    @property
    def name(self) -> str:
        return f"voting_{self.mode}_{len(self.models)}models"

    def fit(self, x: np.ndarray, strategy: np.ndarray, utility: np.ndarray) -> "VotingUtilityEnsemble":
        self.models_ = []
        for model in self.models:
            fitted = model.__class__(**getattr(model, "__dict__", {}))
            fitted.fit(x, strategy, utility)
            self.models_.append(fitted)
        return self

    def predict(self, x: np.ndarray, strategy: np.ndarray) -> np.ndarray:
        groups: dict[tuple[float, ...], list[int]] = {}
        for idx, row in enumerate(x):
            groups.setdefault(tuple(np.round(row.astype(float), 10).tolist()), []).append(idx)

        out = np.zeros(len(strategy), dtype=float)
        for idxs in groups.values():
            idx_array = np.array(idxs, dtype=int)
            group_strategy = strategy[idx_array].astype(int)
            model_scores = []
            for model in self.models_:
                score = model.predict(x[idx_array], group_strategy)
                model_scores.append(_row_minmax(score.astype(float)))
            score_matrix = np.vstack(model_scores)

            if self.mode == "hard":
                votes = np.zeros(len(idx_array), dtype=float)
                for row in score_matrix:
                    votes[int(np.argmax(row))] += 1.0
                out[idx_array] = votes
            elif self.mode == "rank":
                rank_scores = np.zeros(len(idx_array), dtype=float)
                for row in score_matrix:
                    order = np.argsort(row)[::-1]
                    for rank, pos in enumerate(order):
                        rank_scores[int(pos)] += len(row) - rank
                out[idx_array] = rank_scores
            else:
                out[idx_array] = np.mean(score_matrix, axis=0)
        return out
