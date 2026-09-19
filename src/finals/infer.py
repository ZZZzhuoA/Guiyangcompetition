"""实时接口：当前态势 -> LJCL ∈ {1,2,3}。"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from src.finals.context import DecisionContext
from src.finals.features import extract_feature_dict, extract_features, rule_from_features
from src.finals.schema import HISTORY_MIN_DT
from src.finals.snapshot import Snapshot, snapshot_from_interface


class StateCache:
    def __init__(self, horizon: float = 220.0):
        self.ctx = DecisionContext(horizon=horizon)

    def commit(self, snapshot: Snapshot, stats: dict[str, float], ljcl: int) -> None:
        self.ctx.commit(snapshot, stats, ljcl)


def rule_recommend(snapshot: Snapshot) -> int:
    return rule_from_features(extract_features(snapshot))


class FinalsPolicy:
    def __init__(self, scorer=None):
        self.scorer = scorer
        self.cache = StateCache()

    def recommend_snapshot(self, snapshot: Snapshot) -> dict:
        from src.finals.features import FEATURE_NAMES, dict_to_vector

        ctx = self.cache.ctx
        feat_dict = extract_feature_dict(snapshot, context=ctx)
        x = dict_to_vector(feat_dict, FEATURE_NAMES).reshape(1, -1)
        if self.scorer is None:
            choice = rule_from_features(x[0])
            scores = np.zeros(3)
            scores[choice - 1] = 1.0
        else:
            kwargs = {}
            if getattr(self.scorer, "needs_history", False):
                x_hist, a_prev, hist_len = ctx.window_with(x[0])
                kwargs = {"x_hist": x_hist, "a_prev": a_prev, "hist_len": hist_len}
            choice_arr, score_rows = self.scorer.recommend(x, **kwargs)
            choice = int(choice_arr[0])
            scores = score_rows[0]
        should_record = ctx.last_time is None or (float(snapshot.time) - float(ctx.last_time)) >= HISTORY_MIN_DT - 1e-9
        if should_record:
            ctx.push_history(x[0], int(choice))
            self.cache.commit(snapshot, feat_dict, int(choice))
        return {"LJCL": int(choice), "scores": [float(v) for v in scores]}

    def recommend_payload(self, payload: dict, time: float = 0.0) -> dict:
        snapshot = snapshot_from_interface(payload, time=time)
        return self.recommend_snapshot(snapshot)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump({"scorer": self.scorer, "name": getattr(self.scorer, "name", None)}, stream)

    @classmethod
    def load(cls, path: Path) -> "FinalsPolicy":
        with Path(path).open("rb") as stream:
            payload = pickle.load(stream)
        if isinstance(payload, dict) and "scorer" in payload:
            return cls(scorer=payload["scorer"])
        return cls(scorer=payload)
