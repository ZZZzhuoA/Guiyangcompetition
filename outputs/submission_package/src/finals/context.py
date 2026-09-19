"""跨触发缓存：推理时用上一决策点构造时序特征，训练时按决策点递推。

除了时序特征用的几个标量，这里还留了最近 K 次触发的特征向量和已选动作，
供序列模型（见 src/finals/sequence.py）在推理时拼窗口。赛方接口每次只给当前
三张表，序列必须由我们自己缓存；第一次触发窗口里只有一帧，之后逐渐填满。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.finals.sequence import WINDOW, build_window
from src.finals.snapshot import Snapshot


@dataclass
class DecisionContext:
    prev_ids: set[int] = field(default_factory=set)
    seen_ids: set[int] = field(default_factory=set)
    prev_min_range: float | None = None
    prev_min_tclose: float | None = None
    prev_n_alive: float | None = None
    prev_n_can_links: float | None = None
    prev_n_uncovered: float | None = None
    last_ljcl: int = 0
    last_time: float | None = None
    horizon: float = 220.0
    window: int = WINDOW
    hist_x: list = field(default_factory=list)
    hist_a: list = field(default_factory=list)

    def window_with(self, feat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """当前帧 + 缓存里的历史，拼成 (1, K, F) / (1, K-1) / (1,) 的批。"""
        x_hist, a_prev, hist_len = build_window(self.hist_x, self.hist_a, feat, window=self.window)
        return x_hist[None, ...], a_prev[None, ...], np.array([hist_len], dtype=int)

    def push_history(self, feat: np.ndarray, ljcl: int) -> None:
        """必须在 window_with 之后调用：窗口是「过去的帧 + 当前帧」。"""
        self.hist_x.append(np.asarray(feat, dtype=float))
        self.hist_a.append(int(ljcl))
        if len(self.hist_x) > self.window:
            self.hist_x = self.hist_x[-self.window:]
            self.hist_a = self.hist_a[-self.window:]

    def commit(self, snapshot: Snapshot, stats: dict[str, float], ljcl: int) -> None:
        ids = {int(t.target_id) for t in snapshot.targets}
        self.prev_ids = ids
        self.seen_ids |= ids
        self.prev_min_range = float(stats.get("min_range", 0.0))
        self.prev_min_tclose = float(stats.get("min_tclose", 0.0))
        self.prev_n_alive = float(stats.get("n_alive", 0.0))
        self.prev_n_can_links = float(stats.get("n_can_links", 0.0))
        self.prev_n_uncovered = float(stats.get("n_uncovered", 0.0))
        self.last_ljcl = int(ljcl)
        self.last_time = float(snapshot.time)
