"""贯序候选：本段收益、混合标签、fitted-Q，以及带序列编码器的时间序列 RL。

label_key    监督用哪一列（见 src/finals/reward.py 与 seq_set.LABEL_COLUMNS）
reward_key   fitted-Q 的即时奖励用哪一列
needs_history 要不要 x_hist / a_prev / hist_len（见 src/finals/sequence.py）

消融格子：状态表示 × 是否做贝尔曼备份，用来分清增益来自序列还是来自 RL。

|            | 监督（无备份） | fitted-Q       |
| 单帧       | seq_mix        | seq_fqi        |
| K 帧拼接   | seq_stack_mix  | seq_stack_fqi  |
| 递归编码   | seq_recur_mix  | seq_recur_fqi  |
"""

from __future__ import annotations

import numpy as np

from src.finals.learners import BlendScorer, ForestScorer, PairScorer, scores_to_recommend
from src.finals.models import RandomForestUtility
from src.finals.reward import GAMMA
from src.finals.schema import STRATEGIES
from src.finals.sequence import WINDOW, empty_window, make_encoder, push_window


class SeqDeltaScorer(ForestScorer):
    """只学本段拦截增量，最短视的对照。"""

    name = "seq_delta"
    label_key = "r_delta"


class SeqMixScorer(ForestScorer):
    """本段增量 + λ 终局拦截率。"""

    name = "seq_mix"
    label_key = "mix"


class SeqShapedScorer(ForestScorer):
    """本段增量 + 势函数差 - 高威胁漏防修正。"""

    name = "seq_shaped"
    label_key = "r_shaped"


class SeqPairScorer(PairScorer):
    name = "seq_pair"
    label_key = "mix"


class SeqBlendScorer(BlendScorer):
    name = "seq_blend"
    label_key = "mix"


class SeqFittedQ:
    """离线 fitted-Q：y = r + γ max_a' Q(x_next, a')。

    r 默认取 r_shaped（含势函数 shaping），γ 与 shaping 用的同一个值，
    否则势函数的策略不变性不成立。
    """

    name = "seq_fqi"
    label_key = "r_shaped"
    reward_key = "r_shaped"
    needs_transition = True

    def __init__(self, n_iter: int = 5, gamma: float = GAMMA):
        self.n_iter = n_iter
        self.gamma = gamma
        self.base = RandomForestUtility()

    def fit(self, x, strategy, utility, group_id, sample_weight=None, x_next=None, done=None, reward=None):
        r = np.asarray(utility if reward is None else reward, dtype=float)
        y = r.copy()
        self.base.fit(x, strategy, y, sample_weight=sample_weight)
        if x_next is None:
            return self
        done_mask = np.zeros(len(x), dtype=float) if done is None else np.asarray(done, dtype=float)
        strategies = np.array(list(STRATEGIES), dtype=int)
        for _ in range(max(0, self.n_iter - 1)):
            boot = np.zeros(len(x), dtype=float)
            for i, row in enumerate(x_next):
                if done_mask[i] >= 1.0:
                    continue
                cand = np.repeat(np.asarray(row, dtype=float).reshape(1, -1), len(strategies), axis=0)
                boot[i] = float(np.max(self.base.predict(cand, strategies)))
            y = r + self.gamma * boot
            self.base.fit(x, strategy, y, sample_weight=sample_weight)
        return self

    def recommend(self, x: np.ndarray, strategies=STRATEGIES):
        strategies = np.array(list(strategies), dtype=int)
        rows = []
        for row in x:
            cand = np.repeat(row.reshape(1, -1), len(strategies), axis=0)
            rows.append(self.base.predict(cand, strategies))
        score_rows = np.vstack(rows)
        recs = np.array([int(strategies[int(np.argmax(row))]) for row in score_rows], dtype=int)
        return recs, score_rows


class SeqFittedQRaw(SeqFittedQ):
    """不带 shaping 的 fitted-Q，用来看 shaping 到底有没有帮助。"""

    name = "seq_fqi_raw"
    label_key = "r_delta"
    reward_key = "r_delta"


class SeqEncodedScorer:
    """时间序列基座：先把最近 K 拍编码成状态，再用森林回归 Q(h,a)。

    这一层不做贝尔曼备份，只监督 label_key。它存在的意义是当对照：
    如果它已经打过单帧的 seq_mix，说明历史有用；如果 fitted-Q 版又打过它，
    说明多步备份另有增益。两者分开看，才知道该往哪投。
    """

    name = "seq_encoded"
    label_key = "mix"
    needs_history = True
    encoder_kind = "stack"

    def __init__(self, window: int = WINDOW):
        self.window = int(window)
        self.encoder = make_encoder(self.encoder_kind, window=self.window)
        self.base = RandomForestUtility()

    def _encode(self, x, x_hist=None, a_prev=None, hist_len=None) -> np.ndarray:
        if x_hist is None:
            x_hist, a_prev, hist_len = empty_window(x, window=self.window)
        return self.encoder.transform(x_hist, a_prev, hist_len)

    def fit(
        self,
        x,
        strategy,
        utility,
        group_id,
        sample_weight=None,
        x_hist=None,
        a_prev=None,
        hist_len=None,
        **_ignored,
    ):
        if x_hist is None:
            x_hist, a_prev, hist_len = empty_window(x, window=self.window)
        self.encoder.fit(x_hist, a_prev, hist_len)
        h = self.encoder.transform(x_hist, a_prev, hist_len)
        self.base.fit(h, strategy, np.asarray(utility, dtype=float), sample_weight=sample_weight)
        return self

    def recommend(self, x: np.ndarray, strategies=STRATEGIES, x_hist=None, a_prev=None, hist_len=None):
        h = self._encode(x, x_hist, a_prev, hist_len)
        strategies = np.array(list(strategies), dtype=int)
        rows = []
        for row in h:
            cand = np.repeat(row.reshape(1, -1), len(strategies), axis=0)
            rows.append(self.base.predict(cand, strategies))
        return scores_to_recommend(np.vstack(rows), strategies)


class SeqEncodedFittedQ(SeqEncodedScorer):
    """时间序列 RL：状态是序列编码 h_t，目标仍是 r + γ max_a' Q(h_{t+1}, a')。

    下一状态的窗口由 push_window 从当前窗口推出来：末帧换成 x_next，本拍选的
    动作补进 a_prev。所以「这一拍选什么」会改变下一拍的状态表示，这正是它比
    单帧 fitted-Q 多出来的东西。
    """

    name = "seq_encoded_fqi"
    label_key = "r_shaped"
    reward_key = "r_shaped"
    needs_transition = True

    def __init__(self, window: int = WINDOW, n_iter: int = 5, gamma: float = GAMMA):
        super().__init__(window=window)
        self.n_iter = int(n_iter)
        self.gamma = float(gamma)

    def fit(
        self,
        x,
        strategy,
        utility,
        group_id,
        sample_weight=None,
        x_hist=None,
        a_prev=None,
        hist_len=None,
        x_next=None,
        done=None,
        reward=None,
    ):
        if x_hist is None:
            x_hist, a_prev, hist_len = empty_window(x, window=self.window)
        self.encoder.fit(x_hist, a_prev, hist_len)
        h = self.encoder.transform(x_hist, a_prev, hist_len)
        r = np.asarray(utility if reward is None else reward, dtype=float)
        self.base.fit(h, strategy, r.copy(), sample_weight=sample_weight)
        if x_next is None:
            return self
        nxt_x, nxt_a, nxt_len = push_window(x_hist, a_prev, x_next, np.asarray(strategy, dtype=int), hist_len)
        h_next = self.encoder.transform(nxt_x, nxt_a, nxt_len)
        done_mask = np.zeros(len(h), dtype=float) if done is None else np.asarray(done, dtype=float)
        strategies = np.array(list(STRATEGIES), dtype=int)
        for _ in range(max(0, self.n_iter - 1)):
            boot = np.zeros(len(h), dtype=float)
            for i, row in enumerate(h_next):
                if done_mask[i] >= 1.0:
                    continue
                cand = np.repeat(row.reshape(1, -1), len(strategies), axis=0)
                boot[i] = float(np.max(self.base.predict(cand, strategies)))
            self.base.fit(h, strategy, r + self.gamma * boot, sample_weight=sample_weight)
        return self


class SeqStackMix(SeqEncodedScorer):
    name = "seq_stack_mix"
    encoder_kind = "stack"
    label_key = "mix"


class SeqStackFittedQ(SeqEncodedFittedQ):
    name = "seq_stack_fqi"
    encoder_kind = "stack"


class SeqRecurMix(SeqEncodedScorer):
    name = "seq_recur_mix"
    encoder_kind = "recurrent"
    label_key = "mix"


class SeqRecurFittedQ(SeqEncodedFittedQ):
    name = "seq_recur_fqi"
    encoder_kind = "recurrent"
