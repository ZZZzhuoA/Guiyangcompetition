"""序列表示：把最近 K 次触发编码成一个向量，供贯序模型当状态用。

和 seq_fqi 的区别只在状态怎么表示，不在要不要考虑下一拍：
  seq_fqi      状态 = 当前 64 维，历史只通过 last_ljcl / 新增消失等几维进来
  这里         状态 = (x_{m-K+1..m}, a_{m-K+1..m-1}) 编码成一个向量

窗口约定（m 为当前决策点）：
  x_hist[m]  形状 (K, F)，最后一行就是当前帧，靠前是更早的帧，前面不足用 0 填
  a_prev[m]  形状 (K-1,)，是当前帧之前那几拍**已经选过**的动作，0 表示没有
  hist_len[m] 窗口里有多少行是真的，冷启动时为 1
当前帧要选的动作不进编码器，仍走 RandomForestUtility 既有的 append_strategy 机制。

下一状态可以直接推出来，不必另存：
  x_hist' = roll(x_hist) 末行换成 x_next
  a_prev' = roll(a_prev) 末位换成本拍选的 a
"""

from __future__ import annotations

import numpy as np

from src.finals.schema import STRATEGIES

WINDOW = 3
N_ACTIONS = len(STRATEGIES)


def build_window(past_x: list, past_a: list, feat: np.ndarray, window: int = WINDOW):
    """单个决策点的窗口：末行是当前帧，靠前是更早的触发，不足前面补 0。

    训练（seq_set）和推理（DecisionContext）共用这一份，避免两边窗口含义漂移。
    past_x / past_a 是**当前帧之前**已访问过的决策点，最新在末尾。
    """
    feat = np.asarray(feat, dtype=float)
    keep = window - 1
    past_x = list(past_x)[-keep:] if keep > 0 else []
    past_a = list(past_a)[-keep:] if keep > 0 else []
    x_hist = np.zeros((window, len(feat)), dtype=float)
    for j, row in enumerate(past_x):
        x_hist[keep - len(past_x) + j] = np.asarray(row, dtype=float)
    x_hist[-1] = feat
    a_prev = np.zeros(max(0, keep), dtype=int)
    for j, action in enumerate(past_a):
        a_prev[len(a_prev) - len(past_a) + j] = int(action)
    return x_hist, a_prev, min(len(past_x) + 1, window)


def empty_window(x: np.ndarray, window: int = WINDOW) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """只有当前帧的窗口。冷启动（第一次触发）和不带历史的老调用方都走这条。"""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    n, n_feat = x.shape
    x_hist = np.zeros((n, window, n_feat), dtype=float)
    x_hist[:, -1, :] = x
    a_prev = np.zeros((n, max(0, window - 1)), dtype=int)
    hist_len = np.ones(n, dtype=int)
    return x_hist, a_prev, hist_len


def push_window(
    x_hist: np.ndarray,
    a_prev: np.ndarray,
    x_next: np.ndarray,
    action: int | np.ndarray,
    hist_len: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """窗口往前滚一拍：末行放 x_next，已选动作补进 a_prev。"""
    x_hist = np.asarray(x_hist, dtype=float)
    a_prev = np.asarray(a_prev, dtype=int)
    x_next = np.atleast_2d(np.asarray(x_next, dtype=float))
    nxt_x = np.concatenate([x_hist[:, 1:, :], x_next[:, None, :]], axis=1)
    act = np.full(len(x_hist), int(action), dtype=int) if np.isscalar(action) else np.asarray(action, dtype=int)
    if a_prev.shape[1] == 0:
        nxt_a = a_prev
    else:
        nxt_a = np.concatenate([a_prev[:, 1:], act.reshape(-1, 1)], axis=1)
    window = x_hist.shape[1]
    if hist_len is None:
        nxt_len = np.full(len(x_hist), window, dtype=int)
    else:
        nxt_len = np.minimum(np.asarray(hist_len, dtype=int) + 1, window)
    return nxt_x, nxt_a, nxt_len


def _onehot(actions: np.ndarray) -> np.ndarray:
    """0 表示没有动作，编成全零，不占某一类。"""
    actions = np.asarray(actions, dtype=int)
    flat = actions.reshape(-1)
    out = np.zeros((len(flat), N_ACTIONS), dtype=float)
    valid = (flat >= 1) & (flat <= N_ACTIONS)
    out[np.arange(len(flat))[valid], flat[valid] - 1] = 1.0
    return out.reshape(*actions.shape, N_ACTIONS)


class StackEncoder:
    """K 帧原样拼接 + 过去动作 one-hot。

    最直白的序列表示，用来先回答「历史到底有没有用」。输出维度是 K*F + (K-1)*3，
    K=3、F=64 时 198 维，样本少的时候本身就容易过拟合，这一点靠消融看，不靠猜。
    """

    kind = "stack"

    def __init__(self, window: int = WINDOW):
        self.window = int(window)

    def fit(self, x_hist: np.ndarray, a_prev: np.ndarray, hist_len: np.ndarray | None = None) -> "StackEncoder":
        return self

    def transform(self, x_hist: np.ndarray, a_prev: np.ndarray, hist_len: np.ndarray | None = None) -> np.ndarray:
        x_hist = np.asarray(x_hist, dtype=float)
        flat = x_hist.reshape(len(x_hist), -1)
        if np.asarray(a_prev).size == 0:
            return flat
        acts = _onehot(a_prev).reshape(len(x_hist), -1)
        return np.hstack([flat, acts])


class RecurrentEncoder:
    """GRU 单元的递归编码，递归权重固定随机，只训练下游读出头。

    为什么不做 BPTT：赛方电脑上只有 numpy（本项目所有模型都是手写的），一局决策点
    才十来个，合成数据上目前连 r_delta 都恒为 0，反向传播训出来的权重没法验证，
    而且要保证单次推理 <50ms。固定随机递归 + 可训练读出头（储备池 / echo-state 那一类）
    在这个数据量下更稳，也仍然是真正的递归表示：隐状态携带整段窗口的信息。
    真实数据够多之后再把递归权重换成 BPTT 训练的，接口不用动。

    读出向量刻意拼上当前帧：固定随机投影可能损失信息，留一条直连保证下游至少
    不比单帧模型差。
    """

    kind = "recurrent"

    def __init__(self, window: int = WINDOW, hidden: int = 24, seed: int = 0, spectral_radius: float = 0.9):
        self.window = int(window)
        self.hidden = int(hidden)
        self.seed = int(seed)
        self.spectral_radius = float(spectral_radius)

    def _init_weights(self, n_in: int) -> None:
        rng = np.random.default_rng(self.seed)
        h = self.hidden
        self.W = rng.normal(0.0, 1.0 / np.sqrt(n_in), size=(3, n_in, h))
        u = rng.normal(0.0, 1.0, size=(3, h, h))
        for k in range(3):
            radius = np.max(np.abs(np.linalg.eigvals(u[k])))
            u[k] *= self.spectral_radius / max(radius, 1e-9)
        self.U = u
        self.b = np.zeros((3, h))
        # 让更新门初始偏向记住历史，避免隐状态被每帧冲掉
        self.b[0] -= 1.0

    def fit(self, x_hist: np.ndarray, a_prev: np.ndarray, hist_len: np.ndarray | None = None) -> "RecurrentEncoder":
        x_hist = np.asarray(x_hist, dtype=float)
        frames = x_hist.reshape(-1, x_hist.shape[-1])
        lo = frames.min(axis=0)
        span = frames.max(axis=0) - lo
        self.lo_ = lo
        self.scale_ = np.where(span < 1e-12, 1.0, span)
        self._init_weights(x_hist.shape[-1] + N_ACTIONS)
        return self

    def _norm(self, frames: np.ndarray) -> np.ndarray:
        return (frames - self.lo_) / self.scale_

    def transform(self, x_hist: np.ndarray, a_prev: np.ndarray, hist_len: np.ndarray | None = None) -> np.ndarray:
        x_hist = np.asarray(x_hist, dtype=float)
        n, window, _ = x_hist.shape
        a_prev = np.asarray(a_prev, dtype=int)
        if hist_len is None:
            hist_len = np.full(n, window, dtype=int)
        else:
            hist_len = np.asarray(hist_len, dtype=int)
        acts = np.zeros((n, window), dtype=int)
        if a_prev.size:
            acts[:, : a_prev.shape[1]] = a_prev
        act_oh = _onehot(acts)
        h = np.zeros((n, self.hidden), dtype=float)
        for t in range(window):
            # 前面的填充行不进递归，冷启动时只有末行是真的
            active = (hist_len > (window - 1 - t)).astype(float).reshape(-1, 1)
            u = np.hstack([self._norm(x_hist[:, t, :]), act_oh[:, t, :]])
            z = _sigmoid(u @ self.W[0] + h @ self.U[0] + self.b[0])
            r = _sigmoid(u @ self.W[1] + h @ self.U[1] + self.b[1])
            cand = np.tanh(u @ self.W[2] + (r * h) @ self.U[2] + self.b[2])
            h_new = (1.0 - z) * h + z * cand
            h = active * h_new + (1.0 - active) * h
        return np.hstack([h, self._norm(x_hist[:, -1, :])])


def _sigmoid(v: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(v, -30.0, 30.0)))


ENCODERS = {"stack": StackEncoder, "recurrent": RecurrentEncoder}


def make_encoder(kind: str, window: int = WINDOW) -> StackEncoder | RecurrentEncoder:
    if kind not in ENCODERS:
        raise KeyError(f"Unknown encoder {kind}. Choose from {list(ENCODERS)}")
    return ENCODERS[kind](window=window)
