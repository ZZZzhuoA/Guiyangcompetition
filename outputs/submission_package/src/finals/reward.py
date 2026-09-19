"""奖励定义。主奖励对齐赛方指标：整局拦截率 J = N_拦 / N_蓝累计。

设计约束：
  1. 主标量只放拦截增量，不再把漏防率、费效、拦截时间揉进同一个数
  2. 紧迫程度用势函数 shaping，γΦ(x') - Φ(x) 不改变最优策略
  3. 高威胁漏防只作小权重修正，赛方并未按威胁计分
  4. 成本暂不进奖励：集成接口未必给 D_Costgy，训练里变、评估里恒定会学歪
shaping 的 γ 必须与 fitted-Q 的 γ 一致，否则策略不变性不成立。
"""

from __future__ import annotations

GAMMA = 0.85
MIX_LAMBDA = 0.25

# Ui_Zwx 越小威胁越大，取前若干名当高威胁
HIGH_THREAT_MAX_RANK = 3
LEAK_HT_PENALTY = 0.05

POTENTIAL_W_THREAT = 0.04
POTENTIAL_W_URGENCY = 0.02


def blue_seen(world) -> int:
    """本局到目前为止出现过的蓝方数，作为拦截率分母。

    只数已被观测过的实体：还在 pending 里未出现的不算，避免用未来信息。
    出现时刻与策略无关，因此三种策略拿到的分母一致、可比。
    """
    return sum(1 for blue in world.blues if blue.ever_observed)


def high_threat_leaked(world) -> int:
    return sum(1 for blue in world.blues if blue.leaked and int(blue.threat) <= HIGH_THREAT_MAX_RANK)


def potential(features: dict) -> float:
    """Φ(x)：态势越危险越负。逼近要点、高威胁没人管都会压低势能。"""
    threat_pressure = float(features.get("threat_pressure", 0.0))
    urgency = float(features.get("urgency", 0.0))
    return -(POTENTIAL_W_THREAT * threat_pressure + POTENTIAL_W_URGENCY * urgency)


def episode_return(intercepted: int, n_faced: int) -> float:
    return intercepted / max(1, int(n_faced))


def segment_reward(
    intercepted_delta: int,
    n_faced: int,
    phi: float,
    phi_next: float,
    ht_leak_delta: int = 0,
    gamma: float = GAMMA,
    done: bool = False,
) -> dict:
    """本段奖励：拦截增量 + 势函数差 - 高威胁漏防修正。"""
    r_delta = episode_return(intercepted_delta, n_faced)
    shaping = (0.0 if done else gamma * phi_next) - phi
    leak_pen = LEAK_HT_PENALTY * int(ht_leak_delta)
    return {
        "r_delta": float(r_delta),
        "phi": float(phi),
        "phi_next": float(phi_next),
        "shaping": float(shaping),
        "ht_leak_delta": float(ht_leak_delta),
        "r_shaped": float(r_delta + shaping - leak_pen),
    }


def mix_label(r_delta: float, r_term: float, mix_lambda: float = MIX_LAMBDA) -> float:
    """本段收益 + 一点终局拦截率，防止只顾眼前。"""
    return float(r_delta + mix_lambda * r_term)
