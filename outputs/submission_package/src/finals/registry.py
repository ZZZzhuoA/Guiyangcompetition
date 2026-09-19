"""模型注册表：有数据后按同一套指标对比，再决定提交哪一个。"""

from __future__ import annotations

from src.finals.learners import (
    BlendScorer,
    ConstantScorer,
    ForestScorer,
    MLPQ,
    PairScorer,
    RidgeQ,
    RuleScorer,
    SoftmaxPolicy,
)
from src.finals.models import StrategyScorer
from src.finals.seq_models import (
    SeqBlendScorer,
    SeqDeltaScorer,
    SeqFittedQ,
    SeqFittedQRaw,
    SeqMixScorer,
    SeqPairScorer,
    SeqRecurFittedQ,
    SeqRecurMix,
    SeqShapedScorer,
    SeqStackFittedQ,
    SeqStackMix,
)

MODEL_CATALOG = {
    "constant_1": {"family": "baseline", "trainable": False, "builder": lambda: ConstantScorer(1)},
    "constant_2": {"family": "baseline", "trainable": False, "builder": lambda: ConstantScorer(2)},
    "constant_3": {"family": "baseline", "trainable": False, "builder": lambda: ConstantScorer(3)},
    "rule": {"family": "baseline", "trainable": False, "builder": RuleScorer},
    "rf_utility": {"family": "q_tree", "trainable": True, "builder": ForestScorer},
    "pairwise": {"family": "rank", "trainable": True, "builder": PairScorer},
    "rf_pair": {"family": "q_blend", "trainable": True, "builder": BlendScorer},
    "ridge_q": {"family": "q_linear", "trainable": True, "builder": RidgeQ},
    "mlp_q": {"family": "q_net", "trainable": True, "builder": MLPQ},
    "oracle_clf": {"family": "clf", "trainable": True, "builder": SoftmaxPolicy},
    "seq_delta": {"family": "seq", "trainable": True, "builder": SeqDeltaScorer},
    "seq_mix": {"family": "seq", "trainable": True, "builder": SeqMixScorer},
    "seq_shaped": {"family": "seq", "trainable": True, "builder": SeqShapedScorer},
    "seq_pair": {"family": "seq", "trainable": True, "builder": SeqPairScorer},
    "seq_blend": {"family": "seq", "trainable": True, "builder": SeqBlendScorer},
    "seq_fqi": {"family": "seq_rl", "trainable": True, "builder": SeqFittedQ},
    "seq_fqi_raw": {"family": "seq_rl", "trainable": True, "builder": SeqFittedQRaw},
    "seq_stack_mix": {"family": "seq_ts", "trainable": True, "builder": SeqStackMix},
    "seq_stack_fqi": {"family": "seq_ts_rl", "trainable": True, "builder": SeqStackFittedQ},
    "seq_recur_mix": {"family": "seq_ts", "trainable": True, "builder": SeqRecurMix},
    "seq_recur_fqi": {"family": "seq_ts_rl", "trainable": True, "builder": SeqRecurFittedQ},
}

DEFAULT_SUBMIT = "rf_pair"
COMPARE_NAMES = (
    "constant_1",
    "constant_2",
    "constant_3",
    "rule",
    "rf_utility",
    "pairwise",
    "rf_pair",
    "ridge_q",
    "mlp_q",
    "oracle_clf",
)
COMPARE_SEQ_NAMES = (
    "seq_delta",
    "seq_mix",
    "seq_shaped",
    "seq_pair",
    "seq_blend",
    "seq_fqi",
    "seq_fqi_raw",
    "seq_stack_mix",
    "seq_stack_fqi",
    "seq_recur_mix",
    "seq_recur_fqi",
)

# 状态表示 × 是否贝尔曼备份，用来分清增益来自序列还是来自 RL
COMPARE_TS_NAMES = (
    "seq_mix",
    "seq_fqi",
    "seq_stack_mix",
    "seq_stack_fqi",
    "seq_recur_mix",
    "seq_recur_fqi",
)


def make_model(name: str):
    if name not in MODEL_CATALOG:
        raise KeyError(f"Unknown model {name}. Choose from {list(MODEL_CATALOG)}")
    model = MODEL_CATALOG[name]["builder"]()
    model.name = name
    return model


def wrap_legacy(scorer) -> object:
    if hasattr(scorer, "recommend"):
        return scorer
    return StrategyScorer()
