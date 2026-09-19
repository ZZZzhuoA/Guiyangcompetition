"""决赛字段约定：拦截主体是红方实体，实体类型只有 4 种，没有火力层。"""

from __future__ import annotations

import math

STRATEGIES = (1, 2, 3)
STRATEGY_NAMES = {
    1: "闭合时间最短",
    2: "拦截性能最佳",
    3: "综合效益最优",
}


def parse_strategy_label(raw) -> int | None:
    """把结果表/Stu_SJZS 里的策略字段收成 1/2/3。中文类型名和数字都认。"""
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    for strategy, name in STRATEGY_NAMES.items():
        if text == name or name in text:
            return int(strategy)
    compact = text.replace("_", "").replace("策略", "").strip()
    try:
        value = int(float(compact))
    except ValueError:
        return None
    if value in STRATEGIES:
        return value
    return None


_BLANK_TOKENS = {
    "", "none", "null", "nan", "nat", "na", "n/a", "#n/a", "-", "--",
    "#value!", "#ref!", "#div/0!", "#name?",
}
_TRUE_TOKENS = {"true", "t", "yes", "y", "1", "1.0", "是", "真", "对"}
_FALSE_TOKENS = {"false", "f", "no", "n", "0", "0.0", "否", "假", "错"}


def is_blank(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in _BLANK_TOKENS
    return False


def as_float(value, default=None):
    """空单元格 / None / 非数字 → default，避免 float(None) 把建集打崩。"""
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else default
    text = str(value).strip()
    if text.lower() in _BLANK_TOKENS:
        return default
    try:
        number = float(text)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def as_int(value, default=None):
    number = as_float(value, default=None)
    if number is None:
        return default
    return int(number)


def as_bool(value, default=None):
    """True/False、1/0、TRUE、是/否 都认。空单元格 → default。"""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
        return default
    text = str(value).strip().lower()
    if text in _BLANK_TOKENS:
        return default
    if text in _TRUE_TOKENS:
        return True
    if text in _FALSE_TOKENS:
        return False
    return default


ENTITY_TYPES = (1, 2, 3, 4)
ENTITY_TYPE_NAMES = {
    1: "Type1",
    2: "Type2",
    3: "Type3",
    4: "Type4",
}
# 四个布尔字段按实体类型 1–4 对齐。B_GNJGKsslj = 实体4 可实施拦截（赛方改过）。
# 「可实施拦截」= 系统算出该类实体此刻能否拦这个目标，和概率是两回事。
INTERCEPT_FLAGS = ("B_DDKsslj", "B_GPKsslj", "B_HPMKsslj", "B_GNJGKsslj")
# 概率是连续值，样本里见过 0 / 0.7 / 0.8 / 0.9 / 1.0，不要当成 {0.7, 0.9} 枚举。
HIGH_P_THRESHOLD = 0.8
PHYSICS_DT = 0.1   # 系统内部更新
RECORD_DT = 0.5    # CSV 记录步长；FZTime 从 1s 起、按 0.5s 递增
TIME_ALIGN_EPS = 1e-3
HISTORY_MIN_DT = 0.4  # 实装可能 0.1s 刷帧；短于一个记录步不写入决策历史

TEAM_RED = "红方"
TEAM_BLUE = "蓝方"

# 合成仿真用的实体能力。真实赛方数据到来后只用于缺省回退，特征以可拦截表为准。
ENTITY_SPECS = {
    1: {"name": "Type1", "range": 16.0, "track_range": 20.0, "p": 0.9, "cost": 1.00},
    2: {"name": "Type2", "range": 8.0, "track_range": 12.0, "p": 0.7, "cost": 0.25},
    3: {"name": "Type3", "range": 10.0, "track_range": 14.0, "p": 0.7, "cost": 0.35},
    4: {"name": "Type4", "range": 13.0, "track_range": 17.0, "p": 0.9, "cost": 0.70},
}

LOW_ALT_THRESHOLD = 0.6
SWARM_RADIUS = 1.2
SWARM_MIN_SIZE = 3
SWARM_MIN_NEIGHBORS = 2
LEAK_RADIUS = 0.8
PROTECTED_POINT = (0.0, 0.0, 0.0)
TICKS_PER_FILE = 200

RHDL_COLUMNS = [
    ("TargetUnitID", "int"),
    ("TargetUnitName", "string"),
    ("B_Sgzmb", "bool"),
    ("B_l_jbz", "bool"),
    ("B_Sdk", "bool"),
    ("B_QMBflag", "bool"),
    ("Uc_XXYLY", "int"),
    ("Uc_XXYHandle", "int"),
    ("D_Bsm", "double"),
    ("D_Esm", "double"),
    ("Ui_Zwx", "int"),
    ("D_XDPm", "double"),
    ("D_XDTm", "double"),
    ("D_XDRm", "double"),
    ("D_R", "double"),
    ("D_E", "double"),
    ("D_B", "double"),
    ("D_RV", "double"),
    ("D_EV", "double"),
    ("D_BV", "double"),
    ("D_X", "double"),
    ("D_Y", "double"),
    ("D_Z", "double"),
    ("D_XV", "double"),
    ("D_YV", "double"),
    ("D_ZV", "double"),
    ("FZTime", "double"),
    ("DetectTime", "double"),
]

HEALTH_COLUMNS = [
    ("UnitID", "int"),
    ("UnitType", "string"),
    ("UnitName", "string"),
    ("UnitTeam", "string"),
    ("UnitPos_x", "double"),
    ("UnitPos_y", "double"),
    ("UnitPos_z", "double"),
    ("UnitPos_lon", "double"),
    ("UnitPos_lat", "double"),
    ("UnitPos_alt", "double"),
    ("HealthPoint", "double"),
    ("Time", "double"),
]

LJ_COLUMNS = [
    ("UI_Targethandle", "int"),
    ("UI_Hlhandle", "int"),
    ("UC_HlType", "int"),
    ("UI_FSChandle", "int"),
    ("Str_HLMC", "string"),
    ("B_DDKsslj", "bool"),
    ("B_GPKsslj", "bool"),
    ("B_HPMKsslj", "bool"),
    ("B_GNJGKsslj", "bool"),
    ("B_Kgz", "bool"),
    ("B_Ygz", "bool"),
    ("B_TD", "bool"),
    ("G_HLYS", "bool"),
    ("D_Pm", "double"),
    ("D_Rm", "double"),
    ("D_Tmzi", "double"),
    ("D_Costgy", "double"),
    ("G_LJGL", "double"),
    ("Time", "double"),
]

SJZS_COLUMNS = [
    ("S_LJCL", "string"),
]

INTERFACE_TARGET_FIELDS = (
    "TargetUnitID",
    "TargetUnitType",
    "B_l_jbz",
    "Ui_Zwx",
    "D_X",
    "D_Y",
    "D_Z",
    "D_XV",
    "D_YV",
    "D_ZV",
    "Klj_list",
)

INTERFACE_LINK_FIELDS = (
    "UI_FSChandle",
    "Str_HLMC",
    "B_DDKsslj",
    "B_GPKsslj",
    "B_HPMKsslj",
    "B_GNJGKsslj",
    "D_LJGL",
)

INTERFACE_UNIT_FIELDS = (
    "UnitID",
    "UnitType",
    "UnitPos_x",
    "UnitPos_y",
    "UnitPos_z",
    "UnitPos_lon",
    "UnitPos_lat",
    "UnitPos_alt",
)


def flag_for_entity_type(entity_type: int) -> str:
    idx = max(1, min(4, int(entity_type))) - 1
    return INTERCEPT_FLAGS[idx]


def entity_type_from_flags(flags: dict) -> int | None:
    """一行只应对一种类型：哪个 B_* 为真就是哪类实体。"""
    for index, name in enumerate(INTERCEPT_FLAGS, start=1):
        if flags.get(name):
            return index
    return None


def seconds_to_ticks(seconds: float, dt: float) -> int:
    """Δ 是秒。样本 dt=0.5 时 15s = 30 个记录步，不是 15 步。"""
    step = float(dt) if float(dt) > 1e-9 else RECORD_DT
    return max(1, int(round(float(seconds) / step)))


def infer_dt(times: list[float], default: float = RECORD_DT) -> float:
    vals = sorted({t for t in (as_float(x) for x in times) if t is not None})
    diffs = [b - a for a, b in zip(vals, vals[1:]) if b - a > 1e-9]
    if not diffs:
        return float(default)
    diffs.sort()
    return float(diffs[len(diffs) // 2])


def entity_type_from_name(name: str) -> int:
    for key, value in ENTITY_TYPE_NAMES.items():
        if value == name:
            return key
    if isinstance(name, str) and name.startswith("Type"):
        try:
            return int(name.replace("Type", ""))
        except ValueError:
            pass
    return 1
