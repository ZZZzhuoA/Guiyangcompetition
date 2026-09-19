"""从当前态势快照抽取固定维特征。低空/群目标/覆盖范围在这里自算。

特征分三组，便于以后消融：
  snapshot  当前帧
  temporal  相对上一决策点
  derived   由前两组合成的压力/质量指标
缺上下文时 temporal 填 0，旧 CSV 缺列同样按 0 对齐。
"""

from __future__ import annotations

import numpy as np

from src.finals.context import DecisionContext
from src.finals.schema import ENTITY_TYPES, HIGH_P_THRESHOLD, LOW_ALT_THRESHOLD, SWARM_MIN_SIZE, SWARM_RADIUS, as_float
from src.finals.simulate import angular_span, cluster_ids, geometry_to_point
from src.finals.snapshot import Snapshot

SNAPSHOT_FEATURES = [
    "n_alive",
    "n_tracked",
    "n_intercepting",
    "n_low_alt",
    "n_swarm",
    "low_alt_ratio",
    "swarm_ratio",
    "n_swarm_groups",
    "max_swarm_size",
    "mean_swarm_size",
    "swarm_vel_coherence",
    "min_swarm_vel_coherence",
    "min_swarm_center_tclose",
    "mean_swarm_center_tclose",
    "min_threat",
    "mean_threat",
    "min_range",
    "mean_range",
    "min_tclose",
    "mean_tclose",
    "mean_speed",
    "max_speed",
    "pos_std",
    "az_span",
    "el_span",
    "n_red",
    "type1_count",
    "type2_count",
    "type3_count",
    "type4_count",
    "n_links",
    "n_can_links",
    "n_high_p_links",
    "mean_p",
    "mean_cost",
    "min_links_per_target",
    "mean_links_per_target",
    "n_uncovered",
    "high_threat_uncovered",
    "high_threat_min_tclose",
    "high_threat_mean_p",
    "resource_target_ratio",
    "expected_greedy_p",
    "mean_close_speed",
]
TEMPORAL_FEATURES = [
    "n_appeared",
    "n_disappeared",
    "d_min_range",
    "d_min_tclose",
    "d_n_alive",
    "d_n_can_links",
    "d_n_uncovered",
    "remain_ratio",
    "time_frac",
    "dt_since_last",
    "last_ljcl",
    "last_is_1",
    "last_is_2",
    "last_is_3",
]
DERIVED_FEATURES = [
    "urgency",
    "coverage_gap",
    "quality_edge",
    "scarcity",
    "threat_pressure",
    "swarm_pressure",
]
FEATURE_GROUPS = {
    "snapshot": SNAPSHOT_FEATURES,
    "temporal": TEMPORAL_FEATURES,
    "derived": DERIVED_FEATURES,
}
FEATURE_NAMES = SNAPSHOT_FEATURES + TEMPORAL_FEATURES + DERIVED_FEATURES


def _safe_mean(values: list[float], default: float = 0.0) -> float:
    if not values:
        return default
    return float(np.mean(values))


def _velocity_coherence(vels: list[np.ndarray]) -> float:
    if len(vels) < 2:
        return 1.0
    speeds = [float(np.linalg.norm(v)) for v in vels]
    mean_speed = float(np.mean(speeds))
    if mean_speed < 1e-9:
        return 1.0
    mean_v = np.mean(np.vstack(vels), axis=0)
    return float(np.clip(np.linalg.norm(mean_v) / mean_speed, 0.0, 1.0))


def _swarm_motion_stats(targets, labels: dict[int, int], sizes: dict[int, int], point: np.ndarray) -> dict[str, float]:
    coherences = []
    tcloses = []
    for cid, size in sizes.items():
        if size < SWARM_MIN_SIZE:
            continue
        members = [t for t in targets if labels.get(t.target_id, -1) == cid]
        if not members:
            continue
        coherences.append(_velocity_coherence([t.vel for t in members]))
        center_pos = np.mean(np.vstack([t.pos for t in members]), axis=0)
        center_vel = np.mean(np.vstack([t.vel for t in members]), axis=0)
        tcloses.append(geometry_to_point(center_pos, center_vel, point)["t_close"])
    return {
        "swarm_vel_coherence": _safe_mean(coherences),
        "min_swarm_vel_coherence": min(coherences) if coherences else 0.0,
        "min_swarm_center_tclose": min(tcloses) if tcloses else 0.0,
        "mean_swarm_center_tclose": _safe_mean(tcloses),
    }


def extract_snapshot_dict(snapshot: Snapshot, protected_point: np.ndarray | None = None) -> dict[str, float]:
    point = np.zeros(3) if protected_point is None else np.asarray(protected_point, dtype=float)
    reds = [u for u in snapshot.units if u.team == "红方" or str(u.unit_type).startswith("Type")]
    if not reds:
        reds = snapshot.units
    targets = snapshot.targets
    n_alive = len(targets)
    speeds = [float(np.linalg.norm(t.vel)) for t in targets]
    geos = [geometry_to_point(t.pos, t.vel, point) for t in targets]
    ranges = [g["r"] for g in geos]
    tcloses = [g["t_close"] for g in geos]
    threats = [float(t.threat) for t in targets]
    low_alt = [float(target.pos[2]) < LOW_ALT_THRESHOLD for target in targets]
    labels = cluster_ids({t.target_id: t.pos for t in targets}, SWARM_RADIUS) if targets else {}
    sizes: dict[int, int] = {}
    for cid in labels.values():
        sizes[cid] = sizes.get(cid, 0) + 1
    swarm_sizes = [size for size in sizes.values() if size >= SWARM_MIN_SIZE]
    swarm = [sizes.get(labels.get(t.target_id, -1), 1) >= SWARM_MIN_SIZE for t in targets]
    swarm_motion = _swarm_motion_stats(targets, labels, sizes, point)
    az = np.array([g["az"] for g in geos], dtype=float) if geos else np.zeros(1)
    el = np.array([g["el"] for g in geos], dtype=float) if geos else np.zeros(1)
    pos = np.vstack([t.pos for t in targets]) if targets else np.zeros((1, 3))

    type_counts = {k: 0.0 for k in ENTITY_TYPES}
    for unit in reds:
        name = str(unit.unit_type)
        if name.startswith("Type"):
            try:
                type_counts[int(name.replace("Type", ""))] += 1
            except ValueError:
                pass
    if sum(type_counts.values()) == 0:
        for link in snapshot.links:
            type_counts[int(link.entity_type)] += 1

    can_links = [lk for lk in snapshot.links if lk.can_intercept]
    high_p = [lk for lk in can_links if lk.p >= HIGH_P_THRESHOLD]
    links_per = []
    uncovered = 0
    for target in targets:
        n_can = sum(1 for lk in target.links if lk.can_intercept)
        links_per.append(n_can)
        if n_can == 0:
            uncovered += 1
    high_threat = sorted(targets, key=lambda t: t.threat)[: min(3, len(targets))]
    ht_uncovered = 0
    ht_tclose = []
    ht_p = []
    for target in high_threat:
        cans = [lk for lk in target.links if lk.can_intercept]
        if not cans:
            ht_uncovered += 1
        ht_tclose.append(geometry_to_point(target.pos, target.vel, point)["t_close"])
        ht_p.append(_safe_mean([lk.p for lk in cans], 0.0))

    greedy = 0.0
    used_e = set()
    for target in sorted(targets, key=lambda t: t.threat):
        options = [lk for lk in target.links if lk.can_intercept and lk.entity_id not in used_e]
        if not options:
            continue
        best = max(options, key=lambda lk: lk.p)
        greedy += best.p
        used_e.add(best.entity_id)

    n_red = max(1, len(reds)) if reds else max(1, len({lk.entity_id for lk in snapshot.links}))
    return {
        "n_alive": float(n_alive),
        "n_tracked": float(sum(1 for t in targets if t.tracked)),
        "n_intercepting": float(sum(1 for t in targets if t.intercepting)),
        "n_low_alt": float(sum(low_alt)),
        "n_swarm": float(sum(swarm)),
        "low_alt_ratio": float(sum(low_alt) / n_alive) if n_alive else 0.0,
        "swarm_ratio": float(sum(swarm) / n_alive) if n_alive else 0.0,
        "n_swarm_groups": float(len(swarm_sizes)),
        "max_swarm_size": float(max(swarm_sizes) if swarm_sizes else 0.0),
        "mean_swarm_size": _safe_mean([float(v) for v in swarm_sizes]),
        **swarm_motion,
        "min_threat": min(threats) if threats else 0.0,
        "mean_threat": _safe_mean(threats),
        "min_range": min(ranges) if ranges else 0.0,
        "mean_range": _safe_mean(ranges),
        "min_tclose": min(tcloses) if tcloses else 0.0,
        "mean_tclose": _safe_mean(tcloses),
        "mean_speed": _safe_mean(speeds),
        "max_speed": max(speeds) if speeds else 0.0,
        "pos_std": float(np.mean(np.std(pos, axis=0))),
        "az_span": angular_span(az),
        "el_span": float(np.degrees(np.max(el) - np.min(el))) if len(el) else 0.0,
        "n_red": float(len(reds) if reds else len({lk.entity_id for lk in snapshot.links})),
        "type1_count": type_counts[1],
        "type2_count": type_counts[2],
        "type3_count": type_counts[3],
        "type4_count": type_counts[4],
        "n_links": float(len(snapshot.links)),
        "n_can_links": float(len(can_links)),
        "n_high_p_links": float(len(high_p)),
        "mean_p": _safe_mean([lk.p for lk in can_links], 0.0),
        "mean_cost": _safe_mean([lk.cost for lk in can_links], 0.0),
        "min_links_per_target": min(links_per) if links_per else 0.0,
        "mean_links_per_target": _safe_mean([float(v) for v in links_per]),
        "n_uncovered": float(uncovered),
        "high_threat_uncovered": float(ht_uncovered),
        "high_threat_min_tclose": min(ht_tclose) if ht_tclose else 0.0,
        "high_threat_mean_p": _safe_mean(ht_p),
        "resource_target_ratio": float(n_red / max(1, n_alive)),
        "expected_greedy_p": greedy,
        "mean_close_speed": _safe_mean([-g["rv"] for g in geos], 0.0),
    }


def extract_temporal_dict(snapshot: Snapshot, snap: dict[str, float], context: DecisionContext | None) -> dict[str, float]:
    zeros = {name: 0.0 for name in TEMPORAL_FEATURES}
    if context is None:
        return zeros
    ids = {int(t.target_id) for t in snapshot.targets}
    appeared = ids - context.prev_ids if context.prev_ids else ids
    disappeared = context.prev_ids - ids
    seen = max(1, len(context.seen_ids | ids))
    last = int(context.last_ljcl)
    dt = 0.0 if context.last_time is None else max(0.0, float(snapshot.time) - float(context.last_time))
    def _delta(cur: float, prev: float | None) -> float:
        return 0.0 if prev is None else float(cur) - float(prev)
    return {
        "n_appeared": float(len(appeared)),
        "n_disappeared": float(len(disappeared)),
        "d_min_range": _delta(snap["min_range"], context.prev_min_range),
        "d_min_tclose": _delta(snap["min_tclose"], context.prev_min_tclose),
        "d_n_alive": _delta(snap["n_alive"], context.prev_n_alive),
        "d_n_can_links": _delta(snap["n_can_links"], context.prev_n_can_links),
        "d_n_uncovered": _delta(snap["n_uncovered"], context.prev_n_uncovered),
        "remain_ratio": float(snap["n_alive"] / seen),
        "time_frac": float(np.clip(snapshot.time / max(1.0, context.horizon), 0.0, 1.5)),
        "dt_since_last": dt,
        "last_ljcl": float(last),
        "last_is_1": 1.0 if last == 1 else 0.0,
        "last_is_2": 1.0 if last == 2 else 0.0,
        "last_is_3": 1.0 if last == 3 else 0.0,
    }


def extract_derived_dict(snap: dict[str, float], temporal: dict[str, float]) -> dict[str, float]:
    urgency = 1.0 / (1.0 + snap["min_tclose"] / 8.0)
    coverage_gap = snap["n_uncovered"] / max(1.0, snap["n_alive"])
    quality_edge = snap["n_high_p_links"] * snap["mean_p"]
    scarcity = 1.0 / max(0.05, snap["resource_target_ratio"])
    threat_pressure = snap["high_threat_uncovered"] * urgency
    swarm_pressure = snap["n_swarm_groups"] * snap["swarm_ratio"] * max(float(snap.get("swarm_vel_coherence", 0.0)), 0.0)
    _ = temporal
    return {
        "urgency": float(urgency),
        "coverage_gap": float(coverage_gap),
        "quality_edge": float(quality_edge),
        "scarcity": float(scarcity),
        "threat_pressure": float(threat_pressure),
        "swarm_pressure": float(swarm_pressure),
    }


def dict_to_vector(values: dict[str, float], names: list[str] | None = None) -> np.ndarray:
    names = FEATURE_NAMES if names is None else names
    return np.array([as_float(values.get(name), 0.0) for name in names], dtype=float)


def extract_feature_dict(
    snapshot: Snapshot,
    protected_point: np.ndarray | None = None,
    context: DecisionContext | None = None,
) -> dict[str, float]:
    snap = extract_snapshot_dict(snapshot, protected_point=protected_point)
    temporal = extract_temporal_dict(snapshot, snap, context)
    derived = extract_derived_dict(snap, temporal)
    return {**snap, **temporal, **derived}


def extract_features(
    snapshot: Snapshot,
    protected_point: np.ndarray | None = None,
    context: DecisionContext | None = None,
) -> np.ndarray:
    return dict_to_vector(extract_feature_dict(snapshot, protected_point=protected_point, context=context))


def rule_from_features(features: np.ndarray) -> int:
    names = {name: i for i, name in enumerate(FEATURE_NAMES)}
    min_tclose = features[names["min_tclose"]]
    ht_uncovered = features[names["high_threat_uncovered"]]
    high_p = features[names["n_high_p_links"]]
    mean_p = features[names["mean_p"]]
    cost = features[names["mean_cost"]]
    n_alive = features[names["n_alive"]]
    urgency = features[names["urgency"]] if "urgency" in names else 0.0
    if min_tclose < 6.0 or ht_uncovered >= 1 or urgency >= 0.55:
        return 1
    if high_p >= 3 and mean_p >= 0.8:
        return 2
    if cost > 0.55 and n_alive >= 6:
        return 3
    return 3


def utility_from_outcome(intercepted: int, leaked: int, cost: float, n_blue: int, mean_intercept_time: float = 0.0) -> dict:
    n_blue = max(1, int(n_blue))
    intercept_rate = intercepted / n_blue
    leak_rate = leaked / n_blue
    cost_eff = intercept_rate / (0.35 + cost)
    close_pen = mean_intercept_time / 30.0
    utility = 1.15 * intercept_rate + 0.25 * cost_eff - 0.65 * leak_rate - 0.04 * close_pen
    return {
        "intercept_rate": intercept_rate,
        "leak_rate": leak_rate,
        "cost": float(cost),
        "cost_eff": cost_eff,
        "utility": float(utility),
    }
