from __future__ import annotations

import numpy as np


CORE_ENGINEERED_FEATURES = [
    "total_targets",
    "total_threat",
    "max_threat",
    "std_threat",
    "threat_concentration",
    "angle_span",
    "total_capability",
    "total_perf1",
    "total_perf2",
    "active_equipment_count",
    "capability_balance",
    "max_unit_capability",
    "pressure_total",
    "target_pressure",
    "capability_per_target",
    "high_threat_to_best_unit",
    "type1_capability",
    "type2_capability",
    "type3_capability",
    "type_capability_gap",
    "capability_entropy",
    "perf1_to_perf2_ratio",
    "pressure_on_type1",
    "pressure_on_type2",
    "pressure_on_type3",
]


COMPACT_ENGINEERED_FEATURES = [
    "total_targets",
    "total_threat",
    "max_threat",
    "threat_concentration",
    "angle_span",
    "total_capability",
    "active_equipment_count",
    "capability_balance",
    "max_unit_capability",
    "pressure_total",
    "target_pressure",
    "capability_per_target",
    "type_capability_gap",
    "capability_entropy",
    "perf1_to_perf2_ratio",
]


RESOURCE_ENGINEERED_FEATURES = [
    "total_capability",
    "total_perf1",
    "total_perf2",
    "active_equipment_count",
    "capability_balance",
    "max_unit_capability",
    "type1_capability",
    "type2_capability",
    "type3_capability",
    "type_capability_gap",
    "capability_entropy",
    "perf1_to_perf2_ratio",
]


THREAT_ENGINEERED_FEATURES = [
    "total_targets",
    "active_group_count",
    "total_threat",
    "max_threat",
    "mean_threat",
    "std_threat",
    "threat_concentration",
    "weighted_mean_angle",
    "angle_span",
]


MATCH_ENGINEERED_FEATURES = [
    "pressure_total",
    "target_pressure",
    "capability_per_target",
    "high_threat_to_best_unit",
    "type1_capability_to_threat",
    "type2_capability_to_threat",
    "type3_capability_to_threat",
    "pressure_on_type1",
    "pressure_on_type2",
    "pressure_on_type3",
]


def select_features(x: np.ndarray, feature_names: list[str], selected_names: list[str]) -> tuple[np.ndarray, list[str]]:
    name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
    missing = [name for name in selected_names if name not in name_to_idx]
    if missing:
        raise KeyError(f"Missing selected features: {missing}")
    raw_names = [name for name in feature_names if name.startswith("raw_")]
    names = raw_names + selected_names
    indices = [name_to_idx[name] for name in names]
    return x[:, indices], names
