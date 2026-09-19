from __future__ import annotations

import numpy as np


def build_engineered_features(x: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Build domain features from 18 resource features and 24 threat features."""
    x = x.astype(float)
    resource = x[:, :18].reshape(-1, 6, 3)
    threat = x[:, 18:42].reshape(-1, 6, 4)

    perf1 = resource[:, :, 0]
    perf2 = resource[:, :, 1]
    active = resource[:, :, 2]
    distance = threat[:, :, 0]
    angle = threat[:, :, 1]
    speed = threat[:, :, 2]
    count = threat[:, :, 3]

    eps = 1e-6
    effective_capability = (perf1 + perf2) * active
    effective_perf1 = perf1 * active
    effective_perf2 = perf2 * active
    inactive = 1.0 - active

    threat_intensity = count * speed / (distance + eps)
    angle_weighted_threat = threat_intensity * angle
    total_targets = np.sum(count, axis=1)
    active_group_count = np.sum(count > 0, axis=1)
    total_threat = np.sum(threat_intensity, axis=1)
    max_threat = np.max(threat_intensity, axis=1)
    mean_threat = np.mean(threat_intensity, axis=1)
    std_threat = np.std(threat_intensity, axis=1)
    threat_concentration = max_threat / (total_threat + eps)
    weighted_mean_angle = np.sum(angle_weighted_threat, axis=1) / (total_threat + eps)
    angle_span = np.max(angle, axis=1) - np.min(angle, axis=1)

    total_capability = np.sum(effective_capability, axis=1)
    total_perf1 = np.sum(effective_perf1, axis=1)
    total_perf2 = np.sum(effective_perf2, axis=1)
    active_equipment_count = np.sum(active > 0, axis=1)
    inactive_equipment_count = np.sum(inactive > 0, axis=1)
    capability_balance = np.std(effective_capability, axis=1)
    max_unit_capability = np.max(effective_capability, axis=1)
    min_active_capability = np.array(
        [
            np.min(row[row > 0]) if np.any(row > 0) else 0.0
            for row in effective_capability
        ],
        dtype=float,
    )

    pressure_total = total_threat / (total_capability + eps)
    target_pressure = total_targets / (active_equipment_count + eps)
    capability_per_target = total_capability / (total_targets + eps)
    high_threat_to_best_unit = max_threat / (max_unit_capability + eps)
    resilience_ratio = active_equipment_count / 6.0

    # Equipment-type aggregates: every two assets form one equipment type.
    type_capability = effective_capability.reshape(-1, 3, 2).sum(axis=2)
    type_active = active.reshape(-1, 3, 2).sum(axis=2)
    type_perf1 = effective_perf1.reshape(-1, 3, 2).sum(axis=2)
    type_perf2 = effective_perf2.reshape(-1, 3, 2).sum(axis=2)
    type_capability_gap = np.max(type_capability, axis=1) - np.min(type_capability, axis=1)
    type_capability_std = np.std(type_capability, axis=1)
    type_capability_min = np.min(type_capability, axis=1)
    active_pattern_all = (active_equipment_count == 6).astype(float)
    active_pattern_primary = ((active[:, 0] > 0) & (active[:, 2] > 0) & (active[:, 4] > 0)).astype(float)
    paired_redundancy = np.sum(active.reshape(-1, 3, 2), axis=2)
    single_asset_type_count = np.sum(paired_redundancy == 1, axis=1)
    double_asset_type_count = np.sum(paired_redundancy == 2, axis=1)
    no_asset_type_count = np.sum(paired_redundancy == 0, axis=1)
    capability_entropy_base = type_capability / (np.sum(type_capability, axis=1, keepdims=True) + eps)
    capability_entropy = -np.sum(
        np.where(capability_entropy_base > 0, capability_entropy_base * np.log(capability_entropy_base + eps), 0.0),
        axis=1,
    )
    perf_ratio = total_perf1 / (total_perf2 + eps)
    type1_to_threat = type_capability[:, 0] / (total_threat + eps)
    type2_to_threat = type_capability[:, 1] / (total_threat + eps)
    type3_to_threat = type_capability[:, 2] / (total_threat + eps)
    pressure_by_type = total_threat.reshape(-1, 1) / (type_capability + eps)
    target_by_type_active = total_targets.reshape(-1, 1) / (type_active + eps)

    engineered_parts = [
        total_targets.reshape(-1, 1),
        active_group_count.reshape(-1, 1),
        total_threat.reshape(-1, 1),
        max_threat.reshape(-1, 1),
        mean_threat.reshape(-1, 1),
        std_threat.reshape(-1, 1),
        threat_concentration.reshape(-1, 1),
        weighted_mean_angle.reshape(-1, 1),
        angle_span.reshape(-1, 1),
        total_capability.reshape(-1, 1),
        total_perf1.reshape(-1, 1),
        total_perf2.reshape(-1, 1),
        active_equipment_count.reshape(-1, 1),
        inactive_equipment_count.reshape(-1, 1),
        capability_balance.reshape(-1, 1),
        max_unit_capability.reshape(-1, 1),
        min_active_capability.reshape(-1, 1),
        pressure_total.reshape(-1, 1),
        target_pressure.reshape(-1, 1),
        capability_per_target.reshape(-1, 1),
        high_threat_to_best_unit.reshape(-1, 1),
        resilience_ratio.reshape(-1, 1),
        type_capability,
        type_active,
        type_perf1,
        type_perf2,
        type_capability_gap.reshape(-1, 1),
        type_capability_std.reshape(-1, 1),
        type_capability_min.reshape(-1, 1),
        active_pattern_all.reshape(-1, 1),
        active_pattern_primary.reshape(-1, 1),
        single_asset_type_count.reshape(-1, 1),
        double_asset_type_count.reshape(-1, 1),
        no_asset_type_count.reshape(-1, 1),
        capability_entropy.reshape(-1, 1),
        perf_ratio.reshape(-1, 1),
        type1_to_threat.reshape(-1, 1),
        type2_to_threat.reshape(-1, 1),
        type3_to_threat.reshape(-1, 1),
        pressure_by_type,
        target_by_type_active,
        threat_intensity,
    ]

    names = [
        "total_targets",
        "active_group_count",
        "total_threat",
        "max_threat",
        "mean_threat",
        "std_threat",
        "threat_concentration",
        "weighted_mean_angle",
        "angle_span",
        "total_capability",
        "total_perf1",
        "total_perf2",
        "active_equipment_count",
        "inactive_equipment_count",
        "capability_balance",
        "max_unit_capability",
        "min_active_capability",
        "pressure_total",
        "target_pressure",
        "capability_per_target",
        "high_threat_to_best_unit",
        "resilience_ratio",
        "type1_capability",
        "type2_capability",
        "type3_capability",
        "type1_active_count",
        "type2_active_count",
        "type3_active_count",
        "type1_perf1_capacity",
        "type2_perf1_capacity",
        "type3_perf1_capacity",
        "type1_perf2_capacity",
        "type2_perf2_capacity",
        "type3_perf2_capacity",
        "type_capability_gap",
        "type_capability_std",
        "type_capability_min",
        "all_equipment_active",
        "primary_assets_active",
        "single_asset_type_count",
        "double_asset_type_count",
        "no_asset_type_count",
        "capability_entropy",
        "perf1_to_perf2_ratio",
        "type1_capability_to_threat",
        "type2_capability_to_threat",
        "type3_capability_to_threat",
        "pressure_on_type1",
        "pressure_on_type2",
        "pressure_on_type3",
        "targets_per_type1_active",
        "targets_per_type2_active",
        "targets_per_type3_active",
        "group1_threat",
        "group2_threat",
        "group3_threat",
        "group4_threat",
        "group5_threat",
        "group6_threat",
    ]
    engineered = np.hstack(engineered_parts)
    return np.hstack([x, engineered]), [f"raw_f{i:02d}" for i in range(1, x.shape[1] + 1)] + names
