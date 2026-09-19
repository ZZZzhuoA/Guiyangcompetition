"""在独立真实轨迹中匹配相似态势，估计三种策略的条件收益。"""

from __future__ import annotations

import numpy as np

from src.finals.features import FEATURE_NAMES
from src.finals.schema import STRATEGIES


# 只用决策前可观测、含义稳定的态势量做距离；策略编号不进距离。
MATCH_FEATURE_NAMES = (
    "n_alive",
    "n_low_alt",
    "swarm_ratio",
    "n_tracked",
    "n_intercepting",
    "n_swarm_groups",
    "min_threat",
    "mean_threat",
    "min_range",
    "mean_range",
    "min_tclose",
    "mean_tclose",
    "n_can_links",
    "n_high_p_links",
    "mean_p",
    "mean_cost",
    "n_uncovered",
    "high_threat_uncovered",
    "resource_target_ratio",
    "expected_greedy_p",
    "remain_ratio",
    "urgency",
    "coverage_gap",
    "scarcity",
    "threat_pressure",
    "swarm_pressure",
    "time_frac",
)

HISTORY_MATCH_FEATURE_NAMES = (
    "n_alive", "n_intercepting", "min_threat", "min_range",
    "mean_p", "n_uncovered", "urgency", "remain_ratio", "time_frac",
)


def is_observed_dataset(data: dict) -> bool:
    mode = data.get("dataset_mode")
    if mode is None:
        return False
    values = np.asarray(mode).reshape(-1)
    return bool(len(values) and str(values[0]).startswith("observed_"))


def _matching_matrix(data: dict, idx: np.ndarray) -> tuple[np.ndarray, list[str]]:
    feature_index = {name: i for i, name in enumerate(FEATURE_NAMES)}
    names = [name for name in MATCH_FEATURE_NAMES if name in feature_index]
    cols = [feature_index[name] for name in names]
    raw = np.asarray(data["x"][idx][:, cols], dtype=float)
    if "x_hist" in data:
        hist_index = [feature_index[name] for name in HISTORY_MATCH_FEATURE_NAMES if name in feature_index]
        hist = np.asarray(data["x_hist"][idx][:, :-1, hist_index], dtype=float)
        raw = np.concatenate([raw, hist.reshape(len(idx), -1)], axis=1)
        names = names + [
            f"hist{step}_{name}"
            for step in range(hist.shape[1])
            for name in HISTORY_MATCH_FEATURE_NAMES
            if name in feature_index
        ]
    lo = np.quantile(raw, 0.10, axis=0)
    hi = np.quantile(raw, 0.90, axis=0)
    scale = np.where(hi - lo < 1e-9, 1.0, hi - lo)
    return np.clip((raw - lo) / scale, -4.0, 4.0), names


def _projection_candidates(
    z: np.ndarray,
    candidates: np.ndarray,
    *,
    strategy: int,
    wanted: int,
    batch_size: int = 128,
):
    """纯 NumPy 的有界内存近邻候选生成器。

    在多个确定性投影上取 anchor 附近的窗口，再对小候选池计算真实欧氏
    距离。它不创建 n×n 距离矩阵，也不要求赛方环境安装 sklearn/scipy。
    """
    rng = np.random.default_rng(17_003 + int(strategy))
    n_random = 8
    projections = rng.normal(size=(z.shape[1], n_random))
    projections /= np.maximum(np.linalg.norm(projections, axis=0, keepdims=True), 1e-12)
    candidate_scores = z[candidates] @ projections
    orders = np.argsort(candidate_scores, axis=0)
    sorted_scores = [candidate_scores[orders[:, col], col] for col in range(n_random)]
    half_width = max(32, int(wanted) * 4)
    offsets = np.arange(-half_width, half_width + 1, dtype=int)

    for start in range(0, len(z), int(batch_size)):
        stop = min(len(z), start + int(batch_size))
        anchor_scores = z[start:stop] @ projections
        parts = []
        for col in range(n_random):
            positions = np.searchsorted(sorted_scores[col], anchor_scores[:, col])
            window = np.clip(
                positions[:, None] + offsets[None, :], 0, len(candidates) - 1
            )
            parts.append(candidates[orders[window, col]])
        yield start, stop, np.concatenate(parts, axis=1)


def build_matched_groups(
    data: dict,
    mask: np.ndarray,
    *,
    k: int = 12,
    min_neighbors: int = 3,
    max_neighbor_distance: float = 6.0,
    tie_tolerance: float = 0.01,
) -> tuple[dict, dict]:
    """在 mask 内独立匹配，返回每个 anchor 的三策略估计收益组。

    anchor 自己的真实结果保留为事实结果；同一 run 的其他时刻不作为邻居，
    避免一条轨迹在时间上自我复制。同场景的其他独立实验允许参与，因为它们
    正是相同初始条件下的真实策略实验。
    """
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        raise ValueError("cannot match an empty split")
    z, names = _matching_matrix(data, idx)
    action = np.asarray(data["strategy"][idx], dtype=int)
    scene = np.asarray(data["scene_id"][idx], dtype=int)
    time = np.asarray(data["time"][idx], dtype=float)
    run_id_all = np.asarray(data.get("run_id", data["group_id"]))
    run_id = run_id_all[idx].astype(str)
    target_all = np.asarray(data.get("return_to_go", data.get("intercept_rate", data["utility"])), dtype=float)
    target = target_all[idx]
    by_action = {int(a): np.flatnonzero(action == int(a)) for a in STRATEGIES}
    if any(len(rows) == 0 for rows in by_action.values()):
        missing = [int(a) for a, rows in by_action.items() if len(rows) == 0]
        raise ValueError(f"split 中缺少策略 {missing} 的真实样本，无法构造共同支持")
    # 每个 anchor 都必须能找到其他策略的真实邻居；当前策略自己的事实结果
    # 可以额外加入，但不能用它掩盖其他策略没有共同支持的事实。
    supports = []
    for strategy, rows in by_action.items():
        _, counts = np.unique(run_id[rows], return_counts=True)
        supports.append(len(rows) - int(np.max(counts)))
    effective_min_neighbors = min(int(min_neighbors), max(1, min(supports)))

    estimates = np.full((len(idx), len(STRATEGIES)), np.nan, dtype=float)
    valid = np.ones(len(idx), dtype=bool)
    aux_keys = [
        key for key in (
            "x_next", "reward_delta", "r_delta", "r_term", "mix", "r_shaped",
            "phi", "phi_next", "shaping", "ht_leak_delta", "n_faced_delta",
            "n_faced_term", "done", "intercept_delta", "return_to_go",
            "episode_intercept_rate", "has_opportunity",
        ) if key in data
    ]
    aux_estimates = {}
    for key in aux_keys:
        source = np.asarray(data[key])
        shape = (len(idx), len(STRATEGIES), *source.shape[1:])
        aux_estimates[key] = np.full(shape, np.nan, dtype=float)
    neighbor_distances: list[float] = []
    wanted = max(int(k), effective_min_neighbors)
    # 每种策略只建一次多投影候选索引，再分块算精确距离。这样不会逐 anchor
    # 扫描整个 split，也不会生成可能耗尽内存的 n×n 距离矩阵。
    for action_col, strategy in enumerate(STRATEGIES):
        candidates = by_action[int(strategy)]
        for start, stop, candidate_pool in _projection_candidates(
            z, candidates, strategy=int(strategy), wanted=wanted
        ):
            anchors = np.arange(start, stop, dtype=int)
            factual_or_fallback = np.where(
                action[anchors] == int(strategy), anchors, int(candidates[0])
            )
            candidate_pool = np.concatenate(
                [candidate_pool, factual_or_fallback.reshape(-1, 1)], axis=1
            )
            for offset, anchor_local in enumerate(range(start, stop)):
                candidate_rows = np.unique(candidate_pool[offset])
                factual = candidate_rows == anchor_local
                different_run = run_id[candidate_rows] != run_id[anchor_local]
                eligible = factual | different_run
                candidate_rows = candidate_rows[eligible]
                if len(candidate_rows) < effective_min_neighbors:
                    valid[anchor_local] = False
                    continue
                diff = z[candidate_rows] - z[anchor_local]
                distances = np.sqrt(np.sum(diff * diff, axis=1))
                take_n = min(wanted, len(candidate_rows))
                take = np.argpartition(distances, take_n - 1)[:take_n]
                picked_rows = candidate_rows[take]
                picked_dist = distances[take]
                if float(np.max(picked_dist)) > float(max_neighbor_distance):
                    valid[anchor_local] = False
                    continue
                weights = 1.0 / np.maximum(0.10, picked_dist)
                estimates[anchor_local, action_col] = float(
                    np.sum(weights * target[picked_rows]) / np.sum(weights)
                )
                for key in aux_keys:
                    source = np.asarray(data[key][idx])
                    values = source[picked_rows]
                    if source.ndim == 1:
                        aux_estimates[key][anchor_local, action_col] = float(
                            np.sum(weights * values) / np.sum(weights)
                        )
                    else:
                        aux_estimates[key][anchor_local, action_col] = np.sum(
                            values * weights.reshape((-1,) + (1,) * (source.ndim - 1)), axis=0
                        ) / np.sum(weights)
                neighbor_distances.append(float(np.mean(picked_dist)))

    valid &= np.all(np.isfinite(estimates), axis=1)
    valid_rows = np.flatnonzero(valid)
    if len(valid_rows) == 0:
        raise ValueError(
            "相似态势匹配没有得到任何三策略共同支持组；增加场景、减小 min_neighbors，"
            "或检查三种策略是否都被正确识别"
        )
    values_matrix = estimates[valid_rows].copy()
    ordered = np.sort(values_matrix, axis=1)
    ambiguous = ordered[:, -1] - ordered[:, -2] <= float(tie_tolerance)
    # 只把近似并列的最优动作设成同值，仍保留它们相对第三种策略的监督信号。
    for row in np.flatnonzero(ambiguous):
        best = float(np.max(values_matrix[row]))
        near_best = best - values_matrix[row] <= float(tie_tolerance)
        values_matrix[row, near_best] = best
    values = values_matrix.reshape(-1)
    anchor_x = np.asarray(data["x"][idx[valid_rows]], dtype=float)
    groups = [
        f"matched|scene{int(scene[row])}|row{int(idx[row])}" for row in valid_rows
    ]
    matched = {
        "x": np.repeat(anchor_x, len(STRATEGIES), axis=0),
        "strategy": np.tile(np.asarray(STRATEGIES, dtype=int), len(valid_rows)),
        "utility": values.copy(),
        "intercept_rate": values.copy(),
        "scene_id": np.repeat(scene[valid_rows], len(STRATEGIES)).astype(int),
        "time": np.repeat(time[valid_rows], len(STRATEGIES)).astype(float),
        "kind": np.asarray(["matched"] * len(values)),
        "group_id": np.repeat(np.asarray(groups), len(STRATEGIES)),
        "feature_names": np.asarray(FEATURE_NAMES),
        "dataset_mode": np.asarray(["matched_similar_states"]),
    }
    if "x_hist" in data:
        matched["x_hist"] = np.repeat(np.asarray(data["x_hist"][idx[valid_rows]], dtype=float), len(STRATEGIES), axis=0)
    if "a_prev" in data:
        matched["a_prev"] = np.repeat(np.asarray(data["a_prev"][idx[valid_rows]], dtype=int), len(STRATEGIES), axis=0)
    if "hist_len" in data:
        matched["hist_len"] = np.repeat(np.asarray(data["hist_len"][idx[valid_rows]], dtype=int), len(STRATEGIES))
    for key in aux_keys:
        values_aux = aux_estimates[key][valid_rows]
        matched[key] = values_aux.reshape((len(values_aux) * len(STRATEGIES), *values_aux.shape[2:]))
    diagnostics = {
        "raw_rows": int(len(idx)),
        "matched_groups": int(len(values) // len(STRATEGIES)),
        "matched_rows": int(len(values)),
        "neighbor_search": "numpy_multi_projection",
        "dropped_no_common_support": int(len(idx) - len(valid_rows)),
        "tied_groups": int(np.sum(ambiguous)),
        "k": int(k),
        "min_neighbors": int(effective_min_neighbors),
        "requested_min_neighbors": int(min_neighbors),
        "max_neighbor_distance": float(max_neighbor_distance),
        "tie_tolerance": float(tie_tolerance),
        "mean_neighbor_distance": None if not neighbor_distances else float(np.mean(neighbor_distances)),
        "match_features": names,
        "strategy_source_counts": {
            int(strategy): int(np.sum(action == int(strategy))) for strategy in STRATEGIES
        },
    }
    return matched, diagnostics
