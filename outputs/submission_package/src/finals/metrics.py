"""选模指标。赛方主指标是整局拦截率；其余用于诊断和打破平局。"""

from __future__ import annotations

import time

import numpy as np

from src.finals.schema import STRATEGIES


def strategy_histogram(actions: np.ndarray) -> dict[int, int]:
    hist = {s: 0 for s in STRATEGIES}
    for value in actions:
        hist[int(value)] = hist.get(int(value), 0) + 1
    return hist


def collapse_rate(actions: np.ndarray) -> float:
    if len(actions) == 0:
        return 1.0
    hist = strategy_histogram(actions)
    return max(hist.values()) / len(actions)


def _history_slice(history: dict | None, i: int) -> dict:
    """序列模型按行取窗口。没有窗口就交给模型自己退化成单帧冷启动。"""
    if not history:
        return {}
    out = {}
    for key, value in history.items():
        if value is None:
            continue
        out[key] = np.asarray(value[i])[None, ...]
    return out


def group_match(
    scorer,
    x: np.ndarray,
    strategy: np.ndarray,
    utility: np.ndarray,
    group_id: np.ndarray,
    history: dict | None = None,
) -> dict:
    groups: dict[str, list[int]] = {}
    for i, gid in enumerate(group_id):
        groups.setdefault(str(gid), []).append(i)
    correct = 0
    total = 0
    pred_hist = {s: 0 for s in STRATEGIES}
    oracle_hist = {s: 0 for s in STRATEGIES}
    margin = []
    for idxs in groups.values():
        if len({int(strategy[i]) for i in idxs}) < 2:
            continue
        best = int(strategy[idxs[int(np.argmax([utility[i] for i in idxs]))]])
        rec, scores = scorer.recommend(x[idxs[0]].reshape(1, -1), **_history_slice(history, idxs[0]))
        pred = int(rec[0])
        pred_hist[pred] += 1
        oracle_hist[best] += 1
        correct += int(pred == best)
        total += 1
        row = np.asarray(scores[0], dtype=float)
        ordered = np.sort(row)
        margin.append(float(ordered[-1] - ordered[-2]) if len(ordered) >= 2 else 0.0)
    return {
        "n_groups": total,
        "match": None if total == 0 else correct / total,
        "pred_hist": pred_hist,
        "oracle_hist": oracle_hist,
        "collapse": collapse_rate(np.array([s for s, n in pred_hist.items() for _ in range(n)])),
        "mean_score_margin": None if not margin else float(np.mean(margin)),
        "random_baseline": 1.0 / len(STRATEGIES),
    }


def recommend_latency_ms(scorer, x: np.ndarray, repeats: int = 20, history: dict | None = None) -> float:
    """序列模型要连编码一起计时，否则测出来的延迟比实装偏低。"""
    if len(x) == 0:
        return 0.0
    row = x[:1]
    kwargs = _history_slice(history, 0)
    scorer.recommend(row, **kwargs)
    t0 = time.perf_counter()
    for _ in range(repeats):
        scorer.recommend(row, **kwargs)
    return 1000.0 * (time.perf_counter() - t0) / repeats


def selection_score(row: dict) -> float | None:
    """有周期回放时用拦截率；否则只用对照 match，并惩罚策略塌缩。"""
    match = row.get("test_match")
    diversity = 1.0 - float(row.get("test_collapse") or 1.0)
    latency = float(row.get("latency_ms") or 0.0)
    if latency > 50.0:
        return None
    intercept = row.get("episode_intercept_rate")
    lift = row.get("lift_vs_best_fixed")
    if intercept is None:
        if match is None:
            return None
        return 0.7 * float(match) + 0.3 * diversity
    lift_v = 0.0 if lift is None else float(np.clip(lift, -0.2, 0.2) / 0.2)
    return 0.55 * float(intercept) + 0.20 * max(0.0, lift_v) + 0.15 * float(match or 0.0) + 0.10 * diversity


def attach_selection(rows: list[dict]) -> list[dict]:
    for row in rows:
        row["selection_score"] = selection_score(row)
        row["latency_ok"] = bool((row.get("latency_ms") or 0.0) <= 50.0)
    ranked = sorted(
        [row for row in rows if row.get("selection_score") is not None],
        key=lambda item: item["selection_score"],
        reverse=True,
    )
    for i, row in enumerate(ranked):
        row["rank"] = i + 1
    return rows
