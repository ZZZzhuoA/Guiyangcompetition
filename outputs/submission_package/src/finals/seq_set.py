"""贯序训练集：当前态势执行 a 一个间隔 Δ，再可选跑完。

标签（见 src/finals/reward.py）：
  r_delta   本段拦截增量 / 本局已出现蓝方数
  r_term    此后一直用 a 的整局拦截率
  mix       r_delta + λ r_term
  r_shaped  r_delta + γΦ(x') - Φ(x) - 高威胁漏防修正，供 fitted-Q
并记下间隔结束时的 x_next。
"""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np

from src.finals.context import DecisionContext
from src.finals.dataset import (
    _already_intercepted,
    _sample_row,
    decision_times,
    inventory_runs,
    load_grouped_runs,
    missing_sample_hint,
    outcome_from_run,
    time_at_or_after,
)
from src.finals.features import FEATURE_NAMES, dict_to_vector, extract_feature_dict, utility_from_outcome
from src.finals.progress import ProgressBar, SceneCheckpoint
from src.finals.reward import (
    GAMMA,
    MIX_LAMBDA,
    blue_seen,
    episode_return,
    high_threat_leaked,
    mix_label,
    potential,
    segment_reward,
)
from src.finals.schema import STRATEGIES, seconds_to_ticks
from src.finals.sequence import WINDOW, build_window
from src.finals.simulate import clone_world, continue_episode, run_episode, world_from_snapshot
from src.finals.slice_set import drop_reason
from src.finals.snapshot import build_table_index, snapshot_from_tables

LABEL_COLUMNS = (
    "r_delta",
    "r_term",
    "mix",
    "r_shaped",
    "phi",
    "phi_next",
    "shaping",
    "ht_leak_delta",
    "n_faced_delta",
    "n_faced_term",
    "done",
)


def _snapshot_from_logs(logs) -> object:
    health = [row for log in logs for row in log.health]
    last = logs[-1]
    return snapshot_from_tables(last.rhdl, health, last.lj, last.time)


def _fork_seq(snap, feat_dict: dict, ctx: DecisionContext, n_blue: int, scene_id: int, time: float, delta: int, remain: int, seed: int, dt: float = 1.0):
    """三种策略各跑一个 Δ 再跑完，返回同一组内可比的奖励。

    delta / remain 是秒。样本 dt=0.5 时要换成对应步数，否则 15 步只有 7.5 秒。
    """
    world0 = world_from_snapshot(snap, n_blue_init=max(n_blue, len(snap.targets)), dt=dt)
    n_delta = seconds_to_ticks(delta, dt)
    n_remain = seconds_to_ticks(remain, dt)
    phi = potential(feat_dict)
    raw = []
    for strategy in STRATEGIES:
        rng = np.random.default_rng(seed + 1000 * scene_id + 17 * int(time) + 31 * strategy)
        world = clone_world(world0)
        logs_delta = run_episode(world, strategy, rng, n_delta)
        intercepted_delta = int(logs_delta[-1].intercepted)
        faced_delta = blue_seen(world)
        ht_leak_delta = high_threat_leaked(world)
        ctx_next = copy.deepcopy(ctx)
        ctx_next.commit(snap, feat_dict, strategy)
        snap_next = _snapshot_from_logs(logs_delta)
        feat_next_dict = extract_feature_dict(snap_next, context=ctx_next)
        done = not world.pending and not any(b.alive and b.observed for b in world.blues)
        logs_rest = continue_episode(world, strategy, rng, n_remain)
        last = logs_rest[-1] if logs_rest else logs_delta[-1]
        raw.append(
            {
                "strategy": strategy,
                "intercepted_delta": intercepted_delta,
                "faced_delta": faced_delta,
                "ht_leak_delta": ht_leak_delta,
                "feat_next_dict": feat_next_dict,
                "done": done,
                "intercepted_term": int(last.intercepted),
                "faced_term": blue_seen(world),
                "leaked_term": int(last.leaked),
                "cost_term": float(last.cost),
            }
        )
    faced_delta = max(item["faced_delta"] for item in raw)
    faced_term = max(item["faced_term"] for item in raw)
    out = []
    for item in raw:
        feat_next_dict = item["feat_next_dict"]
        reward = segment_reward(
            item["intercepted_delta"],
            faced_delta,
            phi,
            potential(feat_next_dict),
            ht_leak_delta=item["ht_leak_delta"],
            gamma=GAMMA,
            done=item["done"],
        )
        r_term = episode_return(item["intercepted_term"], faced_term)
        labels = {
            **reward,
            "r_term": r_term,
            "mix": mix_label(reward["r_delta"], r_term),
            "n_faced_delta": float(faced_delta),
            "n_faced_term": float(faced_term),
            "done": 1.0 if item["done"] else 0.0,
        }
        diagnostics = utility_from_outcome(
            item["intercepted_term"],
            item["leaked_term"],
            item["cost_term"],
            faced_term,
        )
        out.append((item["strategy"], labels, diagnostics, dict_to_vector(feat_next_dict)))
    return out


def _real_seq_from_runs(by_strategy: dict, snap, feat_dict: dict, ctx: DecisionContext, t0: float, delta: int):
    """t0 上若同一场景真有三种策略对照，用真实轨迹；`_1/_2/_3` 只是重复实验时走分叉仿真。"""
    if any(strategy not in by_strategy for strategy in STRATEGIES):
        return None
    phi = potential(feat_dict)
    raw = []
    for strategy in STRATEGIES:
        run = by_strategy[strategy]
        times = run["times"]
        t_next = time_at_or_after(times, t0 + float(delta))
        intercepted_t0 = _already_intercepted(run["health"], t0)
        intercepted_next = _already_intercepted(run["health"], t_next)
        intercepted_delta = max(0, intercepted_next - intercepted_t0)
        n_blue = max(1, int(run["n_blue"]))
        official = run.get("official_intercept_rate")
        if official is not None:
            intercepted_term = float(official) * n_blue
            leaked_term = max(0.0, n_blue - intercepted_term)
        else:
            intercepted_term = float(run["intercepted"])
            leaked_term = float(run["leaked"])
        snap_next = snapshot_from_tables(run["rhdl"], run["health"], run["lj"], t_next, strategy)
        ctx_next = copy.deepcopy(ctx)
        ctx_next.commit(snap, feat_dict, strategy)
        feat_next_dict = extract_feature_dict(snap_next, context=ctx_next)
        done = abs(t_next - times[-1]) < 1e-9
        raw.append(
            {
                "strategy": strategy,
                "intercepted_delta": intercepted_delta,
                "n_blue": n_blue,
                "feat_next_dict": feat_next_dict,
                "done": done,
                "intercepted_term": intercepted_term,
                "leaked_term": leaked_term,
                "cost_term": float(run["cost"]),
            }
        )
    faced = max(item["n_blue"] for item in raw)
    out = []
    for item in raw:
        feat_next_dict = item["feat_next_dict"]
        reward = segment_reward(
            item["intercepted_delta"],
            faced,
            phi,
            potential(feat_next_dict),
            ht_leak_delta=0,
            gamma=GAMMA,
            done=item["done"],
        )
        r_term = episode_return(item["intercepted_term"], faced)
        if by_strategy[item["strategy"]].get("official_intercept_rate") is not None:
            r_term = float(by_strategy[item["strategy"]]["official_intercept_rate"])
        labels = {
            **reward,
            "r_term": r_term,
            "mix": mix_label(reward["r_delta"], r_term),
            "n_faced_delta": float(faced),
            "n_faced_term": float(faced),
            "done": 1.0 if item["done"] else 0.0,
        }
        diagnostics = outcome_from_run(by_strategy[item["strategy"]])
        out.append((item["strategy"], labels, diagnostics, dict_to_vector(feat_next_dict)))
    return out


_SCENE_ERRORS = (TypeError, ValueError, KeyError, IndexError, AttributeError)


def _pack_seq_extra(feat_next, x_hist, a_prev, hist_len) -> dict:
    return {
        "x_next": np.asarray(feat_next, dtype=float).tolist(),
        "x_hist": np.asarray(x_hist, dtype=float).tolist(),
        "a_prev": np.asarray(a_prev).tolist(),
        "hist_len": int(hist_len),
    }


def _unpack_seq_extra(extra: list[dict]) -> tuple[list, list, list, list]:
    x_next_rows, hist_x_rows, hist_a_rows, hist_len_rows = [], [], [], []
    for item in extra:
        x_next_rows.append(np.asarray(item["x_next"], dtype=float))
        hist_x_rows.append(np.asarray(item["x_hist"], dtype=float))
        hist_a_rows.append(np.asarray(item["a_prev"], dtype=int))
        hist_len_rows.append(int(item["hist_len"]))
    return x_next_rows, hist_x_rows, hist_a_rows, hist_len_rows


def _process_scene(
    scene_id: int,
    by_strategy: dict,
    delta: int,
    max_mid_slices: int,
    n_ticks: int,
    remain: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    ref = by_strategy.get(1) or next(iter(by_strategy.values()))
    index = build_table_index(ref["rhdl"], ref["health"], ref["lj"])
    times = decision_times(ref["times"], delta)
    if not times:
        return [], [{"scene_id": scene_id, "time": None, "kind": "scene", "reason": "no_times"}], []
    ctx = DecisionContext(horizon=max(float(n_ticks), float(ref["times"][-1])))
    prev = None
    kept_mid = 0
    hist_frames: list[np.ndarray] = []
    hist_actions: list[int] = []
    rows: list[dict] = []
    dropped: list[dict] = []
    extra: list[dict] = []
    for i, time in enumerate(times):
        snap = snapshot_from_tables(ref["rhdl"], ref["health"], ref["lj"], time, ref["strategy"], index=index)
        feat_dict = extract_feature_dict(snap, context=ctx)
        feat = dict_to_vector(feat_dict)
        # 历史含**每个被访问到的决策点**，包括训练里被 drop_reason 丢掉的：
        # 推理时策略在每次触发都会被调用，两边窗口含义必须一致。
        x_hist, a_prev, hist_len = build_window(hist_frames, hist_actions, feat)
        hist_frames.append(feat)
        hist_actions.append(int(ref["strategy"]))
        kind = "t0" if i == 0 else "fork"
        reason = drop_reason(feat, prev)
        if i == 0:
            if reason:
                dropped.append({"scene_id": scene_id, "time": time, "kind": kind, "reason": reason})
                ctx.commit(snap, feat_dict, 0)
                prev = feat
                continue
        else:
            if kept_mid >= max_mid_slices:
                break
            if reason:
                dropped.append({"scene_id": scene_id, "time": time, "kind": kind, "reason": reason})
                continue
            kept_mid += 1
        group_id = f"scene{scene_id}|{'t0' if i == 0 else 'slice' + str(time)}"
        if i == 0:
            forks = _real_seq_from_runs(by_strategy, snap, feat_dict, ctx, time, delta)
            if forks is None:
                forks = _fork_seq(
                    snap, feat_dict, ctx, ref["n_blue"], scene_id, time, delta, remain, seed=17,
                    dt=float(ref.get("dt", 1.0)),
                )
        else:
            forks = _fork_seq(
                snap, feat_dict, ctx, ref["n_blue"], scene_id, time, delta, remain, seed=17,
                dt=float(ref.get("dt", 1.0)),
            )
        for strategy, labels, diagnostics, feat_next in forks:
            outcome = {**diagnostics, "utility": labels["mix"]}
            row = _sample_row(scene_id, time, strategy, feat, outcome, group_id, kind)
            row.update({name: float(labels[name]) for name in LABEL_COLUMNS})
            row["hist_len"] = float(hist_len)
            rows.append(row)
            extra.append(_pack_seq_extra(feat_next, x_hist, a_prev, hist_len))
        ctx.commit(snap, feat_dict, int(ref["strategy"]))
        prev = feat
    return rows, dropped, extra


def build_seq_set(
    input_dir: Path,
    output_dir: Path,
    delta: int = 15,
    max_mid_slices: int = 3,
    n_ticks: int = 120,
    results_csv: Path | None = None,
    resume: bool = True,
) -> dict:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped, ingest = load_grouped_runs(input_dir, results_csv=results_csv)
    if not grouped:
        raise ValueError(missing_sample_hint(input_dir))
    remain = min(36, max(delta, n_ticks // 4))
    ckpt = SceneCheckpoint(output_dir)
    if resume:
        done, rows, dropped, extra = ckpt.load()
    else:
        ckpt.clear()
        done, rows, dropped, extra = set(), [], [], []
    if extra and rows and len(extra) != len(rows):
        raise ValueError(
            f"seq checkpoint extra ({len(extra)}) != rows ({len(rows)}); rerun with --fresh"
        )
    if rows and not extra:
        raise ValueError("seq checkpoint missing x_next/x_hist; rerun with --fresh")
    if done:
        print(f"resume seq: skip {len(done)} finished scene(s)", flush=True)
    scenes = [(scene_id, by_strategy) for scene_id, by_strategy in sorted(grouped.items()) if scene_id not in done]
    bar = ProgressBar(len(scenes), prefix="seq")
    for scene_id, by_strategy in scenes:
        try:
            scene_rows, scene_dropped, scene_extra = _process_scene(
                scene_id, by_strategy, delta, max_mid_slices, n_ticks, remain
            )
        except _SCENE_ERRORS as exc:
            scene_rows, scene_dropped, scene_extra = [], [{"scene_id": scene_id, "time": None, "kind": "scene", "reason": f"skip:{exc}"}], []
        rows.extend(scene_rows)
        dropped.extend(scene_dropped)
        extra.extend(scene_extra)
        ckpt.save_scene(scene_id, scene_rows, scene_dropped, scene_extra)
        bar.update(extra=f"scene {scene_id}")
    bar.close()
    if not rows:
        raise ValueError("No sequential samples kept")
    x_next_rows, hist_x_rows, hist_a_rows, hist_len_rows = _unpack_seq_extra(extra)
    fieldnames = [
        "scene_id",
        "time",
        "strategy",
        "group_id",
        "kind",
        "intercept_rate",
        "leak_rate",
        "cost",
        "cost_eff",
        "utility",
        *LABEL_COLUMNS,
        "hist_len",
        *FEATURE_NAMES,
    ]
    csv_path = output_dir / "training_samples.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)
    drop_path = output_dir / "dropped_points.csv"
    with drop_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["scene_id", "time", "kind", "reason"], lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(dropped)
    from src.finals.train import save_training_npz

    # 历史窗口只进 npz：CSV 存不下 (K, F) 的窗口。只有 CSV 时序列模型会退化成单帧冷启动。
    save_training_npz(
        output_dir,
        rows,
        extra={
            "x_next": np.vstack(x_next_rows),
            "x_hist": np.stack(hist_x_rows),
            "a_prev": np.stack(hist_a_rows),
            "hist_len": np.array(hist_len_rows, dtype=int),
        },
    )
    ckpt.clear()
    summary = {
        "mode": "sequential_scoring",
        "delta": delta,
        "gamma": GAMMA,
        "mix_lambda": MIX_LAMBDA,
        "window": WINDOW,
        "n_rows": len(rows),
        "n_scenes": len(grouped),
        "n_runs": ingest.get("n_runs", sum(len(by_strategy) for by_strategy in grouped.values())),
        "n_scene_strategy_pairs": ingest.get("n_scene_strategy_pairs", len(grouped)),
        "n_scenes_with_3_strategies": ingest.get("n_scenes_with_3_strategies", 0),
        "n_scenes_with_3_replicates": ingest.get("n_scenes_with_3_replicates", 0),
        "n_official_matched": ingest.get("n_official_matched", 0),
        "results_csv": ingest.get("results_csv"),
        "n_features": len(FEATURE_NAMES),
        "n_dropped": len(dropped),
        "n_load_errors": ingest.get("n_load_errors", 0),
        "runs": inventory_runs(grouped),
        "kind_counts": {kind: sum(1 for row in rows if row["kind"] == kind) for kind in sorted({row["kind"] for row in rows})},
        "label_means": {name: float(np.mean([row[name] for row in rows])) for name in ("r_delta", "r_term", "mix", "r_shaped", "shaping")},
        "csv": str(csv_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
