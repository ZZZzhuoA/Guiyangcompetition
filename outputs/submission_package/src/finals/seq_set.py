"""真实贯序训练集：历史窗口 + 实际动作 + 区间收益 + 下一状态。

每条 run 都独立读取，不从一条轨迹模拟其它策略。三策略条件收益在训练阶段
按场景切分后由 matching.py 从相似历史态势估计，避免把 fork 标签混入训练。
"""

from __future__ import annotations

import copy
import csv
import json
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

from src.finals.context import DecisionContext
from src.finals.dataset import (
    _already_intercepted,
    _sample_row,
    decision_times,
    inventory_catalog,
    load_run,
    load_catalog_scene,
    load_run_catalog,
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
from src.finals.slice_set import _idle_kept, _terminal_intercepted, drop_reason
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

OBSERVED_SEQ_VERSION = 2

# 下面的 fork 辅助函数只为兼容旧的内部调用保留；当前 build_seq_set 走的是
# 文件中后面的 observed run 路径，不会调用它们，也不会把仿真标签写入新数据集。

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


_SCENE_ERRORS = (TypeError, ValueError, KeyError, IndexError, AttributeError, OSError)
_SEQ_WORKER_OFFICIAL = None
_SEQ_WORKER_CONFIG: tuple[int, int, int, int] | None = None


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


def _build_catalog_scene(
    scene_id: int,
    entries: dict,
    official,
    delta: int,
    max_mid_slices: int,
    n_ticks: int,
    remain: int,
) -> tuple[int, list[dict], list[dict], list[dict], list[dict]]:
    """加载并构造一个场景；进程 worker 与单进程路径共用。"""
    by_strategy, scene_load_errors = load_catalog_scene(entries, official)
    try:
        if not by_strategy:
            raise ValueError("no readable run in scene")
        scene_rows, scene_dropped, scene_extra = _process_scene(
            scene_id, by_strategy, delta, max_mid_slices, n_ticks, remain
        )
    except _SCENE_ERRORS as exc:
        scene_rows = []
        scene_extra = []
        scene_dropped = [
            {"scene_id": scene_id, "time": None, "kind": "scene", "reason": f"skip:{exc}"}
        ]
    for item in scene_load_errors:
        scene_dropped.append(
            {"scene_id": scene_id, "time": None, "kind": "load", "reason": f"skip:{item['error']}"}
        )
    return scene_id, scene_rows, scene_dropped, scene_extra, scene_load_errors


def _init_seq_worker(official, delta: int, max_mid_slices: int, n_ticks: int, remain: int) -> None:
    global _SEQ_WORKER_OFFICIAL, _SEQ_WORKER_CONFIG
    _SEQ_WORKER_OFFICIAL = official
    _SEQ_WORKER_CONFIG = (int(delta), int(max_mid_slices), int(n_ticks), int(remain))


def _seq_worker_task(scene: tuple[int, dict]):
    if _SEQ_WORKER_OFFICIAL is None or _SEQ_WORKER_CONFIG is None:
        raise RuntimeError("sequential worker was not initialized")
    scene_id, entries = scene
    delta, max_mid_slices, n_ticks, remain = _SEQ_WORKER_CONFIG
    return _build_catalog_scene(
        scene_id,
        entries,
        _SEQ_WORKER_OFFICIAL,
        delta,
        max_mid_slices,
        n_ticks,
        remain,
    )


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


def _process_observed_run(
    scene_id: int,
    run: dict,
    run_number: int,
    delta: int,
    max_intervals: int,
    idle_keep_ratio: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    """从一条真实策略轨迹构造带历史窗口的观测转移。

    一条 run 只包含实际执行过的一个策略；三策略标签由训练阶段的
    matching.py 在场景内独立匹配得到。这里绝不从当前态势模拟其它策略。
    """
    strategy = int(run["strategy"])
    replicate = int(run.get("replicate_id") or run_number)
    run_id = str(run["directory"])
    index = build_table_index(run["rhdl"], run["health"], run["lj"])
    starts = decision_times(run["times"], float(delta))
    ctx = DecisionContext(horizon=max(float(run["times"][-1]), float(delta)))
    n_blue = max(1, int(run["n_blue"]))
    terminal_intercepted = _terminal_intercepted(run)
    episode_rate = terminal_intercepted / n_blue
    hist_frames: list[np.ndarray] = []
    hist_actions: list[int] = []
    rows: list[dict] = []
    dropped: list[dict] = []
    extra: list[dict] = []
    kept = 0

    for time in starts:
        next_time = time_at_or_after(run["times"], float(time) + float(delta))
        if next_time <= float(time) + 1e-9:
            continue
        if max_intervals > 0 and kept >= max_intervals:
            break
        snap = snapshot_from_tables(
            run["rhdl"], run["health"], run["lj"], time, strategy, index=index
        )
        feat_dict = extract_feature_dict(snap, context=ctx)
        feat = dict_to_vector(feat_dict)
        x_hist, a_prev, hist_len = build_window(hist_frames, hist_actions, feat)
        hist_frames.append(feat)
        hist_actions.append(strategy)

        snap_next = snapshot_from_tables(
            run["rhdl"], run["health"], run["lj"], next_time, strategy, index=index
        )
        ctx_next = copy.deepcopy(ctx)
        ctx_next.commit(snap, feat_dict, strategy)
        feat_next_dict = extract_feature_dict(snap_next, context=ctx_next)
        feat_next = dict_to_vector(feat_next_dict)
        intercepted_t = _already_intercepted(run["health"], time)
        intercepted_next = _already_intercepted(run["health"], next_time)
        intercept_delta = max(0, intercepted_next - intercepted_t)
        return_to_go = max(0.0, terminal_intercepted - float(intercepted_t)) / n_blue
        has_opportunity = bool(
            float(feat_dict.get("n_alive", 0.0)) > 0.0
            and (
                float(feat_dict.get("n_can_links", 0.0)) > 0.0
                or float(feat_dict.get("n_intercepting", 0.0)) > 0.0
            )
        )
        if intercept_delta > 0:
            kind = "observed_positive"
        elif has_opportunity:
            kind = "observed_hard_negative"
        elif _idle_kept(scene_id, strategy, replicate, time, idle_keep_ratio):
            kind = "observed_idle"
        else:
            dropped.append({
                "scene_id": scene_id, "run_id": run_id, "strategy": strategy,
                "time": float(time), "kind": "observed_idle", "reason": "idle_subsample",
            })
            ctx.commit(snap, feat_dict, strategy)
            continue

        done = abs(float(next_time) - float(run["times"][-1])) <= 1e-9
        reward = segment_reward(
            intercept_delta, n_blue,
            potential(feat_dict), potential(feat_next_dict),
            ht_leak_delta=0, gamma=GAMMA, done=done,
        )
        labels = {
            **reward,
            "r_term": float(return_to_go),
            "mix": mix_label(reward["r_delta"], return_to_go),
            "n_faced_delta": float(n_blue),
            "n_faced_term": float(n_blue),
            "done": 1.0 if done else 0.0,
        }
        outcome = {
            "intercept_rate": float(return_to_go),
            "leak_rate": float(max(0.0, 1.0 - episode_rate)),
            "cost": float(run.get("cost", 0.0)),
            "cost_eff": float(episode_rate / (0.35 + float(run.get("cost", 0.0)))),
            "utility": float(return_to_go),
        }
        group_id = f"scene{scene_id}|run{run_number}|t{float(time):g}"
        row = _sample_row(scene_id, time, strategy, feat, outcome, group_id, kind)
        row.update({name: float(labels[name]) for name in LABEL_COLUMNS})
        row.update({
            "run_id": run_id,
            "replicate_id": replicate,
            "time_next": float(next_time),
            "intercepted_t": float(intercepted_t),
            "intercepted_next": float(intercepted_next),
            "intercept_delta": float(intercept_delta),
            "reward_delta": float(reward["r_delta"]),
            "return_to_go": float(return_to_go),
            "episode_intercept_rate": float(episode_rate),
            "has_opportunity": 1.0 if has_opportunity else 0.0,
            "label_source": str(run.get("label_source", "health")),
        })
        rows.append(row)
        extra.append(_pack_seq_extra(feat_next, x_hist, a_prev, hist_len))
        kept += 1
        ctx.commit(snap, feat_dict, strategy)
    return rows, dropped, extra


def _build_catalog_scene(
    scene_id: int,
    entries: dict,
    official,
    delta: int,
    max_intervals: int,
    idle_keep_ratio: float,
) -> tuple[int, list[dict], list[dict], list[dict], list[dict]]:
    rows: list[dict] = []
    dropped: list[dict] = []
    extra: list[dict] = []
    errors: list[dict] = []
    run_number = 0
    for strategy, strategy_entries in sorted(entries.items()):
        for entry in strategy_entries:
            run_number += 1
            directory = Path(entry["directory"])
            try:
                run = load_run(directory, official=official)
                run["strategy"] = int(strategy)
                run_rows, run_dropped, run_extra = _process_observed_run(
                    scene_id, run, run_number, delta, max_intervals, idle_keep_ratio
                )
                rows.extend(run_rows)
                dropped.extend(run_dropped)
                extra.extend(run_extra)
            except _SCENE_ERRORS as exc:
                errors.append({"directory": str(directory), "error": str(exc)})
                dropped.append({
                    "scene_id": scene_id, "run_id": str(directory),
                    "strategy": int(strategy), "time": None, "kind": "run",
                    "reason": f"skip:{exc}",
                })
    return scene_id, rows, dropped, extra, errors


def _init_seq_worker(official, delta: int, max_intervals: int, idle_keep_ratio: float) -> None:
    global _SEQ_WORKER_OFFICIAL, _SEQ_WORKER_CONFIG
    _SEQ_WORKER_OFFICIAL = official
    _SEQ_WORKER_CONFIG = (int(delta), int(max_intervals), float(idle_keep_ratio))


def _seq_worker_task(scene: tuple[int, dict]):
    if _SEQ_WORKER_OFFICIAL is None or _SEQ_WORKER_CONFIG is None:
        raise RuntimeError("sequential worker was not initialized")
    delta, max_intervals, idle_keep_ratio = _SEQ_WORKER_CONFIG
    return _build_catalog_scene(
        int(scene[0]), scene[1], _SEQ_WORKER_OFFICIAL,
        delta, max_intervals, idle_keep_ratio,
    )


def build_seq_set(
    input_dir: Path,
    output_dir: Path,
    delta: int = 15,
    max_mid_slices: int = 0,
    n_ticks: int = 120,
    results_csv: Path | None = None,
    resume: bool = True,
    workers: int | None = None,
    idle_keep_ratio: float = 0.20,
) -> dict:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = SceneCheckpoint(output_dir)
    version_path = ckpt.root / "builder.json"
    builder_config = {
        "version": OBSERVED_SEQ_VERSION,
        "mode": "observed_sequential_transitions",
        "delta": int(delta),
        "max_intervals_per_run": int(max_mid_slices),
        "idle_keep_ratio": float(idle_keep_ratio),
    }
    if not resume:
        ckpt.clear()
    elif ckpt.exists() and not version_path.exists():
        raise ValueError("seq checkpoint 是旧版贯序数据；请使用新输出目录或加 --fresh")
    if version_path.exists():
        stored = json.loads(version_path.read_text(encoding="utf-8"))
        if int(stored.get("version") or 0) != OBSERVED_SEQ_VERSION:
            raise ValueError("seq checkpoint 版本不匹配；请加 --fresh")
        mismatch = {
            key: (stored.get(key), value)
            for key, value in builder_config.items()
            if stored.get(key) != value
        }
        if mismatch:
            raise ValueError(f"seq checkpoint 构造参数不一致: {mismatch}；请沿用原参数或加 --fresh")
    ckpt.root.mkdir(parents=True, exist_ok=True)
    version_path.write_text(json.dumps(builder_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    catalog, official, ingest = load_run_catalog(
        input_dir,
        results_csv=results_csv,
        checkpoint_path=ckpt.root / "catalog.json",
        resume=resume,
    )
    if not catalog:
        raise ValueError(missing_sample_hint(input_dir))
    idle_keep_ratio = float(np.clip(idle_keep_ratio, 0.0, 1.0))
    if resume:
        done, rows, dropped, extra = ckpt.load()
    else:
        done, rows, dropped, extra = set(), [], [], []
    print(
        f"seq catalog: {ingest.get('n_runs', 0)} run(s), {len(catalog)} scene(s); "
        f"checkpoint: {ckpt.root}",
        flush=True,
    )
    if extra and rows and len(extra) != len(rows):
        raise ValueError(
            f"seq checkpoint extra ({len(extra)}) != rows ({len(rows)}); rerun with --fresh"
        )
    if rows and not extra:
        raise ValueError("seq checkpoint missing x_next/x_hist; rerun with --fresh")
    if done:
        print(f"resume seq: skip {len(done)} finished scene(s)", flush=True)
    # 启动时只需要 done 做跳过判断；旧结果在全部场景完成后从按 scene 排序的
    # 检查点统一装载。这里先释放历史 payload，给并行 worker 留出内存。
    rows, dropped, extra = [], [], []
    scenes = [(scene_id, entries) for scene_id, entries in sorted(catalog.items()) if scene_id not in done]
    worker_count = min(4, os.cpu_count() or 1) if workers is None else int(workers)
    if worker_count < 1:
        raise ValueError(f"workers must be >= 1, got {worker_count}")
    worker_count = min(worker_count, max(1, len(scenes)))
    print(f"seq workers: {worker_count}; pending scenes: {len(scenes)}", flush=True)
    bar = ProgressBar(len(scenes), prefix="seq")

    def accept(result) -> None:
        scene_id, scene_rows, scene_dropped, scene_extra, scene_load_errors = result
        if scene_load_errors:
            ingest["n_load_errors"] = int(ingest.get("n_load_errors", 0)) + len(scene_load_errors)
            ingest.setdefault("load_errors", []).extend(scene_load_errors)
        ckpt.save_scene(scene_id, scene_rows, scene_dropped, scene_extra)
        bar.update(extra=f"scene {scene_id}")

    if worker_count == 1:
        for scene_id, entries in scenes:
            accept(
                _build_catalog_scene(
                    scene_id, entries, official, delta, max_mid_slices, idle_keep_ratio
                )
            )
    elif scenes:
        # 只维持 2×workers 个在途场景。每个 worker 同时只持有一个场景的三张
        # 大表，既获得 CPU 并行，也避免一次提交全部场景后结果堆在主进程内存中。
        pool = ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_init_seq_worker,
            initargs=(official, delta, max_mid_slices, idle_keep_ratio),
        )
        pending = {}
        scene_iter = iter(scenes)

        def submit_one() -> bool:
            try:
                scene = next(scene_iter)
            except StopIteration:
                return False
            future = pool.submit(_seq_worker_task, scene)
            pending[future] = int(scene[0])
            return True

        for _ in range(min(len(scenes), worker_count * 2)):
            submit_one()
        try:
            while pending:
                completed, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
                for future in completed:
                    pending.pop(future)
                    accept(future.result())
                    submit_one()
        except BaseException:
            for future in pending:
                future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)
    bar.close()
    # 并行完成顺序不固定。按场景文件重新装载，保证输出行序和模型训练可复现。
    _, rows, dropped, extra = ckpt.load()
    if not rows:
        raise ValueError("No sequential samples kept")
    x_next_rows, hist_x_rows, hist_a_rows, hist_len_rows = _unpack_seq_extra(extra)
    fieldnames = [
        "scene_id", "run_id", "replicate_id", "time", "time_next", "strategy",
        "group_id", "kind", "intercepted_t", "intercepted_next", "intercept_delta",
        "reward_delta", "return_to_go", "episode_intercept_rate", "has_opportunity",
        "label_source", "intercept_rate", "leak_rate", "cost", "cost_eff", "utility",
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
            "run_id": np.asarray([row["run_id"] for row in rows]),
            "replicate_id": np.asarray([int(row["replicate_id"]) for row in rows], dtype=int),
            "time_next": np.asarray([float(row["time_next"]) for row in rows], dtype=float),
            "intercept_delta": np.asarray([float(row["intercept_delta"]) for row in rows], dtype=float),
            "reward_delta": np.asarray([float(row["reward_delta"]) for row in rows], dtype=float),
            "return_to_go": np.asarray([float(row["return_to_go"]) for row in rows], dtype=float),
            "episode_intercept_rate": np.asarray([float(row["episode_intercept_rate"]) for row in rows], dtype=float),
            "has_opportunity": np.asarray([float(row["has_opportunity"]) for row in rows], dtype=float),
            "done": np.asarray([float(row["done"]) for row in rows], dtype=float),
            "dataset_mode": np.asarray(["observed_sequential_transitions"]),
        },
    )
    summary = {
        "mode": "observed_sequential_transitions",
        "version": OBSERVED_SEQ_VERSION,
        "delta": delta,
        "max_intervals_per_run": max_mid_slices,
        "idle_keep_ratio": idle_keep_ratio,
        "gamma": GAMMA,
        "mix_lambda": MIX_LAMBDA,
        "window": WINDOW,
        "workers": worker_count,
        "n_rows": len(rows),
        "n_scenes": len(catalog),
        "n_runs": ingest.get("n_runs", sum(len(entries) for by_strategy in catalog.values() for entries in by_strategy.values())),
        "n_scene_strategy_pairs": ingest.get("n_scene_strategy_pairs", len(catalog)),
        "n_scenes_with_3_strategies": ingest.get("n_scenes_with_3_strategies", 0),
        "n_scenes_with_3_replicates": ingest.get("n_scenes_with_3_replicates", 0),
        "n_official_matched": ingest.get("n_official_matched", 0),
        "results_csv": ingest.get("results_csv"),
        "n_features": len(FEATURE_NAMES),
        "n_dropped": len(dropped),
        "n_load_errors": ingest.get("n_load_errors", 0),
        "runs": inventory_catalog(catalog),
        "kind_counts": {kind: sum(1 for row in rows if row["kind"] == kind) for kind in sorted({row["kind"] for row in rows})},
        "label_means": {name: float(np.mean([row[name] for row in rows])) for name in ("r_delta", "r_term", "mix", "r_shaped", "shaping")},
        "csv": str(csv_path),
        "checkpoint": str(ckpt.root),
        "checkpoint_preserved": True,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ckpt.mark_complete({"version": OBSERVED_SEQ_VERSION, "n_rows": len(rows)})
    return summary
