"""真实观测切片集：每条实验轨迹独立构造区间转移。

一局实验从头到尾只执行一种策略。这里不再从策略 1 轨迹分叉模拟三种策略，
而是保存每条真实轨迹上的 (x_t, a, r_t, x_next)。三策略的可比标签在完成
场景级 train/test 切分后，由 matching.py 在相似态势中估计，避免数据泄漏。
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
from collections import Counter
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
    load_run_catalog,
    missing_sample_hint,
    time_at_or_after,
)
from src.finals.features import FEATURE_NAMES, dict_to_vector, extract_feature_dict
from src.finals.progress import ProgressBar, SceneCheckpoint
from src.finals.snapshot import build_table_index, snapshot_from_tables


OBSERVED_SLICE_VERSION = 3
_SCENE_ERRORS = (TypeError, ValueError, KeyError, IndexError, AttributeError, OSError)
_SLICE_WORKER_OFFICIAL = None
_SLICE_WORKER_CONFIG: tuple[int, int, float] | None = None

# 旧贯序建集仍从这里导入该过滤器；保留纯函数接口，避免显式运行
# `--with-seq` 时因模块兼容性直接失败。新的 observed_transitions 建集不使用它。
_DROP_KEY = (
    "n_alive", "n_can_links", "n_uncovered", "min_tclose",
    "min_range", "n_swarm_groups", "high_threat_uncovered",
)


def drop_reason(features: np.ndarray, prev: np.ndarray | None) -> str | None:
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    if float(features[idx["n_alive"]]) < 1.0:
        return "no_targets"
    if prev is None:
        return None
    for name in _DROP_KEY:
        current = float(features[idx[name]])
        before = float(prev[idx[name]])
        threshold = 1.0 if name in {
            "n_alive", "n_can_links", "n_uncovered",
            "n_swarm_groups", "high_threat_uncovered",
        } else 1.0
        if abs(current - before) >= threshold:
            return None
    return "too_similar"


def _idle_kept(scene_id: int, strategy: int, replicate: int, time: float, ratio: float) -> bool:
    """确定性抽样空白区间；断点续建和多进程结果保持一致。"""
    if ratio <= 0.0:
        return False
    if ratio >= 1.0:
        return True
    token = f"{int(scene_id)}|{int(strategy)}|{int(replicate)}|{float(time):.6f}".encode()
    bucket = int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), "little")
    return bucket % 1_000_000 < int(round(ratio * 1_000_000))


def _terminal_intercepted(run: dict) -> float:
    rate = run.get("official_intercept_rate")
    n_blue = max(1, int(run["n_blue"]))
    if rate is not None:
        return float(np.clip(float(rate), 0.0, 1.0)) * n_blue
    return float(max(0, int(run["intercepted"])))


def _process_run(
    scene_id: int,
    run: dict,
    run_number: int,
    delta: int,
    max_intervals: int,
    idle_keep_ratio: float,
) -> tuple[list[dict], list[dict], list[dict]]:
    strategy = int(run["strategy"])
    replicate = int(run.get("replicate_id") or run_number)
    run_id = str(run["directory"])
    index = build_table_index(run["rhdl"], run["health"], run["lj"])
    starts = decision_times(run["times"], float(delta))
    ctx = DecisionContext(horizon=max(float(run["times"][-1]), float(delta)))
    n_blue = max(1, int(run["n_blue"]))
    terminal_intercepted = _terminal_intercepted(run)
    episode_rate = terminal_intercepted / n_blue
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
        snap_next = snapshot_from_tables(
            run["rhdl"], run["health"], run["lj"], next_time, strategy, index=index
        )
        ctx_next = copy.deepcopy(ctx)
        ctx_next.commit(snap, feat_dict, strategy)
        feat_next = dict_to_vector(extract_feature_dict(snap_next, context=ctx_next))

        intercepted_t = _already_intercepted(run["health"], time)
        intercepted_next = _already_intercepted(run["health"], next_time)
        intercept_delta = max(0, intercepted_next - intercepted_t)
        return_to_go = max(0.0, terminal_intercepted - float(intercepted_t)) / n_blue
        reward_delta = float(intercept_delta) / n_blue
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
            dropped.append(
                {
                    "scene_id": scene_id,
                    "run_id": run_id,
                    "strategy": strategy,
                    "time": float(time),
                    "kind": "observed_idle",
                    "reason": "idle_subsample",
                }
            )
            ctx.commit(snap, feat_dict, strategy)
            continue

        cost = float(run.get("cost", 0.0))
        outcome = {
            "intercept_rate": float(return_to_go),
            "leak_rate": float(max(0.0, 1.0 - episode_rate)),
            "cost": cost,
            "cost_eff": float(episode_rate / (0.35 + cost)),
            "utility": float(return_to_go),
        }
        group_id = f"scene{scene_id}|run{run_number}|t{float(time):g}"
        row = _sample_row(scene_id, time, strategy, feat, outcome, group_id, kind)
        row.update(
            {
                "run_id": run_id,
                "replicate_id": replicate,
                "time_next": float(next_time),
                "intercepted_t": float(intercepted_t),
                "intercepted_next": float(intercepted_next),
                "intercept_delta": float(intercept_delta),
                "reward_delta": reward_delta,
                "return_to_go": float(return_to_go),
                "episode_intercept_rate": float(episode_rate),
                "has_opportunity": 1.0 if has_opportunity else 0.0,
                "done": 1.0 if abs(float(next_time) - float(run["times"][-1])) <= 1e-9 else 0.0,
                "label_source": str(run.get("label_source", "health")),
            }
        )
        rows.append(row)
        extra.append({"x_next": feat_next.tolist()})
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
                run_rows, run_dropped, run_extra = _process_run(
                    scene_id, run, run_number, delta, max_intervals, idle_keep_ratio
                )
                rows.extend(run_rows)
                dropped.extend(run_dropped)
                extra.extend(run_extra)
            except _SCENE_ERRORS as exc:
                errors.append({"directory": str(directory), "error": str(exc)})
                dropped.append(
                    {
                        "scene_id": scene_id,
                        "run_id": str(directory),
                        "strategy": int(strategy),
                        "time": None,
                        "kind": "run",
                        "reason": f"skip:{exc}",
                    }
                )
    return scene_id, rows, dropped, extra, errors


def _init_slice_worker(official, delta: int, max_intervals: int, idle_keep_ratio: float) -> None:
    global _SLICE_WORKER_OFFICIAL, _SLICE_WORKER_CONFIG
    _SLICE_WORKER_OFFICIAL = official
    _SLICE_WORKER_CONFIG = (int(delta), int(max_intervals), float(idle_keep_ratio))


def _slice_worker_task(scene: tuple[int, dict]):
    if _SLICE_WORKER_OFFICIAL is None or _SLICE_WORKER_CONFIG is None:
        raise RuntimeError("slice worker was not initialized")
    delta, max_intervals, idle_keep_ratio = _SLICE_WORKER_CONFIG
    return _build_catalog_scene(
        int(scene[0]), scene[1], _SLICE_WORKER_OFFICIAL,
        delta, max_intervals, idle_keep_ratio,
    )


def build_slice_set(
    input_dir: Path,
    output_dir: Path,
    with_forks: bool | None = None,
    max_mid_slices: int = 0,
    n_ticks: int = 120,
    results_csv: Path | None = None,
    resume: bool = True,
    time_stride_s: float | None = None,
    delta: int = 15,
    workers: int | None = None,
    idle_keep_ratio: float = 0.20,
) -> dict:
    """构造真实观测切片；旧参数保留兼容，但不再生成模拟分叉。"""
    del with_forks, time_stride_s, n_ticks
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = SceneCheckpoint(output_dir)
    version_path = ckpt.root / "builder.json"
    builder_config = {
        "version": OBSERVED_SLICE_VERSION,
        "mode": "observed_transitions",
        "delta": int(delta),
        "max_intervals_per_run": int(max_mid_slices),
        "idle_keep_ratio": float(idle_keep_ratio),
    }
    if not resume:
        ckpt.clear()
    elif ckpt.exists() and not version_path.exists():
        raise ValueError(
            f"{ckpt.root} 是旧版切片检查点；请改用新输出目录或加 --fresh"
        )
    if version_path.exists():
        stored_config = json.loads(version_path.read_text(encoding="utf-8"))
        version = stored_config.get("version")
        if int(version or 0) != OBSERVED_SLICE_VERSION:
            raise ValueError(f"checkpoint version {version} != {OBSERVED_SLICE_VERSION}; rerun with --fresh")
        mismatched = {
            key: (stored_config.get(key), value)
            for key, value in builder_config.items()
            if stored_config.get(key) != value
        }
        if mismatched:
            raise ValueError(
                f"checkpoint 构造参数与本次不一致: {mismatched}；"
                "请沿用原参数，或加 --fresh 从头构造"
            )
    ckpt.root.mkdir(parents=True, exist_ok=True)
    version_path.write_text(
        json.dumps(builder_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    catalog, official, ingest = load_run_catalog(
        input_dir, results_csv=results_csv,
        checkpoint_path=ckpt.root / "catalog.json", resume=resume,
    )
    if not catalog:
        raise ValueError(missing_sample_hint(input_dir))
    done, _, _, _ = ckpt.load() if resume else (set(), [], [], [])
    scenes = [(sid, entries) for sid, entries in sorted(catalog.items()) if sid not in done]
    worker_count = min(4, os.cpu_count() or 1) if workers is None else int(workers)
    if worker_count < 1:
        raise ValueError(f"workers must be >= 1, got {worker_count}")
    worker_count = min(worker_count, max(1, len(scenes)))
    print(
        f"observed slice catalog: {ingest.get('n_runs', 0)} run(s), {len(catalog)} scene(s); "
        f"workers: {worker_count}; checkpoint: {ckpt.root}", flush=True,
    )
    if done:
        print(f"resume observed slice: skip {len(done)} finished scene(s)", flush=True)
    bar = ProgressBar(len(scenes), prefix="slice-observed")

    def accept(result) -> None:
        scene_id, scene_rows, scene_dropped, scene_extra, errors = result
        if errors:
            ingest["n_load_errors"] = int(ingest.get("n_load_errors", 0)) + len(errors)
            ingest.setdefault("load_errors", []).extend(errors)
        ckpt.save_scene(scene_id, scene_rows, scene_dropped, scene_extra)
        bar.update(extra=f"scene {scene_id}")

    if worker_count == 1:
        for scene_id, entries in scenes:
            accept(_build_catalog_scene(
                scene_id, entries, official, delta, max_mid_slices, idle_keep_ratio
            ))
    elif scenes:
        pool = ProcessPoolExecutor(
            max_workers=worker_count, initializer=_init_slice_worker,
            initargs=(official, delta, max_mid_slices, idle_keep_ratio),
        )
        pending = {}
        scene_iter = iter(scenes)

        def submit_one() -> bool:
            try:
                scene = next(scene_iter)
            except StopIteration:
                return False
            future = pool.submit(_slice_worker_task, scene)
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

    _, rows, dropped, extra = ckpt.load()
    if not rows:
        raise ValueError("No observed slice samples kept")
    if len(extra) != len(rows):
        raise ValueError(f"observed checkpoint extra ({len(extra)}) != rows ({len(rows)})")

    fieldnames = [
        "scene_id", "run_id", "replicate_id", "time", "time_next", "strategy",
        "group_id", "kind", "intercepted_t", "intercepted_next", "intercept_delta",
        "reward_delta", "return_to_go", "episode_intercept_rate", "has_opportunity",
        "done", "label_source", "intercept_rate", "leak_rate", "cost", "cost_eff",
        "utility", *FEATURE_NAMES,
    ]
    csv_path = output_dir / "training_samples.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)
    drop_path = output_dir / "dropped_points.csv"
    with drop_path.open("w", encoding="utf-8-sig", newline="") as stream:
        drop_fields = ["scene_id", "run_id", "strategy", "time", "kind", "reason"]
        writer = csv.DictWriter(stream, fieldnames=drop_fields, extrasaction="ignore", lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(dropped)

    from src.finals.train import save_training_npz

    save_training_npz(
        output_dir, rows,
        extra={
            "x_next": np.asarray([item["x_next"] for item in extra], dtype=float),
            "run_id": np.asarray([row["run_id"] for row in rows]),
            "replicate_id": np.asarray([int(row["replicate_id"]) for row in rows], dtype=int),
            "time_next": np.asarray([float(row["time_next"]) for row in rows], dtype=float),
            "intercept_delta": np.asarray([float(row["intercept_delta"]) for row in rows], dtype=float),
            "reward_delta": np.asarray([float(row["reward_delta"]) for row in rows], dtype=float),
            "return_to_go": np.asarray([float(row["return_to_go"]) for row in rows], dtype=float),
            "episode_intercept_rate": np.asarray([float(row["episode_intercept_rate"]) for row in rows], dtype=float),
            "has_opportunity": np.asarray([float(row["has_opportunity"]) for row in rows], dtype=float),
            "done": np.asarray([float(row["done"]) for row in rows], dtype=float),
            "dataset_mode": np.asarray(["observed_transitions"]),
        },
    )
    kind_counts = Counter(str(row["kind"]) for row in rows)
    strategy_counts = Counter(int(row["strategy"]) for row in rows)
    summary = {
        "mode": "observed_transitions", "version": OBSERVED_SLICE_VERSION,
        "delta": int(delta), "max_intervals_per_run": int(max_mid_slices),
        "idle_keep_ratio": float(idle_keep_ratio), "n_rows": len(rows),
        "n_scenes": len(catalog), "n_runs": ingest.get("n_runs", 0),
        "n_features": len(FEATURE_NAMES), "n_dropped": len(dropped),
        "n_load_errors": ingest.get("n_load_errors", 0),
        "kind_counts": dict(kind_counts),
        "strategy_counts": {str(k): int(v) for k, v in sorted(strategy_counts.items())},
        "label_means": {
            "reward_delta": float(np.mean([float(row["reward_delta"]) for row in rows])),
            "return_to_go": float(np.mean([float(row["return_to_go"]) for row in rows])),
        },
        "results_csv": ingest.get("results_csv"), "runs": inventory_catalog(catalog),
        "csv": str(csv_path), "checkpoint": str(ckpt.root),
        "checkpoint_preserved": True,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    ckpt.mark_complete({"version": OBSERVED_SLICE_VERSION, "n_rows": len(rows)})
    return summary
