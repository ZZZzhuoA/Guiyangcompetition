"""单切片训练集：有决策价值的当前态势 -> 三策略收益。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.finals.context import DecisionContext
from src.finals.dataset import candidate_times, inventory_runs, load_grouped_runs, missing_sample_hint, outcome_from_run, _sample_row
from src.finals.features import FEATURE_NAMES, dict_to_vector, extract_feature_dict, utility_from_outcome
from src.finals.progress import ProgressBar, SceneCheckpoint
from src.finals.schema import STRATEGIES
from src.finals.simulate import clone_world, run_episode, world_from_snapshot
from src.finals.snapshot import build_table_index, snapshot_from_tables

_KEY = (
    "n_alive",
    "n_can_links",
    "n_uncovered",
    "min_tclose",
    "min_range",
    "n_swarm_groups",
    "high_threat_uncovered",
)


def drop_reason(features: np.ndarray, prev: np.ndarray | None) -> str | None:
    idx = {name: i for i, name in enumerate(FEATURE_NAMES)}
    if features[idx["n_alive"]] < 1:
        return "no_targets"
    if prev is None:
        return None
    changed = False
    for name in _KEY:
        a, b = float(features[idx[name]]), float(prev[idx[name]])
        if name in {"n_alive", "n_can_links", "n_uncovered", "n_swarm_groups", "high_threat_uncovered"}:
            if abs(a - b) >= 1:
                changed = True
                break
        elif abs(a - b) >= 1.0:
            changed = True
            break
    if not changed:
        return "too_similar"
    return None


def _fork_slice(snap, n_blue: int, n_ticks: int, scene_id: int, time: float, seed: int, dt: float = 1.0) -> list[tuple[int, dict]]:
    world0 = world_from_snapshot(snap, n_blue_init=max(n_blue, len(snap.targets)), dt=dt)
    remaining = min(24, max(8, n_ticks // 6))
    out = []
    for strategy in STRATEGIES:
        rng = np.random.default_rng(seed + 1000 * scene_id + 17 * int(time) + strategy)
        logs = run_episode(clone_world(world0), strategy, rng, remaining)
        last = logs[-1]
        out.append((strategy, utility_from_outcome(last.intercepted, last.leaked, last.cost, world0.n_blue_init)))
    return out


_SCENE_ERRORS = (TypeError, ValueError, KeyError, IndexError, AttributeError)


def _process_scene(scene_id: int, by_strategy: dict, n_ticks: int, max_mid_slices: int, with_forks: bool, time_stride_s: float) -> tuple[list[dict], list[dict]]:
    ref = by_strategy.get(1) or next(iter(by_strategy.values()))
    index = build_table_index(ref["rhdl"], ref["health"], ref["lj"])
    times = candidate_times(ref["times"], stride_s=time_stride_s, dt=float(ref.get("dt", 0.5)))
    if not times:
        return [], [{"scene_id": scene_id, "time": None, "kind": "scene", "reason": "no_times"}]
    t0 = times[0]
    ctx = DecisionContext(horizon=max(float(n_ticks), float(ref["times"][-1])))
    snap0 = snapshot_from_tables(ref["rhdl"], ref["health"], ref["lj"], t0, ref["strategy"], index=index)
    feat_dict0 = extract_feature_dict(snap0, context=ctx)
    feat0 = dict_to_vector(feat_dict0)
    rows = []
    dropped = []
    reason = drop_reason(feat0, None)
    if reason:
        dropped.append({"scene_id": scene_id, "time": t0, "kind": "t0", "reason": reason})
    elif len(by_strategy) >= 2:
        for strategy, run in by_strategy.items():
            outcome = outcome_from_run(run)
            rows.append(_sample_row(scene_id, t0, strategy, feat0, outcome, f"scene{scene_id}|t0", "t0"))
    elif with_forks:
        for strategy, outcome in _fork_slice(
            snap0, ref["n_blue"], n_ticks, scene_id, t0, seed=17, dt=float(ref.get("dt", 1.0))
        ):
            rows.append(_sample_row(scene_id, t0, strategy, feat0, outcome, f"scene{scene_id}|t0", "t0"))
    else:
        dropped.append({"scene_id": scene_id, "time": t0, "kind": "t0", "reason": "single_strategy"})
    ctx.commit(snap0, feat_dict0, 0)
    prev = feat0
    if not with_forks:
        return rows, dropped
    kept_mid = 0
    for time in times[1:]:
        if kept_mid >= max_mid_slices:
            break
        snap = snapshot_from_tables(ref["rhdl"], ref["health"], ref["lj"], time, ref["strategy"], index=index)
        feat_dict = extract_feature_dict(snap, context=ctx)
        feat = dict_to_vector(feat_dict)
        reason = drop_reason(feat, prev)
        if reason:
            dropped.append({"scene_id": scene_id, "time": time, "kind": "fork", "reason": reason})
            continue
        prev = feat
        kept_mid += 1
        group_id = f"scene{scene_id}|slice{time:g}"
        for strategy, outcome in _fork_slice(
            snap, ref["n_blue"], n_ticks, scene_id, time, seed=17, dt=float(ref.get("dt", 1.0))
        ):
            rows.append(_sample_row(scene_id, time, strategy, feat, outcome, group_id, "fork"))
        ctx.commit(snap, feat_dict, int(ref["strategy"]))
    return rows, dropped


def build_slice_set(
    input_dir: Path,
    output_dir: Path,
    with_forks: bool = True,
    max_mid_slices: int = 4,
    n_ticks: int = 120,
    results_csv: Path | None = None,
    resume: bool = True,
    time_stride_s: float = 2.0,
) -> dict:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped, ingest = load_grouped_runs(input_dir, results_csv=results_csv)
    if not grouped:
        raise ValueError(missing_sample_hint(input_dir))
    ckpt = SceneCheckpoint(output_dir)
    if resume:
        done, rows, dropped, _ = ckpt.load()
    else:
        ckpt.clear()
        done, rows, dropped = set(), [], []
    if done:
        print(f"resume slice: skip {len(done)} finished scene(s)", flush=True)
    scenes = [(scene_id, by_strategy) for scene_id, by_strategy in sorted(grouped.items()) if scene_id not in done]
    bar = ProgressBar(len(scenes), prefix="slice")
    for scene_id, by_strategy in scenes:
        try:
            scene_rows, scene_dropped = _process_scene(
                scene_id, by_strategy, n_ticks, max_mid_slices, with_forks, time_stride_s
            )
        except _SCENE_FAIL as exc:
            scene_rows, scene_dropped = [], [{"scene_id": scene_id, "time": None, "kind": "scene", "reason": f"skip:{exc}"}]
        rows.extend(scene_rows)
        dropped.extend(scene_dropped)
        ckpt.save_scene(scene_id, scene_rows, scene_dropped)
        bar.update(extra=f"scene {scene_id}")
    bar.close()
    if not rows:
        raise ValueError("No slice samples kept")
    fieldnames = ["scene_id", "time", "strategy", "group_id", "kind", "intercept_rate", "leak_rate", "cost", "cost_eff", "utility", *FEATURE_NAMES]
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

    save_training_npz(output_dir, rows)
    ckpt.clear()
    summary = {
        "mode": "single_slice_scoring",
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
        "time_stride_s": time_stride_s,
        "n_load_errors": ingest.get("n_load_errors", 0),
        "runs": inventory_runs(grouped),
        "drop_reasons": {r: sum(1 for d in dropped if d["reason"] == r) for r in sorted({d["reason"] for d in dropped})},
        "kind_counts": {kind: sum(1 for row in rows if row["kind"] == kind) for kind in sorted({row["kind"] for row in rows})},
        "strategy_counts": {str(s): sum(1 for row in rows if row["strategy"] == s) for s in STRATEGIES},
        "conflicts": ingest.get("conflicts", []),
        "csv": str(csv_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
