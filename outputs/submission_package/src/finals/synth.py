"""按赛方样本形态写出 CSV：一个目录 = 一种策略的一次仿真。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.finals.io import write_csv, write_family_parts
from src.finals.schema import SJZS_COLUMNS, STRATEGIES, TICKS_PER_FILE
from src.finals.simulate import clone_world, make_initial_world, run_episode


def generate_scene(scene_id: int, rng: np.random.Generator, n_blue: int, n_red: int, n_ticks: int, dt: float) -> dict:
    world = make_initial_world(rng, n_blue=n_blue, n_red=n_red, dt=dt, n_ticks=n_ticks)
    runs = {}
    for strategy in STRATEGIES:
        seed = int(rng.integers(0, 2**31 - 1))
        logs = run_episode(clone_world(world), strategy, np.random.default_rng(seed), n_ticks)
        runs[strategy] = {"seed": seed, "logs": logs}
    return {"scene_id": scene_id, "n_blue": world.n_blue_init, "n_red": n_red, "n_ticks": n_ticks, "dt": dt, "runs": runs}


def write_run_csv(directory: Path, logs, strategy: int, ticks_per_file: int) -> None:
    rhdl = [row for log in logs for row in log.rhdl]
    health = [row for log in logs for row in log.health]
    lj = [row for log in logs for row in log.lj]
    write_family_parts(directory, "Stu_ZZGLRHDL", rhdl, ticks_per_file=ticks_per_file)
    write_family_parts(directory, "HealthState", health, ticks_per_file=ticks_per_file)
    write_family_parts(directory, "Stu_ZZGLLJ", lj, ticks_per_file=ticks_per_file)
    write_csv(directory / "Stu_SJZS_0.csv", SJZS_COLUMNS, [{"S_LJCL": str(strategy)}])


def generate_dataset(
    output_dir: Path,
    n_scenes: int = 8,
    n_blue: int = 36,
    n_red: int = 8,
    n_ticks: int = 220,
    ticks_per_file: int = TICKS_PER_FILE,
    seed: int = 7,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    manifest = {
        "synthetic": True,
        "layout": "one folder = one strategy = one simulation, file suffix is time segment",
        "ticks_per_file": ticks_per_file,
        "n_scenes": n_scenes,
        "n_blue": n_blue,
        "n_red": n_red,
        "n_ticks": n_ticks,
        "seed": seed,
        "samples": [],
    }
    for scene_id in range(n_scenes):
        scene_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        n_blue_i = int(n_blue + scene_rng.integers(-6, 7))
        n_red_i = int(n_red + scene_rng.integers(-1, 2))
        packed = generate_scene(scene_id, scene_rng, max(24, n_blue_i), max(6, n_red_i), n_ticks, dt=1.0)
        for strategy, payload in packed["runs"].items():
            logs = payload["logs"]
            sample_dir = output_dir / f"sample_{scene_id:03d}_s{strategy}"
            write_run_csv(sample_dir, logs, strategy, ticks_per_file)
            last = logs[-1]
            n_obs = [len({row["TargetUnitID"] for row in log.rhdl}) for log in logs]
            meta = {
                "scene_id": scene_id,
                "strategy": strategy,
                "n_blue": packed["n_blue"],
                "n_red": packed["n_red"],
                "n_ticks_written": len(logs),
                "n_observed_min": int(min(n_obs) if n_obs else 0),
                "n_observed_max": int(max(n_obs) if n_obs else 0),
                "intercepted": last.intercepted,
                "leaked": last.leaked,
                "cost": last.cost,
            }
            (sample_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            manifest["samples"].append(meta)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
