"""列出 --input 下会被建集吃进去的样本目录。换真实 CSV 前先跑一遍核对。

  python scripts/inspect_finals_samples.py --input kemu6_data_72mb
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.finals.dataset import (
    _replicate_from_name,
    _scene_id_from_name,
    _strategy_from_name,
    discover_run_dirs,
    load_run_catalog,
    missing_sample_hint,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="List CSV sample folders that the builders will ingest.")
    parser.add_argument("--input", type=Path, default=ROOT / "kemu6_data_72mb")
    parser.add_argument("--results", type=Path, default=None, help="官方拦截率 CSV；默认在 --input 及其上一级自动找")
    parser.add_argument("--limit", type=int, default=12, help="runs 预览条数")
    args = parser.parse_args()
    dirs = discover_run_dirs(args.input)
    _, _, ingest = load_run_catalog(args.input, results_csv=args.results) if dirs else ({}, None, {})
    preview = []
    for directory in dirs[: max(0, args.limit)]:
        csvs = sorted(p.name for p in directory.glob("*.csv"))
        xlsx = sorted(p.name for p in directory.glob("*.xlsx"))
        preview.append(
            {
                "directory": str(directory),
                "scene_id": _scene_id_from_name(directory),
                "replicate_id": _replicate_from_name(directory),
                "strategy_from_name": _strategy_from_name(directory),
                "csv": csvs,
                "xlsx_ignored": xlsx,
            }
        )
    grouping_ok = bool(
        ingest.get("n_scenes_with_3_replicates") or ingest.get("n_scenes_with_3_strategies")
    )
    payload = {
        "input": str(args.input),
        "n_runs": len(dirs),
        "n_scenes": ingest.get("n_scenes", 0),
        "n_scene_strategy_pairs": ingest.get("n_scene_strategy_pairs", 0),
        "n_scenes_with_3_replicates": ingest.get("n_scenes_with_3_replicates", 0),
        "n_scenes_with_3_strategies": ingest.get("n_scenes_with_3_strategies", 0),
        "n_official_matched": ingest.get("n_official_matched", 0),
        "n_official_strategy_matched": ingest.get("n_official_strategy_matched", 0),
        "index_gaps": ingest.get("index_gaps", []),
        "results_csv": ingest.get("results_csv"),
        "conflicts": ingest.get("conflicts", []),
        "preview": preview,
        "hint": None if dirs else missing_sample_hint(args.input),
        "check": (
            None
            if not dirs or grouping_ok
            else "没有把同场景三次重复收在一起（n_scenes_with_3_replicates），也没有三策略对照（n_scenes_with_3_strategies）。`_1/_2/_3` 是重复实验，不是 LJCL。"
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not dirs:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
