"""整理单切片训练样本：当前态势对三种策略打分。

换真实样本时 --input 指到 kemu6_data_72mb（下面是 1-200/ 200-400/ 400-600/，叶子目录 _1/_2/_3 各是一个样本）：
  python scripts/build_slice_set.py --input kemu6_data_72mb --output outputs/finals_slice
中断后再跑同一条会跳过已完成 scene。推倒重来加 --fresh。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.finals.slice_set import build_slice_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Build single-slice scoring samples from run folders.")
    parser.add_argument("--input", type=Path, default=ROOT / "outputs" / "finals_synth")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "finals_slice")
    parser.add_argument("--n-ticks", type=int, default=120)
    parser.add_argument("--max-mid-slices", type=int, default=4)
    parser.add_argument("--no-forks", action="store_true")
    parser.add_argument("--results", type=Path, default=None, help="官方拦截率 CSV；默认自动找")
    parser.add_argument("--time-stride", type=float, default=2.0, help="候选点时间步长（秒），默认 2s")
    parser.add_argument("--fresh", action="store_true", help="忽略 _ckpt，从头建集")
    args = parser.parse_args()
    summary = build_slice_set(
        args.input,
        args.output,
        with_forks=not args.no_forks,
        max_mid_slices=args.max_mid_slices,
        n_ticks=args.n_ticks,
        results_csv=args.results,
        resume=not args.fresh,
        time_stride_s=args.time_stride,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
