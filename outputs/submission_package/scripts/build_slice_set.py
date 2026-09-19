"""整理真实观测切片：每条实验独立生成区间转移。

换真实样本时 --input 指到 kemu6_data_72mb（下面是 1-200/ 200-400/ 400-600/，叶子目录 _1/_2/_3 各是一个样本）：
  python scripts/build_slice_set.py --input kemu6_data_72mb --output outputs/finals_slice
中断后再跑同一条会续接 `_ckpt/catalog.json` 和 `_ckpt/scenes/*.json`。
推倒重来加 --fresh；完整成功后仍保留 `_ckpt`，以后修改标签或训练逻辑不必重读原始 CSV。
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
    parser.add_argument("--n-ticks", type=int, default=120, help="旧参数兼容，不控制真实轨迹长度")
    parser.add_argument("--delta", type=int, default=15, help="相邻真实转移的决策间隔（秒）")
    parser.add_argument("--max-mid-slices", type=int, default=0, help="每条实验最多保留的区间数；0=全部")
    parser.add_argument("--results", type=Path, default=None, help="官方拦截率 CSV；默认自动找")
    parser.add_argument("--workers", type=int, default=4, help="并行场景进程数；内存紧张设为2")
    parser.add_argument("--idle-keep-ratio", type=float, default=0.20, help="无目标机会且零收益区间保留比例")
    parser.add_argument("--fresh", action="store_true", help="忽略 _ckpt，从头建集")
    args = parser.parse_args()
    summary = build_slice_set(
        args.input,
        args.output,
        max_mid_slices=args.max_mid_slices,
        n_ticks=args.n_ticks,
        results_csv=args.results,
        resume=not args.fresh,
        delta=args.delta,
        workers=args.workers,
        idle_keep_ratio=args.idle_keep_ratio,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
