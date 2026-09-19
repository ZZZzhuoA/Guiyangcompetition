"""整理真实贯序观测样本：历史窗口 + 实际策略 + 区间收益 + 下一状态。

换真实样本时 --input 指到 kemu6_data_72mb（叶子目录 _1/_2/_3 各是一个样本）：
  python scripts/build_seq_set.py --input kemu6_data_72mb --output outputs/finals_seq --delta 15 --workers 4
中断后再跑同一条会续接 `_ckpt/catalog.json` 和 `_ckpt/scenes/*.json`。
推倒重来加 --fresh；完整成功后仍保留 `_ckpt`，便于重新训练和复核。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.finals.seq_set import build_seq_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Build sequential scoring samples from run folders.")
    parser.add_argument("--input", type=Path, default=ROOT / "outputs" / "finals_synth")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "finals_seq")
    parser.add_argument("--delta", type=int, default=15)
    parser.add_argument("--n-ticks", type=int, default=120)
    parser.add_argument("--max-mid-slices", type=int, default=0, help="每条真实轨迹最多保留区间数；0=全部")
    parser.add_argument("--results", type=Path, default=None, help="官方拦截率 CSV；默认自动找")
    parser.add_argument("--workers", type=int, default=4, help="并行场景进程数；内存紧张可设为 2，调试设为 1")
    parser.add_argument("--idle-keep-ratio", type=float, default=0.20, help="无机会零收益区间保留比例")
    parser.add_argument("--fresh", action="store_true", help="忽略 _ckpt，从头建集")
    args = parser.parse_args()
    summary = build_seq_set(
        args.input,
        args.output,
        delta=args.delta,
        max_mid_slices=args.max_mid_slices,
        n_ticks=args.n_ticks,
        results_csv=args.results,
        resume=not args.fresh,
        workers=args.workers,
        idle_keep_ratio=args.idle_keep_ratio,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
