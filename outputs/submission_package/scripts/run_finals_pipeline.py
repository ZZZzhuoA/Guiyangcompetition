"""一键：造数 -> 整理单切片训练表 -> 训练模型。

复用链（换真实 CSV 时跳过造数）：
  scripts/make_finals_synth_data.py
  scripts/build_slice_set.py          # 主路径
  scripts/build_finals_training_set.py  # 旧密时刻路径，仍保留
  scripts/train_finals_model.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.finals.dataset import build_training_set
from src.finals.slice_set import build_slice_set
from src.finals.synth import generate_dataset
from src.finals.train import train_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the constructed-data finals pipeline.")
    parser.add_argument("--synth-dir", type=Path, default=ROOT / "outputs" / "finals_synth")
    parser.add_argument("--dataset-dir", type=Path, default=ROOT / "outputs" / "finals_slice")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "outputs" / "finals_model")
    parser.add_argument("--n-scenes", type=int, default=6)
    parser.add_argument("--n-blue", type=int, default=36)
    parser.add_argument("--n-red", type=int, default=8)
    parser.add_argument("--n-ticks", type=int, default=220)
    parser.add_argument("--ticks-per-file", type=int, default=200)
    parser.add_argument("--max-mid-slices", type=int, default=4)
    parser.add_argument("--no-forks", action="store_true")
    parser.add_argument("--dense", action="store_true", help="Use dense-tick builder instead of slice set.")
    parser.add_argument("--skip-synth", action="store_true", help="Reuse existing scene CSVs.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="真实样本根目录，例如 kemu6_data_72mb。其下为 1-200/200-400/400-600/600-end，每个场景的 _1/_2/_3 是三次重复实验。策略从结果表「策略」列读取。设置后自动跳过造数。",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--results", type=Path, default=None, help="官方拦截率 CSV；默认自动找")
    args = parser.parse_args()
    source = args.input or args.synth_dir
    synth = None
    if args.input is not None:
        if not source.exists():
            parser.error(f"Input dir does not exist: {source}")
    elif not args.skip_synth:
        if args.synth_dir.exists() and any(args.synth_dir.iterdir()):
            parser.error(f"Synth dir must be empty: {args.synth_dir}")
        synth = generate_dataset(
            args.synth_dir,
            n_scenes=args.n_scenes,
            n_blue=args.n_blue,
            n_red=args.n_red,
            n_ticks=args.n_ticks,
            ticks_per_file=args.ticks_per_file,
            seed=args.seed,
        )
    if args.dense:
        data = build_training_set(
            source,
            args.dataset_dir,
            with_forks=not args.no_forks,
            n_ticks=args.n_ticks,
        )
    else:
        data = build_slice_set(
            source,
            args.dataset_dir,
            with_forks=not args.no_forks,
            max_mid_slices=args.max_mid_slices,
            n_ticks=args.n_ticks,
            results_csv=args.results,
            resume=True,
        )
    model = train_policy(args.dataset_dir, args.model_dir)
    print(json.dumps({"synth": None if synth is None else {"n_scenes": synth["n_scenes"]}, "dataset": data, "model": model}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
