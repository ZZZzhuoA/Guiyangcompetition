"""把场景 CSV 整理成训练样本表。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.finals.dataset import build_training_set


def main() -> None:
    parser = argparse.ArgumentParser(description="Build finals training rows from scene CSV folders.")
    parser.add_argument("--input", type=Path, default=ROOT / "outputs" / "finals_synth")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "finals_dataset")
    parser.add_argument("--n-ticks", type=int, default=220)
    parser.add_argument("--no-forks", action="store_true")
    args = parser.parse_args()
    summary = build_training_set(args.input, args.output, with_forks=not args.no_forks, n_ticks=args.n_ticks)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
