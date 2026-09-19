"""构造决赛合成场景 CSV。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.finals.synth import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Construct synthetic finals scenes in organizer CSV format.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "finals_synth")
    parser.add_argument("--n-scenes", type=int, default=6)
    parser.add_argument("--n-blue", type=int, default=36)
    parser.add_argument("--n-red", type=int, default=8)
    parser.add_argument("--n-ticks", type=int, default=220)
    parser.add_argument("--ticks-per-file", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"Output must be empty: {args.output}")
    manifest = generate_dataset(
        args.output,
        n_scenes=args.n_scenes,
        n_blue=args.n_blue,
        n_red=args.n_red,
        n_ticks=args.n_ticks,
        ticks_per_file=args.ticks_per_file,
        seed=args.seed,
    )
    print(json.dumps({"n_scenes": manifest["n_scenes"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
