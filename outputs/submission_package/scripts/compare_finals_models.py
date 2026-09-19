"""对比多模型并按指标排序。有真实数据后复用同一入口。

  python scripts/compare_finals_models.py --dataset outputs/finals_slice --output outputs/finals_model
  python scripts/compare_finals_models.py --with-replay
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.finals.registry import COMPARE_NAMES, COMPARE_SEQ_NAMES
from src.finals.train import compare_models, _json_default


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare finals candidate models.")
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "finals_model")
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--seq", action="store_true", help="Compare sequential models on outputs/finals_seq.")
    parser.add_argument("--with-replay", action="store_true")
    parser.add_argument("--replay-scenes", type=int, default=3)
    parser.add_argument("--replay-ticks", type=int, default=60)
    parser.add_argument("--delta", type=float, default=15.0)
    args = parser.parse_args()
    dataset = args.dataset or (ROOT / "outputs" / "finals_seq" if args.seq else ROOT / "outputs" / "finals_slice")
    names = args.models or list(COMPARE_SEQ_NAMES if args.seq else COMPARE_NAMES)
    summary = compare_models(
        dataset,
        args.output,
        names=tuple(names),
        test_ratio=args.test_ratio,
        with_replay=args.with_replay,
        replay_scenes=args.replay_scenes,
        replay_ticks=args.replay_ticks,
        delta=args.delta,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
