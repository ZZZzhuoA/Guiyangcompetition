"""训练决赛三策略模型并导出 policy.pkl。

换真实样本时复用：
  python scripts/train_finals_model.py --dataset outputs/finals_slice --output outputs/finals_model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.finals.train import train_policy
from src.finals.registry import DEFAULT_SUBMIT, MODEL_CATALOG


def main() -> None:
    parser = argparse.ArgumentParser(description="Train finals strategy scorer from constructed samples.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "outputs" / "finals_slice")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "finals_model")
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--model", type=str, default=DEFAULT_SUBMIT, choices=list(MODEL_CATALOG))
    args = parser.parse_args()
    summary = train_policy(args.dataset, args.output, test_ratio=args.test_ratio, model_name=args.model)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
