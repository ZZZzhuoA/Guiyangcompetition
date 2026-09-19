"""列出注册表里全部候选模型及其关键属性。

  python scripts/list_finals_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.finals.registry import COMPARE_NAMES, COMPARE_SEQ_NAMES, DEFAULT_SUBMIT, MODEL_CATALOG, make_model


def main() -> None:
    print(f"total={len(MODEL_CATALOG)}  default_submit={DEFAULT_SUBMIT}")
    print(f"compare_single={len(COMPARE_NAMES)}  compare_seq={len(COMPARE_SEQ_NAMES)}")
    header = f"{'name':16s} {'family':10s} {'train':6s} {'label':10s} {'reward':10s} {'hist':6s} {'bellman':7s}"
    print(header)
    print("-" * len(header))
    for name, meta in MODEL_CATALOG.items():
        model = make_model(name)
        label = str(getattr(model, "label_key", "utility"))
        reward = str(getattr(model, "reward_key", None) or "-")
        hist = "yes" if getattr(model, "needs_history", False) else "-"
        bellman = "yes" if getattr(model, "needs_transition", False) else "-"
        train = "yes" if meta["trainable"] else "-"
        print(f"{name:16s} {meta['family']:10s} {train:6s} {label:10s} {reward:10s} {hist:6s} {bellman:7s}")


if __name__ == "__main__":
    main()
