"""自检：序列模型在实时循环里能不能正确累积历史窗口，延迟是否还在预算内。

  python scripts/check_seq_inference.py --model outputs/finals_model_ts/seq_recur_fqi.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.finals.infer import FinalsPolicy
from src.finals.simulate import make_initial_world, observe, step
from src.finals.snapshot import snapshot_from_tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test sequence models in the periodic loop.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--delta", type=int, default=15)
    parser.add_argument("--triggers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    with args.model.open("rb") as stream:
        scorer = pickle.load(stream)
    policy = FinalsPolicy(scorer=scorer)
    print(f"model={args.model.name} needs_history={getattr(scorer, 'needs_history', False)}")

    rng = np.random.default_rng(args.seed)
    world = make_initial_world(rng, n_blue=24, n_red=7, dt=1.0, n_ticks=args.delta * args.triggers + 10)
    log = observe(world)
    health = list(log.health)
    worst_ms = 0.0
    for k in range(args.triggers):
        snapshot = snapshot_from_tables(log.rhdl, health, log.lj, log.time)
        t0 = time.perf_counter()
        out = policy.recommend_snapshot(snapshot)
        elapsed = 1000.0 * (time.perf_counter() - t0)
        worst_ms = max(worst_ms, elapsed)
        ctx = policy.cache.ctx
        scores = ", ".join(f"{v:.4f}" for v in out["scores"])
        print(
            f"trigger {k}: t={log.time:6.1f} LJCL={out['LJCL']} "
            f"cached_frames={len(ctx.hist_x)} action_history={ctx.hist_a} "
            f"{elapsed:.2f}ms scores=[{scores}]"
        )
        for _ in range(args.delta):
            log = step(world, out["LJCL"], rng)
            health.extend(log.health)
            if not world.pending and not any(b.alive and b.observed for b in world.blues):
                break
    print(f"worst latency {worst_ms:.2f}ms (budget 50ms)")


if __name__ == "__main__":
    main()
