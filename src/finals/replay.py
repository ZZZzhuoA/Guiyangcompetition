"""周期回放：按 Δ 切策略，对比固定 1/2/3 与 Q 策略的整局拦截率。"""

from __future__ import annotations

import numpy as np

from src.finals.infer import FinalsPolicy
from src.finals.schema import STRATEGIES
from src.finals.simulate import clone_world, make_initial_world, observe, step
from src.finals.snapshot import snapshot_from_tables


def _snapshot_from_log(log, health_acc: list[dict]):
    return snapshot_from_tables(log.rhdl, health_acc, log.lj, log.time)


def run_episode_periodic(world, choose, rng: np.random.Generator, n_ticks: int, delta: float = 15.0):
    logs = [observe(world)]
    health_acc = list(logs[0].health)
    last_t = float(world.time)
    strategy = int(choose(_snapshot_from_log(logs[0], health_acc)))
    decisions = [(float(world.time), strategy)]
    for _ in range(n_ticks):
        if float(world.time) - last_t >= delta - 1e-9:
            strategy = int(choose(_snapshot_from_log(logs[-1], health_acc)))
            last_t = float(world.time)
            decisions.append((float(world.time), strategy))
        logs.append(step(world, strategy, rng))
        health_acc.extend(logs[-1].health)
        alive = any(b.alive and b.observed for b in world.blues)
        if not world.pending and not alive:
            break
    last = logs[-1]
    n_blue = max(1, world.n_blue_init)
    return {
        "intercepted": last.intercepted,
        "leaked": last.leaked,
        "cost": last.cost,
        "intercept_rate": last.intercepted / n_blue,
        "leak_rate": last.leaked / n_blue,
        "n_decisions": len(decisions),
        "decisions": decisions,
        "n_ticks": len(logs),
    }


def evaluate_periodic_policies(
    scorers: dict[str, object],
    n_scenes: int = 4,
    n_ticks: int = 80,
    delta: float = 15.0,
    seed: int = 11,
    n_blue: int = 28,
    n_red: int = 7,
) -> dict:
    rng = np.random.default_rng(seed)
    names = [f"fixed_{s}" for s in STRATEGIES] + list(scorers)
    totals = {name: [] for name in names}
    for scene in range(n_scenes):
        world0 = make_initial_world(
            np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
            n_blue=n_blue,
            n_red=n_red,
            dt=1.0,
            n_ticks=n_ticks,
        )
        for strategy in STRATEGIES:
            packed = run_episode_periodic(
                clone_world(world0),
                lambda snapshot, s=strategy: s,
                np.random.default_rng(scene * 17 + strategy),
                n_ticks,
                delta=delta,
            )
            totals[f"fixed_{strategy}"].append(packed["intercept_rate"])
        for name, scorer in scorers.items():
            policy = FinalsPolicy(scorer=scorer)
            packed = run_episode_periodic(
                clone_world(world0),
                lambda snapshot, p=policy: int(p.recommend_snapshot(snapshot)["LJCL"]),
                np.random.default_rng(scene * 17 + 99),
                n_ticks,
                delta=delta,
            )
            totals[name].append(packed["intercept_rate"])
    summary = {}
    best_fixed = max(float(np.mean(totals[f"fixed_{s}"])) for s in STRATEGIES)
    for name, values in totals.items():
        mean_rate = float(np.mean(values))
        summary[name] = {
            "episode_intercept_rate": mean_rate,
            "lift_vs_best_fixed": None if name.startswith("fixed_") else mean_rate - best_fixed,
            "n_scenes": n_scenes,
            "delta": delta,
        }
    summary["_best_fixed"] = best_fixed
    return summary
