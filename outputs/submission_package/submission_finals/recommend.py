"""决赛集成入口：当前态势 -> LJCL ∈ {1,2,3}。

赛方每隔 Δ（15 或 20 秒）调用一次 recommend()。**必须在同一个进程里反复调用**：
模型的 14 维时序特征（新增/消失目标、距离与闭合时间变化、上次 LJCL）和序列模型
的历史窗口都靠跨触发缓存，进程一换缓存就清零，这些维度会全部退化成 0。
所以这里把 policy 按模型路径缓存成模块级单例，不要每次调用重新 load。

一局结束、换下一局之前调用 reset()，否则上一局的历史会串进新局。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.finals.infer import FinalsPolicy

_POLICY_CACHE: dict[str, FinalsPolicy] = {}


def _default_model() -> Path:
    return ROOT / "outputs" / "finals_model" / "policy.pkl"


def get_policy(model_path: Path | None = None) -> FinalsPolicy:
    """取（或首次创建）常驻 policy。缓存的是实例，跨触发的态势缓存就活在里面。"""
    path = Path(model_path) if model_path else _default_model()
    key = str(path)
    policy = _POLICY_CACHE.get(key)
    if policy is None:
        # 模型缺失时退化成规则托底，不抛异常：赛场上宁可给个合法策略也不能崩
        policy = FinalsPolicy.load(path) if path.exists() else FinalsPolicy(scorer=None)
        _POLICY_CACHE[key] = policy
    return policy


def reset(model_path: Path | None = None) -> None:
    """清掉跨触发缓存。每局开始前调用一次。"""
    if model_path is None:
        _POLICY_CACHE.clear()
        return
    _POLICY_CACHE.pop(str(Path(model_path)), None)


def recommend(payload: dict, model_path: Path | None = None, time: float = 0.0) -> int:
    """集成入口：输入当前态势，输出 LJCL ∈ {1,2,3}。

    time 是本次触发的仿真时刻（赛方 FZTime）。不传的话 time_frac 一直是 0，
    「打到第几段了」这一维就没用了。
    """
    return int(recommend_verbose(payload, model_path=model_path, time=time)["LJCL"])


def recommend_verbose(payload: dict, model_path: Path | None = None, time: float = 0.0) -> dict:
    """同 recommend，但连三个策略的打分一起返回，便于现场排查。

    三个分数几乎相同时说明模型没有区分能力，此时输出等价于固定策略。
    """
    policy = get_policy(model_path)
    out = policy.recommend_payload(payload, time=float(time))
    return {
        "LJCL": int(out["LJCL"]),
        "scores": [float(v) for v in out["scores"]],
        "model": getattr(policy.scorer, "name", "rule_fallback"),
    }


def payload_from_run_dir(run_dir: Path, time: float | None = None) -> dict:
    """把一份场景 CSV 目录转成接口 payload，仅用于本地自测。"""
    from src.finals.io import read_family_dir
    from src.finals.snapshot import snapshot_from_tables, snapshot_to_interface

    rhdl = read_family_dir(run_dir, "Stu_ZZGLRHDL")
    health = read_family_dir(run_dir, "HealthState")
    lj = read_family_dir(run_dir, "Stu_ZZGLLJ")
    times = sorted({float(row["FZTime"]) for row in rhdl})
    chosen = times[0] if time is None else float(time)
    snap = snapshot_from_tables(rhdl, health, lj, chosen)
    return snapshot_to_interface(snap)


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime LJCL recommendation demo.")
    parser.add_argument("--model", type=Path, default=_default_model())
    parser.add_argument("--run-dir", type=Path, help="Optional strategy_* folder used as a demo snapshot.")
    parser.add_argument("--payload-json", type=Path, help="Optional interface payload json.")
    parser.add_argument("--time", type=float, default=0.0)
    args = parser.parse_args()
    if args.payload_json:
        payload = json.loads(args.payload_json.read_text(encoding="utf-8"))
    elif args.run_dir:
        payload = payload_from_run_dir(args.run_dir, time=args.time)
    else:
        parser.error("Provide --payload-json or --run-dir")
    print(json.dumps(recommend_verbose(payload, args.model, time=args.time), ensure_ascii=False))


if __name__ == "__main__":
    main()
