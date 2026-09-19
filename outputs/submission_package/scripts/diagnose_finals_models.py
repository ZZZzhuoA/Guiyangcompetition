"""读 compare.json + 数据集 summary.json，直接给出每个模型的判定和改进方向。

判读规则写在这里而不是留给人记：门槛集中在 THRESHOLDS，改了就是改了，不会两个人
按两套标准看同一份输出。

  python scripts/diagnose_finals_models.py --compare outputs/finals_model_ts/compare.json --dataset outputs/finals_seq
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

THRESHOLDS = {
    "latency_ms": 50.0,      # 硬门：赛方单次触发的时延预算
    "score_margin": 1e-4,    # 三策略打分差低于此值即视为并列
    "overfit_gap": 0.25,     # train_match - test_match 超过即判过拟合
    "min_groups": 10,        # 测试对照组少于此数，match 没有统计意义
    "collapse": 0.999,       # 预测几乎全落在一个策略上
}

BLOCKER = "BLOCK"
WARN = "WARN"
OK = "OK"


def _fmt(value, spec: str = ".4f") -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    return format(value, spec)


def diagnose_dataset(dataset_dir: Path | None) -> list[tuple[str, str]]:
    """数据层体检。标签没信号的话，后面所有模型指标都是噪声。"""
    findings = []
    if dataset_dir is None:
        return findings
    path = Path(dataset_dir) / "summary.json"
    if not path.exists():
        findings.append((WARN, f"没有 {path}，跳过数据层体检"))
        return findings
    summary = json.loads(path.read_text(encoding="utf-8"))
    means = summary.get("label_means") or {}
    for key, value in means.items():
        if abs(float(value)) < 1e-12:
            findings.append((BLOCKER, f"标签 {key} 均值为 0，这一列没有任何信号，学不出东西"))
    n_rows = summary.get("n_rows")
    n_dropped = summary.get("n_dropped")
    if n_rows and n_dropped and n_dropped > n_rows:
        findings.append((WARN, f"丢弃点({n_dropped}) 多于保留样本({n_rows})，检查 drop_reason 是否滤过头"))
    if n_rows is not None and n_rows < 200:
        findings.append((WARN, f"样本只有 {n_rows} 行，任何模型排序都会被方差主导"))
    if not findings:
        findings.append((OK, "数据层体检通过"))
    return findings


def diagnose_row(row: dict) -> list[tuple[str, str]]:
    """单个模型的判定。BLOCK 表示不该提交，WARN 表示有改进空间。"""
    out = []
    latency = row.get("latency_ms")
    if latency is not None and latency > THRESHOLDS["latency_ms"]:
        out.append((BLOCKER, f"延迟 {latency:.1f}ms 超过 {THRESHOLDS['latency_ms']:.0f}ms 预算，不可提交"))
    margin = row.get("test_score_margin")
    if margin is not None and margin < THRESHOLDS["score_margin"]:
        out.append((BLOCKER, f"三策略打分差仅 {margin:.2e}，没有区分能力，等价于固定策略"))
    collapse = row.get("test_collapse")
    if collapse is not None and collapse >= THRESHOLDS["collapse"]:
        out.append((WARN, "预测全部落在同一策略上（塌缩），先确认是标签没差异还是模型没学到"))
    train_match, test_match = row.get("train_match"), row.get("test_match")
    if train_match is not None and test_match is not None:
        gap = train_match - test_match
        if gap > THRESHOLDS["overfit_gap"]:
            out.append((WARN, f"train/test match 差 {gap:.2f}，过拟合，考虑降容量或砍特征维度"))
    groups = row.get("test_groups")
    tied_groups = int(row.get("test_tied_groups") or 0)
    if groups == 0:
        out.append((BLOCKER, "测试集没有收益有差异的策略对照组，无法评价模型"))
    elif groups is not None and groups < THRESHOLDS["min_groups"]:
        out.append((WARN, f"测试对照组只有 {groups} 个，match 没有统计意义"))
    if groups and tied_groups > int(groups):
        out.append((WARN, f"测试集中并列组 {tied_groups} 个，多于有效组 {groups} 个，标签区分度偏低"))
    lift = row.get("lift_vs_best_fixed")
    if lift is not None and lift <= 0.0:
        out.append((BLOCKER, f"整局拦截率相对最好的固定策略 lift={lift:+.4f}，打不过锁死一个策略"))
    if "episode_intercept_rate" not in row:
        out.append((WARN, "没有周期回放结果，不能据此定提交模型（跑 --with-replay）"))
    if not out:
        out.append((OK, "无阻塞项"))
    return out


def diagnose_global(rows: list[dict]) -> list[tuple[str, str]]:
    """跨模型的判定：指标本身有没有分辨力。"""
    findings = []
    if rows and all(int(r.get("test_groups") or 0) == 0 for r in rows):
        findings.append((BLOCKER, "测试集所有策略对照均为并列，没有可用于选模的有效组"))
    rates = [r["episode_intercept_rate"] for r in rows if r.get("episode_intercept_rate") is not None]
    if len(rates) >= 2 and max(rates) - min(rates) < 1e-9:
        findings.append(
            (
                BLOCKER,
                "所有模型的整局拦截率完全相同，回放对策略选择不敏感，"
                "selection_score 里权重最高的那一项没有分辨力，此时的排序不可信",
            )
        )
    margins = [r.get("test_score_margin") for r in rows if r.get("test_score_margin") is not None]
    if margins and max(margins) < THRESHOLDS["score_margin"]:
        findings.append((BLOCKER, "没有任何模型能把三个策略分开，问题在数据或奖励，不在模型选择"))
    oracle = {tuple(sorted((r.get("test_oracle_hist") or {}).items())) for r in rows}
    if len(oracle) == 1 and rows:
        hist = rows[0].get("test_oracle_hist") or {}
        nonzero = [k for k, v in hist.items() if v]
        if len(nonzero) == 1:
            findings.append(
                (BLOCKER, f"oracle 在测试集上恒为策略 {nonzero[0]}，match 退化成「有没有猜中这一个策略」")
            )
    if not findings:
        findings.append((OK, "指标具备分辨力"))
    return findings


def collect_blockers(compare_summary: dict, dataset_dir: Path | None) -> list[str]:
    """整份结果级 BLOCK：数据层、指标分辨力、以及 compare.json 的 chosen。

    某个落选候选延迟超限只在「逐模型判定」里标 BLOCK，不能把整份对比打成不可用。
    """
    rows = compare_summary.get("rows") or []
    hits = [msg for level, msg in diagnose_dataset(dataset_dir) if level == BLOCKER]
    hits += [msg for level, msg in diagnose_global(rows) if level == BLOCKER]
    chosen = compare_summary.get("chosen")
    if chosen:
        row = next((item for item in rows if item.get("name") == chosen), None)
        if row is not None:
            for level, msg in diagnose_row(row):
                if level == BLOCKER:
                    hits.append(f"{chosen}: {msg}")
    return hits


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge finals candidate models from compare.json.")
    parser.add_argument("--compare", type=Path, required=True, help="compare_models 产出的 compare.json")
    parser.add_argument("--dataset", type=Path, default=None, help="训练集目录，用于读 summary.json 做数据层体检")
    args = parser.parse_args()

    summary = json.loads(args.compare.read_text(encoding="utf-8"))
    rows = summary.get("rows", [])

    print("=" * 96)
    print(f"数据层体检  ({args.dataset or '未指定'})")
    print("=" * 96)
    for level, msg in diagnose_dataset(args.dataset):
        print(f"  [{level:5s}] {msg}")

    target = summary.get("target_diagnostics") or {}
    if target:
        print()
        print("=" * 96)
        print("训练目标")
        print("=" * 96)
        for split_name in ("train", "test"):
            item = target.get(split_name) or {}
            print(
                f"  {split_name:5s}: key={item.get('key')}  "
                f"strategy_mean={item.get('strategy_mean')}  winner_hist={item.get('winner_hist')}  "
                f"有效组={item.get('informative_groups')}  并列组={item.get('tied_groups')}"
            )

    print()
    print("=" * 96)
    print("指标分辨力")
    print("=" * 96)
    for level, msg in diagnose_global(rows):
        print(f"  [{level:5s}] {msg}")

    print()
    print("=" * 96)
    print("逐模型指标")
    print("=" * 96)
    header = (
        f"{'name':16s} {'trn_m':>6s} {'tst_m':>6s} {'grp':>4s} {'tie':>4s} {'collap':>6s} "
        f"{'margin':>10s} {'lat_ms':>7s} {'ep_rate':>8s} {'lift':>8s} {'score':>6s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['name']:16s} {_fmt(row.get('train_match'), '.3f'):>6s} {_fmt(row.get('test_match'), '.3f'):>6s} "
            f"{_fmt(row.get('test_groups'), 'd'):>4s} {_fmt(row.get('test_tied_groups'), 'd'):>4s} "
            f"{_fmt(row.get('test_collapse'), '.3f'):>6s} "
            f"{_fmt(row.get('test_score_margin'), '.2e'):>10s} {_fmt(row.get('latency_ms'), '.2f'):>7s} "
            f"{_fmt(row.get('episode_intercept_rate'), '.4f'):>8s} {_fmt(row.get('lift_vs_best_fixed'), '+.4f'):>8s} "
            f"{_fmt(row.get('selection_score'), '.3f'):>6s}"
        )

    print()
    print("=" * 96)
    print("逐模型判定")
    print("=" * 96)
    submittable = []
    for row in rows:
        findings = diagnose_row(row)
        worst = BLOCKER if any(level == BLOCKER for level, _ in findings) else (
            WARN if any(level == WARN for level, _ in findings) else OK
        )
        if worst == OK:
            submittable.append(row["name"])
        print(f"\n{row['name']}  ->  {worst}")
        for level, msg in findings:
            print(f"  [{level:5s}] {msg}")

    print()
    print("=" * 96)
    print("结论")
    print("=" * 96)
    blockers = collect_blockers(summary, args.dataset)
    if blockers:
        print("  数据层、指标分辨力或排序第一的模型有 BLOCK，不能用 chosen 换提交模型。必须先解决：")
        for msg in blockers:
            print(f"    - {msg}")
        print(f"  提交模型保持 default_submit = {summary.get('default_submit')} 不变。")
        if submittable:
            print(f"  无阻塞项的其他候选：{', '.join(submittable)}")
    elif submittable:
        print(f"  无阻塞项的候选：{', '.join(submittable)}")
        print(f"  指标排序第一：{summary.get('chosen')}")
        print("  确认这两者一致、且 lift 明显为正后，再改 registry.DEFAULT_SUBMIT。")
    else:
        print("  所有候选都有阻塞项，逐条看上面的判定。")


if __name__ == "__main__":
    main()
