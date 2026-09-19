"""赛方电脑一键：发现 CSV -> 建真实切片集 -> 对比候选 -> 判读 -> 导出 policy.pkl。

真实样本只在赛方电脑上。工作目录必须是 submission_package/ 这一层。

  python scripts/run_on_site.py --input kemu6_data_72mb
  python scripts/run_on_site.py --input D:\\path\\to\\kemu6_data_72mb --delta 20
  python scripts/run_on_site.py --input kemu6_data_72mb --reuse-built --workers 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_SCRIPTS = str(ROOT / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from diagnose_finals_models import collect_blockers
from src.finals.dataset import discover_run_dirs, missing_sample_hint
from src.finals.registry import COMPARE_NAMES, COMPARE_SEQ_NAMES, DEFAULT_SUBMIT
from src.finals.seq_set import build_seq_set
from src.finals.slice_set import build_slice_set
from src.finals.train import _json_default, compare_models, train_policy


def _print(title: str, payload) -> None:
    print("=" * 88)
    print(title)
    print("=" * 88)
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))


def _blocked(compare_path: Path, dataset_dir: Path) -> list[str]:
    summary = json.loads(compare_path.read_text(encoding="utf-8"))
    return collect_blockers(summary, dataset_dir)


def _completed_dataset(
    dataset_dir: Path,
    expected_mode: str | None = None,
    expected_fields: dict | None = None,
) -> dict | None:
    """显式复用完整数据集；存在 _ckpt 时说明仍需续建，不能误用旧成品。"""
    summary_path = Path(dataset_dir) / "summary.json"
    samples_path = Path(dataset_dir) / "training_samples.npz"
    ckpt = Path(dataset_dir) / "_ckpt"
    if ckpt.exists() and not (ckpt / "complete.json").exists():
        return None
    if not summary_path.exists() or not samples_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if expected_mode is not None and summary.get("mode") != expected_mode:
        return None
    if expected_fields and any(summary.get(key) != value for key, value in expected_fields.items()):
        return None
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="On-site: build datasets, train all models, compare, export policy.pkl.")
    parser.add_argument("--input", type=Path, required=True, help="样本根目录，例如 kemu6_data_72mb")
    parser.add_argument("--delta", type=int, default=15)
    parser.add_argument("--test-ratio", type=float, default=0.25)
    parser.add_argument("--n-ticks", type=int, default=120)
    parser.add_argument("--with-replay", action="store_true", help="额外用本地玩具世界做周期回放；那不是赛方 CSV 测试")
    parser.add_argument("--with-seq", action="store_true", help="额外运行真实历史窗口贯序候选；默认只跑真实观测切片")
    parser.add_argument("--skip-seq", action="store_true", help="兼容旧命令：跳过贯序候选")
    parser.add_argument("--results", type=Path, default=None, help="官方拦截率 CSV；默认在 --input 及其上一级自动找")
    parser.add_argument("--fresh", action="store_true", help="忽略建集 _ckpt，从头建 slice/seq")
    parser.add_argument("--workers", type=int, default=4, help="slice/seq 默认并行进程数")
    parser.add_argument("--slice-workers", type=int, default=None, help="单独覆盖 slice 并行进程数")
    parser.add_argument("--seq-workers", type=int, default=None, help="单独覆盖 seq 并行进程数")
    parser.add_argument("--idle-keep-ratio", type=float, default=0.20, help="slice 空白零收益区间保留比例")
    parser.add_argument("--reuse-built", action="store_true", help="复用已完整生成的 slice/seq；有 _ckpt 的部分仍断点续建")
    parser.add_argument("--slice-dir", type=Path, default=ROOT / "outputs" / "finals_slice")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "outputs" / "finals_model")
    parser.add_argument("--seq-dir", type=Path, default=ROOT / "outputs" / "finals_seq")
    parser.add_argument("--seq-model-dir", type=Path, default=ROOT / "outputs" / "finals_model_ts")
    args = parser.parse_args()
    run_seq = bool(args.with_seq and not args.skip_seq)

    sample_root = args.input
    if not discover_run_dirs(sample_root):
        raise SystemExit(missing_sample_hint(sample_root))

    slice_dir = args.slice_dir
    seq_dir = args.seq_dir
    model_dir = args.model_dir
    seq_model_dir = args.seq_model_dir

    print(f"[1/6] inspect  {sample_root}")
    n_runs = len(discover_run_dirs(sample_root))
    print(f"      found {n_runs} CSV run folder(s)")

    print("[2/6] build slice set")
    slice_summary = (
        _completed_dataset(
            slice_dir,
            "observed_transitions",
            {"delta": args.delta, "idle_keep_ratio": args.idle_keep_ratio},
        )
        if args.reuse_built and not args.fresh
        else None
    )
    if slice_summary is None:
        slice_summary = build_slice_set(
            sample_root,
            slice_dir,
            n_ticks=args.n_ticks,
            results_csv=args.results,
            resume=not args.fresh,
            delta=args.delta,
            workers=args.slice_workers or args.workers,
            idle_keep_ratio=args.idle_keep_ratio,
        )
    else:
        print(f"      reuse completed dataset: {slice_dir / 'training_samples.npz'}")
    _print("slice set", {k: slice_summary[k] for k in ("n_rows", "n_scenes", "n_runs", "n_scenes_with_3_replicates", "n_scenes_with_3_strategies", "n_official_matched", "kind_counts") if k in slice_summary})

    seq_summary = None
    if run_seq:
        print("[3/6] build seq set")
        seq_summary = (
            _completed_dataset(
                seq_dir,
                expected_mode="observed_sequential_transitions",
                expected_fields={"delta": args.delta, "idle_keep_ratio": args.idle_keep_ratio},
            )
            if args.reuse_built and not args.fresh
            else None
        )
        if seq_summary is None:
            seq_summary = build_seq_set(
                sample_root,
                seq_dir,
                delta=args.delta,
                n_ticks=args.n_ticks,
                results_csv=args.results,
                resume=not args.fresh,
                workers=args.seq_workers or args.workers,
                idle_keep_ratio=args.idle_keep_ratio,
            )
        else:
            print(f"      reuse completed dataset: {seq_dir / 'training_samples.npz'}")
        _print("seq set", {k: seq_summary[k] for k in ("n_rows", "n_scenes", "n_runs", "n_scenes_with_3_replicates", "n_scenes_with_3_strategies", "n_official_matched", "label_means") if k in seq_summary})
    else:
        print("[3/6] skip seq set")

    print(f"[4/6] train+compare {len(COMPARE_NAMES)} single-slice models")
    slice_compare = compare_models(
        slice_dir,
        model_dir,
        names=COMPARE_NAMES,
        test_ratio=args.test_ratio,
        with_replay=args.with_replay,
        delta=float(args.delta),
    )
    _print("slice compare", {"chosen": slice_compare.get("chosen"), "split": slice_compare.get("split"), "rows": [
        {k: row.get(k) for k in ("name", "test_policy_value", "test_mean_regret", "test_lift_vs_best_fixed_matched", "test_match", "test_groups", "test_score_margin", "latency_ms", "selection_score")}
        for row in slice_compare["rows"]
    ]})

    seq_compare = None
    if run_seq:
        print(f"[5/6] train+compare {len(COMPARE_SEQ_NAMES)} sequential models")
        seq_compare = compare_models(
            seq_dir,
            seq_model_dir,
            names=COMPARE_SEQ_NAMES,
            test_ratio=args.test_ratio,
            with_replay=args.with_replay,
            delta=float(args.delta),
        )
        _print("seq compare", {"chosen": seq_compare.get("chosen"), "split": seq_compare.get("split"), "rows": [
            {k: row.get(k) for k in ("name", "test_policy_value", "test_mean_regret", "test_lift_vs_best_fixed_matched", "test_match", "test_groups", "test_score_margin", "latency_ms", "selection_score")}
            for row in seq_compare["rows"]
        ]})
    else:
        print("[5/6] skip seq compare")

    print(f"[6/6] export DEFAULT_SUBMIT={DEFAULT_SUBMIT} -> policy.pkl")
    export = train_policy(slice_dir, model_dir, test_ratio=args.test_ratio, model_name=DEFAULT_SUBMIT)

    slice_blockers = _blocked(model_dir / "compare.json", slice_dir)
    seq_blockers: list[str] = []
    if seq_compare is not None:
        seq_blockers = _blocked(seq_model_dir / "compare.json", seq_dir)

    result = {
        "n_runs": n_runs,
        "slice": str(slice_dir / "training_samples.npz"),
        "seq": None if seq_summary is None else str(seq_dir / "training_samples.npz"),
        "split": str(slice_dir / "split.json"),
        "slice_compare": str(model_dir / "compare.json"),
        "seq_compare": None if seq_compare is None else str(seq_model_dir / "compare.json"),
        "policy": export["model_path"],
        "policy_model": DEFAULT_SUBMIT,
        "blockers": slice_blockers,
        "seq_blockers": seq_blockers,
        "next": (
            "数据层、指标分辨力或 slice chosen 有 BLOCK，保持 policy.pkl=rf_pair，不要按 chosen 换模型。"
            if slice_blockers
            else f"slice chosen 无 BLOCK。若要改提交模型：python scripts/train_finals_model.py --dataset outputs/finals_slice --output outputs/finals_model --model {slice_compare.get('chosen')}"
        ),
    }
    _print("DONE", result)
    print("判读明细：")
    print(f"  python scripts/diagnose_finals_models.py --compare {model_dir / 'compare.json'} --dataset {slice_dir}")
    if seq_compare is not None:
        print(f"  python scripts/diagnose_finals_models.py --compare {seq_model_dir / 'compare.json'} --dataset {seq_dir}")


if __name__ == "__main__":
    main()
