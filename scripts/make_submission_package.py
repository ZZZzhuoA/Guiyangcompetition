"""生成拷到赛方电脑的目录：推理入口 + 全部决赛训练代码。

赛方电脑上要用真实 CSV 重训，所以包里带 src/finals 全量和建集/训练脚本。
不带 experiments、初赛代码、合成数据、决赛样本。

  python scripts/make_submission_package.py --output outputs/submission_package
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TRAIN_SCRIPTS = (
    "inspect_finals_samples.py",
    "build_slice_set.py",
    "build_seq_set.py",
    "train_finals_model.py",
    "compare_finals_models.py",
    "diagnose_finals_models.py",
    "list_finals_models.py",
    "check_seq_inference.py",
    "run_finals_pipeline.py",
    "run_on_site.py",
    "make_finals_synth_data.py",
)

SUBMISSION_FILES = ("recommend.py", "requirements.txt", "README_submit.md")


def build(output: Path, model: Path) -> dict:
    output = Path(output)
    if output.exists():
        shutil.rmtree(output, ignore_errors=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "src" / "finals").mkdir(parents=True, exist_ok=True)
    (output / "scripts").mkdir(parents=True, exist_ok=True)
    (output / "submission_finals").mkdir(parents=True, exist_ok=True)
    (output / "outputs" / "finals_model").mkdir(parents=True, exist_ok=True)

    (output / "src" / "__init__.py").write_text("", encoding="utf-8")
    copied = ["src/__init__.py"]
    for src in sorted((ROOT / "src" / "finals").glob("*.py")):
        shutil.copy2(src, output / "src" / "finals" / src.name)
        copied.append(f"src/finals/{src.name}")
    for name in TRAIN_SCRIPTS:
        src = ROOT / "scripts" / name
        if not src.exists():
            raise FileNotFoundError(f"缺少训练脚本 {src}")
        shutil.copy2(src, output / "scripts" / name)
        copied.append(f"scripts/{name}")
    for name in SUBMISSION_FILES:
        src = ROOT / "submission_finals" / name
        if not src.exists():
            raise FileNotFoundError(f"缺少提交文件 {src}")
        shutil.copy2(src, output / "submission_finals" / name)
        copied.append(f"submission_finals/{name}")

    model = Path(model)
    if model.exists():
        shutil.copy2(model, output / "outputs" / "finals_model" / "policy.pkl")
        copied.append("outputs/finals_model/policy.pkl")

    total_bytes = sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    return {"files": copied, "n_files": len(copied), "total_kb": round(total_bytes / 1024, 1)}


def sample_payload() -> dict:
    """造一个最小合法 payload，用于隔离验证。字段与集成接口一致。"""
    return {
        "targets": [
            {
                "TargetUnitID": 101,
                "TargetUnitType": "BlueTarget",
                "B_l_jbz": False,
                "Ui_Zwx": 1,
                "D_X": 9000.0,
                "D_Y": 1200.0,
                "D_Z": 400.0,
                "D_XV": -220.0,
                "D_YV": -30.0,
                "D_ZV": -5.0,
                "Klj_list": [
                    {
                        "UI_FSChandle": 11,
                        "Str_HLMC": "Type1",
                        "B_DDKsslj": True,
                        "B_GPKsslj": False,
                        "B_HPMKsslj": False,
                        "B_GNJGKsslj": False,
                        "D_LJGL": 0.72,
                    }
                ],
            },
            {
                "TargetUnitID": 102,
                "TargetUnitType": "BlueTarget",
                "B_l_jbz": False,
                "Ui_Zwx": 3,
                "D_X": 14000.0,
                "D_Y": -2600.0,
                "D_Z": 1500.0,
                "D_XV": -180.0,
                "D_YV": 40.0,
                "D_ZV": -12.0,
                "Klj_list": [
                    {
                        "UI_FSChandle": 12,
                        "Str_HLMC": "Type2",
                        "B_DDKsslj": False,
                        "B_GPKsslj": True,
                        "B_HPMKsslj": False,
                        "B_GNJGKsslj": False,
                        "D_LJGL": 0.55,
                    }
                ],
            },
        ],
        "units": [
            {"UnitID": 11, "UnitType": "Type1", "UnitPos_x": 0.0, "UnitPos_y": 0.0, "UnitPos_z": 0.0},
            {"UnitID": 12, "UnitType": "Type2", "UnitPos_x": 500.0, "UnitPos_y": 200.0, "UnitPos_z": 0.0},
        ],
    }


VERIFY_SNIPPET = """
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from src.finals.slice_set import build_slice_set
from src.finals.train import train_policy
from submission_finals.recommend import recommend_verbose, reset

payload = json.loads(Path("payload.json").read_text(encoding="utf-8"))
reset()
seen = []
for k in range(4):
    out = recommend_verbose(payload, time=15.0 * k)
    seen.append(out)
    assert out["LJCL"] in (1, 2, 3), out
from submission_finals.recommend import get_policy
ctx = get_policy().cache.ctx
print(json.dumps({
    "ok": True,
    "model": seen[0]["model"],
    "ljcl_sequence": [o["LJCL"] for o in seen],
    "scores_last": seen[-1]["scores"],
    "cached_frames": len(ctx.hist_x),
    "last_ljcl": ctx.last_ljcl,
    "seen_ids": len(ctx.seen_ids),
    "train_import_ok": True,
}, ensure_ascii=False))
"""


def verify(package: Path) -> dict:
    """在只含包内文件的临时目录里跑一次，确认训练模块和推理都能 import。"""
    with tempfile.TemporaryDirectory() as tmp:
        sandbox = Path(tmp) / "pkg"
        shutil.copytree(package, sandbox)
        (sandbox / "payload.json").write_text(
            json.dumps(sample_payload(), ensure_ascii=False), encoding="utf-8"
        )
        (sandbox / "_verify.py").write_text(VERIFY_SNIPPET, encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, "_verify.py"],
            cwd=sandbox,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={"PATH": "", "SYSTEMROOT": "C:\\Windows", "PYTHONPATH": ""},
        )
    if proc.returncode != 0:
        raise RuntimeError(f"隔离验证失败，闭包不完整：\n{proc.stdout}\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the finals package to copy onto the organizer PC.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "submission_package")
    parser.add_argument("--model", type=Path, default=ROOT / "outputs" / "finals_model" / "policy.pkl")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    info = build(args.output, args.model)
    payload_path = args.output / "submission_finals" / "sample_payload.json"
    payload_path.write_text(json.dumps(sample_payload(), ensure_ascii=False, indent=2), encoding="utf-8")
    info["files"].append("submission_finals/sample_payload.json")
    info["n_files"] += 1

    result = {"package": str(args.output), **info}
    if not args.skip_verify:
        result["verify"] = verify(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
