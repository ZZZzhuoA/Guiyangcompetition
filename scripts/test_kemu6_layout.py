"""赛方 kemu6 `_1/_2/_3` 是同一策略的三次重复，不是 LJCL。"""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.finals.dataset import (
    _replicate_from_name,
    _scene_id_from_name,
    _strategy_from_name,
    find_results_csv,
    inventory_index_gaps,
    load_grouped_runs,
    load_official_rates,
    outcome_from_run,
)
from src.finals.io import write_csv
from src.finals.schema import RHDL_COLUMNS, SJZS_COLUMNS

SCENE = "反无大赛-6方向-终版_6方向-1km数据_72km-1km-6方向"
RHDL_ROW = {
    "TargetUnitID": 11,
    "TargetUnitName": "b1",
    "B_Sgzmb": True,
    "B_l_jbz": True,
    "B_Sdk": True,
    "B_QMBflag": False,
    "Uc_XXYLY": 1,
    "Uc_XXYHandle": 1,
    "D_Bsm": 0.0,
    "D_Esm": 0.0,
    "Ui_Zwx": 1,
    "D_XDPm": 0.0,
    "D_XDTm": 0.0,
    "D_XDRm": 5.0,
    "D_R": 5.0,
    "D_E": 0.0,
    "D_B": 0.0,
    "D_RV": -0.1,
    "D_EV": 0.0,
    "D_BV": 0.0,
    "D_X": 5.0,
    "D_Y": 0.0,
    "D_Z": 0.1,
    "D_XV": -0.1,
    "D_YV": 0.0,
    "D_ZV": 0.0,
    "FZTime": 0.0,
    "DetectTime": 0.0,
}


def _leaf(batch: str, index: int, replicate: int) -> Path:
    return Path("kemu6_data_72mb") / batch / f"{SCENE}-{batch}_{index:03d}" / f"_{replicate}"


class LayoutTests(unittest.TestCase):
    def test_replicate_is_not_strategy(self):
        self.assertIsNone(_strategy_from_name(_leaf("1-200", 1, 1)))
        self.assertIsNone(_strategy_from_name(_leaf("1-200", 1, 3)))
        self.assertEqual(_replicate_from_name(_leaf("1-200", 1, 1)), 1)
        self.assertEqual(_replicate_from_name(_leaf("1-200", 1, 3)), 3)
        self.assertEqual(_strategy_from_name(Path("outputs/finals_synth/sample_000_s2")), 2)
        self.assertIsNone(_strategy_from_name(Path("决赛/样本1")))

    def test_scene_ids_unique_across_batches(self):
        self.assertEqual(_scene_id_from_name(_leaf("1-200", 1, 1)), 1)
        self.assertEqual(_scene_id_from_name(_leaf("1-200", 1, 2)), 1)
        self.assertEqual(_scene_id_from_name(_leaf("1-200", 1, 3)), 1)
        self.assertEqual(_scene_id_from_name(_leaf("1-200", 2, 1)), 2)
        self.assertEqual(_scene_id_from_name(_leaf("1-200", 200, 3)), 200)
        self.assertEqual(_scene_id_from_name(_leaf("200-400", 1, 1)), 201)
        self.assertEqual(_scene_id_from_name(_leaf("200-400", 200, 2)), 400)
        self.assertEqual(_scene_id_from_name(_leaf("400-600", 1, 3)), 401)
        self.assertEqual(_scene_id_from_name(_leaf("400-600", 200, 1)), 600)
        self.assertEqual(_scene_id_from_name(_leaf("600-end", 1, 1)), 601)
        self.assertEqual(_scene_id_from_name(_leaf("600-end", 1, 3)), 601)
        self.assertEqual(_scene_id_from_name(_leaf("600-end", 40, 2)), 640)
        nested = Path("kemu6_data_72mb") / "1-200" / f"{SCENE}-600-end_001" / "_1"
        nested264 = Path("kemu6_data_72mb") / "1-200" / f"{SCENE}-600-end_264" / "_3"
        self.assertEqual(_scene_id_from_name(nested), 601)
        self.assertEqual(_scene_id_from_name(nested264), 864)
        self.assertNotEqual(_scene_id_from_name(nested), _scene_id_from_name(_leaf("1-200", 1, 1)))

    def test_synth_and_template_names_unchanged(self):
        self.assertEqual(_scene_id_from_name(Path("outputs/finals_synth/sample_000_s1")), 0)
        self.assertEqual(_scene_id_from_name(Path("outputs/finals_synth/sample_003_s2")), 3)
        self.assertEqual(_scene_id_from_name(Path("决赛/样本1")), 1)

    def test_official_rate_lookup_by_suffix(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            csv_path = root / "文件实验数据路径与结果_72mb.csv"
            rel = _leaf("1-200", 1, 2).as_posix()
            with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["路径", "拦截率"])
                writer.writerow([rel.replace("/", "\\"), "0.73"])
                writer.writerow([_leaf("200-400", 1, 1).as_posix(), "81%"])
            table = load_official_rates(csv_path)
            self.assertEqual(table.lookup(root / rel), 0.73)
            self.assertEqual(table.lookup(Path("D:/data") / _leaf("200-400", 1, 1)), 0.81)

    def test_three_replicates_same_strategy_mean_rate(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "kemu6_data_72mb"
            scene_dir = root / "1-200" / f"{SCENE}-1-200_001"
            rates = {}
            for replicate, rate in ((1, 0.40), (2, 0.70), (3, 0.55)):
                leaf = scene_dir / f"_{replicate}"
                leaf.mkdir(parents=True)
                write_csv(leaf / "Stu_ZZGLRHDL_0.csv", RHDL_COLUMNS, [RHDL_ROW])
                write_csv(leaf / "Stu_SJZS_0.csv", SJZS_COLUMNS, [{"S_LJCL": "2"}])
                rates[leaf.relative_to(root).as_posix()] = rate
            results = root / "文件实验数据路径与结果_72mb.csv"
            with results.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["文件路径", "拦截率"])
                for path, rate in rates.items():
                    writer.writerow([path, rate])
            self.assertEqual(find_results_csv(root), results)
            grouped, meta = load_grouped_runs(root)
            self.assertEqual(meta["n_runs"], 3)
            self.assertEqual(meta["n_scenes"], 1)
            self.assertEqual(meta["n_scenes_with_3_strategies"], 0)
            self.assertEqual(meta["n_scenes_with_3_replicates"], 1)
            self.assertEqual(meta["n_official_matched"], 3)
            self.assertEqual(set(grouped[1]), {2})
            self.assertEqual(len(grouped[1][2]["replicates"]), 3)
            outcome = outcome_from_run(grouped[1][2])
            self.assertAlmostEqual(outcome["intercept_rate"], (0.40 + 0.70 + 0.55) / 3)

    def test_strategy_from_results_type_names(self):
        from src.finals.schema import parse_strategy_label

        self.assertEqual(parse_strategy_label("闭合时间最短"), 1)
        self.assertEqual(parse_strategy_label("拦截性能最佳"), 2)
        self.assertEqual(parse_strategy_label("综合效益最优"), 3)
        self.assertEqual(parse_strategy_label("3"), 3)
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "kemu6_data_72mb"
            scene_dir = root / "600-end" / f"{SCENE}-600-end_001"
            leaf = scene_dir / "_2"
            leaf.mkdir(parents=True)
            write_csv(leaf / "Stu_ZZGLRHDL_0.csv", RHDL_COLUMNS, [RHDL_ROW])
            write_csv(leaf / "Stu_SJZS_0.csv", SJZS_COLUMNS, [{"S_LJCL": "1"}])
            results = root / "文件实验数据路径与结果_72mb.csv"
            scene_rel = scene_dir.relative_to(root).as_posix()
            with results.open("w", encoding="gbk", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["实验分组", "实验名称", "实验次数", "总拦截率", "策略"])
                writer.writerow(["说明", "路径", "重复", "单位", "类型"])
                writer.writerow(["", "", "", "", ""])
                writer.writerow(["600-end", scene_rel, "_1", "0.40", "综合效益最优"])
                writer.writerow(["600-end", "", "_2", "0.50", ""])
                writer.writerow(["600-end", "", "_3", "0.60", ""])
            grouped, meta = load_grouped_runs(root)
            self.assertEqual(_scene_id_from_name(leaf), 601)
            self.assertEqual(meta["n_official_strategy_matched"], 1)
            self.assertEqual(set(grouped[601]), {3})
            self.assertEqual(grouped[601][3]["strategy_source"], "official_csv")
            self.assertNotEqual(grouped[601][3]["strategy"], 1)
            self.assertAlmostEqual(grouped[601][3]["official_intercept_rate"], 0.50)

    def test_missing_indices_are_gaps_not_renumbered(self):
        dirs = [
            _leaf("1-200", 178, 1),
            _leaf("1-200", 178, 2),
            _leaf("1-200", 178, 3),
            _leaf("1-200", 180, 1),
            _leaf("1-200", 180, 2),
            _leaf("1-200", 180, 3),
            _leaf("400-600", 179, 2),
            _leaf("400-600", 179, 3),
            _leaf("400-600", 180, 1),
        ]
        self.assertEqual(_scene_id_from_name(_leaf("1-200", 180, 1)), 180)
        self.assertEqual(_scene_id_from_name(_leaf("400-600", 180, 1)), 580)
        gaps = {row["batch"]: row for row in inventory_index_gaps(dirs)}
        self.assertEqual(gaps["1-200"]["missing_scene_indices"], [179])
        incomplete = {item["index"]: item["missing"] for item in gaps["400-600"]["incomplete_replicates"]}
        self.assertEqual(incomplete[179], ["_1"])
        self.assertEqual(incomplete[180], ["_2", "_3"])


if __name__ == "__main__":
    unittest.main()
