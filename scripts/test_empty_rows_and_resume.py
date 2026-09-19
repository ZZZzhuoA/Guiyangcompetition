"""空行 / None 单元格不崩；slice 建集可断点续传。"""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.finals.dataset import candidate_times, decision_times, load_grouped_runs
from src.finals.io import read_csv
from src.finals.progress import SceneCheckpoint
from src.finals.schema import RHDL_COLUMNS, SJZS_COLUMNS, as_bool, as_float, as_int
from src.finals.slice_set import _process_scene, build_slice_set
from src.finals.snapshot import snapshot_from_interface, snapshot_from_tables
from src.finals.io import write_csv

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
    "FZTime": 1.0,
    "DetectTime": 1.0,
}


def _write_leaf(leaf: Path, strategy: int, extra_blank: bool = False) -> None:
    leaf.mkdir(parents=True)
    path = leaf / "Stu_ZZGLRHDL_0.csv"
    write_csv(path, RHDL_COLUMNS, [RHDL_ROW])
    if extra_blank:
        with path.open("a", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\r\n")
            writer.writerow([])
            writer.writerow([""] * len(RHDL_COLUMNS))
    write_csv(leaf / "Stu_SJZS_0.csv", SJZS_COLUMNS, [{"S_LJCL": str(strategy)}])


class EmptyRowTests(unittest.TestCase):
    def test_as_float_none_and_blank(self):
        self.assertIsNone(as_float(None))
        self.assertIsNone(as_float(""))
        self.assertIsNone(as_float("  "))
        self.assertIsNone(as_float("None"))
        self.assertEqual(as_float("1.5"), 1.5)
        self.assertEqual(as_float(0), 0.0)
        self.assertIsNone(as_float("-"))
        self.assertIsNone(as_float("#N/A"))
        self.assertIs(as_bool("TRUE"), True)
        self.assertIs(as_bool("0"), False)
        self.assertIs(as_bool("是"), True)
        self.assertIs(as_bool("否"), False)
        self.assertIsNone(as_bool(""))

    def test_excel_bools_and_dash_cells_keep_row(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "Stu_ZZGLRHDL_0.csv"
            names = [name for name, _ in RHDL_COLUMNS]
            with path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\r\n")
                writer.writerow(names)
                row = [RHDL_ROW.get(name, "") for name, _ in RHDL_COLUMNS]
                row[names.index("B_Sgzmb")] = "1"
                row[names.index("B_l_jbz")] = "TRUE"
                row[names.index("B_Sdk")] = "0"
                row[names.index("B_QMBflag")] = "FALSE"
                row[names.index("D_Esm")] = "-"
                writer.writerow(row)
            records = read_csv(path, RHDL_COLUMNS)
            self.assertEqual(len(records), 1)
            self.assertIs(records[0]["B_Sgzmb"], True)
            self.assertIs(records[0]["B_l_jbz"], True)
            self.assertIs(records[0]["B_Sdk"], False)
            self.assertIs(records[0]["B_QMBflag"], False)
            self.assertIsNone(records[0]["D_Esm"])
            self.assertEqual(records[0]["TargetUnitID"], 11)

    def test_interface_skips_empty_target_and_missing_coords(self):
        payload = {
            "targets": [
                {"TargetUnitID": None, "D_X": 1, "D_Y": 0, "D_Z": 0, "D_XV": 0, "D_YV": 0, "D_ZV": 0},
                {
                    "TargetUnitID": 7,
                    "D_X": 5.0, "D_Y": 0.0, "D_Z": 0.1,
                    "D_XV": -0.1, "D_YV": 0.0, "D_ZV": 0.0,
                    "Ui_Zwx": "",
                    "Klj_list": [{"UI_FSChandle": "", "D_LJGL": None, "B_DDKsslj": "1"}],
                },
            ],
            "units": [{"UnitID": "", "UnitPos_x": None, "UnitPos_y": None, "UnitPos_z": None}],
        }
        snap = snapshot_from_interface(payload, time=None)
        self.assertEqual(len(snap.targets), 1)
        self.assertEqual(snap.targets[0].target_id, 7)
        self.assertEqual(snap.targets[0].threat, 1)
        self.assertEqual(len(snap.units), 0)
        self.assertTrue(snap.links[0].can_intercept)

    def test_read_csv_skips_blank_rows(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "Stu_ZZGLRHDL_0.csv"
            names = [name for name, _ in RHDL_COLUMNS]
            with path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\r\n")
                writer.writerow(names)
                writer.writerow([])
                writer.writerow([""] * len(names))
                row = [RHDL_ROW.get(name, "") for name, _ in RHDL_COLUMNS]
                row[names.index("B_Sgzmb")] = "True"
                row[names.index("B_l_jbz")] = "True"
                row[names.index("B_Sdk")] = "True"
                row[names.index("B_QMBflag")] = "False"
                writer.writerow(row)
                writer.writerow([])
            records = read_csv(path, RHDL_COLUMNS)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["TargetUnitID"], 11)
            self.assertEqual(records[0]["FZTime"], 1.0)

    def test_snapshot_skips_empty_target_rows(self):
        rhdl = [
            {name: None for name, _ in RHDL_COLUMNS},
            dict(RHDL_ROW),
        ]
        snap = snapshot_from_tables(rhdl, [], [], 1.0, 1)
        self.assertEqual(len(snap.targets), 1)
        self.assertEqual(snap.targets[0].target_id, 11)

    def test_candidate_and_decision_times(self):
        times = [1.0 + 0.5 * i for i in range(80)]
        cand = candidate_times(times, stride_s=2.0, dt=0.5)
        self.assertEqual(cand[0], 1.0)
        self.assertLess(len(cand), len(times))
        dec = decision_times(times, 15)
        self.assertEqual(dec[0], 1.0)
        self.assertGreaterEqual(dec[1] - dec[0], 14.9)


class ResumeTests(unittest.TestCase):
    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as raw:
            ckpt = SceneCheckpoint(Path(raw))
            ckpt.save_scene(7, [{"scene_id": 7, "utility": 0.2}], [{"scene_id": 7, "reason": "x"}])
            done, rows, dropped, extra = ckpt.load()
            self.assertEqual(done, {7})
            self.assertEqual(rows[0]["scene_id"], 7)
            self.assertEqual(dropped[0]["reason"], "x")
            self.assertEqual(extra, [])
            ckpt.clear()
            self.assertFalse(ckpt.exists())

    def test_slice_resume_skips_finished_scene(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "kemu6_data_72mb"
            for index, scene_id in ((1, 1), (2, 2)):
                scene_dir = root / "1-200" / f"{SCENE}-1-200_{index:03d}"
                _write_leaf(scene_dir / "_1", 1, extra_blank=True)
                _write_leaf(scene_dir / "_2", 2, extra_blank=True)
            grouped, meta = load_grouped_runs(root)
            self.assertEqual(meta["n_scenes"], 2)
            self.assertEqual(meta["n_load_errors"], 0)
            self.assertEqual(set(grouped), {1, 2})
            out = Path(raw) / "slice"
            scene_rows, scene_dropped = _process_scene(1, grouped[1], 120, 0, False, 2.0)
            self.assertTrue(scene_rows)
            ckpt = SceneCheckpoint(out)
            ckpt.save_scene(1, scene_rows, scene_dropped)
            summary = build_slice_set(root, out, with_forks=False, max_mid_slices=0, resume=True, n_ticks=120)
            self.assertGreaterEqual(summary["n_rows"], 4)
            self.assertEqual(summary["n_scenes"], 2)
            self.assertFalse((out / "_ckpt").exists())


if __name__ == "__main__":
    unittest.main()
