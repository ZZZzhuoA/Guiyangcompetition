"""赛方 CSV 列可以比本地 schema 多：只抽本地列。"""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.finals.io import detect_csv_encoding, read_csv
from src.finals.schema import LJ_COLUMNS, RHDL_COLUMNS


class ExtraColumnTests(unittest.TestCase):
    def test_drops_unknown_columns_and_keeps_schema(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "Stu_ZZGLLJ_0.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\r\n")
                writer.writerow([
                    "UI_Targethandle", "UC_HlType", "B_GNJGKsslj", "G_LJGL", "Time",
                    "ExtraCol", "AnotherNewField",
                ])
                writer.writerow(["101", "4", "True", "1.0", "1.0", "should-drop", "also-drop"])
            rows = read_csv(path, LJ_COLUMNS)
            self.assertEqual(len(rows), 1)
            self.assertEqual(set(rows[0]), {name for name, _ in LJ_COLUMNS})
            self.assertNotIn("ExtraCol", rows[0])
            self.assertEqual(rows[0]["UI_Targethandle"], 101)
            self.assertEqual(rows[0]["UC_HlType"], 4)
            self.assertIs(rows[0]["B_GNJGKsslj"], True)
            self.assertEqual(rows[0]["G_LJGL"], 1.0)
            self.assertIsNone(rows[0]["B_DDKsslj"])

    def test_d_ljgl_alias_and_stripped_headers(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "Stu_ZZGLRHDL_0.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.writer(stream, lineterminator="\r\n")
                writer.writerow([" TargetUnitID ", "D_X", "D_Y", "D_Z", "FZTime", "BrandNew"])
                writer.writerow(["11", "5.0", "0.0", "0.1", "1.0", "x"])
            rows = read_csv(path, RHDL_COLUMNS)
            self.assertEqual(rows[0]["TargetUnitID"], 11)
            self.assertEqual(rows[0]["D_X"], 5.0)
            self.assertIsNone(rows[0]["D_XV"])
            self.assertNotIn("BrandNew", rows[0])

    def test_gbk_and_leading_info_rows(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "Stu_ZZGLRHDL_0.csv"
            text = (
                "TargetUnitID,D_X,D_Y,D_Z,FZTime\r\n"
                "编号,米,米,米,秒\r\n"
                "说明,,备注,,单位\r\n"
                "11,5.0,0.0,0.1,1.0\r\n"
                "12,6.0,0.0,0.2,1.5\r\n"
            )
            path.write_bytes(text.encode("gbk"))
            self.assertEqual(detect_csv_encoding(path.read_bytes()), "gb18030")
            rows = read_csv(path, RHDL_COLUMNS)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["TargetUnitID"], 11)
            self.assertEqual(rows[1]["TargetUnitID"], 12)
            self.assertEqual(rows[0]["FZTime"], 1.0)

    def test_detects_utf8_before_opening(self):
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "a.csv"
            path.write_bytes("TargetUnitID,FZTime\n11,1.0\n".encode("utf-8"))
            self.assertEqual(detect_csv_encoding(path.read_bytes()), "utf-8")


if __name__ == "__main__":
    unittest.main()
