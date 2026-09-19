"""样本时间步长、四类型可拦截边、概率不限于 0.7/0.9。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.finals.features import extract_snapshot_dict
from src.finals.schema import HIGH_P_THRESHOLD, infer_dt, seconds_to_ticks
from src.finals.snapshot import _link_from_lj, align_time, snapshot_from_interface, snapshot_from_tables


def _lj_row(target: int, etype: int, can: bool, p: float, time: float) -> dict:
    flags = {
        "B_DDKsslj": False,
        "B_GPKsslj": False,
        "B_HPMKsslj": False,
        "B_GNJGKsslj": False,
    }
    flags[("B_DDKsslj", "B_GPKsslj", "B_HPMKsslj", "B_GNJGKsslj")[etype - 1]] = can
    return {
        "UI_Targethandle": target,
        "UI_Hlhandle": 10 + etype,
        "UC_HlType": etype,
        "UI_FSChandle": 10 + etype,
        "Str_HLMC": f"Type{etype}",
        **flags,
        "B_Kgz": True,
        "B_Ygz": False,
        "B_TD": True,
        "G_HLYS": True,
        "D_Pm": 0.0,
        "D_Rm": 5.0,
        "D_Tmzi": 8.0,
        "D_Costgy": 0.5,
        "G_LJGL": p,
        "Time": time,
    }


class TimeAndLinkTests(unittest.TestCase):
    def test_record_dt_and_delta_ticks(self):
        self.assertAlmostEqual(infer_dt([1.0, 1.5, 2.0, 2.5]), 0.5)
        self.assertEqual(seconds_to_ticks(15, 0.5), 30)
        self.assertEqual(seconds_to_ticks(15, 1.0), 15)

    def test_align_0p1_query_to_0p5_record(self):
        times = [1.0, 1.5, 2.0]
        self.assertEqual(align_time(times, 1.0), 1.0)
        self.assertEqual(align_time(times, 1.1), 1.0)
        self.assertEqual(align_time(times, 1.49), 1.0)
        self.assertEqual(align_time(times, 1.5), 1.5)
        self.assertEqual(align_time(times, 0.1), 1.0)

    def test_four_type_rows_and_type4_flag(self):
        rows = [
            _lj_row(101, 1, True, 0.7, 1.0),
            _lj_row(101, 2, True, 0.8, 1.0),
            _lj_row(101, 3, False, 0.9, 1.0),
            _lj_row(101, 4, True, 1.0, 1.0),
        ]
        links = [_link_from_lj(row) for row in rows]
        self.assertEqual([lk.entity_type for lk in links], [1, 2, 3, 4])
        self.assertEqual([lk.p for lk in links], [0.7, 0.8, 0.9, 1.0])
        self.assertEqual([lk.can_intercept for lk in links], [True, True, False, True])
        type4 = links[3]
        self.assertTrue(type4.can_flags["B_GNJGKsslj"])
        self.assertFalse(type4.can_flags["B_DDKsslj"])

    def test_snapshot_aligns_and_high_p_includes_0p8(self):
        rhdl = [{
            "TargetUnitID": 101,
            "TargetUnitName": "b1",
            "B_Sgzmb": True,
            "B_l_jbz": False,
            "B_Sdk": False,
            "B_QMBflag": False,
            "Uc_XXYLY": 1,
            "Uc_XXYHandle": 1,
            "D_Bsm": 0.0,
            "D_Esm": 0.0,
            "Ui_Zwx": 1,
            "D_XDPm": 0.0,
            "D_XDTm": 8.0,
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
        }]
        lj = [
            _lj_row(101, 1, True, 0.7, 1.0),
            _lj_row(101, 2, True, 0.8, 1.0),
            _lj_row(101, 3, True, 0.9, 1.0),
            _lj_row(101, 4, True, 1.0, 1.0),
        ]
        snap = snapshot_from_tables(rhdl, [], lj, time=1.1)
        self.assertEqual(len(snap.links), 4)
        self.assertEqual(len(snap.targets), 1)
        stats = extract_snapshot_dict(snap)
        self.assertEqual(stats["n_can_links"], 4.0)
        self.assertGreaterEqual(HIGH_P_THRESHOLD, 0.8)
        self.assertEqual(stats["n_high_p_links"], 3.0)

    def test_interface_four_types(self):
        payload = {
            "targets": [{
                "TargetUnitID": 101,
                "TargetUnitType": "BlueTarget",
                "B_l_jbz": False,
                "Ui_Zwx": 1,
                "D_X": 5.0, "D_Y": 0.0, "D_Z": 0.1,
                "D_XV": -0.1, "D_YV": 0.0, "D_ZV": 0.0,
                "Klj_list": [
                    {"UI_FSChandle": 11, "Str_HLMC": "Type1", "UC_HlType": 1,
                     "B_DDKsslj": True, "B_GPKsslj": False, "B_HPMKsslj": False, "B_GNJGKsslj": False, "D_LJGL": 0.7},
                    {"UI_FSChandle": 12, "Str_HLMC": "Type2", "UC_HlType": 2,
                     "B_DDKsslj": False, "B_GPKsslj": True, "B_HPMKsslj": False, "B_GNJGKsslj": False, "D_LJGL": 0.8},
                    {"UI_FSChandle": 13, "Str_HLMC": "Type3", "UC_HlType": 3,
                     "B_DDKsslj": False, "B_GPKsslj": False, "B_HPMKsslj": True, "B_GNJGKsslj": False, "D_LJGL": 0.9},
                    {"UI_FSChandle": 14, "Str_HLMC": "Type4", "UC_HlType": 4,
                     "B_DDKsslj": False, "B_GPKsslj": False, "B_HPMKsslj": False, "B_GNJGKsslj": True, "D_LJGL": 1.0},
                ],
            }],
            "units": [],
        }
        snap = snapshot_from_interface(payload, time=15.1)
        self.assertEqual([lk.p for lk in snap.links], [0.7, 0.8, 0.9, 1.0])
        self.assertTrue(snap.links[3].can_intercept)
        self.assertEqual(snap.links[3].entity_type, 4)


if __name__ == "__main__":
    unittest.main()
