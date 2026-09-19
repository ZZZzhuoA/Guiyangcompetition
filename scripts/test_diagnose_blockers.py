"""整份结果级 BLOCK 只看数据层、指标分辨力和 chosen，不把落选候选的失败放大。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnose_finals_models import collect_blockers, diagnose_row


class DiagnoseBlockersTests(unittest.TestCase):
    def test_row_latency_is_block(self):
        findings = diagnose_row({"name": "slow", "latency_ms": 80.0, "test_score_margin": 0.2})
        self.assertTrue(any(level == "BLOCK" and "延迟" in msg for level, msg in findings))

    def test_losing_candidate_does_not_fail_the_comparison(self):
        summary = {
            "chosen": "rf_pair",
            "rows": [
                {"name": "slow_net", "latency_ms": 80.0, "test_score_margin": 0.2, "test_groups": 20},
                {"name": "rf_pair", "latency_ms": 4.0, "test_score_margin": 0.2, "test_groups": 20},
            ],
        }
        hits = collect_blockers(summary, None)
        self.assertFalse(any("slow_net" in msg for msg in hits))
        self.assertFalse(hits)

    def test_chosen_latency_is_submission_block(self):
        summary = {
            "chosen": "slow_net",
            "rows": [
                {"name": "slow_net", "latency_ms": 80.0, "test_score_margin": 0.2, "test_groups": 20},
                {"name": "rf_pair", "latency_ms": 4.0, "test_score_margin": 0.2, "test_groups": 20},
            ],
        }
        hits = collect_blockers(summary, None)
        self.assertTrue(any("slow_net" in msg and "延迟" in msg for msg in hits))

    def test_metric_collapse_still_fails_the_comparison(self):
        summary = {
            "chosen": "rf_pair",
            "rows": [
                {"name": "rf_pair", "latency_ms": 4.0, "test_score_margin": 0.2, "episode_intercept_rate": 0.3},
                {"name": "mlp", "latency_ms": 8.0, "test_score_margin": 0.2, "episode_intercept_rate": 0.3},
            ],
        }
        hits = collect_blockers(summary, None)
        self.assertTrue(any("完全相同" in msg for msg in hits))


if __name__ == "__main__":
    unittest.main()
