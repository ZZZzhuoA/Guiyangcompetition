"""Small independent CSV reader contract checks."""

import csv
import tempfile
import unittest
from pathlib import Path

from make_csv_format_fixtures import read_typed_csv


SCHEMA = {"columns": [
    {"name": "record_id", "type": "int"},
    {"name": "enabled", "type": "bool"},
    {"name": "value", "type": "double"},
    {"name": "label", "type": "string"},
]}


class ReaderTests(unittest.TestCase):
    def read(self, rows, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as stream:
                csv.writer(stream).writerows(rows)
            return read_typed_csv(path, SCHEMA, **kwargs)

    def test_reordered_extra_columns_and_quoted_text(self):
        _, extra, records = self.read([
            ["label", "value", "unused", "enabled", "record_id"],
            ['text,"quoted"\nnext line', "0.25", "x", "False", "9007199254740993"],
        ])
        self.assertEqual(extra, ["unused"])
        self.assertEqual(records, [{"record_id": 9007199254740993, "enabled": False,
                                   "value": 0.25, "label": 'text,"quoted"\nnext line'}])

    def test_blank_is_not_false_or_zero(self):
        _, _, records = self.read([
            ["record_id", "enabled", "value", "label"],
            ["1", "", "", ""], ["2", "False", "0", "x"],
        ])
        self.assertIsNone(records[0]["enabled"])
        self.assertIsNone(records[0]["value"])
        self.assertIs(records[1]["enabled"], False)
        self.assertEqual(records[1]["value"], 0.0)

    def test_reject_missing_and_duplicate_columns(self):
        for headers in (["record_id", "enabled", "value"],
                        ["record_id", "enabled", "value", "label", "label"]):
            with self.subTest(headers=headers), self.assertRaises(ValueError):
                self.read([headers])

    def test_reject_bad_types(self):
        for row in (["1.0", "False", "0", "x"], ["1", "not_false", "0", "x"],
                    ["1", "True", "NaN", "x"], ["1", "False", "inf", "x"]):
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.read([["record_id", "enabled", "value", "label"], row])

    def test_reject_wrong_row_width(self):
        for row in (["1", "True", "0"], ["1", "True", "0", "x", "extra"]):
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.read([["record_id", "enabled", "value", "label"], row])

    def test_strict_blank_and_extra_options(self):
        with self.assertRaises(ValueError):
            self.read([["record_id", "enabled", "value", "label"],
                       ["1", "", "0", "x"]], allow_blank=False)
        with self.assertRaises(ValueError):
            self.read([["record_id", "enabled", "value", "label", "extra"]],
                      allow_extra=False)


if __name__ == "__main__":
    unittest.main()
