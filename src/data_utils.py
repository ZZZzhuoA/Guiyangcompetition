from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetBundle:
    train_file: Path
    test_file: Path
    feature_names: list[str]
    train_x: np.ndarray
    train_y: np.ndarray
    train_metric1: np.ndarray
    train_metric2: np.ndarray
    test_x: np.ndarray


def _read_excel_data(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0)
    data = raw.iloc[2:].copy()
    for column in data.columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data.reset_index(drop=True)


def load_competition_data(data_dir: str | Path = "Data") -> DatasetBundle:
    data_path = Path(data_dir)
    files = sorted(path for path in data_path.glob("*.xlsx") if not path.name.startswith("~$"))
    if len(files) != 2:
        raise FileNotFoundError(f"Expected exactly 2 xlsx files under {data_path}, got {len(files)}")

    loaded = [(len(_read_excel_data(path)), path, _read_excel_data(path)) for path in files]
    loaded.sort(key=lambda item: item[0], reverse=True)

    train_file, train_df = loaded[0][1], loaded[0][2]
    test_file, test_df = loaded[1][1], loaded[1][2]

    feature_slice = slice(1, 43)
    strategy_col = 43
    metric1_col = 44
    metric2_col = 45

    train_x = train_df.iloc[:, feature_slice].to_numpy(dtype=float)
    test_x = test_df.iloc[:, feature_slice].to_numpy(dtype=float)
    train_y = train_df.iloc[:, strategy_col].to_numpy(dtype=int)
    train_metric1 = train_df.iloc[:, metric1_col].to_numpy(dtype=float)
    train_metric2 = train_df.iloc[:, metric2_col].to_numpy(dtype=float)

    feature_names = [f"f{i:02d}" for i in range(1, train_x.shape[1] + 1)]
    return DatasetBundle(
        train_file=train_file,
        test_file=test_file,
        feature_names=feature_names,
        train_x=train_x,
        train_y=train_y,
        train_metric1=train_metric1,
        train_metric2=train_metric2,
        test_x=test_x,
    )


def describe_bundle(bundle: DatasetBundle) -> dict[str, object]:
    unique, counts = np.unique(bundle.train_y, return_counts=True)
    strategy_counts = {int(k): int(v) for k, v in zip(unique, counts)}
    return {
        "train_file": bundle.train_file.name,
        "test_file": bundle.test_file.name,
        "train_rows": int(bundle.train_x.shape[0]),
        "test_rows": int(bundle.test_x.shape[0]),
        "feature_count": int(bundle.train_x.shape[1]),
        "strategy_counts": strategy_counts,
        "metric1_min": float(np.min(bundle.train_metric1)),
        "metric1_mean": float(np.mean(bundle.train_metric1)),
        "metric1_max": float(np.max(bundle.train_metric1)),
        "metric2_min": float(np.min(bundle.train_metric2)),
        "metric2_mean": float(np.mean(bundle.train_metric2)),
        "metric2_max": float(np.max(bundle.train_metric2)),
    }
