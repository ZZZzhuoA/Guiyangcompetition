"""训练并导出可导入赛方电脑的策略模型。"""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import numpy as np

from src.finals.features import FEATURE_NAMES
from src.finals.infer import FinalsPolicy
from src.finals.metrics import attach_selection, group_match, recommend_latency_ms
from src.finals.models import kind_weights
from src.finals.registry import COMPARE_NAMES, DEFAULT_SUBMIT, MODEL_CATALOG, make_model
from src.finals.schema import STRATEGIES


# 贯序数据集额外带的标签列，单切片数据集没有这些列时自动跳过
OPTIONAL_LABEL_KEYS = (
    "r_delta",
    "r_term",
    "mix",
    "r_shaped",
    "phi",
    "phi_next",
    "shaping",
    "ht_leak_delta",
    "n_faced_delta",
    "n_faced_term",
    "done",
)


def _cell(row: dict, name: str) -> float:
    value = row.get(name, 0.0)
    if value is None or value == "":
        return 0.0
    return float(value)


def align_hist(x_hist: np.ndarray, stored_names: list[str] | None = None) -> np.ndarray:
    """(n, K, F) 的历史窗口按特征名对齐，逐帧走 align_x。"""
    x_hist = np.asarray(x_hist, dtype=float)
    n, window, _ = x_hist.shape
    flat = align_x(x_hist.reshape(n * window, -1), stored_names)
    return flat.reshape(n, window, flat.shape[1])


def align_x(x: np.ndarray, stored_names: list[str] | None = None) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if stored_names is None:
        names = list(FEATURE_NAMES[: x.shape[1]])
    else:
        names = [str(name) for name in stored_names]
    index = {name: i for i, name in enumerate(names)}
    out = np.zeros((len(x), len(FEATURE_NAMES)), dtype=float)
    for j, name in enumerate(FEATURE_NAMES):
        if name in index:
            out[:, j] = x[:, index[name]]
    return out


def save_training_npz(output_dir: Path, rows: list[dict], extra: dict | None = None) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "training_samples.npz"
    payload = {
        "x": np.array([[_cell(row, name) for name in FEATURE_NAMES] for row in rows], dtype=float),
        "strategy": np.array([int(row["strategy"]) for row in rows], dtype=int),
        "utility": np.array([float(row["utility"]) for row in rows], dtype=float),
        "scene_id": np.array([int(row["scene_id"]) for row in rows], dtype=int),
        "time": np.array([float(row["time"]) for row in rows], dtype=float),
        "kind": np.array([row["kind"] for row in rows]),
        "group_id": np.array([row["group_id"] for row in rows]),
        "feature_names": np.array(FEATURE_NAMES),
    }
    for key in OPTIONAL_LABEL_KEYS:
        if rows and key in rows[0]:
            payload[key] = np.array([_cell(row, key) for row in rows], dtype=float)
    if extra:
        payload.update(extra)
    np.savez(path, **payload)
    return path


def load_arrays(dataset_dir: Path) -> dict:
    dataset_dir = Path(dataset_dir)
    npz = dataset_dir / "training_samples.npz"
    csv_path = dataset_dir / "training_samples.csv"
    if not npz.exists():
        if not csv_path.exists():
            raise FileNotFoundError(f"Need training_samples.npz or training_samples.csv in {dataset_dir}")
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"Empty training table: {csv_path}")
        save_training_npz(dataset_dir, rows)
    packed = np.load(npz, allow_pickle=True)
    stored = packed["feature_names"].tolist() if "feature_names" in packed.files else None
    data = {
        "x": align_x(packed["x"], stored),
        "strategy": packed["strategy"],
        "utility": packed["utility"],
        "scene_id": packed["scene_id"],
        "time": packed["time"],
        "kind": packed["kind"],
        "group_id": packed["group_id"],
        "feature_names": np.array(FEATURE_NAMES),
    }
    for key in OPTIONAL_LABEL_KEYS:
        if key in packed.files:
            data[key] = packed[key]
    if "x_next" in packed.files:
        data["x_next"] = align_x(packed["x_next"], stored)
    if "x_hist" in packed.files:
        data["x_hist"] = align_hist(packed["x_hist"], stored)
    for key in ("a_prev", "hist_len"):
        if key in packed.files:
            data[key] = packed[key]
    return data


def _label(scorer, data: dict, mask: np.ndarray) -> np.ndarray:
    key = getattr(scorer, "label_key", "utility")
    if key in data:
        return data[key][mask]
    return data["utility"][mask]


def history_kwargs(scorer, data: dict, mask: np.ndarray) -> dict:
    """序列模型要的窗口。数据里没有窗口时返回空，模型会退化成单帧冷启动。"""
    if not getattr(scorer, "needs_history", False) or "x_hist" not in data:
        return {}
    return {
        "x_hist": data["x_hist"][mask],
        "a_prev": data["a_prev"][mask] if "a_prev" in data else None,
        "hist_len": data["hist_len"][mask] if "hist_len" in data else None,
    }


def fit_scorer(scorer, data: dict, mask: np.ndarray, weights: np.ndarray):
    kwargs = history_kwargs(scorer, data, mask)
    if getattr(scorer, "needs_transition", False) and "x_next" in data:
        reward_key = getattr(scorer, "reward_key", "r_shaped")
        if reward_key not in data:
            reward_key = "r_delta" if "r_delta" in data else None
        kwargs["x_next"] = data["x_next"][mask]
        kwargs["done"] = data["done"][mask] if "done" in data else None
        kwargs["reward"] = None if reward_key is None else data[reward_key][mask]
    scorer.fit(
        data["x"][mask],
        data["strategy"][mask],
        _label(scorer, data, mask),
        data["group_id"][mask],
        sample_weight=weights[mask],
        **kwargs,
    )
    return scorer


def scene_split(scene_id: np.ndarray, test_ratio: float = 0.25, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    scenes = np.unique(scene_id)
    rng.shuffle(scenes)
    n_test = max(1, int(round(len(scenes) * test_ratio)))
    test_scenes = set(scenes[:n_test].tolist())
    test = np.array([int(s) in test_scenes for s in scene_id], dtype=bool)
    return ~test, test


def write_split(dataset_dir: Path, data: dict, train_mask: np.ndarray, test_mask: np.ndarray, test_ratio: float, seed: int = 0) -> dict:
    """把按场景切分的训练/测试集落盘。没有单独的测试 CSV：测试就是这些 scene_id 对应的行。"""
    payload = {
        "test_ratio": float(test_ratio),
        "seed": int(seed),
        "train_scenes": sorted({int(s) for s in data["scene_id"][train_mask]}),
        "test_scenes": sorted({int(s) for s in data["scene_id"][test_mask]}),
        "n_train_rows": int(train_mask.sum()),
        "n_test_rows": int(test_mask.sum()),
        "note": "按场景切分，同一局不会同时出现在训练和测试里。测试集不另存 CSV。",
    }
    path = Path(dataset_dir) / "split.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def train_policy(dataset_dir: Path, output_dir: Path, test_ratio: float = 0.25, model_name: str = DEFAULT_SUBMIT) -> dict:
    data = load_arrays(dataset_dir)
    train_mask, test_mask = scene_split(data["scene_id"], test_ratio=test_ratio)
    split = write_split(dataset_dir, data, train_mask, test_mask, test_ratio)
    weights = kind_weights(data["kind"])
    scorer = make_model(model_name)
    fit_scorer(scorer, data, train_mask, weights)
    policy = FinalsPolicy(scorer=scorer)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "policy.pkl"
    policy.save(model_path)

    t0 = _counterfactual_mask(data["kind"], data["group_id"], data["strategy"]) & test_mask
    metrics = evaluate_groups(scorer, data, t0)
    train_t0 = _counterfactual_mask(data["kind"], data["group_id"], data["strategy"]) & train_mask
    train_metrics = evaluate_groups(scorer, data, train_t0)
    summary = {
        "model_path": str(model_path),
        "model_name": model_name,
        "mode": "single_slice_scoring",
        "n_features": int(data["x"].shape[1]),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "split": split,
        "test_groups": metrics,
        "strategies": list(STRATEGIES),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    return summary


def compare_models(
    dataset_dir: Path,
    output_dir: Path,
    names: tuple[str, ...] = COMPARE_NAMES,
    test_ratio: float = 0.25,
    with_replay: bool = False,
    replay_scenes: int = 3,
    replay_ticks: int = 60,
    delta: float = 15.0,
) -> dict:
    data = load_arrays(dataset_dir)
    train_mask, test_mask = scene_split(data["scene_id"], test_ratio=test_ratio)
    split = write_split(dataset_dir, data, train_mask, test_mask, test_ratio)
    weights = kind_weights(data["kind"])
    cf = _counterfactual_mask(data["kind"], data["group_id"], data["strategy"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fitted = {}
    rows = []
    for name in names:
        scorer = make_model(name)
        fit_scorer(scorer, data, train_mask, weights)
        fitted[name] = scorer
        with (output_dir / f"{name}.pkl").open("wb") as stream:
            pickle.dump(scorer, stream)
        train_m = evaluate_groups(scorer, data, cf & train_mask)
        test_m = evaluate_groups(scorer, data, cf & test_mask)
        rows.append(
            {
                "name": name,
                "family": MODEL_CATALOG[name]["family"],
                "label_key": getattr(scorer, "label_key", "utility"),
                "reward_key": getattr(scorer, "reward_key", None),
                "oracle_key": test_m.get("oracle_key"),
                "train_match": train_m["match"],
                "test_match": test_m["match"],
                "train_groups": train_m["n_groups"],
                "test_groups": test_m["n_groups"],
                "test_collapse": test_m.get("collapse"),
                # 三策略打分的最大差，接近 0 说明模型没有区分能力，输出等价于固定策略
                "test_score_margin": test_m.get("mean_score_margin"),
                "test_pred_hist": test_m["pred_hist"],
                "test_oracle_hist": test_m["oracle_hist"],
                "latency_ms": recommend_latency_ms(
                    scorer,
                    data["x"][train_mask],
                    history=history_kwargs(scorer, data, train_mask),
                ),
            }
        )
    replay = None
    if with_replay:
        from src.finals.replay import evaluate_periodic_policies

        replay = evaluate_periodic_policies(
            {name: fitted[name] for name in names},
            n_scenes=replay_scenes,
            n_ticks=replay_ticks,
            delta=delta,
        )
        for row in rows:
            packed = replay.get(row["name"])
            if packed is None and row["name"].startswith("constant_"):
                packed = replay.get("fixed_" + row["name"].split("_")[1])
            if packed:
                row["episode_intercept_rate"] = packed["episode_intercept_rate"]
                row["lift_vs_best_fixed"] = packed.get("lift_vs_best_fixed")
    attach_selection(rows)
    scored = [row for row in rows if row.get("selection_score") is not None]
    chosen = max(scored, key=lambda item: item["selection_score"])["name"] if scored else None
    summary = {
        "mode": "model_compare",
        "n_features": int(data["x"].shape[1]),
        "default_submit": DEFAULT_SUBMIT,
        "chosen": chosen,
        "split": split,
        "note": "chosen 只是当前指标排序，不会覆盖 policy.pkl。赛方 CSV 上的测试是 split.json 里那些测试场景的 group_match；--with-replay 用的是本地玩具世界，不是赛方样本。",
        "rows": rows,
        "replay": None if replay is None else {k: v for k, v in replay.items() if k != "decisions"},
    }
    (output_dir / "compare.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return summary


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _counterfactual_mask(kind: np.ndarray, group_id: np.ndarray, strategy: np.ndarray) -> np.ndarray:
    keep = np.zeros(len(kind), dtype=bool)
    groups: dict[str, list[int]] = {}
    for i, gid in enumerate(group_id):
        groups.setdefault(str(gid), []).append(i)
    for idxs in groups.values():
        if len({int(strategy[i]) for i in idxs}) >= 2:
            for i in idxs:
                keep[i] = True
    return keep


def oracle_label(data: dict) -> tuple[np.ndarray, str]:
    """对照组的“正确答案”按赛方主指标定：有整局拦截率就用它，别用各模型自己的标签。"""
    if "r_term" in data:
        return data["r_term"], "r_term"
    return data["utility"], "utility"


def evaluate_groups(scorer, data: dict, mask: np.ndarray) -> dict:
    if not np.any(mask):
        return {"n_groups": 0, "match": None, "pred_hist": {s: 0 for s in STRATEGIES}, "oracle_hist": {s: 0 for s in STRATEGIES}, "collapse": 1.0, "random_baseline": 1.0 / len(STRATEGIES)}
    oracle, key = oracle_label(data)
    out = group_match(
        scorer,
        data["x"][mask],
        data["strategy"][mask],
        oracle[mask],
        data["group_id"][mask],
        history=history_kwargs(scorer, data, mask),
    )
    out["oracle_key"] = key
    return out
