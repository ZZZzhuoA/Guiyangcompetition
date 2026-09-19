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
from src.finals.matching import build_matched_groups, is_observed_dataset
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

OUTCOME_KEYS = (
    "intercept_rate",
    "leak_rate",
    "cost",
    "cost_eff",
)

OBSERVED_ARRAY_KEYS = (
    "run_id",
    "replicate_id",
    "time_next",
    "intercept_delta",
    "reward_delta",
    "return_to_go",
    "episode_intercept_rate",
    "has_opportunity",
    "dataset_mode",
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
    for key in OUTCOME_KEYS:
        if rows and key in rows[0]:
            payload[key] = np.array([_cell(row, key) for row in rows], dtype=float)
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
    for key in OUTCOME_KEYS:
        if key in packed.files:
            data[key] = packed[key]
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
    for key in OBSERVED_ARRAY_KEYS:
        if key in packed.files:
            data[key] = packed[key]
    # 旧版 NPZ 没有保存 outcome 列。直接从已经生成的 CSV 补齐，避免为了
    # 改训练目标重新跑数小时的原始数据构造。
    missing_outcomes = [key for key in OUTCOME_KEYS if key not in data]
    if missing_outcomes and csv_path.exists():
        values = {key: [] for key in missing_outcomes}
        with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            available = set(reader.fieldnames or [])
            wanted = [key for key in missing_outcomes if key in available]
            for row in reader:
                for key in wanted:
                    values[key].append(_cell(row, key))
        for key, column in values.items():
            if len(column) == len(data["strategy"]):
                data[key] = np.asarray(column, dtype=float)
    return data


def _label(scorer, data: dict, mask: np.ndarray) -> np.ndarray:
    key = label_key(scorer, data)
    if key in data:
        return data[key][mask]
    return data["utility"][mask]


def label_key(scorer, data: dict) -> str:
    """序列模型沿用各自标签；单切片模型统一对齐整局拦截率。"""
    explicit = getattr(scorer, "label_key", None)
    if explicit:
        return str(explicit)
    return "intercept_rate" if "intercept_rate" in data else "utility"


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
    # 单切片模型只从确有策略差异的组学习动作偏好。大量三策略同为 0 的
    # 短分叉样本会稀释信号；若整批都并列则保留原 mask，让后续诊断明确
    # 报出“无有效组”，而不是在训练阶段以空数组崩溃。
    if getattr(scorer, "label_key", None) is None and "intercept_rate" in data:
        informative = informative_group_mask(data, mask, data["intercept_rate"])
        if np.any(informative):
            mask = informative
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


def informative_group_mask(data: dict, mask: np.ndarray, target: np.ndarray) -> np.ndarray:
    keep = np.zeros(len(mask), dtype=bool)
    groups: dict[str, list[int]] = {}
    for i in np.flatnonzero(mask):
        groups.setdefault(str(data["group_id"][i]), []).append(int(i))
    for idxs in groups.values():
        if len({int(data["strategy"][i]) for i in idxs}) < 2:
            continue
        values = np.asarray([target[i] for i in idxs], dtype=float)
        if float(np.max(values) - np.min(values)) > 1e-12:
            keep[idxs] = True
    return keep


def scene_split(scene_id: np.ndarray, test_ratio: float = 0.25, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < float(test_ratio) < 1.0:
        raise ValueError(f"test_ratio must be between 0 and 1 (exclusive), got {test_ratio}")
    rng = np.random.default_rng(seed)
    scenes = np.unique(scene_id)
    if len(scenes) < 2:
        raise ValueError(
            f"Need at least 2 distinct scenes for a train/test split, got {len(scenes)}. "
            "Add another scene before training."
        )
    rng.shuffle(scenes)
    n_test = min(len(scenes) - 1, max(1, int(round(len(scenes) * test_ratio))))
    test_scenes = set(scenes[:n_test].tolist())
    test = np.array([int(s) in test_scenes for s in scene_id], dtype=bool)
    return ~test, test


def split_training_views(
    data: dict,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
) -> tuple[dict, np.ndarray, dict, np.ndarray, dict | None]:
    """真实独立轨迹先切场景，再在各自 split 内做相似态势匹配。"""
    if not is_observed_dataset(data):
        return data, train_mask, data, test_mask, None
    train_data, train_diag = build_matched_groups(data, train_mask)
    test_data, test_diag = build_matched_groups(data, test_mask)
    train_all = np.ones(len(train_data["strategy"]), dtype=bool)
    test_all = np.ones(len(test_data["strategy"]), dtype=bool)
    return train_data, train_all, test_data, test_all, {
        "method": "split_local_knn",
        "train": train_diag,
        "test": test_diag,
        "note": "先按 scene 切分，再分别匹配；测试态势及其收益不参与训练匹配。",
    }


def write_split(dataset_dir: Path, data: dict, train_mask: np.ndarray, test_mask: np.ndarray, test_ratio: float, seed: int = 0) -> dict:
    """把按场景切分的训练/测试集落盘。没有单独的测试 CSV：测试就是这些 scene_id 对应的行。"""
    def count_points(mask: np.ndarray, with_strategy: bool = False) -> int:
        if with_strategy:
            return len({
                (int(scene), int(strategy), float(time))
                for scene, strategy, time in zip(data["scene_id"][mask], data["strategy"][mask], data["time"][mask])
            })
        return len({
            (int(scene), float(time))
            for scene, time in zip(data["scene_id"][mask], data["time"][mask])
        })

    payload = {
        "test_ratio": float(test_ratio),
        "seed": int(seed),
        "train_scenes": sorted({int(s) for s in data["scene_id"][train_mask]}),
        "test_scenes": sorted({int(s) for s in data["scene_id"][test_mask]}),
        "n_train_rows": int(train_mask.sum()),
        "n_test_rows": int(test_mask.sum()),
        "n_train_decision_times": count_points(train_mask),
        "n_test_decision_times": count_points(test_mask),
        "n_train_action_time_samples": count_points(train_mask, with_strategy=True),
        "n_test_action_time_samples": count_points(test_mask, with_strategy=True),
        "note": "按场景切分，同一局不会同时出现在训练和测试里。rows 是数据行数；decision_times 是去重后的场景-时刻数；action_time_samples 还区分策略。",
    }
    path = Path(dataset_dir) / "split.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def train_policy(dataset_dir: Path, output_dir: Path, test_ratio: float = 0.25, model_name: str = DEFAULT_SUBMIT) -> dict:
    data = load_arrays(dataset_dir)
    train_mask, test_mask = scene_split(data["scene_id"], test_ratio=test_ratio)
    split = write_split(dataset_dir, data, train_mask, test_mask, test_ratio)
    train_data, train_view_mask, test_data, test_view_mask, matching = split_training_views(
        data, train_mask, test_mask
    )
    weights = kind_weights(train_data["kind"])
    scorer = make_model(model_name)
    fit_scorer(scorer, train_data, train_view_mask, weights)
    policy = FinalsPolicy(scorer=scorer)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "policy.pkl"
    policy.save(model_path)

    test_cf = _counterfactual_mask(test_data["kind"], test_data["group_id"], test_data["strategy"])
    metrics = evaluate_groups(scorer, test_data, test_cf & test_view_mask)
    train_cf = _counterfactual_mask(train_data["kind"], train_data["group_id"], train_data["strategy"])
    train_metrics = evaluate_groups(scorer, train_data, train_cf & train_view_mask)
    summary = {
        "model_path": str(model_path),
        "model_name": model_name,
        "mode": "single_slice_scoring",
        "n_features": int(data["x"].shape[1]),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "label_key": label_key(scorer, train_data),
        "split": split,
        "matching": matching,
        "train_groups": train_metrics,
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
    train_data, train_view_mask, test_data, test_view_mask, matching = split_training_views(
        data, train_mask, test_mask
    )
    weights = kind_weights(train_data["kind"])
    train_cf = _counterfactual_mask(train_data["kind"], train_data["group_id"], train_data["strategy"])
    test_cf = _counterfactual_mask(test_data["kind"], test_data["group_id"], test_data["strategy"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fitted = {}
    rows = []
    for name in names:
        scorer = make_model(name)
        fit_scorer(scorer, train_data, train_view_mask, weights)
        fitted[name] = scorer
        with (output_dir / f"{name}.pkl").open("wb") as stream:
            pickle.dump(scorer, stream)
        train_m = evaluate_groups(scorer, train_data, train_cf & train_view_mask)
        test_m = evaluate_groups(scorer, test_data, test_cf & test_view_mask)
        rows.append(
            {
                "name": name,
                "family": MODEL_CATALOG[name]["family"],
                "label_key": label_key(scorer, train_data),
                "reward_key": getattr(scorer, "reward_key", None),
                "oracle_key": test_m.get("oracle_key"),
                "train_match": train_m["match"],
                "test_match": test_m["match"],
                "train_groups": train_m["n_groups"],
                "test_groups": test_m["n_groups"],
                "test_total_groups": test_m.get("n_total_groups", test_m["n_groups"]),
                "train_tied_groups": train_m.get("n_tied_groups", 0),
                "test_tied_groups": test_m.get("n_tied_groups", 0),
                "test_collapse": test_m.get("collapse"),
                # 三策略打分的最大差，接近 0 说明模型没有区分能力，输出等价于固定策略
                "test_score_margin": test_m.get("mean_score_margin"),
                "test_pred_hist": test_m["pred_hist"],
                "test_oracle_hist": test_m["oracle_hist"],
                "test_policy_value": test_m.get("mean_policy_value"),
                "test_oracle_value": test_m.get("mean_oracle_value"),
                "test_mean_regret": test_m.get("mean_regret"),
                "test_fixed_policy_values": test_m.get("fixed_policy_values"),
                "test_lift_vs_best_fixed_matched": test_m.get("lift_vs_best_fixed_matched"),
                "latency_ms": recommend_latency_ms(
                    scorer,
                    train_data["x"][train_view_mask],
                    history=history_kwargs(scorer, train_data, train_view_mask),
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
        "matching": matching,
        "target_diagnostics": {
            "train": target_diagnostics(train_data, train_cf & train_view_mask),
            "test": target_diagnostics(test_data, test_cf & test_view_mask),
        },
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
    if "intercept_rate" in data:
        return data["intercept_rate"], "intercept_rate"
    return data["utility"], "utility"


def target_diagnostics(data: dict, mask: np.ndarray) -> dict:
    """汇总实际选模标签，明确均值、有效组和并列组。"""
    target, key = oracle_label(data)
    means = {}
    for strategy in STRATEGIES:
        selected = mask & (data["strategy"].astype(int) == int(strategy))
        means[int(strategy)] = None if not np.any(selected) else float(np.mean(target[selected]))
    groups: dict[str, list[int]] = {}
    for i in np.flatnonzero(mask):
        groups.setdefault(str(data["group_id"][i]), []).append(int(i))
    winners = {int(strategy): 0 for strategy in STRATEGIES}
    tied = 0
    informative = 0
    for idxs in groups.values():
        if len({int(data["strategy"][i]) for i in idxs}) < 2:
            continue
        values = np.asarray([target[i] for i in idxs], dtype=float)
        best_value = float(np.max(values))
        if int(np.sum(np.abs(values - best_value) <= 1e-12)) != 1:
            tied += 1
            continue
        best = int(data["strategy"][idxs[int(np.argmax(values))]])
        winners[best] += 1
        informative += 1
    return {
        "key": key,
        "strategy_mean": means,
        "winner_hist": winners,
        "informative_groups": informative,
        "tied_groups": tied,
    }


def evaluate_groups(scorer, data: dict, mask: np.ndarray) -> dict:
    if not np.any(mask):
        return {"n_groups": 0, "n_total_groups": 0, "n_tied_groups": 0, "match": None, "pred_hist": {s: 0 for s in STRATEGIES}, "oracle_hist": {s: 0 for s in STRATEGIES}, "collapse": 1.0, "mean_policy_value": None, "mean_oracle_value": None, "mean_regret": None, "fixed_policy_values": {s: None for s in STRATEGIES}, "lift_vs_best_fixed_matched": None, "random_baseline": 1.0 / len(STRATEGIES)}
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
