"""从赛方形态的样本目录整理训练表。一个叶子目录就是一次仿真。

赛方 72mb 布局：`_1/_2/_3` 是同一场景的三次重复实验，不是 LJCL=1/2/3。
策略从「文件实验数据路径与结果」的「策略」列读：闭合时间最短→1，拦截性能最佳→2，
综合效益最优→3。对不上再退回 Stu_SJZS.S_LJCL / 合成文件夹名。

    kemu6_data_72mb/
      1-200/   ...-1-200_001/_1 … 序号按文件夹名，缺号就跳过（例如没有 179）
      200-400/
      400-600/
      600-end/ 或仍放在 1-200/ 下：...-600-end_001/_1 … ...-600-end_264/_3
      文件实验数据路径与结果_72mb.csv

scene_id 从**场景文件夹名**里的 `批次_序号` 取，不按目录枚举、不要求连续。
`...-600-end_264` → 864，即使父目录是 `1-200` 也不会和 `...-1-200_001` 撞号。
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.finals.features import FEATURE_NAMES, extract_features, utility_from_outcome
from src.finals.io import read_csv_table, read_family_dir
from src.finals.schema import STRATEGIES, as_float, as_int, infer_dt, parse_strategy_label
from src.finals.simulate import clone_world, run_episode, world_from_snapshot
from src.finals.snapshot import snapshot_from_tables
from src.finals.train import save_training_npz

# 一次仿真的主表。策略表 Stu_SJZS 可能缺，不能当发现依据。
_RUN_MARKERS = ("Stu_ZZGLRHDL_*.csv", "Stu_SJZS_*.csv")
_STRATEGY_LEAF = re.compile(r"^_([123])$")
_BATCH_RANGE = re.compile(r"^(\d+)-(\d+)$")
_BATCH_OPEN = re.compile(r"^(\d+)-(end)$", flags=re.IGNORECASE)
_TRAILING_INDEX = re.compile(r"_(\d+)$")
# 场景名末尾：1-200_179 / 400-600_180 / 600-end_264
_SCENE_BATCH_INDEX = re.compile(r"(?:^|[-_])(\d+)-(end|\d+)_(\d+)$", flags=re.IGNORECASE)
_RESULTS_NAMES = (
    "文件实验数据路径与结果_72mb.csv",
    "文件实验数据路径与结果.csv",
)


def discover_run_dirs(input_dir: Path) -> list[Path]:
    """递归找出每个「一次仿真」目录。xlsx 一律跳过。"""
    input_dir = Path(input_dir)
    found = []
    for pattern in _RUN_MARKERS:
        for path in input_dir.rglob(pattern):
            found.append(path.parent)
    unique = []
    seen = set()
    for directory in found:
        key = str(directory.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(directory)
    unique.sort(key=lambda p: str(p).lower())
    return unique


def missing_sample_hint(input_dir: Path) -> str:
    """建集找不到 CSV 时的说明：常见原因是指到了单个叶子目录、或者目录里还是 xlsx 模板。"""
    input_dir = Path(input_dir)
    xlsx = list(input_dir.rglob("Stu_ZZGLRHDL_*.xlsx")) + list(input_dir.rglob("Stu_SJZS_*.xlsx"))
    lines = [
        f"No CSV samples under {input_dir}.",
        "Expected: .../1-200/...-1-200_001/_1  or  .../1-200/...-600-end_264/_3",
        "Pass --input at kemu6_data_72mb. Scene indices need not be consecutive; missing folders are skipped.",
        "Each _1/_2/_3 folder is a repeat of the same scene+strategy, not LJCL 1/2/3.",
    ]
    if xlsx:
        lines.append(f"Found {len(xlsx)} xlsx table(s) and ignored them, e.g. {xlsx[0]}.")
    return " ".join(lines)


def load_run(directory: Path, official: "OfficialRates | None" = None) -> dict:
    rhdl = read_family_dir(directory, "Stu_ZZGLRHDL")
    health = read_family_dir(directory, "HealthState")
    lj = read_family_dir(directory, "Stu_ZZGLLJ")
    sjzs = read_family_dir(directory, "Stu_SJZS")
    if not rhdl:
        raise ValueError(f"Missing Stu_ZZGLRHDL_*.csv in {directory}")
    meta_path = directory / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    official_rate = None if official is None else official.lookup_rate(directory)
    official_strategy = None if official is None else official.lookup_strategy(directory)
    sjzs_strategy = parse_strategy_label(sjzs[0].get("S_LJCL")) if sjzs else None
    if official_strategy is not None:
        strategy = official_strategy
        strategy_source = "official_csv"
    elif sjzs_strategy is not None:
        strategy = sjzs_strategy
        strategy_source = "sjzs"
    elif meta.get("strategy") is not None:
        strategy = int(meta["strategy"])
        strategy_source = "meta"
    else:
        inferred = _strategy_from_name(directory)
        if inferred is None:
            raise ValueError(
                f"Cannot tell LJCL for {directory}: need 结果表「策略」列 "
                "(闭合时间最短/拦截性能最佳/综合效益最优), Stu_SJZS.S_LJCL, "
                "or a folder like sample_000_s1. `_1/_2/_3` are replicates, not strategy numbers."
            )
        strategy = inferred
        strategy_source = "folder"
    times = sorted({t for t in (as_float(row.get("FZTime")) for row in rhdl) if t is not None})
    if not times:
        raise ValueError(f"No FZTime values in {directory}")
    n_blue = len({row["TargetUnitID"] for row in rhdl if row.get("TargetUnitID") is not None})
    if health:
        n_blue = max(
            n_blue,
            len(
                {
                    row["UnitID"]
                    for row in health
                    if str(row.get("UnitTeam") or "").strip() == "蓝方" and row.get("UnitID") is not None
                }
            ),
        )
    intercepted = len({
        as_int(row.get("UnitID"))
        for row in health
        if str(row.get("UnitTeam") or "").strip() == "蓝方"
        and as_int(row.get("UnitID")) is not None
        and as_float(row.get("HealthPoint")) is not None
        and as_float(row.get("HealthPoint")) <= 0.0
    })
    last_t = times[-1]
    alive_end = len({row["TargetUnitID"] for row in rhdl if as_float(row.get("FZTime")) == last_t and row.get("TargetUnitID") is not None})
    leaked = max(0, n_blue - intercepted - alive_end)
    cost = 0.0
    seen = set()
    for row in lj:
        if not row.get("B_Ygz"):
            continue
        key = (row.get("Time"), row.get("UI_FSChandle"))
        if key in seen:
            continue
        seen.add(key)
        cost += as_float(row.get("D_Costgy"), 0.0)
    scene_id = int(meta.get("scene_id", _scene_id_from_name(directory)))
    rate = official_rate
    payload = {
        "directory": directory,
        "scene_id": scene_id,
        "strategy": int(strategy),
        "strategy_source": strategy_source,
        "rhdl": rhdl,
        "health": health,
        "lj": lj,
        "times": times,
        "n_blue": int(meta.get("n_blue", n_blue)),
        "intercepted": int(meta.get("intercepted", intercepted)),
        "leaked": int(meta.get("leaked", leaked)),
        "cost": float(meta.get("cost", cost)),
        "dt": float(meta.get("dt", infer_dt(times))),
        "official_intercept_rate": None if rate is None else float(rate),
        "official_rate_list": [] if rate is None else [float(rate)],
        "replicate_id": _replicate_from_name(directory),
        "replicates": [str(directory)],
        "label_source": "health",
    }
    if rate is not None:
        n = max(1, int(payload["n_blue"]))
        payload["intercepted"] = int(round(float(rate) * n))
        payload["leaked"] = max(0, n - int(payload["intercepted"]))
        payload["label_source"] = "official_csv"
    return payload


def outcome_from_run(run: dict) -> dict:
    """t0 标签：有官方拦截率就用它，否则用 HealthState 统计。"""
    n = max(1, int(run["n_blue"]))
    rate = run.get("official_intercept_rate")
    if rate is not None:
        intercepted = float(rate) * n
        leaked = max(0.0, (1.0 - float(rate)) * n)
        return utility_from_outcome(intercepted, leaked, run["cost"], n)
    return utility_from_outcome(run["intercepted"], run["leaked"], run["cost"], n)


def inventory_runs(grouped: dict) -> list[dict]:
    """建集 summary 里列出实际吃进去的目录，方便核对 _1/_2/_3 有没有漏。"""
    rows = []
    for scene_id, by_strategy in sorted(grouped.items()):
        for strategy, run in sorted(by_strategy.items()):
            rows.append(
                {
                    "scene_id": int(scene_id),
                    "strategy": int(strategy),
                    "strategy_source": run.get("strategy_source"),
                    "directory": str(run["directory"]),
                    "n_replicates": len(run.get("replicates") or [run["directory"]]),
                    "replicates": list(run.get("replicates") or [str(run["directory"])]),
                    "n_ticks": len(run["times"]),
                    "dt": float(run.get("dt", 0.5)),
                    "n_blue": int(run["n_blue"]),
                    "label_source": run.get("label_source", "health"),
                    "official_intercept_rate": run.get("official_intercept_rate"),
                }
            )
    return rows


def _is_replicate_leaf(name: str) -> bool:
    return _STRATEGY_LEAF.fullmatch(name) is not None


def _scene_folder(directory: Path) -> Path:
    """`_1/_2/_3` 的父目录才是场景；合成 sample_000_s1 则目录本身就是场景。"""
    if _is_replicate_leaf(directory.name):
        return directory.parent
    return directory


def _scene_batch_index(folder_name: str) -> tuple[int, int] | None:
    """从场景文件夹名取 (批次起点, 序号)。

    `...-1-200_179` → (0, 179)；`...-200-400_001` → (200, 1)；
    `...-600-end_264` → (600, 264)。父目录叫 1-200 也以文件名为准。
    """
    match = _SCENE_BATCH_INDEX.search(folder_name)
    if not match:
        return None
    start = int(match.group(1))
    offset = 0 if start <= 1 else start
    return offset, int(match.group(3))


def _batch_label(folder_name: str) -> str | None:
    match = _SCENE_BATCH_INDEX.search(folder_name)
    if not match:
        return None
    end = match.group(2)
    end = "end" if end.lower() == "end" else end
    return f"{match.group(1)}-{end}"


def _batch_offset(path: Path) -> int:
    """批次起点。优先场景文件夹名里的 600-end / 1-200，再看父目录（从右往左）。"""
    folder = _scene_folder(path)
    parsed = _scene_batch_index(folder.name)
    if parsed:
        return parsed[0]
    for part in reversed(path.parts):
        open_match = _BATCH_OPEN.fullmatch(part)
        if open_match:
            return int(open_match.group(1))
        match = _BATCH_RANGE.fullmatch(part)
        if match:
            start = int(match.group(1))
            return 0 if start <= 1 else start
    return 0


def _has_batch_folder(path: Path) -> bool:
    folder = _scene_folder(path)
    if _scene_batch_index(folder.name) is not None:
        return True
    return any(
        _BATCH_RANGE.fullmatch(part) is not None or _BATCH_OPEN.fullmatch(part) is not None
        for part in path.parts
    )


def _scene_id_from_name(directory: Path) -> int:
    """场景编号跨批次唯一，按文件夹名里的序号，不要求连续。

    `.../1-200/...-1-200_001/_1` → 1
    `.../1-200/...-600-end_001/_1` → 601（即使父目录是 1-200）
    `...-600-end_264/_3` → 864
    缺 179 就没有 scene_id 179，后面的 180 仍是 180，不会前移。
    """
    leaf = Path(directory)
    folder = _scene_folder(leaf)
    parsed = _scene_batch_index(folder.name)
    if parsed:
        return parsed[0] + parsed[1]
    trailing = _TRAILING_INDEX.search(folder.name)
    if trailing and (_has_batch_folder(folder) or _is_replicate_leaf(leaf.name)):
        return _batch_offset(folder) + int(trailing.group(1))
    name = folder.name
    match = re.search(r"(?:sample|scene|样本)[_-]?(\d+)", name, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    if trailing:
        return _batch_offset(folder) + int(trailing.group(1))
    for token in name.replace("-", "_").split("_"):
        if token.isdigit():
            return int(token)
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100000


def inventory_index_gaps(run_dirs: list[Path], max_items: int = 80) -> list[dict]:
    """按批次报告缺号、缺重复。只统计实际扫到的 min..max 之间的洞，不臆造 1..200。"""
    by_batch: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    for directory in run_dirs:
        folder = _scene_folder(directory)
        label = _batch_label(folder.name)
        parsed = _scene_batch_index(folder.name)
        if label is None or parsed is None:
            continue
        replicate = _replicate_from_name(directory)
        if replicate is not None:
            by_batch[label][parsed[1]].add(int(replicate))
        else:
            by_batch[label][parsed[1]].add(0)
    reports = []
    for label in sorted(by_batch):
        indices = by_batch[label]
        found = sorted(indices)
        missing_scenes = [i for i in range(found[0], found[-1] + 1) if i not in indices]
        incomplete = []
        for index in found:
            have = {f"_{r}" for r in indices[index] if r in {1, 2, 3}}
            if have and have != {"_1", "_2", "_3"}:
                incomplete.append(
                    {
                        "index": index,
                        "have": sorted(have),
                        "missing": sorted({"_1", "_2", "_3"} - have),
                    }
                )
        reports.append(
            {
                "batch": label,
                "n_scenes": len(found),
                "index_min": found[0],
                "index_max": found[-1],
                "n_missing_scenes": len(missing_scenes),
                "missing_scene_indices": missing_scenes[:max_items],
                "n_incomplete_replicates": len(incomplete),
                "incomplete_replicates": incomplete[:max_items],
            }
        )
    return reports


def _replicate_from_name(directory: Path) -> int | None:
    match = _STRATEGY_LEAF.fullmatch(Path(directory).name)
    if match:
        return int(match.group(1))
    return None


def _strategy_from_name(directory: Path) -> int | None:
    name = Path(directory).name
    if _is_replicate_leaf(name):
        return None
    match = re.search(r"(?:^|[_\-])s(?:trategy)?[_-]?([123])$", name, flags=re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r"策略\s*([123])", name)
    if match:
        return int(match.group(1))
    return None


def _path_lookup_keys(path: Path) -> list[str]:
    text = str(path).replace("\\", "/").strip().strip("/")
    parts = [part for part in Path(text.replace("\\", "/")).parts if part not in {"/", "\\"}]
    if parts and len(parts[0]) == 2 and parts[0][1] == ":":
        parts = parts[1:]
    keys: list[str] = []
    seen: set[str] = set()
    candidates = [text.lower(), "/".join(parts).lower()]
    start = 1 if (parts and not _is_replicate_leaf(parts[-1])) else 2
    for n in range(start, min(6, len(parts) + 1)):
        candidates.append("/".join(parts[-n:]).lower())
    for key in candidates:
        key = key.replace("\\", "/").strip("/")
        if key and key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def _parse_rate(raw) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip().strip("%").replace(",", "")
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if value > 1.5:
        value = value / 100.0
    if value < 0.0 or value > 1.0:
        return None
    return value


def _pick_column(headers: list[str], needles: tuple[str, ...]) -> str | None:
    lowered = [(header, header.lower()) for header in headers]
    for header, low in lowered:
        if any(needle.lower() in header or needle.lower() in low for needle in needles):
            return header
    return None


def _normalize_replicate_token(raw) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    match = re.fullmatch(r"_?([123])", text)
    if match:
        return f"_{match.group(1)}"
    return text


def _read_loose_csv(path: Path) -> tuple[list[str], list[dict]]:
    return read_csv_table(path)


class OfficialRates:
    """结果 CSV：路径(+实验次数) → 官方拦截率和策略类型。"""

    def __init__(
        self,
        rates: dict[str, float],
        strategies: dict[str, int],
        source: Path | None,
        columns: list[str],
        n_rows: int,
    ):
        self._rates = rates
        self._strategies = strategies
        self.source = source
        self.columns = columns
        self.n_rows = n_rows

    @classmethod
    def empty(cls) -> "OfficialRates":
        return cls({}, {}, None, [], 0)

    def _get(self, mapping: dict, directory: Path):
        for key in _path_lookup_keys(directory):
            if key in mapping:
                return mapping[key]
        return None

    def lookup(self, directory: Path) -> float | None:
        return self.lookup_rate(directory)

    def lookup_rate(self, directory: Path) -> float | None:
        return self._get(self._rates, directory)

    def lookup_strategy(self, directory: Path) -> int | None:
        return self._get(self._strategies, directory)


def find_results_csv(input_dir: Path, explicit: Path | None = None) -> Path | None:
    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"Results CSV not found: {path}")
        return path
    folders = [Path(input_dir), Path(input_dir).parent]
    seen: set[str] = set()
    for folder in folders:
        key = str(folder.resolve()) if folder.exists() else str(folder)
        if key in seen:
            continue
        seen.add(key)
        for name in _RESULTS_NAMES:
            candidate = folder / name
            if candidate.exists():
                return candidate
        hits = sorted(folder.glob("*结果*.csv")) + sorted(folder.glob("*拦截率*.csv"))
        hits = [path for path in hits if path.name != "training_samples.csv"]
        if hits:
            return hits[0]
    return None


def _looks_like_sample_path(text: str) -> bool:
    if not text:
        return False
    if "/" in text or "\\" in text:
        return True
    name = Path(text).name
    return _TRAILING_INDEX.search(name) is not None or _is_replicate_leaf(name)


def load_official_rates(csv_path: Path | None) -> OfficialRates:
    if csv_path is None:
        return OfficialRates.empty()
    headers, rows = _read_loose_csv(Path(csv_path))
    path_col = _pick_column(headers, ("实验名称", "路径", "目录", "path", "file", "样本", "folder", "名称"))
    replicate_col = _pick_column(headers, ("实验次数", "次数", "replicate", "重复"))
    rate_col = _pick_column(headers, ("总拦截率", "拦截率", "intercept", "成功率"))
    strategy_col = _pick_column(headers, ("策略", "类型", "ljcl", "LJCL"))
    if path_col is None and headers:
        path_col = headers[0]
    if rate_col is None:
        rate_col = _pick_column(headers, ("rate", "结果"))
    rates: dict[str, float] = {}
    strategies: dict[str, int] = {}
    prev_path = ""
    prev_strategy = ""
    for row in rows:
        raw_path = str(row.get(path_col) or "").strip().strip('"') if path_col else ""
        if _looks_like_sample_path(raw_path):
            prev_path = raw_path
        elif prev_path:
            raw_path = prev_path
        raw_rep = ""
        if replicate_col:
            raw_rep = str(row.get(replicate_col) or "").strip()
        replicate = _normalize_replicate_token(raw_rep)
        raw_strategy = str(row.get(strategy_col) or "").strip() if strategy_col else ""
        if parse_strategy_label(raw_strategy) is not None:
            prev_strategy = raw_strategy
        elif prev_strategy:
            raw_strategy = prev_strategy
        rate = _parse_rate(row.get(rate_col) if rate_col else None)
        strategy = parse_strategy_label(raw_strategy)
        if not _looks_like_sample_path(raw_path) or (rate is None and strategy is None):
            continue
        scene = Path(raw_path)
        keys = list(_path_lookup_keys(scene))
        if replicate:
            keys = list(_path_lookup_keys(scene / replicate)) + keys
        for key in keys:
            if rate is not None:
                rates[key] = rate
            if strategy is not None:
                strategies[key] = strategy
    return OfficialRates(rates, strategies, Path(csv_path), headers, len(rows))


def _merge_replicate(base: dict, extra: dict) -> None:
    """同一场景同一策略的重复实验：保留第一份轨迹，拦截率取平均。"""
    base.setdefault("replicates", [str(base["directory"])])
    extra_dir = str(extra["directory"])
    if extra_dir not in base["replicates"]:
        base["replicates"].append(extra_dir)
    rates = list(base.get("official_rate_list") or [])
    extra_rate = extra.get("official_intercept_rate")
    if extra_rate is not None:
        rates.append(float(extra_rate))
    if rates:
        base["official_rate_list"] = rates
        base["official_intercept_rate"] = float(sum(rates) / len(rates))
        n = max(1, int(base["n_blue"]))
        base["intercepted"] = int(round(base["official_intercept_rate"] * n))
        base["label_source"] = "official_csv"


def load_grouped_runs(input_dir: Path, results_csv: Path | None = None) -> tuple[dict[int, dict[int, dict]], dict]:
    """按 scene_id × 策略收样本。`_1/_2/_3` 是重复实验，会合并。"""
    input_dir = Path(input_dir)
    found = find_results_csv(input_dir, results_csv)
    official = load_official_rates(found)
    grouped: dict[int, dict[int, dict]] = defaultdict(dict)
    n_official = 0
    n_official_strategy = 0
    n_leaf = 0
    load_errors = []
    run_dirs = discover_run_dirs(input_dir)
    n_dirs = len(run_dirs)
    for i, directory in enumerate(run_dirs, start=1):
        if n_dirs >= 20 and (i == 1 or i % 50 == 0 or i == n_dirs):
            from src.finals.progress import render_progress
            render_progress(i, n_dirs, "load runs")
        try:
            run = load_run(directory, official=official)
        except (TypeError, ValueError, KeyError, IndexError, AttributeError) as exc:
            load_errors.append({"directory": str(directory), "error": str(exc)})
            continue
        n_leaf += 1
        if run.get("official_intercept_rate") is not None:
            n_official += 1
        if run.get("strategy_source") == "official_csv":
            n_official_strategy += 1
        scene_id = int(run["scene_id"])
        strategy = int(run["strategy"])
        if strategy in grouped[scene_id]:
            _merge_replicate(grouped[scene_id][strategy], run)
            continue
        grouped[scene_id][strategy] = run
    n_triple = sum(1 for by_strategy in grouped.values() if len(by_strategy) >= 3)
    n_pair = sum(1 for by_strategy in grouped.values() if len(by_strategy) >= 2)
    n_pairs = sum(len(by_strategy) for by_strategy in grouped.values())
    n_with_reps = sum(
        1
        for by_strategy in grouped.values()
        for run in by_strategy.values()
        if len(run.get("replicates") or []) >= 3
    )
    meta = {
        "results_csv": None if found is None else str(found),
        "n_official_rate_rows": official.n_rows,
        "n_official_matched": n_official,
        "n_official_strategy_matched": n_official_strategy,
        "n_runs": n_leaf,
        "n_scene_strategy_pairs": n_pairs,
        "n_scenes": len(grouped),
        "n_scenes_with_3_strategies": n_triple,
        "n_scenes_with_2plus_strategies": n_pair,
        "n_scenes_with_3_replicates": n_with_reps,
        "index_gaps": inventory_index_gaps(run_dirs),
        "n_load_errors": len(load_errors),
        "load_errors": load_errors[:40],
        "conflicts": [],
        "official_columns": official.columns,
    }
    return grouped, meta


def time_at_or_after(times: list[float], target: float) -> float:
    if not times:
        return float(target)
    for time in times:
        if time + 1e-9 >= target:
            return float(time)
    return float(times[-1])


def _already_intercepted(health: list[dict], time: float) -> int:
    return len({
        as_int(row.get("UnitID"))
        for row in health
        if str(row.get("UnitTeam") or "").strip() == "蓝方"
        and as_int(row.get("UnitID")) is not None
        and as_float(row.get("Time")) is not None
        and as_float(row.get("Time")) <= as_float(time, 0.0)
        and as_float(row.get("HealthPoint")) is not None
        and as_float(row.get("HealthPoint")) <= 0.0
    })


def _sample_row(scene_id: int, tick: float, strategy: int, features: np.ndarray, outcome: dict, group_id: str, kind: str) -> dict:
    row = {
        "scene_id": scene_id,
        "time": tick,
        "strategy": strategy,
        "group_id": group_id,
        "kind": kind,
        **outcome,
    }
    for name, value in zip(FEATURE_NAMES, features):
        row[name] = float(value)
    return row


def candidate_times(times: list[float], stride_s: float = 2.0, dt: float = 0.5) -> list[float]:
    """建集不要扫每一条 0.5s 记录。约 stride_s 秒取一个候选点，t0 一定保留。"""
    if not times:
        return []
    step = max(1, int(round(float(stride_s) / max(float(dt), 1e-6))))
    out = [times[0]]
    for time in times[1::step]:
        if time not in out:
            out.append(time)
    if times[-1] not in out:
        out.append(times[-1])
    return out


def decision_times(times: list[float], delta: float) -> list[float]:
    """按决策间隔 Δ 取点，和现场每 15/20 秒触发对齐。"""
    if not times:
        return []
    out = [times[0]]
    target = times[0] + float(delta)
    for time in times[1:]:
        if time + 1e-9 >= target:
            out.append(time)
            target = time + float(delta)
    if times[-1] not in out:
        out.append(times[-1])
    return out


def _append_run_samples(rows: list[dict], run: dict, stride: int) -> None:
    times = candidate_times(run["times"], stride_s=max(1.0, float(stride)), dt=float(run.get("dt", 0.5)))
    for time in times:
        snap = snapshot_from_tables(run["rhdl"], run["health"], run["lj"], time, run["strategy"])
        features = extract_features(snap)
        already = _already_intercepted(run["health"], time)
        remaining_int = max(0, run["intercepted"] - already)
        outcome = utility_from_outcome(remaining_int, run["leaked"], run["cost"], max(1, len(snap.targets)))
        if abs(time - run["times"][0]) < 1e-9:
            outcome = outcome_from_run(run)
            group_id = f"scene{run['scene_id']}|t0"
            kind = "t0"
        else:
            group_id = f"scene{run['scene_id']}|t{time:g}|s{run['strategy']}"
            kind = "onpolicy"
        rows.append(_sample_row(run["scene_id"], time, run["strategy"], features, outcome, group_id, kind))


def _add_forks(rows: list[dict], run: dict, n_ticks: int, seed: int) -> None:
    if len(run["times"]) < 8:
        return
    pick = run["times"][len(run["times"]) // 2]
    snap = snapshot_from_tables(run["rhdl"], run["health"], run["lj"], pick, run["strategy"])
    if len(snap.targets) < 3:
        return
    features = extract_features(snap)
    world0 = world_from_snapshot(snap, n_blue_init=max(run["n_blue"], len(snap.targets)), dt=float(run.get("dt", 1.0)))
    remaining = min(24, max(6, n_ticks // 8))
    group_id = f"scene{run['scene_id']}|fork{pick:g}|from{run['strategy']}"
    for strategy in STRATEGIES:
        rng = np.random.default_rng(seed + 1000 * run["scene_id"] + 10 * int(pick) + strategy)
        logs = run_episode(clone_world(world0), strategy, rng, remaining)
        last = logs[-1]
        outcome = utility_from_outcome(last.intercepted, last.leaked, last.cost, world0.n_blue_init)
        rows.append(_sample_row(run["scene_id"], pick, strategy, features, outcome, group_id, "fork"))


def build_training_set(input_dir: Path, output_dir: Path, with_forks: bool = True, n_ticks: int = 220, stride: int = 8) -> dict:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped, ingest = load_grouped_runs(input_dir)
    if not grouped:
        raise ValueError(missing_sample_hint(input_dir))
    rows = []
    for by_strategy in grouped.values():
        for run in by_strategy.values():
            _append_run_samples(rows, run, stride=stride)
            if with_forks and run["strategy"] == 1:
                _add_forks(rows, run, n_ticks=n_ticks, seed=17)
    fieldnames = ["scene_id", "time", "strategy", "group_id", "kind", "intercept_rate", "leak_rate", "cost", "cost_eff", "utility", *FEATURE_NAMES]
    csv_path = output_dir / "training_samples.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)
    save_training_npz(output_dir, rows)
    summary = {
        "n_rows": len(rows),
        "n_runs": sum(len(by_strategy) for by_strategy in grouped.values()),
        "n_scenes": len(grouped),
        "n_official_matched": ingest.get("n_official_matched", 0),
        "n_features": len(FEATURE_NAMES),
        "kind_counts": {kind: sum(1 for row in rows if row["kind"] == kind) for kind in sorted({row["kind"] for row in rows})},
        "strategy_counts": {str(s): sum(1 for row in rows if row["strategy"] == s) for s in STRATEGIES},
        "csv": str(csv_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
