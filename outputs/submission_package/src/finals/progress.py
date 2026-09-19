"""建集进度条和按场景断点续传。不引入 tqdm。"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


def render_progress(current: int, total: int, prefix: str = "", extra: str = "") -> None:
    total = max(1, int(total))
    current = min(max(0, int(current)), total)
    width = 28
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    suffix = f" {extra}" if extra else ""
    sys.stderr.write(f"\r{prefix} [{bar}] {current}/{total}{suffix}")
    if current >= total:
        sys.stderr.write("\n")
    sys.stderr.flush()


class ProgressBar:
    def __init__(self, total: int, prefix: str = ""):
        self.total = max(0, int(total))
        self.prefix = prefix
        self.current = 0
        self.t0 = time.time()
        if self.total == 0:
            render_progress(0, 1, self.prefix, "nothing to do")
        else:
            render_progress(0, self.total, self.prefix)

    def update(self, step: int = 1, extra: str = "") -> None:
        display_total = max(1, self.total)
        self.current = min(display_total, self.current + step)
        elapsed = max(0.001, time.time() - self.t0)
        rate = self.current / elapsed
        remain = (display_total - self.current) / rate if rate > 0 else 0.0
        eta = f"{elapsed:.0f}s elapsed, ~{remain:.0f}s left"
        clock = f"{extra} | {eta}" if extra else eta
        render_progress(self.current, display_total, self.prefix, clock)

    def close(self) -> None:
        display_total = max(1, self.total)
        if self.current < display_total:
            render_progress(display_total, display_total, self.prefix, "done")


class SceneCheckpoint:
    """每做完一个 scene_id 就落盘。中断后 --resume 跳过已完成场景。"""

    def __init__(self, output_dir: Path):
        self.root = Path(output_dir) / "_ckpt"
        self.done_path = self.root / "done.json"
        self.rows_path = self.root / "rows.jsonl"
        self.dropped_path = self.root / "dropped.jsonl"
        self.extra_path = self.root / "extra.jsonl"
        self._done: set[int] | None = None

    def exists(self) -> bool:
        return self.done_path.exists()

    def load(self) -> tuple[set[int], list[dict], list[dict], list[dict]]:
        if not self.exists():
            self._done = set()
            return set(), [], [], []
        done = {int(v) for v in json.loads(self.done_path.read_text(encoding="utf-8")).get("scene_ids", [])}
        rows = _read_jsonl(self.rows_path)
        dropped = _read_jsonl(self.dropped_path)
        extra = _read_jsonl(self.extra_path)
        self._done = set(done)
        return done, rows, dropped, extra

    def save_scene(self, scene_id: int, rows: list[dict], dropped: list[dict], extra: list[dict] | None = None) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if self._done is None:
            if self.done_path.exists():
                self._done = {int(v) for v in json.loads(self.done_path.read_text(encoding="utf-8")).get("scene_ids", [])}
            else:
                self._done = set()
        self._done.add(int(scene_id))
        _append_jsonl(self.rows_path, rows)
        _append_jsonl(self.dropped_path, dropped)
        if extra:
            _append_jsonl(self.extra_path, extra)
        self.done_path.write_text(json.dumps({"scene_ids": sorted(self._done)}, ensure_ascii=False) + "\n", encoding="utf-8")

    def clear(self) -> None:
        self._done = None
        if self.root.exists():
            for path in self.root.glob("*"):
                path.unlink()
            self.root.rmdir()


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


def _json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)
