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
        self.scenes_root = self.root / "scenes"
        self.done_path = self.root / "done.json"
        self.rows_path = self.root / "rows.jsonl"
        self.dropped_path = self.root / "dropped.jsonl"
        self.extra_path = self.root / "extra.jsonl"
        self._done: set[int] | None = None

    def exists(self) -> bool:
        return self.done_path.exists() or (self.scenes_root.exists() and any(self.scenes_root.glob("*.json")))

    def load(self) -> tuple[set[int], list[dict], list[dict], list[dict]]:
        if not self.exists():
            self._done = set()
            return set(), [], [], []
        # 兼容旧版 append-only 检查点。新检查点按场景单独原子写入，重跑同一
        # 场景不会重复追加，进程在落盘中途退出也只会留下可忽略的 .tmp。
        legacy_done = (
            {int(v) for v in json.loads(self.done_path.read_text(encoding="utf-8")).get("scene_ids", [])}
            if self.done_path.exists()
            else set()
        )
        scene_payloads = []
        file_done = set()
        if self.scenes_root.exists():
            for path in sorted(self.scenes_root.glob("*.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                scene_id = int(payload.get("scene_id", int(path.stem)))
                file_done.add(scene_id)
                scene_payloads.append(payload)

        legacy_rows = _read_jsonl(self.rows_path)
        legacy_dropped = _read_jsonl(self.dropped_path)
        legacy_extra = _read_jsonl(self.extra_path)
        if legacy_extra and len(legacy_extra) != len(legacy_rows):
            raise ValueError("checkpoint extra/rows length mismatch; rerun with --fresh")
        if file_done:
            keep = [int(row.get("scene_id", -1)) not in file_done for row in legacy_rows]
            legacy_rows = [row for row, wanted in zip(legacy_rows, keep) if wanted]
            if legacy_extra:
                legacy_extra = [item for item, wanted in zip(legacy_extra, keep) if wanted]
            legacy_dropped = [row for row in legacy_dropped if int(row.get("scene_id", -1)) not in file_done]

        done = legacy_done | file_done
        rows = legacy_rows
        dropped = legacy_dropped
        extra = legacy_extra
        for payload in scene_payloads:
            rows.extend(payload.get("rows") or [])
            dropped.extend(payload.get("dropped") or [])
            extra.extend(payload.get("extra") or [])
        self._done = set(done)
        return done, rows, dropped, extra

    def save_scene(self, scene_id: int, rows: list[dict], dropped: list[dict], extra: list[dict] | None = None) -> None:
        self.scenes_root.mkdir(parents=True, exist_ok=True)
        if self._done is None:
            if self.done_path.exists():
                self._done = {int(v) for v in json.loads(self.done_path.read_text(encoding="utf-8")).get("scene_ids", [])}
            else:
                self._done = set()
        scene_id = int(scene_id)
        payload = {"scene_id": scene_id, "rows": rows, "dropped": dropped, "extra": extra or []}
        scene_path = self.scenes_root / f"{scene_id:012d}.json"
        scene_tmp = self.scenes_root / f".{scene_id:012d}.json.tmp"
        scene_tmp.write_text(
            json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n",
            encoding="utf-8",
        )
        scene_tmp.replace(scene_path)
        self._done.add(scene_id)
        done_tmp = self.root / ".done.json.tmp"
        done_tmp.write_text(
            json.dumps({"scene_ids": sorted(self._done)}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        done_tmp.replace(self.done_path)

    def clear(self) -> None:
        self._done = None
        if self.root.exists():
            if self.scenes_root.exists():
                for path in self.scenes_root.glob("*"):
                    if path.is_file():
                        path.unlink()
                self.scenes_root.rmdir()
            for path in self.root.glob("*"):
                if path.is_file():
                    path.unlink()
            self.root.rmdir()


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
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
