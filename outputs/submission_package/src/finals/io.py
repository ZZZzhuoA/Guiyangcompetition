"""读写决赛 CSV。

赛方电脑上的表经常是 GBK/GB18030，不一定是 UTF-8。第一行是表头之后，
有的文件第二、三行是单位或说明，第四行才开始是数据。读的时候只抽 schema
里的列，多出来的列丢掉；缺列填 None。
"""

from __future__ import annotations

import csv
import codecs
import math
import re
from pathlib import Path
from typing import TextIO

from src.finals.schema import HEALTH_COLUMNS, LJ_COLUMNS, RHDL_COLUMNS, SJZS_COLUMNS, as_bool, as_float, as_int, is_blank

_MAX_LEADING_SKIP = 6


FAMILY_COLUMNS = {
    "Stu_ZZGLRHDL": RHDL_COLUMNS,
    "HealthState": HEALTH_COLUMNS,
    "Stu_ZZGLLJ": LJ_COLUMNS,
    "Stu_SJZS": SJZS_COLUMNS,
}

# 赛方列名和本地 schema 不一致时的别名；仍然只映射到本地列。
_COLUMN_ALIASES = {
    "G_LJGL": ("D_LJGL",),
    "S_LJCL": ("LJCL",),
}


def _header_key(raw) -> str:
    if raw is None:
        return ""
    return str(raw).replace("\ufeff", "").strip()


def _parse_value(raw, primitive):
    if is_blank(raw):
        return None
    if isinstance(raw, str):
        raw = raw.strip()
    if primitive == "string":
        text = str(raw).strip()
        return text or None
    if primitive == "bool":
        parsed = as_bool(raw)
        if parsed is None:
            raise ValueError(f"Expected True or False, got {raw!r}")
        return parsed
    if primitive == "int":
        if isinstance(raw, bool):
            return None
        parsed = as_int(raw)
        if parsed is None:
            raise ValueError(f"Expected an integer, got {raw!r}")
        return parsed
    if primitive == "double":
        parsed = as_float(raw)
        if parsed is None:
            raise ValueError(f"Expected a finite number, got {raw!r}")
        return parsed
    raise ValueError(f"Unsupported primitive: {primitive}")


def format_csv_value(value, primitive):
    if value is None:
        return ""
    if primitive == "bool":
        return "True" if value else "False"
    if primitive == "int":
        return str(int(value))
    if primitive == "double":
        number = float(value)
        if number.is_integer():
            return f"{number:.1f}"
        return repr(number)
    return str(value)


def detect_csv_encoding(data: bytes) -> str:
    """先看 BOM / 字节再定 encoding，而不是打开时猜错再报错。"""
    if not data:
        return "utf-8"
    if data.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    if data.startswith(b"\xff\xfe"):
        return "utf-16"
    if data.startswith(b"\xfe\xff"):
        return "utf-16-be"
    if b"\x00" in data[:64]:
        for encoding in ("utf-16", "utf-16-be"):
            try:
                data.decode(encoding)
                return encoding
            except UnicodeDecodeError:
                continue
    try:
        # final=False permits a sampled buffer to end in the middle of a
        # multi-byte UTF-8 character.
        codecs.getincrementaldecoder("utf-8")().decode(data, final=False)
        return "utf-8"
    except UnicodeDecodeError:
        pass
    for encoding in ("gb18030", "gbk", "cp936"):
        try:
            data.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    return "gb18030"


def detect_csv_file_encoding(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """流式探测文件编码，内存恒定。

    过去 `_csv_stream` 会 `read_bytes()` 再构造一份完整 Unicode 字符串，单个大
    CSV 会同时存在 bytes、str 和解析后的行，峰值内存很高。这里最多保留一个
    chunk，并完整扫描 UTF-8 合法性；非 UTF-8 赛方表回退为 GB18030。
    """
    path = Path(path)
    with path.open("rb") as stream:
        head = stream.read(4)
        if head.startswith(b"\xef\xbb\xbf"):
            return "utf-8-sig"
        if head.startswith(b"\xff\xfe"):
            return "utf-16"
        if head.startswith(b"\xfe\xff"):
            return "utf-16-be"
        if not head:
            return "utf-8"
        if b"\x00" in head:
            return "utf-16"
        stream.seek(0)
        decoder = codecs.getincrementaldecoder("utf-8")()
        try:
            while True:
                chunk = stream.read(chunk_size)
                if not chunk:
                    break
                decoder.decode(chunk, final=False)
            decoder.decode(b"", final=True)
            return "utf-8"
        except UnicodeDecodeError:
            return "gb18030"


def decode_csv_text(path: Path) -> str:
    """兼容旧调用方；大表读取应使用 `_csv_stream`。"""
    encoding = detect_csv_file_encoding(path)
    return Path(path).read_text(encoding=encoding, errors="replace")


def _csv_stream(path: Path) -> TextIO:
    """返回磁盘流，不再把整个 CSV 复制到内存。调用方必须关闭。"""
    encoding = detect_csv_file_encoding(path)
    return Path(path).open("r", encoding=encoding, errors="replace", newline="")


def _sniff_dialect(sample: str) -> csv.Dialect:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t")
    except csv.Error:
        return csv.excel


def _typed_score(raw: dict, header_to_schema: dict[str, str], types: dict[str, str]) -> tuple[int, int]:
    ok = 0
    fail = 0
    for header, schema_name in header_to_schema.items():
        primitive = types[schema_name]
        value = raw.get(header)
        if is_blank(value):
            continue
        if primitive == "string":
            continue
        try:
            parsed = _parse_value(value, primitive)
        except (TypeError, ValueError):
            fail += 1
            continue
        if parsed is not None:
            ok += 1
    return ok, fail


def _looks_like_data_row(raw: dict, header_to_schema: dict[str, str], types: dict[str, str]) -> bool:
    if not any(not is_blank(value) for value in raw.values()):
        return False
    ok, fail = _typed_score(raw, header_to_schema, types)
    if ok + fail == 0:
        return True
    return ok > 0 and fail == 0


def read_csv_table(path: Path) -> tuple[list[str], list[dict]]:
    """宽松读：编码探测、分隔符探测。给结果表用，不按 schema 抽列。"""
    rows = []
    with _csv_stream(path) as stream:
        sample = stream.read(8192)
        stream.seek(0)
        reader = csv.DictReader(stream, dialect=_sniff_dialect(sample))
        headers = [_header_key(name) for name in (reader.fieldnames or [])]
        if reader.fieldnames:
            reader.fieldnames = headers
        for raw in reader:
            row = {_header_key(key): (None if value is None else str(value).strip()) for key, value in raw.items()}
            if not any(value for value in row.values() if not is_blank(value)):
                continue
            rows.append(row)
    return headers, rows


def write_csv(path: Path, columns: list[tuple[str, str]], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [name for name, _ in columns]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\r\n")
        writer.writerow(names)
        for row in rows:
            writer.writerow([format_csv_value(row.get(name), primitive) for name, primitive in columns])


def read_csv(path: Path, columns: list[tuple[str, str]], allow_missing: bool = True) -> list[dict]:
    """只抽出 `columns` 里的本地列。赛方多出来的列忽略。

    第一行当表头。其后连续的说明/单位行（通常是第 2、3 行）会被跳过，从第一个
    能按类型解析的数据行开始收。
    """
    names = [name for name, _ in columns]
    types = {name: primitive for name, primitive in columns}
    wanted = set(names)
    records = []
    with _csv_stream(path) as stream:
        sample = stream.read(8192)
        stream.seek(0)
        reader = csv.DictReader(stream, dialect=_sniff_dialect(sample))
        raw_headers = list(reader.fieldnames or [])
        if not raw_headers:
            raise ValueError(f"Missing headers: {path}")
        normalized = [_header_key(h) for h in raw_headers]
        reader.fieldnames = normalized
        header_to_schema: dict[str, str] = {}
        for original, key in zip(raw_headers, normalized):
            if key in wanted and key not in header_to_schema.values() and original not in header_to_schema:
                header_to_schema[key] = key
        present = set(header_to_schema.values())
        for schema_name, aliases in _COLUMN_ALIASES.items():
            if schema_name not in wanted or schema_name in present:
                continue
            for key in normalized:
                if key in aliases:
                    header_to_schema[key] = schema_name
                    present.add(schema_name)
                    break
        if not allow_missing:
            missing = [name for name in names if name not in present]
            if missing:
                raise ValueError(f"Required columns missing in {path}: {missing}")
        skipped = 0
        seen_data = False
        for raw in reader:
            if not seen_data:
                if skipped < _MAX_LEADING_SKIP and not _looks_like_data_row(raw, header_to_schema, types):
                    skipped += 1
                    continue
                seen_data = True
            elif not _looks_like_data_row(raw, header_to_schema, types):
                continue
            row = {name: None for name in names}
            for header, schema_name in header_to_schema.items():
                value = raw.get(header)
                try:
                    row[schema_name] = _parse_value(value, types[schema_name])
                except (TypeError, ValueError):
                    row[schema_name] = None
            if not any(value is not None for value in row.values()):
                continue
            records.append(row)
    return records


def family_from_name(stem: str) -> str:
    match = re.fullmatch(r"(.*)_\d+", stem)
    if not match:
        raise ValueError(f"Expected numbered table name: {stem}")
    family = match.group(1)
    if family not in FAMILY_COLUMNS:
        raise ValueError(f"Unknown table family: {family}")
    return family


def _family_part_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if "_" in stem:
        suffix = stem.rsplit("_", 1)[1]
        try:
            return (int(suffix), stem)
        except ValueError:
            return (10**9, stem)
    return (10**9 + 1, stem)


def read_family_dir(directory: Path, family: str) -> list[dict]:
    columns = FAMILY_COLUMNS[family]
    paths = list(directory.glob(f"{family}_*.csv"))
    bare = Path(directory) / f"{family}.csv"
    if bare.exists():
        paths.append(bare)
    rows = []
    for path in sorted(paths, key=_family_part_key):
        try:
            rows.extend(read_csv(path, columns))
        except (TypeError, ValueError, OSError):
            continue
    return rows


# HealthState 整局只写一个文件；态势/可拦截表按时间刻分段。
UNSEGMENTED_FAMILIES = {"HealthState"}


def write_family_parts(directory: Path, family: str, rows: list[dict], ticks_per_file: int = 200) -> None:
    columns = FAMILY_COLUMNS[family]
    time_keys = [name for name, _ in columns if name.lower().endswith("time")]
    if not rows:
        write_csv(directory / f"{family}_0.csv", columns, [])
        return
    if family in UNSEGMENTED_FAMILIES or not time_keys or ticks_per_file <= 0:
        write_csv(directory / f"{family}_0.csv", columns, rows)
        return
    time_key = time_keys[0]
    grouped: dict = {}
    for row in rows:
        grouped.setdefault(row[time_key], []).append(row)
    times = sorted(grouped)
    for part, start in enumerate(range(0, len(times), ticks_per_file)):
        chunk = []
        for tick in times[start:start + ticks_per_file]:
            chunk.extend(grouped[tick])
        write_csv(directory / f"{family}_{part}.csv", columns, chunk)
