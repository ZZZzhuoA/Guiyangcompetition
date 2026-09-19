"""把样本表或集成接口输入收成同一套态势快照。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.finals.schema import (
    ENTITY_TYPE_NAMES,
    INTERCEPT_FLAGS,
    TIME_ALIGN_EPS,
    as_bool,
    as_float,
    as_int,
    entity_type_from_flags,
    entity_type_from_name,
    flag_for_entity_type,
)


@dataclass
class LinkView:
    target_id: int
    entity_id: int
    entity_type: int
    entity_name: str
    can_flags: dict
    can_intercept: bool
    p: float
    cost: float
    tm: float
    slant: float
    miss: float
    tracked: bool
    assigned: bool
    channel_ok: bool
    remaining_ok: bool


@dataclass
class TargetView:
    target_id: int
    name: str
    target_type: str
    tracked: bool
    intercepting: bool
    low_alt: bool
    swarm: bool
    source_type: int
    source_handle: int
    az_span: float
    el_span: float
    threat: int
    xdpm: float
    xdtm: float
    xdrm: float
    slant: float
    el: float
    az: float
    slant_rate: float
    el_rate: float
    az_rate: float
    pos: np.ndarray
    vel: np.ndarray
    links: list[LinkView] = field(default_factory=list)


@dataclass
class UnitView:
    unit_id: int
    unit_type: str
    name: str
    team: str
    pos: np.ndarray
    lon: float
    lat: float
    alt: float
    health: float


@dataclass
class Snapshot:
    time: float
    strategy: int | None
    targets: list[TargetView]
    units: list[UnitView]
    links: list[LinkView]

    def red_units(self) -> list[UnitView]:
        return [u for u in self.units if u.team == "红方" or u.health >= 0.5 and u.unit_type.startswith("Type")]

    def blue_units(self) -> list[UnitView]:
        return [u for u in self.units if u.team == "蓝方"]


def _num(row: dict, key: str, default: float = 0.0) -> float:
    return as_float(row.get(key), default)


def _int(row: dict, key: str, default: int = 0) -> int:
    parsed = as_int(row.get(key))
    return default if parsed is None else parsed


def _bool(row: dict, key: str, default: bool = False) -> bool:
    parsed = as_bool(row.get(key))
    return default if parsed is None else parsed


def _first_num(row: dict, *keys: str, default: float = 0.0) -> float:
    for key in keys:
        parsed = as_float(row.get(key))
        if parsed is not None:
            return parsed
    return default


def _is_red_health(row: dict) -> bool:
    team = str(row.get("UnitTeam") or "").strip()
    unit_type = str(row.get("UnitType") or "").strip()
    return team == "红方" or unit_type.startswith("Type")


def align_time(times: list[float], time: float, eps: float = TIME_ALIGN_EPS) -> float | None:
    """把任意查询时刻对齐到记录拍。样本 0.5s 一记，实装可能 0.1s 给帧。

    优先用不超过查询时刻的最近一拍（因果）；若还没到第一条记录，用最近的一拍。
    """
    vals = sorted({as_float(t) for t in times if as_float(t) is not None})
    if not vals:
        return None
    target = as_float(time, 0.0)
    for tick in vals:
        if abs(tick - target) <= eps:
            return tick
    earlier = [tick for tick in vals if tick <= target + eps]
    if earlier:
        return earlier[-1]
    return vals[0]


def _index_by_time(rows: list[dict], key: str) -> dict[float, list[dict]]:
    grouped: dict[float, list[dict]] = {}
    for row in rows:
        tick = as_float(row.get(key))
        if tick is None:
            continue
        grouped.setdefault(tick, []).append(row)
    return grouped


@dataclass
class TableIndex:
    rhdl_by_t: dict
    lj_by_t: dict
    health: list
    times: list


def build_table_index(rhdl: list[dict], health: list[dict], lj: list[dict]) -> TableIndex:
    rhdl_by_t = _index_by_time(rhdl, "FZTime")
    lj_by_t = _index_by_time(lj, "Time")
    times = sorted(rhdl_by_t) or sorted(lj_by_t)
    return TableIndex(rhdl_by_t=rhdl_by_t, lj_by_t=lj_by_t, health=health, times=times)


def _rows_at_time(rows: list[dict], key: str, time: float, eps: float = TIME_ALIGN_EPS) -> list[dict]:
    grouped = _index_by_time(rows, key)
    if not grouped:
        return []
    tick = align_time(list(grouped), time, eps)
    if tick is None:
        return []
    return grouped.get(tick, [])


def _health_rows_for_time(health: list[dict], time: float) -> list[dict]:
    query = as_float(time, 0.0)
    reds_all = [row for row in health if _is_red_health(row) and as_float(row.get("Time")) is not None]
    reds = []
    if reds_all:
        t0 = min(as_float(row.get("Time"), 0.0) for row in reds_all)
        reds = [row for row in reds_all if as_float(row.get("Time")) == t0]
    latest: dict[int, dict] = {}
    for row in health:
        uid = as_int(row.get("UnitID"))
        tick = as_float(row.get("Time"))
        if _is_red_health(row) or uid is None or tick is None:
            continue
        if tick > query:
            continue
        prev = latest.get(uid)
        if prev is None or tick >= as_float(prev.get("Time"), -1.0):
            latest[uid] = row
    return reds + list(latest.values())


def snapshot_from_tables(
    rhdl: list[dict],
    health: list[dict],
    lj: list[dict],
    time: float,
    strategy: int | None = None,
    index: TableIndex | None = None,
) -> Snapshot:
    if index is None:
        index = build_table_index(rhdl, health, lj)
    tick = align_time(index.times, time)
    query = time if tick is None else tick
    aligned = align_time(list(index.rhdl_by_t) or list(index.lj_by_t), query)
    rhdl_t = index.rhdl_by_t.get(aligned, []) if aligned is not None else []
    lj_tick = align_time(list(index.lj_by_t), query)
    lj_t = index.lj_by_t.get(lj_tick, []) if lj_tick is not None else []
    health_t = _health_rows_for_time(index.health, time)
    links = [_link_from_lj(row) for row in lj_t]
    links_by_target: dict[int, list[LinkView]] = {}
    for link in links:
        links_by_target.setdefault(link.target_id, []).append(link)
    targets = []
    for row in rhdl_t:
        target_id = as_int(row.get("TargetUnitID"))
        if target_id is None or as_float(row.get("D_X")) is None:
            continue
        targets.append(
            TargetView(
                target_id=target_id,
                name=str(row.get("TargetUnitName") or f"T{target_id}"),
                target_type="BlueTarget",
                tracked=_bool(row, "B_Sgzmb", True),
                intercepting=_bool(row, "B_l_jbz"),
                low_alt=_bool(row, "B_Sdk"),
                swarm=_bool(row, "B_QMBflag"),
                source_type=_int(row, "Uc_XXYLY"),
                source_handle=_int(row, "Uc_XXYHandle"),
                az_span=_num(row, "D_Bsm"),
                el_span=_num(row, "D_Esm"),
                threat=_int(row, "Ui_Zwx", 1),
                xdpm=_num(row, "D_XDPm"),
                xdtm=_num(row, "D_XDTm"),
                xdrm=_num(row, "D_XDRm"),
                slant=_num(row, "D_R"),
                el=_num(row, "D_E"),
                az=_num(row, "D_B"),
                slant_rate=_num(row, "D_RV"),
                el_rate=_num(row, "D_EV"),
                az_rate=_num(row, "D_BV"),
                pos=np.array([_num(row, "D_X"), _num(row, "D_Y"), _num(row, "D_Z")], dtype=float),
                vel=np.array([_num(row, "D_XV"), _num(row, "D_YV"), _num(row, "D_ZV")], dtype=float),
                links=links_by_target.get(target_id, []),
            )
        )
    units = []
    for row in health_t:
        uid = as_int(row.get("UnitID"))
        if uid is None:
            continue
        units.append(
            UnitView(
                unit_id=uid,
                unit_type=str(row.get("UnitType") or "Type1"),
                name=str(row.get("UnitName") or row.get("UnitType") or ""),
                team=str(row.get("UnitTeam") or "红方"),
                pos=np.array([_num(row, "UnitPos_x"), _num(row, "UnitPos_y"), _num(row, "UnitPos_z")], dtype=float),
                lon=_num(row, "UnitPos_lon"),
                lat=_num(row, "UnitPos_lat"),
                alt=_num(row, "UnitPos_alt"),
                health=_num(row, "HealthPoint", 1.0),
            )
        )
    return Snapshot(time=as_float(time, 0.0), strategy=strategy, targets=targets, units=units, links=links)


def snapshot_from_interface(payload: dict, time: float = 0.0) -> Snapshot:
    units = []
    for row in payload.get("units", payload.get("装备相关数据列表", [])) or []:
        uid = as_int(row.get("UnitID"))
        if uid is None:
            continue
        unit_type = str(row.get("UnitType") or "Type1")
        units.append(
            UnitView(
                unit_id=uid,
                unit_type=unit_type,
                name=unit_type,
                team="红方",
                pos=np.array(
                    [_num(row, "UnitPos_x"), _num(row, "UnitPos_y"), _num(row, "UnitPos_z")],
                    dtype=float,
                ),
                lon=_num(row, "UnitPos_lon"),
                lat=_num(row, "UnitPos_lat"),
                alt=_num(row, "UnitPos_alt"),
                health=1.0,
            )
        )
    targets = []
    links = []
    for row in payload.get("targets", payload.get("目标相关数据列表", [])) or []:
        target_id = as_int(row.get("TargetUnitID"))
        if target_id is None or as_float(row.get("D_X")) is None:
            continue
        pos = np.array([_num(row, "D_X"), _num(row, "D_Y"), _num(row, "D_Z")], dtype=float)
        vel = np.array([_num(row, "D_XV"), _num(row, "D_YV"), _num(row, "D_ZV")], dtype=float)
        target_links = []
        for item in row.get("Klj_list", []) or []:
            entity_id = as_int(item.get("UI_FSChandle"))
            if entity_id is None:
                entity_id = as_int(item.get("UI_Hlhandle"), 0)
            entity_name = str(item.get("Str_HLMC") or "Type1")
            flags = {name: _bool(item, name) for name in INTERCEPT_FLAGS}
            raw_type = as_int(item.get("UC_HlType"))
            if raw_type in (1, 2, 3, 4):
                entity_type = raw_type
            else:
                entity_type = entity_type_from_flags(flags) or entity_type_from_name(entity_name)
            expected = flag_for_entity_type(entity_type)
            can = bool(flags.get(expected, False))
            link = LinkView(
                target_id=target_id,
                entity_id=int(entity_id if entity_id is not None else entity_type),
                entity_type=int(entity_type),
                entity_name=entity_name or f"Type{entity_type}",
                can_flags=flags,
                can_intercept=can,
                p=_first_num(item, "D_LJGL", "G_LJGL", default=0.0),
                cost=_num(item, "D_Costgy", 0.5),
                tm=_num(item, "D_Tmzi"),
                slant=_num(item, "D_Rm"),
                miss=_num(item, "D_Pm"),
                tracked=True,
                assigned=False,
                channel_ok=True,
                remaining_ok=True,
            )
            target_links.append(link)
            links.append(link)
        slant = float(np.linalg.norm(pos))
        targets.append(
            TargetView(
                target_id=target_id,
                name=str(row.get("TargetUnitName") or f"T{target_id}"),
                target_type=str(row.get("TargetUnitType") or "BlueTarget"),
                tracked=True,
                intercepting=_bool(row, "B_l_jbz"),
                low_alt=False,
                swarm=False,
                source_type=0,
                source_handle=0,
                az_span=0.0,
                el_span=0.0,
                threat=_int(row, "Ui_Zwx", 1),
                xdpm=0.0,
                xdtm=0.0,
                xdrm=slant,
                slant=slant,
                el=0.0,
                az=0.0,
                slant_rate=0.0,
                el_rate=0.0,
                az_rate=0.0,
                pos=pos,
                vel=vel,
                links=target_links,
            )
        )
    return Snapshot(time=as_float(time, 0.0), strategy=None, targets=targets, units=units, links=links)


def snapshot_to_interface(snapshot: Snapshot) -> dict:
    targets = []
    for target in snapshot.targets:
        klj = []
        for link in target.links:
            item = {
                "UI_FSChandle": link.entity_id,
                "Str_HLMC": link.entity_name or ENTITY_TYPE_NAMES.get(link.entity_type, f"Type{link.entity_type}"),
                "D_LJGL": link.p,
            }
            for name in INTERCEPT_FLAGS:
                item[name] = bool(link.can_flags.get(name, False))
            klj.append(item)
        targets.append(
            {
                "TargetUnitID": target.target_id,
                "TargetUnitType": target.target_type,
                "B_l_jbz": bool(target.intercepting),
                "Ui_Zwx": int(target.threat),
                "D_X": float(target.pos[0]),
                "D_Y": float(target.pos[1]),
                "D_Z": float(target.pos[2]),
                "D_XV": float(target.vel[0]),
                "D_YV": float(target.vel[1]),
                "D_ZV": float(target.vel[2]),
                "Klj_list": klj,
            }
        )
    units = []
    for unit in snapshot.units:
        if unit.team != "红方":
            continue
        units.append(
            {
                "UnitID": unit.unit_id,
                "UnitType": unit.unit_type,
                "UnitPos_x": float(unit.pos[0]),
                "UnitPos_y": float(unit.pos[1]),
                "UnitPos_z": float(unit.pos[2]),
                "UnitPos_lon": unit.lon,
                "UnitPos_lat": unit.lat,
                "UnitPos_alt": unit.alt,
            }
        )
    return {"targets": targets, "units": units}


def _link_from_lj(row: dict) -> LinkView:
    flags = {name: _bool(row, name) for name in INTERCEPT_FLAGS}
    raw_type = _int(row, "UC_HlType", 0)
    entity_type = raw_type if raw_type in (1, 2, 3, 4) else (entity_type_from_flags(flags) or 1)
    expected = flag_for_entity_type(entity_type)
    can = bool(flags.get(expected, False))
    entity_id = as_int(row.get("UI_FSChandle"))
    if entity_id is None:
        entity_id = as_int(row.get("UI_Hlhandle"), entity_type)
    return LinkView(
        target_id=_int(row, "UI_Targethandle"),
        entity_id=int(entity_id if entity_id is not None else entity_type),
        entity_type=int(entity_type),
        entity_name=str(row.get("Str_HLMC") or f"Type{entity_type}"),
        can_flags=flags,
        can_intercept=can,
        p=_first_num(row, "G_LJGL", "D_LJGL"),
        cost=_num(row, "D_Costgy", 0.5),
        tm=_num(row, "D_Tmzi"),
        slant=_num(row, "D_Rm"),
        miss=_num(row, "D_Pm"),
        tracked=_bool(row, "B_Kgz", True),
        assigned=_bool(row, "B_Ygz"),
        channel_ok=_bool(row, "B_TD", True),
        remaining_ok=_bool(row, "G_HLYS", True),
    )

