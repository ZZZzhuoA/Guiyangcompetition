"""轻量合成仿真：数量固定、位置随时间变，单策略整局执行。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

from src.finals.schema import (
    ENTITY_SPECS,
    ENTITY_TYPE_NAMES,
    INTERCEPT_FLAGS,
    LEAK_RADIUS,
    LOW_ALT_THRESHOLD,
    PROTECTED_POINT,
    SWARM_MIN_SIZE,
    SWARM_RADIUS,
    TEAM_BLUE,
    TEAM_RED,
    flag_for_entity_type,
)


@dataclass
class BlueAgent:
    unit_id: int
    name: str
    pos: np.ndarray
    vel: np.ndarray
    alive: bool = True
    leaked: bool = False
    tracked: bool = False
    intercepting: bool = False
    observed: bool = False
    ever_observed: bool = False
    appear_time: float = 0.0
    swarm_id: int = -1
    source_handle: int = 0
    threat: int = 1
    hp1_written: bool = False
    hp0_written: bool = False


@dataclass
class RedAgent:
    unit_id: int
    entity_type: int
    pos: np.ndarray
    vel: np.ndarray
    alive: bool = True

    @property
    def name(self) -> str:
        return ENTITY_TYPE_NAMES[self.entity_type]

    @property
    def spec(self) -> dict:
        return ENTITY_SPECS[self.entity_type]


@dataclass
class World:
    time: float
    dt: float
    blues: list[BlueAgent]
    reds: list[RedAgent]
    pending: list[BlueAgent]
    n_blue_init: int
    intercepted: int = 0
    leaked: int = 0
    cost: float = 0.0
    intercept_times: list[float] = field(default_factory=list)


@dataclass
class TickLog:
    time: float
    rhdl: list[dict]
    health: list[dict]
    lj: list[dict]
    intercepted_before: int
    intercepted: int
    leaked: int
    cost: float
    n_alive: int
    assignments: list[tuple[int, int, float, float]]


def angular_span(angles: np.ndarray) -> float:
    if len(angles) <= 1:
        return 0.0
    deg = np.sort((np.degrees(angles) % 360.0))
    gaps = np.diff(np.concatenate([deg, deg[:1] + 360.0]))
    return float(360.0 - np.max(gaps))


def geometry_to_point(pos: np.ndarray, vel: np.ndarray, point: np.ndarray) -> dict:
    rel = pos - point
    r = float(np.linalg.norm(rel))
    if r < 1e-9:
        r = 1e-9
        rel = np.array([1e-9, 0.0, 0.0])
    az = float(np.arctan2(rel[1], rel[0]))
    el = float(np.arcsin(np.clip(rel[2] / r, -1.0, 1.0)))
    radial_speed = float(np.dot(vel, rel) / r)
    az_rate = float((rel[0] * vel[1] - rel[1] * vel[0]) / (rel[0] ** 2 + rel[1] ** 2 + 1e-12))
    el_rate = float((vel[2] * r - rel[2] * radial_speed) / ((r ** 2) * np.cos(el) + 1e-12))
    closing = max(0.0, -radial_speed)
    miss = float(np.linalg.norm(np.cross(rel, vel)) / (np.linalg.norm(vel) + 1e-9))
    if closing > 1e-6:
        t_close = max(0.0, (r - LEAK_RADIUS) / closing)
    else:
        t_close = 1e6
    return {
        "r": r,
        "az": az,
        "el": el,
        "rv": radial_speed,
        "az_rate": az_rate,
        "el_rate": el_rate,
        "miss": miss,
        "t_close": t_close,
    }


def clone_world(world: World) -> World:
    return deepcopy(world)


def cluster_ids(positions: dict[int, np.ndarray], radius: float) -> dict[int, int]:
    ids = list(positions)
    parent = {i: i for i in ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if float(np.linalg.norm(positions[ids[i]] - positions[ids[j]])) <= radius:
                ra, rb = find(ids[i]), find(ids[j])
                if ra != rb:
                    parent[rb] = ra
    roots: dict[int, int] = {}
    labels = {}
    for i in ids:
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
        labels[i] = roots[r]
    return labels


def _inbound_state(rng: np.random.Generator, radius: float, speed: float) -> tuple[np.ndarray, np.ndarray]:
    point = np.array(PROTECTED_POINT, dtype=float)
    az = float(rng.uniform(-np.pi, np.pi))
    el = float(rng.uniform(0.02, 0.18))
    pos = np.array(
        [
            radius * np.cos(el) * np.cos(az),
            radius * np.cos(el) * np.sin(az),
            max(0.12, radius * np.sin(el)),
        ],
        dtype=float,
    )
    direction = point - pos
    direction = direction / (np.linalg.norm(direction) + 1e-9)
    return pos, direction * speed


def make_initial_world(rng: np.random.Generator, n_blue: int = 36, n_red: int = 8, dt: float = 1.0, n_ticks: int = 220) -> World:
    n_blue = max(18, int(n_blue))
    n_swarms = int(rng.integers(4, 8))
    swarm_budget = max(n_swarms * SWARM_MIN_SIZE, int(round(n_blue * rng.uniform(0.65, 0.85))))
    swarm_budget = min(n_blue, swarm_budget)
    sizes = [SWARM_MIN_SIZE] * n_swarms
    extra = swarm_budget - SWARM_MIN_SIZE * n_swarms
    idx = 0
    while extra > 0:
        if sizes[idx % n_swarms] < 12:
            sizes[idx % n_swarms] += 1
            extra -= 1
        idx += 1
        if idx > n_swarms * 20:
            break
    singles = max(0, n_blue - sum(sizes))
    blues: list[BlueAgent] = []
    pending: list[BlueAgent] = []
    uid = 1001
    swarm_id = 0
    wave_times = [0.0]
    for frac in (0.12, 0.28, 0.48):
        if rng.random() < 0.85:
            wave_times.append(float(max(8, int(n_ticks * frac))))
    wave_times = sorted(set(wave_times))
    later = [t for t in wave_times if t > 0]

    def emit(agent: BlueAgent, appear: float) -> None:
        agent.appear_time = appear
        if appear <= 0.0:
            agent.observed = True
            agent.ever_observed = True
            blues.append(agent)
        else:
            pending.append(agent)

    for size in sizes:
        appear = 0.0 if (not later or rng.random() < 0.72) else float(rng.choice(later))
        radius = float(rng.uniform(20.0, 32.0))
        speed = float(rng.uniform(0.08, 0.16))
        center, vel = _inbound_state(rng, radius, speed)
        for _ in range(size):
            offset = rng.normal(0.0, 0.28, size=3)
            offset[2] *= 0.4
            pos = center + offset
            pos[2] = max(0.1, pos[2])
            jitter = rng.normal(0.0, 0.01, size=3)
            agent = BlueAgent(
                unit_id=uid,
                name=f"T{uid}",
                pos=pos,
                vel=vel + jitter,
                swarm_id=swarm_id,
            )
            emit(agent, appear)
            uid += 1
        swarm_id += 1
    for _ in range(singles):
        appear = 0.0 if (not later or rng.random() < 0.55) else float(rng.choice(later))
        radius = float(rng.uniform(18.0, 34.0))
        speed = float(rng.uniform(0.07, 0.18))
        pos, vel = _inbound_state(rng, radius, speed)
        agent = BlueAgent(unit_id=uid, name=f"T{uid}", pos=pos, vel=vel, swarm_id=-1)
        emit(agent, appear)
        uid += 1

    reds = []
    types = np.array([1, 2, 3, 4, 2, 3, 1, 4, 2, 3], dtype=int)
    if rng.random() < 0.45:
        types = np.array([2, 3, 2, 3, 4, 1, 2, 3, 2, 3], dtype=int)
    rng.shuffle(types)
    for i in range(n_red):
        radius = float(rng.uniform(1.5, 5.0))
        az = float(rng.uniform(-np.pi, np.pi))
        pos = np.array([radius * np.cos(az), radius * np.sin(az), float(rng.uniform(0.0, 0.3))], dtype=float)
        reds.append(RedAgent(unit_id=1 + i, entity_type=int(types[i % len(types)]), pos=pos, vel=np.zeros(3)))
    n_total = len(blues) + len(pending)
    return World(time=0.0, dt=dt, blues=blues, reds=reds, pending=pending, n_blue_init=n_total)


def _activate_pending(world: World) -> None:
    keep = []
    for agent in world.pending:
        if agent.appear_time <= world.time:
            agent.observed = True
            agent.ever_observed = True
            world.blues.append(agent)
        else:
            keep.append(agent)
    world.pending = keep


def _refresh_derived(world: World) -> dict:
    point = np.array(PROTECTED_POINT, dtype=float)
    visible = [b for b in world.blues if b.alive and b.observed]
    geos = {b.unit_id: geometry_to_point(b.pos, b.vel, point) for b in world.blues}
    if visible:
        order = sorted(visible, key=lambda b: (geos[b.unit_id]["t_close"], geos[b.unit_id]["r"]))
        for rank, agent in enumerate(order, start=1):
            agent.threat = rank
    positions = {b.unit_id: b.pos for b in visible}
    labels = cluster_ids(positions, SWARM_RADIUS) if positions else {}
    sizes: dict[int, int] = {}
    for cid in labels.values():
        sizes[cid] = sizes.get(cid, 0) + 1
    swarm = {}
    low_alt = {}
    spans = {}
    for agent in visible:
        cid = labels.get(agent.unit_id, -1)
        is_swarm = sizes.get(cid, 1) >= SWARM_MIN_SIZE
        swarm[agent.unit_id] = is_swarm
        low_alt[agent.unit_id] = float(agent.pos[2]) < LOW_ALT_THRESHOLD
        members = [b for b in visible if labels.get(b.unit_id) == cid]
        az = np.array([geos[b.unit_id]["az"] for b in members], dtype=float)
        el = np.array([geos[b.unit_id]["el"] for b in members], dtype=float)
        spans[agent.unit_id] = (
            angular_span(az) if is_swarm else 0.0,
            float(np.degrees(np.max(el) - np.min(el))) if is_swarm and len(el) else 0.0,
        )
    return {
        "geo": geos,
        "swarm": swarm,
        "low_alt": low_alt,
        "spans": spans,
        "positions": positions,
        "n_swarm_groups": sum(1 for size in sizes.values() if size >= SWARM_MIN_SIZE),
    }


def _build_links(world: World, assigned: set[int]) -> list[dict]:
    links = []
    for blue in world.blues:
        if not blue.alive or not blue.observed:
            continue
        for red in world.reds:
            if not red.alive:
                continue
            rel = blue.pos - red.pos
            slant = float(np.linalg.norm(rel))
            geo_e = geometry_to_point(blue.pos, blue.vel, red.pos)
            tracked = slant <= red.spec["track_range"]
            can = slant <= red.spec["range"]
            if not tracked:
                continue
            flags = {name: False for name in INTERCEPT_FLAGS}
            flag = flag_for_entity_type(red.entity_type)
            flags[flag] = bool(can)
            links.append(
                {
                    "target_id": blue.unit_id,
                    "entity_id": red.unit_id,
                    "entity_type": red.entity_type,
                    "entity_name": red.name,
                    "flags": flags,
                    "can": can,
                    "tracked": tracked,
                    "p": float(red.spec["p"]),
                    "cost": float(red.spec["cost"]),
                    "tm": float(geo_e["t_close"]),
                    "slant": slant,
                    "miss": float(geo_e["miss"]),
                    "assigned": red.unit_id in assigned,
                    "channel_ok": red.unit_id not in assigned,
                    "remaining_ok": red.unit_id not in assigned,
                }
            )
    return links


def _assign(strategy: int, world: World, links: list[dict]) -> list[dict]:
    chosen = []
    used_t = set()
    used_e = set()

    def score(link: dict) -> tuple:
        blue = next(b for b in world.blues if b.unit_id == link["target_id"])
        threat_w = 1.0 / max(1, blue.threat)
        if strategy == 1:
            return (-link["tm"], threat_w, link["p"])
        if strategy == 2:
            return (link["p"] * 1.6 + threat_w / (link["tm"] + 0.7), -link["tm"])
        return (link["p"] * threat_w / (link["cost"] + 0.08) / (link["tm"] + 0.5), -link["cost"])

    ranked = sorted([link for link in links if link["can"]], key=score, reverse=True)
    for link in ranked:
        if link["target_id"] in used_t or link["entity_id"] in used_e:
            continue
        chosen.append(link)
        used_t.add(link["target_id"])
        used_e.add(link["entity_id"])
    return chosen


def observe(world: World) -> TickLog:
    _activate_pending(world)
    derived = _refresh_derived(world)
    links = _build_links(world, assigned=set())
    for blue in world.blues:
        blue.intercepting = False
        blue.tracked = False
        blue.source_handle = 0
    for link in links:
        if link["tracked"]:
            blue = next(b for b in world.blues if b.unit_id == link["target_id"])
            blue.tracked = True
            if blue.source_handle == 0:
                blue.source_handle = link["entity_id"]
    rhdl, _, lj = _tables_from_world(world, derived, links)
    return TickLog(
        time=world.time,
        rhdl=rhdl,
        health=flush_health_events(world),
        lj=lj,
        intercepted_before=world.intercepted,
        intercepted=world.intercepted,
        leaked=world.leaked,
        cost=world.cost,
        n_alive=sum(1 for b in world.blues if b.alive and b.observed),
        assignments=[],
    )


def _tables_from_world(world: World, derived: dict, links: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    rhdl = []
    lj = []
    health: list[dict] = []
    for blue in world.blues:
        if not blue.ever_observed:
            continue
        geo = derived["geo"][blue.unit_id]
        if not blue.alive or not blue.observed:
            continue
        swarm = derived["swarm"].get(blue.unit_id, False)
        low_alt = derived["low_alt"].get(blue.unit_id, False)
        az_span, el_span = derived["spans"].get(blue.unit_id, (0.0, 0.0))
        rhdl.append(
            {
                "TargetUnitID": blue.unit_id,
                "TargetUnitName": blue.name,
                "B_Sgzmb": bool(blue.tracked),
                "B_l_jbz": bool(blue.intercepting),
                "B_Sdk": bool(low_alt),
                "B_QMBflag": bool(swarm),
                "Uc_XXYLY": 2,
                "Uc_XXYHandle": int(blue.source_handle),
                "D_Bsm": float(az_span),
                "D_Esm": float(el_span),
                "Ui_Zwx": int(blue.threat),
                "D_XDPm": float(geo["miss"]),
                "D_XDTm": float(geo["t_close"]),
                "D_XDRm": float(geo["r"]),
                "D_R": float(geo["r"]),
                "D_E": float(geo["el"]),
                "D_B": float(geo["az"]),
                "D_RV": float(geo["rv"]),
                "D_EV": float(geo["el_rate"]),
                "D_BV": float(geo["az_rate"]),
                "D_X": float(blue.pos[0]),
                "D_Y": float(blue.pos[1]),
                "D_Z": float(blue.pos[2]),
                "D_XV": float(blue.vel[0]),
                "D_YV": float(blue.vel[1]),
                "D_ZV": float(blue.vel[2]),
                "FZTime": float(world.time),
                "DetectTime": float(world.time),
            }
        )
    for link in links:
        row = {
            "UI_Targethandle": link["target_id"],
            "UI_Hlhandle": link["entity_id"],
            "UC_HlType": link["entity_type"],
            "UI_FSChandle": link["entity_id"],
            "Str_HLMC": link["entity_name"],
            "B_Kgz": bool(link["tracked"]),
            "B_Ygz": bool(link["assigned"]),
            "B_TD": bool(link["channel_ok"]),
            "G_HLYS": bool(link["remaining_ok"]),
            "D_Pm": float(link["miss"]),
            "D_Rm": float(link["slant"]),
            "D_Tmzi": float(link["tm"]),
            "D_Costgy": float(link["cost"]),
            "G_LJGL": float(link["p"]),
            "Time": float(world.time),
        }
        for name in INTERCEPT_FLAGS:
            row[name] = bool(link["flags"][name])
        lj.append(row)
    return rhdl, health, lj


def flush_health_events(world: World) -> list[dict]:
    rows = []
    if abs(float(world.time)) < 1e-12:
        for red in world.reds:
            rows.append(_health_row(red.unit_id, red.name, red.name, TEAM_RED, red.pos, 1.0, world.time))
    for blue in world.blues:
        if not blue.ever_observed:
            continue
        if not blue.hp1_written:
            rows.append(_health_row(blue.unit_id, "BlueTarget", blue.name, TEAM_BLUE, blue.pos, 1.0, world.time))
            blue.hp1_written = True
        if not blue.alive and not blue.hp0_written:
            rows.append(_health_row(blue.unit_id, "BlueTarget", blue.name, TEAM_BLUE, blue.pos, 0.0, world.time))
            blue.hp0_written = True
    return rows


def step(world: World, strategy: int, rng: np.random.Generator) -> TickLog:
    intercepted_before = world.intercepted
    for blue in world.blues:
        if not blue.alive:
            continue
        blue.pos = blue.pos + blue.vel * world.dt
        if float(np.linalg.norm(blue.pos - np.array(PROTECTED_POINT))) <= LEAK_RADIUS:
            blue.leaked = True
            blue.alive = False
            blue.observed = False
            world.leaked += 1
    world.time += world.dt
    _activate_pending(world)
    derived = _refresh_derived(world)
    links = _build_links(world, assigned=set())
    assignments = _assign(strategy, world, links)
    assigned_entities = {item["entity_id"] for item in assignments}
    assigned_targets = {item["target_id"] for item in assignments}
    for blue in world.blues:
        blue.intercepting = blue.unit_id in assigned_targets and blue.alive
        blue.tracked = False
        blue.source_handle = 0
    for link in links:
        if link["tracked"]:
            blue = next(b for b in world.blues if b.unit_id == link["target_id"])
            blue.tracked = True
            if blue.source_handle == 0:
                blue.source_handle = link["entity_id"]
    links = _build_links(world, assigned_entities)
    rhdl, _, lj = _tables_from_world(world, derived, links)

    resolved = []
    for item in assignments:
        success = bool(rng.random() < item["p"])
        world.cost += item["cost"]
        blue = next(b for b in world.blues if b.unit_id == item["target_id"])
        if success and blue.alive:
            blue.alive = False
            blue.intercepting = False
            blue.observed = False
            world.intercepted += 1
            world.intercept_times.append(world.time)
        resolved.append((item["target_id"], item["entity_id"], item["p"], item["cost"]))

    return TickLog(
        time=world.time,
        rhdl=rhdl,
        health=flush_health_events(world),
        lj=lj,
        intercepted_before=intercepted_before,
        intercepted=world.intercepted,
        leaked=world.leaked,
        cost=world.cost,
        n_alive=sum(1 for b in world.blues if b.alive and b.observed),
        assignments=resolved,
    )


def world_from_snapshot(snapshot, n_blue_init: int | None = None, dt: float = 1.0) -> World:
    from src.finals.snapshot import Snapshot

    assert isinstance(snapshot, Snapshot)
    blues = []
    reds = []
    seen_red = set()
    for target in snapshot.targets:
        blues.append(
            BlueAgent(
                unit_id=target.target_id,
                name=target.name,
                pos=np.array(target.pos, dtype=float),
                vel=np.array(target.vel, dtype=float),
                alive=True,
                tracked=target.tracked,
                intercepting=target.intercepting,
                observed=True,
                ever_observed=True,
                hp1_written=True,
                source_handle=target.source_handle,
                threat=target.threat,
            )
        )
    for unit in snapshot.units:
        if unit.team == TEAM_RED or str(unit.unit_type).startswith("Type"):
            if unit.unit_id in seen_red:
                continue
            seen_red.add(unit.unit_id)
            entity_type = 1
            for key, name in ENTITY_TYPE_NAMES.items():
                if name == unit.unit_type:
                    entity_type = key
            for link in snapshot.links:
                if link.entity_id == unit.unit_id:
                    entity_type = link.entity_type
                    break
            reds.append(
                RedAgent(
                    unit_id=unit.unit_id,
                    entity_type=int(entity_type),
                    pos=np.array(unit.pos, dtype=float),
                    vel=np.zeros(3),
                    alive=unit.health >= 0.5,
                )
            )
    if not reds:
        for link in snapshot.links:
            if link.entity_id in seen_red:
                continue
            seen_red.add(link.entity_id)
            reds.append(
                RedAgent(
                    unit_id=int(link.entity_id),
                    entity_type=int(link.entity_type),
                    pos=np.zeros(3),
                    vel=np.zeros(3),
                    alive=True,
                )
            )
    if n_blue_init is None:
        n_blue_init = len(blues)
    return World(time=float(snapshot.time), dt=float(dt), blues=blues, reds=reds, pending=[], n_blue_init=n_blue_init)


def run_episode(world: World, strategy: int, rng: np.random.Generator, n_ticks: int) -> list[TickLog]:
    logs = [observe(world)]
    logs.extend(continue_episode(world, strategy, rng, n_ticks))
    return logs


def continue_episode(world: World, strategy: int, rng: np.random.Generator, n_ticks: int) -> list[TickLog]:
    logs = []
    for _ in range(n_ticks):
        logs.append(step(world, strategy, rng))
        alive = any(b.alive and b.observed for b in world.blues)
        if not world.pending and not alive:
            break
    return logs


def _health_row(unit_id: int, unit_type: str, name: str, team: str, pos: np.ndarray, health: float, time: float) -> dict:
    return {
        "UnitID": unit_id,
        "UnitType": unit_type,
        "UnitName": name,
        "UnitTeam": team,
        "UnitPos_x": float(pos[0]),
        "UnitPos_y": float(pos[1]),
        "UnitPos_z": float(pos[2]),
        "UnitPos_lon": 0.0,
        "UnitPos_lat": 0.0,
        "UnitPos_alt": float(pos[2]),
        "HealthPoint": float(health),
        "Time": float(time),
    }
