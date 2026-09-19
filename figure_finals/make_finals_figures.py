"""决赛机制图：现在可画的 F1 / F2 / F3 / F8。

F1 周期触发决策闭环
F2 三表态势到 64 维特征
F3 决策点分叉与奖励（不要学整局 S_LJCL）
F8 单帧拦截匹配图（合成快照示意，不是评估结果）

定量对比图 F4–F7 不在这里画：合成数据上整局拦截率无差异。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = Path(__file__).resolve().parent / "figures"
SOURCE = Path(__file__).resolve().parent / "source"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "baseline_dark": "#484878",
    "baseline_mid": "#7884B4",
    "baseline_soft": "#B4C0E4",
    "ours_tiny": "#E4E4F0",
    "ours_base": "#E4CCD8",
    "ours_large": "#F0C0CC",
    "bg_lilac": "#E0E0F0",
    "bg_aqua": "#E0F0F0",
    "bg_peach": "#F0E0D0",
    "neutral_light": "#D8D8D8",
    "neutral_mid": "#A8A8A8",
    "neutral_dark": "#606060",
    "ink": "#272727",
    "paper": "#FFFFFF",
    "delta_down": "#E53935",
}


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Arial",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 9,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    base = OUT / name
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def rounded_box(ax, xy, wh, text, fc, ec, fontsize=8.5, tc=None, weight="normal"):
    x, y = xy
    w, h = wh
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.02",
            linewidth=1.0,
            edgecolor=ec,
            facecolor=fc,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=tc or PALETTE["ink"],
        weight=weight,
        linespacing=1.25,
    )


def arrow(ax, start, end, color=None, lw=1.15):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=lw,
            color=color or PALETTE["neutral_dark"],
            shrinkA=2,
            shrinkB=2,
        )
    )


def fig1_realtime_loop() -> None:
    """评估是周期触发后输出 LJCL，并锁到下一次触发。"""
    fig, ax = plt.subplots(figsize=(11.2, 3.7))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    boxes = [
        ((0.015, 0.50), (0.145, 0.30), "周期触发\n$t=k\\Delta$", PALETTE["bg_aqua"], PALETTE["baseline_mid"]),
        ((0.195, 0.50), (0.155, 0.30), "当前态势\n三张表", PALETTE["bg_lilac"], PALETTE["baseline_mid"]),
        ((0.390, 0.50), (0.155, 0.30), "特征 $\\mathbf{x}$\n64 维", PALETTE["ours_tiny"], PALETTE["baseline_dark"]),
        ((0.585, 0.46), (0.175, 0.38), "$Q(\\mathbf{x},1)$\n$Q(\\mathbf{x},2)$\n$Q(\\mathbf{x},3)$", PALETTE["ours_base"], PALETTE["baseline_dark"]),
        ((0.800, 0.50), (0.175, 0.30), "LJCL $\\in\\{1,2,3\\}$\n锁到 $t+\\Delta$", PALETTE["ours_large"], PALETTE["baseline_dark"]),
    ]
    for xy, wh, text, fc, ec in boxes:
        weight = "bold" if xy[0] >= 0.58 else "normal"
        rounded_box(ax, xy, wh, text, fc, ec, fontsize=9, weight=weight)

    rounded_box(
        ax,
        (0.195, 0.14),
        (0.350, 0.22),
        "跨触发缓存：新增 / 消失 / 距离与闭合时间变化\n上次 LJCL · 时间进度  time_frac",
        PALETTE["bg_peach"],
        PALETTE["neutral_mid"],
        fontsize=8,
    )

    arrow(ax, (0.160, 0.65), (0.195, 0.65))
    arrow(ax, (0.350, 0.65), (0.390, 0.65))
    arrow(ax, (0.545, 0.65), (0.585, 0.65))
    arrow(ax, (0.760, 0.65), (0.800, 0.65))
    arrow(ax, (0.468, 0.36), (0.468, 0.50), color=PALETTE["baseline_mid"])

    ax.text(0.50, 0.93, "周期触发决策闭环：每次只输出一个拦截策略", ha="center", va="center", fontsize=12, weight="bold", color=PALETTE["ink"])
    ax.text(
        0.50,
        0.04,
        "评估每隔 $\\Delta\\in\\{15,20\\}$ s 调用 recommend()。一次输出锁定到下一次触发；推理时延预算 $<50$ ms。必须在同一进程内反复调用，否则 14 维时序特征全为 0。",
        ha="center",
        va="center",
        fontsize=7.5,
        color=PALETTE["neutral_dark"],
    )
    save(fig, "fig1_realtime_loop")


def fig2_feature_groups() -> None:
    """输入是动态三表加缓存，不是初赛静态 67 维表。"""
    fig, ax = plt.subplots(figsize=(10.6, 5.4))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    rounded_box(ax, (0.04, 0.70), (0.24, 0.16), "Stu_ZZGLRHDL\n蓝方位置 / 速度 / 威胁", PALETTE["bg_aqua"], PALETTE["baseline_mid"], 8.5)
    rounded_box(ax, (0.04, 0.48), (0.24, 0.16), "Stu_ZZGLLJ\n红–蓝可拦截边 / 概率", PALETTE["bg_lilac"], PALETTE["baseline_mid"], 8.5)
    rounded_box(ax, (0.04, 0.26), (0.24, 0.16), "HealthState\n红方实体位置与存活", PALETTE["ours_tiny"], PALETTE["baseline_mid"], 8.5)
    rounded_box(ax, (0.04, 0.06), (0.24, 0.14), "跨触发缓存\n相对上一决策点", PALETTE["bg_peach"], PALETTE["neutral_mid"], 8.5)

    rounded_box(
        ax,
        (0.38, 0.52),
        (0.26, 0.34),
        "snapshot  44 维\n存活 / 跟踪 / 拦截中\n最短闭合时间、群、四类实体\n可拦截边、未覆盖、贪心期望 $p$",
        PALETTE["bg_aqua"],
        PALETTE["baseline_dark"],
        8.5,
    )
    rounded_box(
        ax,
        (0.38, 0.10),
        (0.26, 0.28),
        "temporal  14 维\n新增 / 消失目标\n距离与闭合时间变化\n剩余比、time_frac、上次 LJCL",
        PALETTE["bg_peach"],
        PALETTE["baseline_dark"],
        8.5,
    )
    rounded_box(
        ax,
        (0.72, 0.38),
        (0.24, 0.32),
        "derived  6 维\nurgency\ncoverage_gap\nquality_edge\nscarcity\nthreat / swarm pressure",
        PALETTE["ours_base"],
        PALETTE["baseline_dark"],
        8.5,
    )
    rounded_box(ax, (0.72, 0.10), (0.24, 0.16), "提交输入\n$\\mathbf{x}\\in\\mathbb{R}^{64}$", PALETTE["ours_large"], PALETTE["baseline_dark"], 9, weight="bold")

    for s, e in [
        ((0.28, 0.78), (0.38, 0.76)),
        ((0.28, 0.56), (0.38, 0.70)),
        ((0.28, 0.34), (0.38, 0.64)),
        ((0.28, 0.13), (0.38, 0.24)),
        ((0.64, 0.70), (0.72, 0.58)),
        ((0.64, 0.24), (0.72, 0.50)),
        ((0.84, 0.38), (0.84, 0.26)),
    ]:
        arrow(ax, s, e)

    ax.text(0.50, 0.94, "动态拦截态势到固定 64 维特征", ha="center", fontsize=12, weight="bold", color=PALETTE["ink"])
    ax.text(
        0.50,
        0.89,
        "44（当前帧）+ 14（跨触发）+ 6（压力合成）。进程重启会使 temporal 全变成 0。",
        ha="center",
        fontsize=8,
        color=PALETTE["neutral_dark"],
    )
    save(fig, "fig2_feature_groups")


def fig3_fork_labels() -> None:
    """监督信号来自同一态势上 1/2/3 分叉，而不是整局 S_LJCL。"""
    fig, ax = plt.subplots(figsize=(10.8, 5.8))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.add_patch(Rectangle((0.03, 0.72), 0.94, 0.18, facecolor="#F4F4F4", edgecolor=PALETTE["neutral_light"], linewidth=1.0))
    ax.text(0.05, 0.86, "不要这样监督", fontsize=8.5, color=PALETTE["neutral_dark"], weight="bold")
    rounded_box(ax, (0.08, 0.75), (0.22, 0.12), "整局样本目录\n锁死一种 S_LJCL", "#F7F7F7", PALETTE["neutral_mid"], 8)
    rounded_box(ax, (0.39, 0.75), (0.22, 0.12), "每个时刻贴上\n同一策略标签", "#F7F7F7", PALETTE["neutral_mid"], 8)
    rounded_box(ax, (0.70, 0.75), (0.22, 0.12), "学到历史锁死策略\n≠ 周期重选", "#F7F7F7", PALETTE["neutral_mid"], 8)
    arrow(ax, (0.30, 0.81), (0.39, 0.81), color=PALETTE["neutral_mid"])
    arrow(ax, (0.61, 0.81), (0.70, 0.81), color=PALETTE["neutral_mid"])

    ax.text(0.05, 0.64, "应该这样构造标签", fontsize=8.5, color=PALETTE["baseline_dark"], weight="bold")
    rounded_box(ax, (0.05, 0.38), (0.18, 0.22), "决策点\n态势 $\\mathbf{x}_t$", PALETTE["bg_aqua"], PALETTE["baseline_dark"], 9, weight="bold")

    branches = [
        (0.62, "1  闭合时间最短", PALETTE["baseline_soft"]),
        (0.46, "2  拦截性能最佳", PALETTE["baseline_mid"]),
        (0.30, "3  综合效益最优", PALETTE["ours_base"]),
    ]
    for y, label, fc in branches:
        rounded_box(ax, (0.30, y - 0.015), (0.26, 0.12), f"此后执行策略 {label}", fc, PALETTE["baseline_dark"], 7.8)
        arrow(ax, (0.23, 0.49), (0.30, y + 0.045), color=PALETTE["baseline_dark"], lw=1.0)

    rounded_box(
        ax,
        (0.64, 0.38),
        (0.32, 0.28),
        "$r_\\Delta$：本段新增拦截 / 已出现蓝方\n$r_{\\mathrm{term}}$：此后一直用 $a$ 的整局拦截率\n$\\mathrm{mix}=r_\\Delta+0.25\\,r_{\\mathrm{term}}$",
        PALETTE["ours_tiny"],
        PALETTE["baseline_dark"],
        8,
    )
    for y, _, _ in branches:
        arrow(ax, (0.56, y + 0.045), (0.64, 0.52), color=PALETTE["baseline_dark"], lw=1.0)

    rounded_box(
        ax,
        (0.30, 0.08),
        (0.66, 0.18),
        "势函数 shaping 只以 $\\gamma\\Phi(\\mathbf{x}')-\\Phi(\\mathbf{x})$ 进入奖励，$\\gamma=0.85$，不改变最优策略。\n高威胁漏防权重 0.05；成本不进奖励（接口未必提供 $D\\_Costgy$）。",
        PALETTE["bg_peach"],
        PALETTE["neutral_mid"],
        8,
    )
    arrow(ax, (0.80, 0.38), (0.80, 0.26), color=PALETTE["neutral_dark"])

    ax.text(0.50, 0.955, "标签来自分叉仿真，而不是整局策略号", ha="center", fontsize=12, weight="bold", color=PALETTE["ink"])
    save(fig, "fig3_fork_labels")


def _schematic_intercept_scene():
    """为可读性构图的匹配图，不是真实样本散点。

    四个来袭方向分开摆：西南/东侧进入拦截窗口并连边，西北远距群和北侧单体在窗外无边。
    红方四类实体围在保护要点周围，边按扇区连接，避免全连接扇形重叠。
    """
    blues = [
        {"id": 11, "x": -6.2, "y": -5.0, "threat": 1, "swarm": True},
        {"id": 12, "x": -7.8, "y": -3.6, "threat": 3, "swarm": True},
        {"id": 13, "x": -5.0, "y": -6.4, "threat": 2, "swarm": True},
        {"id": 21, "x": 7.8, "y": -3.8, "threat": 2, "swarm": True},
        {"id": 22, "x": 8.8, "y": -1.6, "threat": 4, "swarm": True},
        {"id": 23, "x": 6.4, "y": -1.8, "threat": 5, "swarm": True},
        {"id": 31, "x": -1.6, "y": 7.4, "threat": 2, "swarm": False},
        {"id": 41, "x": -8.4, "y": 2.2, "threat": 3, "swarm": False},
        {"id": 51, "x": -12.2, "y": 7.0, "threat": 4, "swarm": True},
        {"id": 52, "x": -13.6, "y": 8.4, "threat": 6, "swarm": True},
        {"id": 53, "x": -11.2, "y": 8.8, "threat": 5, "swarm": True},
        {"id": 61, "x": 4.6, "y": 12.2, "threat": 6, "swarm": False},
    ]
    reds = [
        {"id": 1, "etype": 1, "x": 2.6, "y": 0.9},
        {"id": 2, "etype": 2, "x": 0.2, "y": 2.7},
        {"id": 3, "etype": 3, "x": -2.7, "y": 1.0},
        {"id": 4, "etype": 4, "x": 0.6, "y": -2.6},
    ]
    edges = [
        (4, 11, 0.9),
        (4, 13, 0.9),
        (4, 12, 0.7),
        (1, 21, 0.9),
        (1, 23, 0.7),
        (1, 22, 0.7),
        (2, 31, 0.9),
        (3, 41, 0.7),
    ]
    return blues, reds, edges


def _write_fig8_source(blues, reds, edges, path: Path) -> None:
    rows = []
    for tgt in blues:
        vx, vy = -tgt["x"], -tgt["y"]
        n = max((vx * vx + vy * vy) ** 0.5, 1e-9)
        rows.append(
            {
                "role": "blue",
                "id": tgt["id"],
                "x": tgt["x"],
                "y": tgt["y"],
                "vx": vx / n,
                "vy": vy / n,
                "threat": tgt["threat"],
                "swarm": int(tgt["swarm"]),
                "p": "",
            }
        )
    for unit in reds:
        rows.append(
            {
                "role": "red",
                "id": unit["id"],
                "x": unit["x"],
                "y": unit["y"],
                "vx": 0.0,
                "vy": 0.0,
                "threat": "",
                "swarm": "",
                "p": "",
            }
        )
    red_xy = {u["id"]: u for u in reds}
    for red_id, blue_id, p in edges:
        unit = red_xy[red_id]
        rows.append(
            {
                "role": "edge",
                "id": f"{red_id}-{blue_id}",
                "x": unit["x"],
                "y": unit["y"],
                "vx": "",
                "vy": "",
                "threat": "",
                "swarm": "",
                "p": p,
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["role", "id", "x", "y", "vx", "vy", "threat", "swarm", "p"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _inbound_arrow(ax, x, y, color, gap=0.62, length=1.7):
    """箭头画在目标外侧、指向保护要点，避免和内侧的拦截边叠在一起。"""
    vec = np.array([-x, -y], dtype=float)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return
    toward = vec / norm
    outward = -toward
    start = np.array([x, y], dtype=float) + outward * (gap + length)
    end = np.array([x, y], dtype=float) + outward * gap
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=1.05,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=3,
        )
    )


def fig8_intercept_graph() -> None:
    """一个触发时刻的输入是红–蓝匹配图。配色对齐 figure7_draw.py。"""
    colors = {
        "pareto": "#2E86AB",
        "dominated": "#D3D9DF",
        "selected": "#E69A3C",
        "selected_edge": "#8B5E2B",
        "highlight": "#A23B72",
        "grid": "#E8ECEF",
        "text": "#2C3E50",
        "arrow": "#6C7A89",
        "paper": "#F8FAFC",
        "edge_soft": "#A0AAB5",
    }
    mpl.rcParams["font.sans-serif"] = ["SimSun", "Songti SC", "STSong", "Microsoft YaHei", "SimHei"]

    blues, reds, edges = _schematic_intercept_scene()
    _write_fig8_source(blues, reds, edges, SOURCE / "fig8_snapshot.csv")
    blue_xy = {t["id"]: t for t in blues}
    red_xy = {u["id"]: u for u in reds}

    fig, ax = plt.subplots(figsize=(9.0, 6.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor(colors["paper"])
    range_r = 9.6
    ax.add_patch(
        Circle(
            (0, 0),
            range_r,
            fill=False,
            ls="--",
            lw=0.9,
            edgecolor=colors["arrow"],
            zorder=0,
        )
    )
    ax.text(
        6.9,
        6.7,
        "拦截窗口",
        fontsize=7.5,
        color=colors["arrow"],
        ha="left",
        va="bottom",
        fontfamily="SimSun",
    )

    for red_id, blue_id, p in edges:
        red = red_xy[red_id]
        blue = blue_xy[blue_id]
        high_p = p >= 0.85
        ax.plot(
            [red["x"], blue["x"]],
            [red["y"], blue["y"]],
            color=colors["pareto"] if high_p else colors["edge_soft"],
            lw=1.5 if high_p else 0.9,
            alpha=0.85 if high_p else 0.75,
            zorder=1,
            solid_capstyle="round",
        )

    for tgt in blues:
        _inbound_arrow(ax, tgt["x"], tgt["y"], colors["arrow"])
        size = 70 + 110.0 / tgt["threat"]
        if tgt["swarm"]:
            ax.scatter(
                [tgt["x"]],
                [tgt["y"]],
                s=size,
                facecolors="none",
                edgecolors=colors["pareto"],
                linewidths=1.4,
                zorder=4,
            )
        else:
            ax.scatter(
                [tgt["x"]],
                [tgt["y"]],
                s=size,
                c=colors["pareto"],
                edgecolors="white",
                linewidths=1.2,
                zorder=4,
            )

    markers = {1: "s", 2: "D", 3: "^", 4: "P"}
    for unit in reds:
        ax.scatter(
            [unit["x"]],
            [unit["y"]],
            s=100,
            c=colors["selected"],
            marker=markers[unit["etype"]],
            edgecolors=colors["selected_edge"],
            linewidths=1.1,
            zorder=5,
        )

    ax.scatter(
        [0],
        [0],
        s=130,
        marker="X",
        c=colors["highlight"],
        edgecolors=colors["text"],
        linewidths=0.7,
        zorder=6,
    )
    ax.text(-13.8, 11.0, "射程外，无边", fontsize=8, color=colors["arrow"], ha="center", fontfamily="SimSun")
    ax.text(6.8, 12.2, "射程外", fontsize=8, color=colors["arrow"], ha="left", va="center", fontfamily="SimSun")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-18.0, 12.2)
    ax.set_ylim(-11.2, 16.0)
    ax.set_xlabel("X 位置", fontsize=11, color=colors["text"], fontfamily="SimSun", labelpad=8)
    ax.set_ylabel("Y 位置", fontsize=11, color=colors["text"], fontfamily="SimSun", labelpad=8)
    ax.set_title("单帧拦截匹配图", fontsize=13, weight="bold", color=colors["text"], pad=12, fontfamily="SimSun")
    ax.tick_params(labelsize=9, colors=colors["text"])
    for spine in ax.spines.values():
        spine.set_color(colors["text"])
        spine.set_alpha(0.3)
    ax.grid(color=colors["grid"], linestyle="--", linewidth=0.8, alpha=0.6, zorder=0)
    handles = [
        Line2D([0], [0], color=colors["pareto"], lw=1.5, label="可拦截边 $p=0.9$"),
        Line2D([0], [0], color=colors["edge_soft"], lw=0.9, label="可拦截边 $p=0.7$"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=colors["pareto"], markeredgecolor="white", markersize=7, label="蓝方单体"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none", markeredgecolor=colors["pareto"], markersize=7, label="群目标"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=colors["selected"], markeredgecolor=colors["selected_edge"], markersize=7, label="红方 Type1"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=colors["selected"], markeredgecolor=colors["selected_edge"], markersize=6.5, label="红方 Type2"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=colors["selected"], markeredgecolor=colors["selected_edge"], markersize=7, label="红方 Type3"),
        Line2D([0], [0], marker="P", color="none", markerfacecolor=colors["selected"], markeredgecolor=colors["selected_edge"], markersize=8, label="红方 Type4"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor=colors["highlight"], markeredgecolor=colors["text"], markersize=8, label="保护要点"),
    ]
    legend = ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.48),
        fontsize=7.2,
        frameon=False,
        borderaxespad=0.0,
        prop={"family": "SimSun"},
    )
    fig.subplots_adjust(bottom=0.11, right=0.78)
    save(fig, "fig8_intercept_graph")


def main() -> None:
    setup_style()
    fig1_realtime_loop()
    fig2_feature_groups()
    fig3_fork_labels()
    fig8_intercept_graph()
    print(f"saved figures to {OUT}")


if __name__ == "__main__":
    main()
