from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[0]
OUT = ROOT /  "figures"
OUT.mkdir(parents=True, exist_ok=True)


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Arial Unicode MS",
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


PALETTE = {
    "blue": "#4C78A8",
    "orange": "#F58518",
    "green": "#54A24B",
    "red": "#E45756",
    "purple": "#B279A2",
    "teal": "#72B7B2",
    "gray": "#6B7280",
    "light_gray": "#E5E7EB",
    "dark": "#1F2937",
}


def save(fig: plt.Figure, name: str) -> None:
    for ext, kwargs in {
        "png": {"dpi": 300},
        "svg": {},
    }.items():
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def rounded_box(ax, xy, wh, text, fc="#F8FAFC", ec="#CBD5E1", fontsize=9):
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.1,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=PALETTE["dark"])


def arrow(ax, start, end, color="#64748B"):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.1,
            color=color,
            shrinkA=4,
            shrinkB=4,
        )
    )


def fig1_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    boxes = [
        ((0.03, 0.58), (0.15, 0.18), "训练数据\n场景 + 策略 + M1/M2"),
        ((0.23, 0.58), (0.15, 0.18), "数据预处理\n数值化/分组验证"),
        ((0.43, 0.58), (0.15, 0.18), "机理特征\nraw42 + core25"),
        ((0.63, 0.58), (0.15, 0.18), "RF收益预测\nPriorityScore"),
        ((0.83, 0.58), (0.14, 0.18), "最终推荐\n策略1-4"),
        ((0.43, 0.18), (0.15, 0.18), "同场景策略对\n偏好样本"),
        ((0.63, 0.18), (0.15, 0.18), "Pairwise纠偏\n相对胜率"),
    ]
    colors = ["#EFF6FF", "#F8FAFC", "#ECFDF5", "#FFF7ED", "#FDF2F8", "#F1F5F9", "#EEF2FF"]
    for (xy, wh, text), c in zip(boxes, colors):
        rounded_box(ax, xy, wh, text, fc=c)

    arrow(ax, (0.18, 0.67), (0.23, 0.67))
    arrow(ax, (0.38, 0.67), (0.43, 0.67))
    arrow(ax, (0.58, 0.67), (0.63, 0.67))
    arrow(ax, (0.78, 0.67), (0.83, 0.67))
    arrow(ax, (0.50, 0.58), (0.50, 0.36))
    arrow(ax, (0.58, 0.27), (0.63, 0.27))
    arrow(ax, (0.71, 0.36), (0.71, 0.58))
    ax.text(0.71, 0.49, "融合\n0.65/0.35", ha="left", va="center", fontsize=8, color=PALETTE["gray"])
    ax.text(0.5, 0.93, "算法流程：候选策略收益预测与Pairwise纠偏融合", ha="center", va="center", fontsize=13, weight="bold")
    ax.text(0.5, 0.05, "M1为平均拦截率，M2为平均效费比；模型统一枚举四个候选策略并输出最高分策略。", ha="center", va="center", fontsize=8, color=PALETTE["gray"])
    save(fig, "fig1_algorithm_pipeline")


def fig2_data_feature_structure() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    rounded_box(ax, (0.05, 0.62), (0.22, 0.2), "装备资源状态\n6组 x 3字段\n性能1/性能2/起效", "#EFF6FF")
    rounded_box(ax, (0.05, 0.25), (0.22, 0.2), "无人机来袭态势\n6组 x 4字段\n距离/角度/速度/数量", "#FFF7ED")
    rounded_box(ax, (0.38, 0.62), (0.22, 0.2), "资源能力视角\n总能力/起效数\n类型能力/能力均衡", "#ECFDF5")
    rounded_box(ax, (0.38, 0.25), (0.22, 0.2), "态势威胁视角\n目标数/总威胁\n集中度/角度跨度", "#FEF2F2")
    rounded_box(ax, (0.70, 0.44), (0.23, 0.2), "攻防匹配视角\n压力/单目标能力\n类型承压/高威胁匹配", "#F5F3FF")
    rounded_box(ax, (0.38, 0.04), (0.22, 0.12), "核心25工程特征", "#F8FAFC")
    rounded_box(ax, (0.70, 0.04), (0.23, 0.12), "最终输入\nraw42 + core25 = 67维", "#F8FAFC")
    for s, e in [
        ((0.27, 0.72), (0.38, 0.72)),
        ((0.27, 0.35), (0.38, 0.35)),
        ((0.60, 0.72), (0.70, 0.55)),
        ((0.60, 0.35), (0.70, 0.55)),
        ((0.49, 0.25), (0.49, 0.16)),
        ((0.60, 0.10), (0.70, 0.10)),
    ]:
        arrow(ax, s, e)
    ax.text(0.5, 0.93, "样本结构与双视角特征工程", ha="center", fontsize=13, weight="bold")
    ax.text(0.5, 0.88, "从装备资源状态和来袭态势分别挖掘，再通过攻防压力特征连接两个视角", ha="center", fontsize=8, color=PALETTE["gray"])
    save(fig, "fig2_data_feature_structure")


def fig3_baseline_comparison() -> None:
    data = pd.DataFrame(
        [
            ("直接分类", 0.3130, np.nan, np.nan, np.nan),
            ("平均效用", 0.3949, 0.5949, 0.5947, 0.0324),
            ("场景聚类", 0.4359, 0.6103, 0.6317, 0.0308),
            ("核回归", 0.4103, 0.6154, 0.6340, 0.0300),
            ("随机森林", 0.3744, 0.6359, 0.6326, 0.0292),
            ("Pairwise", 0.4256, 0.6615, 0.6470, 0.0279),
            ("多目标", 0.3692, 0.5744, 0.5807, 0.0332),
            ("双视角聚类", 0.4256, 0.6359, 0.6385, 0.0303),
            ("工程特征回归", 0.4410, 0.6769, 0.6807, 0.0259),
            ("最终模型", 0.5591, 0.8065, 0.6889, 0.0370),
        ],
        columns=["方法", "top1", "top2", "soft", "regret"],
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [2.1, 1]})
    x = np.arange(len(data))
    width = 0.27
    axes[0].bar(x - width, data["top1"], width, label="top1", color=PALETTE["blue"])
    axes[0].bar(x, data["top2"], width, label="top2", color=PALETTE["teal"])
    axes[0].bar(x + width, data["soft"], width, label="soft match", color=PALETTE["orange"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(data["方法"], rotation=35, ha="right")
    axes[0].set_ylim(0, 0.9)
    axes[0].set_ylabel("指标值")
    axes[0].set_title("A  排序类指标对比")
    axes[0].grid(axis="y", color=PALETTE["light_gray"], linewidth=0.7)
    axes[0].legend(ncol=3, loc="upper left")
    axes[1].barh(data["方法"], data["regret"], color=PALETTE["purple"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("mean regret（越低越好）")
    axes[1].set_title("B  收益损失")
    axes[1].grid(axis="x", color=PALETTE["light_gray"], linewidth=0.7)
    fig.suptitle("基线方法与最终模型效果对比", fontsize=13, weight="bold", y=1.02)
    save(fig, "fig3_baseline_metrics")


def fig4_feature_ablation() -> None:
    df = pd.read_csv(ROOT / "outputs" / "feature_selection_cv_results.csv")
    order = [
        "raw42",
        "raw42_plus_threat9",
        "raw42_plus_resource12",
        "raw42_plus_compact15",
        "raw42_plus_match10",
        "raw42_plus_all59",
        "raw42_plus_core25",
    ]
    labels = {
        "raw42": "raw42",
        "raw42_plus_threat9": "态势9",
        "raw42_plus_resource12": "资源12",
        "raw42_plus_compact15": "紧凑15",
        "raw42_plus_match10": "匹配10",
        "raw42_plus_all59": "全部59",
        "raw42_plus_core25": "核心25",
    }
    df = df.set_index("feature_set").loc[order].reset_index()
    x = np.arange(len(df))
    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax1.plot(x, df["top1_match"], marker="o", color=PALETTE["blue"], label="top1")
    ax1.plot(x, df["soft_match"], marker="s", color=PALETTE["orange"], label="soft match")
    ax1.plot(x, df["pareto_hit"], marker="^", color=PALETTE["green"], label="Pareto hit")
    ax1.set_ylabel("命中/合理性指标")
    ax1.set_ylim(0.48, 0.84)
    ax1.grid(axis="y", color=PALETTE["light_gray"], linewidth=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels([labels[v] for v in df["feature_set"]])
    ax2 = ax1.twinx()
    ax2.bar(x, df["regret"], width=0.32, color="#CBD5E1", alpha=0.9, label="regret")
    ax2.set_ylabel("regret（越低越好）")
    ax2.set_ylim(0.02, 0.045)
    lines, labs = ax1.get_legend_handles_labels()
    bars, barlabs = ax2.get_legend_handles_labels()
    ax1.legend(lines + bars, labs + barlabs, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    ax1.set_title("特征消融：核心25特征在综合指标上更稳定", fontsize=13, weight="bold", pad=22)
    save(fig, "fig4_feature_ablation")


def fig5_pairwise_correction() -> None:
    df = pd.read_csv(ROOT / "outputs" / "pairwise_correction_cv_results.csv")
    order = [
        "reg_forest_t15_d5",
        "pairwise_corrected_w0.35_p30",
        "pairwise_corrected_w0.35_p30.05",
        "pairwise_corrected_w0.55_p30.05",
        "pairwise_corrected_w0.55_p30.1",
    ]
    labels = ["RF基线", "Pairwise\n0.35", "Pairwise\n0.35+轻惩罚", "Pairwise\n0.55+轻惩罚", "Pairwise\n0.55+强惩罚"]
    df = df.set_index("model").loc[order].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(len(df))
    w = 0.24
    axes[0].bar(x - w, df["top1_match"], w, color=PALETTE["blue"], label="top1")
    axes[0].bar(x, df["top2_hit"], w, color=PALETTE["teal"], label="top2")
    axes[0].bar(x + w, df["soft_match"], w, color=PALETTE["orange"], label="soft match")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylim(0.48, 0.84)
    axes[0].set_ylabel("指标值")
    axes[0].set_title("A  融合纠偏提升排序质量")
    axes[0].grid(axis="y", color=PALETTE["light_gray"], linewidth=0.7)
    axes[0].legend(ncol=3, loc="upper left")
    axes[1].plot(x, df["strategy3_over_prediction"], marker="o", color=PALETTE["red"], linewidth=2)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha="right")
    axes[1].set_ylabel("策略3过推荐次数")
    axes[1].set_title("B  纠偏强度与策略偏置")
    axes[1].grid(axis="y", color=PALETTE["light_gray"], linewidth=0.7)
    fig.suptitle("Pairwise纠偏与策略3校准消融", fontsize=13, weight="bold", y=1.02)
    save(fig, "fig5_pairwise_correction")


def fig6_strategy_distribution() -> None:
    summary = json.loads((ROOT / "outputs" / "summary.json").read_text(encoding="utf-8"))
    cv = summary["cv_metrics"]
    pred = {int(k): int(v) for k, v in cv["pred_strategy_counts"].items()}
    best = {int(k): int(v) for k, v in cv["best_strategy_counts"].items()}
    test = {int(k): int(v) for k, v in summary["test_prediction_counts"].items()}
    strategies = [1, 2, 3, 4]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    x = np.arange(len(strategies))
    w = 0.25
    ax.bar(x - w, [best.get(s, 0) for s in strategies], w, label="验证集真实最优", color=PALETTE["gray"])
    ax.bar(x, [pred.get(s, 0) for s in strategies], w, label="验证集模型推荐", color=PALETTE["blue"])
    ax.bar(x + w, [test.get(s, 0) for s in strategies], w, label="测试集推荐", color=PALETTE["orange"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"策略{s}" for s in strategies])
    ax.set_ylabel("样本数")
    ax.set_title("策略推荐分布与策略偏置诊断", fontsize=13, weight="bold")
    ax.grid(axis="y", color=PALETTE["light_gray"], linewidth=0.7)
    ax.legend(ncol=3, loc="upper left")
    save(fig, "fig6_strategy_distribution")


def fig7_pareto_schematic() -> None:
    strategies = np.array(["策略1", "策略2", "策略3", "策略4"])
    x = np.array([0.62, 0.82, 0.72, 0.55])
    y = np.array([0.78, 0.69, 0.88, 0.58])
    pareto = np.array([True, True, True, False])
    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    ax.scatter(x[~pareto], y[~pareto], s=120, color="#CBD5E1", edgecolor=PALETTE["gray"], label="被支配策略")
    ax.scatter(x[pareto], y[pareto], s=140, color=PALETTE["blue"], edgecolor="white", linewidth=1.2, label="Pareto候选")
    ax.scatter([x[2]], [y[2]], s=210, color=PALETTE["orange"], edgecolor=PALETTE["dark"], linewidth=1.0, label="拦截率优先选择")
    for sx, sy, lab in zip(x, y, strategies):
        ax.text(sx + 0.012, sy + 0.012, lab, fontsize=9)
    ax.plot([0.62, 0.72, 0.82], [0.78, 0.88, 0.69], color=PALETTE["blue"], linestyle="--", linewidth=1)
    ax.annotate("被策略1同时压制", xy=(0.55, 0.58), xytext=(0.42, 0.66), arrowprops=dict(arrowstyle="->", color=PALETTE["gray"]), fontsize=8, color=PALETTE["gray"])
    ax.set_xlim(0.38, 0.9)
    ax.set_ylim(0.52, 0.94)
    ax.set_xlabel("平均效费比 M2（越大越好）")
    ax.set_ylabel("平均拦截率 M1（越大越好）")
    ax.set_title("双指标Pareto选择逻辑示意", fontsize=13, weight="bold")
    ax.grid(color=PALETTE["light_gray"], linewidth=0.7)
    ax.legend(loc="lower right")
    fig.subplots_adjust(bottom=0.18)
    fig.text(0.5, 0.035, "注：该图为机制示意，用于说明非支配筛选和拦截率优先决策，不表示真实样本散点。", ha="center", fontsize=7.5, color=PALETTE["gray"])
    save(fig, "fig7_pareto_schematic")


def main() -> None:
    setup_style()
    fig1_pipeline()
    fig2_data_feature_structure()
    fig3_baseline_comparison()
    fig4_feature_ablation()
    fig5_pairwise_correction()
    fig6_strategy_distribution()
    fig7_pareto_schematic()
    print(f"saved figures to {OUT}")


if __name__ == "__main__":
    main()
