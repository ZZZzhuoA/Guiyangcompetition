from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ========== 设置中文字体 ==========
plt.rcParams['font.sans-serif'] = ['SimSun', 'Songti SC', 'STSong', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# ========== 优化后的色彩调色板 ==========
PALETTE = {
    # 主色系 - 更现代、更协调
    "blue": "#2E6B9E",          # 深蓝 (top1)
    "teal": "#2A9D8F",          # 青绿 (top2)
    "orange": "#E76F51",        # 暖橙 (soft match)
    "red": "#D62828",           # 中国红 (over_prediction)
    "light_gray": "#EAEAEA",    # 浅灰 (网格)
    "dark_gray": "#333333",     # 深灰 (文字)
    
    # 新增辅助色
    "gold": "#F4A261",          # 金色
    "purple": "#6D597A",        # 紫色
    "mint": "#A8D5BA",          # 薄荷绿
}

ROOT = Path(__file__).resolve().parents[0]
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)

def save(fig: plt.Figure, name: str) -> None:
    for ext, kwargs in {
        "png": {"dpi": 300},
        "svg": {},
    }.items():
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


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
    
    # ===== 左图：柱状图 =====
    # 使用更柔和、专业的渐变色系
    bars1 = axes[0].bar(x - w, df["top1_match"], w, 
                        color=PALETTE["blue"], 
                        edgecolor='white', 
                        linewidth=0.5,
                        label="Top1", 
                        zorder=2)
    
    bars2 = axes[0].bar(x, df["top2_hit"], w, 
                        color=PALETTE["teal"], 
                        edgecolor='white', 
                        linewidth=0.5,
                        label="Top2", 
                        zorder=2)
    
    bars3 = axes[0].bar(x + w, df["soft_match"], w, 
                        color=PALETTE["orange"], 
                        edgecolor='white', 
                        linewidth=0.5,
                        label="Soft Match", 
                        zorder=2)
    
    # 添加数值标签（可选）
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        f'{height:.3f}',
                        ha='center', va='bottom', 
                        fontsize=8, 
                        color=PALETTE["dark_gray"])
    
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=0, ha="center")
    axes[0].set_ylim(0.48, 0.84)
    axes[0].set_ylabel("指标值", fontsize=11, fontweight='bold')
    axes[0].set_title("(a) 融合纠偏提升排序质量", y=-0.28, fontsize=12, fontweight='bold')
    axes[0].grid(axis="y", color=PALETTE["light_gray"], linewidth=0.7, alpha=0.5, zorder=0)
    
    # ===== 图例调小 =====
    axes[0].legend(ncol=3, loc="upper left", 
                   handlelength=0.6,      # 缩短图例线条长度
                   handletextpad=0.2,     # 缩小图标与文字间距
                   columnspacing=0.5,     # 缩小列间距
                   fontsize=8,            # 缩小字体
                   framealpha=0.9, 
                   edgecolor='gray', 
                   fancybox=True)
    
    # ===== 右图：折线图 =====
    # 使用渐变红色系，突出"惩罚"的警示效果
    line = axes[1].plot(x, df["strategy3_over_prediction"], 
                        marker='o', 
                        color=PALETTE["red"], 
                        linewidth=2.5, 
                        markersize=8,
                        markerfacecolor='white',
                        markeredgewidth=2,
                        markeredgecolor=PALETTE["red"],
                        zorder=2,
                        label='过推荐次数')
    
    # 添加数值标签
    for i, v in enumerate(df["strategy3_over_prediction"]):
        axes[1].text(i+0.15, v + 0.15, str(v), 
                     ha='center', va='bottom', 
                     fontsize=9, 
                     color=PALETTE["red"],
                     fontweight='bold')
    
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=15, ha="right")
    axes[1].set_ylabel("策略3过推荐次数", fontsize=11, fontweight='bold')
    axes[1].set_title("(b) 纠偏强度与策略偏置", y=-0.28, fontsize=12, fontweight='bold')
    axes[1].grid(axis="y", color=PALETTE["light_gray"], linewidth=0.7, alpha=0.5, zorder=0)
    
    # 添加阴影区域（突出变化趋势）
    y_min = df["strategy3_over_prediction"].min()
    y_max = df["strategy3_over_prediction"].max()
    axes[1].fill_between(x, y_min, y_max, 
                         color=PALETTE["red"], 
                         alpha=0.08, 
                         zorder=0)
    
    # ===== 整体美化 =====
    fig.suptitle("Pairwise纠偏与策略3校准消融实验", 
                 fontsize=14, 
                 weight="bold", 
                 y=0.98,
                 color=PALETTE["dark_gray"])
    
    # 调整子图间距
    plt.tight_layout()
    
    save(fig, "fig5_pairwise_correction_v2")


if __name__ == "__main__":
    fig5_pairwise_correction()