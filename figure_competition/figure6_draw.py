from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# 设置中文字体为宋体，英文为Times New Roman
plt.rcParams['font.sans-serif'] = ['SimSun', 'Songti SC', 'STSong', 'Microsoft YaHei']
plt.rcParams['font.family'] = ['SimSun', 'Times New Roman', 'serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

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


def fig6_strategy_distribution() -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import json
    from pathlib import Path
    
    summary = json.loads((ROOT / "outputs" / "summary.json").read_text(encoding="utf-8"))
    cv = summary["cv_metrics"]
    pred = {int(k): int(v) for k, v in cv["pred_strategy_counts"].items()}
    best = {int(k): int(v) for k, v in cv["best_strategy_counts"].items()}
    test = {int(k): int(v) for k, v in summary["test_prediction_counts"].items()}
    strategies = [1, 2, 3, 4]
    
    # 与前面柱状图保持一致的配色方案
    colors = {
        "blue": "#4A7CF7",      # 亮蓝 - 对应 Top-1
        "teal": "#20B2AA",      # 青色 - 对应 Top-2
        "orange": "#FF9F45",    # 橙色 - 对应 Soft Match
        "purple": "#A78BFA",    # 紫色 - 对应 regret
        "light_gray": "#E8ECF1",
        "dark_text": "#2D3748",
    }
    
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    x = np.arange(len(strategies))
    w = 0.25
    gap = 0.03
    
    # 使用与前面一致的配色：蓝色、青色、橙色
    bars_best = ax.bar(x - w - gap/2, [best.get(s, 0) for s in strategies], w, 
                       label="验证集真实最优", color=colors["blue"], 
                       edgecolor='white', linewidth=0.8, alpha=0.88, zorder=2)
    bars_pred = ax.bar(x, [pred.get(s, 0) for s in strategies], w, 
                       label="验证集模型推荐", color=colors["teal"],
                       edgecolor='white', linewidth=0.8, alpha=0.88, zorder=2)
    bars_test = ax.bar(x + w + gap/2, [test.get(s, 0) for s in strategies], w, 
                       label="测试集推荐", color=colors["orange"],
                       edgecolor='white', linewidth=0.8, alpha=0.88, zorder=2)
    
    # 在柱子上显示数值
    def add_value_labels(bars, offset=0):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2, height + max(3, height*0.015),
                       f"{int(height):,}", ha='center', va='bottom', 
                       fontsize=9, weight='bold', color=colors["dark_text"],
                       fontfamily='Times New Roman')
    
    add_value_labels(bars_best)
    add_value_labels(bars_pred)
    add_value_labels(bars_test)
    
    ax.set_xticks(x)
    ax.set_xticklabels([f"策略 {s}" for s in strategies], fontsize=11, weight='bold', 
                       color=colors["dark_text"])
    ax.set_ylabel("样本数量", fontsize=11, weight='bold', color=colors["dark_text"], labelpad=10)
    ax.set_xlabel("策略类型", fontsize=11, weight='bold', color=colors["dark_text"], labelpad=10)
    
    # 标题
    ax.set_title("策略推荐分布与偏置诊断", fontsize=13, weight="bold", pad=16, color=colors["dark_text"])
    
    # 美化网格
    ax.grid(axis="y", color=colors["light_gray"], linewidth=0.8, linestyle='-', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    
    # 美化边框
    for spine in ax.spines.values():
        spine.set_color(colors["light_gray"])
        spine.set_linewidth(1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 图例放在图内右上角
    legend = ax.legend(loc="upper right", frameon=True,
                      fontsize=9, framealpha=0.92, 
                      edgecolor=colors["light_gray"],
                      fancybox=False, shadow=False)
    legend.get_frame().set_linewidth(0.8)
    for text in legend.get_texts():
        text.set_fontfamily('SimSun')
        text.set_weight('bold')
    
    # 设置y轴范围，留出空间显示数值标签
    max_val = max(max(best.values(), default=0), 
                  max(pred.values(), default=0), 
                  max(test.values(), default=0))
    ax.set_ylim(0, max_val * 1.12 if max_val > 0 else 100)
    
    # 优化刻度
    ax.tick_params(axis='y', labelsize=10, colors=colors["dark_text"])
    ax.tick_params(axis='x', labelsize=10, colors=colors["dark_text"])
    
    # 设置y轴刻度标签字体
    for label in ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')
        label.set_weight('bold')
    
    plt.tight_layout()
    save(fig, "fig6_strategy_distribution_v2")


if __name__ == "__main__":
    fig6_strategy_distribution()