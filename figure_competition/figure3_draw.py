import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path


# 设置中文字体为宋体，英文为Times New Roman
rcParams['font.family'] = ['SimSun', 'Times New Roman', 'serif']
rcParams['axes.unicode_minus'] = False
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
    
    # 更精致的配色
    colors = {
        "blue": "#4A7CF7",
        "teal": "#20B2AA",
        "orange": "#FF9F45",
        "purple": "#A78BFA",  # 中等饱和度紫色
        "purple_light": "#C4B5FD",  # 稍浅紫色（备用）
        "light_gray": "#E8ECF1",
        "dark_text": "#2D3748",
        "highlight": "#FC8181",
        "highlight_light": "#FCA5A5",  # 稍浅红色
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), 
                             gridspec_kw={"width_ratios": [2.1, 1]})
    
    # ===== 左图：排序类指标 =====
    x = np.arange(len(data))
    width = 0.26
    
    # 添加柱状图
    bars1 = axes[0].bar(x - width, data["top1"], width, 
                        label="Top-1", color=colors["blue"], 
                        edgecolor='white', linewidth=0.5,
                        alpha=0.92, zorder=2)
    bars2 = axes[0].bar(x, data["top2"], width, 
                        label="Top-2", color=colors["teal"],
                        edgecolor='white', linewidth=0.5,
                        alpha=0.92, zorder=2)
    bars3 = axes[0].bar(x + width, data["soft"], width, 
                        label="Soft Match", color=colors["orange"],
                        edgecolor='white', linewidth=0.5,
                        alpha=0.92, zorder=2)
    
    # 突出显示最终模型
    highlight_idx = len(data) - 1
    for bars in [bars1, bars2, bars3]:
        bars[highlight_idx].set_edgecolor(colors["highlight"])
        bars[highlight_idx].set_linewidth(2)
        bars[highlight_idx].set_alpha(1.0)
    
    # 坐标轴设置
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(data["方法"], rotation=30, ha="right", 
                           fontsize=9.5, color=colors["dark_text"],
                           weight="bold")
    axes[0].set_ylim(0, 0.9)
    axes[0].set_ylabel("指标值", fontsize=11, weight="bold", color=colors["dark_text"])
    
    # 美化网格
    axes[0].grid(axis="y", color=colors["light_gray"], linewidth=0.8, 
                 linestyle='-', alpha=0.7, zorder=0)
    axes[0].set_axisbelow(True)
    
    # 边框美化
    for spine in axes[0].spines.values():
        spine.set_color(colors["light_gray"])
        spine.set_linewidth(1.2)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # 图例放在左上角，英文使用 Times New Roman 并加粗
    legend = axes[0].legend(ncol=3, loc="upper left", frameon=True,
                           fancybox=False, shadow=False, 
                           framealpha=0.9, edgecolor=colors["light_gray"],
                           fontsize=9)
    legend.get_frame().set_linewidth(0.8)
    # 设置图例文字为 Times New Roman 并加粗
    for text in legend.get_texts():
        text.set_fontfamily('Times New Roman')
        text.set_weight('bold')
    
    # 在最终模型柱子上添加星标
    axes[0].text(x[highlight_idx] + width*2.2, data["top1"].iloc[highlight_idx] + 0.05,
                "★", color=colors["highlight"], fontsize=14, ha='center', va='top')
    
    # 设置左图刻度标签字体（中文用宋体）并加粗
    for label in axes[0].get_xticklabels():
        label.set_fontfamily('SimSun')
        label.set_weight('bold')
    
    # ===== 右图：收益损失 =====
    # 按数值排序以便更清晰展示
    sorted_data = data.sort_values("regret", ascending=True)
    sorted_data = sorted_data.dropna(subset=["regret"])
    
    # 使用更细的柱子
    bar_height = 0.65
    
    # 使用中等饱和度的紫色
    bars_h = axes[1].barh(sorted_data["方法"], sorted_data["regret"], 
                         height=bar_height,
                         color=colors["purple"], alpha=0.8,
                         edgecolor='white', linewidth=0.5, zorder=2)
    
    # 为最终模型高亮 - 使用稍浅红色
    final_regret_idx = sorted_data.index.get_loc(highlight_idx) if highlight_idx in sorted_data.index else -1
    if final_regret_idx >= 0:
        bars_h[final_regret_idx].set_color(colors["highlight_light"])
        bars_h[final_regret_idx].set_alpha(0.9)
        bars_h[final_regret_idx].set_edgecolor('#FC8181')
        bars_h[final_regret_idx].set_linewidth(1.5)
    
    # 在条形末端添加数值标签 - 英文用 Times New Roman 并加粗
    for i, (idx, row) in enumerate(sorted_data.iterrows()):
        axes[1].text(row["regret"] + 0.001, i, 
                    f"{row['regret']:.4f}",
                    va='center', fontsize=8.5, 
                    color=colors["dark_text"], weight="bold",
                    fontfamily='Times New Roman')
    
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Mean Regret", fontsize=11, 
                       weight="bold", color=colors["dark_text"],
                       fontfamily='Times New Roman')
    
    # 美化网格
    axes[1].grid(axis="x", color=colors["light_gray"], linewidth=0.8,
                 linestyle='-', alpha=0.7, zorder=0)
    axes[1].set_axisbelow(True)
    
    # 边框美化
    for spine in axes[1].spines.values():
        spine.set_color(colors["light_gray"])
        spine.set_linewidth(1.2)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['left'].set_visible(False)
    axes[1].tick_params(axis='y', left=False)
    
    # 设置 y 轴标签字体（中文用宋体）并加粗
    for label in axes[1].get_yticklabels():
        label.set_fontfamily('SimSun')
        label.set_weight('bold')
    
    # ===== 在子图正下方添加 (a)(b) 标签和标题 =====
    # 左图下方 - 加粗
    axes[0].text(0.5, -0.22, "(a) 排序类指标对比", 
                transform=axes[0].transAxes,
                fontsize=11, weight="bold", 
                ha='center', va='top')
    
    # 右图下方 - 加粗
    axes[1].text(0.5, -0.22, "(b) 收益损失", 
                transform=axes[1].transAxes,
                fontsize=11, weight="bold", 
                ha='center', va='top')
    
    # ===== 整体标题 =====
    fig.suptitle("基线方法与最终模型效果对比", fontsize=14, weight="bold", 
                 y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.15, wspace=0.25)
    
    # 保存
    save(fig, "fig3_baseline_metrics_v2")
    plt.show()


if __name__ == "__main__":
    fig3_baseline_comparison()