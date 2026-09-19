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
OUT = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def setup_chinese_font() -> None:
    """配置中文字体为宋体"""
    # 设置全局字体为宋体
    plt.rcParams['font.sans-serif'] = ['SimSun', 'Songti SC', 'STSong', 'SimSong']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    plt.rcParams['font.size'] = 10
    
    # 设置matplotlib字体参数
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['SimSun', 'Songti SC', 'STSong', 'SimSong']


def save(fig: plt.Figure, name: str) -> None:
    for ext, kwargs in {
        "png": {"dpi": 300},
        "svg": {},
    }.items():
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", **kwargs)
    plt.close(fig)


def fig7_pareto_schematic() -> None:
    """双指标Pareto选择逻辑示意（美化版 - 宋体）"""
    # 设置中文字体
    setup_chinese_font()
    
    strategies = np.array(["策略1", "策略2", "策略3", "策略4"])
    x = np.array([0.62, 0.82, 0.72, 0.55])
    y = np.array([0.78, 0.69, 0.88, 0.58])
    pareto = np.array([True, True, True, False])
    
    # 使用更现代的配色方案
    colors = {
        "pareto": "#2E86AB",      # 深邃蓝
        "dominated": "#D3D9DF",   # 柔和灰
        "selected": "#E69A3C",    # 暖橙色
        "highlight": "#A23B72",   # 紫红色（用于标注）
        "grid": "#E8ECEF",
        "text": "#2C3E50",
        "arrow": "#6C7A89"
    }
    
    fig, ax = plt.subplots(figsize=(7, 5.8), dpi=150)
    
    # ========== 背景与网格（更柔和） ==========
    ax.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("white")
    ax.grid(color=colors["grid"], linestyle="--", linewidth=0.8, alpha=0.6, zorder=0)
    
    # ========== 绘制数据点 ==========
    # 被支配策略（灰色）
    ax.scatter(x[~pareto], y[~pareto], s=160, 
               color=colors["dominated"], 
               edgecolor="#A0AAB5", linewidth=1.5,
               label="被支配策略", zorder=2, alpha=0.85)
    
    # Pareto候选（蓝色）- 增大尺寸
    ax.scatter(x[pareto], y[pareto], s=200, 
               color=colors["pareto"], 
               edgecolor="white", linewidth=2.5,
               label="Pareto候选", zorder=3, 
               alpha=0.95)
    
    # 在Pareto点周围添加淡色光晕（使用更大的点模拟阴影效果）
    for xi, yi in zip(x[pareto], y[pareto]):
        ax.scatter(xi, yi, s=450, color=colors["pareto"], 
                   alpha=0.12, zorder=1)
    
    # 为Pareto点添加轻微的阴影效果（偏移的灰色点）
    for xi, yi in zip(x[pareto], y[pareto]):
        ax.scatter(xi - 0.003, yi - 0.003, s=200, 
                   color="white", alpha=0.3, zorder=2)
    
    # 选中点（橙色）- 突出显示
    ax.scatter([x[2]], [y[2]], s=280, 
               color=colors["selected"], 
               edgecolor="#8B5E2B", linewidth=2.0,
               label="拦截率优先选择", zorder=4,
               alpha=1.0)
    # 选中点外发光
    ax.scatter([x[2]], [y[2]], s=500, color=colors["selected"], 
               alpha=0.15, zorder=1)
    # 选中点内层高光
    ax.scatter([x[2]], [y[2]], s=120, color="white", 
               alpha=0.3, zorder=5)
    
    # ========== 添加标签（带背景框） ==========
    for i, (sx, sy, lab) in enumerate(zip(x, y, strategies)):
        # 判断标签偏移方向避免重叠
        if i == 0:  # 策略1 - 右上
            offset_x, offset_y = 0.015, 0.015
        elif i == 1:  # 策略2 - 右下
            offset_x, offset_y = 0.025, -0.025
        elif i == 2:  # 策略3 - 右上
            offset_x, offset_y = 0.015, 0.015
        else:  # 策略4 - 左下
            offset_x, offset_y = -0.035, -0.015
        
        # 标签背景框 - 宋体显示
        bbox_props = dict(boxstyle="round,pad=0.3", 
                         facecolor="white", 
                         edgecolor=colors["text"], 
                         alpha=0.85, 
                         linewidth=0.8)
        ax.text(sx + offset_x, sy + offset_y, lab, 
                fontsize=10, weight="bold", color=colors["text"],
                fontfamily='SimSun',  # 指定宋体
                bbox=bbox_props, zorder=5)
    
    # ========== Pareto前沿连线 ==========
    # 按x排序后连接Pareto点
    pareto_idx = np.where(pareto)[0]
    pareto_order = pareto_idx[np.argsort(x[pareto_idx])]
    ax.plot(x[pareto_order], y[pareto_order], 
            color=colors["pareto"], linestyle="--", 
            linewidth=2.0, alpha=0.6, zorder=1,
            label="Pareto前沿")
    
    # 前沿下方填充淡色区域（显示支配区域）
    # 获取排序后的x和y用于填充
    sorted_x = x[pareto_order]
    sorted_y = y[pareto_order]
    # 添加底部边界
    fill_x = np.concatenate([[0.35], sorted_x, [0.92]])
    fill_y = np.concatenate([[0.48], sorted_y, [0.48]])
    ax.fill(fill_x, fill_y, color=colors["pareto"], alpha=0.06, zorder=0)
    
    # ========== 箭头标注 ==========
    # 策略4被支配的标注
    ax.annotate("被策略1完全支配\n(两项指标均更低)", 
                xy=(0.55, 0.58), 
                xytext=(0.4, 0.5), 
                arrowprops=dict(arrowstyle="->", 
                               color=colors["arrow"], 
                               lw=1.8, 
                               connectionstyle="arc3,rad=-0.15"),
                fontsize=8.5, 
                color=colors["arrow"],
                weight="medium",
                fontfamily='SimSun',  # 指定宋体
                bbox=dict(boxstyle="round,pad=0.3", 
                         facecolor="white", 
                         edgecolor=colors["arrow"], 
                         alpha=0.85))
    
    # 添加"最佳"标注到选中点
    ax.annotate("★ 最终优选", 
                xy=(0.72, 0.88), 
                xytext=(0.78, 0.93), 
                fontsize=9.5, 
                color=colors["selected"],
                weight="bold",
                fontfamily='SimSun',  # 指定宋体
                arrowprops=dict(arrowstyle="->", 
                               color=colors["selected"], 
                               lw=1.5,
                               connectionstyle="arc3,rad=0.1"))
    
    # ========== 坐标轴与标签 ==========
    ax.set_xlim(0.35, 0.92)
    ax.set_ylim(0.48, 0.96)
    
    ax.set_xlabel("平均效费比 M2", 
                  fontsize=11.5, weight="bold", color=colors["text"], 
                  fontfamily='SimSun', labelpad=8)  # 指定宋体
    ax.set_ylabel("平均拦截率 M1", 
                  fontsize=11.5, weight="bold", color=colors["text"], 
                  fontfamily='SimSun', labelpad=8)  # 指定宋体
    
    # 自定义刻度
    ax.set_xticks(np.arange(0.4, 0.91, 0.1))
    ax.set_yticks(np.arange(0.5, 0.96, 0.1))
    ax.tick_params(labelsize=9, colors=colors["text"])
    
    # 设置坐标轴边框颜色
    for spine in ax.spines.values():
        spine.set_color(colors["text"])
        spine.set_alpha(0.3)
    
    # 标题 - 宋体
    ax.set_title("双指标 Pareto 最优选择示意", 
                 fontsize=14, weight="bold", color=colors["text"], 
                 fontfamily='SimSun', pad=18)  # 指定宋体
    
    # ========== 图例 - 右下角，无立体样式，紧凑排列 ==========
    legend = ax.legend(loc="upper left",  # 右下角
                       fontsize=6, 
                       framealpha=0.8, 
                       edgecolor=colors["grid"],
                       shadow=False,  # 去掉阴影
                       fancybox=False,  # 去掉圆角
                       borderpad=0.5,  # 减小边框内边距
                       handletextpad=0.5,  # 减小图标和文字间距
                       labelspacing=0.8,  # 减小条目间距
                       prop={'family': 'SimSun'})
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_linewidth(0.7)
    
    # ========== 右上角方向指示 ==========
    # 更优方向箭头
    ax.annotate("", xy=(0.88, 0.94), xytext=(0.88, 0.89),
                arrowprops=dict(arrowstyle="->", color=colors["text"], 
                               alpha=0.4, lw=1.5))
    ax.text(0.89, 0.915, "更优", 
            fontsize=7, color=colors["text"], alpha=0.5, 
            rotation=90, va="center", fontfamily='SimSun')  # 指定宋体
    
    ax.annotate("", xy=(0.90, 0.52), xytext=(0.85, 0.52),
                arrowprops=dict(arrowstyle="->", color=colors["text"], 
                               alpha=0.4, lw=1.5))
    ax.text(0.875, 0.525, "更优", 
            fontsize=7, color=colors["text"], alpha=0.5, 
            ha="center", va="bottom", fontfamily='SimSun')  # 指定宋体
    
    # ========== 保存 ==========
    plt.tight_layout()
    save(fig, "fig7_pareto_schematic")
    plt.close(fig)


if __name__ == "__main__":
    fig7_pareto_schematic()