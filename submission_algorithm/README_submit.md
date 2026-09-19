# 反无人机策略推荐算法代码提交说明

## 1. 提交代码范围

本目录为最终算法的干净提交版本，仅保留训练和推理所需代码，不包含阶段性实验脚本、论文绘图脚本和中间输出文件。

最终算法为“拦截率优先的核心25工程特征 + 随机森林收益预测 + Pairwise策略纠偏”。

## 2. 文件说明

| 文件 | 作用 |
| --- | --- |
| `run_final_algorithm.py` | 最终训练与测试推理入口，读取训练/测试 Excel，输出测试集推荐策略 |
| `src/data_utils.py` | 读取比赛 Excel 数据，提取原始42维输入、策略编号和两个指标 |
| `src/feature_engineering.py` | 构造59个候选工程特征 |
| `src/feature_selection.py` | 保留最终使用的25个核心工程特征 |
| `src/stage2_eval.py` | 指标归一化、拦截率优先收益函数等工具 |
| `src/ranking_models.py` | 候选策略收益回归模型，包含最终使用的随机森林收益预测器 |
| `src/pairwise_models.py` | Pairwise Logistic策略对偏好学习模型 |
| `src/pairwise_correction.py` | 随机森林收益分与Pairwise相对偏好分的融合纠偏模型 |
| `src/stage3_models.py` | Pairwise模型使用的归一化工具 |
| `src/models.py` | 基础距离函数等公共工具 |
| `requirements.txt` | Python依赖 |

## 3. 运行方式

将比赛提供的训练集和测试集 Excel 文件放入 `Data` 目录，然后在本目录上一级或本目录内运行：

```bash
python submission_algorithm/run_final_algorithm.py --data-dir Data --output-dir outputs/final_submission
```

若在 `submission_algorithm` 目录内运行：

```bash
python run_final_algorithm.py --data-dir ../Data --output-dir ../outputs/final_submission
```

## 4. 输出文件

运行后生成：

| 输出文件 | 内容 |
| --- | --- |
| `test_recommendations.csv` | 每个测试样本的推荐策略以及四个候选策略融合得分 |
| `summary.json` | 模型参数、特征数量、推荐策略分布等摘要信息 |

## 5. 不建议提交的内容

以下内容属于研发和论文辅助材料，正式算法代码提交时可不包含：

- `experiments/stage*.py`：阶段性实验脚本
- `outputs/stage*/`：阶段性实验结果
- `scripts/markdown_to_docx_report.py`、`scripts/make_report_figures.py` 等论文处理脚本
- `docs/` 下的论文、图片和 Word 备份文件
- `src/__pycache__/` 和所有 `.pyc` 文件
