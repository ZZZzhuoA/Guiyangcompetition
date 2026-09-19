# 决赛实时策略推荐 — 提交与操作手册

赛方每隔 Δ（15 或 20 秒）调用一次 `recommend()`，输入当前态势三张表，输出拦截策略 `LJCL ∈ {1,2,3}`。

- `1` 闭合时间最短
- `2` 拦截性能最佳
- `3` 综合效益最优

**训练在赛方电脑上用真实 CSV 做。** 依赖只有 `numpy`，不需要 sklearn / torch / pandas。

---

## 1. 拷到赛方电脑：这一个目录

本地先打包装好：

```powershell
python scripts/make_submission_package.py --output outputs/submission_package
```

然后**只拷 `outputs/submission_package/` 这一个文件夹**。不要拷整个仓库。

```text
submission_package/          ← 拷这一整个文件夹
  scripts/                   建集、训练、对比、判读
  src/finals/                全部决赛代码（训练+推理）
  submission_finals/         recommend.py 入口
  outputs/finals_model/      托底用的 policy.pkl（本地合成数据训的）
```

不要拷：`experiments/`、初赛 `src/*.py`、`Data/`、`docs/`、`决赛/`（赛方电脑上已经有 CSV 样本，别用仓库里的 xlsx 模板覆盖）。

赛方电脑上工作目录就是 `submission_package/` 这一层。`pip install numpy`。

### 到了赛方电脑以后

`--input` 换成赛方样本根目录（`kemu6_data_72mb`）。`600-end` 样本路径和 `1-200` 同构，有的就放在 `1-200/` 下面：

`kemu6_data_72mb/1-200/反无大赛-6方向-终版_6方向-1km数据_72km-1km-6方向-600-end_001/_1`

官方拦截率 CSV 默认在 `--input` 及其上一级自动找 `文件实验数据路径与结果_72mb.csv`。找不到再给 `--results`。正式间隔若是 20 秒，下面所有 `--delta 15` 改成 `20`。

空行 / 空单元格会被跳过，不会再因 `float(None)` 崩掉。建集 stderr 有进度条。`outputs/finals_slice/_ckpt/` 每做完一个场景就落盘；中断后**再跑同一条命令**会跳过已完成场景。成功后 `_ckpt` 仍保留，便于重新训练和复核标签；只有要推倒重来才加 `--fresh`。检查点同时记录 `delta`、idle 抽样比例等构造参数，参数改变时会要求 `--fresh`。

可以直接使用 `run_on_site.py` 完成真实切片建集、候选对比、判读和模型导出；它默认只跑真实观测切片。需要比较带历史窗口的贯序候选时，再显式加 `--with-seq`。

```powershell
cd submission_package
pip install numpy

# 0. 核对样本（快）。n_runs 等于叶子目录数，缺号正常
python scripts/inspect_finals_samples.py --input kemu6_data_72mb

# 1. 真实观测切片集：每个真实实验按 delta 秒形成一个区间转移
python scripts/build_slice_set.py --input kemu6_data_72mb --output outputs/finals_slice --delta 15 --workers 4 --idle-keep-ratio 0.20
# 中断后原样再跑上一行。推倒重来才加 --fresh

# 2. 对比真实切片的 10 个候选。写成各候选 pkl，不覆盖 policy.pkl
python scripts/compare_finals_models.py --dataset outputs/finals_slice --output outputs/finals_model

# 3. 判读数据、价值、后悔值和 BLOCK
python scripts/diagnose_finals_models.py --compare outputs/finals_model/compare.json --dataset outputs/finals_slice

# 4. 按最终决定导出提交模型；尚未确定时先沿用 rf_pair
python scripts/train_finals_model.py --dataset outputs/finals_slice --output outputs/finals_model --model rf_pair

# 可选 A：一键构造并比较带历史窗口的真实贯序候选
python scripts/run_on_site.py --input kemu6_data_72mb --with-seq

# 可选 B：旧版贯序集没有跑完或存在旧版 _ckpt 时，单独用新逻辑从头重建
python scripts/build_seq_set.py `
  --input kemu6_data_72mb `
  --output outputs/finals_seq `
  --delta 15 `
  --workers 4 `
  --idle-keep-ratio 0.20 `
  --fresh
```

上面单独重建命令中的 `--fresh` 只用于清除旧版贯序检查点并从头生成。新版本开始构造后如果中断，续跑时执行同一命令但去掉 `--fresh`，程序会跳过已经完成的场景。

`--with-seq` 只负责构造、训练和比较贯序候选，默认提交模型仍是单切片的 `rf_pair`。如果贯序判读结果通过 BLOCK 检查并决定提交，例如选择 `seq_pair`，再单独导出它的 `policy.pkl`：

```powershell
python scripts/train_finals_model.py --dataset outputs/finals_seq --output outputs/finals_model_ts --model seq_pair
```

当前包的默认模型位置是 `outputs/finals_model/policy.pkl`；确认切换后可执行 `Copy-Item outputs/finals_model_ts/policy.pkl outputs/finals_model/policy.pkl -Force`，再用 `scripts/check_seq_inference.py` 检查历史缓存和延迟。

找不到结果表时第 0/1/2 步加 `--results 文件实验数据路径与结果_72mb.csv`。赛方电脑不要加 `--with-replay`（那是本地玩具世界）。

测试集不是另存一份 CSV：同一局里的切片不会同时进训练和测试，划分写在 `outputs/finals_slice/split.json`。

有 BLOCK 时（数据层、指标没有分辨力、或 **slice chosen** 本身不合格）保持 `policy.pkl=rf_pair`。某个落选候选不合格不影响换 chosen。无 BLOCK 且要换模型，再单独训那个名字覆盖 `policy.pkl`。

训练中途失败时，包里那份托底 `policy.pkl` 仍能跑推理。

---

## 2. 实时调用

```python
from submission_finals.recommend import recommend, reset

reset()                              # 每局开始前调一次
for trigger_time in periodic_times:  # 每隔 15/20 秒
    ljcl = recommend(payload, time=trigger_time)
```

**两条硬约束，违反了模型会静默变差而不是报错：**

1. **必须在同一个进程里反复调用。** 64 维特征里有 14 维是跨触发的（新增/消失目标、距离与闭合时间变化、上次 LJCL、时间进度），序列模型的历史窗口也靠进程内缓存。每次触发新起进程，这些维度全是 0，等于把模型降级成单帧静态打分。`recommend()` 内部把 policy 缓存成模块级单例就是为了这个，不要绕过它去自己 `FinalsPolicy.load()`。
2. **`time` 要传真实仿真时刻（赛方 `FZTime`）。** 不传的话 `time_frac`（打到第几段了）恒为 0。

一局结束换下一局之前调 `reset()`，否则上一局的历史串进新局。

排查现场问题用 `recommend_verbose()`，它连三个策略的打分一起返回：

```python
from submission_finals.recommend import recommend_verbose
print(recommend_verbose(payload, time=30.0))
# {"LJCL": 2, "scores": [0.0, 0.35, 0.175], "model": "rf_pair"}
```

**三个分数几乎相同就说明模型没有区分能力**，此时输出等价于固定策略，不如直接锁一个。

模型文件缺失时会自动退化成规则托底（紧迫→1，高概率边→2，否则→3）而不是抛异常——赛场上宁可给个合法策略也不能崩。这时 `model` 字段是 `rule_fallback`，看到它就说明 pkl 没拷进去。

### payload 结构

```python
{
  "targets": [                     # 集成接口：目标相关数据列表
    {
      "TargetUnitID": 101,
      "TargetUnitType": "BlueTarget",
      "B_l_jbz": False,            # 是否正被拦截
      "Ui_Zwx": 1,                 # 威胁等级，越小越危险
      "D_X": 9000.0,  "D_Y": 1200.0, "D_Z": 400.0,
      "D_XV": -220.0, "D_YV": -30.0, "D_ZV": -5.0,
      "Klj_list": [                # 同一目标同时 4 条：实体类型 1–4 各一行
        {"UI_FSChandle": 11, "Str_HLMC": "Type1", "UC_HlType": 1,
         "B_DDKsslj": True, "B_GPKsslj": False, "B_HPMKsslj": False, "B_GNJGKsslj": False,
         "D_LJGL": 0.7},
        {"UI_FSChandle": 12, "Str_HLMC": "Type2", "UC_HlType": 2,
         "B_DDKsslj": False, "B_GPKsslj": True, "B_HPMKsslj": False, "B_GNJGKsslj": False,
         "D_LJGL": 0.8},
        {"UI_FSChandle": 13, "Str_HLMC": "Type3", "UC_HlType": 3,
         "B_DDKsslj": False, "B_GPKsslj": False, "B_HPMKsslj": True, "B_GNJGKsslj": False,
         "D_LJGL": 0.9},
        {"UI_FSChandle": 14, "Str_HLMC": "Type4", "UC_HlType": 4,
         "B_DDKsslj": False, "B_GPKsslj": False, "B_HPMKsslj": False, "B_GNJGKsslj": True,
         "D_LJGL": 1.0}            # 概率不限于 0.7/0.9；B_GNJGKsslj 只表示实体4能否拦
      ]
    }
  ],
  "units": [                       # 红方相关数据列表
    {"UnitID": 11, "UnitType": "Type1",
     "UnitPos_x": 0.0, "UnitPos_y": 0.0, "UnitPos_z": 0.0}
  ]
}
```

中文键名 `目标相关数据列表` / `装备相关数据列表` 也认。

样本 CSV 的 `FZTime`/`Time` 从 **1s 起、每 0.5s 一记**（系统内部 0.1s 更新，记录抽稀成 0.5s）。实装可能按 **0.1s 给当前帧、不定间隔** 调 `recommend()`：用当前帧打分即可，不必自己对齐 0.5s，也不假设两次调用正好隔 15s。短于约 0.4s 的连刷只更新当前态势、不写入决策历史，避免 0.1s 空转把时序窗口填满。

赛方 CSV / 接口里**多出来的列一律丢掉**，只抽本地 schema 那些字段（`src/finals/schema.py` 里的 `RHDL_COLUMNS` / `LJ_COLUMNS` / `HEALTH_COLUMNS` / `SJZS_COLUMNS`，以及接口的 `INTERFACE_*`）。缺列填空，不因为多列报错。`G_LJGL` 没有时认 `D_LJGL`。

`Stu_ZZGLLJ` / `Klj_list` 对**同一个目标同时 4 行**，`UC_HlType`∈{1,2,3,4}，对应 `B_DDKsslj` / `B_GPKsslj` / `B_HPMKsslj` / `B_GNJGKsslj`。布尔是「系统算出该类实体此刻能否拦」，`G_LJGL`/`D_LJGL` 是拦截概率，取值不限于 0.7 和 0.9（还有 0.8、1 等）。实体4 的可实施拦截用 `B_GNJGKsslj`。

接口没给、代码里用默认值的几处（训练与实装会有分布差）：没有 `B_Sgzmb` 所以当成全部已跟踪；没有 `D_Costgy` 所以链接成本固定 0.5；没给保护要点所以闭合时间按原点 `(0,0,0)` 算；没给红方生命值所以当成全存活。

---

## 3. 数据路径

真实样本全是 **CSV**，不是 xlsx。一条叶子路径 = 一次整局仿真。`_1/_2/_3` 是**同一场景、同一策略的三次重复实验**，不是 LJCL=1/2/3：

```text
kemu6_data_72mb/
  1-200/
    反无大赛-6方向-终版_6方向-1km数据_72km-1km-6方向-1-200_001/
      _1/   Stu_ZZGLRHDL_*.csv  Stu_ZZGLLJ_*.csv  HealthState_*.csv  Stu_SJZS_*.csv
      _2/
      _3/
    ...
    反无大赛-...-1-200_200/_1  _2  _3
  200-400/
    反无大赛-...-200-400_001/_1  _2  _3
    ...
    反无大赛-...-200-400_200/_1  _2  _3
  400-600/
    反无大赛-...-400-600_001/_1  _2  _3
    ...
    反无大赛-...-400-600_200/_1  _2  _3
  600-end/  （或直接放在 1-200/ 下，靠文件夹名里的 600-end_NNN 识别）
    反无大赛-...-600-end_001/_1  _2  _3
    ...
    反无大赛-...-600-end_264/_3
  文件实验数据路径与结果_72mb.csv    官方拦截率 + 「策略」类型
```

`--input` 指到 **`kemu6_data_72mb` 这一层**，建集会递归收齐全部批次（含 `600-end`）。不要只指到 `1-200`。

`scene_id` 从**场景文件夹名**里的 `批次_序号` 取，不按枚举、**不要求连续**。缺 `1-200` 的 179 或 `400-600` 的 179/_1、180/_2、180/_3 就跳过那些叶子，后面的序号不会前移：

- `...-1-200_001` → 1；`...-1-200_180` 仍是 180（不会因为缺 179 变成 179）
- `...-200-400_001` → 201
- `...-400-600_180` → 580
- `...-600-end_001` → 601；`...-600-end_264` → 864（父目录即使是 `1-200` 也不和 `_001` 撞号）

`inspect_finals_samples.py` 的 `index_gaps` 会列出每批 min..max 之间缺的序号、以及 `_1/_2/_3` 不齐的场景。CSV 先探测 BOM/UTF-8/GB18030 再按该 encoding 打开。

**策略号优先读结果表的「策略」列**，中文类型映射为：

- 闭合时间最短 → 1
- 拦截性能最佳 → 2
- 综合效益最优 → 3

结果表「实验名称」是场景目录，「实验次数」是 `_1/_2/_3`。对不上再退回 `Stu_SJZS.S_LJCL`（合成数据才用文件夹名 `sample_000_s1`）。三次重复会作为同一场景下的独立真实轨迹读取。真实切片不再从某一条轨迹模拟其它策略；缺少策略共同支持的场景只能作为原始观测保存，不能提供完整的三策略匹配组。

赛方 CSV 可能是 GBK 而不是 UTF-8；有的表第一行是表头、第二三行是单位/说明、第四行才是数据。读表时会自动试编码并跳过表头下的说明行。

每条真实轨迹都按决策时刻保存 `(x_t, 实际策略, 区间收益, x_next)`。训练和测试先按场景划分，再在各自划分内用可观测态势匹配相似时刻，估计三种策略的条件收益；测试标签不参与训练匹配。

| 路径 | 内容 |
| --- | --- |
| `kemu6_data_72mb/` | 真实样本根目录。其下四个批次（含 `600-end`），每个场景三次重复 |
| `文件实验数据路径与结果_72mb.csv` | 每个样本的官方拦截率和策略类型。建集会自动找 |
| `决赛/样本1/*.xlsx` | 仓库里的**字段模板**，不是训练数据。建集只认 `*.csv` |
| `决赛/集成接口.xlsx` | 接口字段说明（文档，不是样本） |
| `outputs/finals_synth/` | 本地合成场景 CSV（`sample_000_s1` 布局；赛方没给数值时的替代） |
| `outputs/finals_slice/` | 单切片训练集，`training_samples.{csv,npz}` |
| `outputs/finals_seq/` | 贯序训练集，额外带 `x_next` / `x_hist` / `a_prev` / `hist_len` |
| `outputs/finals_model/` | 默认提交模型 `policy.pkl` + 各候选 pkl + `summary.json` / `compare.json` |
| `outputs/submission_package/` | 打包脚本产出，直接拷这个目录 |

换真实数据前先核对发现了几个目录（第 1 节第 0 步）：

```powershell
python scripts/inspect_finals_samples.py --input kemu6_data_72mb
```

期望 `n_runs` 等于实际叶子目录数，不要按 200×4×3 去对。缺号是正常的。`index_gaps` 里能看到 1-200 缺 179、400-600 缺 179/_1 和 180/_2/_3 这类洞。`n_official_strategy_matched` 应接近 `n_runs`。核对完按第 1 节 1→6 分步跑，不要用 `run_finals_pipeline.py` 一把梭。

`summary.json` 里的 `runs` 列出实际吃进去的每个目录、抽到的 `scene_id`、以及标签来源 `official_csv` / `health`。建集若被空行打断过，看 `n_load_errors` 和 `dropped_points.csv` 里 `skip:...`。

历史窗口只写进 `.npz`（CSV 存不下 `(K, F)`）。只有 CSV 时序列模型会退化成单帧冷启动，不报错。

---

## 4. 训练流水线与参数

赛方电脑的主流程是“真实观测切片 → 场景级划分 → 相似态势匹配 → 候选训练与评估 → 导出 `policy.pkl`”。`run_on_site.py` 默认串完这条流程；`--with-seq` 会额外构造真实贯序集：每条真实 run 保留历史窗口、实际动作、区间收益和下一状态，再由相似历史态势估计其它策略的条件收益。贯序结果单独评估，不混入真实切片的默认结论。

### 本地合成数据（赛方电脑跳过）

```powershell
python scripts/make_finals_synth_data.py --output outputs/finals_synth --n-scenes 6 --seed 7
python scripts/build_slice_set.py --input outputs/finals_synth --output outputs/finals_slice --delta 15 --idle-keep-ratio 1.0
python scripts/build_seq_set.py --input outputs/finals_synth --output outputs/finals_seq --delta 15
python scripts/compare_finals_models.py --dataset outputs/finals_slice --output outputs/finals_model
python scripts/compare_finals_models.py --seq --dataset outputs/finals_seq --output outputs/finals_model_ts
python scripts/train_finals_model.py --dataset outputs/finals_slice --output outputs/finals_model --model rf_pair
python scripts/make_submission_package.py --output outputs/submission_package
```

`train_finals_model.py` 和 `compare_finals_models.py` 都会 `fit`，但出口不同。前者只训 `--model` 指定的那一个，写成 `policy.pkl`——这是 `recommend.py` 和打包脚本默认加载的文件。后者把清单里每个候选都训一遍，写成 `rf_pair.pkl`、`seq_fqi.pkl` 和 `compare.json`，**故意不碰 `policy.pkl`**，避免一次对比把正在用的提交模型盖掉。`compare.json` 里的 `chosen` 只是排序，要真换提交模型，先改 `--model`（或 `DEFAULT_SUBMIT`）再跑第 1 节第 5 步。

辅助：

```powershell
# 列出全部 21 个候选：标签列、要不要历史窗口、要不要贝尔曼备份
python scripts/list_finals_models.py

# 序列模型实时循环自检：连触发 6 次，看历史窗口是否累积、单次延迟是否 <50ms
python scripts/check_seq_inference.py --model outputs/finals_model_ts/seq_recur_fqi.pkl
python scripts/check_seq_inference.py --model outputs/finals_model_recur/policy.pkl --delta 15 --triggers 6
```

`check_seq_inference` 正常时会打印类似：

```text
model=seq_recur_fqi.pkl needs_history=True
trigger 0: t=0.0  LJCL=2 cached_frames=1 action_history=[2] 4.6ms
trigger 1: t=15.0 LJCL=2 cached_frames=2 action_history=[2, 2] 7.1ms
trigger 2: t=30.0 LJCL=2 cached_frames=3 action_history=[2, 2, 2] 4.7ms
...
worst latency 7.14ms (budget 50ms)
```

`cached_frames` 应从 1 涨到 3 后停住；若每次都是 1，说明缓存没活（进程被重启，或模型没声明 `needs_history`）。

### 关键参数

**流水线**

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--delta` | 15 | 决策间隔，单位是**秒**。样本按 0.5s 记录时 15s = 30 拍。正式若是 20 秒就改成 20，建集和回放都要一致 |
| `--n-ticks` | 120 | 旧参数兼容；真实切片的轨迹长度由 CSV 的时间列决定 |
| `--delta` | 15 | 相邻真实观测区间的决策间隔（秒）；正式间隔为 20 秒时改成 20 |
| `--max-mid-slices` | 0 | 每条真实轨迹最多保留的区间数；0 表示全部区间 |
| `--idle-keep-ratio` | 0.20 | 无机会、无拦截变化的 idle 区间保留比例；正收益和 hard negative 始终保留 |
| `--workers` | 4 | 按场景并行构造；内存紧张时设为 2 |
| `--fresh` | 关 | 忽略 `outputs/.../_ckpt`，从头建集。默认续传 |
| `--test-ratio` | 0.25 | 按**场景**切分，不是按行，避免同局切片同时进训练和测试 |
| `--replay-scenes` | 3 | 周期回放的场景数；赛方电脑不要开 `--with-replay` |

**模型超参**（在代码里，不走命令行）

| 位置 | 参数 | 值 |
| --- | --- | --- |
| `models.RandomForestUtility` | `n_trees` / `max_depth` / `min_samples_leaf` / `feature_fraction` | 12 / 4 / 6 / 0.75 |
| `models.PairwiseLogistic` | `epochs` / `learning_rate` / `l2` | 200 / 0.08 / 0.003 |
| `models.StrategyScorer` | `pairwise_weight` | 0.35（`rf_pair` = 0.65 森林 + 0.35 pairwise） |
| `models.kind_weights` | 样本权重 | `observed_positive`=2.0，`observed_hard_negative`=1.0，`observed_idle`=0.2 |
| `reward.GAMMA` | 折扣 | 0.85（**必须与势函数 shaping 用的同一个值**，否则策略不变性不成立） |
| `reward.MIX_LAMBDA` | mix 里终局项权重 | 0.25 |
| `reward.LEAK_HT_PENALTY` | 高威胁漏防修正 | 0.05 |
| `sequence.WINDOW` | 历史窗口 K | 3 |
| `seq_models.SeqFittedQ` | `n_iter` | 5 |

奖励定义与四条约束见 `docs/finals_realtime_design.md` 第 5.3 节；序列编码器设计见第 6.5 节。

---

## 5. 怎么根据输出判断模型好坏与改进空间

先跑判读脚本，它把下面所有规则都编码进去了，直接给结论。`--compare` 是对比产出的 `compare.json`，`--dataset` 是对应那次建集的目录（用来读 `summary.json` 做数据层体检）：

```powershell
# 单切片那 10 个候选
python scripts/diagnose_finals_models.py --compare outputs/finals_model/compare.json --dataset outputs/finals_slice

# 贯序 / 时间序列那 11 个候选
python scripts/diagnose_finals_models.py --compare outputs/finals_model_ts/compare.json --dataset outputs/finals_seq

# 真实 CSV 建集之后，路径换成 real 那一套
python scripts/diagnose_finals_models.py --compare outputs/finals_model/compare.json --dataset outputs/finals_slice_real
```

两套不要混：`--compare` 来自 `--seq` 对比时，`--dataset` 必须是 `finals_seq`；单切片对比对应 `finals_slice`。混了会把贯序标签的体检套到单切片数据上，结论是错的。

输出分四段：数据层体检 → 指标分辨力 → 逐模型指标 → 逐模型判定 → 结论。门槛集中在脚本里的 `THRESHOLDS`，改了就是改了，不会两个人按两套标准看同一份输出。

### 三层监测

**第一层：数据层体检**（读训练集的 `summary.json`）

`label_means` 里任何一列均值为 0 就是死信号——那一列学不出东西，后面所有模型指标都是噪声。`n_dropped` 明显多于 `n_rows` 说明 `drop_reason` 滤过头。样本少于 200 行时排序会被方差主导。

**第二层：模型指标**（读 `compare.json` 的 `rows`）

| 字段 | 含义 |
| --- | --- |
| `train_match` / `test_match` | 与 oracle 策略的一致率。真实观测切片按整局拦截率的条件收益定；贯序集按历史态势匹配后的 return-to-go 条件收益定 |
| `train_groups` / `test_groups` | 有唯一最优策略、可计算 match 的对照组数 |
| `test_total_groups` / `test_tied_groups` | 所有完整三策略匹配组，以及最优策略差距不超过 0.01 的近似并列组 |
| `test_collapse` | 最高频策略占比，1.0 即塌缩成固定策略 |
| `test_score_margin` | 三策略打分的最大差 |
| `latency_ms` | 单次推荐耗时，序列模型连编码一起计时 |
| `test_policy_value` | 匹配测试态势上模型实际选中策略的平均拦截收益 |
| `test_mean_regret` | 与该态势 oracle 最优收益的平均差距，越低越好 |
| `test_lift_vs_best_fixed_matched` | 相对测试集上最佳固定策略的收益增量 |
| `episode_intercept_rate` | 周期回放的整局拦截率，**这才是赛方主指标** |
| `lift_vs_best_fixed` | 相对**最好的固定策略**的增量 |
| `selection_score` | 加权总分，`rank` 是排序 |

**第三层：选模打分**。硬门是延迟：超过 50ms 直接判不可提交。有周期回放时使用整局拦截率和回放 lift；真实切片没有回放时，主指标是 `test_policy_value`，`test_mean_regret` 和 `test_match` 只作轻量辅助，避免命中率高但实际拦截收益低的模型排第一。

### 判定规则

**阻塞项（不该提交）**

| 信号 | 说明 |
| --- | --- |
| `latency_ms > 50` | 超时延预算 |
| `test_score_margin ≈ 0` | 三策略打分并列，没有区分能力，输出等价于固定策略 |
| `lift_vs_best_fixed ≤ 0` 或 `test_lift_vs_best_fixed_matched ≤ 0` | 打不过锁死一个策略，整套流程不值得提交 |
| 所有模型 `episode_intercept_rate` 相同 | 回放对策略选择不敏感，权重最高那一项没有分辨力，排序不可信 |
| oracle 在测试集上恒为一个策略 | match 退化成「有没有猜中这一个」 |

**改进空间（可提交但有活干）**

| 信号 | 动作 |
| --- | --- |
| `train_match` 远高于 `test_match`（差 > 0.25） | 过拟合，降模型容量或砍特征维度 |
| `test_collapse = 1.0` | 塌缩，先分清是标签没差异还是模型没学到 |
| `test_groups < 10` | 对照组太少，match 没有统计意义，先扩数据 |
| `seq_fqi` 打不过 `seq_fqi_raw` | 势函数 shaping 没帮上忙，调 `reward.POTENTIAL_W_*` 或去掉 |
| `seq_stack_*` / `seq_recur_*` 打不过单帧 `seq_mix` / `seq_fqi` | 历史没用，别在序列编码上加投入 |

最后两条靠消融格子读（`COMPARE_TS_NAMES`，状态表示 × 是否贝尔曼备份）：

|  | 监督（无备份） | fitted-Q |
| --- | --- | --- |
| 单帧 | `seq_mix` | `seq_fqi` |
| K 帧拼接 | `seq_stack_mix` | `seq_stack_fqi` |
| 递归编码 | `seq_recur_mix` | `seq_recur_fqi` |

同列比较看「序列表示有没有用」，同行比较看「多步备份有没有用」。两者分开看才知道该往哪投。

### 换提交模型的门槛

`compare.json` 里的 `chosen` **只是当前指标排序，不会覆盖 `policy.pkl`**。整份结果能不能过，只看数据层、指标分辨力、以及 **chosen 这一行**；落选候选的 BLOCK 只出现在「逐模型判定」，不否掉整份对比。要真的换提交模型，三个条件同时满足：

1. 判读脚本报告里该模型无阻塞项
2. `chosen` 与它一致
3. `lift_vs_best_fixed` 或 `test_lift_vs_best_fixed_matched` 明显为正（不是 0，不是噪声级的正数）

然后才改 `registry.DEFAULT_SUBMIT`，重训、重打包、重跑隔离验证。

---

## 6. 合成数据与真实数据的边界

`outputs/finals_synth` 只用于检查 CSV 读取、特征构造、断点续建、模型导出和在线推理，不能代替赛方真实实验结果。合成世界的三种策略差异、拦截时刻和场景分布都可能与真实数据不同。

真实数据建集后应重点检查：

- `summary.json` 中 `observed_positive`、`observed_hard_negative` 和 `observed_idle` 的数量；
- `compare.json` 中每个模型的 `test_policy_value`、`test_mean_regret` 和 `test_lift_vs_best_fixed_matched`；
- `matching.train/test` 中的共同支持数量、近邻距离和并列组数量；
- `split.json` 是否按场景划分，且训练集、测试集都包含三种策略。

只有真实测试态势上的收益 lift 稳定为正，并且排序第一模型没有 BLOCK，才考虑替换默认 `rf_pair`。
