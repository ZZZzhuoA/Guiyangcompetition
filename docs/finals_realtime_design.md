# 决赛周期决策设计方案

## 1. 任务定义

赛方样本是「一种策略从头用到尾」的整局 CSV。评估时不是整局锁死一个策略，而是每隔 `Δ ∈ {15s, 20s}` 触发一次，根据当前态势输出

```text
LJCL ∈ {1, 2, 3}
```

目标是**整局拦截率最大**：

```text
J = N_intercepted / N_blue_total
```

`N_blue_total` 含中途新出现的蓝方。一次决策锁定到下一次触发；中间不改策略。

三种策略含义不变：

| LJCL | 名称 | 本方案中的使用假设 |
| --- | --- | --- |
| 1 | 闭合时间最短 | 优先最短遭遇时间 |
| 2 | 拦截性能最佳 | 优先高拦截概率 |
| 3 | 综合效益最优 | 概率 / 消耗 / 时间折中 |

实时入口必须在触发间隔内返回，目标推理时延 **< 50 ms**。

## 2. 为什么不能直接学样本里的 `S_LJCL`

整局单策略轨迹给出的是「这条仿真用了策略 k」，不是「第 t 秒该切到 k」。

若把每个时刻都标成该局的 `S_LJCL`，模型学到的是历史整局标签，和周期重选的评估方式不一致。  
有意义的监督信号是：

```text
在当前态势 x_t 下，从现在起执行策略 a，到下一决策点或终局，能多拦多少
```

## 3. 决策形式

把一局看成决策点序列 `t = 0, Δ, 2Δ, …`。

```text
观测 o_t  -> 特征 x_t  ->  Q(x_t, a)  ->  a* = argmax_a Q
```

`Q(x, a)` 预测的是**从现在起的拦截收益**，主标签为后续拦截增量，辅标签为整局拦截率。

这是离线动作价值估计，不是在线探索型强化学习。推理只做一次前向，满足及时反馈。

不采用在赛方电脑上边打边学的 PPO/DQN：没有稳定仿真回环，且延迟和方差都不可控。

## 4. 状态：当下态势 + 短缓存

接口每次给当前帧。算法自己缓存上一触发时刻，构造「这一段发生了什么」。

### 4.1 当前帧（约 44 维）

- 存活 / 跟踪 / 正在拦截数量
- 威胁排序、最近距离、最短闭合时间、速度
- 低空比例、群数、群规模、群内速度一致性、群中心闭合时间、方位/高低覆盖（自算）
- 四类实体数量、可拦截边数、高概率边、未覆盖目标、高威胁未覆盖
- 资源/目标比、贪心期望拦截概率

### 4.2 跨触发缓存（必须加）

| 特征 | 含义 |
| --- | --- |
| `n_appeared` | 本段新出现蓝方 |
| `n_disappeared` | 本段从态势消失（拦截成功的主证据） |
| `d_min_range` | 最近距离变化 |
| `d_min_tclose` | 最短闭合时间变化 |
| `remain_ratio` | 当前存活 / 本局已见过的蓝方总数 |
| `time_frac` | 已过时间 / 预估局长时间 |
| `last_ljcl` | 上一次输出的策略，首次触发填 0 |

红方只在 `HealthState` 的 `Time=0` 出现，位置固定，从缓存回填，不每帧重读。

## 5. 训练数据：不是所有时刻都有用

原始 CSV 是密时间刻。决策频率是 15/20 秒。密采样里大量相邻帧几乎同质，直接拿来训会稀释真正的切换点。

构造脚本只产出**决策点样本**，并丢掉无意义点。

### 5.1 决策点怎么取

赛方电脑上是多个并列的 CSV 目录，不是 xlsx：

```text
决赛/样本1/*.csv
决赛/样本2/*.csv
决赛/样本3/*.csv
```

每个 `样本N` = 一次仿真、一种策略锁死到底。`--input` 指到 `决赛/`，递归收齐；本地仓库里的 `决赛/样本1/*.xlsx` 只是字段模板，建集会忽略。

1. 读一个样本目录（一次仿真、一种策略）。
2. 按 `FZTime` 对齐整局。
3. 从局开始每隔 `Δ` 取一个决策点；最后一帧若距离上一点 ≥ `Δ/2` 也取。
4. `Δ` 做成参数，默认先按 **15 秒和 20 秒各做一套**，正式间隔确定后只留一套。

时间单位若与仿真刻不是 1:1，用样本里相邻 `FZTime` 差值估计刻长，再换算到 15/20 秒。

### 5.2 哪些点丢掉

满足任一条就丢：

1. 当前监测蓝方为 0。
2. 可拦截边为 0，且最短闭合时间仍很长（现在无事可做）。
3. 与上一决策点的特征 L2 过小（态势几乎没变）。
4. 本段无新增、无消失、最近距离变化低于阈值（空转段）。
5. 已到残局：剩余目标都不可达或全部正在被拦且无新目标。
6. 缺关键列导致位置/句柄无法对齐。

保留的点要写 `keep_reason`，丢掉的点写 `drop_reason`，便于检查是不是滤过头。

### 5.3 标签怎么来（按证据强度分层）

**A. 真对照（权重 3）**  
同一初始态势下有策略 1/2/3 三条整局。`t=0` 的 `x` 相同，三条终局拦截率可直接比。这是最干净的样本。

**B. 决策点分叉（权重 2，周期决策的主数据）**  
在决策点 `t` 把当前快照还原成世界，分别用 1/2/3 往前推 **一个间隔 Δ**（以及可选推到终局）：

```text
r_delta(a)  =  本段新增拦截数 / 本局已出现的蓝方数
r_term(a)   =  若此后一直用 a，整局拦截率
mix(a)      =  r_delta(a) + λ * r_term(a)                        λ = 0.25
r_shaped(a) =  r_delta(a) + γΦ(x') - Φ(x) - 0.05 * 高威胁漏防数   γ = 0.85
```

主目标是整局拦截率，但周期决策首先要选「这一段不漏」。`r_delta` 提供即时差，`r_term` 防止短视。

奖励设计的四条约束写在 `src/finals/reward.py`：

1. **主标量只放拦截增量。** 赛方按整局拦截率计分，漏防率、费效、拦截时间不再揉进同一个数，否则模型在训练集上优化的是一个赛方看不见的量。
2. **紧迫程度走势函数。** `Φ(x) = -(0.04 * threat_pressure + 0.02 * urgency)`，只以 `γΦ(x') - Φ(x)` 的形式进入奖励。势函数型 shaping 不改变最优策略，所以它只能加快学习、不会把策略带偏。shaping 用的 γ 必须和 fitted-Q 的 γ 相同，这条不成立时策略不变性就没了。
3. **高威胁漏防只做小权重修正。** 赛方并不按威胁等级计分，权重 0.05 只是让模型在拦截数相同时偏向先拦高威胁。
4. **成本暂不进奖励。** 集成接口未必提供 `D_Costgy`，训练里让成本变化、评估里它恒定，会学出赛场上兑现不了的取舍。`cost` / `cost_eff` 仍作为诊断列写进 CSV。

分母用「本局已出现的蓝方数」而不是整局蓝方总数：从 `t` 分叉的子局里，`t` 之前已处理掉的目标不该算进分母。三条支路的蓝方出现时刻与策略无关，`build_seq_set` 取三者的最大值作统一分母，保证组内可比。

没有赛方仿真时，用本地构造器做分叉；有真实整局 CSV 时，先用 A + 弱标签 C，等能回放再补 B。

**C. 单策略事后收益（权重 0.4）**  
只有一条整局、不能分叉时，用该策略在 `t` 之后的剩余拦截当弱标签。状态已被该策略塑形，只能当辅样本，不能当主监督。

### 5.4 脚本产出

```text
scripts/build_decision_set.py
  --input outputs/finals_synth   或赛方样本根目录
  --delta 15
  --output outputs/finals_decision_d15
```

产出：

| 文件 | 内容 |
| --- | --- |
| `decision_samples.csv` | 决策点特征、动作、`r_delta`、`r_term`、`utility`、`group_id`、`tier` |
| `dropped_points.csv` | 被滤掉的点及原因 |
| `summary.json` | 保留/丢弃数量、三策略标签分布、分叉覆盖率 |

`group_id = scene_id + time`，同一决策点的三个动作必须成组，供 pairwise 使用。

### 5.5 已知问题：合成数据上标签无信号（待真实样本后复查）

在 `outputs/finals_synth` 上建贯序集，Δ=15 时 48 行 `r_delta` 全为 0，`r_shaped` 退化成只剩 shaping 项；Δ=40 才出现信号：

| Δ | r_delta 均值 | r_term 均值 |
| --- | --- | --- |
| 15 | 0.000 | 0.086 |
| 40 | 0.034 | 0.500 |

即 `world_from_snapshot` 重建出的世界要 20 个 tick 以上才落下第一次拦截。同时 16 个对照组里只有 3 组的 `r_term` 在三策略间有差异。

后果：`r_delta ≡ 0` 时任何贯序模型都学不到东西，`seq_delta` 拿到 `test_match = 1.0` 只是因为它退化成了 `constant_1`。**在合成数据上得到的选模排序不可信。**

更尖锐的证据来自周期回放：`fixed_1`、`fixed_2`、`fixed_3` 和四个贯序/时间序列模型的整局拦截率**完全相同**（都是 0.3571），所以 `lift_vs_best_fixed` 结构性地恒为 0，`selection_score` 里权重最高的那一项根本没有分辨力。这说明问题不只在分叉标签，**本地仿真器的 `_assign` 对策略 1/2/3 就不敏感**——这是上面所有现象的共同根因。

三个待查方向，真实样本到手后一起看：

1. `_assign` 为什么对策略 1/2/3 不敏感：拦截率可能被红方拦截弹数量或可达性卡住了，策略选择根本不影响结果。这一条不解决，任何选模指标都没有意义。
2. 重建世界为什么迟迟不出拦截——快照可能丢了红方的跟踪/分配状态，需要重新捕获。真实分叉会踩同一个坑。
3. 参考轨迹全是策略 1，导致 `a_prev` 恒为 1，动作历史这一路输入携带的信息量为 0。真实样本里参考轨迹会有别的策略，推理时是我们自己的选择，这一维才会活起来。

## 6. 模型：同一接口、多种候选，有数据后再选

所有候选都实现 `fit(...)` 和 `recommend(x) -> (LJCL, scores[3])`。提交入口 `recommend()` 不随模型变。

| 名称 | 类型 | 作用 |
| --- | --- | --- |
| `constant_1/2/3` | 基线 | 全程锁死一个策略 |
| `rule` | 基线 | 紧迫→1，高概率边→2，否则→3 |
| `rf_utility` | Q 树 | numpy 随机森林回归 U(x,a) |
| `pairwise` | 排序 | 同一切片三策略成对比较 |
| `rf_pair` | 默认提交 | 0.65 森林 + 0.35 pairwise |
| `ridge_q` | Q 线性 | 岭回归 U(x,a) |
| `mlp_q` | Q 小网 | 32 隐层，三动作头 |
| `oracle_clf` | 分类 | 用对照样本的最优动作，不是整局 `S_LJCL` |
| `seq_delta` | 贯序 | 只学本段 Δ 拦截增量，最短视的对照 |
| `seq_mix` | 贯序 | `r_delta + 0.25 r_term` |
| `seq_shaped` | 贯序 | 学 `r_shaped`，不做 bootstrap |
| `seq_pair` | 贯序排序 | 对 mix 标签做 pairwise |
| `seq_blend` | 贯序混合 | 森林 mix + pairwise |
| `seq_fqi` | 离线 RL | fitted-Q：`r_shaped + γ max Q(x',a')` |
| `seq_fqi_raw` | 离线 RL | 同上但奖励用裸 `r_delta`，用来验证 shaping 有没有帮助 |
| `seq_stack_mix` | 时间序列 | K 帧拼接编码 + mix 监督 |
| `seq_stack_fqi` | 时间序列 RL | K 帧拼接编码 + fitted-Q |
| `seq_recur_mix` | 时间序列 | 递归编码 + mix 监督 |
| `seq_recur_fqi` | 时间序列 RL | 递归编码 + fitted-Q |

对照组的「正确答案」统一按赛方主指标定：贯序数据集里 `evaluate_groups` 用 `r_term` 当 oracle，而不是各模型自己的标签，否则 match 会奖励那些标签定义得最宽松的模型。

### 6.5 时间序列 RL：贯序 RL + 序列编码器

`seq_fqi` 已经是贯序离线 RL，但状态仍是当前 64 维，历史只通过 `last_ljcl`、新增/消失那几维挤进来。时间序列 RL 差的是**状态表示**：把最近 K 次触发整段编码成状态。约定见 `src/finals/sequence.py`，`WINDOW = 3`：

```text
x_hist[m]   (K, F)   末行是当前帧，靠前是更早的触发，不足前面补 0
a_prev[m]   (K-1,)   当前帧之前那几拍已经选过的动作，0 表示没有
hist_len[m]          窗口里有几行是真的，冷启动为 1
```

当前帧要选的动作不进编码器，仍走 `RandomForestUtility` 既有的 append_strategy 机制。下一状态不另存，由 `push_window` 推出来：末帧换 `x_next`，本拍选的动作补进 `a_prev`。所以「这一拍选什么」会改变下一拍的状态表示，这正是它比单帧 fitted-Q 多出来的东西。

两种编码器：

- **`stack`**：K 帧原样拼接 + 过去动作 one-hot，`K*F + (K-1)*3 = 198` 维。最直白，用来先回答「历史到底有没有用」。维度相对样本量偏高，过拟合风险靠消融看。
- **`recurrent`**：GRU 单元的递归编码，**递归权重固定随机**，只训练下游读出头（储备池 / echo-state 那一类），读出向量拼上当前帧作直连。

递归权重为什么不做 BPTT：赛方电脑上只有 numpy（本项目所有模型都是手写的，`MLPQ` 是手工反向传播），一局决策点才十来个，合成数据上目前连 `r_delta` 都恒为 0，反向传播训出来的权重没法验证，还要保证单次推理 <50ms。固定随机递归在这个数据量下更稳，也仍然是真正的递归表示：隐状态携带整段窗口。真实数据够多之后把递归权重换成 BPTT 训练的即可，接口不用动。

**消融格子**（`COMPARE_TS_NAMES`）：状态表示 × 是否贝尔曼备份，用来分清增益来自序列还是来自 RL。

|  | 监督（无备份） | fitted-Q |
| --- | --- | --- |
| 单帧 | `seq_mix` | `seq_fqi` |
| K 帧拼接 | `seq_stack_mix` | `seq_stack_fqi` |
| 递归编码 | `seq_recur_mix` | `seq_recur_fqi` |

序列必须由我们自己缓存：赛方接口每次只给当前三张表。`DecisionContext.window_with()` 拼窗口，`push_history()` 在推荐之后写入，`FinalsPolicy` 只在模型声明 `needs_history` 时才传。训练和推理共用 `sequence.build_window`，避免两边窗口含义漂移。历史含**每个被访问到的决策点**，包括训练里被 `drop_reason` 丢掉的——推理时策略在每次触发都会被调用。

窗口只写进 `training_samples.npz`，CSV 存不下 `(K, F)`。只有 CSV 时序列模型会退化成单帧冷启动，不会报错。

自检脚本（跑实时循环，看历史累积和延迟）：

```powershell
python scripts/check_seq_inference.py --model outputs/finals_model_ts/seq_recur_fqi.pkl
```

实测最坏单次推理 7ms（预算 50ms），窗口在第 3 次触发后填满并正确滚动。

选模脚本：

```powershell
python scripts/build_seq_set.py --input outputs/finals_synth --output outputs/finals_seq --delta 15
python scripts/compare_finals_models.py --seq
python scripts/compare_finals_models.py --seq --models seq_mix seq_fqi seq_stack_mix seq_stack_fqi seq_recur_mix seq_recur_fqi
python scripts/train_finals_model.py --dataset outputs/finals_seq --model seq_recur_fqi
```

### 6.1 特征分组（可消融）

- `snapshot`（44）：当前帧数量、威胁、距离、闭合时间、群内速度一致性、群中心闭合时间、四类实体、可拦截边、覆盖
- `temporal`（14）：相对上一决策点的出现/消失、距离与闭合时间变化、剩余比、时间进度、上次 LJCL
- `derived`（6）：urgency、coverage_gap、quality_edge、scarcity、threat_pressure、swarm_pressure

旧 CSV 缺新列时按 0 对齐，不必重造也能先训。

### 6.2 规则托底

解析失败或没有 scorer 时走 `rule`。时延超过 50 ms 的候选直接淘汰。

### 6.3 不作为提交主路径

深度 RL、大网络、每帧决策：赛场无稳定仿真，且评估是 15/20 秒触发。

### 6.4 选模指标

1. **主指标**：周期回放整局拦截率，以及相对最好固定策略的提升
2. **对照 match**：同一 `x` 上三策略标签，预测是否等于效用最高者
3. **塌缩率**：是否几乎只输出某一个 LJCL
4. **时延**：单次 `recommend` < 50 ms

没有真实数据时，match 只能说明管线通，不能定提交模型。有回放后按 `selection_score` 排序，再用 `train_finals_model.py --model <name>` 写出 `policy.pkl`。

## 7. 实时循环

```text
初始化缓存
每次触发:
  解析目标列表、可拦截列表、装备列表
  红方位置用 Time=0 的 HealthState / 首次装备表
  算 x_t（含与上次触发的差分）
  输出 LJCL = argmax Q
  更新缓存（目标集合、上次策略、上次特征）
```

约束：

- 无磁盘、无重训。
- 特征维度固定，缺列填默认，多列忽略。
- 输出只允许 1/2/3。

## 8. 本地怎么检验「全程拦截率」

在构造世界里按 Δ 切换策略，比四种基线：

1. 整局固定 1
2. 整局固定 2
3. 整局固定 3
4. 周期 Q 策略

指标：整局拦截率、高威胁漏防数、平均决策时延。  
周期策略应不低于最好的固定策略，并在「前半段该抢时间、后半段该抢概率」的局上拉开差距。

## 9. 和现有代码的关系

单切片打分仍是训练主路径。特征已扩到三组；模型经注册表对比。

```powershell
python scripts/build_slice_set.py --input outputs/finals_synth --output outputs/finals_slice
python scripts/compare_finals_models.py --dataset outputs/finals_slice --output outputs/finals_model
python scripts/train_finals_model.py --dataset outputs/finals_slice --output outputs/finals_model --model rf_pair
```

样本格式维持现状。`recommend()` 入口不动。

## 10. 提交与判读

操作手册在 `submission_finals/README_submit.md`：赛方电脑上需要的文件清单、数据路径、全部训练参数、实时调用契约、以及怎么根据输出判断模型好坏。本节只记要点，细节不在这里重复。

**提交包**由脚本生成并自验，不要手工拷：

```powershell
python scripts/make_submission_package.py --output outputs/submission_package
```

它挑出推理闭包（13 个 `src/finals` 模块 + `submission_finals/` + `policy.pkl`，18 个文件约 128 KB），然后在**只含包内文件**的临时目录里真跑一次 `recommend()`，跑不通就报错。避免「文档说够了、实际少文件」。

**实时调用有两条硬约束**，违反了模型静默变差而不报错：必须在同一进程里反复调用（14 维时序特征和序列窗口都靠进程内缓存，换进程就全是 0）；`time` 要传真实 `FZTime`（否则 `time_frac` 恒为 0）。`recommend()` 把 policy 缓存成模块级单例就是为这个，每局开始前调 `reset()`。

**判读交给脚本**，门槛集中在 `THRESHOLDS`，不靠人记：

```powershell
python scripts/diagnose_finals_models.py --compare <compare.json> --dataset <训练集目录>
```

分四段输出：数据层体检（`label_means` 有列为 0 就是死信号）、指标分辨力（所有模型拦截率相同 / oracle 恒为一个策略 → 排序不可信）、逐模型指标、逐模型判定。阻塞项是延迟超 50ms、`test_score_margin ≈ 0`、`lift_vs_best_fixed ≤ 0`；改进项是过拟合、塌缩、对照组太少。

换提交模型要三个条件同时满足：判读无阻塞项、`chosen` 与之一致、`lift` 明显为正。然后才改 `registry.DEFAULT_SUBMIT`。`compare.json` 里的 `chosen` 本身不会覆盖 `policy.pkl`。

## 11. 落地顺序

1. 先把特征、多模型接口、指标固定（已完成）。
2. 提交包、判读脚本、实时缓存契约固定（已完成）。
3. **让构造仿真里 1/2/3 行为可区分**（当前阻塞点，见 5.5 节）。这一条不解决，第 2 步搭的指标全是瞎的。
4. 本地周期回放对比固定 1/2/3，`lift` 能稳定为正再谈换提交模型。
5. 赛方 CSV 和正式 Δ 到了之后，只改 `--input` / `--delta` 重训、重对比、重打包，推理入口不动。
