# PRBCCMO 当前总文档

## 算法背景

`PRBCCMO` 面向的是 `binary / unknown constraints` 下的约束多目标优化。该场景没有可直接利用的 violation degree，因此真正值得学习的不是“违反了多少”，而是“可行域与不可行域的分界在哪里，以及这些边界信息是否能帮助找到更有价值的可行解”。

相对 `NA-EMT` 一类把模型当作“不可行解价值预测器”的路线，`PRBCCMO` 当前工作树已经转向另一条主线：把学习器作为边界查询调度器，在双群体搜索框架内优先寻找对 Pareto 搜索有意义的边界点，再对选中的种子做标签感知的局部细化。

因此，`PRBCCMO` 的研究问题不是“再做一个双群体算法”，也不是“再套一个 MLP”，而是：

`在 binary unknown constraints 下，能否做 Pareto-relevant 的主动边界查询，并把这些边界查询转化为有用的可行发现。`

## 创新点

严格口径下，目前只保留一个核心创新点：

`Pareto-relevant boundary query under binary unknown constraints`

更贴近当前实现的表述是：

- 用边界概率与不确定性识别 near-boundary 候选，而不是学习传统 violation degree。
- 用 Pareto/sector 相关性筛掉“虽然靠边界、但对多目标搜索没价值”的候选。
- 对选中的 boundary seed 做标签感知 refinement：
  - 可行 seed 向更优可行区域推进。
  - 不可行 seed 做 bracket 收缩或 hard-negative 确认。

下面这些是支撑机制，不应再当作独立创新点：

- `P_C + P_U` 双群体框架
- 轻量 MLP / committee / calibration 变体
- 外部可行档案 `ExternalArchive`
- sector-level 的局部迁移与补充排序项

最新结论同样明确：当前不应继续扩写创新点，也不应继续堆新机制；重点是把这一个创新点验证干净。

## 当前实现方式

### 1. 主循环

当前主流程位于 [PRBCCMO.m](/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Algorithms/Multi-objective%20optimization/PRBCCMO/PRBCCMO.m)：

1. 常规双群体进化产生 `PopulationC / PopulationU` 的 regular offspring。
2. 构建 `FeasibleAnchorPool`：
   - 可行 `PopulationC`
   - 可行 regular offspring
   - `ExternalArchive` 中已有的可行非支配解
3. 基于可行 anchor 与 `PopulationU` 生成 boundary candidate pool。
4. 用已训练的 boundary model 对候选做评分与筛选，给定 boundary budget 后真实评估。
5. 根据真实标签进入 refinement：
   - feasible expansion
   - infeasible bracketing
   - hard-negative confirmation
6. 更新训练、校准、测试缓冲，按触发规则重训/热启动模型。
7. 环境选择，继续下一代。

### 2. 边界建模与 trust 机制

当前实现不是固定单一模型，而是“轻量 committee + calibration/trust 审计”的结构：

- 训练入口：`TrainBoundaryMLP`
- 预测入口：`PredictBoundaryMLP`
- 校准/审计：`EvaluateBoundaryCalibration`、`RefreshBoundaryTrust`
- 变体：`raw / temperature / beta / auto_trust`

当前模型目标已经不是“全局 feasibility classifier”，而是“边界附近的概率是否有可解释语义”。这也是后续必须转向 `boundary semantics first` 训练的原因。

### 3. 候选生成、筛选与 refinement

关键文件如下：

- [GenerateBoundaryCandidates.m](/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Algorithms/Multi-objective%20optimization/PRBCCMO/GenerateBoundaryCandidates.m)
- [SelectBoundaryCandidates.m](/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Algorithms/Multi-objective%20optimization/PRBCCMO/SelectBoundaryCandidates.m)
- [ScoreBoundaryCandidates.m](/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Algorithms/Multi-objective%20optimization/PRBCCMO/ScoreBoundaryCandidates.m)
- [RefineBoundaryWorkers.m](/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Algorithms/Multi-objective%20optimization/PRBCCMO/RefineBoundaryWorkers.m)

当前实现已经具备以下关键特征：

- shared-sector 桥接生成
- two-stage activation gate
- selection eligibility diagnostics
- seed-stage 与 worker-stage 使用一致的 feasible reference 语义
- refinement 阶段区分 feasible / infeasible seed 的后续动作

### 4. 诊断与实验脚本

当前用于验证和续做实验的入口主要是：

- [benchmark_PRBCCMO_experiment0.m](/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Algorithms/Multi-objective%20optimization/PRBCCMO/benchmark_PRBCCMO_experiment0.m)
- [diagnose_PRBCCMO_boundary_activation.m](/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Algorithms/Multi-objective%20optimization/PRBCCMO/diagnose_PRBCCMO_boundary_activation.m)
- [benchmark_PRBCCMO_crossset_suite.m](/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Algorithms/Multi-objective%20optimization/PRBCCMO/benchmark_PRBCCMO_crossset_suite.m)

当前 `Algorithm.metric.sectionB` 已直接保留：

- `activationTrace`
- `selectionTrace`
- `bridgeTrace`

因此后续调试和预实验不再需要额外的大型 trace dump 文件。

## 当前已经验证的内容

### 1. 旧 experiment-0 结果的定位已经明确

保留下来的 `2026-03-23` 正式实验结果只能作为 pre-fix baseline，不再代表当前代码状态。

基线事实如下：

- 设置：`15 runs x 9 DASCMOP_BC x 4 calibration variants`
- 结果：全部 `135000 / 135000` updates invalid，全部 single-class，`pooledCount = 0`
- 解释：这是真实的旧版本失败快照，但不能再被当作当前版本结论

对应保留载体已合并为：

- `benchmark_PRBCCMO_20260324_bundle.csv`
- `benchmark_PRBCCMO_20260324_bundle.mat`

### 2. 当前代码层面已经确认的修复

已并入本文的 `fix_experiment0_context.txt` 曾记录以下修复；这些修复现在已经进入当前工作树并构成现状态前提：

- 可行 anchor 供给已修复：`FeasibleAnchorPool` 不再只依赖 `PopulationC`
- cold-start holdout bootstrapping 已加入，早期 single-class 不再被误判为 calibration 方法本身失效
- `ScoreBoundaryCandidates.m` 的 eligibility bug 已修复
- seed-stage / worker-stage 的 feasible reference 语义已对齐
- `GenerateBoundaryCandidates.m` 增加 two-stage gate 与显式 bridge diagnostics
- Section B traces 已内存化，可直接由诊断脚本消费

### 3. Activation coverage 已经成立

来自 `benchmark_PRBCCMO_20260324_bundle.mat` 中 `activation.family` 子表的当前结论：

| 家族 | 问题数 | boundary_started 问题数 | meanSharedSectorRunRatio | meanActiveSectorRunRatio | 结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| DASCMOP_BC | 9 | 9 | 1.0 | 1.0 | 已全覆盖启动 |
| LIRCMOP_BC | 14 | 14 | 1.0 | 1.0 | 已全覆盖启动 |
| MW_BC | 14 | 14 | 1.0 | 1.0 | 已全覆盖启动 |

这说明最新证据已经足以否定旧结论“边界模块在 BC 问题上普遍死亡”。

### 4. 全量 calibration / trust 结果

来自 `benchmark_PRBCCMO_20260324_bundle.mat` 中 `calibration.family` 子表的家族级汇总如下：

| 家族 | 变体 | pooledECE | pooledCoreNearGap | meanTrustGatePassRate | 当前判断 |
| --- | --- | ---: | ---: | ---: | --- |
| DASCMOP_BC | raw | 0.356622 | 0.377958 | 0.000000 | activation 有了，但 trust 很弱 |
| DASCMOP_BC | temperature | 0.365577 | 0.390688 | 0.000000 | 比 raw 更差 |
| DASCMOP_BC | beta | 0.326113 | 0.382862 | 0.000000 | ECE 最好，但仍远未过关 |
| DASCMOP_BC | auto_trust | 0.330596 | 0.372873 | 0.000000 | near-gap 略优，但 gate 仍全关 |
| LIRCMOP_BC | raw | 0.086462 | 0.094627 | 0.122112 | 三家族中最好 |
| LIRCMOP_BC | temperature | 0.091600 | 0.090332 | 0.139495 | near-gap 略优于 raw |
| LIRCMOP_BC | beta | 0.079807 | 0.067386 | 0.135521 | calibration 指标最好 |
| LIRCMOP_BC | auto_trust | 0.085835 | 0.085371 | 0.187136 | trust gate 通过率最高 |
| MW_BC | raw | 0.219000 | 0.226421 | 0.049221 | 中等，明显弱于 LIR |
| MW_BC | temperature | 0.215434 | 0.225066 | 0.052282 | 略好于 raw |
| MW_BC | beta | 0.198068 | 0.199860 | 0.052070 | 当前 MW 上 calibration 最优 |
| MW_BC | auto_trust | 0.206153 | 0.218171 | 0.052193 | 与 beta 接近，但未根治 |

总体 pooled 汇总来自 `benchmark_PRBCCMO_20260324_bundle.mat` 中 `calibration.pooled` 子表：

- `beta` 当前综合最优：`pooledECE = 0.185900`
- `auto_trust` 当前综合 trust 通过率最高：`meanTrustGatePassRate = 0.078850`
- 但整体仍不能据此宣称“trustworthy boundary querying”已成立

### 5. 当前可下的结论

截至目前，只能稳妥下列结论：

- `boundary activation coverage` 已经被验证
- 当前主瓶颈已从“完全不启动”转为“trust/calibration 语义弱且家族依赖”
- 家族表现分层明显：`LIRCMOP_BC > MW_BC > DASCMOP_BC`
- 当前证据仍不足以证明：
  - scheduler 选到的 query 更接近真实边界
  - 这些 boundary points 有显著 downstream usefulness
  - 整个 boundary 机制栈已经被验证成立

## 当前主要问题

### 1. 科学主张尚未闭环

当前证据链只到：

`模块能启动`

还没有闭合到：

`更可信的边界语义 -> 更靠近真实边界 -> 更有用的可行发现 -> 更好的最终优化`

### 2. 真正瓶颈不是“机制不够多”，而是“概率语义不够强”

最新追加分析已经明确指出，不建议继续往主算法里塞新机制。当前最值得改的不是概念层，而是以下三点：

- shared-sector 内的配对质量
- trust 的使用方式，从 hard gate 改成 soft weighting
- 训练目标转向 `boundary semantics first`

### 3. 历史 hard case 仍有方法学意义

已并入本文的 `fix_experiment0_context.txt` 曾把 `DASCMOP5_BC` 识别为 shared-sector activation 的典型 hard case。虽然最新 `200000 FE` 全量 activation 结果已经显示 `DASCMOP5_BC` 最终能够 `boundary_started`，但它仍应保留为下一阶段 pairing/activation 预实验的重点检验对象，因为它最能暴露 pair constructor 的鲁棒性问题。

## 下一步的计划

以下内容以旧 `fix.md` 最新追加分析为准。

### 总原则

1. 先冻结一个 clean commit/tag，再做正式实验。
2. 先不要改科学主张，只继续收集能支撑或反驳主张的证据。
3. 不加新机制，不扩创新点，不再做“大礼包”式堆叠。
4. 方法上优先改三件事：
   - `shared-sector / top-K pairing`
   - `soft trust weighting`
   - `boundary semantics first` 训练

### 方法调整方向

#### A. 配对方式

把当前桥接配对从“有 shared sector 就配”进一步收紧为更可靠的 sector 内配对，优先尝试 `TopKPair`。目标不是增加候选数，而是提高 shared-sector 内真正有语义的 feasible/infeasible pair 比例。

#### B. trust 使用方式

不要再把 trust 当作硬开关。改成软权重，让模型在 trust 很差时自动回退到 Pareto-driven selection，在 trust 变好时再逐步放大 `p ≈ 0.5` 的作用。

#### C. 训练目标

训练、校准、测试三套缓冲严格隔离；训练时优先过采样 tight brackets；损失设计围绕“边界附近概率是否有语义”，而不是追求一般分类精度。

#### D. 打分简化

主版本只保留最小必要三项：

- eligibility
- trust-corrected uncertainty
- Pareto value

不要再回到 `entropy / hvGain / novelty / penalty` 的厚打分栈。

### 实验 A：桥接配对与激活的修复性预实验

目的：验证 `pair constructor` 是否真正缓解 `activation_gap_not_met`，尤其针对 `DASCMOP5_BC`。

建议设置：

- 问题：`DASCMOP5_BC`、`DASCMOP6_BC`、`LIRCMOP1_BC`、`MW1_BC`
- `Population = 100`
- `MaxFE = 50000`
- `Runs = 20`
- paired seeds
- 比较：`CurrentPair` vs `TopKPair`

建议指标：

- `SSR`：shared-sector 出现比例
- `ASR`：active-sector 出现比例
- `PMR`：shared pairs 中 `RawMargin > 0` 的比例
- `FSG`：首次产生 boundary seed 的代数
- `BSR`：是否成功产生 boundary seed

验收标准：`TopKPair` 在 D5 上显著改善 `ASR / PMR / BSR` 并降低 `FSG`，且不伤害 D6/LIR1/MW1。

### 实验 B：校准 / trust 审计

目的：回答当前 `p ≈ 0.5` 是否已有可用语义。

建议设置：

- 全 `37` 个 BC 问题
- `Population = 100`
- `MaxFE = 200000`
- `Runs = 30`
- 变体：`raw / temperature / beta / auto_trust`
- 只统计 `auditReadyUpdates`
- 主结论单位必须是 `run`，不能再以 pooled all rows 直接下主结论

建议指标：

- `Brier`
- `ECE`
- `CoreNearGap`
- `TGP`：trust gate pass rate
- `TWS`：trust weight score

成功标准：改进版在 `DAS / LIR / MW` 三家族上相对 raw 都显著降低 `ECE / CoreNearGap`，并提高 `TGP / TWS`。

### 实验 C：核心创新的 oracle boundary audit

目的：直接验证 scheduler 选到的 query 是否更靠近真实边界。

建议设置：

- 优先 `9` 个 `DASCMOP_BC`
- 若 `LIR/MW` 也能方便记录原始连续约束，再扩展到全 `37`
- `Population = 100`
- `MaxFE = 200000`
- `Runs = 30`
- 比较：
  - `Full`
  - `Uncertain-only`
  - `HighProb-Boundary`
  - `Rand-Boundary`

关键要求：

- 四个变体必须共享相同 bridge generation、相同 boundary budget、相同主搜索框架
- 离线记录 `oracle_dB`
- 新增 `CandidateAudit`

主指标：

- `mean_dB`
- `median_dB`
- `QP_tau`
- `Spearman(utility, -d_B)`

主假设：`Full` 必须显著优于 `Rand-Boundary / Uncertain-only / HighProb-Boundary`。如果这组不赢，创新点不能成立。

### 实验 D：boundary points 的 downstream usefulness

目的：验证“更靠近边界”是否真的能转化成有价值的可行发现。

建议设置：

- 与实验 C 相同
- 再加入 `No-local-label` 变体

新增日志：

- `BoundaryLineage`
- `ArchiveEvent`

主指标：

- `FRR`
- `TBR`
- `FIR`
- `UBY`
- `MSR`
- `ΔHV_B`
- `TTU`

关键要求：`ΔHV_B` 必须按 `event-level` 统计，不能再用 generation-batch 粗口径。

### 实验 E：最终优化效果

目的：在内部因果证据成立后，再补论文常规性能表。

建议设置：

- 至少全 `9` 个 `DASCMOP_BC`
- 最好扩到全 `37` 个 BC 问题
- `Population = 100`
- `MaxFE = 200000`
- `Runs = 30`

比较方式：

- 内部主比较：`Full vs NoBoundary`
- 外部参考：`NA-EMT` + 2 到 3 个最强 binary/unknown-constraint baselines

指标：

- `HV`
- `IGD`
- `AUC-HV`
- `FHT`

统计：

- 以 `run` 为单位
- paired Wilcoxon signed-rank + Holm
- paired median difference + `95% bootstrap CI`

最终要形成的证据链只能写成：

`pairing 修复 -> trust 改善 -> selected seeds 更接近真实边界 -> boundary discoveries 更有用 -> 最终 HV/IGD/AUC 更好`

只要这条链断一段，就不能宣称创新点“已验证成立”。

## 当前建议保留的文件

代码与脚本：

- `Algorithms/Multi-objective optimization/PRBCCMO/PRBCCMO.m`
- `Algorithms/Multi-objective optimization/PRBCCMO/GenerateBoundaryCandidates.m`
- `Algorithms/Multi-objective optimization/PRBCCMO/ScoreBoundaryCandidates.m`
- `Algorithms/Multi-objective optimization/PRBCCMO/SelectBoundaryCandidates.m`
- `Algorithms/Multi-objective optimization/PRBCCMO/RefineBoundaryWorkers.m`
- `Algorithms/Multi-objective optimization/PRBCCMO/benchmark_PRBCCMO_experiment0.m`
- `Algorithms/Multi-objective optimization/PRBCCMO/diagnose_PRBCCMO_boundary_activation.m`
- `Algorithms/Multi-objective optimization/PRBCCMO/benchmark_PRBCCMO_crossset_suite.m`

当前建议保留的实验结果：

- `benchmark_PRBCCMO_20260324_bundle.csv`
- `benchmark_PRBCCMO_20260324_bundle.mat`

## 一句话现状态

`PRBCCMO` 当前已经证明“边界模块可以稳定启动”，但还没有证明“核心创新点已经成立”；下一阶段不该改题，不该加机制，而该围绕 `pairing -> trust -> oracle boundary audit -> downstream usefulness -> final performance` 这条证据链继续做干净验证。
