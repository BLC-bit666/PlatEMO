# PRBCCMO 当前总文档

## 算法背景

`PRBCCMO` 面向的是 `binary / unknown constraints` 下的约束多目标优化。该场景没有可直接利用的 violation degree，因此真正值得学习的不是“违反了多少”，而是“可行域与不可行域的分界在哪里，以及这些边界信息是否能帮助找到更有价值的可行解”。

相对 `NA-EMT` 一类把模型当作“不可行解价值预测器”的路线，`PRBCCMO` 当前工作树已经进一步收缩成一条更干净的主线：`TopKPair + midpoint placement + ParetoOnly`。学习器仍保留在 calibration/trust 审计与 `Full-v2` 实验分支中，但默认不再在线主导主选择。

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
4. 主线默认固定 `midpoint placement + ParetoOnly`：
   - bridge 候选先由 `TopKPair + two-stage fallback` 生成；
   - probe placement 固定中点，默认关闭 trusted bridge scan；
   - seed 排序主线只按 `paretoValue` 做 `ParetoOnly` 选择；
   - boundary model 产生的 `prob / queryScore / boundaryTrust` 主要保留给审计与 `Full-v2` 实验分支。
5. 根据真实标签进入 refinement：
   - feasible expansion
   - infeasible bracketing
   - hard-negative confirmation
6. 更新训练、校准、测试缓冲，按触发规则重训/热启动模型。
7. 环境选择，继续下一代。

### 2. 边界建模与 trust 机制

当前实现保留“轻量 committee + calibration/trust 审计”的结构，但它已经不再是主线在线调度器：

- 训练入口：`TrainBoundaryMLP`
- 预测入口：`PredictBoundaryMLP`
- 校准/审计：`EvaluateBoundaryCalibration`、`RefreshBoundaryTrust`
- 开发默认 calibrator：`raw / beta`
- `temperature / auto`：降级为补充实验分支
- 默认运行时：`DisableBridgeScan = true`，模型不再在线改写 probe placement

当前模型目标已经不是“全局 feasibility classifier”，而是“边界附近的概率是否有可解释语义”。但这部分语义目前只被允许以审计和实验分支的身份存在，不能再偷偷影响主线基线。

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

- [check_PRBCCMO_step0_engineering.m](/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Algorithms/Multi-objective%20optimization/PRBCCMO/check_PRBCCMO_step0_engineering.m)
- [check_PRBCCMO_step2_query_trust.m](/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Algorithms/Multi-objective%20optimization/PRBCCMO/check_PRBCCMO_step2_query_trust.m)
- [benchmark_PRBCCMO_experiment0.m](/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Algorithms/Multi-objective%20optimization/PRBCCMO/benchmark_PRBCCMO_experiment0.m)
- [diagnose_PRBCCMO_boundary_activation.m](/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Algorithms/Multi-objective%20optimization/PRBCCMO/diagnose_PRBCCMO_boundary_activation.m)
- [benchmark_PRBCCMO_crossset_suite.m](/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Algorithms/Multi-objective%20optimization/PRBCCMO/benchmark_PRBCCMO_crossset_suite.m)

当前 `Algorithm.metric.sectionB` 已直接保留：

- `candidateAudit`
- `activationTrace`
- `selectionTrace`
- `calibrationTrace`
- `updateAudit`

此外，`check_PRBCCMO_step2_query_trust.m` 现在还会直接导出 run-level `queryReport` 和 update-level `queryUpdateReport`，因此 `O_t / D_t` 与 oracle 指标已经不需要再从 MAT 内部手工回推。

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
- 主线代码口径已经统一成：`TopKPair + midpoint placement + ParetoOnly`
- trusted bridge scan 已退出主版本，`Full-v2` 只保留为实验分支
- 开发默认 calibrator 已收缩为 `raw / beta`
- 当前主瓶颈已从“完全不启动”转为“trust/calibration 语义弱且家族依赖”
- 当前更具体的 Step 2 瓶颈是：`Full-v2` 还没有稳定证明自己和 `ParetoOnly` 选得不一样
- 家族表现分层明显：`LIRCMOP_BC > MW_BC > DASCMOP_BC`
- 最新 `37 problems x 3 methods x 5 runs` 的 Step 1 screening 已经给出新的配对结论：
  - `TopKPair` 进入下一轮
  - `RandSectorPair` 保留为唯一强备选
  - `MainBridge` 暂时淘汰，不再进入下一轮模块实验
- 从现在开始，后续模块实验统一以 `TopKPair` 作为 bridge 基准来分析其余模块；`RandSectorPair` 只在需要强结构对照时保留，`MainBridge` 只保留历史对照身份
- `check_PRBCCMO_step2_query_trust.m` 已经补齐：
  - update-level `selection overlap` 与 `boundary score dispersion`
  - run/update 两层 stop/go 字段
  - 自动闸门：若 `Full-v2` 仍塌缩到 `ParetoOnly`，则自动停止后续 oracle audit
- 当前 smoke test 的现实信号并不乐观：自动闸门已经能正常触发，说明现阶段更大的风险仍然是 `Full-v2 ≈ ParetoOnly`
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

### 2. 当前最紧迫的瓶颈不是“再加机制”，而是“先证明 Full-v2 真有分化”

最新追加分析和当前脚本状态已经把优先级压得很明确。现在最值得盯住的不是概念扩写，而是以下三点：

- `Full-v2` 相对 `ParetoOnly` 是否真的产生不同选择
- `boundaryTrust` 在 eligible pool 上是否真的有足够排序分辨率
- 只有在前两项过关后，才值得继续做 oracle/downstream audit

### 3. 历史 hard case 仍有方法学意义

已并入本文的 `fix_experiment0_context.txt` 曾把 `DASCMOP5_BC` 识别为 shared-sector activation 的典型 hard case。虽然最新 `200000 FE` 全量 activation 结果已经显示 `DASCMOP5_BC` 最终能够 `boundary_started`，但它仍应保留为下一阶段 pairing/activation 预实验的重点检验对象，因为它最能暴露 pair constructor 的鲁棒性问题。

## 下一步的计划

以下内容以旧 `fix.md` 最新追加分析为准。

### 总原则

1. 先冻结一个 clean commit/tag，再做正式实验。
2. 先不要改科学主张，只继续收集能支撑或反驳主张的证据。
3. 不加新机制，不扩创新点，不再做“大礼包”式堆叠。
4. 方法上优先改三件事：
   - 保持 `TopKPair + midpoint placement + ParetoOnly` 主线稳定
   - 用 Step 2 自动 stop/go 先判断 `Full-v2` 是否值得继续
   - 只有在 `Full-v2` 真分化后，才继续做 oracle 与 downstream usefulness 验证

### 方法调整方向

#### A. 配对方式

基于最新 full-37 screening，下一轮 bridge 主线固定为 `TopKPair`。`RandSectorPair` 只保留为唯一强备选，负责回答“如果不用 `TopKPair`，最强的结构对照是谁”；`MainBridge` 暂时淘汰，不再进入下一轮模块实验。后续所有模块分析都默认建立在 `TopKPair` bridge 上。

#### B. trust 使用方式

主版本已经不再让 trust 直接主导线上选择。当前更合理的口径是：

- `ParetoOnly` 保持为线上主版本
- `trust / boundary semantics` 只保留给审计与 `Full-v2` 实验分支
- 先证明 `Full-v2` 相比 `ParetoOnly` 真有选点分化，再讨论它是否值得回到主版本

#### C. 训练目标

训练、校准、测试三套缓冲严格隔离；训练时优先过采样 tight brackets；损失设计围绕“边界附近概率是否有语义”，而不是追求一般分类精度。

#### D. 主线收缩

主版本已经明确收缩为：

- `TopKPair`
- `midpoint placement`
- `ParetoOnly`
- `label-aware refinement`
- `minimal migration`

不要再回到 `Full-v1` 乘法主打分，也不要让 trusted bridge scan 回流到主线。

### 实验 A：桥接配对与激活的修复性预实验

目的：完成 bridge 规则筛选，并确定下一轮模块实验的统一基线。

建议设置：

- 问题：`DASCMOP5_BC`、`DASCMOP6_BC`、`LIRCMOP1_BC`、`MW1_BC`
- `Population = 100`
- `MaxFE = 50000`
- `Runs = 20`
- paired seeds
- 比较：`MainBridge / RandSectorPair / TopKPair`

建议指标：

- `SSR`：shared-sector 出现比例
- `ASR`：active-sector 出现比例
- `PMR`：shared pairs 中 `RawMargin > 0` 的比例
- `FSG`：首次产生 boundary seed 的代数
- `BSR`：是否成功产生 boundary seed

筛选决策写死为：

- `TopKPair` 进入下一轮
- `RandSectorPair` 保留为唯一强备选
- `MainBridge` 暂时淘汰

后续实验不再围绕 `CurrentPair vs TopKPair` 反复做 bridge 选择题，而是统一以 `TopKPair` 为 bridge 基线推进其余模块验证。

### 实验 B：校准 / trust 审计

目的：回答当前 `p ≈ 0.5` 是否至少已经有“可审计”的弱语义。

建议设置：

- 全 `37` 个 BC 问题
- `Population = 100`
- `MaxFE = 200000`
- `Runs = 30`
- 开发默认：`raw / beta`
- `temperature / auto`：只作为补充分支
- 固定 bridge = `TopKPair`
- 固定 placement = `midpoint`
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

目的：先证明 `Full-v2` 相对 `ParetoOnly` 真有分化，再验证它是否更靠近真实边界。

建议设置：

- 优先 `9` 个 `DASCMOP_BC`
- 若 `LIR/MW` 也能方便记录原始连续约束，再扩展到全 `37`
- `Population = 100`
- `MaxFE = 200000`
- `Runs = 30`
- 固定 bridge = `TopKPair`
- 固定 runtime 主线 = `ParetoOnly`
- 固定 placement = `midpoint`
- 比较：
  - `ParetoOnly`
  - `Full-v2`
  - `Uncertain-only`
  - `HighProb-Boundary`
  - `Rand-Boundary`

关键要求：

- 所有 query 变体必须共享相同 bridge generation、相同 boundary budget、相同主搜索轨迹
- 脚本先看 stop/go：`O_t` 和 `D_t`
- 只有 stop/go 通过时，才继续读取 `oracle_dB` 相关指标
- 当前脚本已直接导出 run-level `queryReport` 与 update-level `queryUpdateReport`

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
- bridge 仍固定 `TopKPair`

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
- bridge 仍固定 `TopKPair`

比较方式：

- 内部主比较：当前主版本（默认 `ParetoOnly`） vs `NoBoundary`
- 若 `Full-v2` 在 Step 2 真通过 stop/go，再把 `Full-v2` 并入最终内部比较
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

`TopKPair bridge 基线确定 -> ParetoOnly 主线稳定 -> Full-v2 若存在则先证明已分化 -> selected seeds 更接近真实边界 -> boundary discoveries 更有用 -> 最终 HV/IGD/AUC 更好`

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

`PRBCCMO` 当前已经证明“边界模块可以稳定启动”，并且主线代码已经统一到 `TopKPair + midpoint placement + ParetoOnly`；下一步不该盲目把 runs 从 5 加到 30，而应先看 `Full-v2` 能否通过 Step 2 的自动 stop/go 闸门，只有过闸后才继续 oracle / downstream / final performance 证据链。
