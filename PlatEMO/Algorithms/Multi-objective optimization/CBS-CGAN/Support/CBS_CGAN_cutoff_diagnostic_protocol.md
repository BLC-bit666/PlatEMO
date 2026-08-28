# CBS-CGAN 精确截止诊断协议 v1

## 目标与截止定义

本协议重新评价前半程 CGAN 机制，不沿用 PlatEMO 稀疏保存点近似。精确截止种群定义为 `Population1` 在 `auditNotTerminated` 中首次满足

```text
FE >= ganStopFraction * maxFE
```

时的完整种群。它是算法实际从 CGAN 前半程转入后半程的环境选择结果；因每代评价数不恒定，`maxFE=200000` 时可为 `100197` 等值，不强制等于 `100000`。

截止时先计算并保存指标，再调用 PlatEMO 的 sealed `NotTerminated` 写入 `result`，最后用 `PlatEMO:Termination` 正常结束 cutoff-only 任务。正式 runner 固定 `maxRuntime=Inf`。

## 不使用生成样本可行性标签

新诊断对 raw/kept/selected G 不调用 `Problem.Evaluation`，不计算可行率，也不按可行性分类或过滤。`disableOracleAudit=true`，每个结果必须满足：

```text
rawOracleCount == 0
selectedTargetCount == 0
```

种群自身的可行解数量仍是优化结果指标，不属于生成样本 oracle 诊断。

## 指标

截止状态：

- 精确 `cganEndFE`、`cganEndIGD`、`cganEndHV`；
- 完整 `cganEndDecs/Objs/Cons`；
- 可行解数、非支配可行解数、参考向量覆盖数、归一化熵和占用 CV。

前半程轨迹：

- 最早可见种群及 5%、10%、20%、30%、40%、50% FE 检查点的 IGD/HV；
- `frontHalfIGDAUC`、`frontHalfHVAUC` 按实际 `FE/maxFE` 做相邻有限点梯形积分；不跨越 NaN 区间；
- `frontHalf*AUCCoverage` 是已积分区间占完整 `maxFE` 的比例，因此完整前半程约为 0.5，而不是 1。

边界配对云几何代理：

- 只使用当前 BMem 中 finite `x_b/x_i` 配对；
- 决策变量按 `(upper-lower)` 归一化；
- 对每个候选，只在相同 requested reference 内寻找最近配对中点 `(x_b+x_i)/2`；
- distance 为候选到该中点的欧氏距离；
- band hit 表示 distance 不大于同一配对的归一化半宽 `0.5*norm((x_b-x_i)/(upper-lower))`；
- 同方向无配对时 distance 为 NaN，另计 support rate，不回退到其他方向。

BMem 配对由当前搜索云在归一化目标空间中匹配得到，并非约束函数距离或二分认证边界。因此这些字段只解释为“接近当前 CGAN 训练配对云”，不能声称是真实约束边界距离。

候选链条报告 raw、critic-kept、rejected、最终 selected G、映射中心 T 和真实 child 的 pooled median/P90、band rate、support rate。`criticBoundarySpearman` 是实际 condition-wise critic percentile 与负距离的 Spearman 相关；高值表示 critic 排名倾向于更靠近同条件配对云。legacy 池没有 critic 排名，结果为 NaN。

多样性指标包括 requested-reference 覆盖率/归一化熵，以及归一化决策空间按 `1e-6` 量化的近重复率。requested-reference 指标不是 G 的真实目标方向，因为 G 未被评价。

## 正式任务矩阵

共同设置：`N=100`、`D=30`、`maxFE=200000`、固定 seeds `1:5`。

问题：`DASCMOP1_BC`、`DASCMOP5_BC`、`DASCMOP9_BC`、`LIRCMOP10_BC`、`LIRCMOP14_BC`。

Phase 1 共 175 个 cutoff-only 任务：A0、A1、Current、E0、Random20、DE20、GA20，各 5 问题 × 5 seeds。Current 与 A2/E8 默认搜索行为相同，A2/E8 只做别名一致性短测，不重复占用正式任务。

Phase 2 共 100 个 cutoff-only 任务：E1、K10、KAll、Cap10，各 5 问题 × 5 seeds，并复用 Phase 1 的 E0 作为对照。

## 随机性与统计

每个任务在构造 Problem 前执行 `rng(seed,'twister')`，直接调用 `Algorithm.Solve`；不调用会 `rng('shuffle')` 的 `platemo()`。诊断开关必须在相同 seed 下保持 FE、最终种群和终态 RNG 完全一致。

所有候选级数据先汇总成一个 run 级统计量，避免伪重复。跨运行时先删除 NaN/Inf，再报告算术均值与独立 `ranksum`。每题只有 5 次运行，p 值只作为低功效辅助证据；同时保留效应方向和候选/对照比值。IGD、IGD AUC、距离和重复率越低越好；HV、HV AUC、band hit、方向覆盖/熵和 critic-boundary 相关越高越好。

## 执行

```matlab
run_CBS_CGAN_cutoff_diagnostics(rootPath,10,"phase1")
run_CBS_CGAN_cutoff_diagnostics(rootPath,10,"phase2")
```

结果写入独立目录：

```text
Data/CBS_CGAN_cutoff_diagnostics_runs1_5/
```

runner 使用 10 个 process workers、逐任务原子落盘、schema 校验、三次重试和源码快照。已存在但 schema/身份/截止种群不一致的文件不会被视为完成。源码在 campaign 开始后发生变化时拒绝混跑，必须启用新的 campaign 目录。
