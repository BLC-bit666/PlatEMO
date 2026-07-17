# CBS RegionWGAN-GP 固定主线

该目录只保留 `CBS_RegionWGAN_GP` 这一条可执行算法主线。算法用参考向量
条件化的 WGAN-GP（本文档沿用 CGAN 简称）学习边界邻近可行锚点的决策
分布，生成完整决策向量，并把生成解纳入 PlatEMO 的真实评价和双群体环境
选择。

正式实验只比较每个 run 的最终 `IGD`。loss、边界距离、coverage、多样性、
可行率、survival、阶段快照和 query 归因均不计算、不保存。`status`、
`finalFE`、runtime、wall time、源码哈希和任务签名只用于完整性、性能及断点
恢复审计，不是实验指标。

## 固定结论：CGAN 在哪里停止

主线固定 `ganStopFraction = 0.5`。每一代完成 DE 评价后，仅当
`Problem.FE < 0.5 * Problem.maxFE` 时才允许启动新的 CGAN 训练与采样事件。

- 正式默认 `maxFE = 200000`，因此停止阈值是 **FE = 100000**。
- 当检查点已经达到或超过 100000 FE 时，不再启动 CGAN，后半程只运行 DE
  和双群体环境选择。
- 为保持已经验证的 T50 实验语义，一个在阈值前启动的生成批次可能使 FE
  越过阈值；下一代不会再启动 CGAN。代码不会为了卡齐 100000 而截断该批次。

## 主线配置

```text
full warm-start
Batch = 32
G updates = 100 / eligible event
C:G updates = 4:1
G/C hidden = 32x32 / 32x32
zDim = 6
sampleSigma = 0.3
nGen = 30
CGAN active fraction = 0.5
```

PlatEMO 公开参数仍按以下顺序提供，便于常规 GUI/API 调用；正式 runner 不传
参数，因此始终使用默认主线：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `nGen` | 30 | 每个合格事件生成的解数 |
| `zDim` | 6 | 生成器噪声维数 |
| `ganIter` | 100 | 每次训练的生成器更新数 |
| `ganMiniBatch` | 32 | WGAN-GP mini-batch |
| `nCritic` | 4 | 每次生成器更新对应的 critic 更新数 |
| `minGANTrainCount` | 32 | 触发训练所需的最少锚点行数 |
| `sampleSigma` | 0.3 | 生成采样噪声标准差 |

`ganStopFraction`、学习率、gradient penalty、锚点记忆规则和网络结构是固定
主线内部常量，不是实验开关。

## PlatEMO 平台规范

算法类直接继承 `ALGORITHM`，参数通过 `Algorithm.ParameterSet` 读取，种群
通过 `Problem.Initialization` 创建，终止由 `Algorithm.NotTerminated` 管理，
生成解统一通过 `Problem.Evaluation` 计入 FE。通用能力复用平台
`Algorithms/Utility functions` 中的：

- `UniformPoint`：参考向量生成；
- `TournamentSelection`：父代选择；
- `OperatorDE`：差分进化、变异、边界处理与真实评价。

`generateRegionDEOffspring` 只负责最后一代的剩余 FE 分配，不实现新的 DE
算子。由于仓库内 `SSIO-RL` 另有同名 `TournamentSelection.m`，算法入口会把
官方 Utility functions 目录置于路径首位；并列 fitness 先转换为确定性 rank，
从而在修正路径遮蔽的同时保持既有随机选择轨迹。

`CalFitness_CBS`、`EnvironmentalSelection_CBS`、边界记忆和条件 WGAN
属于本算法定义：SPEA2 strength、带容差的边界 Pareto 层和条件生成训练无法
分别由 `NDSort`、`CrowdingDistance` 或其他通用算子保持同一语义。距离计算
保留矩阵实现，以免额外依赖 Statistics and Machine Learning Toolbox，并
保持已经验证的浮点轨迹。

## 算法流程

每一代依次执行：

```text
P1/P2 两个群体
  -> 两组 DE 后代
  -> 在前 50% FE 内更新边界记忆
  -> 构造 TrainX / TrainC / QueryRefs
  -> 合格时 full warm-start 训练 WGAN-GP
  -> one-sixth frontier query 生成完整决策向量并真实评价
  -> 两次环境选择
```

边界记忆只保留 `ref/gap/x_b/y_b`。不可行解仅在本次更新中用于合法伙伴
筛选和 gap 过滤，不作为持久观测字段。训练集固定为 `TrainX = BMem.x_b`；
重复锚点行有意保留为训练权重。有效训练行少于 32 时，本事件不训练、不采样。

query 预算中 `round(nGen/6)` 分配给一跳 frontier，其余分配给 populated
references；没有 frontier 时全部回退到 populated references。

## 唯一正式运行入口

```matlab
root = fileparts(which('platemo'));
outDir = fullfile(root,'Data','CBS_RegionGAN_compare','mainline_igd_runs_v2');
[Summary,outDir] = run_CBS_RegionWGAN_GP_mainline( ...
    outDir,9, ...
    "LIRCMOP" + string((5:10)') + "_BC", ...
    100,30,200000,1:3,struct('resume',true));
```

若省略 `maxFE`，runner 也默认使用 200000。正式并行固定为 9 workers；测试
允许 1 worker。每个任务只保存最终 IGD 和审计字段：

- `run_summary.csv`
- `mainline_config.json`
- `provenance.csv`
- `source_manifest.csv`
- `<problem>_run<seed>/attempt_*/task_result.mat`

schema 为 `cbs_region_wgan_igd_mainline_v2`。旧结果仍是历史证据，但不得与
当前 source hash 或新输出目录混用。

## 主线文件

| 文件 | 职责 |
|---|---|
| `CBS_RegionWGAN_GP.m` | 算法入口、固定 T50 阈值和双群体主循环 |
| `UpdateBoundaryMemory_RC.m` | 边界锚点记忆更新 |
| `BuildBoundaryDataset_RC.m` | 构造训练数据和 query references |
| `RunRegionGAN_RC.m` | one-sixth query 与 WGAN 调用 |
| `BoundaryWGAN_RC.m` | conditional WGAN-GP 训练和采样 |
| `CalFitness_CBS.m` | SPEA2-style fitness |
| `EnvironmentalSelection_CBS.m` | 双群体环境选择 |
| `Support/run_CBS_RegionWGAN_GP_mainline.m` | 最终 IGD runner |
| `Support/CBS_RegionGAN_Provenance.m` | 可复现实验来源记录 |

## 行为中性性能优化

保留的优化不改变网络、loss、Adam、训练次数、随机数顺序、采样或选择语义：

- 在梯度图外生成并 detach critic 所需的 fake；
- 无需求导的张量保持 numeric single，只在 GP 内层启用高阶导数；
- 热循环外提固定标量，每个训练事件只转换一次训练数据；
- 边界记忆预分配，缓存固定参考向量邻域，向量化 Pareto 首前沿判定；
- fitness 密度只选取所需的第 k 近邻距离，避免全矩阵开方和完整排序；
- 删除无用观察字段、研究 observer、多指标/绘图/消融 runner。

固定 seed 的 20k FE 回归中，清理优化前后最终决策、目标、约束、IGD 和 RNG
状态均逐元素一致；IGD 均为 `1.2500475807204314`。WGAN 训练仍占主要耗时。
曾测试的 `dlaccelerate` 虽明显提速，但多问题配对 IGD 退化，故未保留。

本轮 PlatEMO 规范化另使用两个 `600 FE` 快速样本做改前/改后对照，最终
`Dec/Obj/Con/RNG/IGD` 均逐元素一致；对应 IGD 为
`2.6964859435480961` 和 `1.7594293949177635`。这属于快速回归，不是新的
正式实验。

同一单线程条件下，本轮外围优化前 wall time 为 `27.4898 s`；最终代码三次
独立进程计时为 `27.0072 / 27.0690 / 27.0865 s`，中位数 `27.0690 s`，
相对单次同轮对照约缩短 1.5%。profiler 复核中 WGAN 训练占
`37.2428 / 42.3611 s = 87.9%`，critic 更新占 `30.8895 s`；因此在不改变
算法与数值轨迹的约束下，剩余可提速空间有限。

## 回归测试

```matlab
test_CBS_platemo_compliance
test_CBS_region_one_sixth_query
test_CBS_region_boundary_ref_cap
test_CBS_region_wgan_mainline
test_CBS_calfitness_equivalence
test_CBS_RegionGAN_provenance
test_CBS_RegionWGAN_GP_mainline_runner
```

历史调参和被拒绝方案的结果继续保存在 `Data/CBS_RegionGAN_compare`，仅作证据，
不是当前运行依赖。
