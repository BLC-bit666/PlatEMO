# CBS RegionWGAN-GP 固定主线约定

更新时间：2026-07-19

## 1. 当前唯一结论

- 唯一算法：`CBS_RegionWGAN_GP`。
- **主线配置（2026-07-19 定型并已固化为类默认）：
  `operatorMode = ga_de_half` + `boundarySearch = on`（S2+BLS）。**
  默认构造即主线；旧 A1 路径仍可经显式开关
  `('operatorMode','de','boundarySearch','off')` 逐位复现。
  旧 A1 全量战役原生结果归档于 `Data/CBS_RegionWGAN_GP-7-18`，
  规范输出目录 `Data/CBS_RegionWGAN_GP` 已腾空供新战役使用。
- 唯一正式入口：仓库根目录 `run_CBS_RegionWGAN_GP_DAS_LIR_full.m`，内部调用
  `Support/run_CBS_RegionWGAN_GP_mainline.m` 产出 PlatEMO 原生结果文件
  （`Data/CBS_RegionWGAN_GP/*.mat`，顶层变量仅 `result`/`metric`）。
- 正式实验只比较每个 run 的最终 `IGD`。
- 正式默认 `maxFE=200000`；CGAN 固定在前 50% FE 活跃，停止阈值为
  **FE=100000**。
- 检查条件是 `Problem.FE < 0.5*Problem.maxFE`。阈值前已经启动的完整生成
  批次不截断；达到阈值后不再启动新 CGAN 事件。
- 基准为 `_BC` 单比特约束套件：每个解只返回一个 0/1 可行标量
  （`any(PopCon>0)` 折叠），不可行解之间没有违反度梯度。该设定与
  NA-EMT 的 CMOP-UC 一致，比 DRMCMO 论文的逐约束 0/1 计数更严格。
- **LIRCMOP1_BC–LIRCMOP4_BC 已明确排除出一切分析与结论**
  （2026-07-18 决策），正式战役可跑但不进对比表。
- 历史 `Data/CBS_RegionGAN_compare` 只作证据，不是源码依赖，不得删除、覆盖
  或与当前 source hash 混用。

本地 Git 只有 `UC-GAN-2` 分支。远端分支不处理；未明确要求时不提交、不推送、
不创建 PR。

## 2. 固定算法机制

```text
双群体 P1/P2
  -> 每群体一半 SBX+PM(OperatorGAhalf) + 一半原 DE 后代   [S2]
  -> 前 50% FE 内更新 BMem
  -> TrainX / TrainC / QueryRefs
  -> 合格事件 full warm-start 训练 WGAN-GP
  -> one-sixth frontier query，真实 Evaluation
  -> 后 50% FE 每代 ≤20 FE 边界线搜索(二分+前沿中点插值)  [BLS]
  -> 两次环境选择(Union 含 GAN 与 BLS 后代)
```

每代 FE 预算（N=100 稳态）：前半 200(DE/GA)+30(CGAN)=230；
后半 200(DE/GA)+≤20(BLS)=220；全程受 remainingFE 钳制。

固定默认值：

| 配置 | 值 |
|---|---:|
| G updates / eligible event | 100 |
| C updates / eligible event | 400 |
| `ganMiniBatch` | 32 |
| `nCritic` | 4 |
| G/C hidden | `[32 32]` / `[32 32]` |
| `zDim` | 6 |
| `ganLrD/ganLrG` | 1e-4 / 1e-4 |
| `gpLambda` | 10 |
| `sampleSigma` | 0.3 |
| `nGen` | 30 |
| `frontDepth` | 2 |
| `pairNeighborRefRadius` | 2 |
| `maxAnchorsPerRef` | 5 |
| `minGANTrainCount` | 32 |
| `ganStopFraction` | 0.5 |

`nGen`、`zDim`、`ganIter`、`ganMiniBatch`、`nCritic`、
`minGANTrainCount` 和 `sampleSigma` 遵循 PlatEMO `ParameterSet` 约定；正式
runner 不传自定义值。`ganStopFraction` 不是公开实验开关。

边界记忆只持久保存 `ref/gap/x_b/y_b`。不可行伙伴只参与当次合法配对与 gap
过滤；训练固定使用 `TrainX=BMem.x_b`。重复锚点行作为训练权重保留。少于
32 个有效训练行时不训练、不采样、不复用旧模型。

query 将 `round(nGen/6)` 分配给一跳 frontier，其余分配给 populated
references；没有 frontier 时全部回退到 populated references。

不得恢复 schedule、batch/epoch/gap sweep、phased iteration、去重、
structured-z、pointwise、MMD、coherent BMem、observer、阶段快照、绘图、
机制消融或多指标记录分支。

## 3. PlatEMO 实现边界

- 类必须继承 `ALGORITHM`，公开参数必须同时存在于文件头 `---` 注释和
  `Algorithm.ParameterSet`，且顺序与默认值一致。
- 初始化、终止和真实评价使用 `Problem.Initialization`、
  `Algorithm.NotTerminated`、`Problem.Evaluation`。
- 参考向量、父代选择和 DE 分别使用 Utility functions 的 `UniformPoint`、
  `TournamentSelection`、`OperatorDE`，不得在本目录复制这些算子。
- 算法入口必须确保 Utility functions 的 `TournamentSelection` 优先于
  `SSIO-RL` 的同名局部文件；确定性 rank 适配用于保持既有并列选择轨迹。
- `generateRegionDEOffspring` 只做剩余 FE 分配；实际变异与评价由
  `OperatorDE` 完成。
- `CalFitness_CBS`/`EnvironmentalSelection_CBS` 是 CCMO/SPEA2 strength
  survival；`UpdateBoundaryMemory_RC` 使用 1e-12 容差 Pareto 层。这些是
  算法语义，不能用 `NDSort` 或 `CrowdingDistance` 静默替换。
- 不使用 `pdist2`，避免新增 Statistics and Machine Learning Toolbox 依赖
  及浮点轨迹变化。

## 4. 最终 IGD runner

正式范围（全量战役，仅在最终配置定型后运行一次）：

```text
DASCMOP1_BC–DASCMOP9_BC + LIRCMOP1_BC–LIRCMOP14_BC（共 23 题）
N=100, D=各问题类默认, maxFE=200000
runs/seeds=1:10
workers=10（测试允许 1）
输出 Data/CBS_RegionWGAN_GP 原生文件，可断点复用
```

```matlab
run_CBS_RegionWGAN_GP_DAS_LIR_full
```

- 每个 worker 限制 OpenMP/BLAS 为 1 线程。
- 每个任务必须 `status=ok`、`finalFE=maxFE`；IGD 为 NaN 是合法数据
  （无可行解），不是失败。
- 分析时 L1–4 剔除；工程实验（第 8 节）使用 10 题代表集。

## 5. 唯一源码集合

- `Algorithms/Multi-objective optimization/CBS-CGAN/CBS_RegionWGAN_GP.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/UpdateBoundaryMemory_RC.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/BuildBoundaryDataset_RC.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/RunRegionGAN_RC.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/BoundaryWGAN_RC.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/CalFitness_CBS.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/EnvironmentalSelection_CBS.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/Support/run_CBS_RegionWGAN_GP_mainline.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/Support/run_CBS_operator_triage.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/Support/CBS_RegionGAN_Provenance.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/Support/CBS_RegionGAN_SourceManifestSHA256.m`
- `run_CBS_RegionWGAN_GP_DAS_LIR_full.m`（仓库根目录）

## 6. 性能与等价性纪律

允许的优化必须保持网络、loss、Adam、训练次数、采样、选择和 RNG 顺序不变。
固定 `LIRCMOP5_BC, N=100, D=30, maxFE=20000, seed=7001` 的最新回归中，
清理优化前后最终 dec/obj/con/IGD/RNG 状态逐元素一致，IGD 为
`1.2500475807204314`。

当前保留：梯度图外 detach fake、single 数值路径、最小高阶梯度范围、热循环
常量外提、边界记忆预分配和固定邻域缓存、Pareto 判定向量化、fitness 第 k
近邻部分选择。WGAN 训练是主要耗时，进一步改变深度学习数值内核的方案不能
仅凭速度保留。

最终无 profiler 的 20k FE 三次独立 wall time 为
`27.0072 / 27.0690 / 27.0865 s`，中位数 `27.0690 s`；本轮外围优化前同条件
单次对照为 `27.4898 s`。profiler 显示 WGAN 训练占总 wall time 约 87.9%，
critic 更新单独占 30.8895 s。

`dlaccelerate` 曾显著提速，但 18 个 100k FE 配对中 IGD 为 5 胜、4 平、
9 负，因此淘汰。任何改变浮点轨迹的新候选都必须经过多问题、多 seed 的正式
IGD 非退化验证。

## 7. 验证命令

```matlab
test_CBS_platemo_compliance
test_CBS_region_one_sixth_query
test_CBS_region_boundary_ref_cap
test_CBS_region_wgan_mainline
test_CBS_calfitness_equivalence
test_CBS_RegionGAN_provenance
test_CBS_RegionWGAN_GP_mainline_runner
test_CBS_operator_modes
test_CBS_boundary_search
```

双指纹（由 `test_CBS_operator_modes` 守护，条件均为
`LIRCMOP6_BC, N=100, D=30, maxFE=20000, rng(4242,'twister')`）：

- 主线默认路径（S2+BLS）：IGD=`1.3470807122527642`、
  决策和=`1919.0968011138757`、终态 RNG 首字=`368524342`；
- 遗留路径（显式 `de`/`off`）：IGD=`1.3653447524657205`、
  决策和=`1729.4377215220884`、终态 RNG 首字=`415301093`。

任何触碰主循环的改动必须同时逐位复现两条指纹。

## 8. 工程实验状态（不进论文创新点，2026-07-19）

背景诊断：单比特 CV 下全不可行阶段约束适应度退化 + 共享 Union 使
P1≡P2、原 DE（CR=1、精英差分）步长坍缩，导致 D4–8 双稳态崩溃。

- `operatorMode` 构造器开关：`de`、`imtcmo_de`（S1）、`ga_de_half`（S2）。
  `boundarySearch` 构造器开关：`off`/`on`（BLS：50% 停止点后每代 ≤20 FE，
  可行–不可行二分 + 前沿最稀疏对中点插值；出处 Michalewicz 边界算子 /
  GECCO2007 binary interpolation repair）。默认路径与开关前逐位一致，
  由 20k 指纹守护。
- triage 证据（10 题代表集 × 3 seeds，`operator_triage_v1`）：
  - S1/S2 vs A1：失败域 D4–8 崩溃 run 4→0（两臂同效）；
    S2 全局 23/5 胜、守护域非劣；S1 劣化 L6/L10 达 30–76% 淘汰。
  - S2B(S2+BLS) vs S2：L10 三 seed 全胜（−16~−26%，反超本地 MCCMO 与
    DRMCMO 发表值）、D9 三 seed 全胜；失败域中性（崩溃 0/0）；
    唯一代价 L6 +2~6%（绝对量 ≤5e-4）；30 对净 IGD 和 ≈ −0.010。
- **裁决（2026-07-19，用户确认）：主线 = S2+BLS。**
- 已否决：S1；S3 停滞重初始化；IMTCMO Neighbor_Pairing 移植。
- 已知未偿债务：CGAN 相对"无 GAN/平凡采样器"的必要性消融从未运行
  （用户明确搁置）。前半段每代 30 FE 的 CGAN 开销之"值得"目前只有
  间接证据（时机研究、机制指标），无因果对照。
- 已完成（2026-07-19）：S2+BLS 固化为类默认，双指纹录制并入测试，
  启动脚本含主线漂移守卫（`CBSRegionGAN:MainlineDefaultsDrift`）。
- 待执行：唯一一次全量战役（23 题 × 10 seeds）出论文表；分析时
  L1–4 剔除。
