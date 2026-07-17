# CBS RegionWGAN-GP 固定主线约定

更新时间：2026-07-17

## 1. 当前唯一结论

- 唯一算法：`CBS_RegionWGAN_GP`。
- 唯一正式入口：`Support/run_CBS_RegionWGAN_GP_mainline.m`。
- 正式实验只比较每个 run 的最终 `IGD`。
- 正式默认 `maxFE=200000`；CGAN 固定在前 50% FE 活跃，停止阈值为
  **FE=100000**。
- 检查条件是 `Problem.FE < 0.5*Problem.maxFE`。阈值前已经启动的完整生成
  批次不截断；达到阈值后不再启动新 CGAN 事件。
- 历史 `Data/CBS_RegionGAN_compare` 只作证据，不是源码依赖，不得删除、覆盖
  或与当前 source hash 混用。

本地 Git 只有 `UC-GAN-2` 分支。远端分支不处理；未明确要求时不提交、不推送、
不创建 PR。

## 2. 固定算法机制

```text
双群体 P1/P2
  -> 两组 DE 后代
  -> 前 50% FE 内更新 BMem
  -> TrainX / TrainC / QueryRefs
  -> 合格事件 full warm-start 训练 WGAN-GP
  -> one-sixth frontier query，真实 Evaluation
  -> 两次环境选择
```

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

正式范围：

```text
LIRCMOP5_BC–LIRCMOP10_BC
N=100, D=30, M=2, maxFE=200000
runs/seeds=1:3
workers=9
```

```matlab
root = fileparts(which('platemo'));
outDir = fullfile(root,'Data','CBS_RegionGAN_compare','mainline_igd_runs_v2');
[Summary,outDir] = run_CBS_RegionWGAN_GP_mainline( ...
    outDir,9,"LIRCMOP" + string((5:10)') + "_BC", ...
    100,30,200000,1:3,struct('resume',true));
```

- 正式并行使用 9 workers；测试允许 1 worker。
- 每个 worker 限制 OpenMP/BLAS 为 1 线程。
- 每个任务必须 `status=ok`、`finalFE=maxFE` 且最终 IGD 有限。
- 每个任务只保存最终 IGD 与审计字段，不保存最终种群或过程观测。
- schema 为 `cbs_region_wgan_igd_mainline_v2`。

## 5. 唯一源码集合

- `Algorithms/Multi-objective optimization/CBS-CGAN/CBS_RegionWGAN_GP.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/UpdateBoundaryMemory_RC.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/BuildBoundaryDataset_RC.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/RunRegionGAN_RC.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/BoundaryWGAN_RC.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/CalFitness_CBS.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/EnvironmentalSelection_CBS.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/Support/run_CBS_RegionWGAN_GP_mainline.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/Support/CBS_RegionGAN_Provenance.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/Support/CBS_RegionGAN_SourceManifestSHA256.m`

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
```
