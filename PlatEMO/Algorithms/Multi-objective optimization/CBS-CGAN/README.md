# CBS-CGAN 唯一生产主线

本目录只保留一个算法类：`CBS_RegionWGAN_GP`。历史 BT0/F0 已成为主线本身：前半程使用 CGAN 目标，后半程使用真实认证的可行边界端点；不叠加横向 donor 差分。所有消融、算子筛选、兼容别名和 BT 公式实验类已从活动源码删除，冻结证据仍位于 `Data/`。

完整开发裁决和实验数字见项目根目录 `Agent.md`。

## 算法结构

完整代、`N=100` 时的名义配比：

| 阶段 | 约束种群 Pop1 | 无约束种群 Pop2 |
|---|---|---|
| 代首 `FE < 0.5*maxFE` | 25 GA + 55 普通 DE + 20 CGAN 引导 DE | 25 GA + 75 普通 DE |
| 代首 `FE >= 0.5*maxFE` | 25 GA + 55 普通 DE + 20 boundary-target DE | 25 GA + 75 普通 DE |

- 无可用边界目标或同参考方向父代时，引导席位回退为普通 DE。
- 普通 DE 使用互异的 `base/r1/r2`，多项式变异前为 `base + 0.5*(r1-r2)`。
- 引导 DE 调用 `OperatorDE(Problem,A,G,A,{1,F,1,20})`，即变异前为 `A + F*(G-A)`；`F` 循环 `0.4/0.65/0.85`。
- 每代最多使用 20 个真实 FE 做主动边界校准。所有校准解进入公共环境选择；未经评价的 CGAN 原始候选不会直接入种群。
- 前半程边界记忆 `BMem` 服务 pairflag WGAN；后半程 `BoundaryTargetXf/Yf` 服务认证目标引导，两者用途不同。
- 后半程存档 newest-first、FIFO 上限 1500；目标与父代必须在联合归一化后属于同一参考方向。

## 运行

从仓库根目录直接运行：

```matlab
repoRoot = pwd;
addpath(fullfile(repoRoot,'Algorithms','Multi-objective optimization','CBS-CGAN'));
addCBSPaths(repoRoot);
rng(1,'twister');
Problem = LIRCMOP5_BC('N',100,'D',30,'maxFE',200000);
Algorithm = CBS_RegionWGAN_GP('save',2,'metName',{'IGD'});
Algorithm.Solve(Problem);
```

`platemo(...)` 除了执行 `rng('shuffle')`，还会递归加入仓库目录，使 `Data/` 中冻结的历史 `.m` 快照重新出现在 MATLAB 路径上。因此主线运行统一使用上面的 `addCBSPaths + Algorithm.Solve`，批量实验统一使用下述正式 runner。

## 参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `nGen` | 20 | 每次训练后的参考方向查询槽数 |
| `zDim` | 6 | 生成器噪声维数 |
| `ganIter` | 100 | 每次训练的生成器更新数 |
| `ganMiniBatch` | 32 | WGAN-GP mini-batch |
| `nCritic` | 4 | 每次生成器更新前的 critic 更新数 |
| `minGANTrainCount` | 32 | 最少条件训练行数 |
| `sampleSigma` | 0.3 | 生成采样噪声标准差 |

固定内部值包括：50% CGAN cutoff、20% 引导份额、每槽 5 个候选、每参考方向最多 5 个边界锚点、每代最多 20 个校准 FE。

## 正式 runner

唯一正式入口是 `Support/run_CBS_RegionWGAN_GP_mainline.m`。契约固定 `maxFE=200000`，允许 1 个验证 worker 或 10 个正式 worker，默认 runs 为 `1:10`。

```matlab
campaignRoot = '/absolute/path/to/new_campaign';
outDir = fullfile(campaignRoot,'Data','CBS_RegionWGAN_GP');
problems = ["LIRCMOP5_BC";"LIRCMOP7_BC";"CF5_BC";"CF6_BC"];

[Summary,outDir] = run_CBS_RegionWGAN_GP_mainline( ...
    outDir,10,problems,100,30,200000,1:10, ...
    struct('resume',false));
```

摘要报告保存点附近的真实 `FE100K/IGD100K` 和终点 `FE200K/IGD200K`。

`resume=true` 只校验 MAT 结构、种群维度和 FE，不校验源码哈希。代码变化后必须使用新的 campaign root，不能静默复用旧 `Data/CBS_RegionWGAN_GP`。

## 活动文件

- `CBS_RegionWGAN_GP.m`：唯一算法流程、双种群繁殖、CGAN/boundary-target 引导和最小运行诊断。
- `AssignReferenceVectors_CBS.m`：统一目标归一化与参考向量分配。
- `OperatorDEDistinct_CBS.m`：普通 DE 的互异父代选择。
- `UpdateBoundaryMemory_RC.m`、`BuildBoundaryDataset_RC.m`：边界记忆与 pairflag 数据。
- `BoundaryWGAN_RC.m`、`RunRegionGAN_RC.m`：WGAN-GP 与 14+6 查询调度。
- `RefineBoundaryObservations_RC.m`：主动边界校准和认证端点。
- `CalFitness_CBS.m`、`EnvironmentalSelection_CBS.m`：双种群适应度与环境选择。
- `addCBSPaths.m`：移除当前仓库的 `Data/**` 路径后，只添加 Algorithms/Problems/Metrics，避免冻结历史类进入活动路径。
- `Support/run_CBS_RegionWGAN_GP_mainline.m`：正式并行 runner。

## 性能与等价原则

只接受不改变 FE 顺序、RNG 调用数、三组引导 `OperatorDE` 调用顺序和种群轨迹的优化。

清理后的主类已移除 cutoff 全种群快照、实验诊断数组、override 分派和不可达分支，并预分配引导选择输出。WGAN 训练仍占绝大多数运行时间；网络、精度、batch、训练次数和随机调用形状均未改变。

固定指纹为 `LIRCMOP6_BC`、`N=100`、`D=30`、`maxFE=20000`、`rng(4242,'twister')`：

- IGD：`1.3468921272381897`
- 决策和：`1729.8887087629678`
- 目标和：`1056.8965815545916`
- 约束和：`0`
- RNG state 首值：`281257120`
- boundary rows：`222`；晚期目标使用：`904/904`

## 回归测试

保留 10 个生产测试：

- 主线结构与 distinct-parent DE
- 固定轨迹指纹
- boundary transfer 与对象复用
- pairflag 数据
- 14+6 查询分配
- 边界搜索
- 参考方向与边界记忆上限
- `CalFitness_CBS` 等价
- PlatEMO 接口
- 正式 runner 契约

实验专用测试已经随实验类删除。冻结的最终 BT0-vs-NoBT 结果、manifest、41 文件源码快照和诊断仍保存在 `Data/CBS_bt0_vs_nobt_gate_*`。
