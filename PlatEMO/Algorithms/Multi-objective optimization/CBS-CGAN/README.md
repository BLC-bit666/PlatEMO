# CBS-CGAN 生产主线与生成/利用归因实验

生产算法类只有 `CBS_RegionWGAN_GP`。前半程使用全参考向量均衡生成、同条件 critic 筛选和局部尺度目标；后半程使用真实认证的可行边界端点，不叠加横向 donor 差分。`Experimental/CBS_RegionWGAN_GP_Experiment` 提供 A0/A1/A2 配置；`CBS_RegionWGAN_GP_A1/A2` 只是为通用 `test.m` 固定arm和分开保存结果的薄包装类，不覆盖机制。

完整开发裁决和实验数字见项目根目录 `Agent.md`。

## 算法结构

完整代、`N=100` 时的名义配比：

| 阶段 | 约束种群 Pop1 | 无约束种群 Pop2 |
|---|---|---|
| 代首 `FE < 0.5*maxFE` | 25 GA + 55 普通 DE + 20 CGAN 引导 DE | 25 GA + 75 普通 DE |
| 代首 `FE >= 0.5*maxFE` | 25 GA + 55 普通 DE + 20 boundary-target DE | 25 GA + 75 普通 DE |

- 无可用边界目标或同参考方向父代时，引导席位回退为普通 DE。
- 普通 DE 使用互异的 `base/r1/r2`，多项式变异前为 `base + 0.5*(r1-r2)`。
- 前半程每次训练在全部 `W` 上均衡分配 500 个条件，按同条件 critic 百分位保留 200 个。原始 critic 分数不跨条件比较。
- 每个候选在当前可行精英中确定父代 `A`，以全体可行解的稳健局部尺度 `h` 把 `G` 映射为 `T=A+min(1,h/d)(G-A)`；最终在 T-space 做空缺加权全局 maximin，并按本代实际 `guidedCount` 取解。
- 前半程调用 `OperatorDE(Problem,A,T,A,{1,1,1,20})`。`h` 只约束 Polynomial Mutation 前的中心，实际子代步长可以更大。
- 后半程认证目标仍调用 `OperatorDE(Problem,A,G,A,{1,F,1,20})`，`F` 循环 `0.4/0.65/0.85`；同参考方向缺父代时回退普通 DE。
- 每代最多使用 20 个真实 FE 做主动边界校准。校准解进入共享 `Union` 并被两套环境选择共同读取；未经评价的 CGAN 原始候选不会直接入种群。
- 两套环境选择共享 `Population1 + Population2 + Offspring1 + Offspring2 + Calibration`。Pop1 使用约束适应度，Pop2 使用无约束适应度；两者都采用 raw-objective SPEA2 近邻密度和字典序拥挤截断。
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
| `rawGuideCount` | 500 | 每次训练后在全部参考方向上的原始查询数 |
| `zDim` | 6 | 生成器噪声维数 |
| `ganIter` | 100 | 每次训练的生成器更新数 |
| `ganMiniBatch` | 32 | WGAN-GP mini-batch |
| `nCritic` | 4 | 每次生成器更新前的 critic 更新数 |
| `minGANTrainCount` | 32 | 最少条件训练行数 |
| `sampleSigma` | 0.3 | 生成采样噪声标准差 |

固定内部值包括：critic 保留 200、50% CGAN cutoff、20% 引导份额、每参考方向最多 5 个边界锚点、每代最多 20 个校准 FE。旧 20 槽 × 5 候选只由 A0 归因臂复现。

## 正式 runner

生产主线入口是 `Support/run_CBS_RegionWGAN_GP_mainline.m`。契约固定 `maxFE=200000`，允许 1 个验证 worker 或 10 个正式 worker，默认 runs 为 `1:10`。

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

生成/利用价值使用根目录唯一通用启动脚本 `test.m`。其 `algNames` 同时列出 A1/A2，运行 DASCMOP1--9_BC 与 LIRCMOP1--14_BC、runs `1:15`、10 个 process workers，共 690 个新增任务；A0 不在任务列表中。路径、Jobs清理、`platemo(...)`、`rng('shuffle')`、`save=1`、扁平 `parfor`、自动保存和断点跳过均沿用通用脚本原语义。

## 活动文件

- `CBS_RegionWGAN_GP.m`：唯一算法流程、双种群繁殖、CGAN/boundary-target 引导和最小运行诊断。
- `AssignReferenceVectors_CBS.m`：统一目标归一化与参考向量分配。
- `OperatorDEDistinct_CBS.m`：普通 DE 的互异父代选择。
- `UpdateBoundaryMemory_RC.m`、`BuildBoundaryDataset_RC.m`：边界记忆与 pairflag 数据。
- `BoundaryWGAN_RC.m`、`RunRegionGAN_RC.m`：WGAN-GP、条件 critic、全 W 均衡查询及旧 14+6 调度。
- `RefineBoundaryObservations_RC.m`：主动边界校准和认证端点。
- `CalFitness_CBS.m`、`EnvironmentalSelection_CBS.m`：双种群适应度与环境选择。
- `addCBSPaths.m`：移除当前仓库的 `Data/**` 路径后，只添加 Algorithms/Problems/Metrics，避免冻结历史类进入活动路径。
- `Support/run_CBS_RegionWGAN_GP_mainline.m`：生产主线并行 runner；根目录 `test.m`：唯一通用批量启动脚本。
- `Experimental/CBS_RegionWGAN_GP_Experiment.m`：三臂轻量配置包装器；`CBS_RegionWGAN_GP_A1/A2.m`：无机制覆盖的固定arm启动类。
- `Support/analyze_CBS_CGAN_factor_experiment.m`：顺序归因分析器；正式 `test.m` 数据按独立重复而非同run配对解释。

## 确定性验证

固定指纹仍使用 `LIRCMOP6_BC`、`N=100`、`D=30`、`maxFE=20000`、`rng(4242,'twister')` 和 MATLAB 默认计算线程。生成/利用改造会有意改变 RNG 轨迹和种群轨迹；最新数值由 `test_CBS_mainline_fingerprint.m` 锁定，不再沿用改造前指纹。

## 环境选择裁决

四类单因素改造均已被配对实验否决：task-specific survival pool 在 33 题全集未过 gate；归一化参考方向硬保护（NRBT）和仅归一化 SPEA2 截断在 15 个代表问题上都未增加方向覆盖，且后者 200K 的 `ALL G=1.02255`、`LIRCF G=1.04091`、`CF G=1.08653`，四个 CF 全部退化。

第四类 CA-anchored auxiliary truncation（CAAT）只在 Pop2 `Fitness<1` 超额时用更新后 Pop1 作为固定 anchor。15 题 × runs `106:110` 的 200K 结果为 `ALL G=0.995415, 4/5/6`、`LIRCF G=0.990102, 4/1/5`、`LIR G=1.026551`、`CF G=0.937841`；输出覆盖中位变化为 0、W/T/L 为 `0/12/3`，辅助新颖覆盖为 0、`0/15/0`，Pop1/joint PF-GD 的 ALL 比为 `1.097952/1.076822`。它减少了跨种群重复，但没有产生新方向，并且违反主效应、LIR 安全、覆盖和收敛 gates，`PROMOTE=0`。

因此唯一主线保持 `shared Union + constrained/unconstrained SPEA2 + raw-objective peer-only truncation`。task-specific 的初筛与 33 题补齐证据位于 `Data/CBS_env_selection_task_identity_runs101_105/`、`Data/CBS_env_selection_fullsets_runs101_105/`；其余证据位于 `Data/CBS_env_selection_nrbt_runs101_105/`、`Data/CBS_env_selection_normalized_truncation_runs101_105/`、`Data/CBS_env_selection_caat_runs106_110/`。不得通过追加方向配额、PBI、crowding、anchor 距离或问题专用分支挽救失败候选；后续多样性假设必须先产生 Union 中缺失的有效方向。

## 最终主线审计裁决

来源审计确认环境截断不是当前 IGD 瓶颈，稀疏 gap midpoint 才是现有最有效候选源；同时发现 Primitive 2 会把互为最近邻的两个方向重复评价为同一个决策中点。最后一次候选仅按无向 pair 去重，不加入归一化、保护、配额、反馈目标或问题分支。

`LIRCMOP6_BC/CF5_BC/CF6_BC × runs 101:105` 的 15/15 个任务均完成 200K FE。终点中位配对 IGD 比为 `0.961026390/0.817093875/1.317862006`，三题几何比为 `1.011484119`；终态方向覆盖中位变化为 `0/-1/-2`。候选虽然消除了重复 FE 并显著改善 CF5，却使 CF6 中位 IGD 恶化 `31.79%`，违反预注册 no-harm 与总体门槛，故 `PROMOTE=0`。活动代码已恢复实验前 Primitive 2；完整协议、源码快照、结果和分析位于 `Data/CBS_sparse_gap_unique_pairs_runs101_105/`。

## 回归测试

生产和实验入口测试覆盖：

- 主线结构与 distinct-parent DE
- 固定轨迹指纹
- boundary transfer 与对象复用
- pairflag 数据
- 旧 14+6、全 W 均衡 500 和同条件 critic 筛选
- 边界搜索
- 参考方向与边界记忆上限
- `CalFitness_CBS` 等价
- PlatEMO 接口
- 生产 runner 契约与三臂归因分析
- 选择来源审计的聚合计算
- 审计开关的行为中立集成验证

冻结的最终 BT0-vs-NoBT 结果、manifest、41 文件源码快照和诊断仍保存在 `Data/CBS_bt0_vs_nobt_gate_*`。
