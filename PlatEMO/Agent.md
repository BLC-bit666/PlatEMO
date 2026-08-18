# CBS-CGAN 唯一主线开发记录

更新时间：2026-08-18

适用范围：`/Users/lanai/Code/Matlab/PlatEMO/PlatEMO` 当前未暂存工作树。本文件是算法身份、实验裁决、运行协议和后续约束的权威入口；旧类名、旧结果目录或旧文档不能覆盖这里的结论。

## 1. 唯一结论

- 唯一生产算法类是 `CBS_RegionWGAN_GP`。
- 历史实验标签中的 **BT0/F0 已晋级为主线真实语义**：后半程保留认证 boundary-target 引导，不叠加横向 donor 差分；普通 DE 仍走全局 distinct-parent 的 `current/1-like` 路径；引导步长仍循环 `0.4/0.65/0.85`。
- BT0/F0 不再由别名类或 override hook 表达。所有实验算法类、实验专用 helper、runner 和测试已从活动源码清除。
- 最终 BT0-vs-NoBT 预注册实验的 5 个 gate 全部通过，因此保留 boundary transfer，否决“后半程完全取消边界目标”的 NoBT 对照。
- 该裁决只确立项目内部唯一主线，不等于已经证明优于 DRMCMO。现有外部对照证据仍不足以宣称在 LIRCMOP 与 CF 上整体超过 DRMCMO。

## 2. 主线的精确算法语义

### 2.1 双种群、算子配比和 FE

以下为 `N=100` 完整进化批次的名义数量，不含校准评价：

| 代首阶段 | 约束种群 Pop1 | 无约束种群 Pop2 |
|---|---|---|
| `FE < 0.5*maxFE` | 25 GA + 55 普通 DE + 20 CGAN 引导 DE | 25 GA + 75 普通 DE |
| `FE >= 0.5*maxFE` | 25 GA + 55 普通 DE + 20 认证边界目标 DE | 25 GA + 75 普通 DE |

- 后半程若没有认证边界目标，或没有同参考方向的可行父代，引导席位回退为普通 DE；当边界存档为空时，Pop1 直接按 25 GA + 75 普通 DE 分配。
- 每个完整迭代另有最多 20 次真实边界校准评价。`maxFE` 同时约束两套子代和校准评价。
- 阶段由一代开始时的 `generationFE` 判断；末代会受剩余 FE、舍入和回退影响。
- Pop2 全程不读取 CGAN、参考分区或边界目标存档。
- 两套环境选择共享同一候选全集：`Population1 + Population2 + Offspring1 + Offspring2 + Calibration`。Pop1 使用约束适应度，Pop2 使用无约束适应度；两套选择都保留现有 raw-objective SPEA2 近邻密度和字典序拥挤截断。task-specific 父代池、参考方向保护、目标 min-max 截断和更新后 Pop1 anchor 辅助截断均已被配对实验否决。

### 2.2 普通 DE

`OperatorDEDistinct_CBS.m` 选出互异的 `base/r1/r2`，再调用平台 `OperatorDE`。多项式变异前：

```text
x_child = x_base + 0.5 * (x_r1 - x_r2)
```

- 请求数等于种群规模时，每一行恰好作为 base 一次；否则 base 无放回抽取。
- `r1`、`r2` 分别由二元适应度锦标赛选择，并强制三者互异。
- 默认 `CR=1, F=0.5, proM=1, disM=20`，随后执行边界截断和多项式变异。
- 该路径不是 `rand/1`、`best/1` 或 `current-to-pbest/1`。

### 2.3 CGAN 与 boundary-target 的公共引导公式

引导子代统一调用：

```matlab
OperatorDE(Problem,A,G,A,{1,F,1,20})
```

多项式变异前为 `A + F*(G-A)`：

- `A` 是可行父代，`F` 按子代循环 `0.4, 0.65, 0.85`。
- 前半程 `G` 是尚未评价的 CGAN 输出；网络只生成目标点，最终子代才接受真实评价。
- 后半程 `G` 是二分校准后经真实评价认证的可行端点 `xf`。
- 主线不再存在 BT1/BT2/BT3 的横向 `0.5*(R1-R2)` 分支，也不存在相关步长 cap 或诊断容器。

### 2.4 两套边界状态

| 状态 | 生命周期与用途 | 核心规则 |
|---|---|---|
| `BMem` | 仅前半程更新；构造 pairflag 数据并训练/查询 CGAN | 每参考方向最多 5 个前两层可行锚点；不可行配对每代重建；使用邻近参考方向与 MAD gap 过滤 |
| `BoundaryTargetXf/Yf` | 校准全程产生；仅后半程 Pop1 消费 | 只存真实认证的 `xf/yf`；FIFO 上限 1500；newest-first；目标与父代必须在联合归一化后属于同一参考方向 |

`AssignReferenceVectors_CBS.m` 统一采用逐目标 min-max 归一化和最大余弦相似度分区：

- 默认请求 `UniformPoint(max(2,round(N/2)),M)`。
- 前半程复用 `BMem` 返回的 `minimum/span`，使查询方向与下一代父代处于同一尺度。
- 后半程以当前可行父代目标和历史 `yf` 联合定标，不借相邻参考方向。

### 2.5 主动校准与 CGAN

`RefineBoundaryObservations_RC` 每代最多使用 20 FE：

1. 从可行非支配前沿抽取锚点，以变量范围归一化距离寻找最近不可行点，最多进行 4 次二分，输出真实认证的 `xf/xi/yf/ell`；主线存档只消费 `xf/yf`。
2. 用剩余预算搜索可行前沿的稀疏目标间隙并评价决策中点；必要时只修复一次。

校准解进入共享 `Union`，并被两套环境选择共同读取；未经评价的 CGAN 原始点不会直接进入环境选择。

CGAN 的固定语义：

- pairflag 行为 `[x_b, reference, 1]` 与有限的 `[x_i, reference, 0]`；生成时只查询条件 1。
- 每次通常查询 20 个方向槽，其中 70% 来自已有记忆方向、30% 来自一跳空邻域；池为空时预算回流。
- 每槽生成 5 个候选，通过父代局部尺度选一个，再形成真实评价子代。
- 仅在校准后仍满足 `Problem.FE < 0.5*maxFE` 时训练。
- 默认：`zDim=6`、生成器/critic 隐层 `[32 32]`、100 次生成器更新、每次 4 个 critic 更新、mini-batch 32、学习率 `1e-4`、GP 系数 10、最少训练行 32、采样噪声 0.3。

## 3. 开发阶段与实验收敛

1. **主流程修正**：建立 pairflag 条件数据、14+6 查询、主动边界校准和双种群流程；统一参考向量尺度。
2. **算子与选择筛选**：考察 100DE、PBestDE、NoCGAN 和 BE。它们没有形成可重复的统一收益，均未进入主线。
3. **boundary formula 筛选**：BT1/BT2 未过 gate；BT3 的 runs 1:5 只产生初筛信号，runs 6:10 独立复现未过预注册门槛，CF5/CF6 机制诊断也未支持继续 locality arm。
4. **唯一因素确认**：最终只比较 BT0/F0 与 NoBT，隔离“后半程是否保留认证边界目标”这一因素；该实验通过全部预注册 gate，形成当前唯一主线。
5. **task-specific survival pool 否决**：先以 15 题筛选，再补齐 DASCMOP 1–9、LIRCMOP 1–14、CF 1–10 的 33 题全集，runs `101:105`。按项目既有的非有限 IGD 罚值规则（同题同检查点两臂有限最大值的 10 倍），200K 相对 shared Union 的 `ALL G=0.99594956, W/T/L=10/10/13`，`LIRCF G=0.99370181, 8/6/10`，`DAS G=1.00196845, 2/4/3`，`CF G=1.04418140, 3/0/7`；候选/基线分别有 1/5 个非有限终点 IGD，全部预注册 gate 未通过。该结构没有形成统一收益，恢复 shared Union。
6. **归一化参考方向硬保护（NRBT）否决**：在 shared Union 的 `Fitness<1` 超额集合中保护各方向最近理想点，其余仍按原截断删除。15 题 × runs `101:105` 的 200K 结果为 `ALL G=1.00541003, 6/5/4`、`LIRCF G=1.02829455, 3/3/4`、`CF G=1.08757330, 1/1/2`；方向覆盖中位变化为 0，W/T/L 为 `2/9/4`，联合 GD 比为 `1.17791`，独立 leave-one-seed 复算全部拒绝。该机制既未提高覆盖，也未改善 IGD。
7. **仅归一化现有 SPEA2 截断否决**：唯一因素是把 `Fitness<1` 超额集合的目标逐维 min-max 后交给原字典序近邻删除。15 题 × runs `101:105` 的 200K 结果为 `ALL G=1.02254966, 2/7/6`、`LIRCF G=1.04090864, 1/3/6`、`CF G=1.08652755, 0/0/4`；风险几何比为 `1.03063177/1.05340240`（ALL/LIRCF）。PF-GD guard 通过（`0.97703950/0.98979091`），但方向覆盖中位变化为 0，W/T/L 为 `0/13/2`，IGD、风险、CF 与覆盖 gates 同时失败。预注册 `PROMOTE=0`，活动代码恢复 raw-objective 截断。
8. **CA-anchored auxiliary truncation（CAAT）否决**：保持 shared Union、Pop1 constrained raw-SPEA2、Pop2 unconstrained `CalFitness_CBS`、`Fitness<1`、raw 坐标和最终排序不变；唯一因素是在 Pop2 超额截断中把本轮更新后 Pop1 作为固定、不可删除 anchors。15 题 × runs `106:110` 中，诊断基线 75/75 逐元素复现权威基线，候选 75/75 完成且终点 IGD 全部有限。200K 为 `ALL G=0.995414992, 4/5/6`、`LIRCF G=0.990102157, 4/1/5`、`DAS G=1.006126341`、`LIR G=1.026550706`、`CF G=0.937841415`；风险几何比为 `0.989900604/0.982682512`（ALL/LIRCF）。输出方向覆盖中位变化为 0、W/T/L 为 `0/12/3`，Pop2 辅助新颖覆盖为 0、`0/15/0`；75/75 都调用并执行 anchor 删除，11/15 题的 survivor mask 中位差异大于 0，但 Pop1 PF-GD 为 `1.097952499/1.154390691`，联合 Pop1+Pop2 PF-GD 为 `1.076821870/1.121226335`（ALL/LIRCF）。ALL/LIRCF 主效应、LIR 安全、两项覆盖和两项收敛 guard 未同时通过，`PRIMARY_PASS=0`、`MECHANISM_PASS=0`、`PROMOTE=0`；活动主线未改。
9. **最终主线审计与 sparse-gap 无向去重否决**：审计确认 Primitive 2 会把互为最近邻的 `(i,j)`、`(j,i)` 当成两条 gap，并真实评价同一个决策中点。冻结终态代理的前 8 个槽位中，15 个运行有 12 个出现重复，120 个槽位只有 100 个唯一无向边。候选只做无向 pair 去重并继续扫描下一条 gap，不改 raw 目标距离、预算、RNG、双种群、环境选择、CGAN 或边界存档。`LIRCMOP6_BC/CF5_BC/CF6_BC × runs 101:105` 的 15/15 个 200K-FE 任务全部完成；终点中位 IGD 比为 `0.961026390/0.817093875/1.317862006`，三题几何比 `1.011484119`，终态方向覆盖中位变化为 `0/-1/-2`。CF6 no-harm 与总体门槛同时失败，`PROMOTE=0`；活动 Primitive 2 与固定指纹已恢复到实验前行为。该结果说明“增加独特校准候选”本身不足以保证终态多样性或 IGD 改善。

四项实验分别检验 task-specific 候选池、参考方向硬保护、目标距离尺度和跨种群 anchor 冗余。CAAT 是基于 C-TAEA 审计预注册的最后一个纯环境选择假设，没有复活 NRBT、归一化距离或问题专用分支。四项均未在各自协议下同时通过主效应、机制与安全门槛。当前唯一环境选择仍为：**shared Union + constrained/unconstrained SPEA2 + raw-objective peer-only lexicographic truncation**。不得再以 crowding、PBI、方向配额、归一化、CA/DA anchor 距离、动态权重或问题补丁启动第五个纯环境选择分支；后续多样性假设转向候选产生或交配，不再扫描截断指标。

该裁决也与本地对比算法审计一致：DRMCMO、CPCMO、IMTCMO、MCCMO、MTCMO 的生存选择核心与本项目同属 raw-objective SPEA2 适应度、近邻密度和拥挤截断；DRMCMO 的额外机制位于动态约束放宽、候选档案和角度交配，而不是更强的环境截断。NAEMT 的目标归一化 crowding 依附 CDPPV/NDSort 的质量分层，仅在同层边界裁决，不能作为把归一化距离单独移植到 CBS 的依据。C-TAEA 只提供“辅助 archive 相对更新后收敛 archive 去冗余”的 CAAT 假设来源；CAAT 的拒绝不支持移植其完整 CA/DA、参考方向占用或交配流程。方向或角度机制若要借鉴，也应先在候选产生/交配阶段形成缺失方向，而不是在生存选择中硬保留已有点。

历史 BE、BT1、BT2、BT3、100DE、PBestDE、NoCGAN 与兼容别名的活动源码均已删除。它们只能作为已否决历史解释，不得重新被当作当前候选。

## 4. 最终 BT0-vs-NoBT 证据

协议：

- 问题共 15 个：DASCMOP 5 题、LIRCMOP 6 题、CF 4 题。
- 两个算法 × 15 题 × runs `101:110`，共 300 项任务。
- `N=100`、`maxFE=200000`；100K 是同一运行的检查点，200K 是终点。
- 300/300 完成，0 failure，0 unresolved。

`G` 为候选 BT0/F0 相对 NoBT 的跨题几何均值 IGD 比，小于 1 更好；W/T/L 使用预注册的 ±2% 实用区间：

| 分组 | 题数 | G | W/T/L |
|---|---:|---:|---:|
| ALL | 15 | 0.969851945891575 | 8/7/0 |
| LIRCF | 10 | 0.958059765602720 | 7/3/0 |
| LIR | 6 | 0.948667046651121 | 5/1/0 |
| CF | 4 | 0.972323500747906 | 2/2/0 |
| DAS | 5 | 0.993873521467221 | 1/4/0 |

裁决细节：

- `AllEffectGate`、`CoreLIRCFEffectGate`、`DASNoHarmGate`、`LIRNoHarmGate`、`CFNoHarmGate` 全部为真；`GatePass=1`。
- 150/150 配对任务在 100K 保存点的 FE、IGD、FR、种群决策/目标/约束逐项一致；结合冻结源码可确认两臂只在后半程边界目标消费上分离。
- BT0/F0 使用晚期引导席位 `1,352,166 / 1,362,583 = 0.992354961128973`；NoBT 为 `0 / 1,362,583`。
- 15 个逐题 Wilcoxon 结果经 Holm 校正后均未显著。因此正确表述是“预注册聚合效应与实用门槛通过”，不能写成“每个问题均统计显著”。

冻结证据保留在：

- `Data/CBS_bt0_vs_nobt_gate_manifest.mat`
- `Data/CBS_bt0_vs_nobt_gate_source_snapshot/`：41 个冻结源码文件，含已删除的最终对照类和 runner
- `Data/CBS_bt0_vs_nobt_gate_runs101_110_{runs,problems,groups,decision,results,penalties}.*`
- `Data/CBS_bt0_vs_nobt_gate_diagnostics/`
- `Data/CBS_bt0_vs_nobt_gate_attempt1_failures.csv` 与 `Data/CBS_bt0_vs_nobt_gate_unresolved.csv`：仅表头

`Data/` 被 Git 忽略，但本次清理没有删除或改写任何冻结实验凭证。

## 5. 当前活动源码边界

生产与核心 helper：

- `CBS_RegionWGAN_GP.m`
- `AssignReferenceVectors_CBS.m`
- `OperatorDEDistinct_CBS.m`
- `BoundaryWGAN_RC.m`
- `BuildBoundaryDataset_RC.m`
- `CalFitness_CBS.m`
- `EnvironmentalSelection_CBS.m`
- `RefineBoundaryObservations_RC.m`
- `RunRegionGAN_RC.m`
- `UpdateBoundaryMemory_RC.m`
- `addCBSPaths.m`：先移除当前仓库的 `Data/**` 路径，再只加入 Algorithms/Problems/Metrics，避免冻结源码重新暴露旧类

唯一正式 runner：

- `Support/run_CBS_RegionWGAN_GP_mainline.m`
- `Support/test_CBS_RegionWGAN_GP_mainline_runner.m`

根部旧 `run_bt*.m`、`test_arm_*.m`、`test_operator_branches.m` 与重复且不安全的 `test.m` 已删除。当前本地 Git 只有 `UC-GAN-2` 分支；本次没有删除、切换或修改任何本地/远端 Git ref，“清分支”仅指清除代码层实验臂。

## 6. 正式运行协议

- 正式入口是 `Support/run_CBS_RegionWGAN_GP_mainline.m`，固定 `maxFE=200000`，允许 1 个验证 worker 或 10 个正式 worker；默认 run IDs 为 `1:10`。
- 配对实验必须在每项任务中执行 `rng(runId,'twister')` 后直接调用 `Algorithm.Solve`。`platemo(...)` 会调用 `rng('shuffle')`，并以 `genpath(cd)` 将 `Data/` 内冻结源码快照重新加入 MATLAB 路径，不能作为干净主线或配对随机种子入口；正式 runner 使用 `addCBSPaths` 隔离这些快照。
- runner 的 `resume` 只验证 MAT 文件结构与 FE，不验证源码哈希。代码变化后必须使用新结果根目录，或显式设置 `resume=false`；不得仅凭文件存在复用旧版本结果。
- 原生正式输出目录必须是 `<root>/Data/CBS_RegionWGAN_GP`；`Data/` 不会自动进入 Git。

## 7. 2026-08-11—12 清理、环境选择收敛与验证

- 主类由 994 行降为 571 行，删除 cutoff 全状态快照、差分实验诊断、NoCGAN/PBest/份额覆写 hook 和不可达分支。
- 保留三项最小生产观测：认证目标行数、晚期请求/使用/回退席位。
- MATLAB Code Analyzer 对目录内 22 个活动 `.m` 文件报告 0 issue。
- 当前固定指纹为恢复后的 shared Union + raw-objective SPEA2 主线，问题为 LIRCMOP6_BC，`N=100,D=30,maxFE=20000,rng=4242`，运行环境为 MATLAB 默认计算线程：
  - IGD `1.3468921272381897`
  - 决策和 `1729.8887087629678`
  - 目标和 `1056.8965815545916`
  - 约束和 `0`
  - RNG state 首值 `281257120`
  - boundary rows `222`，晚期请求/使用/回退 `904/904/0`
- 两个失败的 15 题环境选择 campaign 均保留冻结源码、协议、runner、分析器和 75/75 候选原生结果：`Data/CBS_env_selection_nrbt_runs101_105/` 和 `Data/CBS_env_selection_normalized_truncation_runs101_105/`。NRBT 保留基线结果与候选源码 SHA-256；归一化截断另保留完整输入/输出 SHA-256，其 75 个候选结果冻结在 `candidate_results_sha256.txt`。
- 归一化截断结果经独立原生文件复算：75 对终点 FE 均为 200000，150 个终态种群可行率均为 1，候选/基线非有限 IGD 与空可行集均为 0；删任一 seed 或任一问题仍不可能通过 gate。PF-GD 使用 `GetOptimum(10000)` 的平台返回值，实际点数依问题而异，因此只作为已通过的收敛保护，不承担拒绝结论；拒绝由 IGD、风险、CF 与覆盖 gate 独立共同确定。
- CAAT 证据保留在 `Data/CBS_env_selection_caat_runs106_110/`：冻结原始/诊断基线/候选三套源码、预注册协议、runner、因子测试、分析器、150/150 个新执行诊断任务和逐文件 SHA-256。诊断基线 75/75 与权威 runs `106:110` 结果在 FE、指标、决策、目标和约束上逐元素一致；候选 75/75 为 0 failure、0 empty、终点 FE 全为 200000，结果 manifest SHA-256 为 `db80d73a1144be08478d90524b65b219b98cf6b9ba4fc2c53e4c7a3365e94122`。实验因子测试属于冻结证据，不计入当前 12 个生产测试。
- CF6_BC 与 DASCMOP9_BC 的独立小预算基线也完成 FE、全保存种群、IGD 和 RNG 逐元素比较。
- 12 个保留测试全部通过（12/12），覆盖主线指纹、boundary transfer、普通 DE 互异父代、pairflag、14+6 分配、边界搜索、参考方向上限、适应度等价、PlatEMO 接口、正式 runner，以及行为中立的选择来源审计与集成验证。
- 重构前剖析中，WGAN 训练约占主流程 `105.5/108.2` 秒；因此未改动 WGAN 数值内核或三组引导调用。性能清理集中在删除生产路径的大对象快照、死诊断分配和动态增长的选择输出。
- 同机、同进程顺序、相同种子和逐元素同结果的 4K-FE 单次复测中，CF6 从 `5.799948s` 降至 `4.559108s`，DASCMOP9 从 `2.018038s` 降至 `1.463814s`。这是工程回归参考而非统计性能基准；长期耗时仍由 WGAN 训练主导。

## 8. 已知限制与后续约束

- `BoundaryTargetXf/Yf` 不去重、没有参考方向配额，也没有利用 `ell` 排序；高频方向仍可能占据晚期席位。
- Primitive 1 的随机锚点可能重复，4 次二分后的 `xf` 未必足够贴近真实边界。
- Primitive 2 的目标距离未归一化；后半程联合尺度也可能受历史极值影响。
- Primitive 2 仍可能重复评价互为最近邻的同一决策中点；无向去重能消除该重复，但正式实验使 CF6 中位 IGD 恶化 `31.79%` 且未增加终态方向覆盖，因此没有保留这项局部正确但全局不安全的修正。
- 前半程 CGAN `RawDec` 未经评价，查询参考方向不等于生成点的真实目标方向或可行性。
- 当前证据只支持保留 boundary transfer，不支持声称已超过 DRMCMO。
- 四类纯生存选择改造均未同时形成可推广的 IGD 与机制收益。CAAT 虽在 56/75 次运行中改变 Pop2 survivor mask，并显著减少部分问题的跨种群精确重复，但 74/75 次运行没有增加任何辅助新颖方向，输出覆盖也没有净增；去重没有转化为有效 PF 覆盖。事后机制诊断中，LIR+CF 的 Pop2 可行率有 32/50 对下降、5/50 上升，平均下降 9.78 个百分点，说明 raw anchor 排斥还会清除靠近可行 Pop1 的辅助桥接点；该诊断不参与预注册 gate。
- 截断只能从已经产生的候选中删除个体，不能生成 Union 中不存在的 PF 方向。若继续“增强多样性以改善 IGD”，必须先为候选产生或交配阶段建立新的、可验证的缺失方向来源假设；后续不启动第五个纯环境选择分支，也不能把新的截断指标包装成候选来源。
- 后续任何机制改动都必须以当前主线为唯一对照，使用新目录、冻结源码、配对种子和预注册门槛；不得因单题、旧版本或初筛信号复活已否决分支。
- 不得改变 GA/DE 数量舍入、评价顺序、三组 `OperatorDE` 调用顺序或 RNG 调用数，除非先建立新的可复现实验协议并明确接受指纹变化。
