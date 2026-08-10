# CBS-CGAN 唯一主线开发记录

更新时间：2026-08-10

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

校准解进入公共 `Union`；未经评价的 CGAN 原始点不会直接进入环境选择。

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

## 7. 2026-08-10 清理与等价性验证

- 主类由 994 行降为 571 行，删除 cutoff 全状态快照、差分实验诊断、NoCGAN/PBest/份额覆写 hook 和不可达分支。
- 保留三项最小生产观测：认证目标行数、晚期请求/使用/回退席位。
- MATLAB Code Analyzer 对目录内 22 个活动 `.m` 文件报告 0 issue。
- LIRCMOP6_BC，`N=100,D=30,maxFE=20000,rng=4242`：重构前后两个保存点的 FE、决策、目标、约束、IGD 与 RNG 末态逐元素完全一致：
  - IGD `1.3468921272381897`
  - 决策和 `1729.8887087629678`
  - 目标和 `1056.8965815545916`
  - 约束和 `0`
  - RNG state 首值 `281257120`
  - boundary rows `222`，晚期使用 `904/904`
- CF6_BC 与 DASCMOP9_BC 的独立小预算基线也完成 FE、全保存种群、IGD 和 RNG 逐元素比较。
- 10 个保留测试全部通过（10/10），覆盖主线指纹、boundary transfer、普通 DE 互异父代、pairflag、14+6 分配、边界搜索、参考方向上限、适应度等价、PlatEMO 接口和正式 runner。
- 重构前剖析中，WGAN 训练约占主流程 `105.5/108.2` 秒；因此未改动 WGAN 数值内核或三组引导调用。性能清理集中在删除生产路径的大对象快照、死诊断分配和动态增长的选择输出。
- 同机、同进程顺序、相同种子和逐元素同结果的 4K-FE 单次复测中，CF6 从 `5.799948s` 降至 `4.559108s`，DASCMOP9 从 `2.018038s` 降至 `1.463814s`。这是工程回归参考而非统计性能基准；长期耗时仍由 WGAN 训练主导。

## 8. 已知限制与后续约束

- `BoundaryTargetXf/Yf` 不去重、没有参考方向配额，也没有利用 `ell` 排序；高频方向仍可能占据晚期席位。
- Primitive 1 的随机锚点可能重复，4 次二分后的 `xf` 未必足够贴近真实边界。
- Primitive 2 的目标距离未归一化；后半程联合尺度也可能受历史极值影响。
- 前半程 CGAN `RawDec` 未经评价，查询参考方向不等于生成点的真实目标方向或可行性。
- 当前证据只支持保留 boundary transfer，不支持声称已超过 DRMCMO。
- 后续任何机制改动都必须以当前主线为唯一对照，使用新目录、冻结源码、配对种子和预注册门槛；不得因单题、旧版本或初筛信号复活已否决分支。
- 不得改变 GA/DE 数量舍入、评价顺序、三组 `OperatorDE` 调用顺序或 RNG 调用数，除非先建立新的可复现实验协议并明确接受指纹变化。
