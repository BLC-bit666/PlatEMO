# CBS-CGAN 生产主线与生成/利用归因记录

更新时间：2026-08-27

适用范围：`/Users/lanai/Code/Matlab/PlatEMO/PlatEMO` 当前工作树。`Algorithms/Multi-objective optimization/CBS-CGAN/` 含本轮经授权的 CGAN 生成/利用改造；仓库其他路径存在与 CBS 无关的本地改动，本文件不覆盖它们。本文件是算法身份、实验裁决、运行协议和后续约束的权威入口；旧类名、旧结果目录或旧文档不能覆盖这里的结论。

## 1. 唯一结论

- 唯一生产算法类是 `CBS_RegionWGAN_GP`。
- 生产类的前半程已固定为 A2：全部参考向量均衡生成 500 个候选、同条件 critic 百分位筛至 200、构造 `A/h/T`，再在 T-space 做空缺加权全局 maximin，按本代运行时 `guidedCount` 产生真实评价子代。
- 历史实验标签中的 **BT0/F0 已晋级为主线真实语义**：后半程保留认证 boundary-target 引导，不叠加横向 donor 差分；普通 DE 仍走全局 distinct-parent 的 `current/1-like` 路径；引导步长仍循环 `0.4/0.65/0.85`。
- BT0/F0 不再由别名类或 override hook 表达。`CBS_RegionWGAN_GP_Experiment` 复用生产核心表达 A0/A1/A2；`CBS_RegionWGAN_GP_A1/A2` 只是分别锁定 arm 1/2 的 PlatEMO 启动包装类，不覆盖算法方法、不复制实现，也不改变后半程 strict boundary-target。
- 最终 BT0-vs-NoBT 预注册实验的 5 个 gate 全部通过，因此保留 boundary transfer，否决“后半程完全取消边界目标”的 NoBT 对照。
- 2026-08-17—19 的环境选择、校准记忆、边界目标新鲜度、稀疏 gap 去重和晚期引导覆盖实验均未通过各自冻结门槛；它们没有进入本轮活动实现。
- 本轮短程回归和确定性指纹只证明代码按设计执行；“新生成方式”和“新利用方式”是否分别有价值，必须由第 10 节的正式归因实验决定，当前不得提前写成已证实收益。A0 直接复用已有 200K-FE 数据，只新增 A1、A2。
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

### 2.3 前半程 CGAN 的 A/h/T 利用

对 critic 保留的每个生成候选 `G`：

1. 从当前可行集取 `Fitness<1` 的精英 `E`；若为空则以全部可行解作为 `E`。
2. 在归一化决策空间寻找最近精英父代 `A`。距离并列时，依次选择与候选查询方向更近、适应度更好、原可行行号更小的父代。
3. 局部尺度 `h` 由全部可行解估计：只使用大于 `1e-12` 的归一化距离，至少需要两个正邻居，取最近至多 3 个距离的中位数；局部无效时用全体有效 `h` 的中位数，全部无效则该席位回退普通 DE。
4. 令 `d=||(G-A)/(upper-lower)||`、`alpha=min(1,h/d)`、`T=A+alpha*(G-A)`，并防御性截断到变量边界。
5. 在 T-space 全局选择：`delta` 是到 `E` 和已选 T 的最小归一化决策距离，评分 `J=delta/(1+o_ref+s_ref)`；`o_ref` 是精英方向占用，`s_ref` 是本轮已选条件数。最佳 `delta<=1e-12` 时停止并回退剩余席位。

最终调用：

```matlab
OperatorDE(Problem,A,T,A,{1,1,1,20})
```

`h` 只约束 Polynomial Mutation 前的中心 `T`，不约束实际子代；机制记录分别保存 `meanCenterStep` 与 `meanActualStep`，二者不能混为一谈。最终选择数不是硬编码 20，而是当前代算子配比产生的 `guidedCount`。

### 2.4 后半程 boundary-target 公式

认证目标仍调用：

```matlab
OperatorDE(Problem,A,G,A,{1,F,1,20})
```

多项式变异前为 `A + F*(G-A)`：

- `A` 是可行父代，`F` 按子代循环 `0.4, 0.65, 0.85`。
- 后半程 `G` 是二分校准后经真实评价认证的可行端点 `xf`。
- 父代和目标仍须属于联合归一化后的同一参考方向；缺失即回退。
- 主线不再存在 BT1/BT2/BT3 的横向 `0.5*(R1-R2)` 分支。该 strict 分支的 slot 次序、固定 F 三档和三组 `OperatorDE` 调用保持原语义。

### 2.5 两套边界状态

| 状态 | 生命周期与用途 | 核心规则 |
|---|---|---|
| `BMem` | 仅前半程更新；构造 pairflag 数据并训练/查询 CGAN | 每参考方向最多 5 个前两层可行锚点；不可行配对每代重建；使用邻近参考方向与 MAD gap 过滤 |
| `BoundaryTargetXf/Yf` | 校准全程产生；仅后半程 Pop1 消费 | 只存真实认证的 `xf/yf`；FIFO 上限 1500；newest-first；目标与父代必须在联合归一化后属于同一参考方向 |

`AssignReferenceVectors_CBS.m` 统一采用逐目标 min-max 归一化和最大余弦相似度分区：

- 默认请求 `UniformPoint(max(2,round(N/2)),M)`。
- 前半程复用 `BMem` 返回的 `minimum/span`，使查询方向与下一代父代处于同一尺度。
- 后半程以当前可行父代目标和历史 `yf` 联合定标，不借相邻参考方向。

### 2.6 主动校准与 CGAN

`RefineBoundaryObservations_RC` 每代最多使用 20 FE：

1. 从可行非支配前沿抽取锚点，以变量范围归一化距离寻找最近不可行点，最多进行 4 次二分，输出真实认证的 `xf/xi/yf/ell`；主线存档只消费 `xf/yf`。
2. 用剩余预算搜索可行前沿的稀疏目标间隙并评价决策中点；必要时只修复一次。

校准解进入共享 `Union`，并被两套环境选择共同读取；未经评价的 CGAN 原始点不会直接进入环境选择。

CGAN 的固定语义：

- pairflag 行为 `[x_b, reference, 1]` 与有限的 `[x_i, reference, 0]`；生成时只查询条件 1。
- 每次让全部 `W` 参与，均衡分配 500 个条件；每方向数量最多相差 1，余数方向和最终查询次序随机，每行独立采样噪声 `z`。
- critic 用生成时的同一条件 `[w,1]` 评分。原始 critic 分数不跨条件比较；先在每个条件内按分数降序形成相对百分位，再按百分位全局保留 200 个，分数并列和二次选择并列均保持原始行优先。
- 200 个候选不直接评价或进入 `Union`；下一代用最新 `Population1` 完成 A/h/T 映射和 T-space 选择，只有最终 CGAN 子代接受真实评价。
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
9. **E1 constrained-selection package 否决**：候选把 Pop1 underflow 改为分层 maximin fill，并保护每次校准产生的认证代表。15 题 × runs `106:110` 的 75/75 个候选终点均为有限值；`ALL G=0.97190`，但逐题 win/loss 为 `3/3`，未满足严格的 wins > losses。CF5/LIRCMOP5 中位比为 `0.67963/0.81850`，CF6/LIRCMOP12 为 `1.14832/1.03597`，收益集中且安全性不足，`PROMOTE=0`。
10. **E2 attribution battery 否决**：fill-only 为 `G=0.9854, W/L=2/4`，protection-only 为 `G=0.9994, W/L=2/3`，E1+E2 为 `G=1.0140, W/L=3/6`。E2 的预注册机制预测被否证：CF6 仍为 `1.067>1.05`，CF5 恶化到 `1.499`。两项因素都伤害 CF6；CF5 收益主要来自 fill，LIRCMOP5 收益主要来自 protection。没有 arm 通过门槛，fresh-seed D/E 两臂在保存 155/375 项后按协议取消，E1/E2 全部回滚。
11. **认证 bracket 直接进入 BMem 否决**：候选在前半程用真实 `xf/xi/yf/ell` 对替代同一可行锚点的重建配对，不改容量、FE 或后半程 transfer。`LIRCMOP6_BC/CF5_BC/CF6_BC × runs 101:105` 的 15/15 项完成，三题几何比 `1.09381015484361`，CF6 为 `1.25875001452221`；no-harm 与 promotion gate 均失败，活动 `BMem` 继续从已评价种群云重建配对。
12. **fresh-only 晚期边界目标否决**：候选每次校准后覆盖 target buffer，只允许下一代消费最新认证目标。15/15 项完成且 100K 两臂完全一致；200K 三题几何比 `1.004077669445274`，LIRCMOP6/CF5/CF6 为 `0.977624622257646/0.995483291979432/1.040149630683574`。CF6 no-harm 失败，活动源码保留累计 FIFO 存档。
13. **sparse-gap 无向去重否决**：审计确认 Primitive 2 会把互为最近邻的 `(i,j)`、`(j,i)` 当成两条 gap，并真实评价同一个决策中点。冻结终态代理的前 8 个槽位中，15 个运行有 12 个出现重复，120 个槽位只有 100 个唯一无向边。候选只做无向 pair 去重并继续扫描下一条 gap，不改 raw 目标距离、预算、RNG、双种群、环境选择、CGAN 或边界存档。`LIRCMOP6_BC/CF5_BC/CF6_BC × runs 101:105` 的 15/15 个 200K-FE 任务全部完成；终点中位 IGD 比为 `0.961026390/0.817093875/1.317862006`，三题几何比 `1.011484119`，终态方向覆盖中位变化为 `0/-1/-2`。CF6 no-harm 与总体门槛同时失败，`PROMOTE=0`；活动 Primitive 2 与固定指纹已恢复到实验前行为。
14. **晚期 guide-pool 方向覆盖修复否决**：H1 每方向每代最多取一个 newest-first target，H2 在同方向无可行父代时改用最近方向父代。小预算机制检查把四题的每代唯一引导方向提高到约 19.6–20，并消除 fallback；但 15 题 × runs `106:110` 的 `ALL G=1.00062`、`CF/LIR/DAS G=0.98956/1.01400/0.99360`，逐题 `W/T/L=4/8/3`，预注册总体门槛失败。当前源码仍按 newest-first slot 和 strict same-reference parent 执行。

早期四项纯环境选择实验检验 task-specific 候选池、参考方向硬保护、目标尺度和跨种群 anchor 冗余；其后的 E1/E2 又检验 constrained fill 与校准点保护。所有方案均未同时通过主效应、机制与安全门槛。当前唯一环境选择仍为：**shared Union + constrained/unconstrained SPEA2 + raw-objective peer-only lexicographic truncation**。不得再以 crowding、PBI、方向配额、归一化、CA/DA anchor 距离、maximin fill、校准点保护、动态权重或问题补丁启动新的选择分支。

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

生产 runner：

- `Support/run_CBS_RegionWGAN_GP_mainline.m`
- `Support/test_CBS_RegionWGAN_GP_mainline_runner.m`

顺序归因入口：

- `Experimental/CBS_RegionWGAN_GP_Experiment.m`
- `Experimental/CBS_RegionWGAN_GP_A1.m`、`CBS_RegionWGAN_GP_A2.m`
- 根目录通用启动脚本 `test.m`
- `Support/analyze_CBS_CGAN_factor_experiment.m`
- `Support/CBS_CGAN_factor_experiment_protocol.md`
- `Support/test_CBS_CGAN_factor_experiment.m`

截至 2026-08-25，活动目录共有 28 个 `.m` 文件（新增两个固定arm薄包装类）。主类为 977 行，SHA-256 为 `8378a015d6ca9d226caed991c95de800a0fff349fa18aaa8ea405a409697e0c7`。生产类公开只读 `guideExperimentSnapshot`，汇总 CGAN 生成数、条件覆盖、筛选百分位、映射/回退、`alpha/h/d`、中心/实际步长、方向余弦、可行率、父代支配改善率和环境选择存活率；它不保存逐个候选大数组。

根部旧 `run_bt*.m`、`test_arm_*.m`、`test_operator_branches.m` 已删除。根部 `test.m` 是包括 CBS A1/A2 在内的唯一通用批量启动脚本；算法列表、问题、runs、workers、`platemo(...)` 调用和自动保存均以该文件为准。当前本地 Git 只有 `UC-GAN-2` 分支；本次没有删除、切换或修改任何本地/远端 Git ref。

## 6. 正式运行协议

- 生产入口是 `Support/run_CBS_RegionWGAN_GP_mainline.m`，固定 `maxFE=200000`，允许 1 个验证 worker 或 10 个正式 worker；默认 run IDs 为 `1:10`。
- 生成/利用正式运行入口是根目录通用脚本 `test.m`：`algNames` 只列 A1/A2，问题为 DASCMOP1--9_BC 与 LIRCMOP1--14_BC，runs `1:15`、`N=100`、`maxFE=200000`、`save=1`、10 个 process workers，共 690 项；A0 不在任务列表中，直接复用已有结果。
- `test.m` 保留项目通用语义：`addpath(genpath(rootPath))`、清理旧 Jobs、扁平 `parfor`、`platemo(...)`、其内部 `rng('shuffle')` 与按算法类名自动保存。run ID 是结果编号，不是 seed；A0、A1、A2 均不能按同 run 声称严格随机配对，正式统计按独立重复运行处理。
- 生产 runner 的 `resume` 只验证 MAT 文件结构与 FE，不验证源码哈希。通用 `test.m` 以对应类名结果 MAT 是否存在判断完成并跳过。
- 原生正式输出目录必须是 `<root>/Data/CBS_RegionWGAN_GP`；`Data/` 不会自动进入 Git。

## 7. 2026-08-11—27 清理、实验收敛与验证

- 2026-08-27 清理 `Data/`：95 个已淘汰顶层条目（6,963 个文件，约 1.48 GiB）移入 macOS 废纸篓；删除范围仅保留为已撤销源码对应的旧 CBS 分支结果、早期 BT/算子筛选产物和已结束任务日志。曾被误移的 `CCMO/CMOEAD/CMOEMT/CPCMO/DRMCMO/IMTCMO/MCCMO/MTCMO/NSGAII/PPS/ToP` 对比算法目录已完整恢复；A0/A1/A2、当前四个外部对照、最终 BT0-vs-NoBT 凭证和本节明确列出的失败实验冻结包均保留。
- 2026-08-19 的精简曾把主类降为 523 行；本轮新增的是生产 A2 核心、轻量聚合机制记录和三臂共用分派，不恢复旧 NoCGAN/PBest/BT 公式分支，也不改变环境选择或后半程认证逻辑。
- MATLAB R2025b Code Analyzer 对目录内 28 个活动 `.m` 文件报告 0 issue（2026-08-25 复核）。
- 当前固定指纹为 A2 CGAN 前半程 + 原 strict boundary-target 后半程 + shared Union/raw-objective SPEA2，问题为 LIRCMOP6_BC，`N=100,D=30,maxFE=20000,rng=4242`，运行环境为 MATLAB 默认计算线程：
  - IGD `1.3473816458691643`
  - 决策和 `1828.5991360038395`
  - 目标和 `625.98054078737562`
  - 约束和 `0`
  - RNG state 首值 `274547642`
  - 同次运行记录 28 次生成事件、`14000/5600` 个原始/critic 保留候选、560 个已选 CGAN 子代、0 回退；这些机制数只作实现审计，不进入固定数值断言或收益结论
- 两个失败的 15 题环境选择 campaign 均保留冻结源码、协议、runner、分析器和 75/75 候选原生结果：`Data/CBS_env_selection_nrbt_runs101_105/` 和 `Data/CBS_env_selection_normalized_truncation_runs101_105/`。NRBT 保留基线结果与候选源码 SHA-256；归一化截断另保留完整输入/输出 SHA-256，其 75 个候选结果冻结在 `candidate_results_sha256.txt`。
- 归一化截断结果经独立原生文件复算：75 对终点 FE 均为 200000，150 个终态种群可行率均为 1，候选/基线非有限 IGD 与空可行集均为 0；删任一 seed 或任一问题仍不可能通过 gate。PF-GD 使用 `GetOptimum(10000)` 的平台返回值，实际点数依问题而异，因此只作为已通过的收敛保护，不承担拒绝结论；拒绝由 IGD、风险、CF 与覆盖 gate 独立共同确定。
- CAAT 证据保留在 `Data/CBS_env_selection_caat_runs106_110/`：冻结原始/诊断基线/候选三套源码、预注册协议、runner、因子测试、分析器、150/150 个新执行诊断任务和逐文件 SHA-256。诊断基线 75/75 与权威 runs `106:110` 结果在 FE、指标、决策、目标和约束上逐元素一致；候选 75/75 为 0 failure、0 empty、终点 FE 全为 200000，结果 manifest SHA-256 为 `db80d73a1144be08478d90524b65b219b98cf6b9ba4fc2c53e4c7a3365e94122`。该实验因子测试只属于冻结证据，不计入当前活动测试入口。
- 后续失败实验分别冻结在 `Data/CBS_env_selection_e1_runs106_110/`、`Data/CBS_env_selection_e2_battery/`、`Data/CBS_certified_bracket_memory_runs101_105/`、`Data/CBS_fresh_boundary_targets_runs101_105/`、`Data/CBS_sparse_gap_unique_pairs_runs101_105/` 和 `Data/CBS_final_guidefix_runs106_110/`。其中 E2 fresh-seed 两臂按冻结 red-line 提前取消，不得表述为完整 375 项 campaign。
- CF6_BC 与 DASCMOP9_BC 的独立小预算基线也完成 FE、全保存种群、IGD 和 RNG 逐元素比较。
- 当前 11 个活动测试全部通过（11/11，2026-08-25），并在结构测试中执行 A0/A1/A2 三条真实短程路径。覆盖范围包括 boundary transfer、pairflag 与 critic 评分、旧 14+6、全 W 均衡 500、条件内筛选、A/h/T 短程机制、PlatEMO 接口、生产 runner、归因分析和默认 20K 指纹。
- WGAN 的训练损失、GP、网络结构和优化器未改；新增了训练后同条件 critic 前向评分。旧性能数字不再代表当前 500 候选主线，不能继续用于性能等价声明。

## 8. 2026-08-19—25 正式对比数据状态

- 文件清点显示 `Data/CBS_RegionWGAN_GP/` 有 1120 个原生 MAT：DASCMOP 1–9 与 LIRCMOP 1–14 各题 40 个，CF 1–10 各题 20 个。
- `Data/DRMCMO-GA/`、`Data/NAEMT/`、`Data/EADMMMOEAD/`、`Data/EADMMNSGAII/` 各有 690 个 MAT，即 DASCMOP 1–9 与 LIRCMOP 1–14 各题 30 个。它们说明对比运行已产生原始输出，不等于已经完成统计裁决。
- 这些顶层结果目录只有 MAT，没有同批源码 SHA-256、manifest、冻结统计协议或 CBS 对比结论；CBS 文件还分布在 2026-08-19—22 的多次执行中。提交 `56940552` 中的确认入口使用 `addpath(genpath(rootPath)) + platemo(...)`，不满足第 6 节规定的路径隔离与配对 RNG 协议。
- `Data/DRMCMO/` 混有多批问题与运行编号，但作为用户指定的对比算法结果已保留；只读抽查曾发现 CBS 终点 FE 为 200000、其中一个 DRMCMO 文件为 200100。所有对比目录仍不能未经统一 FE、源码身份、问题版本、指标重算和统计协议核对就直接比较。
- 因此当前准确状态是“外部确认原始数据已积累，正式可追溯比较尚未完成”。不得从文件数量、单题中位数或 GUI 表格直接宣称 CBS 优于 DRMCMO、NAEMT 或 EADMM 变体。

## 9. 已知限制与后续约束

- `BoundaryTargetXf/Yf` 不去重、没有参考方向配额，也没有利用 `ell` 排序；高频方向仍可能占据晚期席位。
- late guide-pool 的 newest-first slot 会重复消费少数新方向，strict same-reference parent 还会丢弃无匹配方向；H1/H2 已显著修复机制指标但未改善聚合 IGD，因此不能仅凭覆盖改善重新启用。
- Primitive 1 的随机锚点可能重复，4 次二分后的 `xf` 未必足够贴近真实边界。
- Primitive 2 的目标距离未归一化；后半程联合尺度也可能受历史极值影响。
- Primitive 2 仍可能重复评价互为最近邻的同一决策中点；无向去重能消除该重复，但正式实验使 CF6 中位 IGD 恶化 `31.79%` 且未增加终态方向覆盖，因此没有保留这项局部正确但全局不安全的修正。
- 前半程 CGAN `RawDec` 未经评价，查询参考方向不等于生成点的真实目标方向或可行性。
- 同条件 critic 百分位只解决“原始 critic 分数不能跨条件比较”，不证明 critic 已校准，也不保证保留候选可行；必须结合生成/拒绝百分位、最终子代可行率和存活率解释。
- A/h/T 的 `h` 只限制变异前中心。若实际子代步长显著大于中心步长，应先检查 Polynomial Mutation，而不能把它误判为局部尺度实现失效。
- 真实 bracket 直接写入 `BMem`、fresh-only target、环境选择 maximin fill、认证点保护及其 E2 组合均已有负面冻结证据；这些机制不能再作为“未试过”的候选。它们与本轮只用于 CGAN 候选选取的 T-space maximin 不是同一机制。
- 当前证据只支持保留 boundary transfer，不支持声称已超过 DRMCMO。
- 四类早期纯生存选择改造与 E1/E2 constrained-selection package 均未形成可推广的 IGD 与机制收益。CAAT 虽在 56/75 次运行中改变 Pop2 survivor mask，并显著减少部分问题的跨种群精确重复，但 74/75 次运行没有增加任何辅助新颖方向，输出覆盖也没有净增；去重没有转化为有效 PF 覆盖。事后机制诊断中，LIR+CF 的 Pop2 可行率有 32/50 对下降、5/50 上升，平均下降 9.78 个百分点，说明 raw anchor 排斥还会清除靠近可行 Pop1 的辅助桥接点；该诊断不参与预注册 gate。
- 截断只能从已经产生的候选中删除个体，不能生成 Union 中不存在的 PF 方向。若继续“增强多样性以改善 IGD”，必须先为候选产生或交配阶段建立新的、可验证的缺失方向来源假设；后续不再启动新的纯环境选择分支，也不能把新的截断指标包装成候选来源。
- 后续任何机制改动都必须以当前主线为唯一对照，使用新目录、冻结源码、配对种子和预注册门槛；不得因单题、旧版本或初筛信号复活已否决分支。
- 本轮已通过明确协议接受前半程 RNG 与指纹变化；后续不得再改变 GA/DE 数量舍入、评价顺序、A/h/T 选择或后半程三组 `OperatorDE` 调用顺序，除非先建立新的可复现实验协议并重新冻结指纹。

## 10. CGAN 生成/利用正式归因计划

### 10.1 三臂而非强制 2×2

旧候选机制把 20 个查询槽、每槽 5 个候选、父代局部尺度和固定 F 绑定在一起；“旧生成＋新利用”没有唯一自然语义。正式实验采用顺序归因：

| 臂 | 生成/筛选 | 利用 | 比较解释 |
|---|---|---|---|
| A0 | 旧 20 槽 × 5，无 critic | 旧逐槽选择、固定 F 三档 | 历史全链基线 |
| A1 | 全 W 均衡 500、同条件 critic 筛 200、G-space 空缺加权全局 maximin | 原父代选择和固定 F 三档 | A1/A0：生成/筛选包 |
| A2 | 与 A1 相同 | A/h/T、T-space 空缺加权全局 maximin、F=1 | A2/A1：利用包 |

A2/A0 是完整改造的端到端效应。A1/A0 与 A2/A1 是“算法包的顺序归因”，不能表述为两个正交原子因素的独立因果主效应。生产类固定 A2；实验类的 arm 只改变前半程，三臂共用 BMem、WGAN 训练、FE、环境选择和后半程 strict boundary-target。

### 10.2 矩阵、独立重复与输出

- 23 题全集：`DASCMOP1_BC`--`DASCMOP9_BC` 与 `LIRCMOP1_BC`--`LIRCMOP14_BC`。
- A0 已有 `23×15=345` 个 runs `1:15` 结果，位于 `Data/CBS_RegionWGAN_GP`，不重跑；`test.m` 的算法列表只含 A1/A2，共 `2×23×15=690` 项。
- `test.m` 当前固定 `N=100`、`maxFE=200000`、`save=1`、10 个 process workers，并保留 `platemo(...)` 内部 `rng('shuffle')`。run ID 只负责文件编号，各臂按独立重复运行分析。
- A1、A2 通过无机制覆盖的薄包装类分别保存到 `Data/CBS_RegionWGAN_GP_A1`、`Data/CBS_RegionWGAN_GP_A2`；已有主结果按通用脚本逻辑跳过。
- 主指标是 200K FE 的 IGD。`save=1` 只保存终态种群；通用启动脚本不额外保存 `guideSnapshot`。

### 10.3 预注册门槛

A1/A0、A2/A1、A2/A0 分别独立检查：

1. 所有终点 IGD 有限且为正；
2. 每题先分别取两臂15次独立运行的中位数，再计算候选/基线比，跨23题几何均值 `<=0.98`；
3. 明显胜出题数（`<=0.98`）大于明显退化题数（`>1.02`）；
4. DASCMOP 与 LIRCMOP 两个家族的几何比均 `<=1.00`。

任何非有限结果直接失败。A1/A0 不通过时，不得把 A2/A1 的正结果单独解释为“新生成已有效”；A2/A1 不通过时，应保留 A1 作为候选并用机制记录诊断利用路径。正式协议文件为 `Algorithms/Multi-objective optimization/CBS-CGAN/Support/CBS_CGAN_factor_experiment_protocol.md`。
