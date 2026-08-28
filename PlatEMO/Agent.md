# CBS-CGAN 生产主线与生成/利用归因记录

更新时间：2026-08-28

适用范围：`/Users/lanai/Code/Matlab/PlatEMO/PlatEMO` 当前工作树。`Algorithms/Multi-objective optimization/CBS-CGAN/` 含本轮经授权的 CGAN 生成/利用改造；仓库其他路径存在与 CBS 无关的本地改动，本文件不覆盖它们。本文件是算法身份、实验裁决、运行协议和后续约束的权威入口；旧类名、旧结果目录或旧文档不能覆盖这里的结论。

## 1. 唯一结论

- 唯一生产算法类是 `CBS_RegionWGAN_GP`。
- 生产类的前半程自 2026-08-28 本次迁移起固定为“全W均衡500候选＋同条件critic保留200＋新利用”：保留候选使用`A/h/T`局部目标映射、T-space全局maximin和`F=1`产生本代`guidedCount`个真实评价子代。生产主线不称为A0或A2，机制快照仍使用独立的`arm=-1`。
- 机制快照以`arm=-1`标记生产主线；历史A0/A1/A2继续固定为`0/1/2`，避免新主线与旧A0共享诊断身份。
- 历史实验标签中的 **BT0/F0 已晋级为后半程主线真实语义**：保留认证 boundary-target 引导，不叠加横向 donor 差分；普通 DE 仍走全局 distinct-parent 的 `current/1-like` 路径；后半程引导步长仍循环 `0.4/0.65/0.85`。
- `Core/CBS_RegionWGAN_GP_Core.m` 是唯一共享算法引擎；生产类以及`Variants/`中的A1、A2、Random20、DE20、GA20都是可由PlatEMO GUI发现的薄入口。内部`CBS_RegionWGAN_GP_Experiment`冻结历史A0/A1/A2语义，不带GUI标签、不复制实现，也不改变后半程strict boundary-target。历史A2的默认机制现在与生产主线相同，但类名、`arm=2`和已有结果身份保持冻结；A1只为历史结果兼容保留。Random20/DE20/GA20是独立名额消融。
- 最终 BT0-vs-NoBT 预注册实验的 5 个 gate 全部通过，因此保留 boundary transfer，否决“后半程完全取消边界目标”的 NoBT 对照。
- 2026-08-17—19 的环境选择、校准记忆、边界目标新鲜度、稀疏 gap 去重和晚期引导覆盖实验均未通过各自冻结门槛；它们没有进入本轮活动实现。
- 本轮短程回归和确定性指纹只证明代码按设计执行。500→200生成和新利用均为用户指定的当前主线语义，不等同于已经证明收益；第10节旧E8终点裁决保留为历史证据，但后续必须按“精确CGAN截止性能＋不区分可行侧的边界邻近度”重新诊断。
- 该裁决只确立项目内部唯一主线，不等于已经证明优于 DRMCMO。现有外部对照证据仍不足以宣称在 LIRCMOP 与 CF 上整体超过 DRMCMO。

## 2. 主线的精确算法语义

### 2.1 双种群、算子配比和 FE

以下为 `N=100` 完整进化批次的名义数量，不含校准评价：

| 代首阶段 | 约束种群 Pop1 | 无约束种群 Pop2 |
|---|---|---|
| `FE < 0.5*maxFE` | 25 GA + 55 普通 DE + 20 critic保留候选的局部目标DE | 25 GA + 75 普通 DE |
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

对500个原始候选经同条件critic保留的200个生成候选中的每个 `G`：

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

CGAN 的共享训练语义：

- pairflag 行为 `[x_b, reference, 1]` 与有限的 `[x_i, reference, 0]`；生成时只查询条件 1。
- 生产主线让全部`W`参与并均衡分配500个条件；critic按同一条件`[w,1]`评分，先形成条件内百分位，再全局保留200个，之后使用`A/h/T`和T-space选择。
- 历史A2默认使用同一生成与利用机制，仅为旧结果身份和启动兼容保留，不再构成生产主线的单因素对照。
- 历史A0仍表示旧生成＋旧利用，历史A1仍表示新生成＋旧利用；类语义和已有结果身份冻结，不随生产晋级改名。
- 未经评价的CGAN原始候选不直接进入`Union`；只有最终引导子代接受真实评价。
- 仅在校准后仍满足 `Problem.FE < 0.5*maxFE` 时训练。
- 默认：`zDim=6`、生成器/critic 隐层 `[32 32]`、100 次生成器更新、每次 4 个 critic 更新、mini-batch 32、学习率 `1e-4`、GP 系数 10、最少训练行 32、采样噪声 0.3。

当前训练集构造的精确限制：

1. 每次事件收集当前`Population1`、`Offspring1 + Calibration`、`Population2`、`Offspring2`；上一轮只继承`BMem.x_b/y_b`中的可行锚点，不继承旧不可行端点。
2. 可行性严格按真实约束`sum(max(0,cons))<=0`判定。只允许可行解的前两层Pareto前沿作为锚点，并按约束适应度把每个参考方向封顶为5个。
3. 对每个锚点，只在本方向及其最近参考方向邻域内寻找当前事件的不可行点；默认`pairNeighborRefRadius=2`，即最多检查5个参考方向。
4. 若可行锚点在目标上支配某个不可行候选，该候选被排除；其余候选取归一化目标空间距离最近者。锚点找不到合法不可行配对时会整行消失，而不是作为单独正样本保留。
5. 所有完整对再经过全局MAD gap上限过滤；默认至少保留2对（若原本可用）。每对构造成一条`[x_b,W(ref),1]`和一条`[x_i,W(ref),0]`，重复对不去重，以其出现次数形成训练权重。
6. 训练门槛是总训练行数至少`max(minBoundaryLength,minGANTrainCount)=32`且`QueryRefs`非空。现有完整成对结构下通常等价于至少16对，但没有单独检查正样本数、负样本数或条件覆盖数。
7. 训练后生成的池只供下一代使用；消费时还要求当前Pop1至少存在一个真可行父代，否则全部引导席位回退普通DE。

因此早期严格性是“先找到真可行解→进入前两层和方向上限→找到当前代合法不可行配对→累计到32行→下一代仍有可行父代”的复合门槛。当前实现没有放宽可行端点，也没有把不可行端点伪标为正样本。

## 3. 开发阶段与实验收敛

1. **主流程修正**：建立 pairflag 条件数据、14+6 查询、主动边界校准和双种群流程；统一参考向量尺度。
2. **算子与选择筛选**：早期100DE、PBestDE、NoCGAN和BE没有形成可重复的统一收益，均未进入主线。当前Random20/DE20/GA20是重新冻结的“只替换前半程20%名额”对照，不恢复早期分支的其他公式。
3. **boundary formula 筛选**：BT1/BT2 未过 gate；BT3 的 runs 1:5 只产生初筛信号，runs 6:10 独立复现未过预注册门槛，CF5/CF6 机制诊断也未支持继续 locality arm。
4. **后半程唯一因素确认**：最终只比较BT0/F0与NoBT，隔离“后半程是否保留认证边界目标”这一因素；该实验通过全部预注册gate，固定当前后半程语义。
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

生产入口与共享核心：

- `CBS_RegionWGAN_GP.m`：生产 GUI 薄入口，保留算法类名、参数注释和结果目录身份。
- `Core/CBS_RegionWGAN_GP_Core.m`：唯一主循环、状态、FE 分配、CGAN/boundary-target 切换和机制统计。
- `Core/AssignReferenceVectors_CBS.m`、`Core/OperatorDEDistinct_CBS.m`、`Core/CalFitness_CBS.m`、`Core/EnvironmentalSelection_CBS.m`：进化算子与选择。
- `Core/BoundaryWGAN_RC.m`、`Core/RunRegionGAN_RC.m`、`Core/BuildBoundaryDataset_RC.m`、`Core/UpdateBoundaryMemory_RC.m`、`Core/RefineBoundaryObservations_RC.m`：边界与 CGAN。
- `Support/addCBSPaths.m`：先移除当前仓库的 `Data/**` 路径，再只加入 Algorithms/Problems/Metrics，避免冻结源码重新暴露旧类。

生产 runner：

- `Support/run_CBS_RegionWGAN_GP_mainline.m`
- `Tests/test_CBS_RegionWGAN_GP_mainline_runner.m`

历史归因、名额消融与当前顺序筛选入口：

- `Variants/Internal/CBS_RegionWGAN_GP_Experiment.m`：冻结历史A0/A1/A2配置的内部基类，不暴露到 GUI。
- `Variants/CBS_RegionWGAN_GP_A1.m`：历史新生成＋旧利用入口，仅为旧结果兼容保留。
- `Variants/CBS_RegionWGAN_GP_A2.m`：冻结的新生成＋新利用历史入口；默认机制与当前主线相同，但结果身份不同。
- `Variants/CBS_RegionWGAN_GP_Random20.m`：仅以前半程决策空间随机采样替换 CGAN 引导真实评价名额的 GUI 消融入口；后半程 strict boundary-target 不变。
- `Variants/CBS_RegionWGAN_GP_DE20.m`：以前半程普通DE替换同一名额；完整`N=100`代为25 GA + 75普通DE。
- `Variants/CBS_RegionWGAN_GP_GA20.m`：以前半程GA替换同一名额；完整`N=100`代为45 GA + 55普通DE。
- 根目录通用启动脚本 `test.m`
- `Support/analyze_CBS_CGAN_factor_experiment.m`
- `Support/CBS_CGAN_factor_experiment_protocol.md`
- `Tests/test_CBS_CGAN_factor_experiment.m`
- `Variants/Internal/CBS_RegionWGAN_GP_Screening.m`：九个筛选因素的唯一参数解析层；不复制主循环。
- `Variants/CBS_RegionWGAN_GP_E0_Base.m` 至 `CBS_RegionWGAN_GP_E8_GlobalCritic.m`：可被 GUI 发现的顺序筛选薄入口；其中 E5b 只在 split gate 已晋级后重新检查保留未配对锚点。
- `Support/CBS_CGAN_screening_protocol.m`、`run_CBS_CGAN_screening_campaign.m`、`analyze_CBS_CGAN_screening_stage.m`：冻结协议、10-worker 队列与逐级分析。
- `Tests/test_CBS_screening_branches.m`、`test_CBS_screening_campaign_support.m`：因素原语、GUI 配置、NaN 删除和顺序协议回归。

截至2026-08-28，活动目录共有50个`.m`文件：1个生产入口、10个Core文件、17个Variants GUI入口、2个Internal基类、6个Support文件和14个测试，即18个可被GUI发现的算法类。GUI递归扫描子目录；入口文件名必须等于类名，第二行必须含PlatEMO标签。`Tests/test_CBS_platemo_compliance.m`动态发现并短程运行全部CBS GUI入口。共享核心公开只读`guideExperimentSnapshot`，正式筛选结果还把同一快照保存为`metric.CBSAudit`；不保存逐个候选大数组。

根部旧`run_bt*.m`、`test_arm_*.m`、`test_operator_branches.m`已删除；2026-08-27又删除`.DS_Store`和已标记`Superseded`、无调用者的`Support/run_CBS_CGAN_factor_experiment.m`。根部`test.m`是唯一通用批量启动脚本；其当前算法、问题、runs、workers、`platemo(...)`调用和自动保存均以文件自身为准，不再由旧三臂协议反向定义。当前本地Git只有`UC-GAN-2`分支；本次没有删除、切换或修改任何本地/远端Git ref。

## 6. 正式运行协议

- 生产入口是 `Support/run_CBS_RegionWGAN_GP_mainline.m`，固定 `maxFE=200000`，允许 1 个验证 worker 或 10 个正式 worker；默认 run IDs 为 `1:10`。
- 根目录`test.m`当前启动顺序筛选 campaign；冻结问题为`DASCMOP1/5/9_BC`、`LIRCMOP10/14_BC`，runs`1:5`、`N=100`、`maxFE=200000`、`save=20`、10个process workers。
- 队列机制保持项目通用语义：路径隔离、清理旧 Jobs、`problem×run`扁平任务、固定`SubrangeSize=1`、每任务一次`platemo(...)`、其内部`rng('shuffle')`与按算法类名自动保存。run ID 是结果编号，不是seed；统计按独立重复运行处理。
- 筛选 runner 只在原生 MAT 同时含终点`FE>=200000`、`metric.IGD`和`metric.CBSAudit`时跳过；失败任务最多重试3次。首次启动冻结当前源码和`test.m`，逐级保存状态、CSV/MAT分析和失败日志。
- 生产runner的原生输出目录格式必须是`<campaignRoot>/Data/CBS_RegionWGAN_GP`；`Data/`不会自动进入Git。仓库现有`Data/CBS_RegionWGAN_GP`属于旧A0语义，不能被新主线resume、覆盖或混算；新主线必须使用全新campaign root。

## 7. 2026-08-11—28 清理、实验收敛与验证

- 2026-08-27 清理 `Data/`：95 个已淘汰顶层条目（6,963 个文件，约 1.48 GiB）移入 macOS 废纸篓；删除范围仅保留为已撤销源码对应的旧 CBS 分支结果、早期 BT/算子筛选产物和已结束任务日志。曾被误移的 `CCMO/CMOEAD/CMOEMT/CPCMO/DRMCMO/IMTCMO/MCCMO/MTCMO/NSGAII/PPS/ToP` 对比算法目录已完整恢复；A0/A1/A2、当前四个外部对照、最终 BT0-vs-NoBT 凭证和本节明确列出的失败实验冻结包均保留。
- 2026-08-19的精简曾把主类降为523行；当前共享核心同时承载“500→200生成＋新利用”生产主线、历史A1/A2和三个名额消融入口，不恢复旧NoCGAN/PBest/BT公式分支，也不改变环境选择或后半程认证逻辑。
- 当前固定指纹为全W均衡500候选、同条件critic保留200、A/h/T利用前半程 + 原strict boundary-target后半程 + shared Union/raw-objective SPEA2，问题为LIRCMOP6_BC，`N=100,D=30,maxFE=20000,rng=4242`，运行环境为MATLAB默认计算线程：
  - IGD `1.3473816458691643`
  - 决策和 `1828.5991360038395`
  - 目标和 `625.98054078737562`
  - 约束和 `0`
  - RNG state 首值 `274547642`
- 两个失败的 15 题环境选择 campaign 均保留冻结源码、协议、runner、分析器和 75/75 候选原生结果：`Data/CBS_env_selection_nrbt_runs101_105/` 和 `Data/CBS_env_selection_normalized_truncation_runs101_105/`。NRBT 保留基线结果与候选源码 SHA-256；归一化截断另保留完整输入/输出 SHA-256，其 75 个候选结果冻结在 `candidate_results_sha256.txt`。
- 归一化截断结果经独立原生文件复算：75 对终点 FE 均为 200000，150 个终态种群可行率均为 1，候选/基线非有限 IGD 与空可行集均为 0；删任一 seed 或任一问题仍不可能通过 gate。PF-GD 使用 `GetOptimum(10000)` 的平台返回值，实际点数依问题而异，因此只作为已通过的收敛保护，不承担拒绝结论；拒绝由 IGD、风险、CF 与覆盖 gate 独立共同确定。
- CAAT 证据保留在 `Data/CBS_env_selection_caat_runs106_110/`：冻结原始/诊断基线/候选三套源码、预注册协议、runner、因子测试、分析器、150/150 个新执行诊断任务和逐文件 SHA-256。诊断基线 75/75 与权威 runs `106:110` 结果在 FE、指标、决策、目标和约束上逐元素一致；候选 75/75 为 0 failure、0 empty、终点 FE 全为 200000，结果 manifest SHA-256 为 `db80d73a1144be08478d90524b65b219b98cf6b9ba4fc2c53e4c7a3365e94122`。该实验因子测试只属于冻结证据，不计入当前活动测试入口。
- 后续失败实验分别冻结在 `Data/CBS_env_selection_e1_runs106_110/`、`Data/CBS_env_selection_e2_battery/`、`Data/CBS_certified_bracket_memory_runs101_105/`、`Data/CBS_fresh_boundary_targets_runs101_105/`、`Data/CBS_sparse_gap_unique_pairs_runs101_105/` 和 `Data/CBS_final_guidefix_runs106_110/`。其中 E2 fresh-seed 两臂按冻结 red-line 提前取消，不得表述为完整 375 项 campaign。
- CF6_BC 与 DASCMOP9_BC 的独立小预算基线也完成 FE、全保存种群、IGD 和 RNG 逐元素比较。
- 2026-08-28主线与筛选改造后的全部原有CBS回归、两个新增筛选测试和固定20K指纹均通过；核心、边界记忆、WGAN、筛选参数层、协议和分析器的MATLAB Code Analyzer为0 issue。GUI契约会动态发现18个活动入口并排除Core/Internal基类。
- WGAN的训练损失、GP、网络结构、优化器和训练集配对规则均未改；同条件critic前向评分现由生产主线及历史A1/A2使用。旧A2结果只能作为同机制历史证据，不能代替当前源码身份下的新主线实验。

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
- 同条件 critic 百分位只解决“原始 critic 分数不能跨条件比较”，不证明 critic 已校准或保留候选更靠近边界；必须比较保留/拒绝池的边界距离、方向覆盖、最终20个候选及真实子代的存活和边界贡献，不能按原始候选是否可行解释。
- A/h/T 的 `h` 只限制变异前中心。若实际子代步长显著大于中心步长，应先检查 Polynomial Mutation，而不能把它误判为局部尺度实现失效。
- 真实 bracket 直接写入 `BMem`、fresh-only target、环境选择 maximin fill、认证点保护及其 E2 组合均已有负面冻结证据；这些机制不能再作为“未试过”的候选。它们与本轮只用于 CGAN 候选选取的 T-space maximin 不是同一机制。
- 当前证据只支持保留 boundary transfer，不支持声称已超过 DRMCMO。
- 四类早期纯生存选择改造与 E1/E2 constrained-selection package 均未形成可推广的 IGD 与机制收益。CAAT 虽在 56/75 次运行中改变 Pop2 survivor mask，并显著减少部分问题的跨种群精确重复，但 74/75 次运行没有增加任何辅助新颖方向，输出覆盖也没有净增；去重没有转化为有效 PF 覆盖。事后机制诊断中，LIR+CF 的 Pop2 可行率有 32/50 对下降、5/50 上升，平均下降 9.78 个百分点，说明 raw anchor 排斥还会清除靠近可行 Pop1 的辅助桥接点；该诊断不参与预注册 gate。
- 截断只能从已经产生的候选中删除个体，不能生成 Union 中不存在的 PF 方向。若继续“增强多样性以改善 IGD”，必须先为候选产生或交配阶段建立新的、可验证的缺失方向来源假设；后续不再启动新的纯环境选择分支，也不能把新的截断指标包装成候选来源。
- 后续任何机制改动都必须以当前主线为唯一对照，使用新目录、冻结源码、配对种子和预注册门槛；不得因单题、旧版本或初筛信号复活已否决分支。
- 本轮已接受“500→200生成＋新利用”带来的前半程轨迹与指纹变化；后续不得再改变GA/DE数量舍入、评价顺序、主线500→200候选池、局部目标利用或后半程三组`OperatorDE`调用顺序，除非先建立新的可复现实验协议并重新冻结指纹。

## 10. 当前 CGAN 顺序筛选 campaign

### 10.1 冻结矩阵与判定

- 问题：`DASCMOP1_BC`、`DASCMOP5_BC`、`DASCMOP9_BC`、`LIRCMOP10_BC`、`LIRCMOP14_BC`。
- 每个活动分支运行5次，`N=100`、`maxFE=200000`、10个process workers；原生`save=20`同时给出约10K间隔轨迹、100K检查点和200K终点。
- 所有均值、几何比和独立`ranksum`先删除`NaN/Inf`；相同run编号不视为共同seed。
- 候选只有在任务完整、200K跨题几何均值比`<=0.98`、实用胜题数多于负题数，且DAS/LIR家族比都`<=1.02`时晋级。±2%只定义实用win/tie/loss；p值报告但不作为5-run初筛的唯一门槛。

### 10.2 一次只改变一个因素

| 阶段 | 唯一变化 | 执行条件 |
|---|---|---|
| E1 | 丢弃无配对真可行锚点 vs 保留为`pairflag=1`并跨代携带 | 必跑，先形成结论 |
| E2 | 在E1胜者上比较最近5、最近10、全部参考方向配对 | 必跑；始终保留“可行锚点目标支配不可行点则禁止配对” |
| E3 | 前两层可行前沿 vs 全部真可行解 | 至少2题的`frontDropRate>=0.20` |
| E4 | 每方向上限5→10→不限 | 至少2题的`capDropRate>=0.10`；CapAll只在Cap10晋级且瓶颈仍在时运行 |
| E5 | 总行数门槛 vs `总数32 + 正16 + 负8 + 条件4` | 至少2题的unsafe训练事件率`>=0.10` |
| E5b | split gate晋级后，再单独复查保留无配对锚点 | 仅当前仍为drop时运行，避免一次改两个因素 |
| E6 | WGAN uniform mini-batch vs 正负`pairflag`均衡mini-batch | 至少2题的不均衡训练事件率`>=0.20` |
| E7 | 只用当前Pop1可行父代 vs 局部尺度不可用时补充`BMem.x_b/y_b` | 至少2题父代回退率`>=0.10`；历史目标不重评价 |
| E8 | 旧20槽×5候选 vs 全W均衡500条件并按同条件critic保留200 | 在最终已选训练/利用配置上必跑 |

E3—E7的机制触发为“该因素有机会产生行为差异”的资源门槛，不是性能结论；未触发会明确记录为skipped。禁止伪造可行标签、删除目标支配排除条件、跨代保留旧不可行端点或把已否决的认证bracket直接写入BMem。

### 10.3 同时观测机制链而不增加FE

`metric.CBSAudit`保存：首次找到种群可行解/合格锚点/合法配对/首次训练/候选池/引导使用/有用引导的FE；前沿和容量丢弃率；K5/K10/全方向合法配对机会与实际配对方向秩；正负训练行、条件覆盖、训练激活与不均衡率；生成数、critic保留率、条件覆盖和真实方向命中；无池/无父代/无局部尺度/映射回退；内存父代使用；目标、T中心、最终子代的改善与存活链；以及5/10/20/30/40/50/75/100% FE处的Pop1可行数、方向覆盖和累计训练/引导数。旧字段中的原始生成解oracle可行率只作为冻结历史数据保留，后续不得用于筛选、分组或判定生成质量。审计评价在恢复FE和RNG后返回，不进入Union，不改变正式评价预算。

结果、阶段winner和审计表写入`Data/CBS_CGAN_sequential_screening_runs1_5/`；每个算法分支的原生结果仍写入独立`Data/<类名>/`，避免与历史主线/A0/A1/A2混算。

### 10.4 2026-08-28 正式结果与裁决

正式队列从2026-08-28 02:26运行至10:26。实际执行6个分支×5题×5次，共150个`problem×run`任务；150/150完成，0 failure。独立复核确认每个MAT均包含20个保存点、终点`FE=200000`、有限终点IGD和至少183个`CBSAudit`字段。这里的E0是**本次500→200迁移前的旧100候选筛选基线**，不是历史“旧生成＋旧利用”的A0类，也不再代表当前生产语义。

| 阶段/候选 | 100K G | 200K G | 200K W/T/L | 裁决 |
|---|---:|---:|---:|---|
| E1 保留无配对锚点 | 0.9558168814 | 1.0253931187 | 0/2/3 | 拒绝；保留drop |
| E2 最近10方向 | 1.2710961644 | 1.0219873004 | 0/3/2 | 拒绝 |
| E2 全方向 | 0.9923177128 | 1.0031496243 | 1/2/2 | 拒绝；保留K=5 |
| E4 每方向上限10 | 0.9182065469 | 1.0006589150 | 1/2/2 | 拒绝；保留cap=5 |
| E8 全W 500条件＋critic | 1.6132063189 | 1.0418458711 | 0/4/1 | 当时拒绝；本次用户迁移已覆盖该裁决 |

`G`为候选相对当时控制的跨题几何均值IGD比。表中“100K”来自第一个保存的`FE>=100000`点，实际约为109.5K—109.9K，并不是最后一批CGAN候选完成环境选择后的精确截止点；200K则混入了后半程机制。因此这些结果保留为历史轨迹与安全证据，但不再单独裁决前半程CGAN生成机制。E8在DASCMOP1_BC终点退化`22.65%`、独立`ranksum p=0.031746`仍提示必须保留问题级安全检查。

条件阶段严格按E0审计触发：

| 触发字段 | 阈值 | 达阈值问题数/要求 | 结果 |
|---|---:|---:|---|
| `frontDropRate` | 0.20 | 1/2 | E3跳过 |
| `capDropRate` | 0.10 | 5/2 | E4执行 |
| `unsafeTrainingEventRate` | 0.10 | 0/2 | E5、E5b跳过 |
| `imbalancedTrainingEventRate` | 0.20 | 0/2 | E6跳过 |
| `parentFallbackRate` | 0.10 | 0/2 | E7跳过 |

E0五题平均`frontDropRate=0.156995`、`capDropRate=0.597736`。容量确实大量截断锚点，但Cap10把平均保留锚点从`90.03`增到`174.50`后，引导存活率反而从`2.096%`降到`1.967%`，最终总体IGD仍为中性。容量不是有效引导的决定性瓶颈。

机制链进一步给出以下归因：

- E1只产生`2.80%`无配对正样本；训练激活率从`88.39%`升到`91.22%`、引导可行率从`26.06%`升到`26.98%`，但终点`G=1.02539`。更多训练事件本身不等于更好的最终前沿。
- K10把配对率从`94.40%`升到`96.24%`，并把`childUsefulRate`从`0.1135%`升到`0.1441%`，但终点仍退化；局部机制改善没有稳定转化为跨题IGD。
- E8每次训练事件由约100个原始候选扩大到500个，并只保留40%；原始oracle可行率从`23.84%`升到`26.36%`，但真实参考方向命中率从`28.42%`降到`23.48%`，引导存活率从`2.096%`降到`1.666%`，`childUsefulRate`从`0.1135%`降到`0.0785%`。critic扩大并筛选候选后没有改善最终可用性。

旧顺序筛选最终配置为`[keep=0,K=5,frontDepth=2,cap=5,gate=total,batch=uniform,parent=population,generation=legacy,audit=1]`，即参数向量`[0 5 2 5 0 0 0 0 1]`。本次迁移只把生产入口的`generation`切到`global_critic`并保持审计关闭，其余训练、利用和后半程因素不变；冻结E0–E8类及其结果不改名、不覆盖。未触发的E5—E7仍不能表述为已经证明有效。

### 10.5 本次迁移后的重新裁决原则

- 前半程主性能点定义为：最后一批CGAN候选形成真实子代并完成环境选择后的Population1；不能使用第一个粗粒度`save=20`的`FE>=100K`点替代。该精确快照需新增`IGD/HV`，现有50%审计只保存可行数、方向覆盖和累计机制计数。
- 原始500个生成解、critic保留200个和最终选出的20个候选均不得按oracle可行/不可行筛选、分组或判优。pairflag仍用于训练边界两侧结构，但生成后的随机落点允许位于任一侧。
- 生成质量改用不区分边界侧的距离：对每个候选测量其到已知可行—不可行括号的归一化边界距离，并分别报告原始池、critic保留池和最终20个候选的中位数、P90、边界带命中率、方向覆盖与重复率。
- critic是否有价值由“保留池是否比拒绝池更靠近边界”及critic百分位与负边界距离的相关性判断；利用是否有价值由映射成功率、G→T→实际子代的边界距离变化、Pop1/Pop2存活、新边界括号产出和精确截止IGD/HV判断。
- 200K最终IGD继续作为持久性与安全性指标，但不能单独否决或确认只在前50% FE工作的CGAN机制。

冻结结果位于`Data/CBS_CGAN_sequential_screening_runs1_5/`；逐阶段IGD在`analysis/E*_candidates.csv`和`E*_problems.csv`，机制汇总在`analysis/audit_by_problem.csv`、`audit_overall.csv`和`audit_trigger_decisions.csv`，首次源码快照位于`source_snapshot/`。
