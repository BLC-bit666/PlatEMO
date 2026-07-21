# CBS RegionWGAN-GP 固定主线约定

更新时间：2026-07-20

## 1. 当前唯一结论

- 唯一算法：`CBS_RegionWGAN_GP`。
- **主线配置（2026-07-20 模块融合定型并已固化为类默认）：
  `operatorMode = ga_de_half` + `boundarySearch = on` +
  `guideMode = on`（guideShare=0.2, guideCarve=sym,
  guideWindow=half）+ `blsWindow = full` + `blsFeed = on` +
  `scoutMode = off`。**
  模块叙事（BC-CGAN，边界校准条件生成）：算法维护参考方向条件化
  的边界记忆（BMem）；普通进化子代提供**被动边界观测**，
  `RefineBoundaryObservations_RC`（原 BLS，已模块化改名）提供
  **主动边界观测**（可行性反馈校准 + 边界覆盖补全，全程运行、
  每代 ≤20 FE、候选正常参选并同代回流收割）；两类观测统一入
  BMem 训练条件生成器，生成器经免评估 guide 牵引 DE 外扩。
  论文保留一句区间收缩原理引用（Michalewicz；GECCO 2007）。
  诚实边界备忘：融合的实测定位是"无严重恶化"（对 GD20 终值
  15/15、最差单题 L10 +0.001、L13/D9 微赚、L9 免疫保持、
  校准回流未产生可测的 guide 质量提升）——论文只可写闭环
  稳定运行且无损，**不可写校准强化了生成器**。
  引导机制（用户 2026-07-20 裁决，取代 07-19 的保护注入主线）：
  CGAN 输出为**免评估 guide**（不进 Union、不保护、零 FE），每事件
  20 个、6:14 有解:空方向查询（wide 分配）、缓存一代；P1 子代 =
  40 GA + 40 原生 DE + 20 引导 DE（最近可行精英 + F×(guide−精英)，
  F 轮换 {0.4,0.65,0.85}，无 guide/无可行解退化原生 DE）；guide 仅
  前 50% FE 供给（全程供给已被 R3 证明有害）。默认构造即主线；
  保护注入主线经 `('guideMode','off','scoutMode','nofrontier')`、
  旧 S2+BLS 经 `('guideMode','off','scoutMode','off',
  'parameter',{30,...})`、旧 A1 再加
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

## 2. 固定算法机制（BC-CGAN 融合主线执行逻辑图，2026-07-20）

默认路径每代执行序（与代码块序一致）：

```text
双群体 P1/P2
  -> P1 子代 = 40 SBX+PM + 40 原生 DE + 20 引导 DE
       引导 DE：a + F×(guide−a)，F 逐子代轮换 {0.4,0.65,0.85}，
       a = guide 目标方向最近的可行精英（平局取最优 fitness）；
       guide 缓冲为空或无可行解时该 20 席退化为原生 DE
  -> P2 子代 = 50 SBX+PM + 50 原生 DE（S2 配方原样）
  -> 主动边界观测（全程，每代 ≤20 FE）：
       RefineBoundaryObservations_RC =
       可行性反馈校准（可行-不可行区间收缩，≤3 锚点 × ≤4 步）
       + 边界覆盖补全（最稀疏相邻对中点 + 一步收缩修复）；
       候选为真实评价解，正常参加环境选择
  -> 前 50% FE（学习窗口）：
       BMem 收割 [P1, O1+校准候选, P2, O2]（同代回流，blsFeed=on）
       -> TrainX=BMem.x_b / TrainC / QueryRefs
       -> 合格事件（≥32 行）warm-start 训练 WGAN-GP
       -> wide 分配（6:14 有解:空方向，空池逐级回流）生成
          20 个免评估 guide（不进 Union、不保护、缓存一代，
          100k 后清空）
  -> 两次环境选择（Union = [P1,P2,O1,O2,校准候选]，无任何保护）
```

每代 FE（N=100 稳态）：200(子代) + ≤20(校准) = 220 全程恒定；
guide 零评估成本；全程受 remainingFE 钳制。

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
| `nGen` | 20 |
| `frontDepth` | 2 |
| `pairNeighborRefRadius` | 2 |
| `maxAnchorsPerRef` | 5 |
| `minGANTrainCount` | 32 |
| `ganStopFraction` | 0.5 |

`nGen`、`zDim`、`ganIter`、`ganMiniBatch`、`nCritic`、
`minGANTrainCount` 和 `sampleSigma` 遵循 PlatEMO `ParameterSet` 约定；正式
runner 不传自定义值。`ganStopFraction` 不是公开实验开关。

边界记忆持久保存 `ref/gap/x_b/y_b/x_i`（`x_i` 为跳跃研究引入并保留，
主线训练仍固定 `TrainX=BMem.x_b`）。不可行伙伴参与当次合法配对与 gap
过滤。重复锚点行作为训练权重保留。少于
32 个有效训练行时不训练、不采样、不复用旧模型。

主线查询用 wide 分配：`round(0.7*nGen)` 给空方向（一跳/两跳池按
2:1 细分，空池逐级回流），其余给有解方向；全空回退 populated。
legacy（1/6 一跳）、scout（2/3 两跳）、half（1/2 一跳）保留为
消融设施。

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
- `Algorithms/Multi-objective optimization/CBS-CGAN/RefineBoundaryObservations_RC.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/Support/run_CBS_RegionWGAN_GP_mainline.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/Support/run_CBS_scout_pilot.m`
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

2026-07-20 融合主线复测（战役前梳理）：profiler（LIRCMOP6_BC 20k 指纹
条件）显示深度学习内核占总 wall ≈82%（`updateCriticBatch` TotalTime
64.9/79.6 s），自耗时前 30 名全部为 dlarray/dlnetwork 框架内部函数，
算法自有函数（选择/收割/引导繁殖/校准）单项均 <1%——位保真约束下
无值得动的非内核热点，依据"只优化实测瓶颈"纪律本轮零代码改动。
无 profiler 三次 wall：`49.81/47.62/43.89 s`，中位 `47.62 s`（该基线
属融合主线，与旧 S2+BLS 主线的 27.07 s 基线不可比）。机器为
10 物理核：战役 10 workers 已与硬件匹配，12 workers 属超订阅无真实
增益。战役 ETA：融合型任务实测均值 1421 s，230 任务 ≈ 9.1 小时。
清理决议（2026-07-20）：全部实验开关面（operatorMode/scoutMode/
generatorMode/guideMode 变体、分配模式等）保留——默认路径零运行
成本，且为五级指纹可达性、14 项回归测试与历史臂复现的承重结构；
删除清单为空。

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

五级指纹（由 `test_CBS_operator_modes` 守护，条件均为
`LIRCMOP6_BC, N=100, D=30, maxFE=20000, rng(4242,'twister')`）：

- **主线默认路径（模块融合：GD20 + 校准全程 + 同代回流）：
  IGD=`1.3484288502689699`、决策和=`1885.893899968843`、
  终态 RNG 首字=`3538802550`；**
- 融合前 GD20 路径（显式 `blsWindow late` + `blsFeed off`）：
  IGD=`1.3490149458128178`、决策和=`1881.2224281584283`、
  终态 RNG 首字=`2349263053`；
- 保护注入路径（`guideMode off` + `scoutMode nofrontier`）：
  IGD=`1.3471946005101978`、决策和=`1910.3749679275759`、
  终态 RNG 首字=`2297882816`；
- 旧 S2+BLS 路径（`guideMode off` + `scoutMode off` + nGen=30）：
  IGD=`1.3470807122527642`、决策和=`1919.0968011138757`、
  终态 RNG 首字=`368524342`；
- 遗留路径（再加 `de`/`off`）：IGD=`1.3653447524657205`、
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
- 2026-07-19 后续机制研究（scout_pilot_v1 / jump_pilot_v1，
  10 题 × 3 seeds，指标与检查点齐全）：
  - 侧向侦察（frontier 2/3 + 两跳 + 保护）三门未过：空白方向被
    DE/GA 以同速自然填满（覆盖曲线与 A0 重合），机制冗余；
  - 保护注入本身守护题非劣、终值与 A0 同档（SCNF vs A0 终值
    14/16，逐题差均在 margin 内）；用户裁决将其并入主线
    （见第 1 节），事件规模 30→20；
  - 平凡跳跃 TJ（着陆行直采）对 A0 各检查点均微劣，跳跃后代
    晋升率 0.13%；学习式跳跃 JX 终判（JX50 被用户取消）：
    对 TJ 温和占优（终值 17/13，着陆可行率 23% vs 16%——
    学习本身有效），但对 A0 各检查点 7/11、@100k 6/12、
    终值 6/12，L9 无救援——跳跃机制整体未过 G1。
    四轮因果结论一致：在 D=30、200k、S2+BLS 骨干上，
    额外 FE 的生成式注入（复读/侧向/公式跳/学习跳）均无
    IGD 增益；生成器每轮都胜过其平凡对照（学习真实），
    缺的是生态位。主线生成器维持 wgan 复读型
    （JX vs SCNF 终值 12/18，无理由切换）。
- 2026-07-20 引导父代研究（guide_ablation_v1，10 题 × 3 seeds，
  R1 四臂 + R2 剂量/划分 + R3 窗口，用户授权夜间自主执行）：
  - 机制：CGAN 输出为免评估 guide（不进 Union、无保护、零 FE），
    P1 引导子代 = 最近可行精英 + F×(guide−精英)，F∈{0.4,0.65,0.85}；
    查询 6:14（wide 分配）。
  - **胜者配置 GDA20：50 GA + 30 原生 DE + 20 引导 DE
    （guideShare=0.2, guideCarve=de, guideWindow=half）。**
  - 消融证据：GDA20 vs MIX（同配比无 guide）终值 20/10、
    vs CN（copynoise guide）18/12、vs 对称划分 GD30 18/12；
    vs A0（=拆模块后的 50/50）整体持平（@100k 16/14、终值 14/16）。
  - L9 坏吸引子：guide 系三臂 9/9 全逃逸（A0 2/3、MIX 1/3、
    CN 2/3；GDW 破例 1 坏）——(2/3)^9≈2.6%，最强因果案例。
  - per-F：入选率 0.4>0.65>0.85（2.76/1.66/1.41%）；GDA20 引导
    子代入选率 2.21%（前几轮机制为 0.13–0.19%）。
  - R3：guide 全程供给有害（终值 vs GDA20 9/21，墙钟×2），
    半程窗口定案；GD40 按 MIX 定价证据跳过（偏离预注册已记录）。
  - **终裁（2026-07-20，用户确认）：GD20（40/40/20 对称划分）
    为新主线**——终值两两矩阵全场不败（胜其余五臂、对 A0 平
    15/15），L9 3/3 逃逸且数值最好；GDA20 赢前程、GD20 赢终程。
    已固化为类默认、四级指纹录制、入口守卫升级、13/13 测试通过。
  - 严格对照缺口（记录在案）：GD20 的同配比匹配对照（40/40/20
    无 guide 的 MIX-B、copynoise guide 的 CN-B）尚未运行——现有
    MIX/CN 均为 35/35/30 配比；论文消融表定稿前需补这两个便宜臂。
- 2026-07-20 BLS 融合实验（BFF 臂 = GD20 + BLS 全程 + 二分迭代点
  同代直喂 BMem，`blsWindow/blsFeed` 开关，四级指纹验证块前移
  RNG 零扰动）：**不过门**——L10 点名门失守（2/3 seed 劣化，
  中位 +16%），D5/L5 @50k 穿带减速哨兵触发（前半程 BLS 抢占
  穿带预算的旧设计论证首次获实验确认），直喂通道未产生可测的
  guide 质量提升（gd_feas/gd_p1 持平略降），终值仅追平（15/15）。
  附带收获：L13 −0.0016、D9 −0.0009 的小改善。
  后续裁决（2026-07-20，用户确认）：**采用做法二**——BFF 配置
  （校准全程 + 同代回流）晋级主线，标准为"不严重恶化"
  （终值 15/15 满足）；结构统一（v1）完成：校准函数迁出主类为
  `RefineBoundaryObservations_RC`、`OffspringL→CalibrationCandidates`、
  模块语言注释与引用入册，五级指纹逐位验证零行为改动；
  修复一处份额取整边界 bug（count=1 时 gaCount=0 空父代越界，
  仅影响微型 FE 烟测场景）。
  **v2（校准取材记忆化 + BF-iso 隔离对照）经分析后由用户决定
  完全放弃**——理由：BMem 配对按目标空间配对，决策空间线段
  可能变长损害二分精度（正砸 L10/D9 点名门）；后半程记忆冻结
  需搭车第二变更，违反单变量纪律。
- 2026-07-20 战役就绪（用户将自行启动）：两个**独立消融算法类**
  `CBS_RegionWGAN_GP_A00`（模块整删：无记忆/无 guide/无校准，
  50/50 骨干、每代 200 FE，nGen=0 钉死）与
  `CBS_RegionWGAN_GP_CNB`（学习对照：仅生成器换 copynoise，
  其余与主线逐字节一致）——两者均为主线子类、追加钉死开关实现，
  20k 位等价测试证明与手工开关孪生逐位一致。
  正式 runner 泛化 `Options.algorithm`（默认主线，原生保存机制
  按类名自动生成 `Data/<类名>/<类名>_<题>_M_D_run.mat`）并改
  `save=2`（metric.IGD 含 ~100k 与终值两条）。
  启动脚本 `run_CBS_RegionWGAN_GP_DAS_LIR_full.m` 重写为三阶段
  （A00 → CNB → 主线，便宜臂先行趟雷），三算法探针守卫 + 干跑
  验证；15/15 测试通过。预计 A00 ≈ 0.5h、CNB ≈ 1h、主线 ≈ 9h。
- 待执行：用户启动三阶段战役后出论文主表与消融表；分析时
  L1–4 剔除。

## 2026-07-21 全量战役结果与 MW 换域 pilot（假设被否决）

- **三阶段战役完成**（23 题 × 10 seeds × 3 算法，200k FE，save=2）。
  终点 IGD 秩和检验（对主线）：A00 = 2/3/18（主线仅在 L9/L10/L12
  显著赢，D2/L13 反输）；CNB = 1/0/22（学习归因为零，全部贡献
  属于结构）。@100k 复算：A00 = 1/3/19，CNB = 0/0/23。逐 seed
  证据：L9 逃逸 6/10（A00）vs 8/10（主线）vs 9/10（CNB），坏吸
  引子 ≈0.080；L10 终点分布近乎不重叠（主线 0.0061–0.0083 vs
  A00 0.0077–0.0095）。结论与 pilot 一致：模块贡献 = 小而局部，
  且全部归属结构（BMem+BLS+引导），学习份额为零。
- **MW_BC 实现审查通过**（14/14 diff 仅类名+折叠行，数值等价
  500 点逐位验证，随机采样不可行率 100% 确认狭缝几何）。
- **MW 换域 pilot（A00 vs 主线，3 seeds，200k）否决"MW 是模块
  主场"假设**：终点均值 5% 阈值下主线赢 2（MW3/MW10）、A00 赢
  5（MW2/5/6/8/13，MW13 差 −43%）、持平 7；几乎所有题 100k 已
  收敛。裸骨干在狭缝题上已达发表级水平（MW1 0.0016、MW3
  0.0050），无亏空可填；模块在 MW 上是轻微净负（BLS 全程
  ~10% FE 开销 + 无 guide 期 40/60 偏离 + 引导误导）。
- **发现归因漏洞（用户点破）**：sym 分割名额先验固定 40/40/20，
  无 guide 时 20 个引导名额降级为普通 DE → 主线在"训练前 + 100k
  后"实际为 40GA/60DE，而 A00 全程 50/50。函数注释"degrades to
  the S2 mainline"不准确（S2 是 50/50）。后半程消融差异混杂算子
  比例效应，L10/L12 的赢与 D2/L13 的输均受污染。
- 待决（用户选择）：A00-4060 对照臂（guideMode=on + nGen=0 +
  boundarySearch=off，零代码改动量化比例混杂）；M−BLS 臂（BLS
  边际/删除决策）；主线 200k 尺度重调参（含降级分支 1:1 化）；
  机制升级（_BC 二值反馈下用模块重建不可行解排序——用户暂缓）。

## 2026-07-21 A4060 算子比例对照臂（比例混杂分解完成）

- 新增诊断类 `CBS_RegionWGAN_GP_A4060`（模块全关如 A00，但
  guideMode=on → guide 永空 → 全程 40/60 降级构成；位等价+契约
  测试 `test_CBS_a4060_arm` 通过）。7 题 DAS/LIR × 10 seeds +
  MW6/13 × 3 seeds，结果在 `decomp_pilot_v1/Data/`。
- **归因分解（三臂 A00→A4060→主线，200k 终点秩和）**：
  - L9、L10 的主线赢**在匹配骨干后依然显著**（MAIN vs A4060 均
    +）→ 真模块效应（L9=引导逃逸，L10=疑似 BLS 渐近校准）；
  - **L12 的赢大半是比例效应**：A4060 vs A00 显著 +（0.0040 vs
    0.0047），主线对 A4060 不再显著（0.0037，ns）；
  - **L13 的输是纯比例伪影**：A4060 与主线终点完全同值
    （0.1063），显著差于 A00 的 0.1048；模块自身近乎零成本；
  - D2 的输在三臂间每步均不显著（0.0044/0.0045/0.0045 单调微
    移，尘埃级）；
  - **L5 新发现**：40/60 显著优于 50/50（0.0076 vs 0.0083），但
    模块把收益吃回（主线 0.0082 显著差于 A4060）；主线≈A00 是
    两效应相消的巧合；
  - MW（n=3 无检验力，方向参考）：MW13 A00 0.055 → A4060
    0.111 → 主线 0.096；MW6 类似——**MW 上主线的拖累主要是
    40/60 构成，不是模块 FE**。
- 结论：无 guide 降级分支 40/60 ≠ 注释宣称的"退化为 S2 主线
  (50/50)"，该偏离是 D2/L13/MW 反输的主因，同时意外贡献了
  L12"伪赢"。**1:1 化降级分支**预计使 DAS/LIR 消融表变为
  ≈2 赢（L9/L10）/0 输/21 平，且获得"模块失活 ≡ A00 位等价"
  的验证性质；代价：主线重定义、指纹重建、主线战役臂重跑
  （≈9h）。待用户决策；BLS 删除问题（M−BLS 臂）在其后。

## 2026-07-21 后半段 BLS 配对分支实验（branch100k）

- 新增 `initDecs` 注入参数（默认空=行为零变化；指纹回归 + A00/
  A4060/CNB 位等价 + 专项守卫 `test_CBS_initdecs` 全过，冒烟 IGD
  与改动前逐位相同）。新增访问器 `effectiveInitDecs`。
- 设计：从主线战役每个 run 的 ~100k 快照重启（P1/P2 均以快照
  decs 注入——P2 未保存，属文档化近似；预算 = 200000−snapFE+2N
  补偿注入评估；同 seed 配对两臂：bls_on = 主线后半段动力学
  （空 guide、40/60、无训练、每代 20 FE 校准）vs bls_off = 纯
  40/60 骨干）。驱动 `Support/run_CBS_branch100k_pilot.m`，
  7 题 × 10 seeds × 2 臂 = 140 run，数据 `branch_pilot_v1/`。
- **锚定检验全过**（bls_on 分支 vs 真实主线终点，signrank 全部
  p≥0.16，均值几乎重合）→ 重启近似可信。
- **后半段 BLS 结论**（配对 signrank）：
  - **L5 显著有效**：d=−0.00046，9/10 seeds，p=0.006；
  - **L10 显著有效**：d=−0.00109，8/10 seeds，p=0.037——L10 主线
    赢的来源坐实为后半段边界钉扎校准；
  - D2/D4/L12/L13 中性（后半段 9% FE 税在饱和段不转化为损失）；
  - L9 无显著后半段效应（其优势在前半段引导窗口，与逃逸机制
    一致）。
- 归因拼图更新：L5 全程拖累（主线<A4060）来自**前半段**模块活
  动，后半段 BLS 实为正贡献；L13 反输再证为纯比例伪影。
- 对删除 BLS 决策的含义：**后半段 BLS 应保留**（2 显著赢/0 显著
  输）；前半段 BLS 边际仍未测（需 M−BLS 全程臂或 branch-from-0）。
