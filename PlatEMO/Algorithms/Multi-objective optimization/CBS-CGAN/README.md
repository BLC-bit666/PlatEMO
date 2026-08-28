# CBS-CGAN 生产主线与候选生成对照

唯一生产 GUI 入口是 `CBS_RegionWGAN_GP`。自 2026-08-28 本次迁移起，主线固定为“全参考方向均衡生成＋同条件 critic 筛选＋新利用”：每次训练生成 500 个原始候选、按条件内 critic 百分位保留 200 个，再统一进入 `A/h/T` 局部目标映射和 T-space 全局选择。共享算法引擎位于 `Core/CBS_RegionWGAN_GP_Core.m`；历史 A2 的默认行为与当前生产主线相同，但仍以独立类名和 `arm=2`保留已有结果身份。A0/A1 的旧语义和入口仅为历史兼容；Random20、DE20、GA20 是名额消融。所有入口后半程均使用真实认证的可行边界端点，不叠加横向 donor 差分。

机制快照使用`arm=-1`标识生产主线；冻结历史A0/A1/A2仍使用`0/1/2`。

完整开发裁决和实验数字见项目根目录 `Agent.md`。

## 代码结构与核心模块

```text
CBS-CGAN/
├── CBS_RegionWGAN_GP.m          生产 GUI 入口
├── Core/                        唯一共享算法实现
├── Variants/                    每个实验分支一个 GUI 入口类
├── Support/                     runner、分析器、协议与路径隔离
├── Tests/                       回归和 GUI 契约测试
└── README.md
```

核心只分四组：

- **搜索引擎**：`CBS_RegionWGAN_GP_Core.m`，持有双种群主循环、FE 分配、CGAN/boundary-target 切换、环境选择和机制统计。
- **边界与 CGAN**：`UpdateBoundaryMemory_RC.m`、`BuildBoundaryDataset_RC.m`、`BoundaryWGAN_RC.m`、`RunRegionGAN_RC.m`、`RefineBoundaryObservations_RC.m`。
- **进化算子与选择**：`OperatorDEDistinct_CBS.m`、`AssignReferenceVectors_CBS.m`、`CalFitness_CBS.m`、`EnvironmentalSelection_CBS.m`。
- **入口配置**：根目录生产类、`Variants/` 分支类，以及不对 GUI 暴露的 `Variants/Internal/CBS_RegionWGAN_GP_Experiment.m`。

`Core` 文件第二行没有 PlatEMO 标签，因此 GUI 不会把共享基类误列为算法。生产类和每个分支只负责算法身份、GUI 元数据和配置差异。

## 新分支的 GUI 契约

PlatEMO GUI 递归扫描 `Algorithms/**`。每个后续分支只需在 `Variants/` 新建一个薄类，并同时满足：

1. 文件名与 `classdef` 名完全相同；
2. 第二行含 `% <年份> <multi> <real> <constrained>`；
3. 继承 `CBS_RegionWGAN_GP_Core`，只覆盖配置，不复制主循环或 helper；
4. 需要 GUI 参数时使用标准 `% name --- default --- description` 注释并通过 `ParameterSet` 读取；
5. 类名即结果目录名，建立正式实验后不得改名或复用旧目录。

固定配置分支的最小形式：

```matlab
classdef CBS_RegionWGAN_GP_NewBranch < CBS_RegionWGAN_GP_Core
% <2026> <multi> <real> <constrained>
% Short description of the isolated branch factor

    methods
        function Algorithm = CBS_RegionWGAN_GP_NewBranch(varargin)
            Algorithm@CBS_RegionWGAN_GP_Core(varargin{:});
        end
    end
    methods(Access = protected)
        function Config = algorithmConfiguration(Algorithm)
            Config = algorithmConfiguration@CBS_RegionWGAN_GP_Core(Algorithm);
            Config.someFixedOption = "candidate";
        end
    end
end
```

`Tests/test_CBS_platemo_compliance.m` 按 GUI 的第二行标签规则动态发现本目录全部入口，并逐个通过 `platemo(...)` 运行。新增分支无需维护硬编码测试列表。

## 算法运行语义

完整代、`N=100` 时的名义配比：

| 阶段 | 约束种群 Pop1 | 无约束种群 Pop2 |
|---|---|---|
| 主线且代首 `FE < 0.5*maxFE` | 25 GA + 55 普通 DE + 20 critic 保留候选的局部目标 DE | 25 GA + 75 普通 DE |
| Random20 且代首 `FE < 0.5*maxFE` | 25 GA + 55 普通 DE + 20 决策空间随机解 | 25 GA + 75 普通 DE |
| DE20 且代首 `FE < 0.5*maxFE` | 25 GA + 55 普通 DE + 20 普通 DE | 25 GA + 75 普通 DE |
| GA20 且代首 `FE < 0.5*maxFE` | 25 GA + 55 普通 DE + 20 GA | 25 GA + 75 普通 DE |
| 代首 `FE >= 0.5*maxFE` | 25 GA + 55 普通 DE + 20 boundary-target DE | 25 GA + 75 普通 DE |

- 无可用边界目标或同参考方向父代时，引导席位回退为普通 DE。
- 普通 DE 使用互异的 `base/r1/r2`，多项式变异前为 `base + 0.5*(r1-r2)`。
- 主线在全部 `W` 上均衡分配 500 个条件，按同条件 critic 百分位全局保留 200 个；随后使用 `A/h/T` 映射、T-space 全局 maximin 和 `F=1` 产生真实评价子代。
- 历史 A2 默认使用同一“500→200＋局部目标利用”机制，仅为旧结果和启动脚本兼容保留；它不再是当前主线的生成方式对照。
- 历史 A0 为旧生成＋旧利用，历史 A1 为新生成＋旧利用；两者语义被冻结，不能用其类名解释当前生产主线。
- 后半程认证目标仍调用 `OperatorDE(Problem,A,G,A,{1,F,1,20})`，`F` 循环 `0.4/0.65/0.85`；同参考方向缺父代时回退普通 DE。
- 每代最多使用 20 个真实 FE 做主动边界校准。校准解进入共享 `Union` 并被两套环境选择共同读取；未经评价的 CGAN 原始候选不会直接入种群。
- 两套环境选择共享 `Population1 + Population2 + Offspring1 + Offspring2 + Calibration`。Pop1 使用约束适应度，Pop2 使用无约束适应度；两者都采用 raw-objective SPEA2 近邻密度和字典序拥挤截断。
- 前半程边界记忆 `BMem` 服务 pairflag WGAN；后半程 `BoundaryTargetXf/Yf` 服务认证目标引导，两者用途不同。
- 后半程存档 newest-first、FIFO 上限 1500；目标与父代必须在联合归一化后属于同一参考方向。

### 基线训练集与顺序筛选分支

1. 每次训练事件合并当前 `Population1`、`Offspring1 + Calibration`、`Population2`、`Offspring2`，并只追加上一轮 `BMem` 的可行锚点；旧不可行端点不会跨代保留。
2. 真可行解中只保留前两层 Pareto 前沿，每个参考方向最多 5 个锚点。
3. 每个锚点必须在本方向附近的至多 5 个参考方向内找到“当前代不可行且不被该可行锚点目标支配”的候选，并取归一化目标空间最近者；找不到配对时，锚点整行丢弃。
4. 全部配对再经过全局 MAD gap 过滤。每对产生一行 `[x_b,W(ref),1]` 和一行 `[x_i,W(ref),0]`；重复行保留为训练权重。
5. 默认至少需要 32 个条件行（现有完整配对下约等于 16 对）且参考条件非空才训练。候选池在下一代使用，并且当前 Pop1 仍必须至少有一个真可行父代，否则 20% 席位回退普通 DE。

这是一条复合的早期启动门槛，而不只是“可行端点必须贴近边界”。`CBS_RegionWGAN_GP_E0_Base`冻结迁移前的旧100候选筛选基线并开启行为审计。顺序筛选分支依次隔离：保留未配对真可行锚点、配对方向数5/10/不限、前沿层数2/全部、每方向容量5/10/不限、正负样本与条件覆盖门槛、正负均衡mini-batch、边界记忆父代补充，以及旧100候选池/全W均衡500候选critic池。

所有分支继续禁止把不可行点伪标为正样本，也继续排除被可行锚点在目标上支配的不可行候选。保留分支只让无合法负伙伴的**真可行**锚点产生`pairflag=1`行；其`x_i/gap`为`NaN`并可在下一代重新尝试配对。

### 前半程名额消融分支

三个分支都只替换主线前半程的CGAN引导真实评价名额，并跳过边界记忆、WGAN训练、critic筛选和CGAN候选利用：

- `CBS_RegionWGAN_GP_Random20`：调用 `Problem.Initialization(guidedCount)`，在完整决策空间独立随机采样；
- `CBS_RegionWGAN_GP_DE20`：调用与55%普通DE完全相同的 `deOffspring`，完整代等价于25 GA + 75普通DE；
- `CBS_RegionWGAN_GP_GA20`：调用与25%GA完全相同的 `gaOffspring`，完整代等价于45 GA + 55普通DE。

完整代且 `N=100` 时替换20个名额；尾代或其他`N`继续使用主线同一`guidedCount`。校准、双种群、其余算子、共享Union、环境选择、50% cutoff和后半程认证boundary-target全部不变。三个入口均可从PlatEMO GUI直接运行，结果分别写入同名`Data/`目录。

## 运行

从仓库根目录直接运行：

```matlab
repoRoot = pwd;
addpath(genpath(fullfile(repoRoot,'Algorithms','Multi-objective optimization','CBS-CGAN')));
addCBSPaths(repoRoot);
rng(1,'twister');
Problem = LIRCMOP5_BC('N',100,'D',30,'maxFE',200000);
Algorithm = CBS_RegionWGAN_GP('save',2,'metName',{'IGD'});
Algorithm.Solve(Problem);
```

`platemo(...)`会执行`rng('shuffle')`并递归加入仓库目录。共享核心在进入主循环时再次调用活动路径隔离，移除`Data/**`冻结快照，因此GUI、直接`Solve`和批量`platemo(...)`均使用当前活动实现。

## 参数

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `rawGuideCount` | 500 | 主线每次训练事件的全W原始查询数 |
| `zDim` | 6 | 生成器噪声维数 |
| `ganIter` | 100 | 每次训练的生成器更新数 |
| `ganMiniBatch` | 32 | WGAN-GP mini-batch |
| `nCritic` | 4 | 每次生成器更新前的 critic 更新数 |
| `minGANTrainCount` | 32 | 最少条件训练行数 |
| `sampleSigma` | 0.3 | 生成采样噪声标准差 |

固定内部值包括：主线critic保留200、50% CGAN cutoff、20%引导份额、每参考方向最多5个边界锚点、每代最多20个校准FE；旧20槽×5候选仅由冻结历史/筛选分支使用。

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

`resume=true` 只校验 MAT 结构、种群维度和 FE，不校验源码哈希。`Data/CBS_RegionWGAN_GP` 中现有文件属于旧 A0（旧生成＋旧利用），不能代表 2026-08-28 后的新主线。新主线必须使用新的 campaign root；未经明确授权不得移动、覆盖或与旧目录混算。

根目录`test.m`当前启动`Support/run_CBS_CGAN_screening_campaign.m`。协议固定5题、runs`1:5`、`N=100`、`maxFE=200000`、`save=20`和10个process workers。先完成E0/E1并选定“丢弃/保留”，再比较K5/K10/不限；其余前沿、容量、split gate、均衡batch和记忆父代只在审计证明机制有机会生效时逐级运行，最后必跑全W均衡500条件critic池。每一级只相对上一胜者改变一个因素。

分析器删除`NaN/Inf`后再计算均值、独立`ranksum`和候选/对照比。初筛晋级同时要求200K跨题几何比`<=0.98`、实用胜题多于负题，且DAS/LIR家族比均`<=1.02`。结果、阶段状态、CSV/MAT分析、失败日志和首次源码快照位于`Data/CBS_CGAN_sequential_screening_runs1_5/`。

### 顺序筛选正式结论（2026-08-28）

正式队列共完成150/150个任务，0 failure；6个实际运行分支各有25个终点`FE=200000`的有效MAT。E1 KeepAnchor、E2 K10/KAll、E4 Cap10和E8 GlobalCritic均未通过当时以200K终点为主的预设晋级门槛，筛选当时保留参数`[0 5 2 5 0 0 0 0 1]`。本次用户明确改用CGAN截止时性能和边界邻近度重新裁决，并把E8的生成机制迁入生产；因此E0只表示迁移前的冻结筛选基线，不再表示当前生产语义。

200K跨题几何均值比依次为：KeepAnchor `1.02539`、K10 `1.02199`、KAll `1.00315`、Cap10 `1.00066`、GlobalCritic `1.04185`。E3因只有1题的前沿丢弃率达到20%而跳过；E4因5题的容量丢弃率均超过10%而执行；E5—E7对应的不安全训练、pairflag失衡和父代回退触发率在E0上均为0，因此跳过。

机制审计显示，Cap10虽把平均保留锚点从`90.03`增到`174.50`，引导存活率却由`2.096%`降到`1.967%`。GlobalCritic把原始候选扩大到每事件500个并保留40%，原始可行率从`23.84%`升到`26.36%`，但真实参考方向命中率由`28.42%`降到`23.48%`、引导存活率降到`1.666%`，最终在DASCMOP1_BC退化`22.65%`。这些数字仍是有效历史证据，但原始可行率不再作为生成质量指标，200K终点也不再单独决定前半程CGAN机制；后续需在精确CGAN截止点观测IGD/HV，并用不区分可行侧的边界距离评价原始池、critic保留池和最终20个候选。

完整逐阶段表位于campaign的`analysis/E*_candidates.csv`与`E*_problems.csv`；机制表位于`analysis/audit_by_problem.csv`、`audit_overall.csv`和`audit_trigger_decisions.csv`。

## 支持文件

- `Support/addCBSPaths.m`：移除当前仓库 `Data/**` 路径后，只添加 Algorithms/Problems/Metrics，避免冻结历史类进入活动路径。
- `Support/run_CBS_RegionWGAN_GP_mainline.m`：生产主线 runner；根目录 `test.m`：通用批量启动脚本。
- `Support/CBS_CGAN_screening_protocol.m`、`run_CBS_CGAN_screening_campaign.m`、`analyze_CBS_CGAN_screening_stage.m`：当前顺序筛选协议、队列和分析器。
- `Support/analyze_CBS_CGAN_factor_experiment.m` 与协议：保留旧 A0/A1/A2 三臂实验的历史分析，不再定义当前生产实验。
- `Tests/`：14个活动测试；新增筛选测试覆盖配置、锚点保留与跨代转配、方向限制、前沿/容量、均衡batch、NaN删除和顺序协议。

## 确定性验证

固定指纹仍使用 `LIRCMOP6_BC`、`N=100`、`D=30`、`maxFE=20000`、`rng(4242,'twister')` 和 MATLAB 默认计算线程。生成/利用改造会有意改变 RNG 轨迹和种群轨迹；最新数值由 `Tests/test_CBS_mainline_fingerprint.m` 锁定，不再沿用改造前指纹。

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
- 生产 runner 契约与历史三臂分析兼容性
- 选择来源审计的聚合计算
- 审计开关的行为中立集成验证

冻结的最终 BT0-vs-NoBT 结果、manifest、41 文件源码快照和诊断仍保存在 `Data/CBS_bt0_vs_nobt_gate_*`。
