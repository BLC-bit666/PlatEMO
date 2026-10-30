# GANCMO 工作约定

## 唯一生产主线

- 入口为 `Algorithms/Multi-objective optimization/GANCMO/GANCMO.m`。唯一公开算法参数为 `type=Algorithm.ParameterSet(1)`：1 使用 GA（默认），2 使用 DE；主线与 V1–V5 接口相同。CGAN 等固定参数仍只在各版本 `Settings.m` 定义，不增加实验开关、其他外部参数覆盖或诊断分支。
- P1 按约束选择，P2 忽略约束；两者容量均为 N。每代合计 N 次真实评价，尾部截断剩余 FE。P1 的 CGAN 计划比例随 FE 从 0.1 升至 0.9，不足由所选 GA/DE 补足；P2 仅使用所选 GA/DE。两种群共享常规子代，CGAN 子代直接评价且只参与 P1 选择。训练、查询、常规变异的随机流互相独立。
- 必须调用框架 `Algorithms/Utility functions` 中原生 `TournamentSelection`、`OperatorGA`、`OperatorDE`，入口检查其解析路径；不复制或自制交叉、变异。GA 二元锦标赛选取偶数个父代，使用原生默认参数；先生成决策，奇数名额时先裁掉多出的一行再真实评价。DE 保留原父代选择、默认参数、随机调用顺序和评价顺序。CGAN 子代不经 GA/DE 二次加工。
- 两种群保留 SPEA2 强度适应度、近邻密度和逐点拥挤截断。距离使用 `pdist2` 的原始目标欧氏距离，不加目标归一化、方向配额或其他指标。截断可一次排序并保留邻居索引，删点时同步删除对应邻居；剩余距离顺序、字典序比较及同距决胜必须与逐轮重排完全一致。
- W 初始化后固定，R 为实际方向数。Z 累计全部真实评价的逐目标历史最小值，包括不可行和淘汰解。目标仅作 `f-Z` 平移；代内训练与查询共用代初 Z，评价后更新 Z，再统一重配存档全部新旧端点。
- F 属于合并可行池非支配前沿，I 不被该前沿支配；每个 F 在包含自身的五方向邻域内选目标距离最近的合法 I。同距按决策字典序处理。每方向按 F 质量、目标间距、归一化决策间距、决策字典序排序，最多保留五对，全部对训练。不转移空位，不增加独立总容量截断。
- 同方向 F 复用合法 I 索引，但每次存档更新都按当前 Z 重建，不能跨代沿用。存档排序及容量保留规则不因子代生成方式改变。
- 生成器与 Critic 均为 128 宽单隐藏层，输入维数均为 D+M+1。G 输入 `[D维噪声; M维单位方向; 真实侧0/1]`，sine 激活，输出 D 维；C 输入 `[D维决策; 同条件]`，LeakyReLU(0.2)，输出无界标量。G 首层参考方向权重按标准差 128 初始化，随后正常训练；该值不是 sine 的可学习参数。
- 保留 WGAN-GP（系数 10）和同条件最近端点 RMS 损失（权重 50，2026-10-30 用户确认）。训练与查询的噪声标准差均为 1；G/C 学习率始终为 0.001，Adam β1=0、β2=0.9，每批 C2/G1。
- 首训 1000 个完整 epoch，续训 `min(200,floor(1000/K))` 个完整 epoch，K 为每轮批次数。端点同侧去重、F/I 分别无放回打乱，每轮各端点调度一次且不丢尾批；每批两侧损失权重各占一半。首训和续训均要求至少八个有效 F 方向；非有限训练回退网络、Adam、计数和成功快照。

## 动态续训与生成权限

- 保留公式2：当前归一化决策中点集对上次成功训练中点集的平均最近邻距离为 Delta。存档仍逐代重建，检测续训不等于更新存档。
- d 比较当前与上一有效观测中点集；r 为当前去重中点最近邻间距的 RMS；R=size(W,1)。候选尺度 `c=max(sqrt(r*d),sqrt(R)*d)`，`s=s+(c-s)/sqrt(k)`，实际阈值 `theta=min(theta,s)`，仅 `abs(Delta)>theta` 请求续训。
- 初始 `s=theta=sqrt(R)*r0`，k 从首训不同中点数起步；中点全重合时改用不同 F/I 端点的间距及点数。正局部移动使 k 加一；可计算 r 时更新 s/theta。零移动冻结尺度但仍检查 Delta；空或非有限观测暂停检测状态。成功训练仅更换比较快照，不重置尺度状态。
- 未超阈值保持 CGAN 权限；超阈值先停用，仅本次训练成功才重新开启。停用保留网络、Adam、成功快照和原查询随机流消费，名额由所选 GA/DE 补足。无公式3、D 时钟、固定周期、冷却期或间隔反馈。10–20 代只是实验评价偏好。
- 检测器仅保存下一次计算需要的 `theta`、`samples`、`previous`、`smoothed`、`referenceCount`。观测次数、局部移动/间距留存和初值来源等纯诊断信息放在隔离测试中。
- 检测器先求 `pdist2(...,'squaredeuclidean')` 的最近值再开方，减少开方次数；r 仍保留最近距离“开方后再平方”的原始舍入顺序，不作代数消去。环境选择距离计算不变。
- 2026-10-30 根据十二题配对查询结果采用每方向 K=50、端点损失权重50。主循环以实际 R 调用 `GANCMOSettings(R)`，预留 `candidateCount=50*R` 个槽位，每方向恰好50个；查询条件为 `[W(ref,:),1]`。本方向的局部轴对齐包围盒，由当前存档中 `Archive.ref==j` 的全部保留配对F/I端点归一化后逐维取最小值和最大值构成；不合并其他方向，也不是各配对盒的并集。盒内按距本方向最短归一化决策配对中F的距离升序选择；该配对同距沿用存档顺序，只确定排序参照F，不再决定盒范围。无配对方向没有对应盒，按 Critic 降序选择。存档容量保留顺序和全部配对训练不变。
- 多候选保留有限值检查、变量边界裁剪、与 P1/P2/存档端点及已选候选的归一化距离 1e-6 去重。有配对方向优先、组内随机，每方向至多选一个，共享 G 名额；若有配对方向多于 G，按随机顺序选满即停。筛选失败继续尝试其他有配对方向，全部尝试后还有空额才随机访问无配对方向，最终不足由所选 GA/DE 补足。没有新增比例、排序权重或名额超参数。保留噪声预留、极小批次补行及 Critic 近同分的全批回退，均不增加真实 FE。停用时仍消费对应的预留查询流。
- 生产只保留固定 K=50 的多候选筛选路径：双目标 R=100 时5000槽位，三目标 R=91 时4550槽位。K=1直接生成分支已清理；历史对照只能从实验压缩归档恢复，不向生产增加兼容分支。GuideData仅保存查询需要的最短配对F、方向编号及数量，不重复保存未使用的I端点。
- 模型接口为 `train`、`sample`、`score`；score 只做条件 Critic 前向排序，不调用目标/约束或消耗 RNG。存档更新返回 `Archive,Scale,Guide`，Guide 仅供查询筛选；训练仍读取全部有效存档对，保留当前训练数据过滤，不恢复旧的可选训练子集参数。

## 独立消融算法

- 生产GANCMO和独立V1–V5分别放在同级目录，算法辅助文件自包含、使用各自前缀，不跨版本调用，不增加生产版本开关或V6。2026-10-30用户授权将V1/V2/V4/V5公共部分对齐当前主线：K=50、全配对方向局部盒、最短配对F排序、有数据方向优先、条件Critic排序、去重及当前环境选择实现。
- V1仅删除G/C的可行性输入，训练、sample、score均只接收方向，端点匹配按方向分组。V2仅删除G/C的参考方向输入，训练、sample、score均只接收0/1侧标签，查询为1，端点匹配按侧标签分组。两者完整存档元数据仍维持原批次和公共筛选，不把已消融的条件偷偷传回网络。
- V4只有F进入模型真实样本、Critic插值和端点匹配，条件仍M+1；保留F/I存档、检测、候选盒及原批次数。V5仅删除G的端点匹配项及其配置，保留对抗损失和Critic的WGAN-GP。V1/V2/V4端点权重为50；V5没有端点损失。
- V3完全禁用CGAN及其专用存档、训练、查询、检测模块。两种群仅使用所选原生算子：type=1为纯GA，type=2为纯DE；不重新加入CGAN。其环境选择已同步当前主线的等价优化实现，删除版本前缀后源码一致。共同算子模式下，与关闭CGAN的主线保持初始化、FE分配、子代、选择及随机轨迹一致。V1/V2/V4/V5此前单候选实验仍为历史证据，不混作当前主线单因素结果。
- 对应回归为 `Tests/test_GANCMO_ablations.m`、`Tests/test_GANCMO_operators.m`。新增公共主线机制应先对齐保留该机制的消融，再只保留各自目标差异；如用户明确冻结某版本，保留其冻结边界并说明。消融比较须使用同一type；历史DE结果对应type=2，不能当作新默认GA结果。不得擅自改变主线、根目录test.m、其他算法或既有实验结果。

## 对比算法算子接口

- 2026-10-31用户最终要求保留对比算法自带GA/DE机制，仅补充缺少的算子模式；此前“全部替换为原生默认”的方案已撤回。范围为DRMCMO、IMTCMO、MCCMO、MTCMO、NAEMT、EADMMMOEAD。统一第一参数type：1=GA（默认），2=DE。
- DRMCMO保留原GA及原随机参数DE；MCCMO保留原GA/DE两条分支，仅改由type选择。IMTCMO保留双DE组合并新增原生GA；MTCMO保留原GA并新增原生DE；NAEMT保留原自定义DE及K并新增原生GA。新增分支调用Utility functions原生默认算子，不复制交叉/变异公式；每对子代数量需要时使用原生OperatorGAhalf。
- EADMMMOEAD保留原外层GA，新增原生DE；用户明确要求内部MATLAB ga局部求解保持不变。原种群、环境选择、训练、存档、评价数量和整代终止机制不因统一接口另行更改。EADMM其他骨架入口不属于此次范围。
- NAEMT参数顺序为 `{type,alpha,epsilon,N1,K}`，EADMMMOEAD为 `{type,localN,weightFloor}`。其他四个对比算法只增加/保留type。新增默认GA不意味着原DE算法的历史结果可以作为新默认结果。
- 原算法对比按其原模式选type：IMTCMO/NAEMT选2；MTCMO/EADMMMOEAD选1；MCCMO在LIRCMOP/DASCMOP/DOC/RWMOP上选2、其他问题选1；DRMCMO保留所选原配置。统一GA或DE的扩展实验须单独标注，不能混用原模式结果。
- 记录为 `Experiments/Comparators_PreserveOperators_20261031`，回归入口为 `Tests/test_ComparatorOperatorChoice.m`。原有模式与修改前快照核对种群、FE和RNG；新增模式检查实际算子路径、默认值和运行。该阶段没有改写实验名单或启动基准实验。
- 2026-10-31后续用户授权修改test.m：algNames为两列配置，每行 `{算法名,type}`，不再对DRMCMO单独强制DE。proNames包含DASCMOP1–9_BC、LIRCMOP1–14_BC、FCP1–5_BC、DOC1–9_BC、SDC1–15_BC、LSCM1–12_BC，共64题。用户明确自行区分结果，保留原文件命名、存在即跳过规则，不自动添加type目录或结果标签。

## 修改与验证

- 保留用户已有改动；不改根目录 `test.m`、其他算法、问题定义、框架或全局 MATLAB 环境。未经当前任务明确授权，不提交、推送或改 Git 历史；不创建、切换或使用 worktree。
- 提速先测瓶颈，再用改前源码与固定输入核对输出顺序、网络权重、Adam、计数、随机流和 FE。只接入实测有效且通过逐位一致性验证的改动；数学等价或误差很小不够。不要改变批次、浮点精度、训练预算、随机调用顺序或算法判断来换速度。
- 复用 `Tests/test_GANCMO_selection.m`、`Tests/test_GANCMO_mainline.m`、`Tests/test_GANCMO_reference.m`、`Tests/test_GANCMO_training_window.m`、`Tests/test_GANCMO_candidate_localbox.m`。状态观测仅放临时测试副本，不向生产循环加入日志、性能 IGD 或实验开关。
- 局部计时只说明相应模块的速度；短程回归不能冒充完整搜索或最终 IGD 证据。完整对照遵从任务指定问题、种子和 FE，公开失败和缺失，不换种子。已有结果仅在源码、输入和环境核实一致后复用。
- 最终 IGD 优先；历史过程 IGD 口径为 `trapz(20000:2000:150000,IGD)/130000`，各题等权。耗时单列，少量运行不称统计显著。未经授权不追加参数搜索。
- 不递归添加 `Experiments` 到 MATLAB 路径。选型用临时算法、执行脚本和框架副本，在逐文件SHA-256验证压缩归档后移除可运行副本及空目录；保留结果、图表、报告、日志与哈希。正式V1–V5和当前主线的可复用回归不属于选型临时代码。主线不得保存轨迹、覆盖率、边界距离、性能IGD等纯观测数据；训练成功快照、Adam、检测状态和FE/有限值保护是算法必需内容，保留。历史实验不是可切换的生产分支。
- `Tests/`（含测试辅助函数）、根目录 `test.m` 和 `test_*.m` 为本地验证/实验文件，保留在磁盘但由 `.gitignore` 排除；实验、结果、日志及临时输出沿用现有忽略规则。算法源码、问题定义、框架、README 与本文件继续纳入版本管理，不以忽略规则隐藏生产改动。

## 依据与记录

- 当前公式与调用流程：算法目录 `README.md`；参数以当前源码为准。
- 阈值采用：`Experiments/GANCMO_ThresholdIteration_20261019/PRODUCTION_ADOPTION.json`。已测试问题和维度之外，不宣称间隔普遍达标或维度无关。
- 初始化尺度、epoch、共同噪声的采用记录：`Experiments/GANCMO_Sine128_Adoption_20261019`、`Experiments/GANCMO_Epoch1000_200_Adoption_20261018`、`Experiments/GANCMO_Noise1_Adoption_20261018`。各实验保持原设置和结论，不把分项选型结果合并为当前组合的新性能证据。
- 本次清理与等价提速：`Experiments/GANCMO_CleanupSpeed_20261019`；以最终报告和实际通过的核验范围为准。
- 后续局部优化：`Experiments/GANCMO_ExactSpeed_20261019`；范围限存档配对、候选筛选和检测距离计算，验证状态与速度以该目录实际结果为准。

- 单候选采用：`Experiments/GANCMO_SingleCandidate_Adoption_20261021`；候选数量对照与存活/可行率分析分别保留在 `GANCMO_CandidateMultiplicity_Runs5_20261020`、`GANCMO_CandidateSurvival_20261021`。采用范围为每方向一个候选直接评价，其余主线机制保持原逻辑。
- 旧候选接口清理：`Experiments/GANCMO_LegacyCleanup_20261021`；移除评分/方向代表接口并同步本地回归。四项回归通过，120 组存档/训练数据/随机状态对照完全一致，120 组异常训练行过滤检查通过；本次未重跑完整 15 万 FE 搜索。

- 多候选恢复与 runs=1 验证：`Experiments/GANCMO_CandidateFilter_Restore_20261028`。历史规则依据 SingleCandidate_Adoption 的 BEFORE.zip；恢复不回退训练、检测、配对、环境选择或 DE。当时K=1保持直接路径，其余恢复原盒/去重/距离或Critic排序；该K=1分支现已从生产移除。

- 2026-10-30 盒与配额纠正：`Experiments/GANCMO_LocalPairBox_Priority_20261030`。此前权重/K实验使用质量优先代表、共同大盒和全方向混合随机顺序，不符合本次明确的局部最短配对盒及有数据方向优先规则；原结果保留为旧机制证据，不能视作修正版的参数最优性证据。当前筛选回归为 `Tests/test_GANCMO_candidate_localbox.m`，旧兼容入口 `test_GANCMO_candidate_restore` 已清理。

- 2026-10-30 当前主线采用：`Experiments/GANCMO_AllPairsBox_K50_Adoption_20261030`。依据 `GANCMO_BoxScope_K_20261030`、`GANCMO_BoxScope_Weight100_20261030`、`GANCMO_BoxScope_K150_200_20261030` 的配对查询结果，选择权重50、K=50和本方向全部配对的局部盒；最短配对仅保留F排序参照。此前最短配对盒不再是生产机制。离线候选可用性支持成本折中，不证明新主线真实边界距离或最终IGD最优。

- 2026-10-30 主线清理与等价提速：`Experiments/GANCMO_CleanupExactSpeed_20261030`。清理2208个选型执行文件和旧测试兼容入口；移除生产K=1分支及未使用的GuideData.xI。环境选择复用已排序距离及邻居索引；训练缓存与推理调用的试改因收益不稳定未采用。完整FE/IGD等价、局部计时和归档状态以该目录 `REPORT.md`、`FULL_EQUIVALENCE.json`、`FINAL_AUDIT.json` 为准。

- 2026-10-30 四个消融对齐：`Experiments/GANCMO_Ablations_AlignMainline_20261030`。公共源码按当前主线重建，去除前缀后V1/V2各仅Candidates和Model有目标适配，V4仅Model不同，V5仅Model与Settings不同；主线算法源码和V3保持不变。具体回归、独立入口运行及未覆盖范围以该目录REPORT.md和FINAL_AUDIT.json为准。

- 2026-10-30 后续V3修复与算子选择：`Experiments/GANCMO_OperatorChoice_20261030`。V3环境选择对齐当前主线；六个算法均支持原生GA/DE，默认GA。本条更新此前“V3不改”和“无公开参数”的历史状态。权重50、K=50、局部盒、训练、检测与各消融目标均保持不变。完整DE轨迹/IGD核验、两种算子的入口与FE/RNG检查，以该目录 `REPORT.md`、`FULL_EQUIVALENCE.json`、`FINAL_AUDIT.json` 为准；不把GA功能测试称为GA优于DE的性能证据。
