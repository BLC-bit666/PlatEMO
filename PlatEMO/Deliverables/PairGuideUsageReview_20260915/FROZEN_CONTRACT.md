# FROZEN_CONTRACT：本次评审的最高优先级范围

唯一开放变量是“已选q如何参与P1搜索”。允许替换下文当前bridge使用公式；当前guideWeight=0.5描述现状，不强制新公式仍为半拉力。其余参数、机制全部冻结。所选q和原父代的一一匹配也保留，每个q各使用一次，不能通过实质弃用q、缩减引导或提前停用伪装收益。可以研究仅影响使用算子的反馈校准，但不新增训练模型、不改变档案/数据/生成/筛选规则。新算子若需随机数，必须隔离于既有随机流并明确复现方式。

主要消融：configureComparison("fallback_only")，即Guidance slots replaced by DE。它保留P1/P2各25GA+75DE。辅助pure_de将两种群全部算子改成DE，不能替代主要消融。原CCMO另附源码作为祖先骨架，非本次同骨架DE的同义词。

固定实验：LIRCMOP5–8_BC，N100/D30/M2，100000FE，真实P1原生IGD每200FE。开发配对种子1–5，已有仅1–2的筛选必须标注；未来确认用未参与选型新种子。全程/分段/终点同时看，图实线算术均值、阴影min–max、对数Y，无平滑和强制单调。禁止在线使用PF、IGD、离线q真值或未支付的真实目标/连续约束/边界信息，禁止多分支真实评价后免费择优。

以下为当前Agent.md的契约原文片段，历史证据入口不代表开放调参：

# PairGuide 当前工作树契约

更新时间：2026-09-15。

当前 Git 分支为 UC-GAN-2；算法身份为 PairGuide，核心为 PairGuideCore。上一任 CBS_RegionWGAN_GP 与 CBS_RegionWGAN_GP_Core 保持独立。不得将冻结副本加入 MATLAB 活动路径，也不得通过名称映射替换算法。

本文件记录用户历次决定及最新实验确认后的算法契约；2026-09-14 已将当前确定方案迁移至 PairGuide 主线。实现细节、公式、接口和历史审计见 [算法 README](</Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Algorithms/Multi-objective optimization/CBS-CGAN/README.md>) 的“中点桥接主线”部分；其下方 native-v2、共享噪声、旧学习率和旧训练预算属于历史记录，不能覆盖当前契约。后续对照实验可显式覆盖参数并独立记录，不得把实验配置静默替换为主线默认。

**2026-09-14 用户最新确认：采用中点桥接，总名额 Q=20、盒外上限比例 r=0.2（名义16个盒内＋最多4个盒外），关闭自适应，桥接权重 0.5；在 0.5×maxFE 停用 CGAN，本次 maxFE=100000 时截止为 50000 FE。生成点不直接真实评价，只引导 P1 搜索。** 网络采用 zDim=12、G=[8,8]、D=[16,16]，batch 上限64，首训1000/续训20次G优化器更新。

随后完成的 `boundary_local` 使用方式实验使总体过程与终点指标改善，但P6和后期仍退化，尚未达到稳定、全面、大幅优于同骨架DE的目标。它是显式启用的实验候选，不覆盖上述默认。用户已授权继续仅改变“所选q如何引导P1”的研究；完整结果与后续边界见本文末节。

`PairGuideCore`、公开参数说明、网络默认值、候选配额和 README 已同步；主线通过 `PairGuideOffspring_RC` 消费 q。上一版Q30的两个 CGAN 案例及两种 DE 对照的 12000 FE 重放已核验逐代种群、存档、原始/所选 q、真实子代与实验版完全一致，见 [迁移核验](</Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Data/PairGuideMainline_20260914/qa/migration_replay.json>)。上一版Q30选型依据为 [两参数最终报告](</Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Data/PairGuideQuotaCutoff_20260914/FINAL_REPORT.md>)、[最终配置冻结](</Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Data/PairGuideQuotaCutoff_20260914/freezes/stage2.json>) 和 [最终复核](</Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Data/PairGuideQuotaCutoff_20260914/qa/final_review.json>)；网络与训练选择依据为 [训练配置冻结](</Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Data/PairGuideSequentialTuning_20260911/freezes/stage2.json>)。在三配置同条件复核及15＋5实验完成后，用户最终指定当前主线采用 **Q20（16＋4）、0.5×maxFE截止**。依据为 [Q20历史复核](</Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Data/PairGuideQ20Review_20260914/REPORT.md>) 与 [15＋5对照报告](</Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Data/PairGuideQ20R25_20260914/REPORT.md>)。Q20（15＋5）、Q30（24＋6）及旧0.95×maxFE截止保留为历史配置；不能将其写成当前默认。

## 已确定的参数

网络、训练、配额与截止采用下表的已确认配置；其余未重新选择的参数和存档机制沿用既有约定。`PairGuideCore.mainlineDefaults()` 与 `PairBoundaryWGAN_RC` 默认已同步下列网络和训练配置。判别器在代码中也记为 C/netC。

| 项目 | 当前确定值与含义 |
| --- | --- |
| 随机系数 | `trainingSigma=sampleSigma=0.1`；`z=0.1*epsilon`，`epsilon` 为 12 维标准高斯。0.1 是标准差系数，方差为 0.01；训练和生成都独立抽样，系数均必须为有限正数 |
| G/D 学习率 | `lrG=lrD=0.001`（核心字段 `ganLrG/ganLrD`） |
| 网络 | 单个条件 G、单个条件 D；`generatorHidden=[8,8]`、`criticHidden=[16,16]`，均为两层；`zDim=12`；不增加辅助网络或解坐标条件 |
| 优化与批次 | Adam `(beta1,beta2)=(0,0.9)`；`gpLambda=10`；`nCritic=5`，即每次 G 更新配 5 次 D 更新；最终选择端点 mini-batch **上限64**，标签与方向均衡采样。实际batch按可用训练端点数取不超过64的偶数；不能因某次实际不足64而把配置改记为32 |
| 训练预算 | 首训 1000、每次重训 20 次 **G 优化器更新**，不是完整遍历数据集的 epoch 数；续训保留模型及 Adam 状态 |
| 训练门槛 | 至少 8 条 active pair；相对上次训练快照内容变化达到 0.2，或距上次训练达到 10 代时重训 |
| 0/1 条件 | `useSideCondition=true`；训练和生成均区分 `s=1` 可行侧、`s=0` 不可行侧。训练标签由真实可行性确定；请求侧别不保证生成后的真实标签，与 BC 约束返回值的 0/1 含义不能混用 |
| 方向条件 | 直接使用原始 `w`，`stableConditionSpan=false`；训练端点按各自真实目标关联方向 |
| D 错误方向负例 | `mismatchFraction=0`，默认关闭“同侧真实点＋错误方向条件”负例；这是已完成参数实验后的选择，不是遗漏实现 |
| G 损失 | 仅 `-mean(D(G(z,w,s),w,s))`；**不增加同侧、邻近方向的协方差匹配损失**，也不加端点重建、配对差向量回归或逆 gap 权重 |
| 生成与配额 | 每批 500 个原始点，请求 `s=0/1` 各 250；N=100 时总名额 `Q=20`、`r=0.2`、盒外最多 4 个，名义盒内 16 个；自适应关闭。盒外不足由盒内补齐，总数不足由 DE 补齐。20/4 指总数上限20、盒外上限4，不是盒内20再加4，也不保证每代恰好16+4 |
| 生成点使用 | `mode="bridge"`、`guideWeight=0.5`；在归一化决策空间构造 `p+0.5*(q-p)+0.5*(a-b)`，随后执行原 DE 交叉、修复与多项式变异，只评价最终子代 |
| 选择池与存档 | `selectionPool="ccmo"`、`archiveFrontDepth=1`；总容量最多 500 对，包含 active/inactive；每方向最多 1 条 active pair；邻域含自身共 5 个方向 |
| 使用时机 | 确定 `ganStopFraction=0.5`；完整代截止为 `floor(0.5*maxFE/(2*N))*(2*N)`，N=100、maxFE=100000 时为50000 FE。只安排能在截止内完成的下一代引导，之后停用CGAN训练、生成与引导消费，原搜索继续到maxFE；截止前仍须满足存档、模型和下一代配额条件 |

保留 0/1 条件、不给 G 增加协方差匹配损失，是用户明确固定的策略，不再作为本轮待验证选项。sigma、学习率及负例关闭是当前网络和预算下的实用默认，不宣称连续参数空间的唯一最优；负例结论只覆盖已测试的构造与比例，不否定其他尚未实验的负例方案。

## 必须保持的策略

1. P1 约束搜索，P2 无约束搜索；采用 CCMO 式环境选择池，分别为 P1+O1+O2、P2+O1+O2。存档继续使用完整 P1+P2+O1+O2 重建。
2. N=100 时 P1 保留 25 个 GA 与 75 个 DE 搜索槽，已选生成点用于引导其中最后 m 个 DE 槽，m 为实际消费数量；保留原 55/20 两块 DE 父代与随机数流程。P2 仍为 25% GA / 75% DE；引导缺额走原 DE，不增加种群规模或评价预算。
3. 生成点 q 代末生成并按原机制筛选进入 Pending，下一代只用于构造 P1 搜索子代；q 不直接评价目标或约束，也不直接进入环境选择。最终子代按原流程真实评价并参与选择；所有真实评价严格计 FE。
4. 真实可行/不可行独立端点训练，输入原始 w、s 和独立 z，直接生成单点；每方向最多一条 active pair，跨 pair 邻域共 5。
5. 不增加 raw 全局可行精英支配硬筛，不回退上一任算法，不手工指定 D=20。
6. 网络与训练采用上表确定参数。首训/重训至少 8 条 active pair；已有模型在至少一条 active pair 时可向完整方向集合服务，CGAN 候选不绑定 pair。截止前保留启用资格不意味着无存档、无模型或无下一代配额时仍强行生成；不得在截止后消费遗留 Pending。
7. 重训比较上次训练快照内容：变化 >=0.2 或间隔 >=10 代；不使用 2000 FE 冷却、永久验证集或多重质量验收。
8. 真实收紧只要求超过数值容差；39.8% 不作为准入门槛或普遍验收下限。
9. 评价来源与时间只在审计日志中；不再用 age/resumeEligible/lastFE 驱动档案存活或重训。
10. 不把训练损失、构造后窄云或高使用率当成 P1 加速证据；保留 fallback-only 与 pair-only 对照。
11. 2026-09-06 用户追加存档规则：总容量上限 500 对（1000 个端点位置），包含 active 和 inactive。N=100 时旧档案 1000 点与当代完整 Union 最多 400 点竞争，完成反馈、筛选、重配对后最多保留 500 对；每方向仍最多一对 active。真实合法候选不足容量时不伪造、不重复填充。详细竞争规则见 README。
12. 2026-09-06 删除生成候选的坐标修正这一约定继续保留：原生候选经原筛选进入 Pending，生成和筛选阶段不拉回、不移动。原“下一代原样完整评价”已由 2026-09-11 的间接桥接约定替代；桥接属于 P1 子代搜索算子，不是生成候选修正。全档案盒筛选按下述规则执行。
13. 2026-09-07 用户指定：完整 Union 的全部真实可行点与旧存档可行端去重后统一全局非支配排序，包括未配对可行点；仅第一层可作为保留配对的可行端，不额外叠加 P1 支配硬筛。配对前先排除被任一候选可行点支配的不可行端；同一合格不可行端可与多个可行端配对。新配对、反馈后的历史和容量候补使用相同资格，不合格者删除。用户已明确允许不足 500 对或为空，不再强制填满。
14. 每代构建/重建时，对每个合格 F，在含自身的五方向邻域中选取归一化决策欧氏距离最近的合格 I。出现更近的真实合格 I 时参与收紧与重新配对，再经过统一资格及容量竞争；不冻结旧 I，也不保证旧 pair ID 永久保留。同方向选 active pair 时先比较 F 的当前方向质量，再比较 gap、ID；容量竞争先保方向代表，再按方向内排名保留候补。不能把“最近 I 配对”误写成“全档案只保留 gap 最小的 500 对”。

2026-09-07 用户批准单点 CGAN：激活端点按决策与标签去重、按自身真实目标关联方向；标签和方向均衡，生成器只用条件对抗损失，判别器 WGAN-GP；取消重建、配对差向量和逆 gap 权重。完整方向 0/1 各半查询；候选按下述盒内排序与盒外配额规则选择。无配对 ID 的真实子代仍归因为 guided，可通过普通邻域规则更新档案。不得加入辅助网络或解坐标条件。网络参数选择实验已完成，当前采用上表默认。

现有主线证据协议为 `PairGuide-bridge-v4`；`PairGuide-single-v3` 属于历史直接评价协议，候选策略标记为 `archive-box-capped-exploration-v2`。原 16＋4 选择规则实现记录在 Data/PairGuideSelection16plus4_20260910；间接使用方式研究在 Data/PairGuideUseDesign_20260911，网络/训练顺序研究在 Data/PairGuideSequentialTuning_20260911，最新Q与截止选择在 Data/PairGuideQuotaCutoff_20260914。旧无限额未知优先规则记录在 Data/PairGuideExplorationSelection_20260909，参数研究在 Data/PairGuideEarlyMidTuning_20260909，上一轮 100000 FE 三算法对照在 Data/PairGuideFull100k_20260910。前一轮非零噪声研究保留于 Data/PairGuideNonzeroNoise_20260908。PairGuideInterval、PairGuideIntervalBounded、PairGuideBall、Deliverables 中已有资料保持历史用途。旧 first-use Epoch/Sigma probe 已退役，现有主线证据通过 guideExperimentSnapshot().evidence 获取。

2026-09-07 前一轮实现与实验已完成，见[结果报告](</Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Data/PairGuideSinglePoint_20260907/RESULTS.md>)。121 次首用、36 次完整运行、8 次方向留出/续训检查支持上述预算选择，但未证明条件生成偏移已解决；后续工作应区分代码契约通过与模型质量达标。

2026-09-07 已完成 B/C/D 首训首用对照：`archiveFrontDepth=0/1/2` 分别对应 B 的旧 P1 条件、C 的联合第一层、D 的联合前两层，三者均采用 CCMO 选择池；A 复用已有结果。用户随后指定 C 为当前默认：`selectionPool="ccmo", archiveFrontDepth=1`。`configureBoundaryExperiment` 保留显式历史对照覆盖能力。已有实验各五个问题 seed=1、800 个完整 epoch、首次真实使用并环境选择后停止；800 epoch 和首用后停止仅为实验配置，不改变主线默认训练预算与终止条件。

最小回归：

```matlab
test_CBS_pair_guide
test_CBS_pair_guide_archive_capacity
test_CBS_pair_guide_pool_front
test_CBS_pair_guide_training_observation
test_CBS_platemo_compliance
test_CBS_region_wgan_mainline
test_CBS_mainline_fingerprint
```

2026-09-09 用户要求训练与使用均保留随机性，当前以正 sigma 高斯潜变量运行并拒绝零系数；旧零噪声结果只保留为历史配置的证据。侧别 s=1 是可行侧请求，不保证生成点真实可行或有P1收益。非零噪声复验、独立确认与限制见[本轮报告](</Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Data/PairGuideNonzeroNoise_20260908/REPORT.md>)。

## 冻结的生成与筛选机制

2026-09-10 确认原 500→20 筛选规则；2026-09-11 用户要求保留生成和筛选机制，仅允许调整所选总名额 Q 和盒外配额比例 r。2026-09-14完成Q30、历史Q20及15＋5的比较后，用户最终确定 **Q=20、r=0.2，即名义盒内16个＋盒外最多4个**。生成、排序、盒外方向资格及补齐机制不变。15＋5与24＋6保留为历史对照，不覆盖当前默认。

- **已知方向**：当前P1或边界存档已经覆盖的方向；其余为待探索方向。P1全部点及全部存档F/I端点参与覆盖判断。
- **盒内选择**：全部存档F/I端点（含inactive）逐维min/max构成一个决策空间盒。所有盒内候选统一按到最近真实F的归一化决策距离升序选取，不区分已知或待探索请求。
- **盒外探索**：仅允许待探索请求方向，每方向最多1点，总计最多 `floor(Q*r)` 点。只用请求方向判断资格，不将其当成真实外推成功。
- **补齐**：盒外名额不足时继续选盒内点；总数不足由DE补齐。盒内不足不能增加盒外上限。r 是计划配额比例，实际消费总数可低于 Q，实际盒外占比也不保证等于或低于 r；不为凑比例改变筛选机制。

所有候选均须合法、有限且与当前P1/P2及已选点不重复；生成和筛选阶段保留原始决策，不免费评价500点，也不直接评价已选 q。只核验真实搜索子代的方向与可行性，不把子代标签当作 q 的真实标签。方向编号损坏但实际条件合法时恢复或登记编号；缺失合法条件时报告错误。自动查询仍覆盖完整W，不自动扩展W外。请求方向未覆盖仅表示当前缺失，不表示历史从未探索或已到达未知边界。盒子是决策空间包围盒，不是逐对球、线段或目标空间边界。原机制实现与验证见[16＋4记录](</Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Data/PairGuideSelection16plus4_20260910/REPORT.md>)；新实验已核对 Q20/r.2 的筛选结果、顺序及随机状态与该机制一致。

## 已确认的生成点使用方式

1. 每个消费批次，将全部 m 个已选 q 与最后 m 个原 DE 槽的真实 P1 父代，按归一化决策距离贪心一一配对；每个 q 使用一次，不二次筛选、丢弃或重复使用。父代沿用原资格，不另加“仅可行父代”筛选。
2. p、a、b 为原 DE 的真实父代。在归一化决策空间，用 `p+0.5*(q-p)+0.5*(a-b)` 构造桥接变异位置，再调用原 `OperatorDE` 的交叉、边界修复与多项式变异；只评价最终子代一次。若最终提案恰与 q 重合，回退普通 DE，并断言不直接评价已选 q。
3. 原 P2、环境选择、存档、训练数据资格及模型训练规则保持不变；搜索产生不同的真实子代后，允许原反馈规则自然改变存档内容、训练数据与重训次数。
4. 当前确定桥接20/4、自适应关闭、0.5×maxFE截止。用户指定四题统一配置；不得按问题编号硬编码配额或擅自加入逐题切换机制。

隔离实现见 [fixed_use_offspring.m](</Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Data/PairGuideUseDesign_20260911/fixed_use_offspring.m>)。审计中原始 q 位于 `UseSamples.q`，真实子代位于 `UseSamples.childDecs`；兼容字段 `selectedTargetDecs/selectedCenterDecs` 存的是子代，不能用来论证原始生成点的边界拟合改善。主线实现为 [PairGuideOffspring_RC.m](</Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Algorithms/Multi-objective optimization/CBS-CGAN/Core/PairGuideOffspring_RC.m>)。`evidence.useEvents` 与每代 `.use.q` 单独记录原始 q；`.childDecs` 是真实评价子代。迁移核验与正式实验审计检查 q 直接评价 FE 为 0、槽位和总 FE 不增加、所选 q 恰好使用一次。

