# PairGuide 当前工作树契约

更新时间：2026-09-09。

当前 Git 分支为 UC-GAN-2；算法身份为 PairGuide，核心为 PairGuideCore。上一任 CBS_RegionWGAN_GP 与 CBS_RegionWGAN_GP_Core 保持独立。不得将冻结副本加入 MATLAB 活动路径，也不得通过名称映射替换算法。

本次主线以 [2026-09-05 修改对话](https://chatgpt.com/share/6a9bf4a7-f0f8-83ea-8b32-8365ea8dc46c) 最后一轮答复为依据。完整参数、公式、接口、审计口径和测试集中维护于 [算法 README](Algorithms/Multi-objective%20optimization/CBS-CGAN/README.md)，避免重复契约发生漂移。

必须保持：

1. P1 约束搜索，P2 无约束搜索；采用 CCMO 式环境选择池，分别为 P1+O1+O2、P2+O1+O2。存档继续使用完整 P1+P2+O1+O2 重建。
2. P1 25% GA / 55% DE / 20% PairGuide；P2 25% GA / 75% DE；引导缺额走原 DE。
3. donor 代末生成，下一代原样完整评价；所有真实评价严格计 FE。
4. 真实可行/不可行独立端点训练，输入原始 w、s 和独立 z，直接生成单点；每方向最多一条 active pair，跨 pair 邻域共 5。
5. 不增加 raw 全局可行精英支配硬筛，不回退上一任算法，不手工指定 D=20。
6. 首训/重训至少 8 对；首训为 1000 次更新，重训为 20 次（非零噪声复验后保留的基线，不声称最优）；G/C=32×32、学习率 1e-3、训练/生成 sigma 默认 0.1 且均必须大于 0、纯 WGAN-GP。已有模型在至少一对时可向完整方向集合服务，CGAN 候选不绑定 pair。
7. 重训比较上次训练快照内容：变化 >=0.2 或间隔 >=10 代；不使用 2000 FE 冷却、永久验证集或多重质量验收。
8. 真实收紧只要求超过数值容差；39.8% 不作为准入门槛或普遍验收下限。
9. 评价来源与时间只在审计日志中；不再用 age/resumeEligible/lastFE 驱动档案存活或重训。
10. 不把训练损失、构造后窄云或高使用率当成 P1 加速证据；保留 fallback-only 与 pair-only 对照。
11. 2026-09-06 用户追加存档规则：总容量上限 500 对（1000 个端点位置），包含 active 和 inactive。N=100 时旧档案 1000 点与当代完整 Union 最多 400 点竞争，完成反馈、筛选、重配对后最多保留 500 对；每方向仍最多一对 active。真实合法候选不足容量时不伪造、不重复填充。详细竞争规则见 README。
12. 同日用户追加修正：删除 c 到构造 q 的坐标修正，CGAN 原生 c 经筛选后直接进入 Pending，下一代原样完整评价；原生 c 不拉回；2026-09-09 追加的已知方向全档案盒筛选按下述新规则执行。
13. 2026-09-07 用户指定：完整 Union 的全部真实可行点与旧存档可行端去重后统一全局非支配排序，包括未配对可行点；仅第一层可作为保留配对的可行端，不额外叠加 P1 支配硬筛。配对前先排除被任一候选可行点支配的不可行端；同一合格不可行端可与多个可行端配对。新配对、反馈后的历史和容量候补使用相同资格，不合格者删除。用户已明确允许不足 500 对或为空，不再强制填满。

2026-09-07 用户批准单点 CGAN：激活端点按决策与标签去重、按自身真实目标关联方向；标签和方向均衡，生成器只用条件对抗损失，判别器 WGAN-GP；取消重建、配对差向量和逆 gap 权重。完整方向 0/1 各半查询，按方向轮转筛选；无配对 ID 的真实子代仍归因为 guided，可通过普通邻域规则更新档案。不得加入辅助网络或解坐标条件。允许自主调网络、损失和训练方式，必须实验验证。

当前证据协议为 PairGuide-single-v3，当前候选筛选修改目录为 Data/PairGuideExplorationSelection_20260909；前一轮非零噪声研究保留于 Data/PairGuideNonzeroNoise_20260908。PairGuideInterval、PairGuideIntervalBounded、PairGuideBall、Deliverables 中已有资料保持历史用途。旧 first-use Epoch/Sigma probe 已退役，当前证据通过 guideExperimentSnapshot().evidence 获取。

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

2026-09-09 用户追加500→20筛选规则：所有存档F/I端点（含inactive）逐维min/max形成一个全档案决策盒，不按单个pair的球/线段筛选。当前P1全部真实目标及全档案端点均未覆盖的请求方向优先保留，保持未知方向的请求与方向多样性；剩余名额仅从已知方向的盒内原始点中，按到最近真实F的归一化决策距离升序选取。P1含不可行点时其方向也算已有信息；未知候选仍需有限、合法、不重复。只依据请求方向判定待探索资格，实际生成位置在真实评价后核验，不免费评价500点来筛选。不将随机方向ID当成外推条件；合法条件存在时恢复或登记编号。当前自动请求覆盖完整W，尚未自动扩展到W外。保留正噪声、单G/D、原坐标消费、FE及原档案更新机制。规则与验证见[本次报告](</Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Data/PairGuideExplorationSelection_20260909/REPORT.md>)。
