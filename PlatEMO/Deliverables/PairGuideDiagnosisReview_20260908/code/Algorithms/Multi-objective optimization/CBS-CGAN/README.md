# PairGuide 单点条件生成（2026-09-07）

本节为当前执行契约，覆盖下方保留的 native-v2 历史记录。算法入口仍为 PairGuide。

本轮已完成 121 个首次使用实验、36 次完整搜索和 8 个方向留出/续训检查，详见[实现与实验报告](../../../../experiments/PairGuideSinglePoint_20260907/RESULTS.md)。首训选为 1000 次更新；完整实验尚未显示稳定优于对照，方向与 0/1 标签控制仍不可靠，不能将本轮修改描述为已经解决生成偏移。

- 存档先对完整候选可行池统一分层，仅全局第一层有资格；不可行端不被任何候选可行点支配；历史同筛，最多 500 对，可少于上限或为空。每方向一条 active pair、五方向邻域最近配对和不可行端复用不变。
- 输入 `[z,w,s]`，s=1 请求可行、s=0 请求不可行（这是条件标签，BC 问题返回的 0/1 约束值含义相反）；输出一个绝对决策解。无编码器、辅助网络、坐标条件或生成后的坐标拉回。
- 激活配对拆成独立端点，按 `[x,s]` 去重，分别用真实目标在同代 referenceScale 下关联原始 W。每批两侧等额，每侧方向组均衡；不要求同批端点来自同一 pair，也不共享 z。
- G 损失 `-mean(D(G(z,w,s),w,s))`；C 为条件 Wasserstein 差加决策梯度惩罚。无端点回归、差向量回归或逆 gap 权重。`pairVisits=0`；`endpointVisits` 记录实际样本访问，`epochs=endpointVisits/trainingSamples`，更新预算单位仍为 G 更新次数。
- 查询完整 W，0/1 各半；候选按请求方向轮转、合法性和去重筛选，无配对匹配或端点误差排序。ID=0 表示不绑定现有配对，并非评价失败。下一代原坐标真实评价；所有 guided 子代均能经邻域收紧归因。pair-only 保留真实区间采样与配对归因，作为明确的对照。
- P1/P2 选择、25/55/20 与 25/75 算子比例、500 raw、20% 配额、FE 规则保持。无存档或无模型时 DE 回填。训练仍至少 8 active pairs；模型和 Adam 持续保留，内容变化 0.2 或 10 代触发，首训 1000、重训 20 次生成器更新。
- 本轮统一配置：G/C 均为两层 32 单元，学习率均为 1e-3，Adam=(0,0.9)，GP=10，C:G=5:1，batch 上限 32。训练与推理 sigma=0；同一 (w,s) 当前只生成一个确定点，500 次查询会重复，筛选去重。此配置不声称能表达同方向多模态；原始 z 接口保留用于显式噪声对照。错误条件反例实验没有综合优势，主线不保留该分支。
- 首训预算在 200/500/1000/2000/4000、四问题各三种子上比较。1000 的首次使用后平均 IGD 最低，12 组对同 FE 的 fallback 为 11 胜、1 平；此为本轮已测配置的搜索表现选择，并非条件拟合各指标最优或全局最优。
- `configurePairGuideTrainingExperiment` 必填 initialEpoch/retrainEpoch/nCritic，可选 lrG/lrD/miniBatch/generatorHidden/criticHidden/trainingSigma/sampleSigma/gpLambda。主线不能通过此接口输入任意条件或辅助网络。
- 协议 `PairGuide-single-v3`。保存完整 query conditions/refs/sides、原生候选、筛选索引、真实子代、首次/最新模型及各代 referenceScale。无对应配对的几何诊断填 NaN，不能按相对窄带或原配对支配统计证明质量。原端点 RMSE 仅保留作固定探针描述，不作为训练目标或筛选门槛。
- 生图只在 run 1，显示请求可行/不可行两类原生候选和真实消费点，颜色代表请求而非 oracle 标签；标题给出独立真实评价的标签正确率和方向偏差。离线生图对 unique raw 调用 CalObj 与 CalCon，BC 的 CalCon 内部也调用 CalObj，因此账单为每点两行 CalObj、一行 CalCon，不计入搜索 FE。
- `run_PairGuide_single_first_use` 在首训后仅真实消费一批 CGAN 子代、完成环境选择后停止；raw 的目标/可行标签在停止后独立评价。本轮目录 `Data/PairGuideSinglePoint_20260907`，不覆盖旧实验。

以下内容为历史机制、验证与来源记录，不能替代上述单点协议。

# PairGuide：最终对话实现契约

更新：2026-09-07。依据：[修改对话](https://chatgpt.com/share/6a9bf4a7-f0f8-83ea-8b32-8365ea8dc46c) 的最后一轮答复及用户追加的固定容量、删除 q 修正、CCMO 式环境选择池和全局第一层存档资格要求。最新层级资格替换原 P1 支配硬筛；其余网络、时序和成本要求继续生效。

同日独立严格复查确认并修正了五方向邻域反向检索、无下一代引导名额时仍训练/生成、汇总检查点使用未来状态三处偏差，并补齐粗区间/绝对窄带诊断。逐项证据和仍未完成的实验、审计交付见 [严格审查说明](STRICT_REVIEW_20260905.md)。本文的运行机制对应不等于全部实验要求已经完成。

PairGuide 入口仍继承自己的 PairGuideCore。CBS_RegionWGAN_GP 及其核心保持独立，不作名称映射。当前 Git 分支名 UC-GAN-2 不影响算法身份。Deliverables 中旧副本只作冻结证据。

## 对话结论与实现

当前默认 `selectionPool="ccmo", archiveFrontDepth=1`，沿用此前 C 组的选择池和可行分层配置，并已追加不可行端支配筛选及不合格配对删除。直接启动 PairGuide 即生效。CCMO 选择池分别是 `P1+O1+O2`、`P2+O1+O2`；存档输入仍是完整 `P1+P2+O1+O2`。前沿在容量截断前按 `Union` 全部真实可行点与旧存档可行端去重后统一排序，未配对可行点也参与；仅第一层合格，不额外叠加 P1 支配条件。先筛出合格可行点，再剔除被完整可行第一层任一点支配的不可行候选，最后在五方向邻域内配对；同一不可行端可以复用。不合格历史配对直接删除，500 对是上限，允许不足。`retainedFrontRanks` 与最终存档行对应，`eligiblePairs` 记录保留的合格对数，`eligibilityRejectedPairs` 保留兼容字段且当前为 0；`p1DominatedPairs` 仅作诊断。

2026-09-07 新批次采用 LIRCMOP5_BC、6_BC、7_BC、8_BC、9_BC，每问题 seed=1:5，N=100、默认 D=30、maxFE=100000。该已完成批次仅 cgan 共 25 次，首训/重训仍为 200/20 次生成器更新，10 个独立工作进程、每进程单线程。目录 `Data/PairGuideCCMOFront1_5to9_20260907`，明确记录配置和源码哈希；该批次不是 800 epoch 首训后停止实验，其结果对应本次不可行端筛选修正之前的源码。

新 runner 通过 `checkpointFE=[10000 30000 50000 70000 100000]` 记录指定节点，并额外记录首次真实 CGAN 使用完成环境选择后的状态；首次未发生则明确 missing。逐 run 的 `*.mat.checkpoints.csv` 在运行中更新，最终 `analysis/checkpoints_per_run.csv`、`checkpoints_mean.csv`、`checkpoints_std.csv` 汇总 IGD/HV、可行数、存档总对数/激活数/合格数、gap、训练次数/更新数/RMSE、引导数/可行率/留存等。`lastTraining*` 指观测时已完成的最新训练，`consumedTraining*` 指该代实际使用的模型训练，二者可能不同；训练与实际完整遍历 epoch 分别记录。

`distributionLayout="training_pair"` 仅对 seed=1 绘图，每个节点提供全景与 focus 两张双面板图。左侧为该批候选对应模型实际训练时的完整 P1/P2、active/inactive 存档；右侧为原生生成解和最终筛选且原样真实使用的解。标题分别注明训练、生成、消费和观测 FE。全景覆盖全部点；focus 明确报告轴外数量。`plot_data.mat` 同时保存训练、生成及观测时的真实状态。若当代无对应生成事件，图中留空注明，不额外训练或查询。LIRCMOP6/9 的 CalObj 从原 Evaluation 原样提取，CalCon 复用相同算式，并加入与其余问题一致的评价计数；原 Evaluation 输出已做逐项一致性验证。离线绘图 CalObj 单独记账，不反馈搜索或增加搜索 FE。

`Algorithm.configureBoundaryExperiment(struct('selectionPool',pool,'archiveFrontDepth',depth))` 仍用于显式历史对照：A 为 shared/0，B 为 ccmo/0，C 为 ccmo/1，D 为 ccmo/2；depth=0 使用旧 P1 资格，1/2 使用联合前一/前两层。已完成的 B/C/D 实验固定 800 个完整 epoch，底层生成器更新预算按实际批次数换算，首次真实使用后停止；A 只读复用。该隔离实验不改变主线的 200/20 次生成器更新预算与正常终止条件。历史数据及源码哈希保持原样，新默认不覆盖已有实验结果。

| 最终要求 | 实现位置与行为 |
| --- | --- |
| CCMO 式双种群环境选择池 | PairGuideCore.runMainline；P1 从 P1+O1+O2 约束选择，P2 从 P2+O1+O2 无约束选择；完整 Union 留给存档重建 |
| P1 25% GA / 55% DE / 20% 引导；P2 25% / 75% | pairOffspring、scopedBackbone；整数余数和引导缺额由原 DE 路径承担 |
| 六项 pair 核心状态 | id/ref/xf/yf/xi/yi；gap、active 仅为当代派生缓存；nextId 为编号分配器 |
| 合格历史与候补 | 不使用 age、resumeEligible、lastFE；合格但未激活的配对可以再次激活，不合格历史直接删除 |
| 每方向最多一条 active pair，邻域共 5 | 完整候选可行集合全局第一层具备激活资格；不另叠加 P1 硬筛；历史记录不设失活年龄删除 |
| 原 pair 反馈先处理 | 完整评价后、容量竞争前反馈原 ID；保留完整 Union，不要求存活 P1，也不以目标支配或瞬时 ref 变化否决真实收紧 |
| 其他 pair 和普通结果 | 都按真实标签与严格 gap 收紧更新；跨 pair 仅在含自身的 5 方向邻域内 |
| 同方向竞争 | 全局第一层合格者优先，再按当代归一化方向质量、gap、稳定 ID 排序；方向内首个合格者激活；决定保留候选后才分配新 ID |
| 总容量上限 | 最多 500 对 / 1000 端点位置，active 与 inactive 共用；允许不足，完整当代 Union 参加竞争；N=100 时输入最多 1400 点 |
| 8 对首训 / 重训门槛 | 首训最多 200 次生成器更新；后续每次最多 20 次；最多 16 对一个 batch，保留尾批 |
| 重训内容变化 | 相对上次完成训练快照，按方向并集平均变化；至少 0.2 或已过 10 代；每代最多一次尝试 |
| 模型持续服务 | 已训练模型 + 至少一条当前真实 pair；不沿用 8 对训练门槛 |
| 全部当前激活完整 pair 训练 | 无永久验证集、历史验证库、回放混训或多个质量验收阈值 |
| 固定噪声前后诊断 | 复用端点误差、差向量、同条件厚度、相对 gap 诊断；无额外真实评价；数值失败保留旧模型 |
| 两端共享噪声 | Critic、Generator 的对抗项/几何项、配对推理及配对诊断均共享 z |
| 原生候选直接使用 | 保存 Gf/Gi、z、alpha、c、筛选索引、生成时真实端点、ID、ref、模型版本和生成时刻；不构造 q |
| 候选上限 | 先覆盖不同 pair，再第二轮；每 pair 每代最多 2 个，总数不超过 20% 名额 |
| 下一代原样评价 | Pending 保存所选原生 c 与真实端点快照；下一代不匹配 P1 父代、不取第二次中点、不变异 |
| 成本与快照 | raw 无真实评价；缓存已评价结果；分别记录生成、消费、观测 FE；缺测/陈旧状态显式标识 |

2026-09-07 用户追加：真实不可行端 xi 必须不被完整候选可行集合的任一点支配，在配对前筛选，并对更新后的历史配对重新核对。不对 CGAN 未评价的 raw c 增加支配过滤；目标矩形、objective-only、球外拒绝、父代再匹配及二次中点仍不启用。

## 存档筛选、配对与容量

2026-09-07 最新要求覆盖此前“满额后始终维持 500 对”的约定：不合格配对删除，允许存档减少或为空。`pairArchiveCapacity=500` 是上限；`pairArchivePerRef=1` 仅限制每方向的 active pair。合格但未激活的配对可保留竞争，不参与训练或生成。

每代完整评价 Pending 和普通子代，按 CCMO 选择池计算 P1/P2；存档接收完整 Union，先反馈原 ID，再按五方向邻域记录真实收紧。收紧是评价归因；是否保留更新后的配对，还必须通过本代统一资格筛选。

1. 完整 Union 的真实可行点与旧存档可行端按决策向量去重，对全部可行点统一非支配分层，包括未配对的点。默认仅第一层可作存档可行端。
2. 完整 Union 的真实不可行点与旧存档不可行端去重，删除被任一候选可行点支配的点。实现用完整可行第一层作比较；Pareto 支配传递性保证其覆盖其余层的支配作用。
3. 每个合格可行点在含自身的五方向邻域内，选择距离最近的合格不可行端。同一不可行端可被多个可行端复用，不实行一对一占用。
4. 新配对、真实反馈后的历史配对以及旧几何候补都通过相同的两端资格检查。不合格者物理删除，不能以失活候补或填满容量为由恢复。
5. 合格配对去重、按方向质量、gap、稳定 ID 竞争；每方向最多一对激活，总数最多 500 对。只有最终保留的新配对分配 ID。

若合法历史因收紧合并，原几何可作为候补，但同样必须通过两端资格。删除发生在本代真实反馈之后、下一批 Pending 生成之前；`removedIds` 保留谱系退出记录。激活不足 8 对时不训练；已有模型在至少一对激活时可服务，没有合格配对时由普通 DE 回填。

`candidateInputPoints` 仍记录完整输入 `2*历史对数+numel(Union)`，不提前截断到 1000。`feasiblePoolEligible/infeasiblePoolEligible` 记录筛选后两类端点数；`frontRejectedPairs/infeasibleDominatedPairs` 记录历史与新配对合并后各条件拒绝数，二者可重叠；`eligibilityRemoved` 为拒绝并集。保留后 `eligibilityRejectedPairs=0`。新增字段同时写入检查点指标；读取旧证据时缺项为 NaN。`restoredCapacity` 只表示加入筛选前候补数，不保证最终保留。

显式 `archiveFrontDepth=0/2` 仍可覆盖可行端资格规则，但同样执行新的不可行端筛选和不合格配对删除；不能将新的运行当成 2026-09-07 早前 A/B/C/D 对照的逐轨迹重放。历史实验的源码快照、协议和结果保持原样。

以下提速记录对应追加原生候选/P1 资格规则之前的固定容量版本。该次提速范围仅为存档更新：缓存相同 W 的原角度邻域顺序，每次更新只读取一次决策范围；对同一个源点的独立目标 pair 批量算距，仍按原源点顺序更新，保留距离算式、容差和事件顺序；预分配重配对候选；没有引导反馈时复用实际普通更新作为普通影子；影子不再构造无人使用的事件。该次纯提速不更改网络、训练预算、算子、随机流、FE、环境选择和 IGD 公式。

容量规则本身会改变搜索历史，不能声称与原无限存档版逐位相等。纯提速的比较基准是同一容量规则的未优化标量实现，核对完整 Archive、Scale、Trace、随机状态和端到端搜索轨迹及逐代 IGD。16 组整段对照完全相等；20000 FE 的 CGAN 实测提速 8.05 倍；100000 FE 长程存档满额后始终 500 对。完整数据、旧无限存档 IGD 对比与限制见 [性能验证记录](PERFORMANCE_VERIFICATION_20260906.md)。

## 网络与调度公式

训练目标保持归一化真实绝对端点 [xf,w,1]、[xi,w,0]。网络宽度 [32,32]、zDim=6、WGAN-GP=10、Adam 学习率 1e-4、参数 (0,0.9)、Critic:Generator=5:1。训练和推理 sigma 初值均为 0.3。

令 g=||xi-xf||，h=max(g,1e-3*sqrt(D))：

```text
Lend   = mean((||Gf-xf||² + ||Gi-xi||²)/(2h²))
Ldelta = mean(||(Gi-Gf)-(xi-xf)||²/h²)
LG     = Ladv + 10*Lend + Ldelta
```

两次都有的方向：
```text
dr = min(1, max(||xf-new - xf-old||, ||xi-new - xi-old||)
            / max(g-old,1e-3*sqrt(D)))
```
新增或删除方向取 1；实际条件 w 改变也计内容变化；只换 ID 不计变化。Delta 为方向并集上的均值。重训条件为 K>=8 且 (Delta>=0.2 或代数间隔>=10)。

配置接口的第三个公开参数现名 ganUpdates。为兼容既有调用，内部 pairInitialEpoch / initialEpoch / retrainEpoch 字段名称暂保留，但这些预算字段的单位均为生成器更新次数。训练事件 updates 是实际更新数，epochs 仅记录实际遍历次数，不能混用。

## 原生候选直接使用

每次 500 个配对提案，对应 1000 个端点网络输出。相同 z、w 查询两侧；alpha 均匀取 [0.4,0.6]，c=(1-alpha)Gf+alpha*Gi。不查询未训练的 side=0.5。

原生 c 不再向真实配对投影、不再截断轴向位置或垂向偏移。有限、位于决策盒内且匹配当前激活配对的 c 直接进入筛选；不以原生窄带命中或真实可行性作额外拒绝门槛，不调用 CalObj/CalCon。异常越界候选拒绝，不通过裁剪制造另一个点。`Pending.decs` 逐项等于 `rawDecs(pool.keepIdx,:)`。

同 pair 候选仅按两类网络端点的自一致误差/gap 排序，移除投影修正量这一项；方向仍按原局部价值、gap 和 P1 占用优先级轮转，保留原 gstar 数值尺度与去重容差 1e-6。每对先取一个候选，再第二轮，总数受引导配额限制。该筛选不改变候选坐标。

任何 g-new < g-old-1e-12 的真实收紧仍可接受；原 q 构造的 39.8% 收紧推论不再适用。网络拟合及原生候选的实际效果需要独立实验，不能从真实配对本身的几何质量推出 CGAN 有效。

## 审计与对照

guideExperimentSnapshot().evidence 提供：

- evaluations：真实评价编号、来源、决策、目标和约束；端点复用不重新计评价。
- generations：生成/消费代数、原始端点快照、真实子代、原 pair/其他 pair 收紧事件及源评价关联、P1/P2 存活。
- archive.netGapById：同一代前档案上普通结果影子更新与全部结果更新的 gap；影子不调用真实评价、不反馈搜索。
- queries：全部原生提案、两端网络输出、噪声、配对相对位置诊断、筛选索引与 Pending。新协议为 `PairGuide-native-v2`，`pool.candidateDecs` 保存原生候选，不再输出 `constructedDecs/correction`。
- training：触发原因、内容变化、更新次数、前后固定噪声诊断和耗时。
- population：每个已完成代的 P1/P2 真实缓存和档案，可计算 1/5/10 代直接留存与固定 FE 指标。
- 搜索 fullFE / objectiveOnlyFE / constraintOnlyFE，训练/推理/诊断前向行次与耗时。
- 五个本轮 LIRCMOP BC 问题中 CalObj/CalCon 的实际批次数、行次数和包含嵌套调用的耗时。其他问题明确标 instrumented=false，不能把零计数当成零成本。

CalCon 时间包含内部 CalObj 时间，两项不能相加当成总耗时。网络训练总时间包含反向传播。evaluationSeconds 是整个普通/引导子代构造阶段时间，真实 oracle 时间以 oracleCalls 为准。

configureComparison("fallback_only") 禁用训练/查询，20% 名额全走原 DE；configureComparison("pair_only") 在同档案、时序、配额和完整反馈下，用真实端点区间采样代替网络。默认始终为 cgan。初始化、普通算子、训练、查询和诊断的随机状态隔离；普通算子按代和算子分配流。

```matlab
addpath('Algorithms/Multi-objective optimization/CBS-CGAN/Support');
addCBSPaths(pwd);
folder = fullfile(pwd,'Data','PairGuideNative');
run_PairGuide_validation(pwd,10,struct('outputDir',string(folder)));
analyze_PairGuide_interval_validation(folder);
```

正式启动批次采用三组 × 五问题 × 种子 1:5，共 75 次，10 个独立进程、每进程一个计算线程。runner 记录每任务的 `*.progress.txt`、错误日志和总 `state.mat`，完成的结果可复用；协议或源码哈希不同则拒绝混用目录。

原 `Data/PairGuideInterval` 批次已于 2026-09-06 按用户要求停止，保留当时完成的 3 次结果和分布图。固定存档版必须使用新目录（上例 `PairGuideIntervalBounded`）；本次代码修改和性能验证不自动重启 75 次正式实验。

仅 `cgan` 的种子 1 输出生成分布图：首次真实使用及每 10000 FE（短程联调另含终点）。`render_PairGuide_interval_distribution` 在搜索结束后复用真实 P1/P2、端点和子代，对未评价的原生 c 离线调用 CalObj；第三图显示所选原生 c 与真实子代，不再显示“Constructed q”。缓存去重避免对同一原生候选重复评价；保存 `analysis/figures/<problem>/run_01/plot_data.mat`，离线成本与搜索 FE 分开。图上同时注明生成、消费、观测 FE；没有对应生成事件时明确留空。旧 v1 结果仍按其真实投影语义读取和绘图，不改写历史实验数据。

全部任务成功后自动运行 `report_PairGuide_interval_validation`：输出五张 P1 IGD/HV 对 FE 曲线（均值与标准差）、逐运行指标、逐问题汇总和同种子 CGAN 对两个基线的差值。配额占用率同时报告全程与模型首次完成之后；收紧率按唯一真实评价编号统计。AUC 遇到缺测或非有限指标时保持 NaN，不删掉失败阶段后计算较好的数值。五种子只用于首轮判断，不自动给出显著性或边界速度结论。

默认 5 问题 × 5 种子 × 3 组，N=100，FE=100000，D 由问题默认提供。代码准备好不代表已执行这些实验。新结果目录 PairGuideNative 不覆盖已有 PairGuideIntervalBounded 等历史结果。已有结果不能替代新机制下的重新对照。

analyze_PairGuide_interval_validation 不调用真实 oracle；输出原生入带率、修正量、配额使用/回填率、唯一评价收紧率、每引导 FE 对数收紧、净增量、P1/P2 存活、1/5/10 代留存、缺失方向惩罚 gap AUC 与同 FE IGD/HV。pair 覆盖不冒充独立 P1 边界距离认证。约束边界距离、最终 P1 加速及 CGAN 相对 pair-only 的收益仍须独立且计费的实验验证。

`nativeBandRate` 仍诊断 c 相对于真实配对的轴向 [0.4,0.6]、垂向 0.05 gap 命中，不参与筛选。新候选的条件边界距离上界改为 `max(norm(c-xf),norm(c-xi))/sqrt(D)`：在真实配对线段存在连续约束边界交点的条件下，这才适用于未修正的 c，不能继续使用旧 q 的 `sqrt(0.3625)*gap/sqrt(D)` 保证。`coarseIntervalProposals/certifiedIntervalProposals` 保留旧列名，v2 分别表示当前候选未获/获得 RMS 0.003 条件距离上界；只报告，不增加筛选门槛。`projectionMedian` 对新协议为 NaN，表示不存在投影步骤。`pendingPassRate` 与 `fullEvaluationRate` 分别使用提案数和 Pending 数作分母；服务/查询率只统计下一代确有引导名额的生成代。

audit_PairGuide_boundary_validation(folder,coverageTarget,valuableRefs) 是独立离线审计器：使用各组相同初始化中的真实不可行点，与缓存 P1 可行点构造线段，通过 30 步真实标签二分验证交点。报告到该交点的距离上界、RMS 容差 0.003 下的覆盖、首次/连续 5 检查点达标 FE、覆盖 AUC，以及独立 CalObj/CalCon 账单。结果不回传搜索；覆盖阈值和有价值方向集合必须在正式比较前预先指定。省略阈值时只输出距离与覆盖轨迹，不自动宣称达到某个速度标准。该方法不认证全局最近边界，最终解位于更优可行内部也不应被判为失败。

旧 initial_validation 分析脚本继续用于 PairGuideBall 历史格式；旧 Epoch/Sigma probe 接口已显式退役，不默默运行另一套策略。

## 验证

```matlab
test_CBS_pair_guide
test_CBS_pair_guide_archive_capacity
test_CBS_pair_guide_pool_front
test_CBS_pair_guide_training_observation
test_CBS_platemo_compliance
test_CBS_region_wgan_mainline
test_CBS_mainline_fingerprint
```

测试覆盖真实自动微分、共享噪声、更新预算、ID 无关内容变化、10 代保底、少量 pair 复用、细微真实收紧、迁移、原生坐标不变、两轮配额、下一代直接评价、严格 FE、嵌套 oracle 记账与观测独立性。当前默认检查 CCMO 选择池、完整候选先筛后配、跨可行点的不可行端支配排除、端点复用、反馈后的资格删除、500 对容量上限，以及默认与显式 ccmo/1 的完整种群轨迹重放。保留旧算法指纹回归。

2026-09-07 不可行端筛选与删除规则验证：六项相关回归通过；最小反例在旧代码失败、修正后通过，覆盖近处不合格端排除后选择远处合格端、跨可行点全局支配、同一不可行端复用、严格支配与目标相等、真实反馈先归因后删除、空档案与容量竞争。五问题 seed 1 的六节点共 30 个已有真实状态离线重建通过，所有保留端点满足新规则，无额外搜索评价或随机状态变化。6001 FE 端到端轨迹和检查点观测独立性通过；新增检查点计数与旧证据缺项 NaN 兼容性通过。七个相关 MATLAB 文件静态检查无消息，git diff --check 通过。生产网络、损失和预算未修改，既有旧算法指纹差异未在本次重跑或修复；没有重跑正式实验。详情见 [存档修正与训练偏移调查](../../../../experiments/PairGuideDominanceDrift_20260907/RESULTS.md)。

2026-09-07 默认切换验证：上述前六项测试通过。`test_CBS_pair_guide_training_observation` 使用 LIRCMOP5_BC、N=100、默认 D、seed=1、6001 FE，训练预算缩至首训 2 次/重训 1 次生成器更新以验证流程；默认入口与显式 ccmo/1 配置的完整种群历史、真实评价记录和随机状态完全一致，并实际覆盖训练、生成和下一代直接使用。8 个修改 MATLAB 文件的 `checkcode` 无新增消息（pool_front 测试原有一条 isscalar 性能建议保留），`git diff --check` 通过。`test_CBS_mainline_fingerprint` 当前 PairGuide 重放通过，上一任算法仍在第 41 行的既有 IGD 指纹断言失败；未修改旧算法或预期常数。日志：[机制与容量](/tmp/pairguide-default-ccmo-f1-20260907/targeted.log)、[端到端与旧指纹](/tmp/pairguide-default-ccmo-f1-20260907/regression.log)、[静态基线对照](/tmp/pairguide-default-ccmo-f1-20260907/static-baseline.log)。本次只切换默认并回归，没有重新运行正式实验或验证 IGD 优劣。

2026-09-06 原生候选/P1 资格修正验证：先让 `test_CBS_pair_guide` 在旧投影实现上失败，报错“Selection must not project or otherwise move a native CGAN proposal.”；修正后 `test_CBS_pair_guide`、`test_CBS_pair_guide_archive_capacity`、`test_CBS_pair_guide_training_observation`、`test_CBS_platemo_compliance`、`test_CBS_region_wgan_mainline` 五项通过。九个相关核心/支持/测试文件的 checkcode 无消息，`git diff --check` 通过。

短程联调使用默认网络预算和默认 D=30，LIRCMOP5_BC、N=100、seed=1、5000 FE，cgan/fallback_only/pair_only 三组全部完成；CGAN 完成 5 次训练、5 次查询、96 个真实原生引导评价。三组分别精确计入 5000 完整 FE、10000 CalObj 行、5000 CalCon 行。已对保存的所有查询复核 `rawDecs(keepIdx,:)==Pending.decs==下一代引导 childDecs`；新图 Pending 与真实子代目标坐标一致，两张图离线总计 1000 CalObj 行，未重复计算原生池副本。图已目视检查。结果在 [短程验证目录](/tmp/pairguide-native-validation-20260906)，日志见 [针对性回归](/tmp/pairguide-native-targeted.log)、[三分支联调](/tmp/pairguide-native-smoke.log)、[保存数据与绘图复核](/tmp/pairguide-native-artifact.log)。复现联调：

```matlab
O = struct('problems',"LIRCMOP5_BC",'seeds',1,'N',100,'maxFE',5000, ...
    'outputDir',"/tmp/pairguide-native-validation-20260906");
run_PairGuide_validation(pwd,1,O);
```

保留一项失败：`test_CBS_mainline_fingerprint` 的当前 PairGuide 重放部分通过，但上一任 CBS_RegionWGAN_GP 的固定 IGD 指纹不匹配。单独新进程运行上一任算法也未匹配：实际 IGD 为 1.346124110619686，预期为 1.3473816458691643。独立调用记录没有 PairBoundary/PairGuideCore；旧核心、BoundaryWGAN、环境选择、CalFitness 与记账文件相对本次修改前 SHA-256 均一致。没有修改旧算法或放宽预期值；该历史指纹差异的具体来源未在本次范围内继续修复。见 [独立复核日志](/tmp/pairguide-legacy-probe.log)。本次完成的是机制与工具验证，未重新运行新版本 75 次正式实验，不能据此宣称 IGD 改善或保持不变。

2026-09-05 本机 MATLAB R2025b 验证记录：以上五项回归通过；最后的真实近邻反馈修正后，重新通过 pair_guide 与 training_observation 两项。五个问题在不加载 PairGuide 核心目录时的独立完整评价也通过；计费钩子不改变问题公式或独立调用能力。核心及新增支持脚本的 checkcode 和 git diff --check 通过。

默认网络与训练预算、N=100、默认 D、seed=1 的短运行：LIRCMOP10_BC 在 8000 FE 内训练 17 次、真实引导评价 340 个；LIRCMOP12_BC 为 25 次、498 个。LIRCMOP5_BC 的 5000 FE 三组脚本联调分别产生 cgan 160 个、fallback_only 0 个、pair_only 168 个真实引导评价；三组均严格计入 5000 FE，分析脚本及独立离线边界审计完成。这些是机制与工具联调证据，不是正式性能结论；五问题、五种子、100000 FE 的完整对照尚未运行。

来源抓取命令：

```sh
smart-search fetch 'https://chatgpt.com/share/6a9bf4a7-f0f8-83ea-8b32-8365ea8dc46c' --format markdown
```
