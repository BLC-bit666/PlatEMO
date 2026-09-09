# PairGuide 严格对应审查（2026-09-05）

**审查结论：此前实现不是完全一比一。此次修正了三处运行或观测偏差，并补齐粗区间与绝对窄边界带的诊断区分。核心运行规则已逐项核对，但不能把这次代码验证表述为“完整对话的全部实验和审计要求均已完成”。**

基准为[修改对话](https://chatgpt.com/share/6a9bf4a7-f0f8-83ea-8b32-8365ea8dc46c)的最后一轮修订；前文未撤销的网络、几何、时序、成本和验证要求同时核对。已阅读原任务 [GPT复现](codex://threads/01a07164-531c-7431-8b9f-bd2cb4ab576f)，但未将原任务“完成”的声明当作本次验证证据。

当前入口是 `PairGuide < PairGuideCore`。`CBS_PairGuide` 指该主线，不是将 `CBS_RegionWGAN_GP` 改名复用。审查针对工作区现状；此前已有的未提交修改及 Deliverables 冻结副本均予以保留。

## 本次确认并修正的问题

### 1. 跨 pair 更新反向检索邻域，实际超过五个方向

修正前，`tightenRows` 使用 `neighbors(Archive.ref, refs(k))`：寻找“将当前评价点方向列入邻域的所有 pair”，而不是从评价点方向向外查找五个邻居。角度近邻关系不对称，反向集合不受五个方向上限约束。

独立复现使用九个参考方向：方向 5 的合法邻域是 `{3,4,5,6,7}`，原代码却更新了方向 1 至 9；方向 4、6 也分别越界更新了方向 1、9。该偏差直接改变档案反馈、训练内容和后续搜索。

现使用正向邻域。原 matched pair 的直接反馈仍优先处理，仍不因评价点瞬时参考归属变化而被否决；随后对其他 pair 才应用五方向检索。

位置：[PairBoundaryArchive_RC.m:168](<Core/PairBoundaryArchive_RC.m>)。回归同时覆盖普通点的五方向限制和原 pair 归属豁免。

### 2. 剩余 FE 不等于下一代仍有引导名额

修正前仅检查 `Problem.FE < Problem.maxFE`。在 `N=100, maxFE=5001` 下，5000 FE 时仍生成一批候选；下一代 P1 仅分到一个评价名额，其引导配额为零。这违反“没有下一代使用预算时，不训练、不生成 donor”。部分尾批也可能生成超过下一代配额的 Pending。

现先按下一代真实批量计算 P1 引导配额；配额为零时跳过训练与生成，非零时按该配额限制 Pending。完整代仍保持 25/55/20 与 25/75。

位置：[PairGuideCore.m:856](<Core/PairGuideCore.m>)。分析脚本的查询率、服务率和正使用率分母同步排除没有相应名额的尾批。

### 3. 汇总检查点仍使用目标 FE 之后的状态

新的 `objectiveSpaceSnapshots` 已采用目标 FE 之前的缓存，但旧汇总路径 `auditNotTerminated` 仍在越过目标后写入当前状态。同一个 `5001 FE` 复现中，十个汇总检查点有九个使用未来状态。

现汇总检查点也记录目标 FE 之前最近的已完成状态，不跨代插值，不额外评价。早于首次已完成初始化的目标保留缺测。

位置：[PairGuideCore.m:661](<Core/PairGuideCore.m>)。

### 4. 诊断未明确区分粗区间与绝对窄边界带

原生 `c` 满足相对轴向、横向条件，只能说明相对当前 pair 较窄。原分析只有合并的 `nativeBandRate`，没有按对话的绝对距离尺度区分粗区间。

现记录 `sqrt(0.3625)*gap/sqrt(D)` 的条件几何距离上界，按 RMS 阈值 0.003 分为 `coarseInterval` 和 `boundaryCertified`。这只是诊断，不参与候选拒绝、档案更新或训练触发，也不是对全局最近边界距离的测量。

分析新增粗区间提案数、满足该距离上界的提案数、原生且满足该上界的入带率，以及 Pending 通过率和完整评价率。旧 `nativeBandRate` 保留兼容含义：**仅为相对几何入带率**。

位置：[PairBoundaryArchive_RC.m:436](<Core/PairBoundaryArchive_RC.m>)、[analyze_PairGuide_interval_validation.m](<Support/analyze_PairGuide_interval_validation.m>)。

## 最终主线逐项对应

以下“符合”表示当前代码与运行验证所能支持的结论，不表示性能假设已被证明。

| 要求 | 实现证据 | 判定 |
| --- | --- | --- |
| 保持 PairGuide 身份与独立核心 | `PairGuide.m`，独立旧算法指纹回归 | 符合 |
| 双种群从同一个真实 Union 独立选择 | `runMainline` 两次 `EnvironmentalSelection_CBS` | 符合 |
| P1 25/55/20、P2 25/75、引导缺额走原普通 DE | `pairOffspring`、`scopedBackbone`、`OperatorDEDistinct_CBS`；完整代与尾批断言 | 符合 |
| 代末生成、下一代原样完整评价 | Pending 保存 q 与生成代；消费时检查实际决策与 Pending 完全相等 | 符合 |
| 无下一代使用预算则不训练、不生成 | 下一代引导配额保护 | 本次修正 |
| 六项 pair 语义状态 | id/ref/xf/yf/xi/yi；gap、active 为派生缓存 | 符合 |
| 历史两端共同参与更新、可逆资格 | `updateArchive` 统一使用 Historical.xf/xi；不存在年龄淘汰或不可逆恢复标志 | 符合 |
| 每方向最多一个 active pair，先竞争后分配 ID | 方向质量、gap 竞争与末尾 active 选择 | 符合 |
| 原 pair 优先反馈；不依赖 P1 存活 | 环境选择前 `replaceEndpoint` | 符合 |
| 其他 pair 的检索限制在五方向邻域 | `tightenRows` 正向检索；九方向复现 | 本次修正 |
| 任意超过 1e-12 的真实收紧可接受 | `newGap < oldGap - tolerance`；细微收紧回归 | 符合，无 39.8% 准入线 |
| 首训至少 8 对，后续服务至少一对 | 训练与 `Status.useModel`、查询资格分别判断 | 符合 |
| 首训 200、重训 20 次生成器更新 | `trainModel` 以 `eventUpdates` 限额；batch 最大 16 对，支持尾批 | 符合；公开配置仍可显式覆盖实验参数 |
| 内容变化相对上次完成训练快照 | `changesSince` 按方向并集及归一化端点移动计算；只换 ID 不变 | 符合 |
| Delta >= 0.2 或间隔 >= 10 代；每代最多一次尝试 | `trainIfNeeded` | 符合，无固定 2000 FE 冷却 |
| 全部当前 active 完整 pair 构成训练集 | `buildTrainingData`；无永久留出、回放库、多重接受阈值 | 符合 |
| 固定噪声前后诊断、失败保留旧模型 | `pairModelDiagnostics` 与数值失败回退分支 | 源码符合；诊断是拟合检查，不是泛化认证 |
| 两端真实绝对归一化表示、共享 z | Critic、Generator 对抗与几何项、推理、配对诊断均配对前向 | 符合 |
| gap 相对 Lend/Ldelta，LG=Ladv+10Lend+Ldelta | `generatorGradients`；h 下限平方为 `1e-6*D` | 符合 |
| 两层 32、zDim=6、GP=10、Adam 1e-4/(0,0.9)、5:1 | 默认配置与网络构建 | 符合 |
| 训练/推理 sigma 初值同为 0.3 | `mainlineDefaults`、WGAN options | 符合 |
| 500 个双端提案、alpha 属于 [0.4,0.6] | 分层 query、`sampleByCondition`；1000 个端点前向行次 | 符合，无 side=0.5 查询 |
| t 限 [0.4,0.6]、垂距/gap <=0.05、盒内最大标量 beta | `selectCandidates`，几何回归 | 符合 |
| 原生输出与最终 q 分开，记录生成时端点/ID/ref/版本/噪声 | `Evidence.queries` 与 Pending | 符合 |
| 不作目标预筛、全局 raw 支配过滤或二次中点映射 | 主链只做数值/几何/去重/排序，随后完整评价 | 符合 |
| 先不同 pair，再第二轮，每对最多两个 | `selectCandidates` 和消费端双重上限 | 符合 |
| 搜索 FE 与真实 oracle/网络成本分别记录 | `PairGuideCost_RC`、五个问题钩子、Evidence 计数 | 符合；未插桩问题明确不支持实际调用账单 |
| 生成/消费/观测时刻分开，快照复用缓存 | `pairEvidenceSnapshots` 与汇总检查点 | 本次补齐汇总路径 |
| 原生相对窄带与小绝对距离上界分开 | coarse/certified 诊断及分析列 | 本次补齐 |
| 同种子 fallback-only、pair-only，默认维度 | 三组 runner；初始化、普通算子、训练、查询、诊断随机流隔离 | 脚本及短运行通过 |

## 仍未完成的完整对话交付

1. **正式因果实验尚未完成。** 当前五问题、五种子、100000 FE 的三组实验代码存在，本次只作短运行验证；不能据此判定 P1 加速、最终 IGD/HV 改善或 CGAN 独立价值。
2. **前文四组设计中的 A 组缺失。** 当前 runner 只接受 cgan/fallback_only/pair_only。此前 PairGuideBall 或 Deliverables 中的旧机制材料仍是冻结证据，未提供按统一日志、预算和随机流重跑的旧机制 A 组。末轮没有明确撤销该对照；按完整对话验收，这项仍不能记为完成。
3. **冻结同一批真实 pair 的 CGAN/pair-only 比较尚无专门执行流程。** 已保存提案足以支持后续部分分析，但当前全流程 C/D 对照不能替代该项机制诊断。
4. **详细几何和后代归因报告不完整。** 当前保留逐提案的原始端点、c/q、轴向、横向和修正量，但汇总脚本没有完整输出每 pair 的 5%–95% 区间、线段距离及带宽/gap 报告；仅记录直接存活与 1/5/10 代原个体留存，没有普通算子父代关联，因此不能重建完整的“引导点经 P2 后代再影响 P1”贡献链。
5. **P1 速度验收仍须预先定义有价值方向和覆盖目标。** 独立审计器提供真实标签二分得到的距离上界、覆盖和单独成本；它不认证全局最近边界，也未实现所有问题上“越过边界进入更优可行内部”的到达 FE 报告。预注册速度、收敛、成本等验收不能用训练损失替代。

这些是审计或实验交付缺项，不应通过增加算法门槛、撤回最终修订或宣称已有性能收益来掩盖。

## 本次验证与可复查证据

- 修改前五项现有回归全部通过，但独立反例仍发现上述问题，说明原回归覆盖不足。
- 三处行为修正后，重新通过 `test_CBS_pair_guide`、`test_CBS_pair_guide_training_observation`、`test_CBS_platemo_compliance`、`test_CBS_region_wgan_mainline`、`test_CBS_mainline_fingerprint`。
- 加入粗/绝对窄带诊断后，再次通过最相关的 pair_guide 与 training_observation 两项。
- 本次修改的五个 MATLAB 文件 `checkcode` 均为零诊断；`git diff --check` 通过。
- 当前默认网络/训练参数、LIRCMOP5_BC、seed=1、N=100、默认 D=30、5001 FE 的三组脚本联调通过；三组初始化完全一致，均实际执行 10002 行 CalObj、5001 行 CalCon；正常搜索 objective-only/constraint-only 均为零。

| 短运行组 | 完整评价引导子代 | 全程配额占用率 | 原生相对入带率 | 原生且满足绝对距离上界的入带率 |
| --- | --- | --- | --- | --- |
| cgan | 160 | 33.33% | 0% | 0% |
| fallback_only | 0 | 0% | 不适用 | 不适用 |
| pair_only | 168 | 35.00% | 100% | 0.82% |

短运行中 cgan 的 4000 个提案均属于粗区间。这再次说明“候选构造正确”和“网络已学到窄边界带”是不同结论；一个短种子既不能证明效果，也不足以否定最终长程效果。

复查命令（仓库根目录下）：

```matlab
addpath('Algorithms/Multi-objective optimization/CBS-CGAN/Support');
addCBSPaths(pwd);
test_CBS_pair_guide;
test_CBS_pair_guide_training_observation;
test_CBS_platemo_compliance;
test_CBS_region_wgan_mainline;
test_CBS_mainline_fingerprint;
```

```sh
smart-search fetch 'https://chatgpt.com/share/6a9bf4a7-f0f8-83ea-8b32-8365ea8dc46c' --format markdown
matlab -batch "run('/tmp/pairguide_review_controls.m')"
git diff --check
```

共享对话正文：shared-dialogue.md（原工作区文件，未打包）。复现与测试摘录：verification.txt（原工作区文件，未打包）。三组短运行汇总：[interval_summary.csv](/tmp/pairguide-review-controls-20260905/interval_summary.csv)。本次没有提交、推送或修改旧算法身份。
