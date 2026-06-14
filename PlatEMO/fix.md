A. 证据表

- 代码证据
  - `BuildBoundaryTargetTriples_BDG.m` 明确定义训练三元组为 `(x_b, y_t, d_t)`，其中 `x_b` 是 **AF 侧可行边界决策**，并且代码直接设 `XBoundary = AFDec`；condition 是 `AFNorm` 与 `Direction = AINorm - AFNorm`，再可选拼接 `RefToken`。这不是“边界线上的点”，而是“边界 pair 的可行端点”。
  - `CCMO_GAN_BDG.m` 训练时调用 `BuildGANTrainingSet_BDG -> BuildBoundaryTargetTriples_BDG -> FilterBoundaryTargetTriples_BDG`，然后把 `GANTrainDecs` 送入 `BoundaryGAN_BDG('traindirectboundary',...)`；也就是当前主线确实是在学这些 `AFDec`。
  - `BoundaryGAN_BDG.m` 当前 target-conditioned 合约是 `G(z,y_t,d_t)->x, D(x,y_t,d_t)`；判别器只看三类 batch：`matched real / mismatched real / fake matched`。
  - `BoundaryGAN_BDG.m` 的 generator loss 目前只有条件对抗项：`lossG = -mean(log(D(G(z,c),c)))`，没有任何“落在 AF-AI 线段 / 投影点 / 可行-不可行边界零层集附近”的显式项。
  - `BoundaryGAN_BDG.m` 采样时确实用随机 `z ~ N(0,I)`，并从存储的 condition 中抽样，再生成 raw decisions。
  - `TrainDirectBoundaryGAN_BDG` 里 `XAI` 虽然传进来了，但训练循环只对 `XPositive(=AF)` 做 D/G 更新；`XAI` 仅出现在 `pretrainDiag` 诊断里，不参与当前 target-conditioned 训练目标本身。
  - `FilterBoundaryTargetTriples_BDG.m` 注释写得很清楚：它只对 **CGAN training set** 做 filter/weight，**不修改 AF/AI archive 本体**；`condition_knn`、`local_mad_weight`、`random_keep`、`ai_dom_only` 都是数据选择/加权，而不是边界线约束。
  - `UpdateBoundaryArchive_BDG.m` 当前 archive 主要做的是：从 feasibility flip pair 和 cross-pop nearest pair 收集候选，再做 Pareto filter、pair direction gate、pair ref gate，并按 objective gap/rank 评分、按 ref 保留。它是在构造“pair skeleton”，不是在恢复真实边界线。
  - `UpdateBoundaryArchive_BDG.m` 还有 shrink guard，会在 archive 太小的时候保留旧 archive。这说明 archive 逻辑强调稳定性，但不提供更细的边界几何监督。
  - `run_CCMO_GAN_BDG_boundary_diagnostics.m` 的 runner 就是为当前 `GND_keep80_localMAD_nGen20` 主线做 diagnostics，targets 是 `[30000 70000 100000]`，并开启 `normal / fixed_z / z_sweep` 诊断。
  - 诊断指标中明确包含 `condition_entropy`、`condition_top10_mass`、`obj_seg_dist90`、`dec_seg_dist90`、`own_pair_nearest_rate`、`within_obj_width90_median`、`within_dec_width90_median` 等，正是你现在关心的“是否坍缩 / 是否贴 pair / 同一 condition 下是不是厚带”。
  - `test_CCMO_GAN_BDG_target_conditioned.m` 验证了 `GND_keep80_covGate` 的门控语义：`cganTrainMinRefCov=0.50`、`cganTrainMinTargetTriples=50`，且 under-covered 代会 `skip_reason == "target_coverage_gate"`、`refresh_flag == 0`。这说明 covGate 本质上是“是否允许训练”的门，不是几何约束。
  - 当前注入逻辑已经只保留可行 GAN 候选：`keep = find(feasible)`。因此后处理/注入筛选本来就在做，不能回答“raw GAN 是否本身贴边界线”这个核心目标。
- 实验证据
  - bundle 顶部给出的本地总结已经直接指出：`localMAD_nGen20` 没解决 boundary adherence；condition sampling 没坍缩，但 fixed-z 仍落不到自身 AF-AI pair，z-sweep 在同一 condition 下仍是厚分布；根因更像是“AF-like decision distribution under weak target hints”，不是 boundary-line parameterization。
  - FE=100000 的 diagnostics 汇总：`mean_normal_condition_entropy=0.97216`、`mean_normal_condition_top10_mass=0.29833`、`mean_normal_feasible_rate=0.34633`、`mean_normal_obj_seg_dist90=2.34244`、`mean_normal_dec_seg_dist90=0.09236`、`mean_fixedz_obj_seg_dist90=2.19496`、`mean_fixedz_own_pair_nearest_rate=0.09708`、`mean_zsweep_feasible_rate=0.34148`、`mean_zsweep_within_obj_width90_median=0.40475`、`mean_zsweep_within_dec_width90_median=0.22201`。这组数值直接说明：coverage 不塌，但 pair 锚定差，同一 condition 下仍厚。
  - `localMAD_nGen20` vs `localMAD_nGen50`：`SegDist90` 仅从 `1.7283` 变到 `1.7852`，而 feasible 从 `0.278` 升到 `0.3088`，说明多打 GAN 样本并没有把云压成线。
  - `GND_keep80` / `covGate`：`GND_keep80` 的 `SegDist90=6.8754, Feasible=0.359`；`GND_keep80_covGate` 的 `has_GAN=0.5333, BoundaryHit=0.10125, Feasible=0.22969`。这说明 coverage gate 牺牲了训练刷新频率和可行率，但没有把 raw GAN 变成线。
  - `C4_keep60 / 80 / 90`：stage100000 分别约为 `1.3675 / 2.5460 / 2.6401` 的 `SegDist90`，对应 feasible 约 `0.326 / 0.3147 / 0.2942`。keep ratio 会改变距离和可行率，但没有出现“清晰边界线”状态。
  - `weightedCond_nGen20` 在 stage100000 的 `SegDist90=2.3191, Feasible=0.3413`，可行率不错，但贴线仍不如 `localMAD_nGen20`。这再次说明“采样/权重”不是主因。
  - 你上传的代表性 diagnostic 图也一致显示：红色 GAN 星点通常围绕黄线段形成散点云或厚带，而不是沿线收缩成细 1D 结构。
- 外部资料证据
  - cGAN 的基本形式是把 condition 同时喂给 G 和 D；但这并不自动保证 condition 成为“几何坐标”，更不保证 `z` 变得可控。Mirza & Osindero 只是定义了 conditional GAN 框架。([arXiv](https://arxiv.org/abs/1411.1784))
  - pix2pix 明确报告：即便加入 dropout noise，输出也只有“minor stochasticity”，并把“让 conditional GAN 产生高随机且可控的输出”当成未解决问题。这说明在很多 conditional 生成任务里，`z` 的作用要么被忽略，要么不具备明确语义。([CVF开放获取](https://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf))
  - BicycleGAN 进一步说明：若想让 latent 对输出模式有可解释控制，需要显式鼓励 latent-output 的双向一致性/可逆性；否则 latent 的语义是失控的。([arXiv](https://arxiv.org/abs/1711.11586))
  - 在未知/物理约束下，BE-CBO 与 2-OPT-C 都强调：最优点常常位于可行/不可行边界附近，且高效方法需要主动围绕边界而不是只在可行域内部保守搜索。([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v235/tian24g.html))
  - BE-CBO 还给出一个很贴近你问题的路线：用更强的边界建模器（Deep Ensembles 而非 GP）抓复杂边界，并用 uncertainty-aware band 在边界附近搜索。([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v235/tian24g.html))
  - 在“边界是目标”的学习任务里，boundary loss / contour-distance loss 的核心思想是：只做 region/adversarial supervision 不够，要把“离边界多远”直接写进损失。([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v102/kervadec19a.html))
  - 在多目标生成方向，GAN-LMEF 说明 GAN 可以学习低维 manifold 并在 manifold 上插值生成高质量点；但其有效性依赖于 manifold supervision 本身的质量。([arXiv](https://arxiv.org/abs/2101.02932))

B. 五个问题逐项回答

1. 问题 1：G 的输入使用随机采样 z 是否合理？是否是导致厚带/散点云的原因？

- 结论：部分是。`z` 是厚带的放大器，不是第一根因。
- 代码证据：当前 boundary mode 的生成器确实是 `G(z,c)`，采样时每次都从高斯随机取 `z`。
  同时，当前 generator loss 只有 `-log D(G(z,c),c)`，没有任何把 `z` 约束成“边界线内合法一维自由度”的项。
- 实验证据：FE=100000 时，`normal_condition_entropy=0.97216` 与 `normal_condition_top10_mass=0.29833` 说明 condition 覆盖并未坍缩；但 `fixedz_obj_seg_dist90=2.195`、`fixedz_own_pair_nearest_rate=0.09708` 说明把 `z` 固定住以后，样本仍大多不落在自己的 pair 上；同时 `zsweep_within_obj_width90_median=0.40475`、`zsweep_within_dec_width90_median=0.22201` 说明在同一 condition 下扫 `z` 会形成明显厚带。
- 机制解释：
  - 如果 condition 本身已经唯一决定“边界线上哪个点”，`z` 最多只会被网络忽略，或只表达很小的残余模态。pix2pix 的经验就是：conditional generator 往往根本不给 `z` 真正的语义，输出随机性很弱。([CVF开放获取](https://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf))
  - 如果 condition **不能** 唯一决定输出，而你又不给 `z` 加任何 latent-consistency / mutual-information 约束，那么 `z` 就会变成“未被定义的额外自由度”，厚带就会自然出现。BicycleGAN / InfoGAN 的意义正是：要让 latent 真有可控语义，必须显式约束 latent-output 对应关系。([arXiv](https://arxiv.org/abs/1711.11586))
  - 但你这里 even fixed-z 仍然不对，说明“只有 z 导致厚带”这个判断不成立。第一根因是 **condition 与训练目标共同没有定义出唯一的一维边界坐标**。
- 修改建议：
  - 在“边界生成模式”里，把 `z` 从语义变量降级为 0：直接令 `zDim=0`，或训练/采样都用零向量。
  - 不要单独把“去 z”当主线方案；必须和 **新 condition（加标量坐标 t）+ 新目标（不再只用 AFDec）+ 新损失（加重构）** 一起改。
  - 如果你坚持保留 `z`，那就必须额外加入类似 BicycleGAN / InfoGAN 的 latent-output consistency；这条路更复杂，不是我推荐的主线。([arXiv](https://arxiv.org/abs/1711.11586))
- 最小验证实验：
  - 不改其他任何代码，只做一个 `zDim=0` 的 current-pipeline 对照。
  - 跑 `DASCMOP1_BC, DASCMOP2_BC, LIRCMOP7_BC, LIRCMOP8_BC`，`runs=1:3`，看 `fixedz_obj_seg_dist90 / fixedz_own_pair_nearest_rate / zsweep_within_obj_width90_median`。
  - 成功判据：`zsweep` 宽度明显下降；失败判据：即使宽度下降，`fixedz_own_pair_nearest_rate` 仍很低。后一种结果反而更有价值，因为它会直接证明：`z` 不是第一根因。

1. 问题 2：当前 CGAN condition 是否合理？是否足以作为“边界线坐标”？

- 结论：否。

- 代码证据：当前 condition 的定义就是 `ConditionData = [AFNorm, Direction, RefToken]`，其中 `Direction = AINorm - AFNorm`。同时训练 target 只有 `XBoundary = AFDec`。

- 实验证据：condition 覆盖很广，但 `fixedz_own_pair_nearest_rate` 依然只有 `0.09708`，说明“给定 condition 后生成到自己 pair 上”这件事基本没有发生。

- 机制解释：

  - `[AF objective, AI-AF direction]` 最多定义了“一条 segment family”，没有定义“这条 line 上的哪一个位置”。缺失的就是标量坐标 `t∈[0,1]`。
  - 更严重的是：即便 condition 想表示整条 pair 线段，当前 target 却永远是 `AFDec`，等价于把所有 pair 都监督成 `t=0`。
  - 所以当前 condition 不仅“不足以当边界线坐标”，还和当前 target 一起把模型推向了“学 AF family 的 endpoint distribution”。
  - 这和条件生成文献的经验一致：condition 必须承载真正的可识别语义；若想保留多模态，还需要额外约束 latent/condition 到 output 的对应关系。([arXiv](https://arxiv.org/abs/1411.1784))

- 修改建议：

  - 把 condition 改成 **`[AFNorm, Direction, t, RefToken]`**，其中 `t` 是你缺失的“沿 pair 线段的一维坐标”。

  - 关键不是凭空采样 `t`，而是从**已有评估过的真实可行点**里给出 `t`：
    $$
    [
    t_i(x)=\mathrm{clip}\frac{(\bar f(x)-\bar f_{AF,i})^\top(\bar f_{AI,i}-\bar f_{AF,i})}{|\bar f_{AI,i}-\bar f_{AF,i}|^2},,0,1
    ]
    $$
    同时仅保留那些 objective 上靠近 pair segment 的真实可行点。

  - 这样 condition 才从“pair 提示”变成“pair + 线坐标”，真正能参数化一条 1D 边界流形。

- 最小验证实验：

  - 在 `BuildBoundaryTargetTriples_BDG.m` 里只增加 `t`，其余先不变。
  - 仍然用当前 `AFDec` 为 target，先看 `fixedz_own_pair_nearest_rate` 是否提升。
  - 成功判据：该指标显著提升；失败判据：提升有限。这会告诉你“只补 t 但 target 仍错”还不够，从而自然推进到问题 3 的解法。

1. 问题 3：当前 CGAN 训练方式，尤其损失函数，是否合理？是否与目标不一致？

- 结论：否，目标不一致，而且这是主因。

- 代码证据：

  - 训练正样本是 `XPositive = XTrain = AFDec`，不是边界线样本，更不是 boundary projection。`XAI` 不参与主训练，只用于 pretrain/diagnostic。
  - 判别器做的是 `matched real / mismatched real / fake matched` 的条件真假判别；generator loss 只有条件对抗项。
  - 默认主线没有启用 `boundary_quality_eval` 标签，训练依然是标准 binary real label。

- 实验证据：

  - FE=100000 时 `normal_obj_seg_dist90=2.34244` 很大，但 `normal_dec_seg_dist90=0.09236` 反而不算特别差，说明“靠近 AF-AI 决策段”并没有自动变成“objective 空间边界线”。
  - `GND_keep80` feasible 达到 `0.359`，却有很差的 `SegDist90=6.8754`；`weightedCond_nGen20` feasible `0.3413`，`SegDist90=2.3191`；这说明可行与贴边并不等价，现有损失在学“看起来像 AF-like feasible points”，不是“边界线”。

- 机制解释：

  - 你真正想学的是：**给定 pair 坐标，生成一条可行边界流形上的点**。
  - 但当前损失学的是：**给定弱 condition，生成一个能骗过 D 的 AF-like feasible endpoint**。
  - 这两者不是同一个任务。只要 D 认为样本“像这个 condition 对应的 AF 家族”，G 就赢了；D 不检查“是否在自己的 AF-AI 线段上”，更不检查“是否是贴可行/不可行接口的 boundary point”。
  - 这和 boundary-aware learning 的共识一致：如果目标是“边界”，只做 region/adversarial 类损失不够，必须把 boundary distance / boundary consistency 直接写进监督。([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v102/kervadec19a.html))

- 修改建议：

  - 把当前 adversarial-only CGAN 改成：**重构为主、GAN 为辅的边界流形条件生成器**。

  - 具体做法：

    1. 在 `BuildBoundaryTargetTriples_BDG.m` 中，不再输出单个 `AFDec`，而是输出多条 `(x^*, c_i, t_i)` 样本，其中 `x^*` 来自**当前已评估真实可行点**中、objective 上贴近 pair segment 的点。

    2. 在 `BoundaryGAN_BDG.m` 中，边界模式令 `z=0`，用 `G(c_i,t_i)->x`。

    3. generator 损失改为
       $$
       [
       L_G = L_{adv} + \lambda,\mathrm{Huber}(x,;x^*)
       ]
       $$
       其中 `Huber`/`L1` 是主损失，`L_adv` 只做 realism regularizer。

  - 这样改后，GAN 不再试图“自由想象一个 AF-like 点”，而是被迫把同一 `(pair,t)` 映射到真实、已评估、贴 segment 的窄分布上。

  - 这是我最推荐的方向，因为它只引入一个新增标量 `t` 和一个固定重构项，不需要堆很多模块。

- 最小验证实验：

  - 同一套 pair skeleton，下做 `adv-only` vs `adv+Huber` 两个版本。
  - 跑 4 个代表问题、`runs=1:3`。
  - 成功判据：`obj_seg_dist90` 明显下降，同时 `zsweep_within_obj_width90_median` 明显收缩；失败判据：只提升 feasible，不提升 `own_pair_nearest_rate` 与 width。

1. 问题 4：边界存档构造、训练数据选择依据是否合理？是否是问题原因？

- 结论：部分是。它们对“找候选 pair”和“降噪”是合理的，但不是“生成一条边界线”的核心解法。
- 代码证据：
  - archive 的职责是从 flip pair / nearest pair 收集候选，做 Pareto / direction / ref 过滤，再按 objective gap 与 rank 打分保留。
  - `FilterBoundaryTargetTriples_BDG` 只是在训练集上做 `condition_knn`、`local_mad_weight`、`random_keep`、`ai_dom_only`。它并不产生新的 boundary 几何标签。
  - `covGate` 只是训练门：coverage 不够就跳过 GAN refresh。
- 实验证据：
  - `C4_keep60` 的 `SegDist90=1.3675` 明显优于 `C4_keep80=2.5460` 和 `C4_keep90=2.6401`，说明 archive/train filter 当然会影响数值。
  - 但 `keep60` 也只是“更窄一些”，仍远不是一条线。bundle 顶部总结也直接说 none of `C4/GND/covGate/localMAD/weightedCond` turns generated points into a clean line。
  - `localMAD_nGen20` 与 `weightedCond_nGen20` 比较，前者 `1.7283/0.278`，后者 `2.3191/0.3413`，说明权重/过滤能在“贴线”和“可行率”之间摆动，但解不了目标失配。
  - `covGate` 还会把 `mean_has_GAN` 压到 `0.5333`，同时 feasible 掉到 `0.2297`；门控更像“少训一些”，不是“训对几何”。
- 机制解释：
  - AF/AI archive 的价值在于提供 **boundary pair skeleton**。
  - 但 skeleton 不是 geometry supervision。你现在的问题不是“pair skeleton 完全错了”，而是“pair 只有端点，没有线坐标；训练 target 也只有端点”。
  - 所以 archive/filter 是二阶因素，`BuildBoundaryTargetTriples_BDG` 与 `BoundaryGAN_BDG` 才是一阶瓶颈。
- 修改建议：
  - `UpdateBoundaryArchive_BDG.m` 基本保持不动，只把它当 pair proposal 机制。
  - 真正该改的是：在 `BuildBoundaryTargetTriples_BDG.m` 里，基于这些 pair 去当前已评估 feasible 集里挖 **pair-local near-segment real samples**，再生成 `(pair,t)->x^*` 数据。
  - `FilterBoundaryTargetTriples_BDG.m` 继续保留，但角色变成“对这些 real near-segment samples 去噪”，而不是主修复器。
  - 我建议把 `C4_keep60` 当 pair skeleton 默认，因为它是当前 stage100000 里 `SegDist90` 最好的简洁分支。
- 最小验证实验：
  - 固定 archive 逻辑为 `C4_keep60`，比较：
    1. 旧 endpoint triples；
    2. 新的 pair-local multi-`t` triples。
  - 如果 2) 大幅优于 1)，就能把根因从“archive/filter 不够好”排除掉。

1. 问题 5：还有没有更根本的原因？

- 结论：有，而且不止一个。
- 代码证据：
  - 当前训练 target 是 `AFDec`，不是 boundary manifold sample。
  - 当前 D 的任务是“condition-matched realism”，不是“segment / boundary adherence”。
  - 当前注入又只保留 feasible points，会天然偏向更容易的 interior feasible 点。
- 实验证据：
  - FE=100000 时 `normal_dec_seg_dist90=0.09236` 远小于 `normal_obj_seg_dist90=2.34244`，这说明“即使决策空间看起来离 AF-AI segment 不太远，objective 空间仍会大偏离”。也就是说，决策段与 objective 边界线之间存在明显非线性扭曲。
  - `GND_keep80` 的 feasible 高但 `SegDist90` 极差，`weightedCond` feasible 也高于 `localMAD_nGen20` 却更不贴线，这说明“提高可行率”与“贴边界线”存在结构性冲突。
- 机制解释：
  - 第一，**目标空间/决策空间一对多**。同一 objective 邻域可能对应多种 decision 解；如果 condition 只给 pair hint，不给 line coordinate，G 必然学成 cloud。
  - 第二，**“边界”定义错位**。真实想学的是可行/不可行接口；当前拿 AF endpoint 代替 boundary sample，本身就偏内侧。
  - 第三，**D 的任务过弱**。它只学“像不像 matched AF family”，不学“是不是 pair 上的那个位置”。
  - 第四，**z 没有语义约束**。既没有 latent regression，也没有 mutual information，z 的变化不会自动对应“沿边界走一点点”。
  - 第五，**decision-space proxy 不足以保证 objective-space line**。从 `dec_seg_dist90 << obj_seg_dist90` 已经看得很清楚。
  - 第六，**注入逻辑天然鼓励内侧可行点**。越往 interior 走，越容易过 `feasible` 门；所以只看 feasible rate 容易把训练导向“离边界更远的安全点”。
  - 这和未知约束优化文献的共识一致：在这类问题里，关键难点就是边界附近的几何与可行性联合建模，而不是单纯做 feasible classification。([Proceedings of Machine Learning Research](https://proceedings.mlr.press/v235/tian24g.html))
- 修改建议：
  - 不要走“decision 线性插值 + 再赌 objective 会贴线”这条捷径。
  - 直接用**已有评估过的真实 feasible near-segment samples**作为监督，condition 中显式加入 `t`，并用 `L_rec + L_adv` 压缩成窄流形。
  - 这样你绕开了“真实边界零层集不可直接标注”的难题，同时避免了 decision-space 假线段对 objective-space 的误导。
- 最小验证实验：
  - 做一个反证实验：把 target 设成 decision-space 的线性插值 `x(t)=AFDec+t(AIDec-AFDec)`。
  - 如果它能显著改善 `dec_seg_dist90`，却改善不了 `obj_seg_dist90` 或 feasible rate，说明“决策段代理边界线”这条思路本身就是错的。
  - 这个实验会直接验证我上面的结构性判断。

C. 最推荐的一条主线方案

**主线方案：Pair-`t` 边界流形条件生成器（去 `z`，用真实 near-segment feasible 样本监督，GAN 只做真实性正则）**

- 修改模块
  - `BuildBoundaryTargetTriples_BDG.m`：从“每个 pair 只输出一个 `AFDec`”改成“每个 pair 输出多条 `(x^*, AFNorm, Direction, t, RefToken)`”。
  - `BoundaryGAN_BDG.m`：边界模式下去掉随机 `z`，改成 `G(c,t)->x`；generator loss 改成 `L_adv + Huber(x,x^*)`。
  - `FilterBoundaryTargetTriples_BDG.m`：继续存在，但只做 near-segment real sample 的去噪。
  - `CCMO_GAN_BDG.m`：把当前主线 variant 改到这套 boundary mode 上，保留现有 diagnostics。
  - `test_CCMO_GAN_BDG_target_conditioned.m`：新增断言 `target_condition_dim` 包含 `t`，且 `target_triple_count > target_pair_count`。
- 修改后的训练数据定义
  - 对每个 archive pair (i)，在当前已评估的可行集合 (X_F) 中选取 objective 上靠近该 pair segment 的样本：
    [
    d_i(x)=\left|\bar f(x)-\Pi_{\text{seg}_i}\big(\bar f(x)\big)\right|
    ]
    保留 (d_i(x)) 小于阈值的真实 feasible 样本。
  - 给每个保留样本计算
    [
    t_i(x)=\mathrm{clip}\frac{(\bar f(x)-\bar f_{AF,i})^\top(\bar f_{AI,i}-\bar f_{AF,i})}{|\bar f_{AI,i}-\bar f_{AF,i}|^2},0,1
    ]
  - 训练样本变成 `(x^*, c_i=[AFNorm_i, Direction_i, t_i, RefToken_i])`。
  - 若某个 pair 暂无 near-segment feasible 样本，只保留回退样本 `(AFDec_i, t=0)`。
- 修改后的数学目标
  - 判别器仍做条件真假判别：`D(x,c,t)`。
  - 生成器：
    [
    \hat x = G(c,t),\qquad
    L_G = L_{adv} + \lambda,\mathrm{Huber}(\hat x, x^*)
    ]
    其中 `λ` 固定，不做大调参。
  - 采样时不再扫随机 `z`；沿一条 boundary line 的变化只通过 `t` 实现。
- 为什么它会让生成解更接近一条边界线
  - 现在的自由度从“弱 condition + 随机 z 的高维 cloud”变成了“pair + 一维标量 `t` 的 1D 流形”。
  - target 也从 AF 端点变成了真实已评估的 near-segment feasible 样本，监督目标本身就是细线附近的数据。
  - `Huber` 会把同一 `(pair,t)` 压到窄分布；GAN 只负责避免回归均值造成的失真。
  - 这条路线与 manifold-based 生成思路一致，但比当前做法多了最关键的“线坐标 `t`”和“真实 near-line 监督”。([arXiv](https://arxiv.org/abs/2101.02932))
- 最小实验验证
  - 先只在 `DASCMOP1_BC, DASCMOP2_BC, LIRCMOP7_BC, LIRCMOP8_BC` 上跑，`runs=1:3`，FE 取 `30000/70000/100000`。
  - 看 `normal/fixedz obj_seg_dist90`、`fixedz_own_pair_nearest_rate`、`zsweep_within_obj_width90_median`、`RawGANFeasibleRate`。
  - 成功判据：
    - `obj_seg_dist90` 相比当前主线下降至少 40%；
    - `fixedz_own_pair_nearest_rate` 至少提升到 0.30；
    - `zsweep_within_obj_width90_median` 压到 0.15 以下；
    - `RawGANFeasibleRate` 不低于当前主线 0.05 以上。
  - 如果只改善 feasible、不改善 width/own-pair-nearest，就算失败。

D. 不推荐继续做的方向

- 只调 `keep ratio`。`C4_keep60 / 80 / 90` 的 stage100000 `SegDist90` 分别是 `1.3675 / 2.5460 / 2.6401`，差异有，但都没有进入“细线”状态；这说明 keep ratio 只是调厚薄，不是改任务。
- 只把 `nGen` 从 20 加到 50。`localMAD_nGen20` 到 `localMAD_nGen50`，`SegDist90` 从 `1.7283` 变成 `1.7852`，几乎没有本质改善，只是 feasible 从 `0.278` 升到 `0.3088`。
- 只加 `covGate`。它把 `mean_has_GAN` 压到 `0.5333`，`BoundaryHit` 只有 `0.10125`，`Feasible` 还掉到 `0.2297`；它是在“少训”，不是“训对几何”。
- 只改 `weighted condition` 或 `localMAD` 权重。`weightedCond_nGen20` 的 `SegDist90=2.3191` 反而差于 `localMAD_nGen20=1.7283`；这些都是二阶数据分布调整。
- 只做后处理/筛选。当前代码已经只把 feasible GAN 样本注入 `Population1`。继续在注入后筛选，最多只能改进进入种群的点，不能让 raw GAN 本身贴成边界线。
- 只“减小 z / 固定 z”而不改 condition 与 target。诊断已经说明 fixed-z 仍然 `obj_seg_dist90≈2.195`、`own_pair_nearest≈0.097`。单改 z 不能把错任务改对。

E. 下一轮实验计划

1. **实验 1：主线方案小范围验证**
   - 改什么：实现 `pair+t` 数据集、`z=0`、`L_adv + Huber`。
   - 跑哪些问题/runs：`DASCMOP1_BC, DASCMOP2_BC, LIRCMOP7_BC, LIRCMOP8_BC`，`runs=1:3`，FE=`30000/70000/100000`。
   - 看哪些指标：`normal_obj_seg_dist90`、`fixedz_own_pair_nearest_rate`、`zsweep_within_obj_width90_median`、`RawGANFeasibleRate`。
   - 什么结果算成功：`obj_seg_dist90` 至少降 40%，`own_pair_nearest_rate ≥ 0.30`，`zsweep_within_obj_width90_median ≤ 0.15`，且 feasible 不掉超过 0.05。
2. **实验 2：只验证“目标重定义”是否是主因**
   - 改什么：固定同一个 generator/loss，只比较
     - A：旧 `AFDec` endpoint triples；
     - B：新 `pair+t` near-segment real triples。
   - 跑哪些问题/runs：同实验 1。
   - 看哪些指标：`fixedz_obj_seg_dist90`、`fixedz_own_pair_nearest_rate`、`zsweep_within_obj_width90_median`。
   - 什么结果算成功：B 全面优于 A；若 B 仅略优或无优势，则说明还需要更强的 boundary-aware loss。
3. **实验 3：全 10 问题主线回归**
   - 改什么：用实验 1 成功的主线方案替代当前 `GND_keep80_localMAD_nGen20`。
   - 跑哪些问题/runs：10 个问题，`runs=1:3`，FE=`30000/70000/100000`。
   - 看哪些指标：diagnostic by FE 的 `normal_obj_seg_dist90 / fixedz_own_pair_nearest_rate / zsweep_within_obj_width90_median / normal_feasible_rate`，以及 stage100000 的 `BoundaryHit_all / GAN_to_Segment_Dist90 / RawGANFeasibleRate`。
   - 什么结果算成功：
     - diagnostics：`fixedz_own_pair_nearest_rate` 至少是当前的 3 倍；`zsweep_within_obj_width90_median` 至少减半；
     - stage100000：`GAN_to_Segment_Dist90 < 1.0` 且 `RawGANFeasibleRate ≥ 0.30`；
     - 若只提升最终 feasible，而 diagnostics 仍是厚带，则判失败。