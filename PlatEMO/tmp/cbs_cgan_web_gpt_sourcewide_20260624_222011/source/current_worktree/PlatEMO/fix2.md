- ## 1. Verified Facts From Code And Results

  * 我核查了当前 bundle 的最新代码目录 `source/CBS-CGAN/`、修正后离线结果 `results/loss_ablation_lir_offline_evalfix_20260623_161641/`、在线结果 `results/loss_ablation_lir_online_20260623_162606/`、以及 `README.md`、`notes/bundle_summary.txt`、`notes/file_manifest.txt`。其中 `notes/bundle_summary.txt:3-6` 明确写明本 bundle 含 `434` 个文件、`180` 张 PNG、`186` 个 CSV。在线目录下实际 `.png` 文件数也为 `180`。`README.md:72-86`、`notes/bundle_summary.txt:3-6`

  * 最新 active code 已经删除了 pair-margin 和 mismatch-D。`README.md:8-15` 直接写明：

    * `pair-margin` 已从 active generator loss 删除；
    * `mismatch-D` 已从 active discriminator loss 删除；
    * 当前 active loss 是 `adversarial + Huber reconstruction`。
      代码也一致：`source/CBS-CGAN/BoundaryCGAN_CBS.m:216-239` 中，判别器只用 `real/fake` 两路，生成器只加 `advLoss` 和 `huberLoss`，没有 pair-margin 项。

  * 当前 active `BoundaryCGAN_CBS.m` 的默认控制项仍包含 `advWeight`、`reconstructionWeight`、`trainZMode`，且 `trainZMode` 只允许 `"random"` 或 `"zero"`。`source/CBS-CGAN/BoundaryCGAN_CBS.m:256-279`

  * `test_CBS_CGAN_loss_options.m` 明确验证了“被删除的 pair-margin / mismatch-D 选项即使传入也会被忽略”。`source/CBS-CGAN/test_CBS_CGAN_loss_options.m:8-10,55-100`

  * 当前主算法 `CBS_CGAN.m` 的默认 `conditionMode` 其实是 `ref_tau`，不是 `ref_y`。`source/CBS-CGAN/CBS_CGAN.m:56-63`。但本次 loss ablation 的 offline / online runner 都显式改成了 `ref_y`：

    * offline: `source/CBS-CGAN/Support/run_CBS_CGAN_loss_ablation_lir_offline.m:71`
    * online: `source/CBS-CGAN/Support/run_CBS_CGAN_loss_ablation_lir_online.m:95`

  * 当前 `ref_y` 条件的真实代码定义是：

    * `TrainC = [reference direction, normalized BMem.y_b]`
    * `QueryC = [reference direction, normalized QueryY]`
      `README.md:19-31` 与 `source/CBS-CGAN/BuildBoundaryDataset_CBS.m:76-82` 一致。

  * `BuildBoundaryDataset_CBS.m` 当前支持三种 condition：

    * `ref_tau`
    * `ref_y_tau`
    * `ref_y`
      其具体拼接逻辑见 `source/CBS-CGAN/BuildBoundaryDataset_CBS.m:76-82`。

  * 当前 `QueryY` 不是生成点，也不是新评估出的真实解；它是由已有边界记忆 `BMem.y_b` 在相邻边界段上做插值 / gap filling 得到的。`README.md:35-43`；`source/CBS-CGAN/BuildBoundaryDataset_CBS.m:194-208,244-267`

  * 当前 `BMem` 已经不是旧的“单侧可行链”逻辑，而是 **pair-supported thin boundary memory**。`source/CBS-CGAN/UpdateBoundaryMemory_CBS.m:1-3`。具体地：

    * 为每个 ref 形成 `x_f/y_f` 与 `x_i/y_i` 的 pair；
    * 用 `selectThinBoundaryTarget(...)` 在当前 feasible 中选离该 `y_f-y_i` 目标段最近的 `x_b/y_b`；
    * 同时保存 `tau`。
      见 `source/CBS-CGAN/UpdateBoundaryMemory_CBS.m:135-183`

  * `UpdateBoundaryMemory_CBS.m` 末尾还会按 ref 做 canonical 去重和保留。`source/CBS-CGAN/UpdateBoundaryMemory_CBS.m:343-357`

  * 在线 runner 虽然把诊断图第一格标题写成了 `QueryC random z`，但本次 online run 实际上把 **训练和正式采样都固定成了 zero latent**：

    * `trainZMode="zero"`：`source/CBS-CGAN/Support/run_CBS_CGAN_loss_ablation_lir_online.m:95-98`
    * `sampleZMode="zero"`：同上
    * 实际生成前，`CBS_CGAN.m` 会调用 `applyCBSSampleControlOptions(...)`；当 `sampleZMode=="zero"` 时，`Options.sampleZ` 被强制设为全 0。`source/CBS-CGAN/CBS_CGAN.m:180-184,354-365`
    * 诊断图第一格却仍把 `S.generated_objs` 标成 `QueryC random z`。`source/CBS-CGAN/Support/run_CBS_CGAN_boundary_quality_experiments.m:643-645`
      这是本 bundle 里一个已验证的图示命名问题。

  * `visual_z_diagnostic` 的 6 个 panel 顺序固定为：

    1. `QueryC random z`
    2. `QueryC z=0`
    3. `QueryC fixed z`
    4. `TrainC z=0`
    5. `TrainC fixed z`
    6. `Targets only`
       见 `source/CBS-CGAN/Support/run_CBS_CGAN_boundary_quality_experiments.m:643-651`

  * 图例颜色是代码里写死的：

    * 橙色方块：Training set
    * 蓝色菱形：Query target
    * 红色圆点：CGAN generated
    * 绿色区域：feasible domain
    * 米色区域：infeasible domain
      见 `source/CBS-CGAN/Support/run_CBS_CGAN_boundary_quality_experiments.m:710-764`

  * 修正后离线结果之所以要单独放到 `loss_ablation_lir_offline_evalfix_20260623_161641/`，是因为 `README.md:60-61` 明确写了：直接 `Problem.CalObj` 对 `LIRCMOP_BC` 类并不有效；修正目录使用了 `EvaluateDecisions_CBS`。`source/CBS-CGAN/EvaluateDecisions_CBS.m:10-21` 也确实在发现 `CalObj/CalCon` 维度不一致时回退到 `Problem.Evaluation(...)`。

  * 离线 ablation 的关键 summary 是：

    * Huber 相比 mismatch-only 更好：`offline_loss_ablation_summary.csv:2-7`
    * pair-margin 相比 mismatch-only 更差：`offline_loss_ablation_summary.csv:10-17`
    * Full(V4) 相比 Adv+Huber(V2) 更差：`offline_loss_ablation_summary.csv:18-25`
    * mismatch-D 相比 Pure_CGAN(V0) 更差：`offline_loss_ablation_summary.csv:42-49`
    * `pair_violation_rate` 在这些离线 summary 中始终是 `0.975`，没有被拉开。`offline_loss_ablation_summary.csv:6,14,22,30,38,46`

  * 离线按 variant 汇总里，`V0_Pure_CGAN` 的中位数整体最好：

    * `median_train_y_rec90 = 0.302204257184084`
    * `median_query_y_err90 = 0.29910106444686`
    * `median_boundary_dist90 = 0.294353297903252`
    * `median_feasible_rate = 1`
      见 `offline_loss_ablation_by_variant.csv:2`
      相比之下：
    * `V4_Full_current` 的 `median_boundary_dist90 = 0.325968437766546`。`offline_loss_ablation_by_variant.csv:6`
    * `V5_Huber_Pair_no_adv` 的 `median_boundary_dist90 = 0.358944673600781`。`offline_loss_ablation_by_variant.csv:7`

  * 在线按 problem 汇总中，`V5_Huber_Pair_no_adv` 在 `LIRCMOP7_BC` 和 `LIRCMOP8_BC` 上虽然 `feasible_rate` 很高，但 `boundary_dist90 / query_obj_dist90 / segment_width90` 极差：

    * `LIRCMOP7_BC`: `feasible_rate=1`, `boundary_dist90=0.36227307264036`, `query_obj_dist90=0.400614128018036`, `segment_width90=0.400504928221534`。`online_loss_ablation_by_problem.csv:17`
    * `LIRCMOP8_BC`: `feasible_rate=0.75`, `boundary_dist90=0.29102984049351`, `query_obj_dist90=0.293350379982368`, `segment_width90=0.293142077731847`。`online_loss_ablation_by_problem.csv:18`

  * 在线按 problem 汇总中，也存在“指标还不错但 TrainC 面板已经明显失真”的情形。例如 `V0_Pure_CGAN` 在 `LIRCMOP7_BC` 上：

    * `boundary_dist90=0.00494213072563722`
    * `query_obj_dist90=0.0216398413021422`
      见 `online_loss_ablation_by_problem.csv:5`
      但对应图 `results/loss_ablation_lir_online_20260623_162606/V0_Pure_CGAN/LIRCMOP7_BC_run1/figures/LIRCMOP7_BC_run1_targetFE100000_visual_z_diagnostic.png` 的 `TrainC z=0 / TrainC fixed z` 面板已经出现明显离开橙色训练边界的红色团状分叉。

  * 在线按 problem 汇总中，比较干净的例子是：

    * `V0_Pure_CGAN` on `LIRCMOP5_BC`: `feasible_rate=0.7`, `boundary_dist90=0.00341163542851841`。`online_loss_ablation_by_problem.csv:3`
    * `V4_Full_current` on `LIRCMOP10_BC`: `feasible_rate=0.3`, 但 `boundary_dist90=8.55029613108595e-05`, `query_obj_dist90=0.000726319422301347`。`online_loss_ablation_by_problem.csv:8`

  * 我逐一核查了 online run1 的全部标准图和 `visual_z_diagnostic` 图（6 个问题 × 3 个 variant × 5 个 FE × 2 种图 = 180 张 PNG）。其中几个最关键的视觉事实是：

    * `V5_Huber_Pair_no_adv/LIRCMOP7_BC_run1/...targetFE100000_visual_z_diagnostic.png`：`TrainC z=0` 和 `TrainC fixed z` 直接生成出远离橙色训练边界的大环形/蛇形红点，不是薄边界重构。
    * `V5_Huber_Pair_no_adv/LIRCMOP8_BC_run1/...targetFE100000_visual_z_diagnostic.png`：TrainC 和 QueryC 面板都明显偏离橙色边界，形成大弧线。
    * `V4_Full_current/LIRCMOP8_BC_run1/...targetFE100000_visual_z_diagnostic.png`：`TrainC z=0 / fixed z` 也已经不是橙色边界重现，而是一条偏移的弯曲红支路。
    * `V0_Pure_CGAN/LIRCMOP5_BC_run1/...targetFE100000_visual_z_diagnostic.png`：QueryC 和 TrainC 面板都较贴近橙色边界，是本 bundle 中相对干净的例子之一。
    * `V4_Full_current/LIRCMOP10_BC_run1/...targetFE100000_visual_z_diagnostic.png`：QueryC 和 TrainC 面板都较贴近橙色边界，是当前较好的例子。
    * `V5_Huber_Pair_no_adv/LIRCMOP7_BC_run1/...targetFE100000.png`：标准主图里几乎看不到红点，容易让人误判为“没生成”或“生成很少”；但对应 diagnostic 图显示红点其实大面积跑到了当前绘图窗口之外。

  ---

  ## 2. Current Failure Diagnosis

  ### 2.1 已验证的失败链条

  1. **当前 active loss 的删改方向是有证据的，但这还没有解决主问题。**
     离线 ablation 已经支持删除 `mismatch-D` 和 `pair-margin`：

     * `MismatchD_effect`: `V1` 相比 `V0` 在 `train_y_rec90 / query_y_err90 / boundary_dist90` 上都变差。`offline_loss_ablation_summary.csv:42-45`
     * `Pair_vs_mismatch`: `V3` 相比 `V1` 在 `train_y_rec90 / query_y_err90 / boundary_dist90` 上也变差，而 `pair_violation_rate` 没变。`offline_loss_ablation_summary.csv:11-14`
       这解释了为什么 active code 删除了这两项。
       **但** online 图表明，即使现在 active code 只剩 `adversarial + Huber reconstruction`，`ref_y` 主线仍然没有稳定学会“当前薄边界生成”。

  2. **当前 online 失败不能再归因于 random z。**
     本次 online run 实际用了 `trainZMode="zero"` 和 `sampleZMode="zero"`。`run_CBS_CGAN_loss_ablation_lir_online.m:95-98`；`CBS_CGAN.m:354-365`。
     所以 `LIRCMOP7_BC` / `LIRCMOP8_BC` 的失败，是在**零 latent、确定性采样**下发生的，不是随机采样把边界吹散导致的首要问题。

  3. **有些问题已经不是 Query 失败，而是 TrainC 自身重构失败。**
     最强证据是：

     * `V5_Huber_Pair_no_adv/LIRCMOP7_BC_run1/...targetFE100000_visual_z_diagnostic.png`
     * `V5_Huber_Pair_no_adv/LIRCMOP8_BC_run1/...targetFE100000_visual_z_diagnostic.png`
     * `V4_Full_current/LIRCMOP8_BC_run1/...targetFE100000_visual_z_diagnostic.png`
       这些图中，`TrainC z=0 / TrainC fixed z` 的红点已经明显偏离橙色训练边界。
       这意味着：**模型连训练条件下的当前边界都没有稳定复现**。这比“Query 外推失败”更基础。

  4. **当前在线 CSV 里存在“数值看着还行，但模型其实没学住边界”的情况。**
     典型例子是 `V0_Pure_CGAN` 的 `LIRCMOP7_BC`：

     * `boundary_dist90=0.004942...`，看起来不差。`online_loss_ablation_by_problem.csv:5`
     * 但对应 `.../LIRCMOP7_BC_run1_targetFE100000_visual_z_diagnostic.png` 的 `TrainC z=0 / TrainC fixed z` 已经出现明显非边界团状结构。
       这说明：**只看 online summary 的 boundary/query 指标，会漏掉 TrainC 重构失败。**

  5. **`feasible_rate` 不能作为主指标。**
     `V5_Huber_Pair_no_adv` 在：

     * `LIRCMOP7_BC` 上 `feasible_rate=1`，但 `boundary_dist90=0.362273...`、`segment_width90=0.400505...`。`online_loss_ablation_by_problem.csv:17`
     * `LIRCMOP8_BC` 上 `feasible_rate=0.75`，但 `boundary_dist90=0.291030...`、`segment_width90=0.293142...`。`online_loss_ablation_by_problem.csv:18`
       也就是：**可以大量生成“可行点”，但它们并没有形成当前可行/不可行薄边界。**

  6. **当前 `pair_violation_rate` 不是有效主指标。**
     在离线汇总里，V0~V5 的 `pair_violation_rate` 全都是 `0.975`。`offline_loss_ablation_by_variant.csv:2-7`
     所以它在这批实验里并没有真正区分不同 loss 设计。

  ### 2.2 对失败原因的技术判断

  * **[已验证]** 当前实验的主矛盾不是“是否要恢复 pair-margin / mismatch-D”。离线 ablation 已经给出反证，这两项不该回主线。`README.md:88-97`，`offline_loss_ablation_summary.csv:11-17,42-49`

  * **[已验证]** 当前实验的更大问题是：`conditionMode = ref_y` 把目标空间位置 `y` 直接塞进了条件里。`README.md:19-31`，`BuildBoundaryDataset_CBS.m:76-82`

  * **[推断]** 一旦条件直接包含 `normalized y`，CGAN 更容易学成“给定目标空间位置，反推一个决策变量 x”，而不是“从部分当前边界解里学习当前边界分布，再生成这条边界的别处”。这正是 README 自己也点出的 `objective-conditioned inverse mapping` 风险。`README.md:29-31`

  * **[已验证]** 当前在线主图容易误导：主图没有蓝色 Query target，而 diagnostic 图第一格又误标成 `QueryC random z`。所以如果只看标准图，像 `V5_Huber_Pair_no_adv/LIRCMOP7_BC_run1_targetFE100000.png` 这种严重跑飞的情况会被“看起来几乎没红点”掩盖。

  * **[推断]** 因为当前零 latent 下仍大量失败，下一轮最应该改的是**任务定义和 condition**，不是先去纠结随机 z、pair-count 或把已删除 loss 加回来。

  ---

  ## 3. Why The Current `ref_y` Setup Conflicts With The Intended Goal

  当前 `ref_y` 代码做的是：

  * `TrainC = [reference direction, normalized BMem.y_b]`
  * `QueryC = [reference direction, normalized QueryY]`
  * `G(C, z) -> x`
    `README.md:21-31`；`BuildBoundaryDataset_CBS.m:76-82`

  这和你的目标并不等价。

  ### 当前 `ref_y` 实际在学什么

  它更接近：

  > 给定一个目标空间位置提示 `y`，再结合一个方向 `ref`，生成一个决策变量 `x`，使 `x` 评估后的目标值靠近这个 `y`。

  因为 `QueryY` 本身就是从 `BMem.y_b` 插值出来的蓝色目标点，不是真实待发现决策变量。`README.md:35-43`；`BuildBoundaryDataset_CBS.m:194-208`

  ### 你的真实目标是什么

  你的目标应该表述为：

  > 部分当前边界解
  > → 学习“当前这条可行/不可行边界”的分布
  > → 生成这条当前边界上尚未探索到的完整决策变量 `x`

  也就是说，**主语应该是“当前边界分布”**，不是“目标空间点 `y`”。

  ### 两者的差别

  * `ref_y`：把“目标空间点”放进条件，模型很容易退化成 `objective point -> decision variable` 的条件逆映射。
  * 你要的边界生成：条件应该只描述“当前边界上的局部位置/局部身份”，不能把绝对目标点当成生成指令。

  因此，当前 `ref_y` 与研究目标的冲突不是修辞上的，而是**代码语义层面的冲突**。

  ---

  ## 4. Corrected Problem Formulation

  建议把问题严格改写为：

  > 输入：当前搜索已经发现的、由真实评估得到的部分边界记忆 `BMem`。
  > 其中每条记忆都对应当前目标空间可行/不可行边界上的一个局部 pair-supported 边界样本。
  >
  > 学习任务：用 CGAN 学习“当前这条边界”的局部分布规律。
  >
  > 生成任务：对当前边界中尚未充分覆盖的局部位置采样 condition，直接生成完整决策变量 `x`。
  >
  > 验证任务：只用真实 `Problem.Evaluation` / `EvaluateDecisions_CBS` 判断该 `x` 是否真的落在当前目标空间可行/不可行边界附近，并决定是否更新 `BMem`。

  这一定义里：

  * CGAN 仍然直接生成完整决策变量 `x`；
  * “边界”仍然只指目标空间可行/不可行边界；
  * 不需要 surrogate；
  * 也不把问题改写成“给定目标点 `y` 反推 `x`”。

  ---

  ## 5. Recommended Condition Design

  ### 结论

  **不建议继续使用 `ref_y`。
  当前最小、统一、与现有代码最兼容的 condition 是 `ref_tau`。**

  ### 为什么不是 `ref_y`

  因为 `ref_y` 把绝对目标位置塞进条件，直接把任务推向 `y -> x` 的条件逆映射。`README.md:19-31`；`BuildBoundaryDataset_CBS.m:76-82`

  ### 仅用 reference direction 是否足够？

  **我不建议只用 `ref`。**

  * `[已验证]` 当前数据结构里不仅保留了 `ref`，还保留了 `tau`、`source_interval`、`source_type`、`x_f/x_i/y_f/y_i`。`BuildBoundaryDataset_CBS.m:42-49,244-267`
  * `[推断]` 这说明当前代码本身已经默认“仅有方向还不够，边界上的局部位置还需要被表达”。

  ### `ref_tau` 是否可行？

  **可行，而且是当前证据下最合适的最小条件。**

  理由：

  1. `tau` 不是绝对目标点。
     它来自当前 `y_b` 在 `y_f-y_i` 边界段上的局部位相。`UpdateBoundaryMemory_CBS.m:162-183`

  2. `ref_tau` 已经被当前代码支持，不需要发明新框架。
     `BuildBoundaryDataset_CBS.m:76-79`

  3. 它表达的是“当前边界上的局部位置”，而不是“目标空间绝对坐标”。

  ### 是否存在比 `ref_tau` 更好的统一 condition？

  **[推断] 当前 bundle 证据还不足以证明必须再加更复杂的 condition。**

  如果 `ref_tau` 在下一轮实验里仍然不能让 `TrainC z=0 / fixed z` 重现训练边界，那么再考虑加一个**离散的局部 segment identity**（例如由 `source_interval` 派生），而不是回到 `y`。
  但在当前证据下，我不建议一步跳到复杂 condition。

  ### 可行/不可行侧、局部邻域几何要不要进 condition？

  **不建议先放进 condition。**

  当前更合理的位置是：

  * `ref` + `tau` 负责 condition；
  * `x_f/x_i/y_f/y_i`、`source_interval`、`source_type` 继续保存在 metadata 里，用于：

    * 真实评估后的接受/拒绝判断；
    * 边界贴近度指标；
    * `BMem` 更新。

  这样更统一，也更不容易又堆回多条件系统。

  ---

  ## 6. Recommended Training Data Construction

  ### 建议主线

  **保留当前 `pair-supported thin boundary memory`，但把训练/查询条件从 `ref_y` 改成 `ref_tau`。**

  ### 训练集

  * `TrainX = BMem.x_b`
  * `TrainC = [BMem.ref, BMem.tau]`
  * `x_f/x_i/y_f/y_i` 继续作为训练 metadata 保留，但不进主 condition。
    依据：`UpdateBoundaryMemory_CBS.m:135-183` 已经把这些量准备好了；`BuildBoundaryDataset_CBS.m:42-49` 已经会打包。

  ### 查询条件

  不要再把 `QueryY` 当主条件的一部分。

  建议改成：

  * 对 `missing_ref` 和 `large_gap`，仍沿当前边界相邻 pair-supported 段采样；
  * 但 `QueryC` 直接构造成 `[ref_q, tau_q]`；
  * `QueryY` 只保留为**诊断用元信息**，不再作为正式条件，也不再作为主成功标准。

  当前 `buildExternalQueries` 已经有 `ref`、`tau`、`source_interval`、`source_type`；这使这种改法可以直接落在现有代码结构上。`BuildBoundaryDataset_CBS.m:194-208,244-267`

  ### 为什么这比改 BMem 更优先

  因为当前 `BMem` 已经是 thin-boundary、pair-supported 的。`UpdateBoundaryMemory_CBS.m:1-3,135-183,343-357`
  现有证据先指向 **condition 语义不对**，而不是 `BMem` 结构先坏了。

  ---

  ## 7. Recommended Loss And Training Flow

  ### Loss 结论

  **保留 `adversarial + Huber reconstruction`；不要把 `pair-margin` 和 `mismatch-D` 加回来。**

  依据：

  * active code 已这么做。`BoundaryCGAN_CBS.m:216-239`
  * 离线 ablation 支持删除 `pair-margin`、`mismatch-D`。`offline_loss_ablation_summary.csv:11-17,42-49`
  * online `V5_Huber_Pair_no_adv` 在 `LIRCMOP7/8` 上视觉失败最严重。`online_loss_ablation_by_problem.csv:17-18` + 对应 diagnostic 图

  ### 是否需要 pair-margin 替代项？

  **当前证据下，不建议在训练 loss 里引入新的替代项。**

  原因很简单：

  * 旧 pair-margin 在当前 evaluator 下没有把 `pair_violation_rate` 拉开。`offline_loss_ablation_by_variant.csv:2-7`
  * 它还会恶化目标/边界误差。`offline_loss_ablation_summary.csv:11-13`

  更合理的做法是：

  * **训练时**只保留 `adversarial + Huber`；
  * **真实评估后**再用 `x_f/x_i/y_f/y_i` 去做边界贴近与侧向判定，决定是否纳入 `BMem`。

  这不是把方法变成“纯后处理筛选”，因为主生成器仍然是 CGAN；这里只是把 pair 信息从无效的训练损失中移出，转成更可信的真实评估更新规则。

  ### 训练流程建议

  1. `trainZMode` 正式主线固定为 `zero`。
  2. 正式生成 `sampleZMode` 也固定为 `zero`。
  3. 先要求 `TrainC z=0 / fixed z` 重现橙色训练边界；达不到就不要继续宣传 Query 结果。
  4. 训练停止或保留模型，不看 GAN loss 曲线本身，而看：

     * `train_y_rec90`
     * `boundary_dist90`
     * `segment_width90`
     * `TrainC` 诊断图

  ### 一个关键事实

  当前 online 结果已经是 `zero-z` 条件下得到的。`run_CBS_CGAN_loss_ablation_lir_online.m:95-98`
  所以“把 z 固定成 0”不是下一轮的核心创新点，而只是主线的默认约束。
  真正还没改对的是 **condition 与任务表述**。

  ---

  ## 8. Recommended Metrics And Figures

  ### 推荐保留的主指标

  1. **generated 到当前边界的距离**
     继续使用 `boundary_dist50/90`。
     这是当前最贴近真实目标的已有指标之一。`CBS_CGAN.m:194-232`

  2. **boundary thickness / thinness**
     继续使用 `segment_width90`，必要时同时报告 `segment_width90_ratio`。
     因为你的目标是薄边界，不是厚点云。

  3. **coverage of under-explored boundary**
     继续使用 `ref_cover`，但应明确它衡量的是“真实评估并保留的生成点”对当前 ref 覆盖的扩展。

  4. **Train-condition reconstruction**
     把 `train_x_rec90`、`train_y_rec90` 提升为主指标，而不是只放离线分析里。
     因为当前 bundle 已经证明：有些模型 online summary 看着不差，但 `TrainC` 面板已经失败。

  5. **side / interface diagnostics**
     `side_rate` 必须记录，但只能作为辅指标。
     它不能替代边界距离和边界厚度。

  ### 不应作为主成功标准的指标

  1. **`query_obj_dist90` 不应单独成为主指标。**
     因为它是“红点到蓝色 QueryY 的距离”，而蓝色 `QueryY` 本身只是从 `BMem.y_b` 插值出来的诊断点，不是真实主目标。`README.md:35-43`

  2. **`feasible_rate` 不能单独判优。**
     `V5` 在 `LIRCMOP7/8` 上已经给出反例。`online_loss_ablation_by_problem.csv:17-18`

  3. **当前 `pair_violation_rate` 不适合作为主指标。**
     本 bundle 里它几乎常数化了。`offline_loss_ablation_by_variant.csv:2-7`

  ### 推荐图像

  1. **主图**
     只画：

     * feasible/infeasible 背景
     * 橙色当前训练边界
     * 红色真实评估后的生成点
       不要把蓝色 QueryY 画成主图焦点。

  2. **TrainC reconstruction figure**
     单独把 `TrainC z=0` 与橙色训练边界对比画出来。
     当前 bundle 里很多失败是一眼能在这张图看出来的，而 summary CSV 看不出来。

  3. **Query diagnostic figure**
     可以保留，但第一格必须按真实采样模式命名，不能在 `sampleZMode="zero"` 时还写 `QueryC random z`。
     这一点当前代码必须修。`run_CBS_CGAN_boundary_quality_experiments.m:643-645` + `run_CBS_CGAN_loss_ablation_lir_online.m:95-97`

  4. **condition-space coverage figure**
     如果主线改成 `ref_tau`，建议再加一张 `ref × tau` 覆盖图：

     * 训练点
     * query 点
     * 真实保留生成点
       这比画蓝色 QueryY 更贴近“当前边界位置生成”这个真实任务。

  ---

  ## 9. Minimal Next Experiment

  ### 目标

  不是做大而全的 full-factorial。
  只验证三件事：

  1. `ref_only` 条件下，加入 Huber reconstruction 是否改善训练边界重构；
  2. 在不引回 pair-margin / mismatch-D 的前提下，`ref_tau` 是否比 `ref_only` 更适合当前边界位置生成；
  3. A/B/C 三个设置是否能用同一套 zero-z online runner 产出可复查的 CSV 与图。

  ### 问题集

  建议只跑 4 个问题：

  * `LIRCMOP5_BC`：当前相对较好
  * `LIRCMOP7_BC`：当前明显失败
  * `LIRCMOP8_BC`：当前明显失败
  * `LIRCMOP10_BC`：当前相对较好

  ### runs 与线程

  * `runs = 3`
  * `workerCount = 8`
    保留并行执行，但仍要求图和 CSV 完整可复查。

  ### 比较设置

  只比 3 个设置；这是本轮确认后的 A/B/C 合约：

  1. **A_ref_only_adv**: `conditionMode = ref_only`, `advWeight = 1`, `reconstructionWeight = 0`, `trainZMode = zero`, `sampleZMode = zero`
  2. **B_ref_only_adv_huber**: `conditionMode = ref_only`, `advWeight = 1`, `reconstructionWeight = 1`, `trainZMode = zero`, `sampleZMode = zero`
  3. **C_ref_tau_adv**: `conditionMode = ref_tau`, `advWeight = 1`, `reconstructionWeight = 0`, `trainZMode = zero`, `sampleZMode = zero`

  不要再把 `pairMarginWeight` 和 `useMismatchD` 放进这个最小实验。

  ### 必须记录的指标

  * `train_x_rec90`
  * `train_y_rec90`
  * `boundary_dist90`
  * `segment_width90`
  * `feasible_rate`
  * `side_rate`
  * `ref_cover`

  ### 必须输出的图

  保留 5 个 FE target：`10000 / 30000 / 50000 / 70000 / 100000`。
  启用绘图的 run 对每个 target 输出以下图：

  1. 主图：orange boundary + red evaluated generated + background
  2. TrainC reconstruction 图
  3. Query diagnostic 图（正确标注实际 z 模式）
  4. condition-space coverage 图（训练点、query 点、真实评估后的可行生成点在 `ref × tau` 空间中的覆盖）

  ### Pass / Fail 标准

  * **Pass-1**：`LIRCMOP7_BC` 与 `LIRCMOP8_BC` 的 `TrainC z=0 / fixed z` 不再出现当前 bundle 里的团状/环状/蛇形远离橙色边界结构。
    若仍出现，则说明 condition 仍然不对。

  * **Pass-2**：`C_ref_tau_adv` 的 `boundary_dist90` 与 `segment_width90` 在 `LIRCMOP7_BC`、`LIRCMOP8_BC` 上明显优于 `A_ref_only_adv`；同时 `LIRCMOP5_BC`、`LIRCMOP10_BC` 不应明显退化。

  * **Pass-3**：`ref_cover` 至少不低于 `A_ref_only_adv`，不能靠把红点吹成厚云来“提高覆盖”。

  * **Fail-1**：如果 `C_ref_tau_adv` 的 TrainC 图仍明显不能重现训练边界，那么仅靠换 condition mode 还不够，下一步才考虑再加一个离散 segment identity。

  * **Fail-2**：如果 `B_ref_only_adv_huber` 相比 `A_ref_only_adv` 没有改善 TrainC 重构，说明 Huber reconstruction 在 `ref_only` 条件下不足以解决当前边界学习问题。

  ---

  ## 10. Concrete Code Change List

  ### 10.1 `source/CBS-CGAN/BuildBoundaryDataset_CBS.m`

  * **改什么**
    把主线 condition 从 `ref_y` 改为 `ref_tau`；`TrainC` 和 `QueryC` 不再拼接 `normalized Y`。
    `QueryY` 继续保留，但只做诊断元数据，不再作为正式条件。

  * **为什么来自当前证据**
    `ref_y` 直接把目标空间点放进 condition。`README.md:19-31`；`BuildBoundaryDataset_CBS.m:76-82`。
    当前在线 zero-z 仍失败，说明不是 z 的锅。`run_CBS_CGAN_loss_ablation_lir_online.m:95-97`

  * **如何测试**
    跑第 9 节的 `A_ref_only_adv` vs `C_ref_tau_adv`。

  * **什么结果会证伪它**
    若 `ref_tau` 不能修复 `LIRCMOP7/8` 的 `TrainC` 重构失败，且还显著恶化 `LIRCMOP5/10`，则单纯 `ref_tau` 不够。

  ### 10.2 `source/CBS-CGAN/BoundaryCGAN_CBS.m`

  * **改什么**
    保持 active loss 为 `adversarial + Huber reconstruction`；不要恢复 pair-margin 和 mismatch-D。
    把 `trainZMode` 默认改成 `zero`，避免无意回到随机训练。

  * **为什么来自当前证据**
    删除 pair-margin / mismatch-D 已有离线证据。`offline_loss_ablation_summary.csv:11-17,42-49`
    当前 online runner 也已经用 `zero` latent。`run_CBS_CGAN_loss_ablation_lir_online.m:95-97`

  * **如何测试**
    跑 `A_ref_only_adv` vs `B_ref_only_adv_huber`，判断 Huber reconstruction 在 `ref_only` 条件下是否有效；跑 `A_ref_only_adv` vs `C_ref_tau_adv`，判断 `ref_tau` 条件是否更贴合边界位置生成。

  * **什么结果会证伪它**
    如果 `B_ref_only_adv_huber` 相比 `A_ref_only_adv` 没有改善 `TrainC` 图和 `train_x_rec90/train_y_rec90`，说明 Huber reconstruction 不是解决 `ref_only` 条件表达不足的关键。

  ### 10.3 `source/CBS-CGAN/CBS_CGAN.m`

  * **改什么**
    让正式主流程默认使用 `sampleZMode="zero"`；
    把 `TrainC` 重构误差和图纳入主实验记录，而不是只放离线 evaluator；
    用真实 `Problem.Evaluation` 后的边界距离/厚度/覆盖率决定生成点是否更新 `BMem`。

  * **为什么来自当前证据**
    当前 online summary 会漏掉 `TrainC` 重构失败。`V0_Pure_CGAN/LIRCMOP7_BC` 就是反例。
    而且当前 online 实际已经是 zero-z。`run_CBS_CGAN_loss_ablation_lir_online.m:95-97`

  * **如何测试**
    看 `C_ref_tau_adv` 是否能在 `LIRCMOP7/8` 上让 `TrainC` 图恢复正常。

  * **什么结果会证伪它**
    如果加入 TrainC gate 后，所有“看起来好”的 run 都被否决，说明当前 condition / loss 仍没有学对，不能靠 gate 掩盖。

  ### 10.4 `source/CBS-CGAN/Support/run_CBS_CGAN_boundary_quality_experiments.m`

  * **改什么**
    修正 diagnostic panel 命名：第一格按真实采样模式命名，不要固定写 `QueryC random z`。
    增加单独的 `TrainC reconstruction` 输出图。
    增加 `condition-space coverage` 输出图，画训练点、query 点和真实评估后的可行生成点在 `ref × tau` 空间中的覆盖。
    主图不再把蓝色 QueryY 作为主角。

  * **为什么来自当前证据**
    当前 online 图存在命名误导。`run_CBS_CGAN_boundary_quality_experiments.m:643-645` + `run_CBS_CGAN_loss_ablation_lir_online.m:95-97`

  * **如何测试**
    新旧图对照，检查失败是否更容易一眼识别。

  * **什么结果会证伪它**
    如果改图后仍然无法从图上迅速分辨 TrainC 失败与 Query 失败，说明图设计还不够贴合任务。

  ### 10.5 `source/CBS-CGAN/Support/run_CBS_CGAN_loss_ablation_lir_online.m`

  * **改什么**
    下一轮只保留 3 个最小设置：`A_ref_only_adv`、`B_ref_only_adv_huber`、`C_ref_tau_adv`。
    不再继续在线测试 `pairMarginWeight/useMismatchD` 分支。

  * **为什么来自当前证据**
    这两项已被当前 bundle 的离线证据否掉。`offline_loss_ablation_summary.csv:11-17,42-49`

  * **如何测试**
    直接跑第 9 节最小实验。

  * **什么结果会证伪它**
    如果 `C_ref_tau_adv` 全面不如 `A_ref_only_adv`，才值得重新扩 condition 实验；在此之前不应再回到旧 loss 分支。

  ### 10.6 `source/CBS-CGAN/UpdateBoundaryMemory_CBS.m`

  * **改什么**
    **当前阶段不建议先改。**

  * **为什么来自当前证据**
    当前 `BMem` 已明确是 `pair-supported thin boundary memory`。`UpdateBoundaryMemory_CBS.m:1-3,135-183,343-357`
    现有失败先更像是 condition/任务定义不对，而不是 `BMem` 先错。

  * **如何测试**
    先完成 `ref_tau` 最小实验。

  * **什么结果会证伪它**
    如果 `ref_tau` + 现有 loss 仍然无法让 `TrainC` 图在 `LIRCMOP7/8` 上贴边，那时再回头查 `BMem` 是否仍混入了不适合的训练样本。

  补充一点实现层面的可行性：上述改动都可以直接在现有 MATLAB 自定义训练循环内完成，不需要引入第二学习器或改用别的训练框架；MathWorks 官方文档明确支持用 `dlnetwork`、`dlfeval/dlgradient`、`adamupdate` 实现这类自定义训练流程。 ([MathWorks][1])

  ---

  ## 11. Risks And Falsification Criteria

  1. **风险：`ref_tau` 仍然不够表达边界局部身份。**

     * **识别方式**：`TrainC z=0 / fixed z` 在 `LIRCMOP7/8` 仍无法重现橙色边界。
     * **结论**：那时再加一个离散 `segment identity`，而不是回到 `y`。

  2. **风险：去掉 `y` 后，coverage 下降。**

     * **识别方式**：`ref_cover` 明显低于 `A_ref_only_adv`，且没有换来更薄的边界。
     * **结论**：说明 `ref_tau` 的表达力不足，需要补局部 segment 身份，但仍不应回到绝对 QueryY。

  3. **风险：Huber 在 `ref_only` 条件下并不能改善重构。**

     * **识别方式**：`B_ref_only_adv_huber` 相比 `A_ref_only_adv` 没有改善 `train_x_rec90/train_y_rec90` 与 TrainC 图。
     * **结论**：说明 Huber 不是单独解决 `ref_only` 条件表达不足的关键，下一步优先看 `C_ref_tau_adv`。

  4. **风险：指标继续掩盖视觉失败。**

     * **识别方式**：即使 `boundary_dist90` 很小，`TrainC` 图仍出现明显团状/蛇形/大环。
     * **结论**：必须把 `TrainC` 重构指标和图提升为硬性 gate，不然 CSV 仍会误导。

  5. **风险：主图继续掩盖跑飞。**

     * **识别方式**：标准图看似“红点很少”，但 diagnostic 图发现红点跑到了绘图窗口外。
     * **结论**：主图尺度与 diagnostic 图必须联动；至少在失败案例上输出 auto-expanded 视图。

  ---

  一句收束结论：

  **基于这个 bundle，当前最该做的不是恢复 pair-margin / mismatch-D，也不是再纠结 random z；而是把实验主线从 `ref_y` 改回不含目标点的边界位置条件，首选 `ref_tau`，并把“TrainC 是否能重现当前边界”提升为第一验收门槛。**

  [1]: https://www.mathworks.com/help/deeplearning/custom-training-loops.html?utm_source=chatgpt.com "Custom Training Using Automatic Differentiation - MATLAB & Simulink
  "
