# 第一个问题

你是约束多目标优化、生成模型、GAN/CGAN、MATLAB/PlatEMO 方向的严格审稿人和算法设计顾问。请先完整读取我上传的 ZIP，再回答。不要凭经验泛泛建议，不要加入没有证据支持的猜测。每个关键判断都必须引用 ZIP 中的源码、CSV、MAT 或 PNG 路径作为依据。

  你的任务不是简单点评现有实验，而是：根据当前主线源码、历史分支源码、最近 git 状态、实验 CSV/MAT 和代表性图片，分析我的核心算法问题，并提出能真正解决问题的算法设计方案。

  # 1. ZIP 内容说明

  我上传的文件是：

  `CBS_CGAN_web_gpt_sourcewide_representative_20260624_222011.zip`

  这是一个 source-wide review package，源码比较全，图片是代表性筛选。

  包内大致包含：

  - 当前工作区源码快照；
  - 当前 CBS-CGAN 主线源码；
  - 当前 CCMO-GAN-BDG 源码；
  - 已提交历史快照；
  - 部分 PlatEMO 上下文源码；
  - CBS-CGAN 非 PNG 实验文件；
  - 代表性 PNG 图片；
  - git 状态、diff、grep、文件清单等元数据。

  请优先读取：

  - `README.md`
  - `MANIFEST.txt`
  - `metadata/git_status.txt`
  - `metadata/current_uncommitted_diff.patch`
  - `metadata/current_untracked_files.txt`
  - `metadata/key_mechanism_grep.txt`
  - `metadata/selected_png_files.txt`
  - `metadata/source_file_list.txt`
  - `metadata/ccmo_gan_bdg_git_history.txt`
  - `metadata/cbs_cgan_git_history.txt`

  请注意：`README.md` 里有一次 corrected snapshot extraction 说明。历史快照目录里部分路径带 `PlatEMO/` 前缀，请以实际包内路径为准。

  # 2. 当前主线源码

  当前主线源码是本次分析中心，请优先分析：

  - `source/current_worktree/PlatEMO/Algorithms/Multi-objective optimization/CBS-CGAN/`

  重点文件包括但不限于：

  - `CBS_CGAN.m`
  - `BoundaryCGAN_CBS.m`
  - `BuildBoundaryDataset_CBS.m`
  - `UpdateBoundaryMemory_CBS.m`
  - `PairLocalTau_CBS.m`
  - `EvaluateDecisions_CBS.m`
  - `CalFitness_CBS.m`
  - `EnvironmentalSelection_CBS.m`
  - `Support/run_CBS_CGAN_*`
  - `test_CBS_*`

  同时可以快速查看 focused copy：

  - `source/focused/CBS-CGAN_current_worktree/`

  当前源码中出现过 `ref_only`、`ref_tau`、`ref_y`、`ref_y_tau`、endpoint、`y_b_norm`、decision-space Huber、adversarial BCE、pair、tau、train_count 等机制。请结合源码判断这些机制是否语义统一，是否直接服务“生成目标空间边界解”的核心目标。

  # 3. 历史分支源码，只作为背景对比

  历史快照在：

  - `source/committed_snapshots/HEAD/`
  - `source/committed_snapshots/UC-GAN/`
  - `source/committed_snapshots/UC-GAN-2/`
  - `source/committed_snapshots/4212a3d5/`
  - `source/committed_snapshots/cf503f63/`
  - `source/committed_snapshots/7cbe0144/`

  其中历史边界存档对 + KNN 相关实现重点看：

  - `source/committed_snapshots/UC-GAN/PlatEMO/Algorithms/Multi-objective optimization/CCMO-GAN-BDG/CCMO_GAN_BDG.m`
  - `source/committed_snapshots/UC-GAN/PlatEMO/Algorithms/Multi-objective optimization/CCMO-GAN-BDG/UpdateBoundaryArchive_BDG.m`
  - `source/committed_snapshots/UC-GAN/PlatEMO/Algorithms/Multi-objective optimization/CCMO-GAN-BDG/BuildBoundaryTargetTriples_BDG.m`
  - `source/committed_snapshots/UC-GAN/PlatEMO/Algorithms/Multi-objective optimization/CCMO-GAN-BDG/BoundaryConditionKNNKeepMask_BDG.m`
  - `source/committed_snapshots/UC-GAN/PlatEMO/Algorithms/Multi-objective optimization/CCMO-GAN-BDG/BuildBoundaryTrainBundle_BDG.m`
  - `source/committed_snapshots/UC-GAN/PlatEMO/Algorithms/Multi-objective optimization/CCMO-GAN-BDG/FilterBoundaryTargetTriples_BDG.m`

  也可以看当前 focused 版本：

  - `source/focused/CCMO-GAN-BDG_current_worktree/`
  - `source/focused/KNN.m`

  这些旧分支只作为背景和对比：它们能帮助理解边界存档、边界对、KNN 过滤、target triples 等思路，但不要把旧分支当成当前主方案。

  # 4. 实验结果位置

  代表性 PNG 在：

  - `experiments/selected_png/Data/CBS_CGAN/`

  包内约 72 张 PNG。请重点看这些图，尤其关注训练阶段红点和橙点的关系。

  非 PNG 实验文件在：

  - `experiments/all_non_png/Data/CBS_CGAN/`
  - `experiments/all_non_png/Data/CCMO_GAN_BDG/`

  其中包括 CSV、MAT、manifest、log/config-like 文件。请不要只看 CSV，也要结合 PNG；如果需要验证训练集、snapshot、条件重复、样本规模，可以读取 MAT 或相关 manifest。

  重点实验目录包括：

  - `experiments/selected_png/Data/CBS_CGAN/A_ref_only_adv_LIR6_run1_nGen30_20260623_220259/flat_train_reconstruction/`
  - `experiments/selected_png/Data/CBS_CGAN/B_ref_only_adv_huber_LIR6_run1_nGen30_20260623_234001/flat_train_reconstruction/`
  - `experiments/selected_png/Data/CBS_CGAN/train_quality_baseline_ABCD_LIR6_run1_20260624_024446/contact_sheets_baseline/`
  - `experiments/selected_png/Data/CBS_CGAN/train_quality_mechanism_sweep_LIR6_run1_20260624_094837/contact_sheets_mechanism/`
  - `experiments/selected_png/Data/CBS_CGAN/train_quality_epoch_sweep_D_default_20260624_130858/contact_sheets_epoch/`
  - `experiments/selected_png/Data/CBS_CGAN/train_quality_pair6_vs_pair3_D_default_epoch50_20260624_153705/contact_sheets_pair_compare/`
  - `experiments/selected_png/Data/CBS_CGAN/train_quality_endpoint_vs_default_D_epoch50_20260624_181601/contact_sheets_endpoint_compare/`
  - `experiments/selected_png/Data/CBS_CGAN/train_quality_endpoint_yb_norm_vs_default_D_epoch50_20260624_192053/contact_sheets_endpoint_yb_norm_compare/`
  - `experiments/selected_png/Data/CBS_CGAN/single_stage_overfit_endpoint_yb_norm_20260624_203803/`
  - `experiments/selected_png/Data/CBS_CGAN/single_stage_overfit_adv_only_no_huber_20260624_210752/`

  对应 CSV/MAT 通常在：

  - `experiments/all_non_png/Data/CBS_CGAN/A_ref_only_adv_LIR6_run1_nGen30_20260623_220259/`
  - `experiments/all_non_png/Data/CBS_CGAN/B_ref_only_adv_huber_LIR6_run1_nGen30_20260623_234001/`
  - `experiments/all_non_png/Data/CBS_CGAN/train_quality_baseline_ABCD_LIR6_run1_20260624_024446/`
  - `experiments/all_non_png/Data/CBS_CGAN/train_quality_mechanism_sweep_LIR6_run1_20260624_094837/`
  - `experiments/all_non_png/Data/CBS_CGAN/train_quality_epoch_sweep_D_default_20260624_130858/`
  - `experiments/all_non_png/Data/CBS_CGAN/train_quality_pair6_vs_pair3_D_default_epoch50_20260624_153705/`
  - `experiments/all_non_png/Data/CBS_CGAN/train_quality_endpoint_vs_default_D_epoch50_20260624_181601/`
  - `experiments/all_non_png/Data/CBS_CGAN/train_quality_endpoint_yb_norm_vs_default_D_epoch50_20260624_192053/`
  - `experiments/all_non_png/Data/CBS_CGAN/single_stage_overfit_endpoint_yb_norm_20260624_203803/`
  - `experiments/all_non_png/Data/CBS_CGAN/single_stage_overfit_adv_only_no_huber_20260624_210752/`

  请重点读取这些 CSV：

  - `train_quality_sweep_summary.csv`
  - `train_quality_sweep_stage_metrics.csv`
  - `train_quality_sweep_manifest.csv`
  - `train_quality_sweep_figure_manifest.csv`
  - `stage_metrics_all.csv`
  - `history_metrics_all.csv`
  - `run_summary.csv`
  - `single_stage_overfit_metrics.csv`
  - `single_stage_overfit_trace.csv`
  - `stage_trainset_profile.csv`
  - `stage_trainset_by_ref.csv`
  - `snapshot_summary.csv`

  # 5. 核心目的

  我的核心目的非常明确：

  我想用边界解训练 GAN/CGAN 相关网络，使 GAN/CGAN 直接生成边界解。

  更具体地说：

  - 输入来自当前优化过程中已经探索到的部分目标空间可行/不可行边界解。
  - 网络要学习当前边界的分布。
  - 生成器要直接生成完整决策变量 X。
  - 生成的 X 经问题评价后，目标值应该落在目标空间的可行/不可行边界上。
  - 我需要的是生成一条窄的边界线，而不是生成厚点云、普通散点、多样性点云，或者偏离边界的点。

  这里的“边界”只指目标空间中的可行/不可行约束边界，不是决策空间边界，也不是 Pareto front 本身。

  # 6. 最重要的算法价值点

  请特别注意：

  我的想法不是做 `目标空间坐标 y -> 决策变量 x` 的逆映射。

  我的目标是：

  通过部分边界解的学习，学到当前目标空间可行/不可行边界的分布，从而直接生成还未探索到的当前边界的其他部分。

  这是算法的核心创新价值。请围绕这个目标分析，不要把方案带偏成普通逆映射、普通插值、普通采样、普通局部搜索、普通 repair 或普通 surrogate。

  # 7. 当前遇到的问题

  请基于 ZIP 中的源码、CSV、MAT 和 PNG 证据分析以下问题。

  ## 问题 1：边界存档和训练集构造不够好

  当前的边界存档、边界记忆、训练样本构造可能没有干净表达目标空间可行/不可行边界。

  请分析：

  - 当前 `UpdateBoundaryMemory_CBS.m` 和 `BuildBoundaryDataset_CBS.m` 中，边界样本到底是怎么来的？
  - 训练集中的点是否真的是目标空间可行/不可行边界点？
  - 是否存在重复点、错误点、非边界点、厚带点？
  - 每个参考向量/分区保留样本的逻辑是否会导致训练集过稀、重复、退化或不连续？
  - 当前 pair、endpoint、tau、ref、`y_b_norm` 等字段是否语义一致？
  - 当前训练集是否足以支撑 GAN/CGAN 学到一条连续边界？
  - 历史 `CCMO-GAN-BDG` 的边界 archive pair、KNN、target triples 思路中，哪些部分值得保留，哪些部分不应继承？

  请给出一个更统一、更少分支、更语义干净的边界存档和训练集定义。

  ## 问题 2：不是逆映射，而是学习边界分布并生成未探索边界

  请围绕这个核心价值分析：

  - 如果条件中加入 `y_b_norm`，是否会让问题退化为 `y -> x` 的逆映射？
  - 如果使用 endpoint 作为目标，是否会让模型只记忆已有训练点？
  - 怎样设计 CGAN 条件，才能表达“当前边界上下文”而不是指定某个目标坐标？
  - 如果不用 CGAN，只用传统 GAN，是否能从部分边界样本中学习边界分布并生成未探索边界段？
  - 如果传统 GAN 可以，缺点是什么？
  - 如果必须使用 CGAN，条件应该是什么，才能既可控又不退化为逆映射？

  请给出清晰判断：传统 GAN、CGAN、或某种更干净的条件化 GAN，哪个更符合我的核心目标。

  ## 问题 3：CGAN 生成结果不够贴边

  当前图片中，红色生成点经常不能贴近橙色训练边界点，也没有形成窄边界线。常见现象包括：

  - 红点偏离橙点；
  - 红点形成弯曲轨迹；
  - 红点变成厚点云；
  - 红点集中在错误区域；
  - 早期和部分问题上生成点远离训练边界；
  - 训练阶段 reconstruction 都不稳定。

  请根据 `experiments/selected_png/Data/CBS_CGAN/` 中的图逐类分析：

  - A_ref_only_adv 原始训练图说明什么？
  - B_ref_only_adv_huber 原始训练图说明什么？
  - Huber 是否显著改善？
  - epoch 增大是否显著改善？
  - pair=6 是否比 pair=3 有实质提升？
  - endpoint 是否比默认目标更合理？
  - endpoint + `y_b_norm` 是否带来本质区别？
  - 单 stage 过拟合诊断说明当前训练方式、损失函数、网络结构或数据定义中哪一部分最可疑？

  请不要只说“训练不够”或“多加 epoch”。必须解释为什么训练阶段都贴不回训练点，以及如何让生成结果变成窄边界。

  ## 问题 4：请你总结当前最严峻的问题

  请你基于源码、CSV、MAT 和图片，总结当前最严峻的问题。

  我希望你判断：

  - 最严重的问题是边界存档/训练集本身不干净？
  - 是条件定义语义混乱？
  - 是 adversarial loss 本身无法约束边界贴合？
  - 是 decision-space Huber 与 objective-space 边界目标不一致？
  - 是网络容量/训练量不足？
  - 是固定 `z=0` 导致生成退化？
  - 是模型只会记忆已有点，不能学习边界分布？
  - 是评价指标没有真正刻画“边界窄度”？

  请按优先级排序，并给出证据路径。

  # 8. 我的疑问

  请重点回答以下疑问。

  ## 疑问 1：不用 CGAN，只用传统 GAN 是否真的不行？

  我想知道：如果目标是“通过部分边界解学习当前边界分布，并生成还未探索到的边界其他部分”，传统 GAN 是否真的无法完成？

  请严格分析：

  - 传统 GAN 能否学习边界样本分布？
  - 它是否可以生成未探索边界段？
  - 它的问题是否只是不可控、覆盖不足、容易生成厚带？
  - CGAN 的条件到底应该提供什么信息，才能解决传统 GAN 的不足？
  - 如果 CGAN 条件只是 `y_b_norm`，是否偏离了我的创新点？

  ## 疑问 2：当前算法连训练阶段都做不好，说明什么？

  当前算法在训练阶段 reconstruction 都经常达不到要求。这是否说明：

  - 训练方式有问题？
  - 损失函数有问题？
  - 网络结构有问题？
  - 训练集定义有问题？
  - 条件定义有问题？
  - GAN/CGAN 本身在这个训练集规模和条件设置下不适合？

  请基于单 stage 过拟合实验和训练阶段图像分析，不要泛泛而谈。

  # 9. 已知客观事实

  请把以下事实作为输入，但你仍需要自己从 ZIP 中核对。

  1. 当前 ZIP 是 source-wide 版本，源码较全，实验非图文件较多，PNG 是代表性筛选。
  2. 包内包含约 72 张代表性 PNG、约 2949 个 CSV、约 51 个 MAT、当前源码和多个历史快照。
  3. 当前主线源码位于：
     - `source/current_worktree/PlatEMO/Algorithms/Multi-objective optimization/CBS-CGAN/`
  4. 当前 focused 源码位于：
     - `source/focused/CBS-CGAN_current_worktree/`
  5. 历史 `CCMO-GAN-BDG` 边界 archive pair + KNN 背景位于：
     - `source/committed_snapshots/UC-GAN/PlatEMO/Algorithms/Multi-objective optimization/CCMO-GAN-BDG/`
  6. 当前实验图片位于：
     - `experiments/selected_png/Data/CBS_CGAN/`
  7. 当前实验 CSV/MAT 位于：
     - `experiments/all_non_png/Data/CBS_CGAN/`
  8. 当前实现和实验中出现过 `ref_only`、`ref_tau`、`ref_y`、`ref_y_tau`、endpoint、`y_b_norm`、decision-space Huber、adversarial BCE 等设计。
  9. 部分实验固定或常用：
     - `z=0`
     - `nGen=30`
     - `batch=32`
     - `runs=1`
     - LIRCMOP5-10
     - FE targets 包含 10000、30000、50000、70000、100000
  10. A/B/C/D 类型实验大致对应：
      - `A_ref_only_adv`：ref_only + adversarial BCE，无 Huber。
      - `B_ref_only_adv_huber`：ref_only + adversarial BCE + decision-space Huber。
      - `C_ref_tau_adv`：ref_tau + adversarial BCE。
      - D 系列通常是 ref_tau 或后续条件版本 + adversarial BCE + decision-space Huber。
  11. 单 stage 过拟合诊断请重点看：
      - `experiments/all_non_png/Data/CBS_CGAN/single_stage_overfit_endpoint_yb_norm_20260624_203803/single_stage_overfit_metrics.csv`
      - `experiments/all_non_png/Data/CBS_CGAN/single_stage_overfit_endpoint_yb_norm_20260624_203803/stage_trainset_profile.csv`
      - `experiments/all_non_png/Data/CBS_CGAN/single_stage_overfit_adv_only_no_huber_20260624_210752/single_stage_overfit_metrics.csv`
      - `experiments/all_non_png/Data/CBS_CGAN/single_stage_overfit_adv_only_no_huber_20260624_210752/single_stage_overfit_trace.csv`
  12. 在 `single_stage_overfit_endpoint_yb_norm_20260624_203803` 中，LIRCMOP7_BC targetFE=50000 的记录显示：
      - `train_count=42`
      - `condition_unique_count=16`
      - `condition_duplicate_count=26`
      - `condition_dim=4`
      - `condition_mode=ref_y`
      - `boundaryTargetMode=feasible_endpoint`
      - `online_like_adv_huber_epoch50` 的 `train_y_rec90_norm` 约为 `0.16757`
      - `supervised_huber_epoch200` 的 `train_y_rec90_norm` 约为 `0.01713`
      - `supervised_huber_epoch1000` 的 `train_y_rec90_norm` 约为 `0.01262`
  13. 在 `single_stage_overfit_adv_only_no_huber_20260624_210752` 中，LIRCMOP7_BC targetFE=50000 的记录显示：
      - `train_count=144`
      - `condition_unique_count=48`
      - `condition_duplicate_count=96`
      - `adv_only_epoch500_no_huber` 的 `train_y_rec90_norm` 约为 `0.059057`
  14. 这些数字只能作为事实背景；请结合图片判断，不要只根据 CSV 下结论。

  # 10. 核心约束

  请严格遵守以下约束。

  1. 禁止建议放弃 GAN/CGAN。
  2. 禁止把算法改成普通采样、普通局部搜索、普通修复算子或普通 surrogate 模型。
  3. 必须保留核心创新点：
     - 使用 GAN 或 CGAN；
     - 直接生成完整决策变量 X；
     - 边界只指目标空间可行/不可行边界；
     - 目标是通过部分边界解学习边界分布并生成未探索边界部分。
  4. 除核心创新点外，其他都可以改：
     - 边界存档；
     - 训练集构造；
     - 条件定义；
     - 网络结构；
     - 训练方式；
     - 损失函数；
     - 生成策略；
     - 评价指标。
  5. 算法设计优先级：
     - 统一；
     - 减法；
     - 收敛；
     - 语义干净；
     - 直接服务核心创新点。
  6. 不要提出没有依据、大概率没有效果的花里胡哨小技巧。
  7. 不要堆很多看起来复杂但不解决核心问题的小模块。
  8. 可以设计少量分支实验，但不要设计太多分支；每个分支必须回答一个关键问题。
  9. 请用最严格的标准分析，不要为了给建议而给建议。

  # 11. 请重点分析的方向

  ## A. 边界存档和训练集结构

  请给出你认为更合理的边界存档结构。至少说明：

  - 每条边界记录应该保存什么？
  - 是否应该保存 feasible endpoint、infeasible endpoint、边界插值点、约束符号、约束违反值、局部法向、局部切向、参考向量分区、stage/FE 信息？
  - 如何定义“这条样本确实是边界样本”？
  - 如何过滤掉非边界点、重复点、厚带点？
  - 如何保持边界覆盖和连续性？
  - 每个参考向量/局部片段应该保存多少样本？
  - 训练集最终应该从边界存档中怎样抽取？

  请给出字段级定义，而不是只说原则。

  ## B. GAN/CGAN 输入输出

  请明确设计：

  - 生成器输入是什么？
  - 生成器输出是什么？
  - 判别器输入是什么？
  - 是否需要条件？
  - 条件是什么？
  - 条件如何避免变成 `y -> x` 逆映射？
  - `z` 是否应该使用？如果使用，语义是什么？如果不使用，模型是否退化？
  - 训练阶段和生成阶段的条件语义是否完全一致？

  ## C. 损失函数

  请分析当前 adversarial BCE 和 decision-space Huber 的问题。

  请设计更合理的损失组合，但不要堆料。候选方向可以包括但不限于：

  - adversarial loss；
  - decision-space reconstruction；
  - objective-space boundary consistency；
  - feasible/infeasible side consistency；
  - local tangent/normal thickness penalty；
  - diversity/coverage loss；
  - boundary projection consistency。

  你必须说明每个损失项为什么必要，解决什么具体失败现象。如果某个损失项不是必要项，请不要加。

  ## D. 评价指标

  请提出能量化“生成的是窄边界而不是厚点云”的指标。至少考虑：

  - 到目标空间可行/不可行边界的距离；
  - 局部法向厚度；
  - `segment_width90`；
  - `side_rate`；
  - 可行/不可行两侧覆盖；
  - 生成点沿边界的覆盖率；
  - 与训练边界点的距离；
  - 是否只记忆训练点；
  - 未探索区域覆盖；
  - 目标空间厚度与切向长度的比值。

  请给出指标定义、计算方式和通过/失败标准。

  ## E. 最小诊断实验

  请设计最小实验矩阵，不要设计大规模堆料实验。

  每个实验必须说明：

  - 固定什么；
  - 改变什么；
  - 用哪些图片判断；
  - 用哪些 CSV/MAT 指标判断；
  - 什么结果说明假设成立；
  - 什么结果说明假设不成立。

  优先区分以下因素：

  1. 训练集是否不干净；
  2. 条件定义是否错误；
  3. adversarial loss 是否不足；
  4. Huber 是否帮错方向；
  5. 训练量是否只是次要因素；
  6. `z=0` 是否导致生成退化；
  7. 当前网络是否根本无法在小训练集上稳定学习边界；
  8. 是否需要先做单 stage 可学习性，再回到在线算法。

  # 12. 输出格式要求

  请严格按以下结构回答：

  ## 1. 证据事实

  先列出你从源码、CSV、MAT、PNG 中看到的事实。
  每条事实必须带路径，例如：

  - `source/current_worktree/PlatEMO/...`
  - `source/committed_snapshots/...`
  - `experiments/selected_png/...`
  - `experiments/all_non_png/...`
  - `metadata/...`

  不要在这一节提出方案。

  ## 2. 主要失败机制

  按严重程度排序。
  每个失败机制必须说明：

  - 现象；
  - 证据路径；
  - 为什么它会导致红点不贴边或生成厚点云；
  - 它和核心目标的冲突是什么。

  ## 3. 对 GAN vs CGAN 的判断

  请明确回答：

  - 传统 GAN 是否可能完成目标？
  - CGAN 是否必要？
  - 如果用 CGAN，条件应该是什么？
  - 怎样避免 CGAN 退化成目标到决策变量的逆映射？

  ## 4. 最干净的算法重构方案

  请给出你认为最应该采用的重构方案。
  要求：统一、减法、收敛、语义干净，不要堆模块。

  必须包含：

  - 边界存档结构；
  - 训练集字段；
  - G/D 输入输出；
  - 条件定义；
  - 损失函数；
  - 训练流程；
  - 生成流程；
  - 如何保证生成窄边界；
  - 如何避免只记忆已有训练点；
  - 如何生成未探索边界部分。

  ## 5. 最小实验矩阵

  给出 3 到 6 个最关键实验即可。
  不要给十几个分支。
  每个实验必须有明确目的和通过/失败标准。

  ## 6. 优先修改的源码位置

  请列出最应该优先修改的文件和方向，例如：

  - `UpdateBoundaryMemory_CBS.m`
  - `BuildBoundaryDataset_CBS.m`
  - `BoundaryCGAN_CBS.m`
  - `CBS_CGAN.m`
  - `Support/run_CBS_CGAN_*`

  每个文件说明要改什么、为什么改、如何验证。

  ## 7. 最终建议

  最后给一个简洁结论：

  - 当前最大问题是什么；
  - 第一优先级该改什么；
  - 第二优先级该验证什么；
  - 不应该继续做什么。

# 第一个问题的回答

我先做了包级清单扫描，再按你指定优先级读取了 `README.md`、`MANIFEST.txt`、`metadata/*`、当前主线 CBS-CGAN、历史 CCMO-GAN-BDG 关键文件、你点名的关键 CSV/MAT，以及代表性 PNG。下面只对能落到明确路径的内容下结论。

## 1. 证据事实

- `README.md` 说明这是 corrected snapshot extraction 后的 source-wide 包；当前工作分支是 `endpoint-yb-norm`；包内统计为 `2890` 个源码文件、`2984` 个非 PNG 实验文件、`72` 张代表性 PNG。`metadata/git_status.txt` 也显示当前 branch 为 `endpoint-yb-norm`。
- `metadata/git_status.txt` 和 `metadata/current_uncommitted_diff.patch` 显示当前主线仍在强烈变动，核心改动集中在
  `source/current_worktree/PlatEMO/Algorithms/Multi-objective optimization/CBS-CGAN/BoundaryCGAN_CBS.m`、
  `BuildBoundaryDataset_CBS.m`、`CBS_CGAN.m`、`UpdateBoundaryMemory_CBS.m`；同时还存在未跟踪但与你问题直接相关的
  `Support/run_CBS_CGAN_holdout_ref_eval.m`、
  `Support/run_CBS_CGAN_non_cgan_baseline_eval.m` 等脚本。说明你已经在做“holdout ref”和“non-CGAN baseline”这两个正确方向的验证。
- `BoundaryCGAN_CBS.m` 文件头仍把模型写成“`Minimal conditional GAN for boundary point inverse mapping`”；默认项里又把
  `trainZMode="zero"`、`sampleZMode="zero"`、`advWeight=1`、`reconstructionWeight=1` 写死。也就是说，当前主线默认是“零噪声 + 对抗 BCE + 决策空间 Huber”，而不是“边界分布生成 + objective-space boundary 约束”。
- `metadata/current_uncommitted_diff.patch` 里，接口从 `G(z,y_b)->x` 漂到 `G(z,[u_ref,tau])->x`；但 `BuildBoundaryDataset_CBS.m` 仍保留 `ref_y`、`ref_y_tau` 两类直接把 `y_b_norm` 或 `QueryY` 喂进条件的分支。接口语义已经漂移，但没有收敛成一个统一定义。
- `UpdateBoundaryMemory_CBS.m` 中，边界记忆不是从“所有靠近可行/不可行边界的可行点”构造，而是先取
  `mainFeasible = firstFrontMask(Y(feasible,:))`，即只从可行集的 PF 主层取 feasible 端；infeasible 端再从 `neighborRefs(..., pairNeighborRefRadius)` 里取，默认半径在 `CBS_CGAN.m` 是 `4`。然后目标点 `y_b` 还会在 `feasible_endpoint` 与 “near-segment feasible support” 之间切换；最后又被 `filterParetoMainLayer(Candidate)` 再做一次 Pareto 主层过滤。你的 archive 从定义上就是 PF-biased，而不是“纯边界”。
- `BuildBoundaryDataset_CBS.m` 中的四种条件
  `ref_only / ref_tau / ref_y / ref_y_tau` 语义并不统一：
  `ref_only` 只有 partition；
  `ref_tau` 加的是局部 chord 坐标；
  `ref_y` 直接包含 `y_b_norm`；
  `ref_y_tau` 则接近“目标坐标 + chord 坐标”。
  同文件里 `TrainTau` 会被 `PairLocalTau_CBS.m` 重新按 `(y_f,y_i)` 的 objective chord 投影重算，覆盖掉 archive 中原先的 `tau`。所以当前 `tau` 不是全局边界参数，只是 pair chord 上的局部投影。
- `BuildBoundaryDataset_CBS.m` 的 `buildExternalQueries` 会为 `missing_ref` 和 `large_gap` 构造 query；但 query 的“边界顺序”来自 `sortedBoundaryNodes(BMem,W)`，它按 reference vector 的 SVD 排序分数来排，不是按真实 objective-space boundary 链的几何邻接来排。这会把不该相连的边界节点接成 query interval。
- `UpdateBoundaryMemory_CBS.m` 的 endpoint 模式会直接把 `y_b=y_f`，因此 `tau=PairLocalTau_CBS(y_f,y_f,y_i)=0`。这一点不只是代码层面的推断，`experiments/all_non_png/Data/CBS_CGAN/train_quality_endpoint_vs_default_D_epoch50_20260624_181601/train_quality_sweep_stage_metrics.csv` 与
  `.../train_quality_endpoint_yb_norm_vs_default_D_epoch50_20260624_192053/train_quality_sweep_stage_metrics.csv` 中，LIRCMOP7_BC 各 targetFE 行的 `train_tau_nonzero_rate=0`、`train_tau_range=0`，与代码完全一致。也就是说，endpoint 一开，`ref_tau` 条件事实上退化。
- 在 `experiments/all_non_png/Data/CBS_CGAN/single_stage_overfit_endpoint_yb_norm_20260624_203803/stage_trainset_profile.csv`，LIRCMOP7_BC、targetFE=50000 的记录是
  `train_count=42`、`condition_unique_count=16`、`condition_duplicate_count=26`、`train_tau_range=0`。
  同目录 `single_stage_overfit_metrics.csv` 显示：
  `online_like_adv_huber_epoch50` 的 `train_y_rec90_norm=0.167571`；
  `supervised_huber_epoch200=0.017130`；
  `supervised_huber_epoch1000=0.012617`。
  再对 `.../LIRCMOP7_BC_run1_targetFE050000_snapshot.mat` 按当前 `BuildBoundaryDataset_CBS.m` 公式重算，`ref_only/ref_tau/ref_y/ref_y_tau` 的 unique 条件数分别只有 `14/14/16/16`。这不是“小数据学不会”，而是“同一 condition 对应多个 x”。
- 在 `experiments/all_non_png/Data/CBS_CGAN/single_stage_overfit_adv_only_no_huber_20260624_210752/single_stage_overfit_metrics.csv`，对应 LIRCMOP7_BC、targetFE=50000 的
  `adv_only_epoch500_no_huber` 仍只有 `train_y_rec90_norm=0.059057`；同目录 `stage_trainset_profile.csv` 给出 `train_count=144`、`condition_unique_count=48`、`condition_duplicate_count=96`。训练样本变多了，但 condition collision 仍然严重。
- 重要的是，底层 pair 本身并不厚。
  `single_stage_overfit_endpoint_yb_norm_20260624_203803/stage_trainset_profile.csv` 里 `train_pair_obj_dist90_norm=0.001941`；
  `single_stage_overfit_adv_only_no_huber_20260624_210752/stage_trainset_profile.csv` 里 `train_pair_obj_dist90_norm=0.000943`。
  所以最严重的问题不是“pair gap 太厚”，而是 pair 之上的 archive/condition/loss 语义错位。
- `experiments/all_non_png/Data/CBS_CGAN/train_quality_baseline_ABCD_LIR6_run1_20260624_024446/train_quality_sweep_manifest.csv` 证实
  A=`ref_only+adv`，B=`ref_only+adv+huber`，C=`ref_tau+adv`，D=`ref_tau+adv+huber`，而且都还是 `trainZMode=zero`、`sampleZMode=zero`。
  同目录 `train_quality_sweep_stage_metrics.csv` 的整体中位数 `train_y_rec90` 分别是
  A `1.0270`，B `0.9949`，C `1.1840`，D `0.6307`。
  说明 Huber 有帮助，但 D 仍明显不够好。
- `experiments/all_non_png/Data/CBS_CGAN/train_quality_epoch_sweep_D_default_20260624_130858/train_quality_sweep_stage_metrics.csv` 的整体中位数 `train_y_rec90` 是
  epoch20 `0.7904`，epoch50 `0.6459`，epoch100 `0.7938`；
  在 LIRCMOP7_BC 上更不单调，例如 FE50000 是 `0.4597 -> 1.0467 -> 1.4972`。所以“多训一点 epoch”不是主解。
- `experiments/all_non_png/Data/CBS_CGAN/train_quality_pair6_vs_pair3_D_default_epoch50_20260624_153705/train_quality_sweep_stage_metrics.csv` 相对 pair3 的整体中位数变化是
  `train_y_rec90 +0.154`、`query_obj_dist90 +0.0034`、`ref_cover -0.036`，总体变差。
  LIRCMOP7_BC FE100000 更极端：`train_y_rec90 0.5403 -> 2.9533`，`boundary_dist90 0.0060 -> 0.0572`。
  增加 pair 数并没有解决主问题。
- `experiments/all_non_png/Data/CBS_CGAN/train_quality_endpoint_yb_norm_vs_default_D_epoch50_20260624_192053/train_quality_sweep_stage_metrics.csv` 的 LIRCMOP7_BC FE50000 行显示：
  default 到 endpoint+y_b_norm，`boundary_dist90 0.0138 -> 0.0459`，`train_y_rec90 1.0467 -> 3.5041`，但 `feasible_rate 0.1667 -> 0.3333`，`ref_cover 0.1316 -> 0.4286`。
  这说明当前 coarse 指标会把“更差的几何贴边”误判成“更好”。
- 代表性 PNG 与 CSV 一致：
  `experiments/selected_png/Data/CBS_CGAN/A_ref_only_adv_LIR6_run1_nGen30_20260623_220259/flat_train_reconstruction/LIRCMOP7_BC_run1_targetFE050000_train_reconstruction.png`
  有明显 detached U 型下支和上方离边簇；
  `.../B_ref_only_adv_huber.../LIRCMOP7_BC_run1_targetFE050000_train_reconstruction.png`
  更紧，但仍有平直离边支；
  `.../train_quality_epoch_sweep_D_default_20260624_130858/contact_sheets_epoch/LIRCMOP7_BC_epoch_contact_sheet.png`
  里 epoch100 不是稳定优于 epoch50；
  `.../train_quality_endpoint_yb_norm_vs_default_D_epoch50_20260624_192053/contact_sheets_endpoint_yb_norm_compare/LIRCMOP7_BC_default_vs_endpoint_yb_norm_epoch50.png`
  在 FE50000、FE100000 出现更糟的分叉和错区块；
  `.../single_stage_overfit_endpoint_yb_norm_20260624_203803/LIRCMOP7_BC_run1_targetFE050000_overfit_reconstruction.png`
  则直观看到 `online_like_adv_huber_epoch50` 远离训练边界，而 `supervised_huber_epoch200/1000` 已明显贴回去。
- `source/current_worktree/PlatEMO/Algorithms/Multi-objective optimization/CBS-CGAN/CBS_CGAN.m` 里的 `generatedPairSideStats` 用的是 decision-space 距离 `dist(x,x_f)` 与 `dist(x,x_i)` 去算 `side_rate`；同文件的 `generatedSegmentWidth90` 不是“生成点局部法向厚度”，而是来自 `queryMeta.source_interval` 的源区间宽度。再加上 `segment_width90_ratio` 在
  `train_quality_baseline_ABCD.../train_quality_sweep_stage_metrics.csv`、
  `train_quality_epoch_sweep_D_default.../train_quality_sweep_stage_metrics.csv`、
  `train_quality_endpoint_vs_default.../train_quality_sweep_stage_metrics.csv`、
  `train_quality_endpoint_yb_norm_vs_default.../train_quality_sweep_stage_metrics.csv`
  里全为 `NaN`，所以当前 sweep 实际上没有可靠量化“窄边界”。
- `metadata/ccmo_gan_bdg_git_history.txt` 直接写着：`5-23：边界数据训练有效，GAN生成无效`、`6-14...生成解仍然不是很贴边`。这和当前结果是连续的，不是最近一两个 patch 才出现的问题。

## 2. 主要失败机制

先给结论：**最严重的问题不是 GAN 本身，也不是底层 pair gap 太厚；而是你把“边界 pair → archive target → 条件 → 训练损失 → 评价指标”这五层语义接歪了。**

### 1. 边界 archive 不是“目标空间可行/不可行边界”的干净表达

- 现象：训练集看起来在学 boundary，实际上更像在学“可行 PF 主层附近的一条被 pair 过、再被 Pareto 过滤过的 support 链”。
- 证据路径：
  `source/current_worktree/PlatEMO/Algorithms/Multi-objective optimization/CBS-CGAN/UpdateBoundaryMemory_CBS.m`；
  `.../BuildBoundaryDataset_CBS.m`；
  `experiments/all_non_png/Data/CBS_CGAN/single_stage_overfit_*/*stage_trainset_profile.csv`。
- 为什么会导致红点不贴边或变厚：
  1. feasible 端先被 `firstFrontMask` 限死在可行 PF 主层；
  2. `y_b` 又在 endpoint 和 near-segment support 两种语义间切换；
  3. 再被 `filterParetoMainLayer` 过滤；
  4. query interval 还按 ref-SVD 顺序连边，而不是真实几何邻接。
     结果是：网络看到的“边界”本身就不是统一对象，query 还会把本不相邻的局部片段接起来，红点自然会被拉成弯曲支、竖直支、厚云团。
- 和核心目标的冲突：你明确说“边界不是 Pareto front 本身”。当前 `UpdateBoundaryMemory_CBS.m` 却两次把 archive 往 PF 主层上压，这与目标直接冲突。

### 2. 条件定义同时存在“过弱退化”和“过强逆映射”两种失败；再叠加 `z=0`，条件塌缩非常严重

- 现象：
  `ref_only/ref_tau` 太弱，很多不同 `x` 共用同一条件；
  `ref_y/ref_y_tau` 又太强，开始逼近 `y -> x`；
  endpoint 模式更是把 `tau` 直接清零。
- 证据路径：
  `source/current_worktree/PlatEMO/Algorithms/Multi-objective optimization/CBS-CGAN/BoundaryCGAN_CBS.m`；
  `.../BuildBoundaryDataset_CBS.m`；
  `.../PairLocalTau_CBS.m`；
  `experiments/all_non_png/Data/CBS_CGAN/single_stage_overfit_endpoint_yb_norm_20260624_203803/stage_trainset_profile.csv`；
  `.../single_stage_overfit_metrics.csv`；
  `.../LIRCMOP7_BC_run1_targetFE050000_snapshot.mat`；
  `.../train_quality_endpoint_vs_default_D_epoch50_20260624_181601/train_quality_sweep_stage_metrics.csv`；
  `.../train_quality_endpoint_yb_norm_vs_default_D_epoch50_20260624_192053/train_quality_sweep_stage_metrics.csv`。
- 为什么会导致红点不贴边或变厚：
  当 `trainZMode=zero`、`sampleZMode=zero` 时，生成器对每个 condition 是确定映射。
  但你在关键 snapshot 上，`42` 个训练样本只有 `14~16` 个 unique condition；也就是**同一 condition 必须解释多个不同 x**。
  这时：
  - 加 Huber，会回归到平均值，生成“中间厚带”；
  - 只做 adversarial，会在几个模式之间抖，形成 detached 支和错区块。
    而 `ref_y` 虽然能提高部分 seen-point 重建，但它把问题带向“给定目标坐标找 x”，偏离了你的创新点。
- 和核心目标的冲突：你的目标是“学边界分布并补全未探索边界”，不是“把一个 `y` 精确反解成 `x`”。`ref_y` 类条件在语义上已经向逆映射滑过去了。

### 3. 训练损失与目标错位：当前优化的是 “x-space 可分辨/可重构”，不是 “objective-space 窄边界”

- 现象：A/C 的 adversarial-only 明显不够；B/D 的 x-space Huber 虽有帮助，但仍无法稳定贴回训练边界；甚至某些配置下 x-space 指标略好，而 y-space 更差。
- 证据路径：
  `source/current_worktree/PlatEMO/Algorithms/Multi-objective optimization/CBS-CGAN/BoundaryCGAN_CBS.m`；
  `experiments/all_non_png/Data/CBS_CGAN/train_quality_baseline_ABCD_LIR6_run1_20260624_024446/train_quality_sweep_stage_metrics.csv`；
  `.../single_stage_overfit_endpoint_yb_norm_20260624_203803/single_stage_overfit_metrics.csv`；
  `.../single_stage_overfit_adv_only_no_huber_20260624_210752/single_stage_overfit_metrics.csv`；
  `.../train_quality_endpoint_yb_norm_vs_default_D_epoch50_20260624_192053/train_quality_sweep_stage_metrics.csv`。
- 为什么会导致红点不贴边或变厚：
  当前 `BoundaryCGAN_CBS.m` 没有任何直接面向 objective-space boundary 的训练约束。
  这会出现两类错配：
  1. x-space 上“折中得还可以”的解，评价后 objective-space 仍可能离边很远；
  2. adversarial 只关心“像不像训练分布”，不关心“是不是落在目标 interval 的窄边界上”。
     所以你会反复看到：训练阶段连 orange 都贴不回，更不用说生成未探索边界。
- 和核心目标的冲突：你的成功标准是 objective-space 上一条窄边界线，而不是 decision-space 上一个看起来像训练样本的点。

### 4. 现在的评价指标会把错误方向当成进步

- 现象：某些设置 `feasible_rate`、`ref_cover` 变好，但图片更差；而真正该看“窄边界”的指标要么定义错位，要么根本没产出。
- 证据路径：
  `source/current_worktree/PlatEMO/Algorithms/Multi-objective optimization/CBS-CGAN/CBS_CGAN.m`；
  `experiments/all_non_png/Data/CBS_CGAN/train_quality_endpoint_yb_norm_vs_default_D_epoch50_20260624_192053/train_quality_sweep_stage_metrics.csv`；
  `.../train_quality_baseline_ABCD.../train_quality_sweep_stage_metrics.csv`；
  `.../train_quality_epoch_sweep_D_default.../train_quality_sweep_stage_metrics.csv`。
- 为什么会导致红点不贴边或变厚：
  `side_rate` 现在是 decision-space 的“更靠近 `x_f` 还是 `x_i`”，不是 objective-space 的“在边界哪一侧”；
  `segment_width90_ratio` 在主 sweep 里全是 `NaN`；
  `boundary_dist90` 只是到当前 `BMem.y_b` polyline 的距离，而 `BMem` 自己已经 PF-biased。
  因此 sweep 很容易选中“coverage 更高但更不贴边”的方向。
- 和核心目标的冲突：你的核心指标应该是“窄、贴边、能补 missing interval”，而不是“覆盖了一些 ref、出了更多 feasible”。

### 5. 训练量和网络容量是次要问题，不是第一问题

- 现象：更多 epoch 不稳定；更多 pair 甚至更差；但同一个 snapshot，纯 supervised Huber 又能很好地拟合。
- 证据路径：
  `experiments/all_non_png/Data/CBS_CGAN/train_quality_epoch_sweep_D_default_20260624_130858/train_quality_sweep_stage_metrics.csv`；
  `.../train_quality_pair6_vs_pair3_D_default_epoch50_20260624_153705/train_quality_sweep_stage_metrics.csv`；
  `.../single_stage_overfit_endpoint_yb_norm_20260624_203803/single_stage_overfit_metrics.csv`。
- 为什么不是主因：
  如果主因是“网络太小/epoch 不够”，那 pair6 不该整体变差，epoch100 也不该经常比 epoch50 更差；更关键的是，同一 42 点 snapshot，supervised Huber 200/1000 epoch 已能把 `train_y_rec90_norm` 压到 `0.017/0.013`。
  说明**可学习性在局部是存在的**，但当前在线目标和条件设计没把它转成正确的生成行为。
- 和核心目标的冲突：继续堆 epoch、堆 pair，只会在错误语义上更用力。

## 3. 对 GAN vs CGAN 的判断

- **传统 GAN 是否可能完成目标？**
  可以。不是“不行”。如果 boundary archive 真的是干净的、薄的、连续的边界样本分布，无条件 GAN 完全可以学习这个经验分布，并生成训练集中没见过但仍位于同一分布上的边界点。
  你的包里也已经把这条路当成正式对照了：
  `source/current_worktree/PlatEMO/Algorithms/Multi-objective optimization/CBS-CGAN/Support/run_CBS_CGAN_non_cgan_baseline_eval.m`；
  历史上 `source/committed_snapshots/UC-GAN/PlatEMO/Algorithms/Multi-objective optimization/CCMO-GAN-BDG/CCMO_GAN_BDG.m` 也是“直接 GAN 生成完整 X”的路线。
- **传统 GAN 的真正短板是什么？**
  不是“学不到分布”，而是**不可控**：
  1. 容易偏向当前高密度、易学的边界片段；
  2. 难以有意识地补 `missing_ref` / `large_gap`；
  3. 在线时很难把生成预算集中到“还没探索到的边界区间”。
     所以传统 GAN 应该保留为 baseline，但不适合作主线。
- **CGAN 是否必要？**
  对“在线补未探索边界区间”的主线，**需要**。
  但这个条件不能是 `y_b_norm`，也不该是完整 `QueryY`。那会把问题推成 `y -> x` 逆映射。
- **我建议的条件是什么？**
  用**边界上下文**，不用完整 objective 坐标。
  最干净的条件是
  `c = [component_id, interval_id, u]`，
  其中 `interval_id` 表示当前边界链上的一个局部区间，`u∈[0,1]` 表示该区间内的相对位置。
  在当前代码里，最容易落地的等价编码是
  `c = [component_id, s_arc_norm]`，
  即把条件定义成“当前边界链上的低维参数化位置”，而不是完整 `y`。
  这仍然是在学“边界分布”，不是在做 full-objective inverse mapping。
- **怎样避免退化成逆映射？**
  三条硬约束：
  1. 条件里不放完整 `y_b_norm`，不放完整 `QueryY`；
  2. 条件只描述“当前边界的局部片段/相对位置”；
  3. query 只从“缺失区间/大 gap 区间”发起，不从“训练点 ID”发起。
     这样模型学的是“边界链上的局部生成规律”，不是“给定 2D 目标坐标，反推出 x”。

从外部文献看，GAN 在演化优化里确实存在三条不同路线：GMOEA 是无条件地从当前样本分布生成后代；G2S 是用 objective guiding points 条件化地把目标空间信息映回决策空间；GAN-LMEF 是沿 learned manifold 做插值；而 BE-CBO 则强调 unknown-constraint 场景里 boundary-aware exploration 本身就是核心。因此，你这里最合适的位置不是 full `y -> x`，而是 **boundary-context-conditioned direct-X generation**。([IEEE Xplore](https://ieeexplore.ieee.org/document/9082904/?utm_source=chatgpt.com))

## 4. 最干净的算法重构方案

我建议把当前主线收缩成一个单一方案：**Certified Boundary Segment CGAN**。核心是：**archive 的基本单元不再是混合语义的 `x_b/y_b` 点，而是“有一侧 feasible、一侧 infeasible 的 boundary segment”。**

### A. 边界 archive 结构

每条 archive 记录保存一个 **segment**，不是一个 `b-point`：

- `stageFE`：采样阶段。
- `component_id`：边界连通分量编号。
- `ref_left, ref_right`：该 segment 所属的 coarse ref 区间。
- `x_f, y_f`：feasible 端点。
- `x_i, y_i`：infeasible 端点。
- `flag_f=1, flag_i=0`：两侧可行性标签。
- `gap_obj_norm = ||y_i-y_f||`：归一化 objective gap。
- `gap_dec = ||x_i-x_f||`：decision gap。
- `mid_y = 0.5*(y_f+y_i)`：objective 中点。
- `n_obj = normalize(y_i-y_f)`：局部 objective 法向，直接表示“可行→不可行”方向。
- `t_obj`：由相邻 segment 的 `mid_y` 估计的局部切向。
- `arc_s`：该 segment 在当前 boundary chain 上的弧长坐标。
- `conf`：置信度，可由 `gap_obj_norm`、连续性、一致性组合得到。

**不再把 `x_b/y_b` 作为主 archive target 字段保存。**
`boundary interpolation point` 可以按 query 临时算，不应成为 archive 的主对象；这是当前语义混乱的源头之一。

### B. “这条样本确实是边界样本”的定义

一条 segment 进入 archive，至少满足：

1. 一端 feasible，一端 infeasible。
2. `gap_obj_norm <= ε_gap(stage)`；先按你现在的 profile，起步阈值就用当前 pair 的 90% 分位附近。
3. 只允许同 ref 或相邻 ref 配对；**去掉默认 radius=4 的宽邻域**。
4. 不再要求 feasible 端来自 PF 第一层；**去掉 `firstFrontMask` 和 `filterParetoMainLayer`**。
5. 去重规则改为 `(mid_y, n_obj)` 联合去重，而不是“每 ref 留一个点”：
   若两个 segment 的 `||mid_y^a-mid_y^b|| < ε_mid` 且 `acos(n_a·n_b) < ε_ang`，只保留 gap 更小者。
6. 每个局部区间最多保留 `2` 条 segment：
   一条最小 gap 的主记录；
   一条与主记录 `mid_y` 足够分离的 continuity 备份。
   这样既不稀，也不爆重复。

### C. 训练集字段

训练样本用 **feasible near-boundary endpoint** 作为 real data：

- `x_real = x_f`
- 条件 `c = [component_id, interval_id, u]`
  或当前代码更易落地的 `c = [component_id, s_arc_norm]`
- batch metadata 保留：`x_i, y_f, y_i, n_obj, t_obj, gap_obj_norm, arc_s`

这里的关键减法是：

- 主线删除 `ref_y`
- 主线删除 `ref_y_tau`
- 主线删除 endpoint / near-support 两套 targetMode 分支
- 主线不再用 `y_b_norm` 作为条件

历史 BDG 里值得保留的是
`UpdateBoundaryArchive_BDG.m` 那种“显式 pair archive”的思想；
不应该继续继承的是
`BuildBoundaryTargetTriples_BDG.m`、`FilterBoundaryTargetTriples_BDG.m`、`BoundaryConditionKNNKeepMask_BDG.m` 这种 `targetMode/conditionMode/KNN` 分支堆叠。

### D. G / D 输入输出

- 生成器：`G(z, c) -> x`
- 判别器：`D(x, c) -> score`
- `z` 必须启用，`zDim=2~4` 即可
- 默认：
  - `trainZMode = random`
  - `sampleZMode = random`

`z` 的语义不是“全局随机性”，而是**同一局部边界上下文下的候选多样性**。
这一步不是小修，是必须改。你现在的 zero-z 在 condition collision 下会强迫一个条件只输出一个 x。

### E. 损失函数

我建议只保留三项，别再堆：

1. **对抗损失 `L_adv`**
   保留现有 adversarial loss 即可，先不把重点放在 BCE/WGAN 体制切换上。
2. **pair-side contrastive loss `L_side`**
   对每个 batch 记录的 `(x_f, x_i)`，让生成解更靠近 feasible 端、远离 infeasible 端：
   `L_side = max(0, m + ||x_hat-x_f|| - ||x_hat-x_i||)`
   它解决的是“生成点漂到错误区域/错误侧”。
3. **短时 warm-start Huber `L_ws`**
   只在训练前 `T_ws` 个 epoch 开启，用来稳住初期训练；之后关闭。
   你现有 single-stage 结果已经证明 Huber 在小样本局部可学习性上有用，但不该一直作为主目标。

所以主线就是：

- `L_D = L_adv`
- `L_G = λ_adv L_adv + λ_side L_side + λ_ws L_ws`

**不加长期 decision-space Huber。**
**不加一堆 tangent/diversity 小模块。**

### F. objective-space boundary consistency 不放到反向传播里，放到“候选选择”里

你的问题是黑盒 `Problem.Evaluation`。在 `PlatEMO` 当前框架里，不应硬做一个伪 surrogate 再把 objective loss 反传。最干净的办法是：

1. 生成器对每个 query context 采样 `K=4~8` 个候选 `x_hat`；
2. 用真实 `Problem.Evaluation` 或 `EvaluateDecisions_CBS.m` 评价；
3. 在 **objective space** 上给每个候选打 boundary score；
4. 只从 GAN 已生成的候选里选最贴边者，不做 repair，不做 local search。

对 query interval `I_q`，定义：

- `d_seg(y, I_q)`：`y` 到目标 interval 的最短距离
- `d_tan(y, q)`：`y` 在 interval 切向上的相对位置误差
- `S_f = d_seg + 0.2*d_tan + M*[infeasible]`
- `S_i = d_seg + 0.2*d_tan + M*[feasible]`

然后：

- 取 `S_f` 最小的 feasible 候选注入主种群；
- 取 `S_i` 最小的 infeasible 候选只用于刷新 archive / 辅助种群，不直接当最终解。

这样同时保留了两侧信息，但**核心动作仍然是 GAN/CGAN 直接生成完整 `X`**。

### G. 训练流程

1. 从当前两种群更新 certified boundary segments。
2. 按 objective-space 几何邻接重建 boundary chain / components。
3. 由 chain 提取局部区间与 `arc_s`。
4. 组装 `(x_f, c, metadata)` 训练集。
5. 用 `random z` 训练 CGAN。
6. 每个 query context 采样多候选并真实评价。
7. 选择最贴边 feasible 候选注入主种群；最贴边 infeasible 候选更新 archive。

### H. 生成流程

query 仍然可以沿用你现在 `BuildBoundaryDataset_CBS.m` 里的两类来源：

- `missing_ref`
- `large_gap`

但**query 不再以 `QueryY` 作为条件**，而是以“哪个区间 + 区间内相对位置”作为条件。
也就是保留你现在“补缺口”的思路，删除“给定目标坐标”的逆映射味道。

### I. 如何保证“生成的是窄边界，不是厚点云”

主线保证来自四层同时收紧：

1. archive 本身只保留 certified thin segments；
2. 条件只描述局部 boundary context，不描述完整 `y`；
3. `L_side` 把生成点拉向正确边界侧；
4. 多 `z` 候选经过真实 objective-space interval score 再筛一次。

### J. 如何避免“只记忆已有点”

- query 只发往 `missing_ref` / `large_gap` 区间，不发往训练点本身；
- 条件是区间上下文，不是训练点 ID；
- `z` 允许同一上下文出多个候选；
- 用 holdout-ref 评估，不只看 train reconstruction。

### K. 评价指标

在当前 2 目标、归一化 objective space 下，我建议只保留下面几项作为主指标：

- `d_seg90`：90% 生成点评到目标 interval 的距离。
  通过：`<=0.01`；失败：`>0.02`
- `width90(q)`：区间 `q` 上生成点相对局部法向的 90% 厚度。
  `width_ratio90 = median_q width90(q) / |I_q|`
  强通过：`<=0.05`；可接受：`0.05~0.10`；失败：`>0.10`
- `two_side_query_rate`：对每个 query，用 `K` 个候选看是否同时出现近边 feasible 与近边 infeasible。
  通过：`>=0.2`
- `novel_interval_cover`：heldout/missing interval 中，被至少一个近边生成点覆盖的区间比例。
  通过：`>=0.3`
- `memorization_rate`：生成点中，最近训练边界点距离 `< ε_mem` 的比例。
  单看没意义；必须和 `novel_interval_cover` 联合看。
  目标是“memorization 不高，同时 novel cover 不低”。

现有的 `feasible_rate`、`ref_cover` 可以保留，但只能当副指标。
现有 `side_rate` 应改名为 `decision_side_rate`，降级使用。
现有 `segment_width90` 如果保留名字，也必须改成“局部法向厚度”，不能再是 source interval 宽度。

## 5. 最小实验矩阵

只做 5 个。每个实验都回答一个核心问题。

### 实验 1：训练集本身是否是主因

- 固定什么：
  固定一个 snapshot：
  `experiments/all_non_png/Data/CBS_CGAN/single_stage_overfit_endpoint_yb_norm_20260624_203803/LIRCMOP7_BC_run1_targetFE050000_snapshot.mat`
  固定网络结构、学习率、epoch=200、`z~N(0,I)`。
- 改什么：
  当前 `BuildBoundaryDataset_CBS.m` 数据集
  vs
  新的 certified-segment 数据集。
- 看什么图片：
  输出与
  `.../single_stage_overfit_endpoint_yb_norm_20260624_203803/LIRCMOP7_BC_run1_targetFE050000_overfit_reconstruction.png`
  同风格的 overfit reconstruction 图。
- 看什么 CSV/MAT：
  `single_stage_overfit_metrics.csv`；
  `stage_trainset_profile.csv`；
  snapshot MAT。
- 假设成立的结果：
  新数据集在 200 epoch 内把 `train_y_rec90_norm` 压到 `<=0.02`，且 detached branch 明显消失。
- 假设不成立的结果：
  新数据集仍然学不回去，那才轮到怀疑网络/损失本身。

### 实验 2：GAN vs CGAN，到底谁更符合你的核心目标

- 固定什么：
  固定清洗后的 certified-segment archive；固定同一网络和损失。
- 改什么：
  1. 传统 GAN
  2. 现在这种 `ref_y` / exact-y CGAN
  3. 推荐的 local-context CGAN
- 看什么图片：
  held-out ref reconstruction / missing-interval generation 图。
- 看什么 CSV/MAT：
  直接用现有
  `source/current_worktree/PlatEMO/Algorithms/Multi-objective optimization/CBS-CGAN/Support/run_CBS_CGAN_holdout_ref_eval.m`
  和
  `.../Support/run_CBS_CGAN_non_cgan_baseline_eval.m`
  产出 holdout 指标。
- 假设成立的结果：
  local-context CGAN 的 `novel_interval_cover` 最高，`d_seg90` 与 `width_ratio90` 最好；
  `ref_y` 在 seen-point reconstruction 上也许更好，但 holdout/missing-interval 更差。
- 假设不成立的结果：
  若 plain GAN 与 local-context CGAN 持平甚至更好，说明 condition 还没定义干净。

### 实验 3：问题到底在 adversarial 不足，还是 Huber 帮错了方向

- 固定什么：
  固定清洗后的 local-context CGAN。
- 改什么：
  1. `adv only`
  2. `adv + permanent x-Huber`
  3. `adv + side contrastive + warm-start Huber`
- 看什么图片：
  用
  `source/current_worktree/PlatEMO/Algorithms/Multi-objective optimization/CBS-CGAN/Support/run_CBS_CGAN_single_stage_overfit_diagnostic.m`
  先做 offline；再做一个短 online 图。
- 看什么 CSV/MAT：
  `single_stage_overfit_metrics.csv`；
  `stage_metrics_all.csv`。
- 假设成立的结果：
  只有第 3 组能同时去掉 detached branches，并把 `d_seg90` 和 `width_ratio90` 压下来。
- 假设不成立的结果：
  若 `adv only` 已足够好，说明前面的 archive/condition 还没完全对准。

### 实验 4：`z=0` 是否导致生成退化

- 固定什么：
  固定清洗后的 local-context CGAN 和第 3 实验中最优损失。
- 改什么：
  1. `z=0, K=1`
  2. `z~N(0,I), K=4`
  3. `z~N(0,I), K=8`
- 看什么图片：
  missing-interval generation 图。
- 看什么 CSV/MAT：
  `novel_interval_cover`、`two_side_query_rate`、`memorization_rate`。
- 假设成立的结果：
  random-z 能明显提高 novel cover 和 two-side rate，而 width 不恶化。
- 假设不成立的结果：
  如果 random-z 基本没提升，说明 condition 仍过强或 query interval 仍定义错了。

### 实验 5：在线主线是否真的变好了

- 固定什么：
  只在 LIRCMOP6/7/8 上跑；先只看 `targetFE = 30000, 50000, 100000`；`runs=1` 即可。
- 改什么：
  当前主线 `D_ref_tau_adv_huber_epoch50`
  vs
  重构后的单一主线。
- 看什么图片：
  与
  `experiments/selected_png/Data/CBS_CGAN/train_quality_epoch_sweep_D_default_20260624_130858/contact_sheets_epoch/*.png`
  同风格的 contact sheet。
- 看什么 CSV/MAT：
  用现有
  `source/current_worktree/PlatEMO/Algorithms/Multi-objective optimization/CBS-CGAN/Support/run_CBS_CGAN_train_quality_sweep_lir_online.m`
  输出 `stage_metrics_all.csv` 和 `run_summary.csv`。
- 假设成立的结果：
  至少 3/4 的关键 targetFE 上，`d_seg90`、`width_ratio90` 更好，且 PNG 中不再出现明显 detached branch / thick cloud。
- 假设不成立的结果：
  若 single-stage 已经学好但 online 仍坏，那问题就转到 archive 更新节奏与 query 选择，而不是生成器本体。

## 6. 优先修改的源码位置

- `source/current_worktree/PlatEMO/Algorithms/Multi-objective optimization/CBS-CGAN/UpdateBoundaryMemory_CBS.m`
  改什么：去掉 `firstFrontMask`、去掉 `filterParetoMainLayer`、archive 单元改成 certified segment、配对半径收缩到同 ref/相邻 ref。
  为什么：这是当前“边界≠PF”冲突的源头。
  如何验证：`stage_trainset_profile.csv` 里 objective pair gap 仍小，但 coverage/连续性提升，PNG 少掉假 interval 分叉。
- `.../BuildBoundaryDataset_CBS.m`
  改什么：删除 `ref_y/ref_y_tau` 主线分支；把条件改成 boundary context；query 改成 interval context；边界顺序改成几何链，不再按 ref-SVD 排。
  为什么：这是当前 condition drift 和 fake interval 的源头。
  如何验证：`condition_unique_count/train_count` 显著上升；不再出现 endpoint 模式 `tau` 全 0 的退化，因为 endpoint 分支被删。
- `.../BoundaryCGAN_CBS.m`
  改什么：默认 `random z`；保留 adversarial；加入 `L_side`；把 Huber 改成 warm-start。
  为什么：当前 zero-z + x-Huber 在 condition collision 下必然抹厚。
  如何验证：single-stage overfit 上 `train_y_rec90_norm` 显著下降，且 detached branch 消失。
- `.../CBS_CGAN.m`
  改什么：每个 query context 采样 `K` 个 candidate；真实评价后做 objective-space boundary score 选择；新增 `d_seg90/width_ratio90/novel_interval_cover/two_side_query_rate` 日志。
  为什么：当前 online 主线没有真正把“窄边界”当最终筛选目标。
  如何验证：online contact sheets 和 `stage_metrics_all.csv` 一起变好。
- `.../PairLocalTau_CBS.m`
  改什么：主线废弃为条件变量，改造成 boundary arc/interval 辅助函数，或者新建 `BoundaryArcCoord_CBS.m`。
  为什么：当前 `tau` 只是 pair chord 坐标，endpoint 一开就全零。
  如何验证：主线不再依赖“会塌掉的 tau”。
- `.../Support/run_CBS_CGAN_single_stage_overfit_diagnostic.m`、
  `.../Support/run_CBS_CGAN_holdout_ref_eval.m`、
  `.../Support/run_CBS_CGAN_non_cgan_baseline_eval.m`、
  `.../Support/run_CBS_CGAN_train_quality_sweep_lir_online.m`
  改什么：把它们变成正式诊断流水线，而不是临时脚本。
  为什么：你现在真正缺的是“最小可证伪实验”，不是更多花式 sweep。
  如何验证：每次改主线先过 single-stage，再过 holdout，再过 online 短 sweep。

## 7. 最终建议

- **当前最大问题**：不是 GAN/CGAN 本身“不适合”，也不是网络太小；而是 **boundary archive、condition、loss、metric 四层同时没有对准“目标空间可行/不可行窄边界”**。其中第一优先级是 archive/trainset 定义错误，第二优先级是 condition 语义塌缩并被 `z=0` 放大。
- **第一优先级该改什么**：先重写
  `UpdateBoundaryMemory_CBS.m` 和 `BuildBoundaryDataset_CBS.m`。
  不把 archive 清干净，后面换多少 GAN/CGAN 细节都只是换壳。
- **第二优先级该验证什么**：先做 single-stage 可学习性和 holdout-ref 可泛化性；只有这两关过了，再回到在线算法。你现有
  `single_stage_overfit_*`、`holdout_ref_eval`、`non_cgan_baseline_eval`
  三套脚本已经足够支撑这一步。
- **不应该继续做什么**：
  1. 不要继续在当前混合语义上堆 `pair=6 / endpoint / endpoint+y_b_norm / epoch100` 这种分支；
  2. 不要再把 `ref_y` 当主线；
  3. 不要再默认 `z=0`；
  4. 不要再用 `feasible_rate`、`ref_cover`、当前 `side_rate` 来代替“窄边界”判断。

# 第二个问题

我认同你的分析，但是我还有以下要求需要你在你提出的算法的思想上改造
1:本算法要求统一收敛，不要对于不一定有作用或者不一定有价值的参数、模块不要过度设计（比如边界 archive 结构设计过于复杂？都是必要信息吗？请你重构为适合我们主线的精简版本！）
2:损失函数是非常核心的内容，你要重新慎重思考，慎重设计（不确定的因素不要添加，要契合主线的统一收敛）
3:“生成器对每个 query context 采样 K=4~8 个候选 x_hat；用真实 Problem.Evaluation 或 EvaluateDecisions_CBS.m 评价”这一步是否会额外产生大量的真实评估次数？每一次真实评估次数都是十分珍贵的，不能只用于后处理来浪费！
4:我们的核心是在通过GAN/CGAN学习一个分布，要学习的分布是目标空间的边界，所以你要分清楚主线是什么？中心是什么？不要乱加参数和模块，要精简主线，把思考核心放在我们的主线上；
5:指标和实验需要关注主线和核心创新点，尤其是当前难点，确保稳扎稳打的验证；
你重新给出完整的且仅给出：重新思考且重构后的“算法思想”+“代码实现方式（基于A_ref_only_adv分支算法如何修改）”，注意：优先做统一、减法和收敛，而不是分支、加法和堆料；

# 第二个问题的回答

## 算法思想

我建议把当前主线收缩成一个**唯一语义、唯一条件、唯一损失组合**的版本：**Ref-only Certified Boundary CGAN**。它本质上是把现在的 `A_ref_only_adv` 分支清洗成真正能收敛的主线，而不是再往 `ref_tau / ref_y / endpoint / y_b_norm` 那些分支上继续加东西。

先说我为什么这样收缩。

当前 `A_ref_only_adv` 的纯 adversarial 训练本身不够，LIRCMOP7_BC、FE=50000 时 `train_y_rec90=1.180998`、`boundary_dist90=0.020033`，对应图里红点明显脱离橙色训练边界（证据：`experiments/all_non_png/Data/CBS_CGAN/train_quality_baseline_ABCD_LIR6_run1_20260624_024446/train_quality_sweep_stage_metrics.csv`；`experiments/selected_png/Data/CBS_CGAN/A_ref_only_adv_LIR6_run1_nGen30_20260623_220259/flat_train_reconstruction/LIRCMOP7_BC_run1_targetFE050000_train_reconstruction.png`）。单 stage 过拟合里，`adv_only_epoch500_no_huber` 也只有 `train_y_rec90_norm=0.059057`，说明“只靠 GAN 对抗项自己贴边”在你现在这个任务上不成立（证据：`experiments/all_non_png/Data/CBS_CGAN/single_stage_overfit_adv_only_no_huber_20260624_210752/single_stage_overfit_metrics.csv`）。

但这不等于要去加 `y_b_norm`、endpoint 分支、pair tau 分支。当前 `ref_y`/endpoint 这条线在 LIRCMOP7_BC、FE=50000 的 snapshot 上，`train_count=42` 只有 `condition_unique_count=16`，`condition_duplicate_count=26`，而且 `train_tau_range=0`、`train_tau_nonzero_rate=0`；同一 snapshot 下 `online_like_adv_huber_epoch50` 的 `train_y_rec90_norm=0.167571`，远差于 `supervised_huber_epoch200=0.017130`（证据：`experiments/all_non_png/Data/CBS_CGAN/single_stage_overfit_endpoint_yb_norm_20260624_203803/stage_trainset_profile.csv`；`.../single_stage_overfit_metrics.csv`）。这说明现在的问题不是“Huber 方向错”，而是**训练样本和条件先天一对多、语义混乱**。`BuildBoundaryDataset_CBS.m` 里现在确实同时保留了 `ref_only / ref_tau / ref_y / ref_y_tau` 四套条件（`source/current_worktree/PlatEMO/Algorithms/Multi-objective optimization/CBS-CGAN/BuildBoundaryDataset_CBS.m:84-96`），而 `BoundaryCGAN_CBS.m` 文件头还直接写着 “inverse mapping”（`.../BoundaryCGAN_CBS.m:1-7`）。这条线越做越容易滑向 `y -> x`，不应该再当主线。

再看 archive。`UpdateBoundaryMemory_CBS.m` 现在先把 feasible 端压到 `firstFrontMask` 主层（`.../UpdateBoundaryMemory_CBS.m:100-105`），然后再经过 `filterParetoMainLayer`（`.../UpdateBoundaryMemory_CBS.m:278-289`）。这会把“目标空间可行/不可行边界”继续压成“PF 主层附近的边界 support”。这和你的核心目标不一致。

所以我建议把主线彻底收缩成下面这一个算法思想：

**1）边界 archive 只保存“每个 ref 的一条最薄 certified 边界段”，不再保存混合语义的 `x_b / y_b / tau / endpoint / y_b_norm`。**

每个参考向量 `ref=r` 只保留一条记录：

- `x_f^r, y_f^r`：该 ref 上当前最靠近边界的 feasible 端点
- `x_i^r, y_i^r`：与之配对的 infeasible 端点
- `gap^r = || normalize(y_f^r) - normalize(y_i^r) ||_2`：这条 pair 的目标空间薄度证书
- `ref=r`

就这四类核心信息。`x_b / y_b` 不再单独有自己的语义；为了兼容当前代码，直接令 `x_b = x_f`、`y_b = y_f` 即可。`tau` 在主线里废弃，不再承担任何语义。

**2）条件只保留一个：`ref_only`。**

条件就是参考向量 `W(r,:)`，不是 `tau`，不是 `y_b_norm`，不是 endpoint 坐标。这样条件表达的是“边界方向上下文”，不是“给定目标坐标找决策变量”。这正好保留了你的核心创新点：**通过已探索的边界片段，学习当前边界分布，然后对未覆盖 ref 直接生成新的边界解**。

当前 `D_ref_tau_adv_huber` 比 `B_ref_only_adv_huber` 好，不是因为 `tau` 本身语义更对，而是因为当前 archive 允许一 ref 多样本，`tau` 在给脏数据做被动消歧；一旦 archive 改成“1 ref 1 sample”，这个消歧需求就不存在了。反过来，endpoint 版本下 `tau` 已经直接塌成全 0（`train_tau_range=0`，证据见上面的 endpoint CSV），所以 `tau` 不适合做主线条件。

**3）训练目标只用 `x_f`，不再用“最近 support 点”或其他混合 target。**

你当前 `UpdateBoundaryMemory_CBS.m:140-151` 里会通过 `selectThinBoundaryTarget(...)` 在 feasible endpoint 和 near-segment support 之间切换，再把结果写进 `x_b / y_b`。这正是 target 语义漂移的源头（证据：`source/current_worktree/PlatEMO/Algorithms/Multi-objective optimization/CBS-CGAN/UpdateBoundaryMemory_CBS.m:140-189`）。

主线里不再做这个切换。因为在 binary feasibility 场景下，真正可靠、可监督、且最靠近边界的实测样本，就是 certified pair 的 feasible 端点 `x_f`。你要学的是“边界附近的 feasible 决策分布”，而不是“某个插值支撑点”。

**4）CGAN 仍保留，但改成确定性 `G(ref) -> x`。**

不引入 random z，不做 `K=4~8` 候选再真实评估筛选。你自己已经明确说每次真实评估都很珍贵，我同意，所以主线必须做到：**一个 query condition，只生成一个 X，只做一次真实评估。**

这里固定 `z=0` 不是问题，因为当前真正的问题不是 zero-z 本身，而是“脏 archive + 一对多条件”。在“1 ref 1 sample”的 clean archive 下，`ref_only` 已经是近似一一对应条件，固定 `z=0` 反而更符合你要的“窄边界线”，而不是厚点云。`CBS_CGAN.m` 当前采样后只做一次真实评价并直接并入种群（`source/current_worktree/PlatEMO/Algorithms/Multi-objective optimization/CBS-CGAN/CBS_CGAN.m:197-208,273-277`），这个主流程可以保留；只是不要再增加任何“多候选后处理评估”。

**5）损失函数只保留两项：`adversarial BCE + anchored Huber`。**

不要再加 objective-side penalty、normal penalty、pair side loss、projection loss 这些当前都没有充分证据必须存在的项。你的 FE 预算和当前训练现状都不支持把损失堆复杂。

主线损失就用：

[
L_D = \text{BCE}(D(x_f, r), 1) + \text{BCE}(D(G(r), r), 0)
]

[
L_G = \text{BCE}(D(G(r), r), 1) + \lambda \cdot \text{Huber}(G(r), x_f)
]

这里 `r` 就是参考向量条件。`λ` 先固定为 1，不再做 sweep。理由很直接：

- `A_ref_only_adv` 和 adv-only overfit 已证明纯 adversarial 不够（上面两组 CSV/PNG）。
- 当前 `BoundaryCGAN_CBS.m` 里实际上已经有这两个损失块，不需要新发明，只是目前 Huber 对着混乱的 `TrainX=x_b`，才显得语义不稳（证据：`source/current_worktree/PlatEMO/Algorithms/Multi-objective optimization/CBS-CGAN/BoundaryCGAN_CBS.m:289-301`）。
- 一旦 `TrainX` 改成 `x_f`，而且 archive 强制一 ref 一样本，Huber 的含义就变成“把 ref->boundary-x 这条确定性边界流形先锚住”，这和你的主线是对齐的，不再是“把厚点云往平均值上抹”。

**6）query 只做 missing-ref 插值，不做外推，不做多分支。**

当前 `BuildBoundaryDataset_CBS.m` 的 `buildExternalQueries` 同时做 `missing_ref` 和 `large_gap`（`.../BuildBoundaryDataset_CBS.m:181-223`）。主线我建议只保留 `missing_ref`：

- 已占据 ref 之间的空 ref，才是“当前已知边界片段之间的未探索边界”；
- 这正好是你的核心创新点；
- 它天然是插值，不是外推，更稳。

如果某一代没有 missing ref，就这一代不触发 CGAN 生成，继续普通协同进化。不要为了“每代都用 GAN”去造 query。

整体上，这个重构版其实很简单：**用最薄 pair 证书把 archive 清干净；用 `ref` 做唯一条件；用 `x_f` 做唯一监督目标；用 `adv + Huber` 做唯一损失；用 missing refs 做唯一 query 来源；每个 query 只评估一次。**

------

## 代码实现方式（基于 A_ref_only_adv 分支）

下面按“尽量少改文件、尽量复用现有结构”的方式给出落地改法。

### 1. `CBS_CGAN.m`：把主线默认值改回 ref-only，并冻结成唯一主线

当前默认 `DatasetOptions.conditionMode` 是 `"ref_tau"`（`source/current_worktree/PlatEMO/Algorithms/Multi-objective optimization/CBS-CGAN/CBS_CGAN.m:57-64`），而 `pairNeighborRefRadius` 默认是 4（`.../CBS_CGAN.m:17,27-33`）。这里直接改成主线常量：

```matlab
% old
DatasetOptions.conditionMode = cbsConditionModeFromControl("ref_tau");

% new
DatasetOptions.conditionMode = "ref_only";
% old default
pairNeighborRefRadius = 4;
maxCandidatePairsPerRef = 1; % already 1 in main parameter set

% new default
pairNeighborRefRadius = 1;
maxCandidatePairsPerRef = 1;
queryPerCondition = 1;
```

同时把主线采样固定成 zero-z，不再给 random-z 开入口。当前 `applyCBSSampleControlOptions` 已支持 `"zero"`（`.../CBS_CGAN.m:388-409`），直接沿用；不要再增加 K 候选评估逻辑。也就是：

- `trainZMode = "zero"`
- `sampleZMode = "zero"`
- `queryPerCondition = 1`

`GANOptions.reconstructionWeight` 当前默认已经是 1（`.../CBS_CGAN.m:75-77`），这里保留；真正要改的是上游 dataset 语义，不是这里再去堆新 loss。

此外，把文件头注释里的主线说明改掉。当前第 4 行写的是 “thin-boundary memory with adversarial + Huber CGAN, ganIter=50”，但实际 condition 默认还是 `ref_tau`，语义不统一（`.../CBS_CGAN.m:3-4,61-64`）。改成：

```matlab
% Mainline: ref-only certified-boundary CGAN, TrainX = feasible boundary endpoints.
```

### 2. `UpdateBoundaryMemory_CBS.m`：把 archive 改成“1 ref 1 certified pair”

这是最关键的改动。

#### 2.1 去掉 PF 主层过滤

当前 feasible 端先走 `firstFrontMask`（`.../UpdateBoundaryMemory_CBS.m:100-105`），这一步删掉：

```matlab
% old
mainFeasible = false(size(Feasible));
fAll = find(Feasible);
if ~isempty(fAll)
    front = firstFrontMask(Y(fAll,:));
    mainFeasible(fAll(front)) = true;
end

% new
mainFeasible = Feasible;
```

这样 archive 表达的是“可行/不可行边界”，不是“PF 主层边界”。

#### 2.2 配对半径固定成 1，只保留每 ref 最小 gap pair

当前配对逻辑本身可以保留框架（`.../UpdateBoundaryMemory_CBS.m:108-159`），但条件固定：

- `pairNeighborRefRadius = 1`
- `maxCandidatePairsPerRef = 1`

也就是每个 ref 最多留一条最薄 pair，不再做 `pair=3/6` 这种分支。你自己的对比里，`pair6` 在 LIRCMOP7_BC、FE=50000 反而把 `train_y_rec90` 从 `1.046732` 拉到 `1.245837`，`query_obj_dist90` 从 `0.016824` 拉到 `0.02197`（证据：`experiments/all_non_png/Data/CBS_CGAN/train_quality_epoch_sweep_D_default_20260624_130858/train_quality_sweep_stage_metrics.csv`；`experiments/all_non_png/Data/CBS_CGAN/train_quality_pair6_vs_pair3_D_default_epoch50_20260624_153705/train_quality_sweep_stage_metrics.csv`）。

#### 2.3 删掉 `selectThinBoundaryTarget(...)`，主线 target 固定为 feasible endpoint

当前第 140-151 行会写入 `x_b / y_b / tau`，而 `x_b / y_b` 来自 `selectThinBoundaryTarget(...)`（`.../UpdateBoundaryMemory_CBS.m:140-189`）。主线里直接改成：

```matlab
Candidate.ref(end+1,1) = r;
Candidate.gap(end+1,1) = gapSorted(t);

Candidate.x_f(end+1,:) = X(f,:);
Candidate.y_f(end+1,:) = Y(f,:);
Candidate.x_i(end+1,:) = X(i,:);
Candidate.y_i(end+1,:) = Y(i,:);

% compatibility fields
Candidate.x_b(end+1,:) = X(f,:);
Candidate.y_b(end+1,:) = Y(f,:);
Candidate.tau(end+1,1) = 0;
```

也就是说：

- `x_b = x_f`
- `y_b = y_f`
- `tau = 0` 仅为兼容旧字段，主线不用它

这样就彻底消除了 endpoint / support target 的分支漂移。

#### 2.4 去掉 `filterParetoMainLayer`

当前 `filterBoundaryCandidates()` 里先 `isfinite(gap)`，然后 `filterParetoMainLayer`，再 `filterGapCap`（`.../UpdateBoundaryMemory_CBS.m:261-275`）。主线改成只保留：

```matlab
Candidate = subsetMemory(Candidate,isfinite(Candidate.gap));
Candidate = filterGapCap(Candidate,Options);
```

`filterParetoMainLayer` 整段（`.../UpdateBoundaryMemory_CBS.m:278-289`）从主线删除。

#### 2.5 `sortedBoundaryNodes` 简化成按 `ref` 排序

当前 `sortedBoundaryNodes` 用 ref 点做 SVD 排序（`.../UpdateBoundaryMemory_CBS.m:388-401`），主线没必要。改成：

```matlab
function idx = sortedBoundaryNodes(BMem,~)
    [~,ord] = sort(BMem.ref(:),'ascend');
    idx = ord(:);
end
```

因为主线 query 只做 missing-ref，排序只需要 ref 顺序。

### 3. `BuildBoundaryDataset_CBS.m`：只保留 ref-only dataset，TrainX 直接用 `x_f`

这是第二个关键文件。

#### 3.1 主线只保留 `ref_only`

当前 `conditionModeFromOptions()` 和 `referenceConditionsFromRefsTau()` 同时支持四种模式（`.../BuildBoundaryDataset_CBS.m:69-103,159-179`）。主线里直接收缩到一种：

```matlab
function mode = conditionModeFromOptions(~)
    mode = "ref_only";
end
```

`referenceConditionsFromRefsTau()` 也只保留：

```matlab
C = double(W(refs,:));
```

不再拼 `tau`，不再拼 `normalizeObjectiveCondition(...)`，也就不再有 `ref_y / ref_y_tau`。

#### 3.2 TrainX 改成 `BMem.x_f`

当前第 32-39 行会先算 `TrainTau`，然后 `TrainX = BMem.x_b(validTrain,:)`（`.../BuildBoundaryDataset_CBS.m:32-39`）。主线改成：

```matlab
validTrain = all(isfinite(BMem.x_f),2) & all(isfinite(BMem.x_i),2) & ...
             all(isfinite(BMem.y_f),2) & all(isfinite(BMem.y_i),2) & ...
             isfinite(BMem.gap(:));

TrainX = BMem.x_f(validTrain,:);
TrainC = double(W(BMem.ref(validTrain),:));
```

这样 `TrainX` 的语义就是唯一的：**当前已证实的 near-boundary feasible decision**。

同时 `Info.trainXf/trainXi/trainYf/trainYi/trainRef` 这些诊断字段可以保留；`Info.trainTau` 统一置零或删除。

#### 3.3 query 只保留 `missing_ref`

当前 `buildExternalQueries()` 同时做 `missing_ref` 和 `large_gap`（`.../BuildBoundaryDataset_CBS.m:181-223`）。主线只保留前半段 missing-ref 逻辑，删掉 208-222 行那段 `large_gap`。

最简单的实现就是：

```matlab
usedRefs = unique(BMem.ref(:)');
allRefs  = min(usedRefs):max(usedRefs);
missingRefs = setdiff(allRefs,usedRefs,'stable');

if numel(missingRefs) > budget
    missingRefs = missingRefs(1:budget);
end

QueryC = double(W(missingRefs,:));
QueryY = zeros(numel(missingRefs),Problem.M);   % diagnostics only
QueryMeta.ref = missingRefs(:);
QueryMeta.source_type = repmat("missing_ref",numel(missingRefs),1);
```

这样 query condition 就是“未覆盖参考向量”，而不是 query objective。完全不需要 `PairLocalTau_CBS.m`。

如果 `missingRefs` 为空，这一代就不触发 CGAN 生成，继续普通进化。主线先只做插值式边界补全，不做外推。

### 4. `BoundaryCGAN_CBS.m`：损失结构不加新项，只保留现有 `adv + Huber`

这个文件反而不需要大改。原因是你要的最小损失结构，其实代码已经有了。

当前 `generatorGradients(...)` 里已经是：

- 对抗项 `advLoss`（289-297 行）
- Huber 重构项 `recLoss`（298-301 行）

见：`source/current_worktree/PlatEMO/Algorithms/Multi-objective optimization/CBS-CGAN/BoundaryCGAN_CBS.m:289-301`。

主线这里不要再加新 loss。真正要改的是两点：

#### 4.1 注释改掉“inverse mapping”

把文件头第 2 行：

```matlab
% Minimal conditional GAN for boundary point inverse mapping.
```

改成：

```matlab
% Ref-conditioned CGAN for direct generation of certified boundary decisions.
```

#### 4.2 继续用 zero-z，不引入 random-z 和多候选评估

当前 `fillOptions()` 默认支持 `trainZMode/sampleZMode = zero`（`.../BoundaryCGAN_CBS.m:319-345`），`trainingLatentSamples()` 和 `latentSamplesForCondition()` 也已经支持 zero 模式（`.../BoundaryCGAN_CBS.m:188-224`）。主线就沿用，不要再改成 random-z，更不要引入 “一个 condition 采 K 个样本再真实评估选一个” 的逻辑。

你担心 FE 浪费，这里正好可以避免：**一条 query condition -> 一次 forward -> 一个 X -> 一次真实评估**。

也就是说，这个文件的主线改法不是“加更多 loss”，而是“把它喂干净的 `TrainX=x_f, TrainC=ref`”。

### 5. `Support/run_CBS_CGAN_*`：停止 A/B/C/D 扫描，把支持脚本收敛成单一主线

当前 `run_CBS_CGAN_loss_ablation_lir_online.m` 里 A/B/C 的 variant 表是：

- `A_ref_only_adv`
- `B_ref_only_adv_huber`
- `C_ref_tau_adv`

见：`source/current_worktree/PlatEMO/Algorithms/Multi-objective optimization/CBS-CGAN/Support/run_CBS_CGAN_loss_ablation_lir_online.m:139-145`。

主线里这张表直接收缩成一行，例如：

```matlab
'variant',{"REF_BOUNDARY_MAINLINE"}, ...
'conditionMode',{"ref_only"}, ...
'advWeight',{1}, ...
'reconstructionWeight',{1}, ...
'trainZMode',{"zero"}, ...
'sampleZMode',{"zero"}
```

同时在运行参数里固定：

```matlab
pairNeighborRefRadius = 1;
maxCandidatePairsPerRef = 1;
queryPerCondition = 1;
```

`run_CBS_CGAN_train_quality_sweep_lir_online.m`、`run_CBS_CGAN_single_stage_overfit_diagnostic.m` 也同步收成这一条主线。也就是：

- 不再扫 `ref_tau`
- 不再扫 `ref_y`
- 不再扫 `endpoint`
- 不再扫 `endpoint_yb_norm`
- 不再扫 `pair=3/6`

这些分支现在只保留为历史对照，不再作为主线脚本继续扩展。

### 6. 这套改法为什么不会额外浪费真实评估

这个问题你提得对，所以代码上必须明确做到两件事：

第一，不引入 `K` 候选后选。
第二，生成出来的点不是“只用于后处理”，而是直接进入当前进化与下一轮边界记忆。

当前 `CBS_CGAN.m` 已经是这个主流程：生成后 `Problem.Evaluation(RawDec)` 一次，然后 `OffspringG` 会并入 `UnionPopulation` 做环境选择（`source/current_worktree/PlatEMO/Algorithms/Multi-objective optimization/CBS-CGAN/CBS_CGAN.m:207-208,273-277`）。这个主流程保留即可。新的生成点无论可行还是不可行，都不是浪费：

- 可行点可直接竞争进入 `Population1`
- 不可行点也会进入下一轮 `UpdateBoundaryMemory_CBS.m` 的样本池，成为新的 `x_i / y_i` 候选

所以这里没有“额外 FE 只用于后处理”的问题。

------

如果你按上面这套改，主线实际上会变得非常干净：

- **条件只有 ref**
- **target 只有 `x_f`**
- **archive 只有一 ref 一 pair**
- **query 只有 missing_ref**
- **loss 只有 adv + Huber**
- **每个 query 只评估一次**
- **不再引入 `y_b_norm`、tau、endpoint/support 分支、pair 数扫描、多候选评估**

这才是和你当前证据最一致、也最符合“统一、减法、收敛”的版本。





# 第三个问题

不对，我要的统一，精简的主线，不是让你无条件的删除，无条件的简化，我要求的是能实现我的要求，能解决的当前问题的最小，最精简的版本；而不是你当前这个只为了精简而精简却解决不了问题的版本；你不要受到（A_ref_only_adv）的约束；请你参考你的第一次回答，重新给出能实现我的要求，能解决的当前问题的最小，最精简的版本



# 第三个问题的回答

你说得对。上一版把主线压成 `ref_only`，确实过度了。结合当前源码、CSV 和过拟合诊断，我认为**最小、统一、且真正对准当前失败机制**的版本，不是 A/B/C/D 里的任何一个现成分支，而是：

**“thin boundary pair memory + component/ref/arc 条件 CGAN + Huber 预热后再对抗微调”**

它保留你要的核心：

- 仍然是 GAN/CGAN；
- 仍然直接生成完整决策变量 `X`；
- 学的是**目标空间可行/不可行边界分布**；
- 用的是当前已探索到的部分边界；
- 生成的是**未探索边界段**；
- 不额外浪费 FE，不做多候选后筛，不做 surrogate/repair/local search。

下面我只给这一个主线。

------

## 一、重构后的最小主线算法思想

### 1）边界 archive 的唯一对象：**thin boundary pair**，不是混合语义的 `x_b/y_b`

当前主线真正稳定、可信的边界信息，其实不是 `x_b/y_b`，而是已经成对保存的 `(x_f,y_f,x_i,y_i,gap,ref)`。源码里这一对字段已经完整存在，`UpdateBoundaryMemory_CBS.m` 在追加 candidate 时就保存了 `x_f,y_f,x_i,y_i,gap,ref`（`source/current_worktree/PlatEMO/Algorithms/Multi-objective optimization/CBS-CGAN/UpdateBoundaryMemory_CBS.m:143-151`）。同时，单 stage profile 里这类 pair 的目标空间厚度已经很薄：
`single_stage_overfit_endpoint_yb_norm_20260624_203803/stage_trainset_profile.csv` 中 `train_pair_obj_dist90_norm=0.001941`；
`single_stage_overfit_adv_only_no_huber_20260624_210752/stage_trainset_profile.csv` 中 `train_pair_obj_dist90_norm=0.000943`。
这说明 pair 本身已经足够“贴边”，主问题不在 pair 太厚，而在后面把它转成训练目标和条件时语义变脏了。

相反，当前 `x_b/y_b` 会在 `selectThinBoundaryTarget(...)` 里在 feasible endpoint 和 near-segment support 之间切换（`UpdateBoundaryMemory_CBS.m:162-189`），这正是训练目标语义漂移的来源。再加上前面先把 feasible 端压到 `firstFrontMask`（`UpdateBoundaryMemory_CBS.m:100-105`），后面又做 `filterParetoMainLayer`（`UpdateBoundaryMemory_CBS.m:261-289`），archive 被不断往“PF 主层 support”上扭，不再是干净的约束边界。

所以最小主线应当是：

[
\mathcal{B}_j={r_j,x^f_j,y^f_j,x^i_j,y^i_j,g_j},
\quad
g_j=|\hat y^f_j-\hat y^i_j|_2
]

也就是每条记录只保留：

- `ref`
- `x_f, y_f`
- `x_i, y_i`
- `gap`

**不再把 `x_b/y_b/tau` 当主语义字段。**
如果为了兼容现有代码必须保留 `x_b/y_b`，那也只做 alias：`x_b = x_f`，`y_b = y_f`，不再允许第二套 target 语义。
证据路径：`source/current_worktree/.../UpdateBoundaryMemory_CBS.m:143-151,162-189,261-289`；`experiments/all_non_png/Data/CBS_CGAN/single_stage_overfit_*/*stage_trainset_profile.csv`。

------

### 2）条件不能是 `ref_only`，也不能是 `ref_y`；最小正确条件是 **`comp_ref_s`**

我修正上一版的关键点就在这里。

#### 为什么不能回到 `ref_only`

现有结果已经说明只给 ref 不够。
在 `train_quality_baseline_ABCD_LIR6_run1_20260624_024446/train_quality_sweep_stage_metrics.csv` 的 `LIRCMOP7_BC, target_FE=50000` 上：

- `A_ref_only_adv`：`train_y_rec90=1.180998`
- `B_ref_only_adv_huber`：`train_y_rec90=1.122742`
- `D_ref_tau_adv_huber`：`train_y_rec90=0.524007`

也就是说，哪怕加了 Huber，`ref_only` 仍明显弱于“ref + 一个位置标量”。
对应图也能看出 `A/B` 红点更容易离开橙色边界：
`experiments/selected_png/Data/CBS_CGAN/train_quality_baseline_ABCD_LIR6_run1_20260624_024446/contact_sheets_baseline/LIRCMOP7_BC_ABCD_contact_sheet.png`。

#### 为什么不能继续用当前 `tau`

当前 `tau` 不是“沿边界的位置”，而是**在单条 feasible–infeasible chord 上的投影**。
`PairLocalTau_CBS.m` 明确是把 `Y` 投影到 `(Yf,Yi)` 线段上（`source/current_worktree/.../PairLocalTau_CBS.m:25-37`）。
这意味着它表达的是“穿过边界厚度方向”的位置，不是“沿边界走向”的位置。

更糟的是，一旦切到 endpoint 模式，`y_b=y_f`，于是 `tau=0`，条件直接塌掉。
代码上这是 `UpdateBoundaryMemory_CBS.m:164-168` + `PairLocalTau_CBS.m:25-37` 的直接结果。
CSV 也完全验证了这一点：
`train_quality_endpoint_vs_default_D_epoch50_20260624_181601/train_quality_sweep_stage_metrics.csv` 和
`train_quality_endpoint_yb_norm_vs_default_D_epoch50_20260624_192053/train_quality_sweep_stage_metrics.csv`
里 `LIRCMOP7_BC` 各 target_FE 的 `train_tau_range=0`、`train_tau_nonzero_rate=0`。

所以 `ref_tau` 比 `ref_only` 好，并不意味着“tau 对了”，而只是说明**你确实需要一个位置标量**；只是这个标量不该是 current tau。

#### 为什么不能用 `ref_y`

`BuildBoundaryDataset_CBS.m` 里 `ref_y`/`ref_y_tau` 是直接把归一化 objective 拼进 condition（`source/current_worktree/.../BuildBoundaryDataset_CBS.m:84-92,105-125`）。
这会把问题带向 `y -> x`。
单 stage 过拟合里它确实更容易拟合训练点，但这正是因为它更接近逆映射，不是因为它更符合“学边界分布再补未探索边界”的主线。

更关键的是，它在线并不稳定。
`train_quality_endpoint_yb_norm_vs_default_D_epoch50_20260624_192053/train_quality_sweep_stage_metrics.csv` 的 `LIRCMOP7_BC, target_FE=50000` 上，`endpoint + y_b_norm` 已经恶化到：

- `train_y_rec90=3.504068`
- `boundary_dist90=0.045948`

对应图：
`experiments/selected_png/Data/CBS_CGAN/train_quality_endpoint_yb_norm_vs_default_D_epoch50_20260624_192053/contact_sheets_endpoint_yb_norm_compare/LIRCMOP7_BC_default_vs_endpoint_yb_norm_epoch50.png`
可直接看到红点分叉和错位更严重。

#### 最小正确条件：`comp_ref_s`

所以最小、正确、且不退化成逆映射的条件应该是：

[
c_j=[\mathrm{comp}_j,\ W(r_j,:),\ s_j]
]

其中：

- `comp_j`：边界连通分量标识；
- `W(r_j,:)`：当前 ref 上下文；
- `s_j`：**沿当前边界分量弧长的归一化位置**。

这里 `s` 才是你真正需要的那个“位置标量”。

它不是给定目标坐标；
它只表达“当前边界链上的相对位置”。
这正好服务你的核心目标：**从已有边界片段学边界分布，并补全边界链上未探索的空段。**

`comp` 和 `s` 不需要写进 archive，**只在建 dataset 时从 pair 派生**，这样 archive 仍然保持最小。
当前 `BuildBoundaryDataset_CBS.m` 用 ref-SVD 排序 `sortedBoundaryNodes(...)`（`BuildBoundaryDataset_CBS.m:285-299`），再按这个顺序造 `missing_ref`/`large_gap` query（`BuildBoundaryDataset_CBS.m:181-223`），这会把 ref-space 相邻误当成 boundary-space 相邻。
最小修正不是加一堆 KNN，而是：**用 pair 的 objective 中点去建边界链，再从链上算 `s`。**

具体做法：

[
m_j=\frac{y^f_j+y^i_j}{2}
]

对 `m_j` 做一次几何排序，得到每个 component 内的顺序；然后定义

[
s_1=0,\quad
s_j=\frac{\sum_{t<j}|m_{t+1}-m_t|*2}{\sum_t|m*{t+1}-m_t|_2}
]

`m_j` 只用于**上下文与 query**，不用于 condition 中的完整坐标输入，因此不会退化成 `y -> x`。

------

### 3）训练目标只用 `x_f`，因为它已经足够贴边，而且能直接用

这一点必须明确。

我上一版把目标压成 `ref_only + x_f`，错在 `ref_only`，不在 `x_f`。
`x_f` 反而是当前最干净的训练目标，因为：

- 它是已经评估过的、可直接注入主种群的 feasible 决策；
- 与 paired infeasible 组成的 gap 已经很薄，上面提到 `train_pair_obj_dist90_norm` 已到 `1e-3` 量级；
- 它比当前会漂移的 `x_b` 更稳定。

所以训练样本应当是：

[
\text{TrainX}=X_f,\quad \text{TrainC}=[\mathrm{comp},W(ref),s]
]

不是 `x_b`，也不是 near-segment support。

`x_i,y_i` 仍然保留，但它们的作用只剩两个：

1. 证明这个样本确实来自一条薄 boundary pair；
2. 用于构造 `m_j=(y_f+y_i)/2` 和 query interval。

------

### 4）损失函数最小改成两阶段：**先 Huber 锁边，再 adversarial 微调**

这一点是这次重构的核心，不需要再加第三、第四个 loss。

当前证据非常清楚：

- `single_stage_overfit_endpoint_yb_norm_20260624_203803/single_stage_overfit_metrics.csv` 里
  `online_like_adv_huber_epoch50` 的 `train_y_rec90_norm=0.167571`；
  同一 snapshot 下
  `supervised_huber_epoch200=0.017130`，
  `supervised_huber_epoch1000=0.012617`。
- `single_stage_overfit_adv_only_no_huber_20260624_210752/single_stage_overfit_metrics.csv` 里
  `adv_only_epoch500_no_huber` 也只有 `train_y_rec90_norm=0.059057`。
- 对应图
  `experiments/selected_png/Data/CBS_CGAN/single_stage_overfit_endpoint_yb_norm_20260624_203803/LIRCMOP7_BC_run1_targetFE050000_overfit_reconstruction.png`
  和
  `.../single_stage_overfit_adv_only_no_huber_20260624_210752/LIRCMOP7_BC_run1_targetFE050000_overfit_reconstruction.png`
  都能直接看到：从 scratch 的 adversarial 主导训练，连训练点都贴不稳；而纯 Huber 明显能贴回去。

另外，我这里也修正我上一版的一个过度判断：**condition duplicate 本身不是当前最小主线必须首先解决的问题。**
在 `single_stage_overfit_adv_only_no_huber_20260624_210752/stage_trainset_profile.csv` 里，虽然 `condition_duplicate_count=96`，但同时：

- `duplicate_condition_x_spread50=0`
- `duplicate_condition_y_spread50_norm=0`

也就是这些 duplicate 在这个 snapshot 里其实是 exact duplicates，不是一对多歧义。
所以当前最应该修的不是“再加复杂多样性分支”，而是**训练启动方式**。

因此，最小损失不是加新项，而是把当前已有的两项改成**单一固定训练日程**：

#### 阶段 A：Huber 预热

前半程只训生成器，判别器冻结：

[
L_G^{(A)}=\mathrm{Huber}(G(c),x_f)
]

#### 阶段 B：Huber + adversarial 联合微调

后半程恢复当前 conditional BCE：

[
L_D=\mathrm{BCE}(D(x_f,c),1)+\mathrm{BCE}(D(G(c),c),0)
]

[
L_G^{(B)}=\mathrm{Huber}(G(c),x_f)+\mathrm{BCE}(D(G(c),c),1)
]

关键点是：**不新增必须 sweep 的权重参数。**
直接沿用当前 `BoundaryCGAN_CBS.m` 里已有的 BCE + Huber 形式（`source/current_worktree/.../BoundaryCGAN_CBS.m:289-303`），只是把训练顺序从“从第 1 步就 joint”改成“先 Huber 锁住边界，再对抗微调”。
这样最小，而且和过拟合证据完全一致。

我不建议在这一版再加：

- pair-side margin；
- objective-space projection loss；
- normal/tangent penalty；
- diversity penalty。

这些都不是当前“最小且能解决问题”的必要项。

------

### 5）生成方式仍然是一条 query 只生成一个 `X`，不额外浪费 FE

这一点我完全收回上一版多候选筛选的建议。你这里 FE 很贵，主线不应该依赖多候选。

当前 `CBS_CGAN.m` 已经是：

- 对每个 `QueryC` 调 `samplebycondition`
- 得到 `RawDec`
- 直接 `Problem.Evaluation(RawDec)`
- 再并入 `UnionPopulation`

见 `source/current_worktree/.../CBS_CGAN.m:177-208,273-277`。

这个流程应当保留。
最小主线只改两件事：

1. `QueryC` 的语义改成 `comp_ref_s`；
2. query 只从**同一 component 内**的 `missing_interval` / `large_gap` 生成。

所以在线流程就是：

- boundary memory 刷新；
- 从 thin pairs 派生 boundary chain；
- 找同 component 内的空段；
- 每个空段给一个 `QueryC`；
- CGAN 生成一个 `X`；
- 做一次真实评价；
- 直接进当前两种群/下一轮 memory。

没有额外 FE 浪费。

------

## 二、代码实现方式

下面只给最小必要改动。

### 1. `UpdateBoundaryMemory_CBS.m`

#### 改动 1：去掉 PF 主层过滤

当前：

- `firstFrontMask`：`UpdateBoundaryMemory_CBS.m:100-105`
- `filterParetoMainLayer`：`UpdateBoundaryMemory_CBS.m:261-289`

这两块都要从主线拿掉。

直接改成：

- `mainFeasible = Feasible;`
- `filterBoundaryCandidates()` 只保留：
  - finite gap 过滤；
  - gap cap 过滤；
  - exact duplicate 去重。

#### 改动 2：archive 固定保存 thin pair，不再切 targetMode

当前 `selectThinBoundaryTarget(...)` 在 `UpdateBoundaryMemory_CBS.m:162-189` 会让 `x_b/y_b` 语义漂移。
主线改成固定：

```matlab
Candidate.ref(end+1,1) = r;
Candidate.gap(end+1,1) = gapSorted(t);

Candidate.x_f(end+1,:) = X(f,:);
Candidate.y_f(end+1,:) = Y(f,:);
Candidate.x_i(end+1,:) = X(i,:);
Candidate.y_i(end+1,:) = Y(i,:);

% compatibility only
Candidate.x_b(end+1,:) = X(f,:);
Candidate.y_b(end+1,:) = Y(f,:);
Candidate.tau(end+1,1) = 0;
```

也就是：

- `x_b/y_b` 仅做兼容字段；
- 主语义只剩 `x_f/y_f/x_i/y_i/gap/ref`；
- `boundaryTargetMode` 不再有主线分支。

#### 改动 3：pair 只允许同 ref / 邻 ref

当前 pairing 用 `neighborRefs(...,pairNeighborRefRadius)`，默认半径来自 `CBS_CGAN.m:17,27-33`，现主线默认 4 太宽。
主线固定成：

- `pairNeighborRefRadius = 1`
- `maxCandidatePairsPerRef = 3`

这里的 `3` 不是再做 pair sweep，而是把当前很多实验实际在用的 pair3 固定成主线；pair6 你已经做过比较，收益不成立。
证据：`experiments/all_non_png/Data/CBS_CGAN/train_quality_pair6_vs_pair3_D_default_epoch50_20260624_153705/train_quality_sweep_stage_metrics.csv`。

#### 改动 4：不要在 memory 阶段做“每 ref 只留 1 条”的全局信息损失

当前 `deduplicateRef()` 在 `UpdateBoundaryMemory_CBS.m:404-417` 是按 ref 直接保留前几条 gap 最小 pair。
主线里不要再在 memory 阶段强行压成单条主线边界。
更合理的是：

- memory 只做 exact duplicate 去重；
- component 与 chain 在 dataset 阶段派生。

这样 archive 仍然最小，但不会过早丢掉多分支边界信息。

------

### 2. `BuildBoundaryDataset_CBS.m`

这是主改文件。

#### 改动 1：新增唯一主线 condition mode：`comp_ref_s`

当前只有：

- `ref_only`
- `ref_tau`
- `ref_y`
- `ref_y_tau`

见 `BuildBoundaryDataset_CBS.m:84-92,133-147,159-179`。

主线新增且只默认使用：

- `comp_ref_s`

实现上可写成：

```matlab
case "comp_ref_s"
    C = [CompToken, Ref, S];
```

其中：

- `CompToken`：`component_id` 归一化标量；
- `Ref`：`W(ref,:)`；
- `S`：弧长坐标。

`ref_y` 和 `ref_y_tau` 只留作历史对照，不再作为主线。

#### 改动 2：TrainX 改为 `x_f`

当前 `TrainX = BMem.x_b(validTrain,:)`（`BuildBoundaryDataset_CBS.m:28-39`）。
主线改为：

```matlab
TrainX = BMem.x_f(validTrain,:);
```

并把 `Info.trainObjs` 里记录的 boundary 位置改成 pair 中点或 `y_f`，但 condition 不喂完整 objective。

#### 改动 3：删除主线里的 `PairLocalTau_CBS`

当前 `BuildBoundaryDataset_CBS.m:32-39,54-64` 会反复重算 `tau`。
主线里这块全部替换成 boundary chain context。

增加一个派生步骤，例如新 helper：

- `BuildBoundaryChainContext_CBS.m`

输入：`BMem`
输出：

- `order`
- `component_id`
- `s_arc`
- `mid_y`

实现最小版本就够：

1. `mid_y = 0.5*(y_f + y_i)`
2. 在 `mid_y` 上按第一主方向排序
3. 相邻 `mid_y` 距离如果大于 `median + 3*MAD`，切 component
4. component 内按累计弧长算 `s_arc`

这里我建议**只把 chain context 作为 dataset 的派生视图，不写回 archive**。
这样 archive 仍然是最小 pair archive。

#### 改动 4：query 从“同 component 内的空段”产生，不再由 `QueryY` 驱动 condition

当前 `buildExternalQueries()` 先用 `sortedBoundaryNodes(BMem,W)` 得序，再做 `missing_ref` 和 `large_gap`（`BuildBoundaryDataset_CBS.m:181-223,285-299`）。
主线改成：

- 排序基于 `mid_y` chain，不基于 ref-SVD；
- query 只在**同一 component 内**构造；
- `QueryC` 由 `comp_ref_s` 直接生成；
- `QueryY` 只作为诊断/画图字段，不再参与条件。

对区间 `(j,j+1)`，query 条件就是：

```matlab
comp_q = comp(j);
ref_q  = nearestReference(...);   % 或 missing ref 本身
s_q    = 0.5*(s(j) + s(j+1));
QueryC = [comp_q, W(ref_q,:), s_q];
```

如果中间有多个 missing refs，就按缺失 ref 顺序均匀插 `s_q`；
如果没有 missing ref 但 `arc gap` 特别大，就放一个 `large_gap` query。
主线不再从任何给定 `y_b_norm` 反推 `x`。

------

### 3. `BoundaryCGAN_CBS.m`

#### 改动 1：把主线语义从 inverse mapping 改掉

文件头当前写的是：

- `Minimal conditional GAN for boundary point inverse mapping`
  (`source/current_worktree/.../BoundaryCGAN_CBS.m:1-7`)

这句要改，因为新主线已经不是 `y -> x`。

#### 改动 2：训练改成固定两阶段，不增加新 sweep 参数

当前训练从 `trainBoundaryCGAN()` 一开始就是 joint（`BoundaryCGAN_CBS.m:60-106`）。
主线改成：

- 前半程：`advWeight=0`，只训 G；
- 后半程：恢复当前 `advWeight=1, reconstructionWeight=1`。

最小实现方式是不新增外部参数，直接在 `trainBoundaryCGAN()` 里按 `iter` 自动切半。例如：

```matlab
warmIters = ceil(Options.iter/2);
jointIters = Options.iter - warmIters;
```

然后：

- phase A：只走 `updateGeneratorBatch(...)`，且临时 `advWeight=0`
- phase B：按当前 `updateDiscriminatorSteps + updateGeneratorSteps`

这样：

- BCE 和 Huber 的代码都不用改；
- `generatorGradients()` 也不用新加 loss；
- 只改训练顺序。

#### 改动 3：主线固定 `trainMode='epoch'`

当前 `fillOptions()` 默认 `trainMode='iter'`（`BoundaryCGAN_CBS.m:319-346`）。
主线固定成 `epoch`，因为两阶段 schedule 在 epoch 语义下更自然，且你现在的 overfit/epoch sweep 也已经主要按 epoch 在看。

#### 改动 4：保持 `sampleZMode='zero'`

这一版主线里，边界分布是由 `comp_ref_s` 参数化表达的，不靠 `z` 去撒点。
所以：

- `trainZMode = zero`
- `sampleZMode = zero`

保留当前 zero-z 生产逻辑（`BoundaryCGAN_CBS.m:188-224`），不引入多候选随机采样，也不引入额外 FE。

------

### 4. `CBS_CGAN.m`

#### 改动 1：默认主线配置直接改死成一套

当前默认项见 `CBS_CGAN.m:17-24,27-77`。
主线统一成：

- `pairNeighborRefRadius = 1`
- `maxCandidatePairsPerRef = 3`
- `conditionMode = 'comp_ref_s'`
- `boundaryTargetMode = 'feasible_endpoint'`
- `trainMode = 'epoch'`
- `trainZMode = 'zero'`
- `sampleZMode = 'zero'`
- `queryPerCondition = 1`

也就是：

- 不再让主线默认落在 `ref_tau`
- 不再做 `endpoint / y_b_norm / ref_y / ref_y_tau` 分支切换
- 不再把 z 当作线上探索来源

#### 改动 2：保留当前单次真实评价流程

`CBS_CGAN.m:177-208,273-277` 这段保留。
即：

- 生成一次 `RawDec`
- 真实评价一次
- 直接并入 `UnionPopulation`

不增加任何 K 候选逻辑。

#### 改动 3：tau 相关日志退出主线

`LastDiag.train_tau_*` 这套指标在新主线里不再有主语义。
主线可改成记录：

- `train_s_range`
- `component_count`
- `component_cover`

但这只是诊断，不是算法主体。

------

### 5. `PairLocalTau_CBS.m`

主线退役。
可以保留文件供旧实验复现，但新主线不再调用。
建议新增或内嵌替代 helper：

- `BuildBoundaryChainContext_CBS.m`

如果你不想增加新文件，也可以直接把 chain context 函数写在 `BuildBoundaryDataset_CBS.m` 末尾。

------

### 6. `Support/run_CBS_CGAN_*`

主线 runner 不再继续做 A/B/C/D。

新的线上主线只保留一个 setting，例如：

- `conditionMode = comp_ref_s`
- `boundaryTargetMode = feasible_endpoint`
- `advWeight = 1`
- `reconstructionWeight = 1`
- `trainMode = epoch`
- `trainZMode = zero`
- `sampleZMode = zero`
- `pairNeighborRefRadius = 1`
- `maxCandidatePairsPerRef = 3`

也就是说，支持脚本要从“分支扫参器”改成“主线验证器”。

------

这版主线的本质可以概括成一句话：

**不是让 CGAN 记一个目标点 `y`，而是让 CGAN 学一条边界链的局部分布坐标 `(\mathrm{comp}, ref, s)` 到 near-boundary feasible `x_f` 的映射；边界分布由 pair archive 和 arc context 表达，对抗项只负责分布微调，不再从第 1 步主导训练。**

这就是我认为现在**最小、最统一、最能解决当前问题**的版本。

