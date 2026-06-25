下面按你要求，把内容严格分成三类：

- `[事实]`：能被 zip 内源码、CSV、PNG、README 直接确认。
- `[推断]`：我基于这些事实做出的解释。
- `[待验证]`：还需要补实验才能定论。

## 1. 附件中可直接确认的事实

- `[事实]` 我核查了 `notes/selected_images.txt:1-43` 列出的全部 43 张 PNG；其中包含 20 张最新 `visual_z_diagnostic` 图、15 张最新标准图、2 张 K3/K5 contact sheet、6 张旧主线参考图。`README.md:19-20,47-50`
- `[事实]` `README.md` 把 `source/fix.md` 定义为“Original repair proposal that guided the current CBS-CGAN implementation”，即历史修补提案，不是当前源码规范。`README.md:11-12`
- `[事实]` `README.md` 自身还带有一段 `Key Local Conclusion To Re-check`。这只是 bundle 作者的说明文字，不是独立实验结果。`README.md:26-35`
- `[事实]` 最新重点视觉实验比较的是 `B_ref_y_tau` 和 `C_ref_y`，并且问题集正是 `DASCMOP1/2/4/5` 与 `LIRCMOP5/6/7/8/9/10`。`README.md:13-16`，`source/new_CBS_CGAN/Support/run_CBS_CGAN_query_condition_ablation.m:138-149`
- `[事实]` `visual_z_diagnostic` 一共 6 个 panel，顺序固定为：`QueryC random z`、`QueryC z=0`、`QueryC fixed z`、`TrainC z=0`、`TrainC fixed z`、`Targets only`。`source/new_CBS_CGAN/Support/run_CBS_CGAN_boundary_quality_experiments.m:612-632`
- `[事实]` 图例颜色在代码里固定：训练集是橙色方块，Query target 是蓝色菱形，CGAN generated 是红色圆点。`source/new_CBS_CGAN/Support/run_CBS_CGAN_boundary_quality_experiments.m:704-742`
- `[事实]` 当前源码已经不是旧的“direction-only / ref-only / no pair-margin / no reconstruction”版本。
  1. `UpdateBoundaryMemory_CBS.m` 顶部注释已明确写成 `Build pair-supported thin boundary memory`，且上一轮回灌的是 `x_f/y_f` 与 `x_i/y_i`，只接受 finite-gap 的旧 pair。`source/new_CBS_CGAN/UpdateBoundaryMemory_CBS.m:1-3,64-88`
  2. `BuildBoundaryDataset_CBS.m` 现在显式构造 `QueryY`、`QueryMeta`，并支持 `ref_tau`、`ref_y_tau`、`ref_y` 三种条件模式。`source/new_CBS_CGAN/BuildBoundaryDataset_CBS.m:26-56,76-82,123-165`
  3. `BoundaryCGAN_CBS.m` 当前 generator loss 已包含 adversarial、Huber reconstruction、pair margin。`source/new_CBS_CGAN/BoundaryCGAN_CBS.m:238-255,272-286`
- `[事实]` 当前 `BMem` 的边界 pair 是这样采的：
  先在当前 feasible 中只保留第一前沿，按 ref 找 feasible 端，再从邻域 ref 找 infeasible 端；若 feasible 端支配 infeasible 端，则跳过；对保留 pair，再从当前 feasible 样本中挑与该 `y_f-y_i` 目标段最近的 `x_b/y_b`，并记录 `tau`。`source/new_CBS_CGAN/UpdateBoundaryMemory_CBS.m:100-159,162-183`
- `[事实]` 当前 `TrainC` 和 `QueryC` 已经走同一套条件构造函数 `referenceConditionsFromRefsTau`；`ref_y_tau` 和 `ref_y` 都把 `Y` 归一化后拼进条件。`source/new_CBS_CGAN/BuildBoundaryDataset_CBS.m:33-36,54-56,61-116`
- `[事实]` 当前 query 不是盲目乱造，而是两类：`missing_ref` 和 `large_gap`。`buildExternalQueries` 会在边界相邻段之间插值构造 `QueryY`，同时保留 `source_interval/source_type/tau`。`source/new_CBS_CGAN/BuildBoundaryDataset_CBS.m:167-208`
- `[事实]` `TrainC z=0` / `TrainC fixed z` 这两个 panel 不是“训练集重画”，而是把 `TrainC` 真正喂给生成器，再用 `Problem.CalObj` 投影到目标空间画出来。`source/new_CBS_CGAN/CBS_CGAN.m:520-561`
- `[事实]` `QueryC z=0` / `QueryC fixed z` / `QueryC random z` 也都是同一个生成器，只是 latent `z` 被强制成全零、固定行、或随机行。`source/new_CBS_CGAN/CBS_CGAN.m:526-549`
- `[事实]` GAN 训练时没有把 `Problem.CalObj` 或 `Problem.Evaluation` 放进训练图里。训练时只用到了 `TrainX`、`TrainC`、`trainXf/trainXi` 与上下界；真正的 `Problem.Evaluation` 发生在采样之后。`source/new_CBS_CGAN/BoundaryCGAN_CBS.m:29-95,238-255`，`source/new_CBS_CGAN/CBS_CGAN.m:188-223`
- `[事实]` 当前算法已经记录了你关心的几类指标：`boundary_dist50/90`、`query_obj_dist50/90`、`segment_width90`、`segment_width90_ratio`、`side_rate`、`pair_margin50`、`ref_cover`。`source/new_CBS_CGAN/CBS_CGAN.m:194-232`
- `[事实]` 旧主线 `CCMO_GAN_BDG` 的默认设置仍是：`archivePairDirectionMode="af_not_dominates_ai"`、`archivePairRefMode="neighbor4"`、`conditionMode="yt_dt_t_ref"`、`targetMode="near_segment_feasible"`、`decisionInterpCount=5`、`trainFilterMode="condition_knn"`。`source/old_CCMO_GAN_BDG/CCMO_GAN_BDG.m:1177-1211`
- `[事实]` 旧主线 archive 的核心确实更强调 AF/AI 配对过滤：先 Pareto 过滤，再做 `af_not_dominates_ai` 方向过滤，再做 `neighbor4` ref 邻域过滤，再按分数每个 ref 留少量 pair。`source/old_CCMO_GAN_BDG/UpdateBoundaryArchive_BDG.m:473-575,697-733`
- `[事实]` 旧主线 target 构造默认就是 `near_segment_feasible`，并且允许 `decisionInterpCount=5` 的扩带式 target；后面还可以做 condition-kNN 的决策空间散度筛除。`source/old_CCMO_GAN_BDG/BuildBoundaryTargetTriples_BDG.m:64-99`，`source/old_CCMO_GAN_BDG/BoundaryConditionKNNKeepMask_BDG.m:1-31`

## 2. 逐图观察摘要

图像记号约定：

- `B诊断图`：`results/latest_visual_z/images/B_ref_y_tau/<problem>_run1/figures/..._visual_z_diagnostic.png`
- `C诊断图`：`results/latest_visual_z/images/C_ref_y/<problem>_run1/figures/..._visual_z_diagnostic.png`
- `标准图`：同目录下 `...targetFE100000.png`

### DASCMOP1

- `[事实]` `Targets only` 中，蓝色 Query target 基本沿着橙色训练边界分布，几何上看是合理的。
- `[事实]` B 与 C 的 `QueryC random z / z=0 / fixed z` 都没有贴到蓝点链上；红点形成明显脱离目标区域的弧线/蛇形支路。
- `[事实]` 更关键的是，B 与 C 的 `TrainC z=0` 和 `TrainC fixed z` 也都不能重现橙色训练边界；红点仍落在另一条弯曲支路上。
- `[推断]` 这不是单纯的 query 外推失败，因为连训练条件下都没能回到训练边界。

### DASCMOP2

- `[事实]` `Targets only` 仍然合理，蓝点与橙点关系清楚。
- `[事实]` B 的 Query 面板红点分成多个脱离目标的簇；C 略接近一些，但仍有明显偏离。
- `[事实]` B 和 C 的 `TrainC z=0 / fixed z` 都出现明显的三角形/回环状红点结构，不是橙色训练边界的重现。
- `[推断]` 条件里加入 `y` 有帮助，但没有解决“训练条件下仍失真”的根问题。

### DASCMOP4

- `[事实]` B 与 C 的 `Targets only` 都是短而合理的橙蓝边界链。
- `[事实]` B 的三张 QueryC 面板都出现一条远离目标区的上升红线；C 也一样，只是尺度略小。`z=0` 和 `fixed z` 都没有把这条线拉回目标边界。
- `[事实]` B 与 C 的 `TrainC z=0 / fixed z` 同样是一条脱离训练集的上升红线，而不是橙色训练边界。
- `[事实]` `C_ref_y_stage_metrics_all.csv` 在 `DASCMOP4_BC, target_FE=100000` 上给出 `boundary_dist90=0.2177089`、`query_obj_dist90=0.2194150`、`feasible_rate=0.05`；B 同行是 `0.7648838 / 0.7673746 / 0`。`results/latest_visual_z/csv/C_ref_y_stage_metrics_all.csv:16`，`results/latest_visual_z/csv/B_ref_y_tau_stage_metrics_all.csv:16`
- `[推断]` 数字上 C 比 B 好很多，但图上红线仍然明显脱靶；这正是“CSV 改善但视觉目标仍未达成”的反例。

### DASCMOP5

- `[事实]` `Targets only` 里蓝点本身仍是合理的小范围边界链。
- `[事实]` B 与 C 的三张 QueryC 面板都出现远离目标区、且量级明显异常的对角红线；C 比 B 更夸张。
- `[事实]` `TrainC z=0 / fixed z` 也同样离得很远，说明不是 random z 单独造成的。
- `[事实]` `C_ref_y_stage_metrics_all.csv` 在 `DASCMOP5_BC, target_FE=100000` 上 `train_count=114`、`query_count=20`、`feasible_rate=0`、`boundary_dist90=2.1891479`；B 同行是 `train_count=66`、`feasible_rate=0`、`boundary_dist90=1.4073620`。`results/latest_visual_z/csv/C_ref_y_stage_metrics_all.csv:21`，`results/latest_visual_z/csv/B_ref_y_tau_stage_metrics_all.csv:21`
- `[推断]` 到 `FE100000` 时并不是“没有训练数据”，而是“有训练数据但生成仍整体跑飞”。

### LIRCMOP5

- `[事实]` B 与 C 的 `Targets only` 都合理。
- `[事实]` `TrainC z=0 / fixed z` 在两种 condition 下都基本能贴着橙色训练边界。
- `[事实]` `QueryC z=0 / fixed z` 也大体贴边；主要问题出在 `QueryC random z`，会出现小钩子或偏侧支路。
- `[推断]` 这个问题更像“latent z 把已经学会的边界吹散”，而不是训练集/条件本身完全错误。

### LIRCMOP6

- `[事实]` B 与 C 的 `TrainC z=0 / fixed z` 基本都能重现训练边界。
- `[事实]` `QueryC z=0 / fixed z` 也贴近蓝色 target；但 `QueryC random z` 会产生上方额外分支，且 C 比 B 更明显。
- `[推断]` 这里主问题也是 random z 的扩散，而不是 condition 本身不足。

### LIRCMOP7

- `[事实]` B 与 C 的 `TrainC z=0 / fixed z` 几乎都能和橙色训练边界重合。
- `[事实]` `QueryC z=0 / fixed z` 也基本贴蓝点；但 `QueryC random z` 会出现短弧/离群支路。
- `[事实]` `C_ref_y_stage_metrics_all.csv` 在 `LIRCMOP7_BC, target_FE=100000` 上 `boundary_dist90=0.0189397`，但 `feasible_rate=0`。`results/latest_visual_z/csv/C_ref_y_stage_metrics_all.csv:36`
- `[推断]` 这个问题说明“离边界近”不等于“落在正确且可行的一侧”。

### LIRCMOP8

- `[事实]` B 与 C 的 `TrainC z=0 / fixed z` 都能很好地重现橙色训练边界。
- `[事实]` `QueryC z=0 / fixed z` 也基本贴蓝点。
- `[事实]` 但 `QueryC random z` 会产生最明显的远离点，尤其 B 版本最严重。
- `[事实]` 在 `FE100000`，C 的 `boundary_dist90=0.0360476`，明显小于 B 的 `0.3422332`。`results/latest_visual_z/csv/C_ref_y_stage_metrics_all.csv:41`，`results/latest_visual_z/csv/B_ref_y_tau_stage_metrics_all.csv:41`
- `[推断]` 这里 `ref_y` 明显比 `ref_y_tau` 更稳，但仍需要管住 random z。

### LIRCMOP9

- `[事实]` B 与 C 的 `TrainC z=0 / fixed z` 都能重构训练边界。
- `[事实]` `QueryC z=0 / fixed z` 大体贴近蓝点；`random z` 仍有小偏移钩子。
- `[推断]` 这是“基本学到边界，但 production sampling 还不够收敛”的类型。

### LIRCMOP10

- `[事实]` 这是当前 bundle 里最接近目标的一类图。B 与 C 的 `TrainC z=0 / fixed z` 和 `QueryC z=0 / fixed z` 都很贴边。
- `[事实]` `QueryC random z` 仍有轻微漂移，但幅度远小于前面几类。
- `[推断]` 当前方案在一部分 LIRCMOP 上已经具备“可学会边界”的能力，所以问题不是“CGAN 完全不能做这件事”，而是“在 harder DASCMOP 上闭环还断着”。

### 旧主线 CCMO-GAN-BDG 对照

- `[事实]` 我核查了 `results/old_CCMO_GAN_BDG/images/DASCMOP2_BC_run1_FE100000_domain_gan_train.png`、`DASCMOP4_BC_run1_FE100000_domain_gan_train.png`、`DASCMOP5_BC_run1_FE100000_domain_gan_train.png`、`LIRCMOP5_BC_run1_FE070000_domain_gan_train.png`、`...FE100000...png` 以及 `old_ccmo_contact_sheet.png`。
- `[事实]` 旧主线在 `DASCMOP4/5` 上有明显长斜射线和厚带；在 `DASCMOP2`、`LIRCMOP5` 上也更像厚点云而不是窄边界。
- `[推断]` 旧主线能提供某些“边界趋势”启发，但它本身也没有达到你要的“薄边界生成”。

## 3. 当前问题链条

### 3.1 训练数据链条

- `[事实]` 当前 `UpdateBoundaryMemory_CBS.m` 已经是 pair-supported、thin-boundary 版本，不再是旧的“单侧 feasible 链 + inf gap 回填”逻辑。`source/new_CBS_CGAN/UpdateBoundaryMemory_CBS.m:3,64-88,100-183`
- `[事实]` 当前 `BuildBoundaryDataset_CBS.m` 训练 target 直接取 `TrainX = BMem.x_b`，并把 `trainXf/trainXi/trainYf/trainYi/trainTau` 一起打包到 `Info`。`source/new_CBS_CGAN/BuildBoundaryDataset_CBS.m:28-49`
- `[推断]` 所以，**训练集构造不是现在最先坏掉的环节**。它已经比旧版更接近“当前一条薄边界”的目标。

### 3.2 条件链条

- `[事实]` 当前 TrainC 和 QueryC 都走 `referenceConditionsFromRefsTau(...)`；`ref_y_tau` 与 `ref_y` 都没有丢弃 `QueryY`。`source/new_CBS_CGAN/BuildBoundaryDataset_CBS.m:33-36,54-56,76-82`
- `[事实]` Query 也不是瞎采，而是基于 `missing_ref` 与 `large_gap` 的边界插值。`source/new_CBS_CGAN/BuildBoundaryDataset_CBS.m:167-208`
- `[推断]` 因此，**当前 bundle 的根因已经不是旧 `fix.md` 中那种“direction-only conditioning”**。

### 3.3 CGAN 学习链条

- `[事实]` 当前 G/D 结构都很小：G 是 `64-64-64` 三层隐藏层，D 是 `64-64-32`。`source/new_CBS_CGAN/BoundaryCGAN_CBS.m:191-216`
- `[事实]` 训练时 generator loss 是
  `L_adv + reconstructionWeight * Huber(X_gen - X_b) + pairMarginWeight * margin(X_f,X_i)`。`source/new_CBS_CGAN/BoundaryCGAN_CBS.m:238-255`
- `[事实]` 这些损失都发生在**决策空间**；训练过程中没有直接约束 `Problem.CalObj(G(z,C))` 去等于条件里的 `y`。`source/new_CBS_CGAN/BoundaryCGAN_CBS.m:29-95,238-255`
- `[推断]` 当前模型学到的是“给定条件后输出某个决策向量 `X`”，但**最终验收目标**却是“该 `X` 投影后必须落在目标空间的薄边界上”。这个闭环在训练里没有被直接关上。

### 3.4 投影到目标空间后的结果链条

- `[事实]` `TrainC z=0` / `TrainC fixed z` 是直接把训练条件送入当前 generator，再用 `Problem.CalObj` 投影出的结果。`source/new_CBS_CGAN/CBS_CGAN.m:520-561`
- `[事实]` 在 DASCMOP1/2/4/5 上，这两个 TrainC panel 已经失败；而在 LIRCMOP5–10 上，这两个 panel 大多成功。
- `[推断]` 这说明当前问题分成两类：
  1. DASCMOP1/2/4/5：**训练条件下都不能重现训练边界**，根问题发生在“学什么 / 怎么学”这一级；
  2. LIRCMOP5–10：训练条件下能重现，但 random z 把 Query 结果吹散，根问题更偏向“怎么采样”。

### 3.5 对核心目标的直接回答

- `[推断]` 基于当前源码、CSV 和 43 张图，我的结论是：**当前 CBS-CGAN 并没有在这批问题上稳定实现“目标空间可行/不可行薄边界生成”**。
  它在一部分 LIRCMOP 问题上已经能在 `TrainC z=0/fixed z` 下复现薄边界，但在 DASCMOP1/2/4/5 上仍然没有把“训练边界 → 生成边界 → 目标空间薄边界”这个链条闭合。

## 4. 候选原因排序

### 1) 目标空间监督缺位，是当前最主要原因

- `[事实]` 训练损失只约束了决策空间 `X` 与 pair-side 的决策距离；没有直接约束 `CalObj(G(z,C))` 贴近条件中的 `y`。`source/new_CBS_CGAN/BoundaryCGAN_CBS.m:238-255`
- `[事实]` DASCMOP1/2/4/5 的 `TrainC z=0 / fixed z` 已经失败；这一步根本还没进入 query 外推。
- `[事实]` `DASCMOP4` 的 C 版本在 CSV 上比 B 版本好很多，但图上仍是一条脱离边界的红线。`results/latest_visual_z/csv/C_ref_y_stage_metrics_all.csv:16`，`results/latest_visual_z/csv/B_ref_y_tau_stage_metrics_all.csv:16`
- `[推断]` 这说明当前核心矛盾不是“没 pair / 没 QueryY / 没 tau”，而是：**训练目标主要在决策空间闭合，验收目标却在目标空间闭合。**
- `[待验证]` 需要通过加入 objective-space consistency 的可微代理损失来证伪或证实。

### 2) random z 会显著放大误差，但它不是 DASCMOP 失败的根因

- `[事实]` 在 LIRCMOP5/6/7/8/9/10 上，`QueryC z=0` 和 `QueryC fixed z` 往往明显好于 `QueryC random z`。
- `[事实]` 在 DASCMOP4/5 上，连 `z=0` 和 `fixed z` 都同样失败。
- `[推断]` 所以 random z 是**二级放大器**：它解释了多数 LIRCMOP 的钩子/离群点，但解释不了 DASCMOP 的训练条件重构失败。
- `[待验证]` 将 production sampling 固定为 `z=0` 或低方差固定 latent 后，LIR 问题应明显收敛；若 DASCMOP 仍失败，则更能证明主因不在 z。

### 3) `ref_y` 通常比 `ref_y_tau` 更稳，但 tau 不是稳定收益项

- `[事实]` 当前 A/B/C 条件分支是 `A_ref_tau / B_ref_y_tau / C_ref_y`。`source/new_CBS_CGAN/Support/run_CBS_CGAN_query_condition_ablation.m:138-141`
- `[事实]` 在 `ABC_K3_paired_variant_deltas.csv` 中，`C_ref_y` 相比 `B_ref_y_tau` 的 `delta_query_obj_dist90` 中位数为负，且 150 个配对 stage 里有 98 个是改善；`delta_boundary_dist90` 也有 90 个 stage 改善。`results/ablations/csv/ABC_K3_paired_variant_deltas.csv`
- `[事实]` 但这种改善不是全局稳定的；例如 `LIRCMOP8` 的 median `boundary_dist90` 是 B 优于 C。`results/ablations/csv/ABC_K3_comparison_stage_metrics_by_problem.csv:22-31`
- `[推断]` 现阶段最稳的事实不是“tau 有用”，而是“**加入 y 往往比加入 tau 更能定位目标**”。tau 至少在当前实现里，没有呈现稳定的正收益。
- `[待验证]` 需要把 `ref_y` 作为新的单主线，再单独检验 tau 是否还能带来增益。

### 4) 训练样本数 / pair 数量不是主因，只是次要调节项

- `[事实]` `K3_K5_comparison_analysis_summary.csv` 里，`pair5` 显著提高了很多问题的 `train_count_med`，例如 DASCMOP4 是 `54 -> 105`，LIRCMOP7 是 `132 -> 230`。`results/ablations/csv/K3_K5_comparison_analysis_summary.csv:2-21`
- `[事实]` 但这些增长并没有稳定换来视觉修复：
  DASCMOP4 的 `feasible_rate_med` 仍为 0；LIRCMOP7 的 `boundary_dist90_med` 反而变差。`results/ablations/csv/K3_K5_comparison_analysis_summary.csv:4,8,14,18`
- `[推断]` “多 pair / 多训练点”不是当前最核心的堵点。否则 K5 应该更稳定地修复红点形态，但实际没有。
- `[待验证]` 如果后续加了 objective-space bridge 后，pair-count 的收益可能才会变得更可解释。

### 5) QueryY 构造不是当前主因

- `[事实]` 当前代码会显式构造 `QueryY`、`source_interval`、`source_type`，而不是只靠 ref。`source/new_CBS_CGAN/BuildBoundaryDataset_CBS.m:167-208`
- `[事实]` 从 20 张 `visual_z_diagnostic` 观察，`Targets only` 面板在多数问题上都几何合理，蓝点大多沿着橙色训练边界或其缺口插值位置分布。
- `[推断]` 当前主要异常不在蓝点 target 本身，而在红点没有学会去追这些 target。
- `[待验证]` 只有在加入 objective-space bridge 后，如果仍然大量脱靶，才应回头质疑 QueryY 构造。

### 6) 旧主线不是答案；不能回退到旧厚带逻辑

- `[事实]` 旧主线确实更强调 AF/AI 成对筛选。`source/old_CCMO_GAN_BDG/UpdateBoundaryArchive_BDG.m:473-575,697-733`
- `[事实]` 但旧主线默认 target 是 `near_segment_feasible`，还带 `decisionInterpCount=5`。`source/old_CCMO_GAN_BDG/CCMO_GAN_BDG.m:1189-1193`
- `[事实]` 旧图在 `DASCMOP4/5` 上有明显厚带和斜射线。
- `[推断]` 旧主线可借鉴的是“严格 pair 证据”，不是“厚带 target”或“回到旧图形态”。

### 7) 网络宽度、epoch、mismatch condition、D/G 强弱目前都不是前排证据

- `[事实]` 当前 zip 里没有任何一项实验能把“网络容量不足”与“目标空间闭环缺失”明确分离出来。
- `[推断]` 它们当然可能有影响，但在现有证据下不应排在前四，更不应优先用“加网络/加 epoch/调 lr”去解释全部问题。

## 5. 最小改造方案

我建议只走一条主线，不分叉：

**主线：保留当前 paired thin BMem，默认条件改为 `ref_y`，在现有 CGAN 上补一条 objective-space consistency 闭环，并把 production z 收紧到固定/低方差。**

### 5.1 新的训练数据如何构造

- `[方案]` **保留当前 `UpdateBoundaryMemory_CBS.m` 的 paired thin boundary memory 逻辑**，不要回退到旧 `near_segment_feasible` 厚带 target。
- `[方案]` 训练行继续使用当前 `(x_b, x_f, x_i, y_b, y_f, y_i, ref)`；不额外扩增 decision-interp 样本。

### 5.2 条件 C 应该包含什么，不应包含什么

- `[方案]` 主条件统一为 `C = [ref, y]`，即直接采用当前已经实现的 `ref_y`。
- `[方案]` `tau` 继续保留在 metadata 里，继续用于 `missing_ref / large_gap` query 的构造与分析，但**不再作为主条件默认输入**。
- `[推断]` 这样最符合当前证据：`y` 已经在多数问题上比 `tau` 更稳定地提供了几何定位。

### 5.3 z 应该如何使用

- `[方案]` 训练阶段保留小方差 latent；生产阶段默认不用 broad random z，而是固定 `z=0` 或固定低方差 latent。
- `[方案]` `random z` 只保留给 `visual_z_diagnostic` 和 ablation，不作为正式边界生成模式。
- `[推断]` 这与 LIRCMOP5–10 的视觉证据一致。

### 5.4 Generator 应该输出什么

- `[方案]` 不改核心创新点：`G(C,z) -> 完整决策变量 X`。

### 5.5 Discriminator 应该判别什么

- `[方案]` 保留当前 `D(X,C)` 的 conditional 判别器和 mismatch condition 机制。`source/new_CBS_CGAN/BoundaryCGAN_CBS.m:219-235`

### 5.6 损失函数应该包含什么

- `[方案]` 保留当前三项中的两项：
  1. `L_adv`
  2. `L_pair`（pair margin）
- `[方案]` 保留当前 `X` 空间 Huber，但把它从“唯一监督主项”降成“分支选择锚点”。
- `[方案]` **新增唯一一个必要桥梁**：objective proxy consistency。
  做法是在 `BoundaryCGAN_CBS.m` 内部增加一个轻量 `R: X -> Y` 代理回归器，只用当前已经评估过的 `(X,Y)` 样本训练。然后用
  `L_obj = Huber(R(X_gen), y_target)`
  约束生成结果在目标空间上追条件里的 `y_target`。
- `[推断]` 这是当前最小、最统一的补丁，因为现在真正缺的是“从 `X` 到目标空间边界”的训练闭环。

### 5.7 哪些旧损失应该删除或降权

- `[方案]` 不删除当前 `L_adv`、`L_pair`、`L_x`，但要让 `L_obj` 成为主监督；`L_adv` 退到分布正则器角色，`L_x` 退到单逆分支锚点角色。
- `[方案]` 不要引回旧主线那套 thick target / decision interpolation / near-segment 扩带监督。

### 5.8 如何保证生成的是目标空间边界而不是厚点云

- `[方案]` 四个动作一起做：
  1. 训练数据仍是一条 pair-supported thin boundary；
  2. 条件用 `ref_y` 固定几何位置；
  3. `L_obj + L_pair` 同时约束“目标空间贴边”和“站在 feasible 一侧”；
  4. production 用 fixed/low-variance z，避免把一条线吹成一团云。

### 5.9 如何处理 missing ref / large gap query

- `[方案]` 保留当前 `buildExternalQueries` 的 `missing_ref` 与 `large_gap`，因为当前图里蓝色 target 大多数是合理的。
- `[方案]` 改的是：这些 query 不再交给 broad random z，而是交给 fixed/low-variance z，并由 `L_obj` 让生成结果主动追向 query `y`。

### 5.10 如何把生成解用于优化流程

- `[方案]` 仍沿用当前主流程：`G -> RawDec -> Problem.Evaluation -> EnvironmentalSelection_CBS`。`source/new_CBS_CGAN/CBS_CGAN.m:188-223`
- `[方案]` 但把 `side_rate / pair_margin50 / query_obj_dist90` 作为正式保留前诊断门槛；如果 fixed-z 生成仍严重偏离，则不让它进入主优化流程。

## 6. 修改文件清单

- `source/new_CBS_CGAN/BuildBoundaryDataset_CBS.m`
  目的：把 `ref_y` 设为新的主线条件；保留 `tau` 作为 query metadata，不再把 `tau` 当默认主条件。
- `source/new_CBS_CGAN/BoundaryCGAN_CBS.m`
  目的：在现有 `L_adv + L_x + L_pair` 基础上加入 `L_obj`；同时提供正式的 fixed/low-variance production sampling 入口。
  这是最核心的改动位点。
- `source/new_CBS_CGAN/CBS_CGAN.m`
  目的：
  1. 把 `trainObjs/queryObjs` 显式传给 `BoundaryCGAN_CBS`；
  2. 默认生产采样改为 fixed/low-variance z；
  3. 记录新的 objective reconstruction 指标。
- `source/new_CBS_CGAN/Support/run_CBS_CGAN_boundary_quality_experiments.m`
  目的：保留现有 `visual_z_diagnostic`，并补 `TrainC objective reconstruction error`、`QueryC target error`、`z sensitivity` 的 CSV。
- `source/new_CBS_CGAN/Support/run_CBS_CGAN_query_condition_ablation.m`
  目的：在同一 runner 下，复核 `ref_y` 主线在 fixed-z 与 objective-consistency 加入后的变化。
- `[不建议大改] source/new_CBS_CGAN/UpdateBoundaryMemory_CBS.m`
  `[事实]` 当前它已经比旧版更接近你的目标。
  `[推断]` 它不是第一修改位点。只有在补上 `L_obj` 后仍出现明显历史拖尾，才再考虑是否削弱 `appendPreviousPairedCandidates`。

## 7. 最小验证实验

### 7.1 实验设置

- `[方案]` 线程数：`1`，先保证可复现实验图。
- `[方案]` 两步最小实验：
  1. **视觉验证**：`runs=1`，开 `visualDiagnostics=1`，问题只跑 4 个；
  2. **确认统计**：`runs=3`，不开大图，只记 CSV。
- `[方案]` 问题集最少保留 4 个：
  - `DASCMOP4_BC`：当前最明显失败；
  - `DASCMOP2_BC`：当前是“有些改善但仍失真”的中间态；
  - `LIRCMOP8_BC`：当前最典型的 random-z 敏感；
  - `LIRCMOP10_BC`：当前相对最好，用来防回归。

### 7.2 必须画的图

- `[方案]` 每个问题都画：
  - 标准最终图 `...targetFE100000.png`
  - `visual_z_diagnostic`
  - 边界区域放大图
  - 可行/不可行区域背景图
- `[方案]` 同时保留当前 bundle 的 old-mainline 对照图，至少对 `DASCMOP4_BC` 做新旧对比。

### 7.3 必须记录的指标

- `[方案]` 保留现有：
  `feasible_rate`、`boundary_dist50/90`、`query_obj_dist50/90`、`segment_width90`、`segment_width90_ratio`、`side_rate`、`pair_margin50`、`ref_cover`
- `[方案]` 新增三类：
  1. `TrainC objective reconstruction error`：`CalObj(G(TrainC,z_fixed))` 到 `trainObjs` 的误差；
  2. `QueryC objective target error`：`CalObj(G(QueryC,z_fixed))` 到 `queryObjs` 的误差；
  3. `z sensitivity`：同一 `QueryC` 下 `random z` 与 `fixed z` 的目标空间差异。

### 7.4 成功判定

- `[方案]` 视觉判定必须先过：
  1. `DASCMOP4_BC` 的 `TrainC fixed z` 不再出现那条脱离目标区的上升红线；
  2. `DASCMOP2_BC` 的 `TrainC fixed z` 不再是三角回环；
  3. `LIRCMOP8_BC` 的 `QueryC fixed z` 贴住蓝点，且 `random z` 不再产生大范围离群点；
  4. `LIRCMOP10_BC` 不得明显退化。
- `[方案]` 数值判定至少满足：
  - `DASCMOP4_BC`、`DASCMOP2_BC` 在 `FE100000` 的 `boundary_dist90` 和 `query_obj_dist90` 相比当前 `C_ref_y` 至少下降 50%；
  - `side_rate >= 0.8`；
  - `segment_width90_ratio` 不高于当前最好版本；
  - `LIRCMOP10_BC` 的 `query_obj_dist90` 不劣化超过 20%。

### 7.5 能证伪方案的条件

- `[待验证]` 如果加入 `L_obj` 后，`TrainC fixed z` 在 DASCMOP4/5 仍明显不能重构橙色训练边界，那么根因就不只是 objective-space 闭环缺失，BMem target 本身还要重查。
- `[待验证]` 如果 `Targets only` 仍合理，但 `QueryC fixed z` 继续大偏离，则说明 `ref_y` 条件本身仍不足以确定逆映射。
- `[待验证]` 如果 CSV 指标改善，但图上仍存在脱离边界的弧线/对角线/蛇形支路，则该方案应判失败。
- `[待验证]` 如果 hard DASCMOP 改善，但原本表现较好的 LIRCMOP10 明显退化，则说明方案过拟合困难问题，不应收为主线。

## 8. 风险和反例

- `[待验证]` 轻量 `X -> Y` proxy 本身可能在边界附近不准；如果 proxy 学歪了，`L_obj` 会把 generator 往错误目标空间推。
- `[待验证]` 如果最后只有 fixed z 有效，而 random z 仍然一吹就散，那就说明 latent 在这条任务线上更像“扰动源”而不是“有益多样性源”；这未必违背你的目标，但意味着 production 不应再指望 random z。
- `[待验证]` 当前 `UpdateBoundaryMemory_CBS.m` 还会回灌上一轮 pair 端点；如果在补 `L_obj` 后仍出现明显历史拖尾，再考虑对历史 pair 加时效衰减。
- `[事实]` 旧主线已经证明“更厚的数据、更多插值、更多扩带”并不自动等于更好的边界生成。
  `[推断]` 所以后续不应靠回退旧 thick-target 思路来赌运气。

一句话总结：

- `[事实]` 当前源码已经修掉了不少旧问题：pair-supported BMem、`ref_y/ref_y_tau`、Huber、pair margin 都已在代码里。
- `[推断]` 但当前最关键的断点已经变成：**CGAN 训练仍主要在决策空间闭合，而你要的成功标准是在目标空间的可行/不可行薄边界闭合。**
- `[方案]` 最小且统一的修补，不是继续堆 pair 数、tau、epoch，而是：**保留当前 thin paired BMem，默认切到 `ref_y`，补一条 objective-space consistency 闭环，并把 production z 固定/收紧。**
