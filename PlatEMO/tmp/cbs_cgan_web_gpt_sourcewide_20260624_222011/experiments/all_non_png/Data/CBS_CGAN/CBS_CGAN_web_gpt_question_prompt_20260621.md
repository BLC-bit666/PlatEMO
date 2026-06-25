# 给网页版 GPT 的严格评审提问稿

请同时阅读我上传的两个附件：

1. `CBS_CGAN_current_fixmd_source_results_web_gpt_20260621_7M.txt`
2. `CBS_CGAN_problem_images_for_web_gpt_20260621.zip`

第一个 txt 包含当前 `fix.md`、CBS-CGAN 当前主线源码、相关分支源码、诊断脚本、实验结果摘要和关键 CSV 内容。第二个 zip 包含少量能够体现当前问题的图像 contact sheet。

请不要把当前实现、当前 `fix.md` 或我们目前的设想默认当成正确答案。你的任务是基于附件中的源码、实验结果和图片，对“当前设想、当前实现思路、当前问题诊断”进行严格评判、纠错和优化，然后提出一条能落地到源码的主线方案。

## 我们实际遇到的问题

下面这些是我们现在真正需要解决的问题。请把这些问题作为分析中心，而不是只评价某个已有分支是否跑得更好。

### 问题 1：边界存档和训练集构造不够可靠

当前训练集对 CGAN/GAN 的影响非常大，但现有边界存档和训练集构造仍不稳定。图像中可以看到：

- 一部分训练点可能位于一个边界段，另一部分训练点可能位于另一个更好或更差的边界段；
- 一些训练集不是一条窄边界线，而是离散段、厚点云或多个区域的混合；
- 当训练集混合不同边界段或不同质量层级时，CGAN 生成点可能出现跨越两个边界的弧形、厚带或散点。

我们需要你判断：这是当前实现的问题、`fix.md` 规则的问题、还是当前算法设想本身的问题。

### 问题 2：CGAN/GAN 的核心价值还没有被证明

我们希望 CGAN/GAN 的价值不是复现已有训练集，而是：

```text
通过已探索到的局部边界，学习当前边界分布；
再按照 reference 或目标空间条件，在还没有探索到的区域生成有价值的边界解。
```

当前疑问包括：

- 如果 CGAN 只能在已有训练点附近生成，那么它对算法的核心贡献不足；
- `z=0` 或小 `z` 是否限制了生成未探索边界区域的能力；
- reference 是否能指导 CGAN 在缺失方向生成边界解；
- 目标空间 condition、reference、chain 信息和 z 之间到底应该如何分工；
- 如何证明生成点确实补到了未探索 reference 区域，而不是只在已有训练区域附近抖动。

### 问题 3：CGAN/GAN 生成点不够贴边，且不是一条窄边界

我们的目标是生成一条目标空间可行/不可行边界附近的窄边界线，而不是厚点云。当前图像和指标中仍能看到：

- CGAN 生成点有时明显偏离训练集分布；
- CGAN 生成点有时不贴近可行/不可行边界；
- CGAN 生成点可能形成厚带、散点、局部簇或跨段弧线；
- LIRCMOP 上有些阶段比 DASCMOP 更接近边界趋势，但整体仍不稳定；
- DASCMOP1/2 与 LIRCMOP5-10 的整体问题比 DASCMOP4/5 个别特殊现象更重要。

我们需要你判断：问题主要来自 BMem/TrainC、QueryC、z、网络结构、训练方式、损失函数，还是这些因素共同造成。

### 问题 4：CGAN/GAN 起作用阶段可能太晚

我们不希望 CGAN/GAN 只在已经找到真实边界后才起作用。我们希望每代都能把当前搜索前沿或当前自认为的边界当作训练对象，让 CGAN/GAN 生成当前阶段认为有价值的边界解，即使早期这个边界还不是真实可行/不可行边界。

当前矛盾是：

- 核心创新要求边界最终必须是目标空间可行/不可行边界；
- 但算法早期可能没有可靠 F/I pair；
- 如果硬等真实 F/I pair，CGAN/GAN 可能启动太晚；
- 如果随意定义伪边界，又可能偏离“目标空间 F/I 边界”这个核心创新点。

我们需要你给出一个不偏离硬约束的早期机制。

## 我们对你的明确要求

请你完成的是“严格评审 + 主线重构建议”，不是简单点评实验。

你必须做到：

1. **先判断问题本质**  
   判断当前失败主要来自训练集/BMem、QueryC、GAN 训练、损失函数、早期机制，还是算法设想本身。

2. **评判我们当前设想**  
   对后文设想 A-F 逐条判断：保留、修改、废弃或证据不足。不要默认我们当前想法是对的。

3. **提出一条单一主线方案**  
   不要给多个并列方向。请提出一条收敛、统一、尽量少分支的方案，说明 BMem、TrainC/TrainX、QueryC、reference、z、GAN 训练和早期伪边界如何统一。

4. **方案必须能落地到源码**  
   需要具体到函数级别说明应改哪些文件，例如 `UpdateBoundaryMemory_CBS.m`、`BuildBoundaryDataset_CBS.m`、`BoundaryCGAN_CBS.m`、`CBS_CGAN.m` 等，每个文件改什么、数据结构如何变化、调用关系如何变化。

5. **不要用后处理掩盖问题**  
   不要主要依赖投影、筛选、修复或兜底规则制造好看的边界图。我们需要生成器本身学到边界解分布。

6. **给出最少但有效的验证指标**  
   指标要围绕 CGAN/GAN 生成边界解质量：窄边界、贴边、目标条件一致性、可行性、未探索 reference 覆盖价值。不要把 IGD/HV 当核心指标。

7. **明确哪些结论有证据，哪些只是推断**  
   你必须区分源码事实、CSV/图片事实、推断和不确定项。证据不足时直接说证据不足。

## 核心目的

我要做的是：用边界解训练 CGAN/GAN，并让 CGAN/GAN 直接生成完整决策变量 `x`，使 `f(x)` 在目标空间形成可行/不可行边界附近的一条窄边界。

核心目标不是：

- 生成厚点云；
- 只复现已有训练点附近的重复点；
- 只让图像看起来更好；
- 用目标空间点再反解决策变量；
- 依赖 IGD/HV 判断 CGAN 是否有价值。

核心目标是：

- 训练数据应能代表当前阶段的边界或当前阶段自认为的边界；
- CGAN/GAN 生成的完整决策变量经真实 `Problem.Evaluation` 后应贴近目标空间可行/不可行边界；
- 生成结果应尽量形成一条窄边界线，而不是厚带或散点云；
- CGAN/GAN 应能在未覆盖 reference 区域生成有价值边界解，而不仅仅复现已有训练区域；
- 早期还没有找到真实边界时，也希望 CGAN/GAN 能基于当前搜索前沿生成“当前自认为的边界”，以推动搜索。

## 硬约束

1. 必须使用 CGAN 或 GAN，不能把核心创新替换成普通插值、局部搜索、回归模型或修复算子。
2. 生成器必须直接生成完整决策变量 `x`。
3. “边界”最终必须指目标空间中的可行/不可行边界。早期可以讨论 current frontier / pseudo boundary / self-believed boundary，但必须解释它如何过渡或服务于最终目标空间 F/I 边界，不能把最终边界定义偷换成非 F/I 边界。
4. 算法设计优先统一、减法和收敛，不要靠大量分支、小技巧、后处理、兜底规则或堆 loss 掩盖本质问题。
5. 不要围绕 IGD/HV 展开。当前只关注 CGAN/GAN 生成边界解的质量：是否窄、是否贴边、是否覆盖未探索 reference 区域、是否直接生成可用决策解。
6. 不要只聚焦 DASCMOP4、DASCMOP5 等个别特殊问题。请分析整体规律，尤其是 DASCMOP1/2 与 LIRCMOP5-10 的整体表现。

## 当前候选设想：请评判，不要默认接受

下面是我们目前形成的候选设想。它们不一定正确，请你逐条评判是否成立、是否违反约束、是否需要修改或废弃。

### 设想 A：训练目标从 F/I 中点改为可行侧端点

历史上曾考虑：

```text
y_b = (y_f + y_i) / 2
```

后来认为 F/I 中点可能不可达，且 `condition -> target` 与 `x_b` 不自洽，所以当前 feasible-side 分支改成：

```text
F/I pair:
    (x_f, y_f, feasible)
    (x_i, y_i, infeasible)

BMem node:
    y_b = y_f
    x_b = x_f
    gap = ||norm(y_f) - norm(y_i)||
```

请评判：

- 用可行侧端点替代中点是否合理；
- 它是否仍然能代表目标空间 F/I 边界；
- 它是否会把训练目标从“边界”偏移为“可行侧前沿”；
- 如果该设想不充分，应该如何修正而不回到不可达中点。

### 设想 B：每代都应构造 current self-believed boundary

我们目前认为：CGAN 不应只在已经找到真实边界后才起作用。早期即使没有可靠 F/I pair，也应基于当前搜索前沿构造“当前自认为的边界”，训练 CGAN/GAN 生成当前阶段认为有价值的边界解。

请评判：

- 这个设想是否合理；
- 它是否与“边界最终必须是目标空间 F/I 边界”冲突；
- 早期 current frontier / pseudo boundary 应如何定义，才能不偏离最终创新点；
- 这个机制应放在 `UpdateBoundaryMemory_CBS.m`、`BuildBoundaryDataset_CBS.m`、主循环，还是需要重构数据流。

### 设想 C：BMem 必须从点级一致性升级到 chain-level 一致性

当前图像中可能出现训练集多个边界段、多个质量层级、不同阶段前沿混合的现象。我们目前怀疑 BMem/TrainC 混合不一致 chain 会导致 CGAN 生成弧形跨越、厚带或散点。

请评判：

- 从源码看，当前 `UpdateBoundaryMemory_CBS.m` 和 `BuildBoundaryDataset_CBS.m` 是否真正保证了 chain-level 一致性；
- 这个怀疑是否有源码和图像证据支持；
- 如果支持，应该如何构造更统一的 chain-level BMem；
- 如果不支持，请指出更主要的问题在哪里。

### 设想 D：reference 只做调度，CGAN condition 仍应是目标空间条件

当前思路是：

```text
reference -> 选择哪个方向缺少边界点
BMem chain -> 在目标空间构造 QueryC
CGAN/GAN -> G(z, QueryC) 生成完整决策变量 x
```

我们暂时没有把 reference 单独作为 CGAN condition，因为同一个 reference 上可能存在多条边界，仅 reference 可能无法表达目标空间位置。

请评判：

- reference 是否应该只做调度；
- 是否应该把 reference、目标空间 condition、chain id 或其他信息一起作为 condition；
- 如果加入更多 condition，如何避免堆料和欠定；
- 如何验证生成点确实补到了未覆盖 reference 区域，而不是只复现已有训练区域。

### 设想 E：z 只做局部扰动，边界覆盖主要靠多 QueryC

当前思路是：未探索边界区域不应主要靠随机 `z` 扩散发现，而应靠 QueryC/reference 调度决定；为了生成窄边界，倾向于多 QueryC、每个 QueryC 少量或一个 `z`。

请评判：

- `z=0`、小 `z`、随机 `z` 对当前任务分别意味着什么；
- 多 QueryC 与多 z 的职责应如何划分；
- `queryPerCondition=1` 是否更符合“窄边界线”目标；
- 是否存在需要保留非零 z 的必要性。

### 设想 F：是否需要把 `f(G(z,c)) ~= c`、`CV(G(z,c)) <= 0` 放入训练目标

当前诊断已经关注：

- 训练条件重构的决策 RMSE；
- 训练条件目标一致性；
- 训练条件可行率；
- Query 条件目标一致性。

但当前 CGAN/GAN 生成点仍然可能不贴边、不贴训练分布、形成厚带或散点。

请评判：

- 当前 adversarial loss + reconstruction loss 是否足以约束生成器；
- 是否需要把 `f(G(z,c)) ~= c` 加入训练损失；
- 是否需要把 `CV(G(z,c)) <= 0` 加入训练损失；
- 如果加入，MATLAB/PlatEMO 框架下如何落地梯度或近似优化；
- 如果不加入，应该如何让生成器稳定学习 `c -> x` 的边界映射。

## 当前实现的客观事实

当前主线是：

- `Algorithms/Multi-objective optimization/CBS-CGAN/CBS_CGAN_FS.m`
- 它继承自 `CBS_CGAN.m`
- 当前版本含义是 feasible-side boundary support variant

当前 CGAN 调用流程来自 `CBS_CGAN.m`：

- 每代更新 `BMem = UpdateBoundaryMemory_CBS(...)`。
- 用 `BuildBoundaryDataset_CBS(...)` 从 `BMem` 构造 `TrainX, TrainC, QueryC`。
- 用 `BoundaryCGAN_CBS('train', GAN, TrainX, TrainC, Problem, GANOptions)` 训练网络。
- 用 `BoundaryCGAN_CBS('samplebycondition', GAN, QueryC, queryPerCondition, GANOptions)` 生成 `RawDec`。
- 用 `Problem.Evaluation(RawDec)` 得到真实目标值与约束值。
- 诊断记录 `train_dec_rmse`、`train_obj_dist50/90`、`train_feasible_rate`、`query_obj_dist50/90`、`feasible_rate`、`boundary_dist50/90`。

当前默认参数片段：

- `nGen = 20`
- `zDim = 2`
- `ganIter = 100`
- `queryPerCondition = 4`
- `reconstructionWeight = 100`
- `reconstructionHuberDelta = 0.10`

当前 feasible-side 分支已做过的修改：

- 原旧设计中 `y_b` 曾使用 F/I pair 中点。
- 当前分支改为 feasible-side endpoint：`y_b = y_f`，`x_b = x_f`。
- `TrainC` 是归一化后的 `BMem.y_b`。
- `TrainX` 是 `BMem.x_b`。
- `QueryC` 由 `BMem` chain 和缺失 reference 区域构造，使用同一套 objective condition scale。
- reference 当前用于调度缺失方向；CGAN 的条件仍是目标空间 condition。

请注意：上述只是当前实现事实，不代表这些实现一定正确。

## 当前实验与图片观察

最新主图目录：

- `Data/CBS_CGAN/boundary_quality_FS_qpc1_runs1_20260621_110631`

实验设置由 `run_summary.csv` 记录：

- 问题：`DASCMOP1_BC, DASCMOP2_BC, DASCMOP4_BC, DASCMOP5_BC, LIRCMOP5_BC, LIRCMOP6_BC, LIRCMOP7_BC, LIRCMOP8_BC, LIRCMOP9_BC, LIRCMOP10_BC`
- `runs = 1`
- `N = 100`
- `D = 30`
- `maxFE = 100000`
- 每个问题 5 张阶段图：`FE = 10000, 30000, 50000, 70000, 100000`
- 每张图显示目标空间可行域/不可行域、训练集、CGAN 生成点。

最新 runs=1 的 `stage_metrics_all.csv` 记录：

- 多数阶段 `raw_generated_count` 接近 `query_count` 或 `nGen` 限制。
- `DASCMOP1_BC` 后期 `feasible_rate` 可到 0.75，但部分中期明显低。
- `DASCMOP2_BC` 早期 `feasible_rate` 可为 1，但后期有阶段降到 0.15-0.25。
- `DASCMOP4_BC` 在该实验中多个阶段 `feasible_rate = 0`，但请不要把分析只集中到这个个别问题。
- LIRCMOP 图像中部分阶段红点比 DAS 更接近边界趋势，但仍存在离训练分布、局部厚带、跨段或散点问题。

诊断实验目录：

- `Data/CBS_CGAN/train_condition_diag_evalfix_runs3_20260621_140821`

该诊断记录：

- 诊断对象包括 `DASCMOP4_BC, DASCMOP5_BC, LIRCMOP9_BC, LIRCMOP10_BC`。
- `LIRCMOP*_BC` 诊断已修正为使用真实 `Problem.Evaluation`，不是绕过到基类 `CalObj/CalCon`。
- `train_count/query_count/bmem_count/chain_count` 在不同问题和 run 中变化明显。
- 存在 `empty_train` 情况，例如 `DASCMOP5_BC` 的部分 runs。

zip 中的当前主线图像显示的客观现象：

- 一些问题中，训练集可以沿目标空间边界或当前边界段排列。
- 一些问题中，训练集由多个离散段组成，而不是一条连续窄线。
- CGAN 生成点有时贴近训练段，有时明显偏离训练分布。
- CGAN 生成点在多张图中呈厚带、散点、局部簇或跨越不同边界段的弧形。
- 部分 LIR 图像相比 DAS 图像更接近边界趋势，但仍没有稳定形成窄边界线。

zip 中旧分支/对照图像仅作为历史参考：

- `ALL_PROBLEMS_FE_contact_GND_keep80_localMAD_weightedCond.png`
- `DASCMOP2_old_epoch50_vs_current.png`
- `LIRCMOP8_old_epoch50_vs_current.png`

这些图像用于说明：在旧分支和当前/历史对照中，也能看到训练集、边界段和 CGAN 生成点之间存在不一致、厚带、散点或跨段现象。

## 请重点回答的问题

请基于上传的源码、fix、实验结果和图片，严格分析并回答以下问题。

### 问题 1：当前候选设想哪些应保留、修改或废弃

请逐条评判设想 A-F：

- 是否成立；
- 证据是什么；
- 可能违反什么约束；
- 应保留、修改还是废弃；
- 如果修改，应如何改成更统一的主线。

### 问题 2：边界存档/训练集到底应该怎么构造

请分析：

- 当前 BMem/TrainC/TrainX 的构造是否是主要问题；
- 当前实现是否混合了不同边界段、不同质量层级或不同阶段前沿；
- 如果是，应如何在源码层面保证训练集是一条窄的当前边界/当前前沿；
- 如果不是，请指出真正导致 CGAN 生成厚带或散点的主要原因。

### 问题 3：CGAN/GAN 如何产生未探索 reference 区域的边界解

请分析：

- 当前 `QueryC` 构造是否足以支持这个目标；
- reference 调度、目标空间 condition、chain 信息、z 的职责如何划分；
- 生成点如何证明不是只复现已有训练点附近区域；
- 需要什么最小指标来验证“未探索 reference 区域生成价值”。

### 问题 4：为什么生成点不够贴边、不够窄

请分析：

- 是训练集问题、条件设计问题、网络结构问题、训练方式问题、损失函数问题，还是组合问题；
- 当前 adversarial loss + reconstruction loss 是否足够；
- 是否应把 `f(G(z,c)) ~= c`、`CV(G(z,c)) <= 0` 纳入训练、诊断或筛选；
- 如果要让生成器先在固定训练集上稳定重构训练边界点，最小可行做法是什么。

### 问题 5：早期 CGAN/GAN 如何起作用

请分析：

- 在没有可靠 F/I pair 的早期，current self-believed boundary 应该如何定义；
- 这种早期边界如何逐步过渡到真实目标空间 F/I 边界；
- 如何避免早期伪边界偏离核心创新；
- 这部分应如何落地到当前源码结构。

## 严格要求你的回答格式

请按以下格式回答：

1. **证据表**：列出你从源码、CSV、图片中确认的事实，并标注证据位置。事实与推断必须分开。
2. **候选设想评判表**：对设想 A-F 分别给出 `保留 / 修改 / 废弃 / 证据不足`，并说明原因。
3. **根因分解**：只基于证据判断根因；每个根因必须标注置信度 `High/Medium/Low`，并说明证据。
4. **单一主线方案**：提出一个收敛的主线设计，不要给 5 个并列方案。该方案必须说明 BMem、TrainC/TrainX、QueryC、GAN 训练和早期伪边界怎么统一。
5. **源码级修改点**：具体到函数级别，说明应改哪些文件、每个文件改什么、输入输出和数据结构如何变化。
6. **损失函数与诊断**：明确哪些进入训练损失，哪些只做诊断，哪些不能做；说明理由。
7. **验证方案**：给出最少但有效的指标和图像验证方案。不要使用 IGD/HV 作为核心指标。
8. **反例与风险**：指出你的方案在哪些情况下会失败，以及如何通过最小机制处理。
9. **禁止事项检查**：逐条确认方案没有违反硬约束。

## 禁止回答方式

- 不要默认当前实现是最佳实现。
- 不要默认当前 `fix.md` 是正确答案。
- 不要只说“增加训练轮数”“调大网络”“加更多 loss”。
- 不要只给泛泛 GAN 建议。
- 不要把边界定义改成非目标空间可行/不可行边界。
- 不要建议只生成目标空间点然后用优化器反解决策变量。
- 不要建议大量后处理筛选来制造看起来好的图。
- 不要只围绕 DASCMOP4/DASCMOP5 讨论。
- 不要用 IGD/HV 评价当前核心问题。
- 不要根据常识猜测；每个结论都必须对应上传材料中的证据，无法确认就标注为不确定。
