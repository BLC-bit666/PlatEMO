在禁止代理模型后，我的判断更收敛：

**当前 CBS-CGAN 的最小改造，不是重写 BMem，也不是再加新学习器，而是先把现有模型收缩成一条确定性主线：**

**`C = [ref, y]`，`z = 0`，`Huber + pair-margin` 主导，`adversarial` 只做轻正则，`Problem.CalObj / Problem.Evaluation` 只做训练后验收、回滚和指标。**

这条结论直接来自当前源码和图，而不是泛泛经验：

* 当前 `BMem` 已经不是旧版“厚带/单侧回填”逻辑，而是 **pair-supported thin boundary memory**；训练目标 `x_b` 由 `y_f-y_i` 段附近的当前 feasible 样本选出。见 `source/new_CBS_CGAN/UpdateBoundaryMemory_CBS.m:1-3,90-183,310-367`。
* 当前 CGAN 也已经不是“只有 adversarial”。`generatorGradients` 里已经有 `advLoss + Huber reconstruction + pair margin`。见 `source/new_CBS_CGAN/BoundaryCGAN_CBS.m:238-255`。
* 但当前训练和正式采样都**默认随机 z**：训练时 `BoundaryCGAN_CBS.m:64,77,88` 用 `sigma*randn`，正式生成时 `sampleByCondition` 默认也是 `Options.sigma*randn`。见 `source/new_CBS_CGAN/BoundaryCGAN_CBS.m:158-170`。
* 图上已经分成两类问题：

  1. `DASCMOP4`、`DASCMOP5` 这类问题，连 `TrainC z=0 / TrainC fixed z` 都不能复现训练边界，例如
     `results/latest_visual_z/images/C_ref_y/DASCMOP4_BC_run1/figures/DASCMOP4_BC_run1_targetFE100000_visual_z_diagnostic.png`、
     `results/latest_visual_z/images/C_ref_y/DASCMOP5_BC_run1/figures/DASCMOP5_BC_run1_targetFE100000_visual_z_diagnostic.png`。
     这说明先要解决“**训练条件都学不住**”。
  2. `LIRCMOP6/7/8/10` 这类问题，`TrainC z=0/fixed z` 和 `QueryC z=0/fixed z` 多数已经贴边，但 `QueryC random z` 明显更散、更偏，例如
     `results/latest_visual_z/images/C_ref_y/LIRCMOP8_BC_run1/figures/LIRCMOP8_BC_run1_targetFE100000_visual_z_diagnostic.png`。
     这说明 random z 在当前阶段主要是“**吹散器**”。

下面只回答你这次问的最小改法。

---

### 1. 是否先固定 `z=0`，把目标收敛到稳定的 `G(C,0) -> x_b`？

**是。先这样做。**

理由很直接：

* 当前正式生成默认随机 z，代码在 `source/new_CBS_CGAN/BoundaryCGAN_CBS.m:168-170`。
* 但你最关心的不是“有多少随机多样性”，而是“能不能先稳定复现一条薄边界”。
* 从图上看，`LIRCMOP` 多个问题里 `z=0/fixed z` 明显比 `random z` 更贴边；而 `DASCMOP4/5` 里连 `z=0/fixed z` 都失败，说明**现在首先要把条件映射学稳**，不是放大 latent 变化。

**最小落地改法：**

* 训练阶段也把 latent 先固定成 0，而不是现在的 `sigma*randn`。直接改 `source/new_CBS_CGAN/BoundaryCGAN_CBS.m:64,77,88`。
* 正式生成阶段明确传 `sampleZ = zeros(n,zDim)`，不要再走默认随机 latent。可直接复用现有 `sampleByCondition` 对 `Options.sampleZ` 的支持，见 `source/new_CBS_CGAN/BoundaryCGAN_CBS.m:158-166`。
* `random z` 继续保留，但只留在 `visual_z_diagnostic` 里做诊断，不参与正式生成。

---

### 2. 是否默认使用 `C=[ref,y]`，暂时去掉 tau？

**是。主线先默认 `ref_y`。**

这不是凭感觉，而是当前代码和 CSV 已经给出支持：

* `BuildBoundaryDataset_CBS.m` 已经支持三种模式：`ref_tau`、`ref_y_tau`、`ref_y`。见 `source/new_CBS_CGAN/BuildBoundaryDataset_CBS.m:76-82`。
* `C_ref_y` 相比 `B_ref_y_tau`，在 `results/ablations/csv/ABC_K3_comparison_stage_metrics_by_problem.csv:12-31` 的 10 个问题里，`boundary_dist90` 的中位数 **9/10 更小**；只有 `LIRCMOP8_BC` 是例外。比如：

  * `DASCMOP2_BC`：`0.1102578838 -> 0.0415959215`，见 `:13` 与 `:23`
  * `DASCMOP4_BC`：`1.6378811690 -> 1.0142391904`，见 `:14` 与 `:24`
  * `LIRCMOP10_BC`：`0.0006867756 -> 0.0002031783`，见 `:16` 与 `:26`
  * 例外 `LIRCMOP8_BC`：`0.0145433640 -> 0.0198165422`，见 `:20` 与 `:30`

所以最小主线应先定成：

* **默认条件：`ref_y`**
* **tau 不删，但先退回 metadata / QueryMeta / BMem 里，不作为默认训练输入**

这样改最省，也最符合你“统一、减法、收敛”的要求。

---

### 3. 是否应先强化 Huber reconstruction，让 `TrainC fixed z` 能稳定重现训练边界？

**是，但不是新增重构项，而是把现有 Huber 从“并列项”改成“主项”。**

因为 Huber 已经在代码里了：

* `source/new_CBS_CGAN/BoundaryCGAN_CBS.m:244-247`

当前真正的问题不是“没有 Huber”，而是**现在的训练逻辑还在把随机 z 和 adversarial 放在太靠前的位置**。

我建议的最小原则是：

* `L_rec`（Huber）负责把 `G(C,0)` 钉到当前 `x_b`
* `L_pair`（现有 pair margin）继续保留，因为它是当前唯一明确的“靠 feasible 侧、远 infeasible 侧”的项，见 `BoundaryCGAN_CBS.m:248-253`
* `L_adv` 退到轻正则

也就是说，先别再发明新 loss；**先把已有 loss 的主次关系摆正。**

---

### 4. 是否应降低 adversarial loss 权重，避免 D/G 对抗在小数据下扰乱训练？

**是。**

当前代码里 `advLoss` 是 generator loss 的起点：

* `source/new_CBS_CGAN/BoundaryCGAN_CBS.m:241-243`

但它**没有独立权重**，相当于默认 1。与此同时，D 还做了 matched / mismatched / fake 三路判别，见 `source/new_CBS_CGAN/BoundaryCGAN_CBS.m:219-236`。

在你这类“小而窄的边界数据”任务里，如果 `TrainC fixed z` 都没学住，说明当前阶段最该优化的是**条件到 `x_b` 的稳定映射**，不是更强的分布对抗。

**最小改法：**

* 在 `BoundaryCGAN_CBS.m` 新增 `Options.advWeight`
* 把 `lossG = advLoss;` 改成
  `lossG = Options.advWeight * advLoss;`
* 保持 `Huber` 与 `pairMargin` 原有结构不变

我建议这里先只确定**相对顺序**，不在没有补实验前拍脑袋给死参数：

* `λ_rec >= λ_pair >> λ_adv`

具体数值属于**待验证**，但结构顺序已经很明确。

---

### 5. 是否让 `random z` 只作为诊断，不作为正式生成？

**是。**

而且当前代码已经天然支持这么做：

* 诊断图 6 个 panel 就是专门分开的，见
  `source/new_CBS_CGAN/Support/run_CBS_CGAN_boundary_quality_experiments.m:621-632`
* `CBS_CGAN.m` 里已经有 `visualSampleObjs(..., "zero")` 和 `visualSampleObjs(..., "fixed")`，见 `source/new_CBS_CGAN/CBS_CGAN.m:526-533,543-555`

所以最小改造不是“删 random z”，而是：

* **正式生成：固定 `z=0`**
* **诊断：保留 `random z / z=0 / fixed z` 三套图**

这样你仍能观察 latent 敏感性，但不会让生产分支继续被 random z 干扰。

---

### 6. 是否使用真实 `Problem.CalObj` / `Problem.Evaluation` 只做训练后验收、筛选、早停和指标记录，而不进入反向传播？

**是。这个方向最合适。**

而且当前代码已经接近这样：

* 正式生成后，`CBS_CGAN.m:186-233` 会对 `RawDec` 做 `Problem.Evaluation`
* 视觉诊断里，`CBS_CGAN.m:536-560` 已经会对 `TrainC/QueryC` 的 `z=0` 和 `fixed z` 采样结果调用 `Problem.CalObj`

所以最小改造不是把 `Problem.CalObj` 拉进反向传播，而是把它变成**训练后验收门**：

每次 GAN 重训后，直接算三组量：

1. `TrainC z=0 objective reconstruction error`
   比较 `Problem.CalObj(G(TrainC,0))` 与 `trainObjs`
2. `QueryC z=0 target error`
   比较 `Problem.CalObj(G(QueryC,0))` 与 `queryObjs`
3. `fixed-z feasible rate / side_rate`

如果这三组里关键指标比上一版 GAN 变差，就**回滚**到上一版 GAN，不接受这次重训。

这条路不需要任何第二学习器，完全可以继续放在当前 MATLAB 的自定义训练循环里：`dlnetwork` 支持自定义训练循环，自定义 loss 需要时可直接使用；`adamupdate` 也就是为 `dlnetwork` 的自定义训练循环更新参数而设计的。([MathWorks][1])

---

### 7. 如果 `TrainC fixed z` 都不能复现训练边界，应优先改训练流程、loss 权重、condition，还是 BMem？

**优先顺序应当是：`condition -> z -> loss 权重/训练流程 -> post-train gate -> 最后才看 BMem`。**

原因是当前 `BMem` 已经相对干净：

* `UpdateBoundaryMemory_CBS.m:1-3` 明确就是 `pair-supported thin boundary memory`
* `:90-183` 已经把 `x_b/x_f/x_i, y_b/y_f/y_i, tau, gap` 组织成当前配对边界行
* `:310-367` 已经做了 canonical 化和每 ref 去重

相反，当前更直接的证据是：

* `DASCMOP4_BC` 在 `C_ref_y_stage_metrics_all.csv:16` 已经有 `train_count=48`，但 `feasible_rate=0.05`
* `DASCMOP5_BC` 在 `C_ref_y_stage_metrics_all.csv:21` 已经有 `train_count=114`，但 `feasible_rate=0`

这不是“先没数据”的问题，而是“**有数据，TrainC/QueryC 仍学不住**”。

另外，也不建议先去调 `K` 或 `pair-count`：

* `results/ablations/csv/K3_K5_comparison_analysis_summary.csv:4,14` 中，`DASCMOP4_BC` 的 `train_count_med` 从 `54` 提到 `105`，但 `feasible_rate_med` 仍是 `0`
* `:5,15` 中，`DASCMOP5_BC` 的 `train_count_p90/max` 变大了，但 `feasible_rate_med` 仍是 `0`，`boundary_dist90_med` 还变差了

所以如果 `TrainC fixed z` 失败，**先别动 BMem，也别先堆数据量**。

---

### 8. 如果 `TrainC fixed z` 能复现训练边界，但 `QueryC fixed z` 失败，应优先改 QueryC 构造还是条件表达？

**先改 QueryC 的保守性，不先改 BMem，也不先把 tau 加回来。**

因为这时已经说明：

* `TrainC -> X` 这条映射是能学住的
* 失败主要来自 **train/query support gap**

当前 `BuildBoundaryDataset_CBS.m` 的 query 已经不是瞎造，而是基于相邻边界段生成 `missing_ref` / `large_gap`，并保存了 `source_interval / source_type / tau`。见 `source/new_CBS_CGAN/BuildBoundaryDataset_CBS.m:167-208`。

所以最小改造应是：

* **保持 `C=[ref,y]` 不变**
* **先收缩 QueryC，而不是扩 QueryC**
* 具体做法：只保留“由两端都有当前 finite-gap pair 支撑”的局部内插 query；对 `missing_ref` 更保守，对 `large_gap` 优先；必要时先临时关闭最激进的 `missing_ref` 查询

一句话就是：

> **TrainC 能重现而 QueryC 失败时，先缩 query，不先加条件维度。**

只有在这种收缩后仍然稳定失败，才值得再单独验证 tau 是否需要回到条件里。

---

## 最小改动清单

**1. `source/new_CBS_CGAN/CBS_CGAN.m`**

* 把默认 `conditionMode` 从 `"ref_tau"` 改成 `"ref_y"`
  位置：`CBS_CGAN.m:56-61`
* 正式生成前显式传入 `sampleZ=zeros(...)`，不要再让正式采样走随机 latent
  位置：`CBS_CGAN.m:179-181`
* 每次 GAN 重训后，复用现有 `visualSampleObjs` 路径做 `TrainC z=0` / `QueryC z=0` 的 objective error 和 feasible rate 验收；若变差则回滚
* 把 `emptyCBSMainDiag()` 里的默认 `condition_mode` 也改成 `"ref_y"`
  位置：`CBS_CGAN.m:282-321`

**2. `source/new_CBS_CGAN/BoundaryCGAN_CBS.m`**

* 增加 `advWeight`
* 让训练阶段支持 `trainZMode="zero"`，把 `:64,77,88` 的 `sigma*randn` 切到 `zeros`
* 保留现有 `Huber` 与 `pairMargin`，不要再加新 loss
* 正式生产采样继续通过已有 `Options.sampleZ` 走零 latent

**3. `source/new_CBS_CGAN/BuildBoundaryDataset_CBS.m`**

* 不重写数据集结构
* 只把主线默认切到 `ref_y`
* 若出现“TrainC 能过，QueryC 过不了”，再在 `buildExternalQueries` 里把 query 收紧到更保守的局部内插

**4. `source/new_CBS_CGAN/UpdateBoundaryMemory_CBS.m`**

* 第一轮最小改造**不动**
* 只有当 `ref_y + z=0 + Huber主导 + adv降权` 之后，`TrainC fixed z` 仍稳定失败，才回头查 BMem

---

## 不建议现在做的事

* 不建议再引入 `R:X→Y`、surrogate、learned objective model
* 不建议先改网络宽度、epoch、学习率
* 不建议先加 pair-count / K
* 不建议先把 tau 再塞回默认条件
* 不建议把 random z 继续当正式生成分支

最短总结：

**禁止代理模型后，当前最小可落地改法就是：先把 CBS-CGAN 训练成一个稳定的、确定性的条件重构器，再让 adversarial 只做轻正则；先证明 `TrainC z=0` 能重现训练边界，再谈 query 扩展。**

[1]: https://www.mathworks.com/help/deeplearning/ref/dlnetwork.html "dlnetwork - Deep learning neural network - MATLAB
"
