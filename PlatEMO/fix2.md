According to a document from 2026-03-24，基于当前 PRBCCMO 工作树、代码、fix 文档和最新结果包，我的结论如下。 

**1：一句话回答**

创新点**未明显偏离**，但**尚未被验证成立**。 

**2：实验验证了什么？是否有说服力？揭露的问题是什么？**

它现在**已经验证了“模块能启动”**，但**还没有验证“创新成立”**。最新包里明确写了：旧的 2026-03-23 experiment-0 只能作为 pre-fix failure baseline，不代表当前代码；而当前代码在 200000 FE、N=100、Runs=1 的 activation-only 诊断里，三大家族全部出现 `boundary_started`，覆盖了 DASCMOP_BC 9/9、LIRCMOP_BC 14/14、MW_BC 14/14。这说明“边界模块不再普遍死亡”这件事基本坐实了。

它也验证了**当前瓶颈已经从“完全不启动”转成“trust/calibration 语义弱且家族依赖”**。最新 cross-family calibration 结果里，DASCMOP_BC 的 pooled ECE 仍在 0.326–0.366、core near-gap 在 0.373–0.391，所有变体的 meanTrustGatePassRate 都是 0；LIRCMOP_BC 明显最好，beta 的 pooled ECE 和 near-gap 最优，auto_trust 的 meanTrustGatePassRate 最高，约 0.187；MW_BC 介于两者之间，但仍明显偏弱，best beta 的 pooled ECE 也还在约 0.198，trust gate 通过率只有约 0.05。最新总结已经直接写明：当前证据只支持 activation coverage，不支持“trustworthy boundary querying and downstream usefulness”这个目标性主张。 

这些实验对**工程诊断**是有说服力的，因为它们已经把旧问题的根因拆开了：先是 feasible-anchor starvation，再是 hard-negative eligibility bug，最后剩下的是 shared sectors 已存在、但 `RawMargin > DeltaG` 经常达不到的桥接激活问题。最新文档还明确指出，这已经不是“全局无可行点”或“桥接模块完全坏掉”的问题，而是**shared sector 内部的配对/标量排序质量**问题。

但这些实验对**论文创新点**仍然**不够有说服力**。原因有四个：第一，当前关键 cross-family 结果很多还是 Runs=1 的 evidence collection，不是 publication-level validation；第二，当前 bundle 本身就是 modified but uncommitted working tree，复现实验前必须先冻结版本；第三，最新上下文已经明说“当前证据并不能证明 Pareto-relevant boundary scheduler 选得更好，也不能证明 seed 更靠近真实边界，更不能证明 downstream usefulness 足以支撑这套机制”；第四，创新验证还缺 oracle boundary audit、Rand/Uncertain-only/HighProb 的直接因果 ablation、以及 archive-entry / migration success / boundary-induced delta HV 这类 downstream usefulness 证据。 

**4：针对上述问题，算法应该怎样改进？**

我不建议改创新点，也不建议再往里加大机制栈。当前代码主线已经比旧版瘦很多：有 trust gate、reliability bins、label-aware refinement，而且 `ScreenBoundaryMigrants` 已经会把真实可行的 boundary discoveries 汇总进 `BoundaryFeasiblePool` 再筛成 `MigrationPool`，这说明早先“feasible seed 完全进不了主搜索”的大缺口在当前快照里基本已经补上了。现在该改的是**桥接配对质量**和**概率语义质量**，不是再发明新故事。

第一，**优先修桥接对构造，不要再盲目放松 DeltaG**。当前最新诊断已经把主阻塞点定位到 shared sector 内部的 `activation_gap_not_met`，而且文档明确写了“下一步不应该盲目继续放松 DeltaG”。我建议把每个 sector 的桥接从“单个 best feasible champion 对单个 best infeasible helper”改成“top-K feasible anchors × top-K infeasible helpers”的局部配对搜索。对 sector (s)，令 (F_s^K) 是该 sector 标量值最好的 (K) 个可行 anchor，(U_s^K) 是最好的 (K) 个 infeasible helpers，用下面的分数选桥接对：
[
(f_s^*,u_s^*)=\arg\max_{f\in F_s^K,\ u\in U_s^K}
\Big([g_s(f)-g_s(u)]_+ - \lambda_d|\hat x_f-\hat x_u|_2\Big),
]
其中 (\hat x) 是归一化决策向量。这样做的目的很直接：**不再只追求“同 sector 最优”，而是追求“同 sector 且真像一条可跨的桥”**。这正对应了当前文档里暴露出的 shared-sector pairing quality 问题。 

第二，**把 trust gate 从“硬开关”改成“软信任权重”**。现在 DAS 几乎全程 gate=0，LIR 有一部分可用，MW 只有极弱可用，这说明当前 family-dependent 语义更适合用**连续 trust**，而不是布尔开关。现代神经网络的概率本来就常失准，temperature scaling 是很强的简单基线，但并不是在所有 score distortion 和分布漂移下都够用；beta calibration 在某些二分类形态上比 logistic/platt 更稳，而单纯 post-hoc calibration 在 dataset shift 下常常仍会失效，ensembles 往往更稳。([Proceedings of Machine Learning Research][1])
我建议固定成：
[
T_t=\max!\left(0,1-\frac{\text{ECE}_t}{\tau_E}\right)\cdot
\max!\left(0,1-\frac{\text{NearGap}*t}{\tau_N}\right),
]
例如 (\tau_E=\tau_N=0.10)，再把当前 seed utility 改成
[
U_s(x)=\text{Eligible}(x),(\epsilon+V_s(x))
\Big[(1-T_t)+T_t,R(p_c(x)),q(x),(1+\lambda*\sigma s_p(x))\Big],
]
其中
[
q(x)=\max(0,1-2|p_c(x)-0.5|).
]
这样当 trust 很差时，算法自动退回 Pareto-driven selection；当 trust 变好时，再逐步放大 (p\approx0.5) 的权重，而不是硬切换。这个改法不改创新点，只是让创新点从“全-or-无”变成“可信则多用，不可信则少用”。 ([Proceedings of Machine Learning Research][1])

第三，**把训练完全改成“boundary semantics first”**。如果你还存在 train/calib/test 混用，必须拆成三套互不重叠缓冲：(D_{\text{train}},D_{\text{cal}},D_{\text{test}})。校准器只在 (D_{\text{cal}}) 上拟合，trust 只在 (D_{\text{test}}) 上评估；训练时过采样 tight brackets，而不是平均抽取历史点。建议固定损失为
[
L=L_{\text{WBCE}}+\lambda_B L_{\text{Brier}}+\lambda_P L_{\text{pair}}+\lambda_M L_{\text{mid}},
]
其中
[
L_{\text{pair}}=\frac1{|B|}\sum_{(x_f,x_i)\in B}\max(0,m-p(x_f)+p(x_i)),
]
[
L_{\text{mid}}=\frac1{|M|}\sum_{x_m\in M}(p(x_m)-0.5)^2,
]
且 midpoint 只来自足够 tight 的 bracket。这样做是因为你的论文核心不是“全局 feasibility classifier”，而是“边界附近的概率有没有语义”。现代校准文献和不确定性文献都支持你把重点放在 held-out calibration、dataset shift robustness 和 ensemble uncertainty 上。([Proceedings of Machine Learning Research][1])

第四，**保持打分极简，不要再加花活**。主动学习里，单纯 uncertainty 容易选到不具代表性的点，所以你需要保留 Pareto relevance；但除此之外，不必再加更多 heuristic。你现在真正该保留的只有三样：`eligibility × trust-corrected uncertainty × Pareto value`。也就是说，不要再回到“entropy / hvGain / novelty / penalty 大礼包”，更不要再做全局 component migration 这类会让首稿重度膨胀的机制。主动学习文献对“informativeness 需要配 representativeness”这点说得非常清楚，你当前的 (V_s(x)) 已经承担了这个角色，够用了。([NeurIPS 论文集][2])

**3：下一步的实验验证应该怎样设计（详细版）？**

先做一个前提：**先冻结一个 clean commit/tag，再跑正式实验**。因为当前 bundle 明确写了它是 modified but uncommitted working tree；不先冻结版本，后面所有 30-run 结果都不适合进论文。

我建议下一步实验分成 5 组，顺序不能乱。

### 实验 A：桥接配对与激活的修复性预实验

这组不是论文主结果，但必须先做。目标是验证你改的 pair constructor 确实解决 `activation_gap_not_met`，尤其是 DASCMOP5_BC。

设置：DASCMOP5_BC、DASCMOP6_BC，再加 LIRCMOP1_BC、MW1_BC 作对照；Population=100，MaxFE=50000，20 runs，paired seeds。比较 `CurrentPair` 与 `TopKPair` 两个版本。

日志：直接复用现有 `activationTrace / selectionTrace / bridgeTrace`。

指标定义：
[
SSR_r=\frac1T\sum_{t=1}^T \mathbf 1[\text{SharedSectorCount}*t>0],
]
[
ASR_r=\frac1T\sum*{t=1}^T \mathbf 1[\text{ActiveSectorCount}*t>0],
]
[
PMR_r=\frac1T\sum*{t=1}^T
\frac{#{\text{pairs with RawMargin}>0}}{\max(#{\text{shared pairs}},1)},
]
[
FSG_r=\min{t:\text{BoundarySeedCount}_t>0},
]
[
BSR_r=\mathbf 1[\exists t,\ \text{BoundarySeedCount}_t>0].
]

验收标准：`TopKPair` 在 D5 上显著提高 (ASR_r, PMR_r, BSR_r)，并降低 (FSG_r)，同时不损害 D6/LIR1/MW1。若这一步不过，就先别做创新验证。

### 实验 B：校准 / trust 审计

这组回答“当前 (p\approx0.5) 到底有没有可用语义”。它不是最终创新验证，但它是创新验证前提。

设置：全 37 个 BC 问题；Population=100，MaxFE=200000，30 runs；校准变体至少比较 `raw / temperature / beta / auto_trust`，并且只用 **auditReadyUpdates**。30 runs 和 PlatEMO 设置建议直接对齐 NA-EMT 的实验尺度。

数据收集：每次模型更新后，只在**独立 (D_{\text{test}})** 上记一次 `UpdateAudit`，记录 (p_i,y_i)、所选 calibrator、是否 audit-ready，绝不把 (D_{\text{test}}) 用回拟合。主统计单位必须是 **run**，不能再用 pooled all rows 做主结论。

指标：
[
\text{Brier}*{r,u}=\frac1n\sum*{i=1}^n (p_i-y_i)^2,
]
[
\text{ECE}*{r,u}=\sum*{b=1}^{10}\frac{|I_b|}{n}
\left|\frac1{|I_b|}\sum_{i\in I_b}y_i-\frac1{|I_b|}\sum_{i\in I_b}p_i\right|,
]
[
\text{CoreNearGap}*{r,u}=
\left|
\frac1{|J|}\sum*{i\in J}y_i-0.5
\right|,
\quad J={i:p_i\in[0.45,0.55]},
]
[
TGP_r=\frac1{U_r}\sum_{u=1}^{U_r}\mathbf 1[\text{ECE}*{r,u}\le\tau_E,\ \text{CoreNearGap}*{r,u}\le\tau_N,\ |J|\ge 20],
]
[
TWS_r=\frac1{U_r}\sum_{u=1}^{U_r}T_{r,u}.
]

主汇总：对每个 run 先取 update-level 中位数，再对 30 runs 做中位数和 95% bootstrap CI。
成功标准不是“所有家族都过 0.05 门槛”，而是：改进版在 DAS/LIR/MW 三家族上对 raw 均显著降低 run-level `ECE` 和 `CoreNearGap`，并显著提高 `TWS/TGP`。

### 实验 C：核心创新的 oracle boundary audit

这是论文里最关键的一组。目标是直接验证：**你的 scheduler 选到的 query 是否更靠近真实边界。**

设置：优先做 9 个 DASCMOP_BC；如果 LIR/MW 也能方便拿到原始连续约束，就扩展到全 37 个 BC 问题。Population=100，MaxFE=200000，30 runs。比较 4 个内部变体：

* `Full`: trusted + Pareto-relevant query
* `Uncertain-only`: 只看 (q(x)) / (p\approx0.5)
* `HighProb-Boundary`: 只偏向高 (p)
* `Rand-Boundary`: 在 eligible pool 里随机选

注意：这 4 个变体**必须共享相同 bridge generation、相同 boundary budget、相同主搜索框架**。这样才能把因果锁定在“query rule”。

离线 oracle 距离定义：
[
d_B(x)=\min_j \frac{|g_j^{raw}(x)|}{s_j+\epsilon},
]
其中 (g_j^{raw}) 是原始连续约束，(s_j) 由每个问题离线 50000 个均匀样本估计，例如取 median absolute value。

需要增加 `CandidateAudit`：对每个 active bridge candidate 记录
`run, problem, gen, sector, gateOn, p, s_p, reliability, paretoValue, utility, selected, dec`。
离线再补上 `oracle_dB`。

主指标：
[
\overline d_B^{(r)}=\frac1{n_r}\sum_{i=1}^{n_r} d_B(x_i),
\qquad
\widetilde d_B^{(r)}=\mathrm{median}*{i=1}^{n_r} d_B(x_i),
]
[
QP*\tau^{(r)}=
\frac1{n_r}\sum_{i=1}^{n_r}\mathbf 1[d_B(x_i)\le\tau],
\quad \tau\in{10^{-3},10^{-2},5\times10^{-2}},
]
[
\rho_r=\mathrm{Spearman}(U_i,-d_B(x_i)).
]

主假设：`Full` 必须显著优于 `Rand-Boundary / Uncertain-only / HighProb-Boundary` 的 `mean_dB, median_dB, QP_tau, rho`。
如果这一组不赢，创新点就不能成立。

### 实验 D：这些 boundary points 有没有 downstream usefulness

这组回答“靠近边界”是否真的转化成“更有用的可行发现”。

设置：同实验 C，但再加一个 `No-local-label` 变体，以拆分 query 与 refinement 的作用：

* `Full`
* `No-local-label`
* `Uncertain-only`
* `Rand-Boundary`
* `HighProb-Boundary`

需要增加两类日志：

1. `BoundaryLineage`：每个 seed 的 label、worker 类型、后继点 label、oracle_dB、bracket gap。
2. `ArchiveEvent`：每个 boundary-originated feasible point 首次进入 external archive 的 FE、事件贡献 HV、是否被主种群接纳、接纳后存活代数。

指标：
[
FRR_r=
\frac{#{\text{infeasible seeds yielding at least one feasible descendant}}}
{#{\text{infeasible seeds}}},
]
[
TBR_r=
\frac{#{\text{infeasible seeds with }\Delta_f\le\varepsilon_b}}
{#{\text{infeasible seeds}}},
\quad \varepsilon_b=0.03,
]
[
FIR_r=
\frac{#{\text{feasible seeds whose boundary outputs improve sector scalar}}}
{#{\text{feasible seeds}}},
]
[
UBY_r=
\frac{#{\text{boundary-originated feasible points first entering external archive}}}
{#{\text{selected seeds}}},
]
[
MSR_r=
\frac{#{\text{boundary-originated feasible migrants surviving after }H=10\text{ generations}}}
{#{\text{boundary-originated feasible migrants}}},
]
[
\Delta HV_B^{(r)}=
\sum_{x\in \mathcal E_r}
\Big(HV(A_{t_x^-}\cup{x})-HV(A_{t_x^-})\Big),
]
[
TTU_r=
\mathrm{median}_{x\in\mathcal E_r}\big(FE^{enter}(x)-FE^{seed}(x)\big).
]

关键点：(\Delta HV_B) 必须是**event-level**，不是 generation-batch。你旧结论里 `boundary_delta_hv = 0`，很大概率就是统计口径太粗导致的。

主假设：

* `Full > Uncertain-only / Rand / HighProb` 在 `UBY, MSR, ΔHV_B, TTU` 上成立；
* `Full > No-local-label` 在 `FRR, TBR, FIR, MSR` 上成立。

如果 C 组赢了而 D 组不赢，说明你只找到了“更靠边界的点”，但没证明这些点对多目标优化真有价值。

### 实验 E：最终优化效果

最后才接论文常规性能表。

设置：至少全 9 个 DASCMOP_BC，最好扩到全 37 个 BC；Population=100，MaxFE=200000，30 runs。内部比较用 `Full vs NoBoundary`，外部参考再加 NA-EMT 和你认为最强的 2–3 个 binary/unknown-constraint baselines。内部 ablation 才是创新验证主证据，外部 baselines 只做补充。NA-EMT 本身就是用 N=100、MaxFE=200000、30 runs、PlatEMO 跑的，所以这套设置是合理对齐的。

指标：
[
HV(A)=\mathrm{Leb}\left(\bigcup_{x\in A}[\tilde f_1(x),r_1]\times\cdots\times[\tilde f_M(x),r_M]\right),
]
[
IGD(A,R)=\frac1{|R|}\sum_{y\in R}\min_{x\in A}|\tilde f(x)-\tilde y|*2,
]
[
AUC\text{-}HV_r=\frac1K\sum*{k=1}^K HV_r(t_k),
]
[
FHT_r=\min{FE:HV_r(FE)\ge 0.9,HV^*}.
]

统计：以 **run 为单位**，paired Wilcoxon signed-rank + Holm；同时报告 paired median difference 和 95% bootstrap CI。
主结论链要写成：

**pairing 修复成功 → trust 指标改善 → selected seeds 更靠近真实边界 → 这些 boundary discoveries 更有 downstream usefulness → final HV/IGD/AUC 更好。**

这条链只要断一段，创新点就不能写“已验证成立”。

如果你愿意，我下一条可以直接把这套“下一步实验”改写成**可执行的 Matlab/PlatEMO 实验计划表**，包括每个脚本该加哪些日志字段、每个 CSV 应该长什么样。

[1]: https://proceedings.mlr.press/v70/guo17a.html "https://proceedings.mlr.press/v70/guo17a.html"
[2]: https://papers.nips.cc/paper/4176-active-learning-by-querying-informative-and-representative-examples "https://papers.nips.cc/paper/4176-active-learning-by-querying-informative-and-representative-examples"

