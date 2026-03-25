According to a document from 2026-03-25，基于当前 PRBCCMO 工作树、代码、fix 文档以及 2026-03-25 已完成的正式实验 A/B 结果，我的结论如下。 

**1：一句话回答**

截至 2026-03-25，创新点方向**未明显偏离**，但**A/B 正式实验仍不支持两个关键主张**：`TopKPair` 还没有改善整体 `activation outcome`，`(p\approx0.5)` 也还不能被视为“可放心依赖”的边界语义。 

**2：实验验证了什么？是否有说服力？揭露的问题是什么？**

它现在**已经验证了“模块能运行、能审计、能稳定暴露瓶颈”**，但**还没有验证“创新成立”**。和更早那批 Runs=1 的 evidence collection 相比，这次真正有说服力的变化在于：实验 A 已经用 `4 problems × 2 variants × 30 paired runs` 检查了 `TopKPair`，实验 B 已经用 `37 problems × 4 variants × 10 runs` 检查了校准与 trust 审计。也就是说，当前判断不再是基于 smoke 或单次诊断，而是基于正式批量结果。

它也验证了**当前主矛盾已经不再是“模块起不来”，而是“局部信号没有稳定传导成最终收益”**。具体说，实验 A 已经把 `TopKPair` 的断点定位到 `pair quality -> activation outcome`：`DASCMOP5/6` 仍然没有被激活，`LIRCMOP1` 基本无变化，`MW1` 虽然出现了 `PMR` 的局部改善，但没有转化成更好的 `ASR / BSR / FSG`，反而暴露出 `GPR / FDR` 下行。这说明当前 `TopKPair` 还只是一个局部 pairing quality 修补器，不是一个真正把桥接激活链路修通的模块。

实验 B 则把 `q(x)=\max(0,1-2|p(x)-0.5|)` 的断点定位到 `calibrated probability -> trustworthy boundary semantics`。这次最大的积极结果是：独立 held-out audit 基本已经打通，绝大多数 run-problem-variant 都有 audit-ready updates，因此“没有证据可审”这个工程性 hole 基本已经补上。但方法学上，结论仍然偏保守：`beta` 和 `auto_trust` 确实能改进一部分 run-level calibration 指标，可是这种改进并没有升级成强而稳定的 trust semantics。`DASCMOP_BC` 仍然几乎没有可调度的 `q(x)` 语义；`LIRCMOP_BC` 目前最好，但也只能算部分可用；`MW_BC` 介于两者之间，仍属弱语义。最新结果只支持“`(p\approx0.5)` 可部分利用”，不支持“`(p\approx0.5)` 可放心依赖”。 

这些实验对**工程诊断**是有说服力的，因为它们已经把问题从“模块死亡”推进到了“哪一段没有传导成功”这个层次。对 `TopKPair` 来说，问题不再是“有没有更多正 margin pair”，而是“新增的 pair 为什么没有变成更有用的 seed”；对 `p\approx0.5` 来说，问题不再是“有没有校准器能降 ECE”，而是“为什么校准后的概率仍然不等于可信的边界语义”。换句话说，当前代码已经能把症状暴露到机制层，而不是停在系统层报错或静默失败。

但这些实验对**论文创新点**仍然**不够有说服力**。原因现在也更清楚了：第一，实验 A 已经正式表明 `TopKPair` 没有通过 outcome 层验收，当前不能再把“pair quality 改善”当成“activation outcome 改善”；第二，实验 B 已经正式表明 `q(x)` 的语义仍然明显 family-dependent，当前不能把“calibration 变好”直接当成“trustworthy boundary semantics 成立”；第三，当前仍缺实验 C 的 oracle boundary audit 与实验 D 的 downstream usefulness 证据链，因此即便 B 达到了“可部分利用”，也仍然不能直接支撑核心创新主张。 

**4：针对上述问题，算法应该怎样改进？**

我不建议改创新点，也不建议再往里加大机制栈。当前代码主线已经比旧版瘦很多：有 trust gate、reliability bins、label-aware refinement，而且 `ScreenBoundaryMigrants` 已经会把真实可行的 boundary discoveries 汇总进 `BoundaryFeasiblePool` 再筛成 `MigrationPool`，这说明早先“feasible seed 完全进不了主搜索”的大缺口在当前快照里基本已经补上了。现在该改的是**桥接配对质量**和**概率语义质量**，不是再发明新故事。

用 2026-03-25 的正式实验 A/B 结果概括，当前算法改进要打通的不是更多模块，而是两条已经被实验证实存在的传导断点：

1. `better pair score -> better boundary seed / feasible descendant`
2. `better calibrated probability -> stronger trust admission / useful boundary query semantics`

前者对应 `TopKPair` 为什么没能从 `PMR` 走到 `ASR / BSR / FSG`，后者对应 `beta / auto_trust` 为什么没能从 `ECE / Brier` 走到 `TWS / TGP / trustworthy semantics`。这两个断点不打通，后面无论再加多少局部机制，都只会继续停留在“局部信号变好，但最终收益不稳定”。

第一，**优先修桥接对构造，不要再盲目放松 DeltaG**。当前最新诊断已经把主阻塞点定位到 shared sector 内部的 `activation_gap_not_met`，而且文档明确写了“下一步不应该盲目继续放松 DeltaG”。我建议把每个 sector 的桥接从“单个 best feasible champion 对单个 best infeasible helper”改成“top-K feasible anchors × top-K infeasible helpers”的局部配对搜索。对 sector (s)，令 (F_s^K) 是该 sector 标量值最好的 (K) 个可行 anchor，(U_s^K) 是最好的 (K) 个 infeasible helpers，用下面的分数选桥接对：
[
(f_s^*,u_s^*)=\arg\max_{f\in F_s^K,\ u\in U_s^K}
\Big([g_s(f)-g_s(u)]_+ - \lambda_d|\hat x_f-\hat x_u|_2\Big),
]
其中 (\hat x) 是归一化决策向量。这样做的目的很直接：**不再只追求“同 sector 最优”，而是追求“同 sector 且真像一条可跨的桥”**。这正对应了当前文档里暴露出的 shared-sector pairing quality 问题。 

如果目标不只是“提高 PMR”，而是**让 `TopKPair` 真正改善整体 `activation outcome`**，那么下一步不能继续只调 `K` 或只调 `\lambda_d`，而必须按下面清单推进：

2026-03-25 的正式实验 A 已经把这件事坐实了：当前结果正是“`PMR` 局部改善，但 `ASR / BSR / FSG` 没过线”的情形，所以以下改动不再是可选优化，而是下一步必须执行的主线修正。

1. 先把激活链路拆开记录为  
   `SharedSector -> PositiveMarginPair -> GatePass -> SelectedCandidate -> BoundarySeed -> FeasibleDescendant`。
2. 在现有 `activationTrace / selectionTrace / bridgeTrace` 基础上补四个 run-level 转化指标：
[
GPR_r=
\frac{\#\{\text{gate-passed positive-margin pairs}\}}
{\max(\#\{\text{positive-margin pairs}\},1)},
]
[
SPR_r=
\frac{\#\{\text{selected candidates}\}}
{\max(\#\{\text{gate-passed candidates}\},1)},
]
[
SeedPR_r=
\frac{\#\{\text{boundary seeds}\}}
{\max(\#\{\text{selected candidates}\},1)},
]
[
FDR_r=
\frac{\#\{\text{boundary seeds yielding feasible descendants}\}}
{\max(\#\{\text{boundary seeds}\},1)}.
]
3. 若下一轮结果再次出现 `PMR` 提升但 `ASR / BSR / FSG` 不动，就不要继续把注意力放在 `TopK` 本身，而要把配对打分从“只看 midpoint raw margin”改成“面向 pair-to-seed 转化”的代理目标。建议对每条桥接线段先做 3 点 probing：
[
x^\star_{f,u}=
\arg\max_{\alpha\in\{0.25,0.5,0.75\}}
q(\alpha x_f+(1-\alpha)x_u),
]
再用
[
S_{\text{pair}}(f,u)=
[m(f,u)]_+e^{-\lambda_d d(f,u)}(\epsilon+q(x^\star_{f,u}))
]
替代“只在单个 midpoint 上取最大分”的做法。
4. 每个 sector 不要过早压成单个 winner pair，先保留 `m=2` 或 `3` 个候选 pair，再让 gate / utility 去竞争；否则 pairing 排名误差会被放大成 seed-stage 的单点失败。
5. pairing 主线的 stop/go 条件必须提高到 outcome 层：只有当 `TopKPair` 相对 `CurrentPair` 在 `D5 / D6 / LIR1 / MW1` 上稳定改善 `ASR / BSR / FSG` 中至少两项时，才算“TopKPair 改善了整体 activation outcome”；如果只改善 `PMR`，则只说明 pair quality 变好，不能算主目标达成。

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

但若目标是让 **`(p\approx0.5)` 变成“可放心依赖”的调度语义**，仅仅看到 `ECE / Brier` 下降还不够；下一步必须按下面清单执行：

1. 先补 audit hole。只要仍存在整组问题长期 `auditReadyRuns = 0`，当前结论就只能写成“局部可用”，不能写成“可放心依赖”。
2. 给 trust 增加显式准入门槛，而不是只用连续权重。建议定义
[
\Gamma_t=
\mathbf 1[\text{ECE}_t\le\tau_E,\ 
\text{CoreNearGap}_t\le\tau_N,\ 
|J_t|\ge n_{\min}],
\quad
J_t=\{i:p_i\in[0.45,0.55]\},
]
并要求连续 `K_a` 次 update 满足 `\Gamma_t=1` 后，才允许把 `q(x)` 的权重上限提升到高权区；否则一律保持 soft fallback 到 Pareto-driven selection。
3. 在 `\Gamma_t` 达标前，`q(x)` 只能作为软因子，不能单独主导 query rule；也就是说，当前阶段只能说 “`p\approx0.5` 可部分利用”，不能直接说 “`p\approx0.5` 可放心依赖”。
4. 校准器选择要允许 family-aware 或 problem-aware，不必强行要求单一 calibrator 全局统一。当前结果如果继续表现为 `beta` 更擅长压低 `ECE / Brier`、`auto_trust` 更擅长改善 `CoreNearGap / TGP`，就应保留这种分工，而不是人为统一成一条校准路线。
5. “可放心依赖”的最终判据必须升级到三层同时成立：  
   `run-level calibration pass`、`oracle boundary audit pass`、`downstream usefulness pass`。  
   也就是说，只有当实验 B、C、D 同时过关时，才可以把 `(p\approx0.5)` 写成可依赖的边界语义。

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

如果后续继续新增正式实验，前提仍然是：**先冻结一个 clean commit/tag，再跑新的正式批次**。A/B 之前之所以强调这一点，是因为不冻结版本，后面的多 run 结果都不适合进论文；这条规则在后续 C/D/E 中同样成立。

截至 2026-03-25，正式实验 A/B 已经完成，因此“下一步实验”不再是从 A 重新开始，而应切换成下面这个顺序：

1. `当前主线`：先做实验 `C -> D`，直接检查 `q(x)` 与真实边界、以及 boundary discovery 的 downstream usefulness。
2. `结果收口`：只有当 `C` 和 `D` 都过关后，才进入实验 `E` 做最终优化效果表。
3. `诊断回归`：实验 `A` 只在 `pair score / sector keep-m / gate-selection linkage / local refinement` 这些 pair-to-seed 传导逻辑发生实质变化后才重跑。
4. `校准回归`：实验 `B` 只在 `train/calib/test` 划分、校准器集合、trust admission 规则或 query semantics 发生实质变化后才重跑。

### 实验 A：桥接配对与激活的修复性预实验

这组不是论文主结果，但它已经作为前置诊断完成。目标是验证你改的 pair constructor 是否真的解决 `activation_gap_not_met`，尤其是 DASCMOP5_BC。

当前已完成设置：DASCMOP5_BC、DASCMOP6_BC，再加 LIRCMOP1_BC、MW1_BC 作对照；`Population=100`，`MaxFE=50000`，`30 paired runs`。比较 `CurrentPair` 与 `TopKPair` 两个版本。

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

2026-03-25 实际结果更新：

1. 本轮 `30 paired runs` 的正式实验 A **没有通过**上述 stop/go 条件。
2. `DASCMOP5_BC / DASCMOP6_BC` 在 `CurrentPair` 与 `TopKPair` 下都没有形成有效 activation outcome，说明当前瓶颈并没有被 `TopKPair` 解除。
3. `LIRCMOP1_BC` 基本表现为平局，说明 `TopKPair` 没有形成稳定正收益。
4. `MW1_BC` 的结果最有信息量：`TopKPair` 的确提升了局部 `PMR`，但没有带来更好的 outcome，反而暴露出 `GPR` 明显下降、`ASR / SSR / FDR` 走弱。这说明当前 pair score 更容易选出“看起来像桥”的 pair，却没有提高“最终能产生有用 seed/feasible descendant”的概率。
5. 因而，实验 A 当前揭露的真正断点不是 “top-K 不够大” 或 “keep-m 不够多”，而是 **`pair quality -> activation outcome` 没有打通**。也就是说，当前难点在于 pair score 还没有对准 downstream usefulness，gate、selection、local refinement 也还不能把新增 pair 中真正有用的桥筛出来。
6. 结论上，`TopKPair` 目前只能保留为诊断性变体，不能进入主版本，更不能被写成“已修复 activation outcome”。

补充执行清单：

1. 这一轮实验的主结论不再允许只停留在 `PMR`；必须同时报告 `GPR / SPR / SeedPR / FDR`，明确 `TopKPair` 究竟卡在 gate 前、selection 后，还是 seed 后。
2. 若出现“`PMR` 上升但 `ASR / BSR / FSG` 不变”的结果，下一步应回到方法部分第一个清单，改 pair score 与 sector 内保留策略，而不是继续盲调 `K`。
3. 若 `TopKPair` 在 `MW1` 仅改善 `PMR`，但在 `D5 / D6` 不改善 `ASR / BSR / FSG`，则只能下“pair quality 局部改善”的结论，不能下“activation outcome 已修复”的结论。
4. 只有当 outcome 层指标过关后，才允许把 `TopKPair` 固化进主版本；否则它应继续作为诊断性变体保留。
5. 实验 A 后续不再作为默认重复项。只有当 `pair score`、`sector 内保留策略`、`gate-selection` 传导逻辑或 `local refinement` 的 pair-to-seed 链路被实质修改时，才值得重跑 A；如果只是继续扫 `K / \lambda_d`，原则上不再单独立项。
6. 下一次重跑 A 的目标也必须收紧为“能否打通 `pair quality -> activation outcome`”，而不再是“`PMR` 会不会再涨一点”。

### 实验 B：校准 / trust 审计

这组回答“当前 (p\approx0.5) 到底有没有可用语义”。它不是最终创新验证，但它是创新验证前提。

当前已完成设置：全 37 个 BC 问题；`Population=100`，`MaxFE=200000`，`10 runs`；校准变体比较 `raw / temperature / beta / auto_trust`，并且只用 **auditReadyUpdates**。若后续逻辑稳定且需要论文主证据，可再扩到 `30 runs` 与 NA-EMT 尺度对齐。

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

主汇总：对每个 run 先取 update-level 中位数，再对当前批次的 runs 做中位数和 95% bootstrap CI；若后续扩到论文版 `30 runs`，仍保持同一汇总口径。
成功标准不是“所有家族都过 0.05 门槛”，而是：改进版在 DAS/LIR/MW 三家族上对 raw 均显著降低 run-level `ECE` 和 `CoreNearGap`，并显著提高 `TWS/TGP`。

2026-03-25 实际结果更新：

1. 本轮 `37 problems × 4 variants × 10 runs` 的正式实验 B 在工程层面是**通过**的：独立 held-out audit 基本已经打通，`auditReadyRuns=0` 的系统性空洞已不再是主问题。
2. 但在方法层面，实验 B **没有通过**“`(p\approx0.5)` 可放心依赖”的验收。当前更准确的结论是：`q(x)` 已经不是纯噪声，但也还远远不是稳定可信的边界语义。
3. `DASCMOP_BC` 仍然是最难的一组。虽然 `beta / auto_trust` 能降低一部分 `ECE`，但 `TWS/TGP` 仍接近于零，说明这里的 `p\approx0.5` 还没有形成可调度语义。
4. `LIRCMOP_BC` 目前最好。`beta` 与 `auto_trust` 已经体现出部分可用语义，但这种语义仍不足以支撑“trustworthy”主张；它更适合作为 soft factor，而不是 query rule 的主导信号。
5. `MW_BC` 比 DAS 更好，但仍明显弱于 LIR。也就是说，`q(x)` 的语义强度仍然是 family-dependent 的，这正是当前最难的点。
6. 因而，实验 B 当前揭露的真正断点不是 “有没有校准器能降 ECE”，而是 **`calibrated probability -> trustworthy boundary semantics` 没有打通**。现在的难点在于：校准器确实改善了概率误差，但这种改善并没有稳定转化成强的 `TWS / TGP / trust admission`。
7. 结论上，B 只支持“`(p\approx0.5)` 可部分利用，因此允许进入实验 C/D 继续验证”；它**不支持**直接把 `p\approx0.5` 写成“可放心依赖”的边界语义。

补充执行清单：

1. 每次模型更新都必须导出独立 `D_{\text{test}}` 上的原始 `UpdateAudit(p_i,y_i)`；没有原始审计行时，只能做工程诊断，不能支撑“可放心依赖”的方法学结论。
2. 任何整组问题若仍长期 `auditReadyRuns = 0`，必须先定位并修复，再重新统计 B；否则只能写成“当前 `(p\approx0.5)` 在部分问题上有弱语义”，不能写成“整体可依赖”。
3. B 组结束后，要显式区分三种结论层级：  
   `不可用`：`ECE / CoreNearGap / TGP` 对 raw 无改进；  
   `可部分利用`：`ECE / CoreNearGap` 改进，但 `TGP / TWS` 或家族覆盖仍弱；  
   `可放心依赖`：`ECE / CoreNearGap / TGP / TWS` 全部稳定改进，且无系统性 audit hole。
4. 即使 B 组达到 “可部分利用”，也不能直接把 `(p\approx0.5)` 写成安全语义；还必须在实验 C 证明 `q(x)` 与 `-d_B(x)` 稳定相关，在实验 D 证明这种相关性确实转化成 `UBY / MSR / \Delta HV_B / TTU` 的 downstream gain。
5. 因此，B 组的真正 stop/go 条件应写成：  
   `B 过关 -> 允许进入 C/D`；  
   `C 和 D 也过关 -> 才允许宣称 (p\approx0.5) 可放心依赖`。
6. 实验 B 后续也不再默认重复。只有当 `train/calib/test` 划分、校准器集合、trust admission、query rule 语义或 held-out audit 机制发生实质变化时，才值得重跑 B。
7. 下一次重跑 B 的目标，不应再只是继续压低 `ECE / Brier`，而必须看这些改动能否把当前 family-dependent 的弱语义提升成更稳定的 `TWS / TGP / trust admission`。

### 实验 C：核心创新的 oracle boundary audit

这是论文里最关键的一组，也是当前 A/B 完成之后的**第一优先级主线**。目标是直接验证：**你的 scheduler 选到的 query 是否更靠近真实边界。**

A/B 到这里已经把前置问题压缩得很明确了：A 说明 `TopKPair` 还没有把好 pair 变成好 outcome，B 说明 `p\approx0.5` 还没有自动变成可信语义。C 的任务就是直接回答“当前 `Full` query rule 选出来的点，到底有没有更靠近真实边界”，否则 B 里所有关于 `q(x)` 的讨论都还停留在 proxy 层。

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

这组必须紧跟在 C 之后。它回答“靠近边界”是否真的转化成“更有用的可行发现”。

原因很直接：即使 C 赢了，也只能说明 scheduler 更会找边界；只有 D 再赢，才能说明这种 proximity 确实变成了 boundary discovery 的实际收益。如果 C 赢而 D 不赢，那么当前主问题就会从 “找不准边界” 转成 “找到了边界，但这些边界点并没有被正确转化为有价值的搜索收益”。

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

这一步现在不应提前做。只有当 C 和 D 都过关后，E 才有资格作为论文常规性能表；否则它最多只能记录现象，不能作为创新点成立的主证据。

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
