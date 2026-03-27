综合审查后，我采纳这份评价的大方向，并做四个纠偏：

1. **核心创新点只保留一个**：`Pareto-relevant boundary query under binary unknown constraints`。这和你当前文档、主流程、实现口径是一致的。`P_C+P_U`、MLP/committee、calibration、archive 都是支撑件，不再单列创新点。`PRBCCMO` 的主流程也确实已经是“regular offspring → feasible anchor pool → boundary candidate → seed selection → label-aware refinement → model/buffer update → environment selection”。

2. **上一版分析里有两类“已实现却被写成未来建议”的内容，要纠正**：
   `trust admission + capped soft weight` 已经在代码里；`boundary-semantics-first` 的训练目标也已经在代码里；校准器也已经是在 held-out calibration buffer 上选一个写回模型，而不是运行时四路并行竞争。

3. **最新 bridge 筛选结论已经改写主线**。基于最新 `37 problems x 3 methods x 5 runs` 的 screening，`TopKPair` 在 `FDR` 主导、`FSG` 次级的筛选口径下应进入下一轮；`RandSectorPair` 保留为唯一强备选；`MainBridge` 暂时淘汰，不再进入下一轮模块实验。后续所有实验默认以 `TopKPair` 为 bridge 基准来分析 query、trust、refinement、migration 等其余模块；`RandSectorPair` 只在需要强结构对照时保留，`MainBridge` 只保留历史对照身份。

4. **migration 不能一刀切删掉**。它是当前“边界发现 → PopulationC/最终收益”的直接落地通道；正确做法不是删除，而是压缩成**最小必要实现**。另外，当前确实存在模式开关过多、算法平台化过强的问题，`selMode / localMode / pairMode / gateMode / calMode / trust*` 这些都不应继续污染论文主版本。主版本只保留一条固定主线，其他全部降为 ablation。

---

## 最终精简版算法：PRBCCMO-Lite

一句话概括：

**最简单的 shared-sector bridge 产出候选，主线固定 `midpoint placement + ParetoOnly`，把 boundary semantics 收缩到 `Full-v2` 实验分支，再用 label-aware refinement 把边界点转成可行收益，最后只保留最小 migration 把收益送回主搜索。**

## 当前代码进度（2026-03-26）

已经落地的部分：

* 主线 bridge 已固定为 `TopKPair + two-stage fallback`。
* 主线 probe placement 已固定为 `midpoint`，trusted bridge scan 默认关闭，不再在线改写候选坐标。
* 主线 query 已固定为 `ParetoOnly`；`Full-v2` 只保留为实验分支。
* 开发默认 calibrator 已缩到 `raw / beta`。
* `check_PRBCCMO_step2_query_trust.m` 已经补齐：
  * update-level `queryUpdateReport / QueryUpdateCsv`
  * `selection overlap` 与 `boundary score dispersion`
  * 自动 stop/go 闸门
  * 严格按公式使用固定 `B_t` 计算 `O_t`

当前真实状态：

* 脚本已经能自动阻止“`Full-v2` 还没分化，却继续做 oracle audit”这种脏证据链。
* 现有 smoke test 下，自动闸门已经会关闭 oracle audit，这说明当前更大的风险仍然是 `Full-v2 ≈ ParetoOnly`，而不是“只差把 runs 从 5 加到 30”。

### 1. 主版本只保留 5 个模块

#### 模块 A：常规搜索 + 可行锚点池

先照常做双群体 regular evolution，不把它当创新点。
定义 generation (t) 的可行锚点池：

[
A_t=\operatorname{Feasible}\big(P_C^t \cup O_{\text{reg}}^t \cup E_t\big)
]

其中 (E_t) 是外部可行档案。这个模块不改逻辑，只压口径：它只是给 boundary query 提供“可行侧参照”。

#### 模块 B：最小化 bridge generation

主版本下一轮固定为 **TopKPair + two-stage fallback**，不再保留 `current-pair` 主线。
对每个 shared sector (s)：

[
f_s=\arg\min_{x\in A_t\cap s}\phi_s(x),\qquad
u_s=\arg\min_{x\in U_t^{-}\cap s}\phi_s(x)
]

其中 (U_t^{-}) 是 `PopulationU` 中不可行解，(\phi_s(\cdot)) 是 sector scalar。
定义桥激活 margin：

[
m_s=\phi_s(f_s)-\phi_s(u_s)
]

主激活集合：

[
\mathcal S_t^{\text{strict}}={s:m_s>\Delta_g}
]

若 strict 为空，再启用一次弱回退：

[
\mathcal S_t=
\begin{cases}
\mathcal S_t^{\text{strict}}, & \mathcal S_t^{\text{strict}}\neq\emptyset\
{s:m_s>0}, & \text{otherwise}
\end{cases}
]

**每个 active sector 只保留 1 对 ((f_s,u_s))**。
删掉 `current_pair` 的主版本地位；`RandSectorPair` 只保留唯一强备选身份；`pairKeepM / pairLd` 不进入主线。这样做的原因很直接：最新 screening 已经把下一轮基线收敛到 `TopKPair`。

#### 模块 C：`ParetoOnly` 主线 + `Full-v2` 实验分支

主版本先把 online probe placement 也收回到最干净的中点版本，不再让模型在线改写候选坐标：

[
x_s^{mid}=\frac{f_s+u_s}{2}
]

也就是：主线运行时固定 midpoint placement，不再启用 trusted bridge scan。
原先的 `0.25/0.5/0.75` 三点 probe scan 只保留为实验分支，因为它虽然不直接决定最终排序，但仍会在线改动 candidate 坐标，容易污染 `ParetoOnly` 主线的归因。

主版本 query 不再用乘法软权重直接主导排序，而是先回退成最干净的 `ParetoOnly`：

[
\mathcal C_t^{\text{elig}}=\{x:\mathbf 1_{\text{elig}}(x)=1\}
]

[
\mathcal C_t^{\text{main}}=
Top_{B_{\text{seed}}}^{V}\bigl(\mathcal C_t^{\text{elig}}\bigr)
]

也就是：主线只按 Pareto relevance (V(x)) 排序，boundary semantics 先不再主导线上选择。

如果要继续验证 boundary semantics 是否真有额外排序力，则只保留一个实验分支：`Full-v2`，采用
**Pareto shortlist + boundary rerank**：

[
q(x)=\max\bigl(0,1-2|p(x)-0.5|\bigr)
]

[
B(x)=r(x),q(x),\bigl(1+\lambda_\sigma \sigma_p(x)\bigr)
]

[
\mathcal C_t^{pre}=Top_{\lceil \kappa B_t\rceil}^{V}\bigl(\mathcal C_t^{\text{elig}}\bigr),\qquad \kappa=3
]

[
\alpha_t=\min(T_t,\alpha_{\max}),\qquad \tau_T=0.10,\qquad \alpha_{\max}=0.50
]

[
U_{\text{Full-v2}}(x)=
\begin{cases}
\widetilde V(x), & T_t<\tau_T \\
(1-\alpha_t)\widetilde V(x)+\alpha_t\widetilde B(x), & x\in\mathcal C_t^{pre}
\end{cases}
]

其中 (\widetilde V,\widetilde B) 在 shortlist 内做 min-max normalization。
这样做的原因是直接的：当前乘法软权重在低 trust 区间容易塌缩回 `ParetoOnly`，所以主版本先诚实回退为 `ParetoOnly`，而把 `Full-v2` 留作单独实验分支去证明它是否真能改变选点。

其中：

* (p(x))：校准后的边界概率；
* (r(x))：reliability；
* (\sigma_p(x))：committee disagreement；
* (V(x))：Pareto relevance；
* (T_t)：trust weight；
* (\mathbf 1_{\text{elig}}(x))：是否落在 hard-negative 禁区外。

`Uncertain-only / HighProb-Boundary / Rand-Boundary` 全部只保留为对照。
下一轮主版本口径不再写成 “trusted query”，而是明确写成：
**`ParetoOnly` 是主线，`Full-v2` 只是待证明的实验分支。**

#### 模块 D：label-aware refinement

这一步保留，而且要按你当前真实实现来写，不另造新数学形式。

对 **feasible seed** (x)：

1. 先沿 `seed -> helper` 做一次 forward trial：
   [
   y_1=\operatorname{Forward}(x,u_s)
   ]
2. 若 (y_1) 变 infeasible，再做一次 midpoint 回退：
   [
   y_2=\frac{x+y_1}{2}
   ]

对 **infeasible seed** (x)：

1. 初始化 bracket：
   [
   (x^F,x^I)=(f_s,x)
   ]
2. 最多做 (K_b=3) 次 midpoint shrink：
   [
   x^m=\frac{x^F+x^I}{2}
   ]
   若 (x^m) 可行，则 (x^F\leftarrow x^m)，否则 (x^I\leftarrow x^m)。
3. 若 bracket 过程中找到可行端点，再补一次 forward trial：
   [
   y=\operatorname{Forward}(x^F,u_s)
   ]
4. 若 3 次 shrink 后仍没找到可行端，则做 hard-negative confirm，并把该区域写入 hard-negative archive。

这一步不要再拆 `label-aware / isotropic` 两条主版本；主版本固定 `label-aware`。`isotropic` 只保留给 ablation。

#### 模块 E：模型更新 + 最小 migration

训练模块**保留现有实现，不再重复发明**。
训练目标：

[
L=L_{\text{WBCE}}+\lambda_B L_{\text{Brier}}+\lambda_P L_{\text{pair}}+\lambda_M L_{\text{mid}}
]

其中

[
L_{\text{pair}}=
\mathbb E\big[\max(0,m-p(x^F)+p(x^I))\big]
]

[
L_{\text{mid}}=
\mathbb E\big[(p(x^M)-0.5)^2\big]
]

校准保持 held-out buffer 逻辑，但开发迭代默认只保留 `raw / beta` 两个候选；`temperature / auto` 暂时降级为实验分支，只在最终确认时再补跑。
trust 保持现有 admission 逻辑，但在论文主版本里只写成一条规则：

[
T_t^{raw}=
\max(0,1-\tfrac{ECE_t}{\tau_E})
\cdot
\max(0,1-\tfrac{CoreNearGap_t}{\tau_N})
]

[
\text{pass}*t=
\mathbf 1\big[ECE_t\le\tau_E,\ CoreNearGap_t\le\tau_N,\ |J_t|\ge n*{\min}\big]
]

若连续通过次数 (<K)，则

[
T_t=\min(T_t^{raw},T_{\max})
]

否则

[
T_t=T_t^{raw}
]

其中 (J_t={i:p_i\in[0.45,0.55]})。

migration 只保留**每个 sector 一个最优边界收益点**。
对 boundary-originated feasible 点 (y)：

[
\Delta_s(y)=\phi_s(c_s)-\phi_s(y)
]

其中 (c_s) 是当前 `PopulationC` 在 sector (s) 的 champion。
只保留 (\Delta_s(y)>0) 的点，且每个 sector 只迁入最优一个。
这条路径必须保留，因为它是 boundary discovery 真正变成最终收益的最短通道。

---

## 主版本明确删掉什么

* `MainBridge / current-pair`：**暂时删出主版本**，只保留历史对照身份。
* `RandSectorPair`：不进主线，只保留唯一强备选身份。
* `Full-v1` 式乘法主打分：**退出主版本**，只保留 `Full-v2` 实验分支。
* `uncertain-only / random-bridge / highprob-boundary / strict-only / isotropic`：**全部改成 ablation**。
* trusted bridge scan：**退出主版本**；主线固定 midpoint placement，`0.25/0.5/0.75` 三点 probe scan 只保留为实验分支。
* `traceOn / traceProbLabel`：只在实验脚本中开启，不进入方法主描述。
* `calMode`：方法里不再写成多分支结构；开发默认只保留 `raw / beta`，`temperature / auto` 降级到实验分支。
* 不再扩写 archive/migration 机制，只允许“每 sector 一个有效收益点”的最小实现。

---

## 分步骤验证：每个模块/功能是否合格

不要直接上总性能表。按下面 5 步走，每一步都设 **stop/go**。

### Step 0：工程合法性检查

目的：先保证后面实验的结论有效，而不是“日志没打通”。

要求：

* `D_train / D_cal / D_test` 严格分离；
* 每次模型更新都导出独立 `UpdateAudit(p_i,y_i)`；
* `CandidateAudit / BoundaryLineage / ArchiveEvent` 三类日志都完整；
* 统计单位必须是 **run**，不能把 pooled rows 当主结论。

**过关条件**：没有系统性 audit hole、没有日志缺口、没有数据泄漏。
这一步不过，后面全不算。

---

### Step 1：桥接模块验证（只验证 bridge，不验证 query）

目的：验证哪种 shared-sector pairing 应该进入下一轮，并据此固定后续模块实验的统一 bridge 基线。

设置：

* 问题集：先用 A 组最有信息量的 4 个问题：`DASCMOP5_BC / DASCMOP6_BC / LIRCMOP1_BC / MW1_BC`；
* `Population=100, MaxFE=200000, 30 paired runs`；
* 关闭 trust 影响，seed placement 固定中点；
* 暂时关闭 label-aware refinement，只看 bridge → seed 的直接质量。

比较：

* `MainBridge`: current-pair + weak fallback；
* `RandSectorPair`: shared-sector 内随机配对；
* `TopKPair`: top-k shared-sector 配对。

记录：

* `sharedSectorCount / activeSectorCount / gatePassedPairCount / candidateCount / selectedCount / feasibleDescendantSeedCount`；
* 首个 seed 出现代数 `FSG`；
* 若能离线拿 oracle，则对 selected seeds 计算 `d_B(x)`。

建议主指标：

[
SAR=\frac{#\text{active sectors}}{#\text{shared sectors}}
]

[
GPR=\frac{#\text{gate-passed pairs}}{#\text{positive-margin pairs}}
]

[
FDR=\frac{#\text{selected seeds yielding feasible descendants}}{#\text{selected seeds}}
]

**筛选决策**：若 `TopKPair` 在 `FDR` 主导、`FSG` 次级的口径下领先，则直接进入下一轮；`RandSectorPair` 保留为唯一强备选；`MainBridge` 暂时淘汰。
这一步的逻辑已经更新为：桥好不好，不能再只看 `PMR/GPR`，必须以 outcome 为主；下一步所有实验都统一以 `TopKPair` 为 bridge 基线。

---

### Step 2：query + trust 模块验证

目的：先证明 `Full-v2` 和 `ParetoOnly` 真的“选得不一样”，再验证它是否真的把 seeds 选得更靠近真实边界。

设置：

* 优先做 `9 个 DASCMOP_BC`；若能拿到 LIR/MW 的原始连续约束，再扩到全 `37 BC`；
* `Population=100, MaxFE=200000, 30 runs`；
* 固定 bridge = `TopKPair`；
* **所有变体共享同一 bridge generation、同一 boundary budget、同一主搜索框架**；
* runtime 主线固定为 `ParetoOnly`，`Full-v2` 只在离线重排中评估；
* 开发默认只跑 `raw / beta` 两个 calibrator；
* 暂时不看下游收益，只看“选点靠不靠边界”。

比较：

* `ParetoOnly`: 主线，只用 (V(x))；
* `Full-v2`: Pareto shortlist + trust-aware boundary rerank；
* `Uncertain-only`: 只用 (q(x))；
* `HighProb-Boundary`: 只偏向高 (p(x))；
* `Rand-Boundary`: eligible pool 随机选。

oracle 距离：

[
d_B(x)=\min_j \frac{|g_j^{raw}(x)|}{s_j+\epsilon}
]

其中 (g_j^{raw}) 是原始连续约束，(s_j) 用离线 50000 个均匀样本估计。

执行口径：

* **运行问题固定为 `DASCMOP_BC`**。Step 2 的 runtime benchmark 仍然是 binary-constraint 版本，不把原始 `DASCMOP` 直接拿来替代主实验问题。
* **`PRBCCMO` 算法本体不改**。实验中不允许把原始连续约束的 violation degree 或 CV 幅值喂回主搜索、训练、校准、筛选或环境选择；算法在线只使用当前 binary feasibility 语义。
* **原始 `DAS-CMOP` 连续约束只用于离线 oracle 审计**。它们只负责计算 `d_B(x)` 及其派生统计，用来回答“selected seeds 是否更靠近真实边界”，不参与算法决策。
* **不建议把运行问题直接切换成原始 `DASCMOP` 作为主证据**。那样虽然未必会改变当前代码的核心用法，但会把 `binary / unknown constraints` 的设定和 oracle 评估混在一起，导致实验口径变脏。若后续确实要做，也只能作为补充敏感性检查，不能替代这里的主实验。

主指标：

[
\overline d_B,\ \widetilde d_B,\ QP_\tau,\ \rho=\mathrm{Spearman}(U,-d_B)
]

其中

[
QP_\tau=\frac1n\sum_i \mathbf 1[d_B(x_i)\le \tau]
]

先加两个 stop/go 诊断量：

[
O_t=\frac{|S_t^{\text{Full-v2}}\cap S_t^{\text{Pareto}}|}{B_t}
]

[
D_t=\operatorname{std}\big(B(x)\big),\qquad x\in\mathcal C_t^{\text{elig}}
]

其中 (O_t) 是 selection overlap，(D_t) 是 eligible pool 上的 boundary score dispersion。

当前脚本实现状态已经前进一步：

* `O_t` 现在按固定 `B_t` 作为分母；
* 另有独立的“集合是否真的不同”统计，避免因为某次 update 没选满预算而误把 `O_t<1` 当成真分化；
* stop/go 已经是脚本里的自动闸门，而不是事后人工解释。

同时保留 trust 审计指标：

[
ECE,\ CoreNearGap,\ TWS,\ TGP
]

**过关条件**：

1. `beta` 相比 `raw`，run-level `ECE` 和 `CoreNearGap` 不恶化，且 `TWS/TGP` 至少不退步；`temperature / auto` 只作为补充审计，不再是开发默认。
2. `Full-v2` 必须先在相当一部分 problem/run/update 上让 `O_t` 明显低于 1；若 `median(O_t)\approx 1`，说明它仍然塌缩成 `ParetoOnly`，直接 stop。
3. `Full-v2` 的 `D_t` 不能长期接近 0；若 boundary term 本身没有分辨率，就没有继续做 oracle audit 的意义。
4. 只有在 2、3 成立后，才要求 `Full-v2` 显著优于 `Rand-Boundary / Uncertain-only / HighProb-Boundary` 的 `mean_dB / median_dB / QP_tau / rho`。
5. 若 `Full-v2 \le ParetoOnly`，或虽然数值不同但没有稳定收益，则主版本继续保持 `ParetoOnly`。

这一步不过，核心创新点就还没成立。

当前进度判断：

* Step 2 的脚本改造已经完成；
* 但正式 30-run 主实验还没启动；
* 在当前 smoke test 里，闸门会自动关闭 oracle audit，因此现阶段还不能把 `Full-v2` 当作已通过的 query 分支。

---

### Step 3：refinement 模块验证

目的：验证“找到边界点”之后，label-aware refinement 能不能把它们转成更有用的可行发现。

设置：

* 固定 bridge = `TopKPair`；
* 若 Step 2 通过，则固定 query = `Full-v2`；
* 若 Step 2 未通过，则 Step 3 不再继续验证 `Full-v2`，主线直接回退成 `ParetoOnly`；
* 只改 refinement。

比较：

* `LabelAware`：主版本；
* `No-local-label`：去掉标签感知；
* `Feasible-forward-only`：只保留 feasible 分支；
* `Infeasible-bracket-only`：只保留 infeasible 分支。

日志：

* `BoundaryLineage`：seed label、worker 类型、后继点 label、oracle_dB、bracket gap；
* 每个 seed 的 local eval 数、是否成功得到 feasible descendant。

主指标：

[
FRR=
\frac{#\text{infeasible seeds yielding feasible descendant}}
{#\text{infeasible seeds}}
]

[
TBR=
\frac{#\text{infeasible seeds with bracket gap}\le \varepsilon_b}
{#\text{infeasible seeds}}
]

[
FIR=
\frac{#\text{feasible seeds whose descendants improve sector scalar}}
{#\text{feasible seeds}}
]

并补一个效率指标：

[
LY=\frac{#\text{new feasible descendants}}{#\text{local evaluations}}
]

**过关条件**：`LabelAware` 必须显著优于 `No-local-label` 的 `FRR / TBR / FIR / LY`。
若只有 feasible 分支贡献明显，就删掉 infeasible 分支；反之亦然。
主版本不保留“两个分支都写上，但其中一个长期不贡献”的臃肿结构。

---

### Step 4：migration / downstream usefulness 验证

目的：验证这些 boundary discoveries 是否真的带来了**下游收益**，而不是只找到“更像边界”的点。

设置：

* 同 Step 3；
* 以当前通过 Step 2 的 query 主版本为基线；
* 加 `QueryMain-no-migration` 作为唯一结构对照。

日志：

* `ArchiveEvent`：每个 boundary-originated feasible point 首次进入 external archive 的 FE、event-level HV 贡献、是否进入主种群、存活代数。

主指标：

[
UBY=
\frac{#\text{boundary-originated feasible points first entering archive}}
{#\text{selected seeds}}
]

[
MSR=
\frac{#\text{boundary-originated feasible migrants surviving }H=10\text{ generations}}
{#\text{boundary-originated feasible migrants}}
]

[
\Delta HV_B=
\sum_{x\in\mathcal E}
\big(HV(A_{t_x^-}\cup{x})-HV(A_{t_x^-})\big)
]

[
TTU=\mathrm{median}\big(FE^{enter}(x)-FE^{seed}(x)\big)
]

这里 (\Delta HV_B) 必须按 **event-level** 统计，不能按 generation-batch 粗算。

**过关条件**：当前 query 主版本必须显著优于 `QueryMain-no-migration` 的 `UBY / MSR / ΔHV_B / TTU`。
若 Step 2 赢而 Step 4 不赢，说明你只是“更会找边界”，但还不会把边界变成收益；此时不能宣称创新闭环成立。

---

### Step 5：最终性能表

这一步最后做，不提前做。

设置：

* 至少 `9 个 DASCMOP_BC`，最好扩到全 `37 BC`；
* `Population=100, MaxFE=200000, 30 runs`；
* 内部比较：当前主版本（默认 `ParetoOnly`） vs `NoBoundary`；
* 若 `Full-v2` 在 Step 2 已通过，再把 `Full-v2` 作为额外实验分支并入最终表；
* 外部只加 2–3 个最强参考方法即可，外部表不是创新点主证据。

主指标：

[
HV,\ IGD,\ AUC\text{-}HV,\ FHT
]

统计：

* **run 为单位**；
* paired Wilcoxon signed-rank + Holm；
* 同时报 paired median difference 和 95% bootstrap CI。

**最终结论链必须完整写成**：

**`TopKPair` bridge 基线合格 → `ParetoOnly` 主线稳定 → `Full-v2` 若存在则先证明选点已分化 → selected seeds 更靠近真实边界 → refinement/migration 把它们转成有用可行发现 → 最终 HV/IGD/AUC 更好。**

这条链断任意一段，创新点就不能写成“已验证成立”。

---

## 最后的建议

你的主版本现在应该收缩为：

**`TopKPair shared-sector bridge`
→ `midpoint placement`
→ `ParetoOnly`（主线） / `Full-v2`（实验分支）
→ `Label-aware refinement`
→ `Minimal migration`
→ `Boundary-semantics model update`**

这是按最新 screening 结论收缩后的下一轮主线：`TopKPair` 进入下一轮，`RandSectorPair` 作为唯一强备选，`MainBridge` 暂时淘汰。
如果你愿意，我下一步可以直接把这套内容整理成论文里的“方法部分 + 实验部分”写法。
