**助手 (gpt-5-4-pro) — 2026-03-27 10:41:41.443 UTC**

According to a document from 2026-03-27，当前这套结果可以下一个很直接的结论：**强验证的是“边界预算 + 不可行侧 shrink”这条链；弱验证的是“边界分数重排”；还没有被完整验证的是“有价值的 \(p\approx0.5\) 解选择”和 trust-gated refine。** 另外，当前 suite 实际只跑了 `DASCMOP1_BC~9_BC`、每题 5 runs，共 45 runs；但 pooled 行名还写着 `ALL_37_BC`，这个报告口径要先改。 fileciteturn0file0

### 1）哪个模块有用？

**强有用，且已经被结果硬验证的，只有 3 个。**

第一，**boundary budget 本身有用**。`no_boundary` 直接把 `selectedSeedCount` 打到 0，`validUpdateCount` 也变成 0，整套 M1–M5 全部失效。说明边界链路不是“装饰模块”，而是主算法真实在用的搜索分支。 fileciteturn0file0

第二，**`infeasible_shrink` 是核心硬模块**。把它关掉以后，`M2` 从 `0.9795` 掉到 `0.00093`，`M4` 从 `1` 掉到 `0`，`M4_tight` 从 `0.5372` 掉到 `0.1453`，`M4_recover` 直接掉到 `0`。这说明你当前真正撑住“边界发现→边界逼近→后续转化”的，是不可行侧 bracket shrink，不是别的。 fileciteturn0file0

第三，**finder 里的“三点扫描”有用，但 trust-refine 没证明有用**。`midpoint_only` 明显变差：`M1_selected=0.03477`、`M4_tight=0.3230`、`M5_boundaryGap=0.4490`；而 `probe3_only` 和主线几乎重合：`M1_selected=0.02873 vs 0.02880`，`M2=0.98033 vs 0.97953`，`M4_tight=0.53617 vs 0.53724`。这说明**真正起作用的是 3-probe placement，本地 trust refine 目前基本没起作用。** fileciteturn0file0

**弱有用，但还不能写成核心创新的，有 2 个。**

第一，**boundary 分数重排有一定作用**。`selector_random_shortlist` 比主线差：`M1_selected=0.04077`，`M4_tight=0.4691`；说明 shortlist 内不是随便挑，boundary score 确实有用。**但** `selector_boundary_only` 和主线几乎一样，说明现在真正被验证的是“boundary 重排”，不是“Pareto shortlist + boundary 联合价值”。 fileciteturn0file0

第二，**beta calibrator 只验证了“校准更好”，没验证“搜索更好”**。`boundarycore_beta` 的 `M5_boundaryEce/GAP` 明显好于 `raw`（`0.3403/0.3368` vs `0.4093/0.4092`），但 `raw` 的 `M1_selected` 反而更小（`0.00970` vs `0.02880`），`M2/M3` 也不差。结论只能写成：**beta 提升了概率校准，但还没证明它让 query 更有价值。** fileciteturn0file0

**目前没有通过验证、甚至处于失活状态的模块，有 2 个。**

第一，**trust-gated refine 没过验证**。`boundarycore_beta` 和 `probe3_only` 几乎一样，而 `boundarycore_no_trust` 之所以变了，是因为它同时设置了 `DisableTrust=true` 和 `ForcePlacementRefine=true`。这说明现在不是 “refine 无效”，而是**trust gate 把 refine 基本关死了**。我按 `variant_run_summary` 对齐 45 runs 看，主线 `finalTrustGate` 实际没有打开。 fileciteturn0file0

第二，**`feasible_forward` 不是当前主驱动**。把它关掉以后，`M3` 按定义归零，但 `M2` 基本不变（`0.97977` vs `0.97953`），`M3_downstream` 只小幅变化（`0.2155` vs `0.2227`），`M4_tight` 也几乎不变。说明它更像辅助技巧，不是当前主方法的核心贡献。 fileciteturn0file0

### 2）核心创新点是否验证了？

**结论：只验证了一半。**

你真正想证明的不是“找到了更靠近 \(0.5\) 的点”，而是**“找到了对 Pareto 搜索更有价值的 \(p\approx0.5\) 点”**。  
当前结果只把前半句基本做出来了：主线比 `selector_pareto_only`、`selector_random_shortlist` 的 `M1_selected` 更小，说明它**更靠近边界**。但后半句还没立住：  
- `selector_boundary_only` 几乎和主线一样，说明 **Pareto shortlist 这层没有被证明必要**；  
- `M2` 和 `M4` 在大部分变体里都接近饱和，区分度很差；  
- 当前 suite 没有直接统计“边界点带来的绝对下游收益”，也没有最终 `HV/IGD+` 这类终局性能指标，所以还不能说“主线确实找到了更有价值的边界点”。 fileciteturn0file0

更直白一点：**当前过的是 boundary-ness，不是 value。**  
所以论文题眼现在不能写成“valuable boundary query 已验证”，最多只能写成“boundary-local query mechanism 已初步验证”。 fileciteturn0file0

### 3）没过验证的模块，怎么让它通过验证？

#### A. trust 模块：先改算法，再改实验
这部分不是实验不够，而是**算法门槛太硬**。

把现在的二值 gate 改成连续 trust weight：
\[
T_t=\max\!\left(0,1-\frac{\mathrm{ECE}_{bd}}{\tau_E}\right)\cdot
\max\!\left(0,1-\frac{\mathrm{Gap}_{bd}}{\tau_G}\right)
\]
再用
\[
B_t^{refine}=\lfloor B_t\cdot T_t\rfloor
\quad\text{或}\quad
U'(x)=T_t\cdot U(x)
\]
来控制 refine，而不是“连续若干次通过才开闸”。  
同时把 `TrustAdmissionStreak` 从 3 降到 1，或者直接去掉；否则它会继续退化成 `probe3_only`。 fileciteturn0file0

实验上，加 3 个直接量：`gateOpenRate`、`refineUseCount`、`refineGain/seed`。  
**过关标准：** 主线必须满足 `gateOpenRate > 0`，且在下游收益上显著好于 `probe3_only`，否则 trust 不能算贡献。 fileciteturn0file0

#### B. selector / 核心 query：先改实验；若还不稳，再改算法
你这个创新本质上是一个**active query rule**。在主动学习里，uncertainty sampling 关注不确定样本，query-by-committee 关注最大分歧，而当概率本身参与 query 时，校准质量会直接影响 query 是否可信；beta calibration 也是专门面向二分类概率校准提出的方法。换句话说，你最终应该证明的是 **uncertainty + disagreement + calibrated probability** 能带来更多下游收益，而不只是更小的 \(|p-0.5|\)。 citeturn903232search0turn230844search2turn230844search4

所以实验上，不要再只看 `M1/M2/M3/M4`，要补 3 个**绝对收益**指标：
\[
UBY=\frac{\#\text{boundary-originated archive additions}}{\#\text{selected seeds}},
\]
\[
ABS=\#\text{boundary-originated archive additions / run},
\]
\[
AGS=\frac{\Delta HV_{\text{archive from boundary}}}{\#\text{selected seeds}}.
\]

然后固定同一条搜索轨迹，直接比较  
`boundarycore_beta` vs `selector_pareto_only` vs `selector_boundary_only` vs `selector_random_shortlist`。  
**过关标准：** 主线必须同时满足  
1）`M1_selected` 优于 `pareto_only` / `random_shortlist`；  
2）`UBY / ABS / AGS` 也优于它们。  
只有这样，才能说“你找到了**有价值**的 0.5 点”。 fileciteturn0file0

如果这样做完仍然不稳，就要改算法：把现在的“两阶段 shortlist”改成一个**单一 utility**：
\[
S(x)=Q(x)\,(1+\lambda D(x))\,(1+\eta G(x)),
\]
其中
\[
Q(x)=4p(x)(1-p(x)),
\quad
D(x)=\mathrm{Var}_k[p_k(x)],
\quad
G(x)=\max(0,g_s^\star-\hat g_s(x)).
\]
也就是：边界性、分歧、对当前 Pareto 搜索的预期收益，三者同时进分数。  
再在选中的 \(x^\*\) 周围做小半径主动采样：
\[
x'=x^\*+\sigma_t\epsilon,\quad \epsilon\sim\mathcal N(0,I),
\]
而不是只在线段上停一次。这样才真正接近你最初的想法。 fileciteturn0file0

#### C. feasible_forward：先改实验，不必先改算法
这部分当前最大问题不是“没有作用”，而是**证据太弱**。因为主线里 feasible seed 占比本来就很低，我按 `run_summary` 汇总，大约只占已选 seed 的 `2.7%`。所以单看条件成功率 `M3`，信息量不够。 fileciteturn0file0

你应该改成看绝对量：
\[
FFC=\#\text{forward improvements / run},
\qquad
FGY=\frac{\#\text{forward descendants entering archive}}{\#\text{forward local evals}}.
\]
如果这两个量仍然很小，就把 `feasible_forward` 从“创新点”降级成“工程增强项”；不要硬写成核心模块。 fileciteturn0file0

#### D. bridge 参数：先别当创新点，只当超参数
`BridgeTopK` 和 `BridgeActivationGap` 当前没有被干净证明。尤其 `bridge_topk_1` 有些条件成功率看起来更高，但这是因为 feasible seeds 数量塌得太厉害，分母变小了，不能直接说明它更好。  
所以这块先别写成创新，只保留为可调超参数；等你用 `ABS/AGS/HV` 这类绝对收益指标重新比完，再决定默认值。 fileciteturn0file0

### 4）最终创新点应该如何验证？

我建议你最后只做 **3 层验证**，别再加新模块了。

**第一层：query 机制验证。**  
比较 `boundarycore_beta / pareto_only / boundary_only / random_shortlist / probe3_only`。  
看 `M1_selected + UBY + ABS + AGS`。  
结论目标：**主线不但更靠近边界，而且绝对下游收益更高。** fileciteturn0file0

**第二层：refinement 机制验证。**  
比较 `mainline / feasible_forward_off / infeasible_shrink_off / no_trust(or always_refine)`。  
看 `ABS + AGS + tight-bracket absolute count + recover absolute count`。  
结论目标：**证明真正把边界点转成收益的，是 shrink；forward 只是辅助；trust 如果不开闸就不算贡献。** fileciteturn0file0

**第三层：最终算法效果验证。**  
这一步当前还没做。必须补。  
比较 `boundarycore_beta` vs `no_boundary` vs 合理基线（至少 CCMO / 你当前去掉 boundary 的版本），在**完整 BC 套件**上报最终 `HV / IGD+ / feasibility ratio / first-archive-entry FE`，做 paired Wilcoxon 或 Friedman。当前 suite 只有模块指标，没有终局优化指标，所以现在**不能**宣称“创新点已被最终验证”。 fileciteturn0file0

**一句话收口：**  
现在已经可以写进论文、而且站得住的，是：**“未知二值约束下，基于边界查询的搜索链路是有效的，其中不可行侧 shrink 是核心驱动。”**  
现在还不能写死的，是：**“我们已经验证了最有价值的 \(p\approx0.5\) 解选择。”**  
要把这句话写稳，你下一步不是再堆机制，而是把 **value 的实验指标** 和 **最终优化效果** 补齐。


---