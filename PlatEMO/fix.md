先给结论：**你当前的主矛盾不是“边界创新没用”，而是“边界采样已经能找到局部可行条带，但 budget 没有足够均匀地铺到更多条带上”，所以 IGD 主要输在 coverage，而不是先输在局部收敛。**  
这和你的测试集特性是吻合的：DAS-CMOP 本身同时包含 feasibility / convergence / diversity 三类难度；LIR-CMOP 则专门用来刻画 **large infeasible regions + small feasible regions**，而 CMOP/BC 又进一步削弱了 CV guidance，因此算法很容易“先贴住某几段可行条带，再在这些条带附近变密，但其余条带缺口很大”。citeturn845809view0turn845809view1turn220401view0

另外，我先纠正一个判断：**按你上传的 3-30-2.xlsx 直接看均值，当前 PRBCCMO 并不是整体失效。**  
它在 DASCMOP_BC 上其实很强：DASCMOP1/2/3/6/9 都是当前三者里最优；真正短板集中在 DASCMOP4/5/7/8，以及几乎全部 LIRCMOP_BC（尤其对 MCCMO）。这说明你的核心思路在“边界可定位、且 CPF 与局部 bridge 对应关系较清楚”的场景里是有效的，问题主要出在**大不可行区 / 窄可行区 / 条带式 CPF**下的覆盖不足，而不是整个创新点方向错了。

## 1）图里的“缺口 + 不密集”，本质上是什么问题？

从图上看，点已经基本落在若干灰色可行条带上，说明**已访问条带内的收敛并不差**；真正拉高 IGD 的，是：

- **未访问条带太多**：前沿有明显缺口；
- **已访问条带内只保留了极少数点**：密度不够。

所以这**不是优先靠加一个局部搜索模块就能解决的问题**。  
局部搜索最多把“已找到的那几段”变密，**不能自动补齐没访问到的条带**。对 DASCMOP_BC / LIRCMOP_BC 这类问题，更关键的是**如何把边界采样 budget 分散到更多 sector / subregion**，而不是先在少数 sector 里做更深的 exploit。CCMO、EGDCMO、DRMCMO 这类工作之所以有效，也都强调了：面对 small feasible regions、large infeasible regions、narrow feasible areas，必须保住 **well-distributed infeasible/bridge exploration**，否则很容易卡在局部可行域。citeturn983893view0turn291736view0turn226524view1turn226524view3

## 2）你当前代码里，真正导致 IGD 偏大的点

你当前上传版的核心行为是：

- `BuildSingleBridgePool`：每个 sector 只取**单个最优** feasible anchor 和 infeasible helper；
- `Order`：按 `anchorScalar` 从小到大扫；
- `ExecuteBoundaryCore`：只要 budget 够，就继续 `coarse → refine → shrink`；
- migration：只看最终 `Seed / Shrink`；
- `TrimLabelArchive`：只做类别平衡，不优先保留 boundary samples。fileciteturn0file1

这会直接导致两个后果：

**第一，coverage 不足。**  
因为排序是“好 sector 优先”，budget 会优先砸在 already-good sectors 上；而 DASCMOP_BC / LIRCMOP_BC 这类问题恰恰最怕这个。你图里的“只覆盖几段条带”就是这个结果。fileciteturn0file1

**第二，boundary budget 被局部 exploit 吃掉。**  
你当前版即使没有真实 sign-flip bracket，也会继续做 local refine；shrink 也是对“最终选中的 infeasible seed”做 0.5 插值，而不是对“当前最紧的 feasible-infeasible bracket”做 midpoint。这样会让 boundary FE 花在局部 false uncertainty 上，而不是更多 sector 的真实边界上。fileciteturn0file1

你其实**已经有一版更接近正确方向的骨架**：那一版已经带了  
`coverage-first bridge scheduling + topK=5 local bridge pairing + bracket-only local refine + midpoint shrink + local-best migration`。这个骨架正好对应你现在的问题，不算加新模块，本质上只是把你自己更对路的一版拿回来。fileciteturn0file0turn0file3  
而最早的 PRBCCMO-Core 只有单桥 + 三探针 + shrink，coverage 更弱，不建议回退到那一版。fileciteturn0file2

## 3）把创新点改成更有学术价值的表述

你原来的说法“在概率 0.5 附近主动采样”方向是对的，但表述太口语，也容易被 reviewer 追问“为什么一定是 0.5？概率是否可靠？”。

我建议改成：

**“A sector-wise, bracket-constrained active boundary learning strategy for CMOP/BC.”**

更完整一点可以写成：

**在 objective-space 的 reference sectors 中，先构造 Pareto-relevant feasible–infeasible bridges；再利用 feasibility score 的不确定性，仅在 locally verified sign-flip brackets 内执行主动边界采样；新获得的真实可行/不可行标签持续更新 boundary discriminator；当边界附近发现能改善当前 sector champion 的 feasible point 时，将其迁移回 constrained population。**

这个表述比“0.5 采样”更强，原因有两个：

第一，**0.5 本身有理论依据**。  
主动学习里，二分类模型最有信息的点，就是 posterior 最接近 0.5 的点；这等价于“最靠近当前决策边界的点”。citeturn375756view0turn375756view1turn375756view3

第二，**但不能把 0.5 用成全局规则**。  
Settles 也指出，纯 uncertainty sampling 可能会选到“不代表整体分布的边界点/离群点”，所以更合理的做法不是“全局找 0.5”，而是**先构造局部可行–不可行 bracket，再在这个局部 bracket 内做 0.5 邻域 refine**。这正好就是你该强化的学术点。citeturn375756view2

另外，**不要再把 MLP 输出硬写成“真实概率”**。  
更稳妥的表述是 `feasibility score` 或 `boundary uncertainty score`。因为现代神经网络的概率往往并不天然校准；如果不做专门校准，把它当“排序分数”比当“真概率”更安全。citeturn227912search0turn227912search1

---

## 4）最有效、最小改动的修正方案

### 方案 A：恢复 coverage-first + topK local bridge pairing  
**这是最重要、最值的一刀。**

当前你最缺的不是“更深 refine”，而是“更多 sector 被真正摸到”。

对每个 sector \(s\)，先定义 sector 标量：

\[
g_s(x)=\sum_{m=1}^M w_{s,m}\,
\frac{f_m(x)-z_m^{\min}}{z_m^{\max}-z_m^{\min}+\varepsilon}
\]

其中 \(w_s\) 是 reference vector，\((z^{\min},z^{\max})\) 来自当前参考目标集。

然后在每个 sector 内，不再只取单个最优 feasible / infeasible，而是先取 topK：

\[
F_s^K=\operatorname{TopK}_K\{x\in \mathcal F \mid \mathrm{sec}(x)=s\}
\]
\[
U_s^K=\operatorname{TopK}_K\{x\in \mathcal U \mid \mathrm{sec}(x)=s\}
\]

推荐直接用你已有的 \(K=5\)。fileciteturn0file0turn0file3

bridge pair 选择改成：

\[
(a_s,u_s)=
\arg\min_{f\in F_s^K,\;u\in U_s^K,\; g_s(u)\le g_s(f)}
|g_s(f)-g_s(u)|
\]

如果不存在 \(g_s(u)\le g_s(f)\) 的组合，再退化成纯最小 gap。  
这个改动的目的很明确：**让桥更“局部穿边界”，而不是“从一个很强的 anchor 拉一条过长的线”。**

接着，把 sector 扫描顺序改成 coverage-first，而不是 anchorScalar-first。  
定义当前覆盖数：

\[
c_s=\left|\{x\in P_C^{F}\cup EA^{F}\mid \mathrm{sec}(x)=s\}\right|
\]

然后排序：

\[
\pi=\operatorname{sortrows}\big([c_s,\;g_s(a_s),\;s],[1,2,3]\big)
\]

即：**先扫覆盖最稀疏的 sector，再在同一 coverage 层内选 anchor 更优的。**

这一步与你的创新点是完全一致的，因为它不是在加新模块，而是在把“边界主动采样”真正变成 **coverage-aware active boundary learning**。  
而且这和文献共识一致：小可行域/窄可行域/多分离可行域下，需要通过 subregions / weight vectors 维持 well-distributed infeasible exploration，而不是只盯着少数已优 sector。citeturn291736view0turn983893view0turn226524view0

### 方案 B：把“0.5 邻域 refine”改成 **bracket-constrained refine**  
这是你创新点最该强化的地方。

先沿桥做三探针：

\[
\Lambda_c=\{0.25,0.5,0.75\}
\]
\[
x(\lambda)=(1-\lambda)x^F+\lambda x^I
\]

coarse probing 之后，只有当相邻 probe 的真实标签出现翻转时，才定义局部 bracket：

\[
[\lambda_L,\lambda_U],\qquad
y(\lambda_L)\neq y(\lambda_U)
\]

其中 \(y(\lambda)\in\{0,1\}\) 表示真实 feasible / infeasible 结果。

然后用 MLP 的边界不确定性在 bracket 内选 refine seed。  
建议直接用 **logit 绝对值**，而不是裸概率：

\[
u(x)=-|z(x)|
\]

其中 \(z(x)\) 是 MLP 的 pre-sigmoid logit；若拿不到 logit，再退回：

\[
u(x)=1-|2s(x)-1|
\]

其中 \(s(x)\in(0,1)\) 是 feasibility score。

然后 refine 点只在 bracket 内生成：

\[
\lambda^\*=\arg\max_{\lambda\in \{\lambda_L,\lambda_U\}\cup\Lambda_c} u(x(\lambda))
\]

\[
\Lambda_r=
\Big\{
\operatorname{clip}(\lambda^\*-\delta,\lambda_L,\lambda_U),
\operatorname{clip}(\lambda^\*+\delta,\lambda_L,\lambda_U)
\Big\},
\qquad \delta=0.125
\]

**没有真实 bracket，就不 refine。**

这一条比“全局在 0.5 附近 refine”更学术，也更有效。  
因为 uncertainty sampling 的理论前提是“靠近决策边界更有信息”，但它也会被不代表整体结构的边界点误导；所以你必须加上“locally verified bracket”这一层，才能让主动采样真正服务于边界拟合，而不是服务于噪声。citeturn375756view0turn375756view2

### 方案 C：shrink 只对最紧 bracket 做 midpoint  
当前最小改动是：

在本 sector 的所有已评估点中，找最近的一对 feasible / infeasible：

\[
(\lambda_F,\lambda_I)=
\arg\min_{y(\lambda_F)=1,\;y(\lambda_I)=0}
|\lambda_F-\lambda_I|
\]

然后只做一次 midpoint shrink：

\[
\lambda_{\text{mid}}=\frac{\lambda_F+\lambda_I}{2}
\]

再评估 \(x(\lambda_{\text{mid}})\)。

**如果没有真实 bracket，就不 shrink。**

这一步的意义是：把 shrink 从“对最后一个 infeasible seed 做 0.5 插值”，改成“对最可信的真实边界 bracket 做 midpoint”。  
这样既不加模块，也最符合你“边界拟合”的主线。fileciteturn0file0turn0file1

### 方案 D：migration 不只看最后一个 seed，要看“本地所有已评估 feasible 点”
这是一个非常小，但对 IGD 很有用的改动。

对 sector \(s\)，把本地所有已评估点合起来：

\[
\mathcal E_s=
\mathcal Q_s^{coarse}\cup
\mathcal Q_s^{refine}\cup
\mathcal Q_s^{shrink}
\]

然后选本地最优 feasible：

\[
x_s^{mig}=
\arg\min_{x\in\mathcal E_s,\;x\;\text{feasible}} g_s(x)
\]

若

\[
g_s(x_s^{mig})<g_s(a_s)
\]

则迁移到 \(P_C\)。

你当前版只看最终 `Seed / Shrink`，会白白浪费已经评估到的其他 feasible 点；而“把所有本地真实评估点都纳入 sector champion 替换”恰好强化了你的边界主动采样创新，因为它让**边界采样得到的真实标签，不仅训练 MLP，还直接改进 CPF coverage**。fileciteturn0file0turn0file1

### 方案 E：LabelArchive 改成 boundary-first  
当前 `TrimLabelArchive` 只按正负类别做平衡，**没有把 boundary-local samples 当成更高价值样本**。这会稀释 MLP 对真实边界的感知。fileciteturn0file1

建议改成：

设 archive 上限为 \(M\)，其中 boundary quota 为

\[
M_B=\lfloor 0.6M\rfloor,\qquad M_R=M-M_B
\]

保留策略：

\[
\mathcal A=
\operatorname{Recent}_{M_B/2}(\mathcal B^+)
\cup
\operatorname{Recent}_{M_B/2}(\mathcal B^-)
\cup
\operatorname{Recent}_{M_R/2}(\mathcal R^+)
\cup
\operatorname{Recent}_{M_R/2}(\mathcal R^-)
\]

其中：

- \(\mathcal B^+\)：boundary-local feasible 样本  
- \(\mathcal B^-\)：boundary-local infeasible 样本  
- \(\mathcal R^+\)、\(\mathcal R^-\)：非 boundary 的常规样本

也就是说：**先保留最近的边界样本，再补普通样本。**

这一步不改变框架，只是让你的单 MLP 真正变成“boundary learner”，而不是一般的 feasible/infeasible classifier。  
你当前代码已经正确地“每次重训重算 Mu/Sigma，只 warm-start 权重”，这一点是对的，保留即可。fileciteturn0file1turn0file0

---

## 5）参数怎么调，最稳

我建议先不要动 MLP 容量，先动边界预算与 pairing：

- `BridgeTopK = 5`  
- `RefineStep = 0.125`  
- `hidden = 20, epoch = 25, lr = 0.01` 先不动  
- `trainRho = 3`  
- `bRho` 统一先从 **0.25** 开始；如果 LIRCMOP_BC 仍然明显稀疏，再提到 **0.30**

这里要强调：  
**不要优先加局部搜索模块，不要优先加更大的 MLP，不要先上 calibration / trust / committee。**  
因为你现在的主矛盾不是模型能力，而是 **bridge coverage 与 boundary FE 的使用方式**。CMOP/BC 文献也说明了：在 binary constraints 下，真正决定成败的是如何在 feasible / infeasible 之间保持有效探索，而不是简单堆模型。citeturn220401view0turn640970view0

---

## 6）最终推荐：最小改动、最有效的一版

如果你只愿意做**一轮最小改动**，我建议就做这四件事，别多：

1. **把当前 `BuildSingleBridgePool` 改成 coverage-first + topK=5 local pairing**  
2. **把 current refine 改成 bracket-only 0.5-neighborhood refine**  
3. **把 shrink 改成 tight bracket midpoint shrink**  
4. **把 migration 改成“本地所有已评估 feasible 点里的 sector-best”**

这四条都属于**现有边界创新主线内部的修正**，不是加新模块。  
而且你其实已经有对应骨架了，最快路径就是：**以你那份 coverage-first / topK / bracket-only / local-best 的版本为主，替换当前上传版的 `BuildSingleBridgePool + ExecuteBoundaryCore`，其余框架尽量不动。** fileciteturn0file0turn0file1turn0file3

最后把创新点定成一句话：

**“We propose a sector-wise, bracket-constrained active boundary learning strategy for CMOP/BC, where uncertainty-guided queries are restricted to locally verified feasible–infeasible bridges, and newly discovered sector-improving feasible solutions are injected back to the constrained population.”**

这句话比“在 0.5 附近采样”强得多，也更能直接解释你为什么能降低 IGD。


---