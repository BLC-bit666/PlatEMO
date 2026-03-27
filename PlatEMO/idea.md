## 2）怎么利用 MLP 输出找到边界点：我建议的最终做法

### 核心判断

**不要在全空间里找 (p\approx 0.5) 的点。**
要在**已经被证明“有优化意义”的 feasible–infeasible bridge 上**找 (p\approx 0.5) 的点。

原因很简单：

* 主动学习里，二分类最经典的 query 原则就是**不确定性采样**；对概率分类器而言，二分类的不确定性最大点就在 (p=0.5) 附近。Lewis 与 Gale 最早的 uncertainty sampling 就是这个思路。([arXiv][1])
* 但你这里不是普通分类，你是**约束多目标优化**。所以“边界点”必须同时满足两件事：
  1）它像边界；
  2）它挂在一个**有 Pareto 价值的桥**上。
* 对未知约束，文献里常用的也是**可行性概率**来建模 feasibility，而不是去猜 violation degree。约束 BO 里就是这么处理 unknown constraints 的。([arXiv][2])

### 最终推荐：**桥上找边界，不在全局找边界**

保留你当前已经证实比较稳的 `TopKPair` 共享 sector 配对。对每个 active sector (s)，已有一对真实评估过的：

[
f_s \in \text{feasible}, \qquad u_s \in \text{infeasible}
]

并定义桥上线段：

[
x_s(\lambda)=f_s+\lambda (u_s-f_s), \qquad \lambda\in[0,1]
]

然后让 MLP 做的事情只剩一件：

[
\lambda_s^\star=\arg\min_{\lambda\in[0,1]}
\left|p_{\text{cal}}!\left(x_s(\lambda)\right)-0.5\right|
]

[
x_s^\star=x_s(\lambda_s^\star)
]

这里 (p_{\text{cal}}(x)) 是**校准后的可行概率**，不是原始 logits。之所以要校准，是因为深模型原始概率往往不可靠；而 beta calibration 对二分类、偏斜分数更稳，且比普通 logistic calibration 更不容易把原本已校准的输出“校坏”。([Proceedings of Machine Learning Research][3])

### 但你现在不该上“连续 root-finding 主线”

因为你现在的实验状态说明：`Full` 分支**还没通过 trust/audit**，所以不适合让模型大幅在线改写 candidate 坐标。更合适的是一个**极简的、低风险的桥上线搜索**：

先做 3 点 probe：

[
\Lambda_0={0.25,\ 0.5,\ 0.75}
]

[
\lambda_s^{(0)}=
\arg\min_{\lambda\in\Lambda_0}
\left|p_{\text{cal}}(x_s(\lambda))-0.5\right|
]

如果当前 boundary-local trust 通过，再补 **1 次局部细化**：

[
\Lambda_1=
\left{
\max(0,\lambda_s^{(0)}-0.125),\
\lambda_s^{(0)},\
\min(1,\lambda_s^{(0)}+0.125)
\right}
]

[
\lambda_s^\star=
\arg\min_{\lambda\in\Lambda_1}
\left|p_{\text{cal}}(x_s(\lambda))-0.5\right|
]

否则直接取 (\lambda_s^\star=\lambda_s^{(0)})。

这套机制的优点是：

* **完全贴合你的创新点**：就是在桥上找 (p\approx0.5)；
* **不重**：最多 3 到 6 次模型前向，不增加真实评估；
* **不依赖过强 trust**：trust 弱时直接退回 3 点 probe；
* **和你当前代码/实验兼容**：保留 `TopKPair`，保留主线的稳健性，只把 midpoint 升级成“轻量条件边界定位”。 


## 3）找到“点”后，怎样设计最本质、最简洁的选择器？

### 我建议：**两阶段选择器**，不要做一个大杂烩分数

你现在最容易犯的错，是把 Pareto、boundary、trust、disagreement、novelty 全乘/全加进去，最后又回到“权重汤”。

真正本质的选择器应该只回答两个问题：

1. **这个桥值不值得查？**
2. **在值得查的桥里，哪个点最像边界？**

所以选择器做成两阶段最干净。

### 阶段 A：Pareto 预筛

对每个桥 (s)，只用**可行侧**的已知信息计算一个 utility proxy。
沿用你现有的 sector 标量化思想即可：

[
V_s=\max\bigl(0,\ \phi_s(c_s)-\phi_s(f_s)\bigr)
]

其中 (c_s) 是当前该 sector 的 constrained champion，(f_s) 是桥上的 feasible 端。
先按 (V_s) 排序，取 top-(\kappa B) 个桥进入 shortlist。这里 (B) 是 boundary budget，(\kappa) 取 2 或 3 就够了。

**解释：**
这一步不是边界模块的创新；它只是确保你查的边界不是“没优化价值的边界”。

### 阶段 B：boundary 重排

对 shortlist 中每个桥，只看边界性：

[
B_s = 1 - 2\left|p_{\text{cal}}(x_s^\star)-0.5\right|
]

它天然落在 ([0,1]) 内；越接近 1，越靠近预测边界。

可选 tie-break：

[
D_s=\operatorname{Var}_k(p_k(x_s^\star))
]

最终排序规则：

* 先按 (V_s) 做 shortlist；
* 再按 (B_s) 降序；
* 若相同，再按 (D_s) 降序。

### 为什么这个选择器最接近“边界价值”？

因为你这里的“边界价值”本质上不是单一概念，而是：

[
\text{Boundary Value}
=====================

\text{Pareto Relevance}
+
\text{Boundary-ness}
]

但这两者不该混成一个连续加权大分数。
更好的表达是**层级关系**：

* 先保证“有价值”；
* 再在有价值的桥里找“最像边界”的点。

这比 `Full-v2` 那类混合打分更容易解释，也更容易做消融。你的当前结果也说明：`ParetoOnly` 主线是稳定的，而 query 分支尚未通过验收，所以现在最合理的架构不是“让模型全权排序”，而是“让模型只在 Pareto shortlist 内决定谁更像边界”。 

---

## 4）真实评估后怎么利用：可行解怎样做，不可行解怎样做？

这里我建议你把逻辑压到**两个动作**，足够了。

### A. 若 (x_s^\star) 真实评估后是可行解

做两件事：

**动作 1：先用于优化**
若它改进该 sector 的 champion，就直接进入 constrained population / external archive：

[
\phi_s(x_s^\star) < \phi_s(c_s)
\quad \Rightarrow \quad
c_s \leftarrow x_s^\star
]

**动作 2：再做一次极简前推**
沿着“远离 infeasible 端”的方向再试一次：

[
z = x_s^\star + \alpha (x_s^\star-u_s),\qquad \alpha\in(0,1)
]

只做 **1 次**额外真实评估。
若 (z) 仍可行且更优，则保留 (z)；否则保留 (x_s^\star)。

**解释：**
可行边界点最有价值的地方，不是停在边界上，而是它很可能就在**新可行域入口**。所以可行解的第一使命是“转收益”。

### B. 若 (x_s^\star) 真实评估后是不可行解

做两件事：

**动作 1：先用于缩 bracket**
把原桥 ((f_s,u_s)) 缩成更紧的局部 bracket：

[
(f_s,\ u_s)\ \rightarrow\ (f_s,\ x_s^\star)
]

因为 (x_s^\star) 比原来的 (u_s) 更靠近预测边界。

**动作 2：只补 1 次 midpoint shrink**
再测一次：

[
m=\frac{f_s+x_s^\star}{2}
]

* 若 (m) 可行：新 bracket 记为 ((m,\ x_s^\star))；
* 若 (m) 不可行：新 bracket 记为 ((f_s,\ m))。

同时把不可行点记为 hard negative。

**解释：**
不可行点的使命不是“优化”，而是“定位边界”。
所以不可行解不要乱交叉、不要乱迁移，**只做 bracket tightening** 就够了。

### 这一步和你当前 label-aware refinement 的关系

本质一致，但我建议你把它压缩成：

* **可行分支：1 次前推**
* **不可行分支：1 次收缩**

不要保留更多分支，除非后续消融证明它们真的长期贡献。你自己的文档也已经明确：若 Step 3 发现某一分支长期不贡献，主版本就该删掉，不保留臃肿结构。

---

## 5）模型怎样更新？

### 我建议的最终更新方式：**分类 + 配对 + 中点**

只保留 3 类监督。

### 数据构成

每轮新增训练集：

[
\mathcal D_t^{new}
==================

\mathcal D_t^{sel}
\cup
\mathcal D_t^{bracket}
\cup
\mathcal D_t^{hardneg}
]

其中：

* (\mathcal D_t^{sel})：本轮选中并真实评估的边界点；
* (\mathcal D_t^{bracket})：tight bracket 的 feasible/infeasible 端点与 midpoint；
* (\mathcal D_t^{hardneg})：确认过的不可行 hard negatives。

再加一个小 replay buffer：

[
\mathcal D_t = \mathcal D_t^{new} \cup \mathcal D_t^{replay}
]

这样能防止模型只记住最近那一小圈边界。

### 损失函数

#### 1. 基础分类损失

[
L_{\text{cls}}
==============

-\Bigl[
y\log p(x)+(1-y)\log(1-p(x))
\Bigr]
]

#### 2. 配对顺序损失

对每个 tight bracket 中的 feasible / infeasible 对 ((f,i))：

[
L_{\text{pair}}
===============

\max\bigl(0,\ \mu - p(f) + p(i)\bigr)
]

它要求模型学会：

[
p(f) > p(i)
]

#### 3. 中点 0.5 损失

对 tight bracket 的 midpoint (m)：

[
L_{\text{mid}}
==============

\bigl(p(m)-0.5\bigr)^2
]

#### 总损失

[
L
=

L_{\text{cls}}
+\lambda_1 L_{\text{pair}}
+\lambda_2 L_{\text{mid}}
]

这和你当前代码里 `lPair / lMid` 的方向是完全一致的，但口径更清楚：
**模型学的不是“违反程度”，而是“边界语义”。**


### 校准：用 beta，不要再扩 calibrator 菜单

训练完后，对 held-out buffer 做 beta calibration，得到 (p_{\text{cal}}(x))。
用 beta 的原因不是“更先进”，而是它对 skewed binary scores 更稳，而且 logistic calibration 不包含 identity，确实可能把本来已经不错的概率再校坏。([Proceedings of Machine Learning Research][3])

### Trust：只看 boundary-local，不看全局

这一步很关键。
你现在最大的问题不是“全局分类准不准”，而是：

> **在 (p\approx0.5) 这一带，概率有没有语义。**

所以我建议 trust gate 只在 boundary-local 子集上做：

[
\mathcal B_t
============

{x:\ |p_{\text{cal}}(x)-0.5|\le \delta}
\cup
{\text{tight bracket midpoints}}
]

定义两个指标：

[
ECE_{bd}
]

[
Gap_{bd}
========

\left|
\operatorname{mean}(p_{\text{cal}})-\operatorname{feasible_rate}
\right|
]

然后设 gate：

若连续 (r) 次更新都满足

[
ECE_{bd}\le \tau_E,\qquad
Gap_{bd}\le \tau_G,\qquad
|\mathcal B_t|\ge n_{\min}
]

则开启“1 次局部细化”；否则退回 3 点 probe / midpoint。

这其实正好呼应你当前代码已经在做的事：它本来就有基于 `ECE + coreNearGap + streak + fallbackCap` 的 trust gate。我的建议不是推翻，而是把它**更明确地限定到 boundary-local 语义**。

---

## 我给你的最终主版本：一句话概括

### `PRBCCMO-BoundaryCore`

[
\texttt{TopKPair bridge}
\rightarrow
\texttt{segment-conditioned } p\approx 0.5 \texttt{ localization}
\rightarrow
\texttt{Pareto shortlist}
\rightarrow
\texttt{boundary rerank}
\rightarrow
\texttt{real evaluation}
\rightarrow
\begin{cases}
\texttt{feasible: one-step forward exploit}\
\texttt{infeasible: one-step bracket shrink}
\end{cases}
\rightarrow
\texttt{pair+mid boundary update + beta calibration}
]

---

## 每个模块怎么单独验证

为了让论文证据链干净，我建议每个模块只看一个主指标：

找点模块：
[
M_1=\operatorname{mean}\left|p_{\text{cal}}(x_s^\star)-0.5\right|
]

选择器模块：
[
M_2=
\frac{#\text{selected seeds that either tighten bracket or yield improving feasible}}
{#\text{selected seeds}}
]

可行分支：
[
M_3 = FIR
]

不可行分支：
[
M_4 = TBR
]

模型更新：
[
M_5 = ECE_{bd},\ Gap_{bd}
]

你文档里已经在用 `FRR / TBR / FIR / LY` 这一类拆分指标；继续沿这个口径走是对的。

---

## 最后一句判断

你这篇论文现在最该写成的，不是“一个复杂的边界平台”，而是：

> **在有价值的 feasible–infeasible bridge 上，用校准后的 MLP 概率去定位 (p\approx0.5) 的边界点；再用一个极简的 Pareto-then-Boundary 选择器，把这些点转成更有用的可行发现。**

这才是最贴你的原始思路、也最容易被实验真正验证的版本。

[1]: https://arxiv.org/abs/cmp-lg/9407020 "https://arxiv.org/abs/cmp-lg/9407020"
[2]: https://arxiv.org/abs/1403.5607 "https://arxiv.org/abs/1403.5607"
[3]: https://proceedings.mlr.press/v54/kull17a.html "https://proceedings.mlr.press/v54/kull17a.html"
[4]: https://arxiv.org/abs/1112.5745 "https://arxiv.org/abs/1112.5745"

