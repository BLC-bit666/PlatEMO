
## 最有效、最小改动的修正方案

我建议你只改 **1 个主方案 + 1 个配套修复**。  
**不加新模块，不改论文主线。**

### 主方案：把当前 boundary core 改成“coverage-first bridge scheduler”

#### 1）先改 sector 排序：从“最好 sector 优先”改成“最稀疏 sector 优先”
你现在隐式在做：

\[
\text{Order} = \operatorname{sort}(g_s(a_s))
\]

这会导致预算反复花在已经好的 sector 上。

改成先统计每个 sector 的当前覆盖数：

\[
n_s = \left| \{x \in P_C^F \cup EA^F : \operatorname{sec}(x)=s \} \right|
\]

其中 sector 标量仍用你现在这套：

\[
g_s(x)=\sum_{m=1}^M w_{s,m}\frac{f_m(x)-z_m^{\min}}{z_m^{\max}-z_m^{\min}+\varepsilon}
\]

然后排序改成：

\[
\text{Order} = \operatorname{sortrows}\big( n_s \uparrow,\; g_s(a_s)\uparrow \big)
\]

也就是：**先扫当前覆盖稀疏的 sector，再在同覆盖层里选更有潜力的 anchor。**

这一步是**最关键**的。因为你的问题不是“不会 refine”，而是“根本没把预算分到足够多的方向”。这一改，正好对应文献里反复强调的“先保证 diverse search directions，再谈 convergence”。citeturn334384search14turn334384search10turn894511search7 fileciteturn9file0

#### 2）把 boundary budget 改成“两段式”：先 coarse 覆盖，再 local refine
当前版的问题是：一个 sector 经常会吃掉

- coarse 3 次：\(\{0.25,0.5,0.75\}\)
- refine 2 次
- shrink 1 次

即单 sector 5~6 次 FE。这样同样的 \(B_t\) 只能摸到很少几个 sector。fileciteturn9file0

改成：

先定义每代 boundary budget：

\[
B_t = \lfloor \rho_b N \rfloor
\]

coarse 只做：

\[
\Lambda_c=\{0.25,0.5,0.75\}, \qquad
K_{\text{sec}}=\min\left(|\mathcal S^{shared}|,\left\lfloor \frac{B_t}{3}\right\rfloor\right)
\]

即：**先保证 \(K_{\text{sec}}\) 个 sector 都至少被 coarse 扫一次。**

然后只对**出现了真实 sign-flip bracket** 的 sector 做 refine。  
设 coarse 上某 sector 有相邻探针 \((\lambda_j,\lambda_{j+1})\) 满足一可行一不可行，则记：

\[
[\lambda_L,\lambda_U]=[\lambda_j,\lambda_{j+1}]
\]

只对这类 sector 做 local refine：

\[
\Lambda_r=
\left\{
\operatorname{clip}(\lambda^\*-\delta,[\lambda_L,\lambda_U]),
\operatorname{clip}(\lambda^\*+\delta,[\lambda_L,\lambda_U])
\right\},
\qquad \delta=0.125
\]

**没有 bracket，就不 refine。**

这一步特别重要。因为你现在的问题不是 refine 不够，而是 refine **太早、太满、太平均**，把 coverage 吃掉了。文献上对小可行域/离散可行域也都更支持这种“先 feasibility–diversity，再局部 exploitation”的节奏。citeturn334384search0turn334384search10 fileciteturn9file0

#### 3）shrink 也只在 bracket 内做，不要对任意 infeasible seed 都 shrink
当前 shrink 实际还是偏“局部硬补救”。  
改成只对**tightest feasible–infeasible pair** 做 midpoint：

若当前局部已评估样本里最紧的一对是 \((\lambda_F,\lambda_I)\)，则

\[
\lambda_{\text{mid}}=\frac{\lambda_F+\lambda_I}{2}
\]

只评估 \(x(\lambda_{\text{mid}})\)。

**没有 bracket，不 shrink。**

这样你保住了“边界追踪”创新，但不会再把 FE 浪费在不可信的 seed 上。fileciteturn9file0

#### 4）migration 不再只看最后 seed，要看“本地所有已评估 feasible 点”
这是你当前实现里一个很伤的细节。

每个 sector 里，把 coarse / refine / shrink 所有已评估点并起来：

\[
\mathcal E_s = \mathcal Q_s^{coarse}\cup \mathcal Q_s^{refine}\cup \mathcal Q_s^{shrink}
\]

然后 migration 候选改成：

\[
x_s^{mig}=
\arg\min_{x\in \mathcal E_s,\; x\ \text{feasible}} g_s(x)
\]

若

\[
g_s(x_s^{mig}) < g_s(a_s)
\]

则迁移。

也就是说：**不是“最后选中的那个点”迁移，而是“本地已评估 feasible 点里最好的那个”迁移。**

这一步几乎零风险，但收益很大。因为你现在很可能已经评到了好点，只是被后续 infeasible refine 覆盖掉了。fileciteturn9file0

---

### 配套修复：恢复旧版的 `TopK`，但只恢复这一条

#### 5）每个 sector 内恢复 `topK` bridge pairing
这个我建议恢复，而且我判断**它确实有用**。

对每个 sector \(s\)，先取：

\[
F_s^K = \text{Top-}K \text{ feasible anchors by } g_s,\qquad
U_s^K = \text{Top-}K \text{ infeasible helpers by } g_s
\]

推荐 \(K=5\)，沿用你旧版。

然后 bridge pair 不要再固定用“全 sector 最优 feasible + 全 sector 最优 infeasible”，改成在 topK 里选**最局部**的一对：

\[
(a_s,u_s)=
\arg\min_{f\in F_s^K,\;u\in U_s^K,\; g_s(u)\le g_s(f)}
|g_s(f)-g_s(u)|
\]

如果没有 \(g_s(u)\le g_s(f)\) 的组合，再退化成纯最小 gap。

这个改动的本质是：**让 bridge 更像“局部穿边界”，而不是“从一个很强 anchor 拉一条过长的跨区段线”。**  
旧版 `BridgeTopK=5` 的真正价值就在这里。fileciteturn9file2turn9file0

---

## 一个很小但该做的训练修复

#### 6）LabelArchive 改成 boundary-first，而不是纯类别平衡
当前 `TrimLabelArchive` 只做最近的正负样本平衡，会把大量真正有用的 boundary points 淹掉。fileciteturn9file0

改成：

\[
M_B = \lfloor 0.6M \rfloor,\qquad M_R = M - M_B
\]

其中 \(M\) 是 archive 最大容量。

保留策略：

- 先保最近的 boundary positives / boundary negatives，各占一半左右；
- 再用 non-boundary positives / negatives 补满。

这样 MLP 学到的才是**边界几何**，不是普通远离边界的可行/不可行分类。  
这个改动不加模块，只是把你的单 MLP 真正变成 boundary MLP。fileciteturn9file0

---

## 我不建议你现在动的东西

先不要动这些：

- 不要把 trust / committee / calibration / hard-negative 再搬回来  
- 不要先上 adaptive reference vectors  
- 不要先开 `UseDirectMix`
- 不要先加更大的 MLP

因为你现在的主矛盾不是“模型不够强”，而是**coverage policy 错了**。文献里真正有效的点，也是先把 feasibility、diversity、subregion coverage 理顺，再谈更复杂的 exploitation。citeturn151625search3turn672068search1turn334384search14

---

## 你这个问题里，topK 到底是不是关键？

我的判断：

- **是关键，但不是唯一关键**
- **topK 解决的是“桥是否局部、是否靠谱”**
- **coverage-first scheduler 解决的是“预算有没有分到足够多的 sector”**

所以答案是：**上一版不是单靠 topK 赢的，而是靠“topK + shortlist + 先 coverage 后 boundary”一起赢的。**  
你这次掉到 1e-1，主因更偏后者。fileciteturn9file2turn9file0

---

## 最终建议：就按这个顺序改

第一轮只改 4 处：

1. `Order` 改成 `(sector occupancy asc, anchorScalar asc)`  
2. `ExecuteBoundaryCore` 改成 **coarse-first / bracket-only refine / bracket-only shrink**  
3. migration 改成从 **所有已评估 feasible 点** 里选 local-best  
4. 恢复 `TopK=5` bridge pairing

第二轮再加 1 处：

5. `LabelArchive` 改成 boundary-first

这 5 个改动都属于**当前主线内部修复**，不是新模块；它们最符合你的论文叙事，也最可能直接修正你现在的三件核心诉求：**创新点不破、IGD下降、PF覆盖恢复。**

一句话总结：**你不是要把算法再做复杂，而是要把当前 PRBCCMO-core 从“best-sector-first 的局部深挖器”改回“coverage-first 的边界搜索器”。**


**助手 (gpt-5-4-pro) — 2026-03-29 14:50:45.717 UTC**

*思考小结：Thought for 39m 29s*


---
**参考链接**

- https://burrsettles.com/pub/settles.activelearning.pdf?source=post_page
- https://arxiv.org/pdf/2108.09588
- https://link.springer.com/article/10.1007/s00366-021-01441-4
- https://www.sciencedirect.com/science/article/pii/S221065022300175X
- https://colab.ws/articles/10.1109%2Ftevc.2020.3004012
- https://arxiv.org/abs/2411.08437
- https://arxiv.org/abs/2301.03317
- https://www.mdpi.com/2227-7390/13/19/3206
- https://link.springer.com/article/10.1007/s40747-023-01042-2
- https://www.sciencedirect.com/science/article/pii/S2210650221001826
- https://link.springer.com/article/10.1007/s40747-022-00851-1
- https://link.springer.com/article/10.1007/s10898-019-00860-4
- https://arxiv.org/pdf/1707.08767
- https://link.springer.com/article/10.1007/s44336-024-00006-5
- https://arxiv.org/pdf/1612.07603
- https://www.openai.com
- https://discovery.researcher.life/article/a-multipopulation-evolutionary-algorithm-using-new-cooperative-mechanism-for-solving-multiobjective-problems-with-multiconstraint/6a80cf1fbbec34609f2a6900f5b251ce
- https://link.springer.com/chapter/10.1007/978-3-642-00619-7_7
- https://arxiv.org/abs/1711.07907
- https://link.springer.com/article/10.1007/s00158-022-03473-w
- https://proceedings.mlr.press/v70/guo17a.html
