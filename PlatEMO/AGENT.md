# CCMO 改造算法开发指引

## 1. 文档目的

本分支的唯一目标，是基于 `Idea.md` 的最终结论，在 `CCMO` 的骨架上逐步实现一个新的未知/二值约束多目标优化算法。本文档不是论文草稿，而是后续每一步编码、验证、消融和收敛范围控制的执行说明。

本文档中的新算法现已命名为 `PRBCCMO`，含义为 `Pareto-Relevant Boundary CCMO`。其实际落地目录为：

- `Algorithms/Multi-objective optimization/PRBCCMO/`

## 2. 必须坚持的结论

### 2.1 第一阶段真正要验证的命题

第一阶段不是证明“MLP 更强”，而是证明：

> 单个 MLP 输出的 near-0.5 候选，经过目标支配与拥挤度过滤后，比随机候选或高预测值候选更容易产生 Pareto-useful feasible offspring；把这些候选用于与约束种群交叉和局部边界邻域采样，在相同评估预算下能更快发现新的 feasible objective sectors，并改善整体优化性能。

### 2.2 当前版本明确不做

- 不上 MLP ensemble
- 不做同代内二次扩展，即“不在本代发现 near-0.5 种子后继续同代局部搜索/继续同代交叉/继续同代评估”
- 不做二分括逼、切向搜索、复杂边界校准
- 不改成复杂的 surrogate-training 技巧对比
- 不把所有 `P_U` 侧 offspring 无脑灌入 `P_C`

### 2.3 当前版本必须保留

- `CCMO` 的双种群基本语义
- 一代只做一批真实评估的时序
- 主创新只落在“near-0.5 候选怎么选、怎么用、怎么进入下一代”
- MLP 训练机制尽量简单，收益归因到 boundary candidate 利用方式

## 3. 代码基线事实

### 3.1 CCMO 当前骨架

`CCMO` 当前实现非常简单：

- 维护两个种群：约束种群 `Population1` 与忽略约束的辅助种群 `Population2`
- 两个种群各自产生 offspring
- offspring 共享给两个环境选择
- `Population1` 的适应度计算考虑约束；`Population2` 不考虑约束

对应文件：

- `Algorithms/Multi-objective optimization/CCMO/CCMO.m`
- `Algorithms/Multi-objective optimization/CCMO/EnvironmentalSelection.m`
- `Algorithms/Multi-objective optimization/CCMO/CalFitness.m`

### 3.2 可复用范围

本算法只允许基于以下两类内容推进：

- `CCMO` 的双种群主循环、共享 offspring、双环境选择骨架
- PlatEMO 通用基础设施，如 `NDSort`、`CrowdingDistance`、`TournamentSelection`、`UniformPoint`

明确排除：

- 任何已经验证失败的分支算法实现
- 通过拼接其它算法机制来替代 `Idea.md` 的最终路线

### 3.3 当前算法定位

这次不是做“完整主动边界学习算法”，而是做一个基于 `CCMO` 的最小可验证版本：

- 目标不是恢复精确决策空间边界
- 目标是找到对 Pareto 搜索有价值的 near-boundary candidates
- 论文表述优先使用
  - `Pareto-relevant boundary candidate`
  - `promising feasible objective sector`
  - `near-boundary cue`

## 4. 最终算法骨架

### 4.1 代际状态

每一代至少维护以下状态：

- `P_C^t`：约束种群
- `P_U^t`：无约束辅助种群
- `M_t`：单个 MLP 分类器
- `A_F^t`：真实评估过的 feasible near-0.5 档案
- `A_I^t`：真实评估过的 infeasible near-0.5 档案
- `D_train^t`：MLP 训练数据档案
- 可选 `W`：reference vectors / sectors，用于后续稀疏性过滤

### 4.2 一代内允许出现的候选

在第 `t` 代开始时生成以下候选池：

- `Q_C^t`：原 CCMO 约束种群常规 offspring
- `Q_U^t`：原 CCMO 无约束种群常规 offspring
- `Q_X^t`：`P_C^t × P_U^t` 的 cross-pop candidates
- `Q_LS^t`：上一代 `A_F^{t-1} ∪ A_I^{t-1}` 周围的局部扰动 candidates
- `Q_M^t`：上一代 `A_I^{t-1} × P_C^t` 的交叉 candidates

注意：

- `Q_*` 只是候选池，不等于本代真正 offspring
- 本代真正参与环境选择的，只能是本代统一真实评估后的那一批

### 4.3 一代内真正参与环境选择的 offspring

定义：

- `O_C^t`：来自 `Q_C^t` 的真实 offspring
- `O_U^t`：来自 `Q_U^t` 的真实 offspring
- `O_B^t`：从 `Q_X^t ∪ Q_LS^t ∪ Q_M^t` 中筛出并真实评估的 boundary offspring

因此本代真正 offspring 为：

- `O^t = O_C^t ∪ O_U^t ∪ O_B^t`

### 4.4 near-0.5 评分

第一阶段只用最简单、最稳定的分数：

- `u(x) = 1 - 2 * abs(p(x) - 0.5)`

其中：

- `p(x)` 是单个 MLP 给出的可行概率/边界线索
- `u(x)` 越大，说明越接近 `0.5`

### 4.5 boundary offspring 的筛选原则

第一阶段推荐两层筛选：

1. 先按 `u(x)` 取 top-k
2. 再做目标空间过滤

第一版目标空间过滤的最低要求：

- 用目标值做非支配关系判断
- 同 front 内按 crowding distance 排序

第二版增强时再加入：

- sector-aware feasible filter
- reference-vector / niche sparsity

### 4.6 环境选择原则

#### `P_C` 更新

候选集合：

- `R_C^t = P_C^t ∪ O_C^t ∪ O_B^t`

规则：

- 先选 feasible
- 若 feasible 不足，再从 infeasible 中补少量 bridge points
- 这些 bridge points 不能只看 `p≈0.5`
- 必须满足至少以下逻辑：
  - near-0.5
  - 不被当前已选 feasible 在同一区域完全压死
  - 具有目标空间多样性

#### `P_U` 更新

候选集合：

- `R_U^t = P_U^t ∪ O_U^t ∪ O_X^t`

其中：

- `O_X^t = O_B^t` 中来自 `Q_X^t` 且真实评估后的 cross-pop offspring

规则：

- 尽量保持 `CCMO` 原味
- 只做 objective-only 选择
- 第一阶段不把 `Q_LS^t`、`Q_M^t` 生成的 offspring 再注入 `P_U`

#### `A_I` 更新

主战场是保存“未来还有桥接价值的 infeasible near-0.5 点”，而不是长期塞进 `P_C`。

优先保留：

- 真实评估为 infeasible
- `u(x)` 高
- 目标空间分布稀疏
- 没有被当前已选 feasible 在同一区域完全压死

#### `A_F` 更新

保存“已确认靠近边界且值得下一代继续局部探索”的 feasible near-0.5 点。

### 4.7 一代时序红线

必须遵守：

- 本代 newly discovered near-0.5 seeds 只能进入 `A_F^t` / `A_I^t` 和训练集
- 它们的局部扩展和边界交叉必须延迟到下一代再做
- 不能把一代写成“先评估第一批，再扩展第二批，再补做评估”的异步流程

## 5. 分阶段实施路线

## 阶段 0：复制基线并跑通

目标：

- 新建 `<NEW_ALGO>` 文件夹
- 复制 `CCMO` 为最小可运行版本
- 在不改变行为的前提下完成重命名与参数接口梳理

验收：

- 新算法可在 PlatEMO 中单独运行
- 关闭新增模块时，行为退化为 `CCMO` 风格

## 阶段 1：先只加单 MLP 与训练档案

目标：

- 引入最小 MLP train/predict wrapper
- 建立 `D_train`
- 先只完成 `p(x)` 和 `u(x)` 计算链路

实现要求：

- 不改变 `CCMO` 的环境选择
- MLP 没准备好时必须能降级为“无模型模式”
- 未真实评估样本绝对不能进训练集

实现约束：

- MLP 必须按当前算法需求单独实现
- 可以使用 MATLAB/PlatEMO 的通用函数与基础语法
- 不允许参考或迁移任何失败分支算法的实现细节

## 阶段 2：先实现最小创新版，只做 `Q_X`

目标：

- 只从 `P_C × P_U` 生成 `Q_X`
- 只用 `u(x)` + 目标支配 + crowding 选出一批 uncertain seeds
- uncertain seeds 只用于与 `P_C` 交叉，暂不做局部搜索

这一步对应消融：

- `A1: Uncertain-Mating Only`

验收：

- 可以统计 U-seed / H-seed / R-seed 的 seed-level 成功率

## 阶段 3：补上局部边界邻域采样

目标：

- 在上一代 `A_F / A_I` 周围做轻量局部扰动，形成 `Q_LS`
- 保持扰动简单，不引入二分括逼和切向搜索

建议实现：

- polynomial mutation 或 Gaussian perturbation
- 可选“朝匹配 feasible parent 的方向轻推，再加少量噪声”

这一步对应消融：

- `A2: Neighborhood-Probing Only`

## 阶段 4：合并为第一阶段完整版

目标：

- 同时启用 `Q_X` 与 `Q_LS`
- 增加 `A_I × P_C -> Q_M`
- 完成 `P_C` / `P_U` / `A_F` / `A_I` 四套更新

这一步对应主版本：

- `A3: Full-Minimal`

验收：

- 代内时序仍然是“一次生成候选 -> 一次统一真实评估 -> 一次环境选择”

## 阶段 5：补强目标空间过滤

目标：

- 当简单 crowding 不够时，再加入 sector-aware filter
- 用 reference vectors 或统一 bins 定义 sector / niche

注意：

- 这一步是增强项，不是第一阶段阻塞项
- 如果 `A3` 已经明显优于对照，不要过早把机制复杂化

## 阶段 6：实验、消融与论文证据链

必须按三层证据链组织实验：

1. 机制成立
2. 边界假设成立
3. 优化收益成立

## 6. 推荐文件拆分

第一阶段建议至少拆成以下文件，避免主类过胖：

- `<NEW_ALGO>.m`
- `GenerateBoundaryCandidates.m`
- `SelectBoundaryCandidates.m`
- `EnvironmentalSelectionC.m`
- `EnvironmentalSelectionU.m`
- `UpdateBoundaryArchives.m`
- `UpdateTrainingArchive.m`
- `TrainBoundaryMLP.m`
- `PredictBoundaryMLP.m`
- `LocalBoundaryPerturbation.m`

若 `Q_M` 的交叉逻辑较特殊，再单独拆：

- `GenerateBoundaryMatingCandidates.m`

### 文件职责要求

- 主类只负责一代流程编排
- 候选生成、筛选、环境选择、档案维护、MLP 训练预测分开
- 不要把所有 helper 都塞回一个大文件里

## 7. 具体实现顺序约束

后续编码必须遵守以下顺序：

1. 先保证 `<NEW_ALGO>` 可以跑通且保留 `CCMO` 骨架
2. 再补 MLP 预测链路
3. 再补 `Q_X`
4. 再补 `Q_LS`
5. 再补 `Q_M`
6. 再改 `P_C` / `P_U` / `A_F` / `A_I`
7. 最后再做 sector-aware filter 和论文增强项

禁止跳步：

- 不允许一上来就同时写完全部模块再调
- 不允许在没有 seed-level 证据前就只看最终 HV/IGD

## 8. 第一阶段实验计划

### 8.1 核心对照

至少保留以下版本，且总评估预算必须一致：

- `A0`：`CCMO` 基线
- `A1`：只启用 uncertain mating
- `A2`：只启用 neighborhood probing
- `A3`：完整第一阶段版本
- `C1`：随机 seed 替换 near-0.5
- `C2`：高预测值 seed 替换 near-0.5
- 可选 `C3`：只按 near-0.5 选，不做目标过滤

### 8.2 机制指标

必须统计：

- `SSR`：Seed Success Rate
- `FY`：Feasible Yield
- `NSY`：New-Sector Yield
- `BHR`：Boundary Hit Rate

### 8.3 优化指标

不能只报最终值，必须加入早期指标：

- 10% 预算时的 `HV` / `IGD`
- 20% 预算时的 `HV` / `IGD`
- `HV-AUC` / `IGD-AUC`
- `Evaluations to first feasible`
- `Evaluations to first new sector`
- `Evaluations to k feasible sectors`

### 8.4 问题集与统计

优先沿用 `Idea.md` 中建议的未知约束测试协议：

- `DASCMOP-UC1–9`
- 30 次独立运行
- Wilcoxon
- Friedman 排名

## 9. 明确的非目标

以下内容不属于第一阶段交付：

- ensemble 或校准型 MLP
- bisection / bracketing
- tangential boundary crawling
- component-level 资源重分配
- 复杂 archive 注入策略
- 为了指标好看而大幅偏离 `CCMO` 原骨架
- 复用任何失败分支算法的模块与思路

## 10. 每次提交前的检查项

每次进入编码前先回答：

- 这一步是在验证创新点，还是在引入无关复杂度？
- 新增模块是否真的服务于 near-0.5 候选的选择与利用？
- 是否破坏了“一代一批评估”的时序？
- 是否让收益归因变得不清楚？

每次改完后至少确认：

- 能运行
- 预算没有失控
- 未真实评估样本没有进入训练集
- newly discovered seeds 没有在同代递归扩展
- `P_U` 仍保持 helper role，而不是被边界模块污染

## 11. 实施口径

后续所有实现、调试和回答都默认以本文档为准。若新想法与本文档冲突，优先遵循以下原则：

1. 先保住第一阶段命题的可验证性
2. 先保住 `CCMO` 骨架和单代预算可控
3. 先做最小创新版，再做增强版

如果必须偏离本文档，先在实现前更新本文档，再动代码。
