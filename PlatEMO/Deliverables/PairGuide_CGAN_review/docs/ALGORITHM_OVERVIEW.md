# 当前算法介绍

## 1. 研究对象与入口

算法面向约束多目标优化，沿用 CBS 双种群框架：

- P1：约束种群，适应度和环境选择考虑约束；研究目标是更快逼近约束边界及可行 Pareto 前沿。
- P2：无约束种群，忽略约束进行搜索，提供跨越不可行区域的目标方向信息。
- 每代把 P1、P2 及两者子代合并后，分别进行有约束与无约束环境选择。

当前仓库保留两个主要入口：

- `code/algorithms/CBS-CGAN/CBS_RegionWGAN_GP.m`：较早生产路径，使用全局 critic 与局部目标映射。
- `code/algorithms/CBS-CGAN-PairGuide/CBS_RegionWGAN_GP_PairGuide.m`：当前研究主线，使用原子可行/不可行端点对、绝对端点 CGAN 和 PairGuide 差分引导。

最新消融与配额实验中的 “mainline” 均指第二个入口。

## 2. 每代双种群流程

主循环位于 `code/algorithms/CBS-CGAN/Core/CBS_RegionWGAN_GP_Core.m` 的 `runMainline`。

1. 初始化两个大小均为 N 的种群 P1、P2。
2. P1 生成 N 个子代。默认组成是 25% GA、55% 普通 DE、20% PairGuide；PairGuide 不足的名额由普通 DE 回填。
3. P2 生成 N 个子代，固定为 25% GA、75% 普通 DE。
4. 合并 P1、P2 及双方子代；从同一 Union 分别选择下一代 P1 和 P2。
5. 用已真实评价的 Union 更新 PairGuide 边界档案。
6. 满足训练门槛时训练或继续训练 CGAN，随后生成 500 个原始候选，供下一代 P1 引导使用。

因此当前 PairGuide 不是把原始 CGAN 点直接加入种群。原始点 G 不评价、不进入 Union；它先和 P1 可行精英形成局部目标 T，再由 `OperatorDE` 生成真正接受评价的子代。

## 3. 原子边界档案

实现：`code/algorithms/CBS-CGAN/Core/PairBoundaryArchive_RC.m`。

每条记录保存一对真实评价端点：

```text
id, xf, yf, xi, yi, ref, gap, age, lastFE, active
```

- `xf/yf`：真实可行端点及目标值。
- `xi/yi`：真实不可行端点及目标值。
- `ref`：由可行端点目标值分配的参考方向。
- 新 pair：当前 P1 的可行 `Fitness<1` 精英，与 evaluated Union 中归一化决策距离最近的普通真实不可行解配对。
- 更新：可行或不可行端点只有在严格缩小 pair 决策空间 gap 时才替换。
- active pair 才参与训练和查询；连续 10 代无局部支持证据会删除。
- 每个参考方向最多保留 5 对。

此档案逼近的是决策空间中的可行/不可行端点对，不是显式训练出的分类超曲面，也没有直接把 P1 与 P2 成员一一配对。

## 4. CGAN 数据、网络与损失

实现：`code/algorithms/CBS-CGAN/Core/PairBoundaryWGAN_RC.m`。

每个 active pair 形成两个绝对决策向量训练样本：

```text
[xf_norm, reference_vector, side=1]
[xi_norm, reference_vector, side=0]
```

当前网络：

- 生成器输入：6 维高斯噪声 `z` 加条件 `[w, side]`。
- 生成器：全连接隐藏层 `[32,32]`、LeakyReLU，输出 D 维 `tanh` 绝对归一化决策向量。
- Critic：输入生成/真实决策向量与条件，隐藏层 `[32,32]`，线性标量输出。
- WGAN-GP：学习率 `1e-4`，梯度惩罚 10，当前 5 Critic : 1 Generator。
- 生成器损失：对抗损失、同噪声可行/不可行端点关系的 energy-distance 损失、mode-seeking 损失。

当前损失没有直接约束 “生成点到分界面的法向距离”、点云厚度或一条低维连续边界；训练对象是两个 side 条件下的端点分布。生产查询固定使用 `side=0`，即请求不可行侧候选，而不是显式请求 pair 中点或 `side=0.5` 分界点。

## 5. 训练方式

当前暂定参数：

| 项目 | 当前值 |
|---|---:|
| 首次训练 | 500 Epoch |
| 后续重训 | 10 Epoch |
| 完整 pair 门槛 | 32 |
| active reference 门槛 | 4 |
| mini-batch | 32 个完整 pair，即 64 个端点 |
| 训练遍历 | 每 Epoch 对全部 active pair 随机打乱、无放回访问一次，尾批保留 |
| 重训触发 | pair 新增/收紧/移除合计至少 8，或新增 reference 至少 2 |
| 模型更新 | warm start；训练结果立即采用，无验证集、早停或 checkpoint 接受门槛 |
| 推理噪声 | `sigma=1` |

首次实际训练时数据通常只有几十个 pair，训练数据规模远小于 500 个每次生成的候选数。

## 6. 当前 PairGuide 子代形成

相关函数：`PairBoundaryArchive_RC('querycontexts'/'selectcandidates')` 与 `pairGuideDecisions`。

1. 在 active reference 上均衡构造 500 个 `[w,side=0]` 条件。
2. CGAN 生成 500 个绝对决策向量 G；critic 不筛选、不排序。
3. 去除越界、重复、无可归属 pair、缺乏局部支持的候选；每个 G 匹配同一或邻近参考方向中最近的 active `xi`。
4. 从当前 P1 可行精英中选择能局部支持该 pair 的父代 `xc`。
5. 计算局部尺度 `h`、距离 `d`，构造 `Fg=min(0.5,h/(d+eps))` 和 `T=clip(xc+Fg*(G-xc))`。
6. 在 T 空间按参考方向空缺权重进行 maximin 选择。
7. 调用 `OperatorDE(Problem,xc,T,xc,{1,1,1,20})` 产生并评价子代；不足名额由普通 DE 回填。

这意味着 G 当前承担“方向/目标向量”角色，不是直接作为被评价的子代，也不是与整个 P1 共同构成一个常规交配池。

## 7. 已保留算法分支

| 分支 | 唯一变化 | 用途 |
|---|---|---|
| `CBS_RegionWGAN_GP_PairGuide` | 20% PairGuide | 当前研究主线 |
| `CBS_RegionWGAN_GP_PairGuide_DE20` | 20% PairGuide 名额改为普通 DE | CGAN 贡献消融 |
| `..._Quota30` | PairGuide 名额 30%，普通 DE 45% | 配额实验 |
| `..._Quota40` | PairGuide 名额 40%，普通 DE 35% | 配额实验 |
| `..._Quota50` | PairGuide 名额 50%，普通 DE 25% | 配额实验 |
| `CBS_RegionWGAN_GP` | 旧全局 critic/局部目标路径 | 机制参照 |

## 8. 与当前疑问直接相关的已知事实

- 原始 500 个 G 不消耗搜索 FE。图中对 G 的目标和可行性检查由独立问题实例离线完成，不反馈算法。
- 图中的厚度主要是目标空间投影；真实模型输出位于 D 维决策空间，两者不能视为同一几何厚度。
- 推理 `sigma` 从 1 降至 0.5 可把决策空间和目标空间点云厚度中位数降至约 0.60，但这只是缩小采样范围，不证明学到了正确分界线。
- 把训练噪声也降到 0.5 反而使相同 `use sigma=0.5` 下的厚度中位数扩大约 1.31 倍。
- 提高 PairGuide 名额没有稳定、单调改善最终 IGD；当前继续保留 20%。
- 当前实验不能把厚点云归因于过拟合、网络容量或档案构造中的任何单一因素；这些仍需隔离验证。
- 核心疑问讨论 20 维问题；本资料包中最新 DE20、配额及大部分可视化实验实际使用 D=30。算法支持可变 D，但现有 D=30 证据不能直接等同于 20 维结论。
