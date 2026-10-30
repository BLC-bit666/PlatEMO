# GANCMO_V2：删除参考方向条件

2026-10-30以已确定的GANCMO主线重新对齐公共部分，仅保留本版本指定消融。版本自包含，全部算法专用函数使用GANCMO_V2前缀；无需加载主线或其他消融目录。六个算法统一支持原生GA/DE选择，V3仍完全禁用CGAN，不新增V6。

## 唯一消融

G与Critic在训练、生成和评分时都只接收一个真实可行性标签s，输入宽度为D+1。训练保留0/1两侧，生成及候选Critic排序使用s=1。端点匹配仅按s分组。

参考方向继续用于存档配对、八方向门槛、原批次预算、候选槽位和方向局部盒；方向坐标不进入任何网络。这个版本消融的是网络方向条件，不删除整个算法的方向组织。

## 与当前主线一致的公共机制

- 每方向K=50；以实际方向数R设置 `candidateCount=50*R`。N=100时双目标5000槽位，三目标R=91时4550槽位。端点损失权重50。
- 本方向全部当前保留配对的F/I端点归一化后逐维取最小值和最大值，构造局部盒；盒内按距本方向最短归一化决策配对中F的距离升序选择。
- 有数据方向优先，组内随机，每方向至多一个，共享原CGAN配额；全部有数据方向尝试后仍有名额，再访问无数据方向，按本版本剩余网络条件的Critic分数降序选择。最终不足由所选GA/DE补足。
- 保留有限值、变量边界和归一化距离1e-6去重，以及按需推理、预留查询随机流、极小批次补行和Critic近同分全批回退。
- 存档配对、每方向最多5对、全部保留对的数据组织、动态训练权限、P1/P2、GA/DE选择、FE尾部和当前等价优化后的环境选择均沿用主线。
- 两网络128宽、噪声标准差1、学习率0.001、Adam(0,0.9)、每批C2/G1；首训1000完整epoch，续训最多200完整epoch且不超过1000次G更新，至少8个有效F方向。

上述范围内，仅网络条件/训练样本/训练损失按本节定义发生必要变化；不叠加其他版本的消融。筛选不预知真实可行性，所有保留子代仍须真实评价。

## 使用和验证

按PlatEMO通常方式加入框架、通用算子与问题路径，再只加入本版本目录：

```matlab
addpath(fullfile(pwd,'Algorithms','Multi-objective optimization','GANCMO_V2'));
P=LIRCMOP7_BC('N',100,'D',30,'maxFE',150000,'maxRuntime',Inf);
A=GANCMO_V2('parameter',{1},'save',-1,'outputFcn',@(varargin)[]); % 默认GA
A.Solve(P);
```

公开参数type：1使用GA（默认），2使用DE。省略 `parameter` 等同于 `{1}`，改为 `'parameter',{2}` 保留原DE模式。P1常规子代和P2均采用所选算子，CGAN子代直接评价、不经过GA/DE。使用 `Algorithms/Utility functions` 原生 `OperatorGA`、`OperatorDE` 和 `TournamentSelection`；GA奇数名额先裁掉多出的决策行再评价，DE原父代及随机顺序不变。消融比较须与主线使用相同type。

本地回归入口：`addpath('Tests'); test_GANCMO_ablations`。当前验证与公共源码差异见[对齐报告](../../../Experiments/GANCMO_Ablations_AlignMainline_20261030/REPORT.md)。覆盖剩余条件的训练/sample/score接口、局部盒、方向优先级、去重、Critic排序、查询RNG、V4的I学习隔离、V5与零端点权重主线的训练一致性及独立目录生产入口。

旧单候选版本及其结果保持历史口径，不能作为当前多候选版本的性能结论。旧源码保存在本次对齐记录的BEFORE.zip中；完整基准IGD结论需使用对应版本的实际实验结果。

算子与V3修复记录见[算子选择报告](../../../Experiments/GANCMO_OperatorChoice_20261030/REPORT.md)；对应回归 `test_GANCMO_operators` 与 `test_GANCMO_ablations` 均检查GA/DE两种模式。历史DE结果对应type=2，GA入口跑通不代表GA的最终IGD优于DE。
