# GANCMO_V3：完全禁用 CGAN

以当前GANCMO主线为基线，只消融CGAN及其专用支持模块。算法自包含，所有专用辅助函数使用GANCMO_V3前缀；只依赖PlatEMO框架、原生通用算子和所选问题，无需加载主线或其他消融目录。

## 唯一消融与公共机制

不创建G/C、训练/查询随机流、参考方向、配对存档或动态检测器，不调用训练、推理、筛选或GAN配额模块。公开参数type与主线及其他消融一致：1为GA（默认），2为DE。因为V3完全禁用CGAN，两个模式分别为纯GA、纯DE。

保留主线初始化、每代FE分配、常规子代随机流与两种群子代共享。P1仍考虑约束，P2仍忽略约束。每代总计N次真实评价，剩余FE不足时精确截断。2026-10-30已同步主线环境选择的等价优化：复用已排序的距离及邻居索引；去除版本前缀后Selection源码与主线一致，删点和同距决胜规则不变。

仅调用 `Algorithms/Utility functions` 原生 `TournamentSelection`、`OperatorGA`、`OperatorDE`，入口核对其解析路径。GA二元锦标赛选择偶数个父代，原生算子先返回决策；奇数名额时裁去多出一行后再真实评价。DE保留原父代选择、默认参数、随机调用及评价顺序。没有自制交叉、变异或隐藏GAN分支。

## 使用

按PlatEMO通常方式加入框架、通用函数和问题路径，再只加入本目录：

```matlab
addpath(fullfile(pwd,'Algorithms','Multi-objective optimization','GANCMO_V3'));
P = LIRCMOP7_BC('N',100,'D',30,'maxFE',150000,'maxRuntime',Inf);
A = GANCMO_V3('parameter',{1},'save',-1,'outputFcn',@(varargin)[]); % 默认纯GA
A.Solve(P);
```

省略 `parameter` 等同于 `{1}`；改为 `'parameter',{2}` 使用纯DE。消融比较时，主线和V1–V5必须使用相同type。历史DE结果对应type=2，不能当作新默认GA的性能结果。

## 验证

本地回归：`addpath('Tests'); test_GANCMO_operators; test_GANCMO_ablations`。覆盖两种算子的原生调用路径、默认值、非法参数、精确FE、随机流与独立目录生产入口；V3在两种算子下均与关闭CGAN的主线进行子代和完整种群轨迹对照。其他版本仍只删除各自目标机制。

本次修复、完整DE主线核验及实际覆盖范围见[算子选择报告](../../../Experiments/GANCMO_OperatorChoice_20261030/REPORT.md)。实现和入口验证不等于GA与DE的完整基准性能比较。
