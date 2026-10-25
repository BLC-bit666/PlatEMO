# GANCMO_V3：完全禁用 CGAN

以 2026-10-24 当前 GANCMO 单子代主线为基线，独立进行一个指定消融，不叠加其他版本的改动。所有算法专用辅助函数使用本版本前缀并随目录提供，只依赖 PlatEMO 框架、通用算子及所选问题。原 GANCMO 目录保持不变。

## 唯一消融与实现边界

不创建 G/C、训练/查询随机流、参考方向、配对存档或动态检测器，不调用训练、推理或 GAN 配额模块。P1 和 P2 的全部子代都由 DE 产生。

保留原初始化、每代 FE 分配、DE 父代选择及随机流种子、SPEA2 环境选择与两种群 DE 子代共享。P1 仍考虑约束，P2 仍忽略约束。此目录不需要其他 GANCMO 目录，也不包含神经网络辅助文件。

保留每代总计 N 次真实评价及剩余 FE 尾部截断；没有训练/推理消耗或隐藏 GAN 分支。

不提供 V6：基线已经是每方向单子代。V1–V5 均不是论文旧版多候选筛选主线的消融。

## 使用

在 PlatEMO 框架、通用函数和问题已加入路径后，只需加入本目录即可运行，不必加入 GANCMO 或其他消融目录。

```matlab
addpath(fullfile(pwd,'Algorithms','Multi-objective optimization','GANCMO_V3'));
P = LIRCMOP7_BC('N',100,'D',30,'maxFE',150000,'maxRuntime',Inf);
A = GANCMO_V3('save',-1,'outputFcn',@(varargin)[]);
A.Solve(P);
```

## 验证

本地回归入口：`addpath('Tests'); test_GANCMO_ablations`。覆盖网络实际输入、训练池、完整 epoch/续训上限、单子代与查询随机流、V3 的 DE 轨迹等价、V4 对 I 学习内容的隔离、V5 与零端点权重主线的逐位等价，以及仅加入本版本目录的生产入口运行。验证记录见 `Tests/GANCMO_ablation_validation_summary.md`；最终入口与主线回归日志为 `Tests/GANCMO_ablation_production_final.log`。这些是实现与运行验证，不是完整基准性能或最终 IGD 结论。
