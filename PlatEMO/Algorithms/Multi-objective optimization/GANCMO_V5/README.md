# GANCMO_V5：删除生成器端点匹配损失

以 2026-10-24 当前 GANCMO 单子代主线为基线，独立进行一个指定消融，不叠加其他版本的改动。所有算法专用辅助函数使用本版本前缀并随目录提供，只依赖 PlatEMO 框架、通用算子及所选问题。原 GANCMO 目录保持不变。

## 唯一消融与实现边界

生成器仅优化 -E[C(G(z,c),c)]，移除端点距离计算、最近端点查询和 endpointWeight 配置。Critic 的 WGAN-GP 与系数 10 保持不变。

G/C 条件仍为 M+1 维，F/I 训练样本、批次、各半权重、网络、噪声、学习率和训练预算均保留。推理本来不计算训练损失，因此生成调用、单子代和 DE 配额无需额外改变。

除上述消融及其必要的维数/数据适配外，保留主线 G128/C128、噪声标准差 1、固定学习率 0.001、Adam (0,0.9)、每批 C2/G1、初训 1000 epoch、续训最多 200 个完整 epoch 且不超过 1000 次 G 更新，以及每方向一个子代、变量边界裁剪、有限值检查、DE 补足和原 FE 预算。V5 不保留端点项，V3 不启用上述神经网络机制。

不提供 V6：基线已经是每方向单子代。V1–V5 均不是论文旧版多候选筛选主线的消融。

## 使用

在 PlatEMO 框架、通用函数和问题已加入路径后，只需加入本目录即可运行，不必加入 GANCMO 或其他消融目录。

```matlab
addpath(fullfile(pwd,'Algorithms','Multi-objective optimization','GANCMO_V5'));
P = LIRCMOP7_BC('N',100,'D',30,'maxFE',150000,'maxRuntime',Inf);
A = GANCMO_V5('save',-1,'outputFcn',@(varargin)[]);
A.Solve(P);
```

## 验证

本地回归入口：`addpath('Tests'); test_GANCMO_ablations`。覆盖网络实际输入、训练池、完整 epoch/续训上限、单子代与查询随机流、V3 的 DE 轨迹等价、V4 对 I 学习内容的隔离、V5 与零端点权重主线的逐位等价，以及仅加入本版本目录的生产入口运行。验证记录见 `Tests/GANCMO_ablation_validation_summary.md`；最终入口与主线回归日志为 `Tests/GANCMO_ablation_production_final.log`。这些是实现与运行验证，不是完整基准性能或最终 IGD 结论。
