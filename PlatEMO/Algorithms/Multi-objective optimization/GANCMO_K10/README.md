# GANCMO_K10：每方向 K=10 的候选数量实验

从当前 GANCMO 主线完整复制；除类名、函数名及错误标识的版本前缀外，唯一代码差异为 `GANCMO_K10Settings.m` 中的 `candidateCount=10*referenceCount`。K=50 直接使用原 GANCMO，不建立重复版本。

端点损失权重50、局部盒、最短配对F排序、有数据方向优先、Critic排序、去重、训练、检测、种群选择和FE预算均保持主线实现。K=1同样执行候选筛选；不恢复历史的单候选直接评价分支。

版本自包含，全部算法专用辅助函数使用 `GANCMO_K10` 前缀，不依赖 GANCMO 或其他消融目录。公开参数保持 `type=1` 为默认GA、`type=2` 为DE，均调用 Utility functions 中的原生算子。

按PlatEMO通常方式添加框架及问题路径后使用：

```matlab
addpath(fullfile(pwd,'Algorithms','Multi-objective optimization','GANCMO_K10'));
A = GANCMO_K10('parameter',{1},'save',-1,'outputFcn',@(varargin)[]);
A.Solve(P); % P为已创建的问题对象；DE模式改为parameter={2}
```

实现差异和短程验证记录见 `Experiments/GANCMO_KVariants_20261107`。短程验证不代表完整基准实验或IGD性能结果。
