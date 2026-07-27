# CBS-CGAN 主线与唯一消融

本目录保留 `CBS_RegionWGAN_GP` 唯一主线，以及一个明确命名的无 CGAN 消融类 `CBS_RegionWGAN_GP_NoCGAN`。历史参数臂、观测指标和专属实验脚本均已移除。构造器只接受 PlatEMO 标准参数；旧实验开关会明确报错，不会被静默忽略。

## 固定算法结构

1. 两个种群分别按约束目标和无约束目标协同进化。
2. 边界记忆保存配对的可行解与不可行解。
3. CGAN 训练数据固定为：
   - 可行解：条件 `[参考方向,1]`；
   - 与其配对的不可行解：条件 `[参考方向,0]`。
4. 生成时只请求条件 `[参考方向,1]`。
5. 每次查询 20 个方向槽位：14 个属于已有边界记忆的方向，6 个属于其一跳空方向。
6. 每槽生成 5 个候选，共 100 个不评价的原始生成解。
7. 对每个槽位，从 5 个候选中选择一个，使其引导步长最接近相应可行父代的局部间距。
8. 第一种群在完整代时产生 40% GA、40% 普通 DE、20% 引导 DE 子代。引导形式为可行父代朝生成解移动，强度循环使用 `0.4、0.65、0.85`；最终子代必须真实评价并参加普通环境选择。
9. CGAN 的记忆更新、训练与生成只在前 50% 评价预算内进行。边界校准全程开启，每代最多使用 20 次真实评价，并将结果回流边界记忆。

原始 CGAN 生成解从不直接计入函数评价，也不直接进入种群。

## 无 CGAN 消融

`CBS_RegionWGAN_GP_NoCGAN` 不建立边界记忆，不构造条件数据，不初始化、训练或查询 WGAN：

- `Problem.FE < 0.5*Problem.maxFE`：第一种群产生 40% GA、40% 普通 DE、20% 决策空间均匀随机采样子代；
- `Problem.FE >= 0.5*Problem.maxFE`：与主线的 CGAN 停止阶段一致，产生 40% GA、60% 普通 DE；
- 第二种群、边界校准、环境选择和真实评价预算与主线完全共用。

随机采样通过 PlatEMO 的 `Problem.Initialization` 生成，会真实评价并参加环境选择；它不同于主线中不消耗 FE 的原始 CGAN 候选。正式 200K 预算下，分界点就是 100K。

直接运行示例：

```matlab
platemo('algorithm',@CBS_RegionWGAN_GP_NoCGAN, ...
    'problem',@LIRCMOP5_BC,'N',100,'D',30, ...
    'maxFE',200000,'save',2,'metName',{'IGD'});
```

## 默认参数

| 参数 | 默认值 |
|---|---:|
| 查询槽位数 | 20 |
| 噪声维数 | 6 |
| 每次生成器更新数 | 100 |
| 小批量大小 | 32 |
| 每次生成器更新前的判别器更新数 | 4 |
| 最少训练行数 | 32 |
| 生成采样标准差 | 0.3 |

固定内部值包括：每方向最多 5 个边界锚点、每槽 5 个生成候选、20% 引导份额、CGAN 在 50% 预算处停止。

## 正式实验

正式入口为：

```matlab
[Summary,outDir] = run_CBS_RegionWGAN_GP_mainline( ...
    fullfile(pwd,'Data','CBS_RegionWGAN_GP'),10, ...
    "LIRCMOP"+string((5:12)')+"_BC",100,30,200000,1:5, ...
    struct('resume',true));
```

正式契约固定为 `maxFE=200000`，摘要只报告：

- `IGD100K`：`save=2` 在 100K 附近保留的第一份种群；同时报告其真实 `FE100K`；
- `IGD200K`：200K 终点 IGD。

新结果写入 `Data/CBS_RegionWGAN_GP`。当前 pairflag 主线在旧 `PF` 类名下生成的 23 个问题 × 5 次历史结果，已经校验数量并迁移到该正式目录，同时统一为 `CBS_RegionWGAN_GP_*.mat` 文件名；仓库中不再存在 `PF` 算法类或 `PF` 数据目录。

## 文件职责

- `CBS_RegionWGAN_GP.m`：唯一主流程、GA/DE/引导子代。
- `CBS_RegionWGAN_GP_NoCGAN.m`：前半程随机席位、后半程普通 DE 的无 CGAN 消融。
- `UpdateBoundaryMemory_RC.m`：可行—不可行边界配对记忆。
- `BuildBoundaryDataset_RC.m`：唯一 pairflag 数据集。
- `BoundaryWGAN_RC.m`：WGAN-GP 训练与条件生成。
- `RunRegionGAN_RC.m`：14+6 查询和训练/生成调度。
- `RefineBoundaryObservations_RC.m`：全程边界校准。
- `CalFitness_CBS.m`、`EnvironmentalSelection_CBS.m`：双种群适应度与环境选择。
- `Support/run_CBS_RegionWGAN_GP_mainline.m`：正式 100K/200K IGD runner。

## 性能原则

只接受不改变随机数调用、浮点计算路径和种群轨迹的优化。目前已删除所有观测数据构造，并在判别器热点中复用同一批 `single` 实数数据。没有启用 GPU、混合精度、并行训练、`dlaccelerate`，也没有改变网络、批量、训练次数或随机调用形状。

最新 20K profiler 中，WGAN 训练约占主循环时间的 98.4%；其余搜索逻辑合计不足 2%。`dlaccelerate` 微基准虽然更快，但已观察到单精度梯度末位差异，因此按严格等价原则拒绝。批量实验的主要工程加速仍是正式 runner 的问题/种子级 10-worker 并行。

## 回归测试

核心测试包括：pairflag 数据、14+6 查询、无 CGAN 消融、边界记忆、边界校准、PlatEMO 接口、正式 runner 和固定轨迹指纹。固定指纹为 `LIRCMOP6_BC`、`N=100`、`D=30`、`maxFE=20000`、随机种子 4242：

- IGD：`1.346550324710176`
- 决策和：`1883.8784633646248`
- 随机状态首值：`3522559217`
