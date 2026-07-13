# CBS RegionWGAN-GP

该目录只保留一个算法：`CBS_RegionWGAN_GP`。它用参考向量条件化的 WGAN-GP 直接生成完整决策向量，并把生成解送入 PlatEMO 的真实评价和双群体环境选择。

## 运行机制

```text
双群体 + DE
  -> BMem（可行 anchor / 当前不可行邻点）
  -> TrainX = 可行侧完整决策，TrainC = reference vector
  -> online warm-start WGAN-GP
  -> 25 populated + 5 one-hop frontier（nGen=30）
  -> 真实 Evaluation
  -> 双群体选择和 query/BMem 归因
```

唯一 query 使用 `round(nGen/6)` 个一跳 frontier 样本，其余给 populated refs；remote refs 不采样。若不存在 frontier，则全部预算回退到 populated refs。

## 主要文件

| 文件 | 职责 |
|---|---|
| `CBS_RegionWGAN_GP.m` | PlatEMO 算法入口和主线参数 |
| `CBS_RegionGAN_Base.m` | 双群体进化、GAN 评价、选择和诊断 |
| `UpdateBoundaryMemory_RC.m` | BMem 更新 |
| `BuildBoundaryDataset_RC.m` | reference-only 训练集和 query pool |
| `RunRegionGAN_RC.m` | one-sixth query、WGAN 调用和边界诊断 |
| `BoundaryWGAN_RC.m` | conditional WGAN-GP |
| `RegionQueryTracker_RC.m` | frontier 转化和 BMem 进入归因 |
| `Support/run_CBS_RegionWGAN_GP_mainline.m` | 非绘图、可复现实验 runner |

## 默认参数

```text
trainGap=1, archiveGap=1, nGen=30
zDim=6, sampleSigma=0.3
ganIter=100, nCritic=2, miniBatch=32
lrD=1e-4, lrG=1e-4, gpLambda=10
frontDepth=2, pairNeighborRefRadius=2
maxAnchorsPerRef=5, minGANTrainCount=32
```

## 正式实验

```matlab
outDir = fullfile(pwd,"Data","CBS_RegionGAN_compare","mainline");
run_CBS_RegionWGAN_GP_mainline( ...
    outDir,8, ...
    ["LIRCMOP5_BC";"LIRCMOP6_BC";"LIRCMOP7_BC"; ...
     "LIRCMOP8_BC";"LIRCMOP9_BC";"LIRCMOP10_BC"], ...
    100,30,100000,1:3,struct());
```

runner 不绘图，保存 provenance、source manifest、事件诊断、最终 P1/P2、最终可行非支配集、query 样本、ref 转化和直接 BMem 进入记录。

上一轮 BMem 的可行 anchor 可继续参与当前竞争并重新配对；若连续入选，age 会累积，因此该机制不是严格 TTL=1。

## 验证

```matlab
test_CBS_region_one_sixth_query
test_CBS_region_wgan_mainline
test_CBS_region_query_group_diagnostics
test_CBS_region_query_loop_diagnostics
test_CBS_region_query_tracker
test_CBS_region_boundary_ref_cap
test_CBS_boundary_wgan_mapping_diagnostics
test_CBS_true_boundary_diagnostics
test_CBS_decision_eval_helper
test_CBS_RegionGAN_provenance
test_CBS_RegionWGAN_GP_mainline_runner
```

历史 query 和被拒绝机制只保留在 `Data/CBS_RegionGAN_compare/retained_evidence`，不再有可执行源码分支。能量距离明确排除。
