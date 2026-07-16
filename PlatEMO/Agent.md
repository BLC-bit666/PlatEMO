# CBS RegionWGAN-GP 唯一主线约定

更新时间：2026-07-16

## 1. 当前结论

仓库内 CBS-CGAN 只保留一个算法和一套正式 runner：

- 算法：`CBS_RegionWGAN_GP`
- 唯一主线：`full warm-start + Batch32 + fixed iter100 + trainGap1 + nCritic4 + G/C 32×32`
- 训练配置晋级的唯一性能指标：`gen_to_train_dec_dist50` 与 `gen_to_train_dec_dist90`，越小越好
- 生产主线 benchmark runner 只记录最终 `IGD`；IGD 不参与 Batch/iter/gap/nCritic 的晋级或淘汰
- 过程中的 loss、边界距离、coverage、多样性、可行率、survival、query 归因、checkpoint 和阶段快照均不再计算或保存
- `status`、`finalFE`、runtime、wall time、source hash 和 task signature 仅用于完整性、性能与断点恢复审计，不是性能指标

本地 Git 只有 `UC-GAN-2` 分支。远端分支未修改；不提交、不推送、不创建 PR。

## 2. 固定算法机制

每代执行：

```text
双群体 P1/P2
  -> 两组 DE 后代
  -> 用本代已评价个体更新 legacy BMem
  -> 构造 TrainX / TrainC / QueryRefs
  -> 合格事件 full warm-start 训练 WGAN-GP
  -> one-sixth query 生成完整决策向量
  -> 真实 Evaluation
  -> 两次环境选择
```

固定参数：

| 参数 | 值 |
|---|---:|
| `trainGap/archiveGap` | 1 / 1 |
| G updates / eligible event | 100 |
| C updates / eligible event | 400 |
| `ganMiniBatch` | 32 |
| `nCritic` | 4 |
| G/C hidden | `[32 32]` / `[32 32]` |
| `zDim` | 6 |
| `ganLrD/ganLrG` | 1e-4 / 1e-4 |
| `gpLambda` | 10 |
| train/sample sigma | 1.0 / 0.3 |
| `nGen` | 30 |
| `frontDepth` | 2 |
| `pairNeighborRefRadius` | 2 |
| `maxAnchorsPerRef` | 5 |
| `minGANTrainCount` | 32 |

`BMem` 每代更新。`TrainX=BMem.x_b`，保留 legacy 重复行权重与有放回抽样。原始 `TrainX<32` 时不训练、不采样、不 fallback 复用旧模型；达到阈值的每个事件都先训练再采样。

query 使用 populated refs 与一跳 frontier：`round(nGen/6)` 个样本给 frontier，其余给 populated；没有 frontier 时全部回退到 populated，不采样 remote refs。

## 3. 为什么固定为该配置

已完成的嵌套 replay、正式闭环和独立 seeds 验证显示：

- Batch 8/16、weighted epoch 与其它 iter 候选没有可靠优于 `Batch32/iter100`；
- gap2、gap3 和预算匹配 gap5 均未通过当前训练边界距离门；
- iter60 太少，iter75 的独立确认失败，iter90 没有形成足够性价比，iter110/125/150 更慢且无可靠改善；
- 分阶段 `100/85/75` 在独立确认末段 dist90 非劣失败。
- nCritic=5 的 18-run 正式筛选中，overall dist50/dist90 配对中位差为 `-0.0060808/-0.0044824`，但两项都不可靠；末段 dist50 的 95% cluster CI 上界 `0.0216389` 超过历史非劣界 `0.0189459`，且每事件 critic updates 从 400 增至 500，因此保留 nCritic=4，不启动独立确认。

因此研究配置已经收口，运行主线时不得恢复 batch/epoch/gap/phased、去重、structured-z、pointwise、MMD、coherent BMem 或额外诊断分支。

历史证据保留在：

- `Data/CBS_RegionGAN_compare/batch_epoch_gap_runs3_v1`
- `Data/CBS_RegionGAN_compare/gap123_iter_refine_runs3_v1`
- `Data/CBS_RegionGAN_compare/phased_iter_100_85_75_runs3_v1`
- `Data/CBS_RegionGAN_compare/ncritic45_distance_runs3_v1`
- `Data/CBS_RegionGAN_compare/retained_evidence`

历史 `Data` 只作证据，不是当前源码依赖，不得覆盖或删除。

## 4. 唯一源码入口

- `Algorithms/Multi-objective optimization/CBS-CGAN/CBS_RegionWGAN_GP.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/CBS_RegionGAN_Base.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/UpdateBoundaryMemory_RC.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/BuildBoundaryDataset_RC.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/RunRegionGAN_RC.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/BoundaryWGAN_RC.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/CalFitness_CBS.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/EnvironmentalSelection_CBS.m`
- `Algorithms/Multi-objective optimization/CBS-CGAN/Support/run_CBS_RegionWGAN_GP_mainline.m`

## 5. 正式实验纪律

正式默认范围：

```text
LIRCMOP5_BC–LIRCMOP10_BC
N=100, D=30, M=2, maxFE=100000
runs/seeds=1:3（需要更多独立 seeds 时另设不重叠范围）
```

调用：

```matlab
root = fileparts(which('platemo'));
outDir = fullfile(root,'Data','CBS_RegionGAN_compare','mainline_igd_runs');
[Summary,outDir] = run_CBS_RegionWGAN_GP_mainline( ...
    outDir,9,"LIRCMOP" + string((5:10)') + "_BC", ...
    100,30,100000,1:3,struct('resume',true));
```

- 正式并行必须使用 9 MATLAB workers；测试允许 1 worker。
- 每个 worker 将 OpenMP/BLAS 线程限制为 1，避免过度订阅。
- runner 按 source-tree SHA-256 与 task signature 断点续跑；只复用成功、完整、源码兼容的任务。
- 失败任务不得被忽略；诊断后写入新 attempt，只重跑失败项。
- 每个任务必须 `status=ok`、`finalFE=maxFE`、最终 IGD 有限。
- 不同 source hash 不得混用同一输出根目录。
- 每个任务仅保存最终 IGD 与审计字段，不保存最终种群和过程观测。

## 6. 当前性能基准与优化决策

第三轮只保留不改变随机数顺序、网络、loss、Adam、训练次数、采样和选择
语义的实现优化。固定
`LIRCMOP5_BC, N=100, D=30, maxFE=10000, seed=7001`
的同环境单线程基准为：

| 版本 | wall time | 最终 IGD | 判定 |
|---|---:|---:|---|
| 第三轮安全起点 | 28.5329 s | 2.575514678635725 | 对照 |
| exact-safe A | 26.8254 s | 2.575514678635725 | 保留 |

exact-safe A 相对同轮起点缩短 5.98%，约 1.064×；最终决策、目标、约束和
IGD 逐元素完全一致。保留的实现优化包括：

- critic 更新时先在梯度图外生成并 detach fake；
- 将无需求导的 fake 与 gradient-penalty 插值保留为 numeric single；
- 训练随机张量直接生成 single，同时保持原 RNG 状态和数值；
- 只在 gradient penalty 的内层梯度启用高阶导数，最终 G/C 参数梯度显式关闭；
- 热循环外提取固定标量，避免反复传递和解析完整 Options；
- 缓存 Deep Learning Toolbox 可用性并删除冗余 CPU gather；
- 每个训练事件只转换一次固定的 `TrainX/TrainC`。

critic-only `dlaccelerate` 候选在 18 个
`problem×seed`、`maxFE=100000` 正式配对中将 wall time 中位数从
1323.50 s 降至 505.30 s，逐对加速中位数为 2.688×；但 IGD 为
5 胜、4 平、9 负，`B-A` 配对中位差为 `9.0250e-05`，
Hodges–Lehmann 为 `3.4074e-04`，95% problem-cluster bootstrap CI
为 `[0, 1.7976e-03]`。它没有可靠改善且多数配对变差，因此按
“IGD 不变差”硬约束淘汰。全量和 generator-only acceleration 也因改变数值
轨迹或无速度收益而淘汰。

最后测试的 numeric `xBatch` 复用候选保持 dec/obj/con/IGD 逐元素一致，
但 3 次独立进程计时的中位数为 24.6078 s，慢于 exact-safe A 的
24.5295 s，未形成稳定收益，已回退。

最终 CBS-CGAN source-tree SHA-256：

`ad330ed678980767f0ecfe236dcc3c4a3f89a47ee230a9ba6bfdcebb4aaecf60`

最终 `BoundaryWGAN_RC.m` SHA-256：

`0b4b4e44c5b2f050438005e81d0dcbc1c1892c30c94854def67f5aebddba45de`

第三轮全部基准、18×100k 正式结果、统计、被拒绝候选和最终报告位于：

`/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Data/CBS_RegionGAN_compare/mainline_optimization_round3_20260716`

最终源码已通过 6 项回归和全目录 `checkcode`。同一 100k manifest 使用
9 workers 复跑时 18/18 `reused=1`，attempt 数保持 18。

## 7. 验证命令

```matlab
test_CBS_region_one_sixth_query
test_CBS_region_boundary_ref_cap
test_CBS_region_wgan_mainline
test_CBS_calfitness_equivalence
test_CBS_RegionGAN_provenance
test_CBS_RegionWGAN_GP_mainline_runner
```

任何后续行为中性优化都必须证明固定种子最终解与 IGD 逐元素不变后再比较
速度；会改变浮点轨迹的数值内核候选必须通过多问题、多 seed 的正式配对
IGD 非退化验证。不得以改变训练、采样或选择语义换取提速。
