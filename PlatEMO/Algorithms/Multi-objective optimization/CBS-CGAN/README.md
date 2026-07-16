# CBS RegionWGAN-GP 唯一主线

该目录只保留 `CBS_RegionWGAN_GP`。算法使用参考向量条件化的 WGAN-GP 生成完整决策向量，并将生成解纳入 PlatEMO 的真实评价和双群体环境选择。

## 固定配置

```text
full warm-start
Batch = 32
G updates = 100 / eligible event
C:G updates = 4:1
trainGap = 1
G/C hidden = 32x32 / 32x32
zDim = 6
sampleSigma = 0.3
```

`nCritic=5` 已使用 10 workers 完成 18 个正式对比 run。它在 overall
dist50/dist90 上只有不可靠的小幅下降，并在末段 dist50 未通过非劣门，
同时每个事件增加 25% critic updates，因此最终仍固定 `nCritic=4`。

`BMem` 每代更新。原始 `TrainX` 行数少于 32 时，该事件不训练、不采样，也不复用旧 GAN；达到阈值后使用 legacy 重复行权重和有放回抽样。

query 预算中 `round(nGen/6)` 分配给一跳 frontier，其余分配给 populated refs；没有 frontier 时全部回退到 populated refs，不向 remote refs 采样。

## 主线文件

| 文件 | 职责 |
|---|---|
| `CBS_RegionWGAN_GP.m` | 算法入口及固定参数 |
| `CBS_RegionGAN_Base.m` | 双群体进化主循环 |
| `UpdateBoundaryMemory_RC.m` | legacy BMem 更新 |
| `BuildBoundaryDataset_RC.m` | 构造 `TrainX/TrainC/QueryRefs` |
| `RunRegionGAN_RC.m` | one-sixth query 与 WGAN 调用 |
| `BoundaryWGAN_RC.m` | conditional WGAN-GP 训练和采样 |
| `CalFitness_CBS.m` | 向量化 SPEA2 fitness |
| `Support/run_CBS_RegionWGAN_GP_mainline.m` | 最终 IGD 正式 runner |

算法内部不再计算或保存 loss、边界距离、coverage、多样性、可行率、survival、query 归因、checkpoint 或阶段快照。训练配置晋级的唯一性能指标是历史研究中的 `gen_to_train_dec_dist50/90`；生产主线 runner 只保存最终 IGD，`status/finalFE/runtime/wall time/provenance` 只用于完整性、恢复和运行审计。

## 正式运行

```matlab
root = fileparts(which('platemo'));
outDir = fullfile(root,'Data','CBS_RegionGAN_compare','mainline_igd_runs');
[Summary,outDir] = run_CBS_RegionWGAN_GP_mainline( ...
    outDir,9, ...
    "LIRCMOP" + string((5:10)') + "_BC", ...
    100,30,100000,1:3,struct('resume',true));
```

正式并行固定为 9 workers；每个 worker 将 BLAS/OpenMP 线程限制为 1，避免过度订阅。runner 支持按 source hash 和 task signature 断点续跑；成功且兼容的任务复用，失败任务写入新的 immutable attempt 后单独重跑。输出目录不会与不同源码版本混用。

每个任务只保存最终 IGD 行和运行审计字段，不保存最终种群或过程观测。根目录包含：

- `run_summary.csv`
- `mainline_config.json`
- `provenance.csv`
- `source_manifest.csv`
- `<problem>_run<seed>/attempt_*/task_result.mat`

## 性能优化

主线已删除诊断 forward、loss/history、边界距离、coverage、多样性、可行率、
survival、query tracker、checkpoint 与阶段快照。当前 exact-safe 优化还：

- 在梯度图外计算并 detach critic 所需的 generator fake；
- 让无需求导的 fake 和 GP 插值保持 numeric single；
- 直接生成 single 随机张量，保持原随机序列；
- 仅在 GP 内层梯度启用高阶导数；
- 外提热循环标量、缓存工具箱检查并删除冗余 gather；
- 每个训练事件只转换一次固定的 `TrainX/TrainC`。

固定 seed 的 10k FE 同轮基准由 28.5329 s 降至 26.8254 s，提速
1.064×；最终决策、目标、约束和 IGD 均逐元素一致。

critic-only `dlaccelerate` 在 18 个 100k FE 正式配对上将 wall time
中位数从 1323.50 s 降到 505.30 s，但 IGD 为 5 胜、4 平、9 负，
配对中位差为正，未满足“IGD 不变差”，因此淘汰。numeric `xBatch` 复用
虽逐元素等价，但 3 次独立计时没有稳定收益，也未保留。

最终 source-tree SHA-256：

`ad330ed678980767f0ecfe236dcc3c4a3f89a47ee230a9ba6bfdcebb4aaecf60`

验证证据：

`/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Data/CBS_RegionGAN_compare/mainline_optimization_round3_20260716`

## 回归测试

```matlab
test_CBS_region_one_sixth_query
test_CBS_region_boundary_ref_cap
test_CBS_region_wgan_mainline
test_CBS_calfitness_equivalence
test_CBS_RegionGAN_provenance
test_CBS_RegionWGAN_GP_mainline_runner
```

历史调度、nCritic=5 和其它被拒绝机制的结果仍保留在 `Data/CBS_RegionGAN_compare`，但没有可执行源码分支，不是当前运行依赖。nCritic 对比证据位于 `Data/CBS_RegionGAN_compare/ncritic45_distance_runs3_v1`。
