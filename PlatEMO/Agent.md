# CBS-RegionGAN 当前主线状态（2026-07-08）

## 当前主线决策

Region-conditioned GAN 分支当前主线确定为：

- 算法：`CBS_RegionWGAN_GP`
- GAN 类型：`wgan-gp`
- `queryMode`：`random_all_w`
- `prevBMemMode`：`prev1_fair_union`，上一代 `BMem` 的可行 anchor 在配对前并入当前可行候选池，与当前 `[Population1, Offspring1, Population2, Offspring2]` 的可行解一起竞争
- `nGen`：30，每代 GAN 最多注入 30 个生成解
- `zDim`：6
- `ganIter`：100
- 训练 Z：随机，默认标准正态
- 采样 Z：随机，`sampleSigma=0.3`
- `prescreenMultiplier`：1，不做 critic prescreen
- `ganMiniBatch`：32
- `minGANTrainCount`：32，训练集少于 32 行时跳过本代 GAN 训练与生成
- `maxAnchorsPerRef`：5，在配对不可行解之前，每个 ref 最多保留 5 个按 `CalFitness_CBS` 排序最优的可行边界 anchor
- WGAN-GP 参数：`gpLambda=10`，`nCritic=2`

这条主线对应三步实验结论和一次机制修正：先由 `mb32 + CBS_RegionWGAN_GP` 确定 WGAN-GP 主分支，再由 6 问题、`runs=3` 的 `current_only` vs 旧版 `prev1_fair_union` 实验证明上一代 `BMem` 短记忆能提高训练覆盖；随后图像诊断暴露“整条 BMem row 晚合并”会沿用旧配对结构，因此当前主线已改为“上一代可行 anchor 配对前合并，且只与当前不可行解重新配对”；最后在 2026-07-08 的真实边界指标复核后，操作主线按本轮决策切到 `C_iter100_ncritic2`，即 `ganIter=100,nCritic=2`。普通 `CBS_RegionCGAN` 保留为 BCE/sigmoid CGAN 对照分支，不再作为当前主线。

2026-07-08 的新增现实：

- 真实边界指标已经成为当前主判据：先看 `bdist50_true`，再看 `bwidth90_10_true`；`bcover_eps_true` 只作为旁证，不能覆盖距离和宽度上的失败。
- A/B/C/D 已用真实边界指标重算。有效目录为 `Data/CBS_RegionGAN_compare/ncritic_true_boundary_20260708_105015`，四支均为 18/18 ok，且均有 true-boundary columns。当前操作主线已改为 `C_iter100_ncritic2`，但文档必须保留一个事实：A 和 C 在不同聚合口径下各有优势，不能写成 C 在所有统计上绝对最优。
- `region_slocal` 与 `region_rho` 两个条件分支已完成正式对照。有效目录为 `Data/CBS_RegionGAN_compare/condition_slocal_rho_20260708_153442`；`condition_slocal_rho_20260708_153359` 和 `condition_true_boundary_20260708_153234` 不是有效结论目录。
- `s_local/rho` 的实现是生效的，不是接线失败：`BuildBoundaryDataset_RC` 会把 `TrainC/QueryC` 扩成 `[W(ref,:),scalar]`，`random_all_w` 也会从扩展后的 all-ref condition pool 采样。但实验没有证明它们能替代 C 基线。
- 新增 scalar 没有稳定收益的主因是训练目标仍是 `BMem.x_b/y_b` 的可行侧 boundary cloud，而不是 pair-supported thin boundary。`s_local/rho` 只是在解释当前 cloud 内部位置；当 cloud 本身偏厚、偏移或多层时，条件更细并不会自动生成真实窄边界带。
- 下一步训练集方向应从“继续加 scalar”转为“利用 `BMem.x_f/y_f` 与 `BMem.x_i/y_i` 的可行/不可行 pair 构造 thin TrainX”。直接把 `x_f` 和 `x_i` 无 side 条件混入同一 real distribution 不建议作为主线，因为它会让生成器学习可行/不可行两侧混合带。
- `ganIter` 动态线性调度、采样 Z 控制和 WGAN critic prescreen 已经实现为实验控制项，但不是当前主线默认。
- 已完成的 `linear_iter` 正式分支只验证了 `ganIter=150->30`，未启用 prescreen；结果显示早期生成边界贴合明显改善，但最终 18-run 汇总没有稳定优于固定 `ganIter=75`。
- 已完成的可视化分支 `linear150->75 + prescreen8` 同时改变了训练强度和 critic 筛选；run=1 五阶段图和指标均显示中后期生成点比固定 `ganIter=75` 更偏移、更离散，因此该组合不能进入主线。
- `trainSigma/sampleSigma` 已拆分。训练仍用随机标准 Z，采样端单独缩小 Z 方差后，`sampleSigma=0.3` 是当前最均衡的默认值；`0.15` 无收益，`0.25/0.5/zero` 只在局部指标或局部阶段有优势。
- `zDim=2/3/4/8` 都没有足够证据替代 `zDim=6`。`zDim=4` 有若干最终中位贴边信号，但 tail 与全阶段稳定性较弱；`zDim=8` 不稳；`zDim=2/3` 明显不作为主线候选。
- 2026-06-30 已补跑 `sampleSigma=0.3,zDim=6` 的 run=1 五阶段图，分别覆盖 `ganIter=50` 和 `ganIter=75`。两者都只是视觉诊断产物，后续已由 2026-07-08 的 A/B/C/D true-boundary 复核取代为更高优先级证据。
- 因此当前主线保持 `CBS_RegionWGAN_GP.mainlineDefaults()` 的固定参数语义：`ganIter=100,nCritic=2,zDim=6,sampleSigma=0.3,prescreenMultiplier=1`。动态 `ganIter`、critic prescreen、`s_local/rho` 条件只能作为拆因子诊断，不能作为主线堆叠。

## 为什么形成这条主线

这条主线不是从“目标空间点 y 反推唯一决策变量 x”的逆映射思路来的，而是从传统条件生成模型的思想收敛来的：

- 早期 `ref_y` / `ref_y_tau` 方向把条件写成 `[参考方向, 归一化目标 y, 可选 tau]`，再用 `G(z,c)->x` 生成决策变量。这个设计在诊断上逐渐暴露出一个问题：当 `c` 里包含很具体的目标位置 y，且 `z=0` 或固定 z 经常用于重建时，模型很容易退化成“给定 y 找一个 x”的确定性逆映射。
- 逆映射不是当前目标。我们真正想要的是：在当前目标空间的可行/不可行边界附近，按粗区域学习一团决策变量分布 `p(x | region)`，再由随机 `z` 在该区域内采样不同的边界候选。也就是说，`c` 只负责告诉生成器“目标空间的大致区域/参考分区”，`z` 负责表达同一区域内的决策空间多样性。
- 这也是为什么后来改成 RegionCGAN/RegionWGAN：条件从精确 y 收缩为粗 region/ref 条件，训练目标从“重建某个 y 对应的 x”转为“拟合该 region 内当前边界决策云的分布”。这更接近传统 CGAN 的分布拟合逻辑。

这里的“传统 CGAN 思想”不等于必须坚持 BCE/sigmoid 损失。普通 BCE CGAN 是对照分支；主线采用 WGAN-GP，是因为它仍然学习条件分布 `p(x | region)`，但把对抗损失换成更适合小样本在线训练的 Wasserstein critic + gradient penalty。

关键设计取舍如下：

1. `z` 不再固定为主线用法。固定 z 只能诊断生成器是否能重建/回放条件，不能代表生成分布；主线训练使用随机标准 Z，采样使用随机 Z 但缩小到 `sampleSigma=0.3`，以减少后期厚带和离群点。
2. 条件 `c` 不再承担精确目标坐标的逆映射职责，而是表示粗目标区域。当前主线用 `queryMode=random_all_w` 在所有 ref 上随机抽 QueryC，避免只在已有训练条件附近回放。
3. 不做无限跨代累计训练集。边界会随进化过程移动，长期历史会污染当前分布；当前只把上一代 `BMem` 的可行 anchor 作为短记忆，在配对前与当前可行候选公平竞争，过旧边界和旧不可行配对不进入主线。
4. 训练集不能退回成每个 ref 一个点的薄骨架，但 2026-07-08 的真实边界指标和条件分支实验已经说明：当前前二可行 PF + 每 ref 最多 5 个 anchor 的 boundary cloud 也不是最终答案。它保留了区域内分布，但会把厚带、多层边界和偏移片段一起交给 WGAN。下一步应把 `x_f/y_f` 与 `x_i/y_i` 的 pair 结构显式用于 thin boundary target。
5. 网络必须小。单次刷新样本量受当前种群窗口限制，不能靠跨代累计扩样；因此采用 `[32 32]` 小网络。
6. `zDim=6` 来自之前对密集边界云的 SVD 诊断和 2026-06-30 的 zDim 拆因子实验：区域内决策云内在维大约 3-5，`zDim=2/3` 偏小，`zDim=4` 有局部最终收益但 tail 较差，`zDim=8` 不稳，6 仍是最稳的折中。
7. WGAN-GP 优先于 BCE CGAN。BCE 诊断能看到 D/G 对抗过程，但在小样本、窄边界、易模式坍缩的设置下不够稳；WGAN-GP 的线性 critic 和梯度惩罚更适合作为主线。

因此，当前主线可以概括为：**用传统条件生成分布的思想学习 `p(x | coarse objective region)`，但用 WGAN-GP 训练而不是 BCE CGAN；避免确定性逆映射，避免长期历史累计污染，用上一代 BMem 可行 anchor 的短记忆缓解 ref 覆盖不足，并用当前不可行解重新定义边界配对。** 但截至 2026-07-08，新的主矛盾已经从 query/采样尺度转到训练集定义：仅学习可行侧 cloud 不足以稳定得到真实窄边界带。

## 已落地代码

- `Algorithms/Multi-objective optimization/CBS-CGAN/CBS_RegionWGAN_GP.m`
  - 角色从 comparison branch 明确改为 mainline branch。
  - 新增 `CBS_RegionWGAN_GP.mainlineDefaults()`，集中固定当前主线默认参数。
  - `main()` 通过 `mainlineDefaults()` 注入默认参数，当前锁定 `queryMode=random_all_w`、`prevBMemMode=prev1_fair_union`、`ganIter=100`、`nCritic=2`、`zDim=6`、`sampleSigma=0.3`，避免后续手改 `ParameterSet` 时不小心偏离主线。
- `Algorithms/Multi-objective optimization/CBS-CGAN/CBS_RegionGAN_Base.m`
  - `QueryMode` 默认值改为可从 `Config.queryMode` 读取。
  - `prevBMemMode` 可从 `Config.prevBMemMode` 或实验控制读取；若分支未提供，仍保持旧默认 `current_only`，因此 `CBS_RegionCGAN` 的对照语义不被强行改变。
  - 支持实验控制字段 `ganIterSchedule`、`ganIterStart`、`ganIterEnd`、`sampleZMode`、`trainZMode`、`sigma`、`trainSigma`、`sampleSigma`、`prescreenMultiplier`。
  - `applyRegionGANConfigOptions` 已让 `Config/default mainline` 的 `sampleSigma` 传入 `GANOptions`，再由实验控制覆盖。
  - 事件级和阶段级诊断已记录 `gan_iter_used`、`gan_iter_schedule`、`sample_z_mode`、`train_z_mode`、`train_z_sigma`、`sample_z_sigma`、`prescreen_multiplier`、`prescreen_candidate_count`、`prescreen_selected_count`。
- `Algorithms/Multi-objective optimization/CBS-CGAN/RunRegionGAN_RC.m`
  - 新增 `resolveganoptions` helper，用 `currentFE/maxFE` 解析动态 `ganIter`。
  - `linear_fe` 公式：`round(ganIterStart + (ganIterEnd - ganIterStart) * currentFE / maxFE)`，并夹在 start/end 区间内。
  - `trainandsample` 现在保留 `last_sample_info`，供主循环写入 prescreen 诊断。
  - 已实现真实边界诊断 `trueboundarydiagnostics`，输出 `bdist50_true`、`bwidth90_10_true`、`bcover_eps_true`。当前主排序只使用前两个：先 `bdist50_true`，再 `bwidth90_10_true`。
- `Algorithms/Multi-objective optimization/CBS-CGAN/BoundaryWGAN_RC.m`
  - `latentSamples(..., purpose)` 已区分训练和采样：`purpose="train"` 使用 `trainSigma`，`purpose="sample"` 使用 `sampleSigma`；未显式设置时回退到旧 `sigma`。
  - 支持 `prescreenMultiplier>1`：每个最终生成槽位先生成 `prescreenMultiplier` 个候选，用 WGAN critic 分数选最高者，只有选中的最终 `nGen` 个解进入 `Problem.Evaluation`。
  - 注意：critic prescreen 不增加 FE，但当前证据显示 critic 高分不等价于目标空间边界贴合。
- `Algorithms/Multi-objective optimization/CBS-CGAN/UpdateBoundaryMemory_RC.m`
  - `prev1_fair_union` 将上一代 `BMem` 的可行 anchor 在配对前注入可行候选池，与当前可行解一起做前 `frontDepth=2` 层筛选和每 ref 最多 5 个 anchor 的 cap。
  - 配对阶段只使用当前 `[Population1, Offspring1, Population2, Offspring2]` 中的不可行解；上一代 `BMem` 的旧 `x_i/y_i/gap` 不再整条沿用。
  - `BMem` 当前确实保存 `x_f/y_f` 与 `x_i/y_i` pair，但 region 版训练集尚未真正利用 `x_i/y_i` 作为训练监督。
- `Algorithms/Multi-objective optimization/CBS-CGAN/BuildBoundaryDataset_RC.m`
  - 已支持 `conditionMode=region/region_slocal/region_rho`。
  - `region_slocal` 和 `region_rho` 会把条件从 `W(ref,:)` 扩成 `[W(ref,:),scalar]`；`random_all_w` 会从扩展后的 all-ref condition pool 采样。
  - 当前 `TrainX` 仍是 `BMem.x_b`，在 region 版里等价于可行侧 `x_f`。这意味着新增 scalar 只是解释可行侧 cloud，没有把可行/不可行 pair 变成 thin boundary target。
- `Algorithms/Multi-objective optimization/CBS-CGAN/Support/run_CBS_RegionCGAN_training_diagnostics.m`
  - 诊断 runner 在 `algorithmClass=CBS_RegionWGAN_GP` 且未显式覆盖时，默认跟随 `CBS_RegionWGAN_GP.mainlineDefaults()`。
  - WGAN 的 `event_summary_all.csv` 和 `stage_snapshots_all.csv` 已能正常汇总；`train_history_all.csv` 对 WGAN 仍为空，因为当前 WGAN-GP 不记录逐 iter 内部训练历史。
- `Algorithms/Multi-objective optimization/CBS-CGAN/Support/run_CBS_RegionWGAN_GP_ncritic_true_boundary_branches.m`
  - 已用于 A/B/C/D 的真实边界指标复核，输出目录 `Data/CBS_RegionGAN_compare/ncritic_true_boundary_20260708_105015`。
- `Algorithms/Multi-objective optimization/CBS-CGAN/Support/run_CBS_RegionWGAN_GP_condition_true_boundary_branches.m`
  - 已用于 `C_region_slocal` 与 `C_region_rho` 的条件分支复核，输出目录 `Data/CBS_RegionGAN_compare/condition_slocal_rho_20260708_153442`。
- `Algorithms/Multi-objective optimization/CBS-CGAN/Support/run_CBS_RegionWGAN_GP_qw_plan_branches.m`
  - 新增并实际使用拆因子分支：`query_boundary_iter50_z_zero`、`query_boundary_iter50_z_sigma025`、`query_boundary_iter50_z_sigma03`、`query_boundary_iter50_z_sigma015`、`query_boundary_iter50_z_sigma05`、`query_boundary_iter50_zdim2`、`query_boundary_iter50_zdim3`、`query_boundary_iter50_sigma03_zdim4`、`query_boundary_iter50_sigma03_zdim8`。
  - 旧 `linear_iter_*` 和 `*_prescreen*` 分支保留为诊断候选，但当前不作为主线。
- `Algorithms/Multi-objective optimization/CBS-CGAN/test_CBS_region_gan_branches.m`
  - 主线默认参数回归测试应锁定 `wgan-gp + random_all_w + prev1_fair_union + ganIter=100 + zDim=6 + sampleSigma=0.3 + ganMiniBatch=32 + minGANTrainCount=32 + nGen=30 + maxAnchorsPerRef=5 + gpLambda=10 + nCritic=2`。
  - 增加动态 `ganIter` 公式、WGAN sampling Z、`trainSigma/sampleSigma` 拆分和 critic prescreen 的回归测试。

## 训练集与生成流程现状

截至 2026-07-08，region 版边界训练集仍按当前简化门控构造：

1. 从 `[Population1, Offspring1, Population2, Offspring2]` 中收集当前候选。
2. 先筛选可行解。
3. 对可行解做目标空间非支配排序，保留前 `frontDepth=2` 层。
4. 在配对不可行解之前，对每个 ref 按 `CalFitness_CBS` 升序最多保留 `maxAnchorsPerRef=5` 个可行 anchor。
5. 若启用 `prev1_fair_union`，把上一代 `BMem.x_f/y_f` 作为可行 anchor 加入当前可行候选池；不把上一代 `x_i/y_i/gap` 作为已有配对直接带入。
6. 在合并后的可行候选池上执行前二 PF 和每 ref 最多 `maxAnchorsPerRef=5` 的 anchor cap。
7. 将保留的可行 anchor 与当前窗口里的邻近不可行解重新配对，形成新边界 cloud，再执行 adaptive gap cap，得到新 `BMem` 与 `TrainX/TrainC`。
8. 若 `TrainX < minGANTrainCount=32`，本代跳过 WGAN-GP 训练和生成。
9. 若训练样本充足，则用 WGAN-GP 训练，并在所有 ref 上完全随机抽取 `QueryC=random_all_w`，总生成数受 `nGen=30` 限制。训练 Z 默认标准随机；采样 Z 当前默认 `sampleSigma=0.3`。

注意：当前只保留上一代 `BMem` 这一层短记忆，不做多代历史累计，也不做需要额外真实评估的后验边界筛选或局部修复。

2026-07-08 的训练集结论：

1. 当前 `BMem` 已经保存 pair 信息：`x_f/y_f` 是可行侧点，`x_i/y_i` 是最近不可行侧点，`gap` 是两端在归一化目标空间的距离。
2. 但 `BuildBoundaryDataset_RC` 当前训练时只取 `TrainX=BMem.x_b`、`TrainY=BMem.y_b`；在 region 版 `UpdateBoundaryMemory_RC` 中，`x_b/y_b` 等价于可行侧 `x_f/y_f`。
3. 因此当前 WGAN 学到的是 `p(x_f | condition)`，不是 `p(boundary pair | condition)`，也不是 `p(feasible-side thin boundary point | condition)`。
4. 直接把 `[x_f;x_i]` 作为同一 condition 下的普通 `TrainX` 不建议进入主线。没有 `side/tau` 条件时，这会让模型学习可行/不可行两侧混合带，可能提高不可行生成并增大 `bwidth90_10_true`。
5. 下一步应优先做 `pair-supported thin feasible-side TrainX`：从每个可行/不可行 pair 中选择或构造更靠近边界的可行侧 target，例如迁移旧 `UpdateBoundaryMemory_CBS` 的 `selectThinBoundaryTarget` 思路，或先保守保留 gap 最小、非支配、每 ref 少量 pair 的 `x_f`。
6. 如果要显式使用 `x_i` 训练两侧分布，必须新增 `side/tau/gap_norm` 条件，并且生成时只 query 可行侧 `side=feasible` 或接近可行侧的 `tau`；不能让生成器无条件采样不可行侧。

## 最近实验依据

两组实验均已完成 `LIRCMOP5_BC` 到 `LIRCMOP10_BC`，`runs=1`，每个问题 5 个 FE 图点，共四组：

- `mb32 + CBS_RegionCGAN`
- `mb32 + CBS_RegionWGAN_GP`
- `mb16 + CBS_RegionCGAN`
- `mb16 + CBS_RegionWGAN_GP`

实验目录：

- `Data/CBS_RegionGAN_compare/refcap5_min32_LIR5_10_runs1_20260626_193916`
- `Data/CBS_RegionGAN_compare/refcap5_min16_mb16_LIR5_10_runs1_20260626_201509`

关键汇总表：

- `Data/CBS_RegionGAN_compare/mb32_vs_mb16_boundary_snapshot_metrics.csv`
- `Data/CBS_RegionGAN_compare/mb32_vs_mb16_domain_boundary_dist_metrics.csv`

全阶段 30 个快照的核心指标：

| 组别 | near_boundary_rate_gap1 越高越好 | gap_ratio50 越低越好 | gap_ratio90 越低越好 | domain_boundary_dist50 越低越好 | domain_boundary_dist90 越低越好 |
|---|---:|---:|---:|---:|---:|
| `mb32 + CBS_RegionCGAN` | 0.1167 | 4.0545 | 14.6459 | 0.0761 | 0.1845 |
| `mb32 + CBS_RegionWGAN_GP` | 0.3833 | 1.6549 | 9.6636 | 0.0565 | 0.1535 |
| `mb16 + CBS_RegionCGAN` | 0.1667 | 5.0057 | 19.0620 | 0.0713 | 0.1919 |
| `mb16 + CBS_RegionWGAN_GP` | 0.3000 | 2.1882 | 10.7947 | 0.0579 | 0.1628 |

后期 `FE>=70000` 的核心指标：

| 组别 | near_boundary_rate_gap1 越高越好 | gap_ratio50 越低越好 | gap_ratio90 越低越好 | domain_boundary_dist50 越低越好 | domain_boundary_dist90 越低越好 |
|---|---:|---:|---:|---:|---:|
| `mb32 + CBS_RegionCGAN` | 0.2667 | 3.4312 | 15.7226 | 0.0479 | 0.1437 |
| `mb32 + CBS_RegionWGAN_GP` | 0.4000 | 1.5302 | 9.9233 | 0.0280 | 0.1176 |
| `mb16 + CBS_RegionCGAN` | 0.1667 | 5.0057 | 17.9454 | 0.0526 | 0.1778 |
| `mb16 + CBS_RegionWGAN_GP` | 0.2833 | 2.5880 | 12.0457 | 0.0512 | 0.1324 |

结论：`mb16 + WGAN-GP` 在少数“红点到训练集最近点”的中位距离上有优势，但 tail 更差，gap 指标更差，后期 FE 表现也弱于 `mb32 + WGAN-GP`。因此第一阶段主线选 `mb32 + CBS_RegionWGAN_GP`。

第二阶段已完成 `current_only` vs 旧版 `prev1_fair_union` 对照：

- 实验目录：`Data/CBS_RegionGAN_compare/prev1_fair_union_6prob_runs3_10w_20260626_225234`
- 问题：`LIRCMOP5_BC` 到 `LIRCMOP10_BC`
- `runs=1:3`
- `N=100`
- `maxFE=100000`
- 对照组：`current_only`
- 新主线候选：`prev1_fair_union`
- 后处理事件表：
  - `current_only/generated_event_summary_from_metric.csv`
  - `prev1_fair_union/generated_event_summary_from_metric.csv`
  - `current_only/all_history_summary_from_metric.csv`
  - `prev1_fair_union/all_history_summary_from_metric.csv`

关键结果：

| 指标 | `current_only` | `prev1_fair_union` | 结论 |
|---|---:|---:|---|
| 生成事件数 | 5979 | 7297 | 生成触发更稳定 |
| 生成触发率 | 0.7386 | 0.9239 | 覆盖错配明显缓解 |
| 平均 `train_count` | 61.0214 | 207.8288 | 训练边界云大幅增厚 |
| 平均 `region_count` | 23.0271 | 44.6707 | ref 覆盖接近翻倍 |
| `gap_ratio50` 中位数 | 2.0632 | 1.2992 | 中位贴边距离改善 |
| `gap_ratio90` 中位数 | 10.6326 | 6.9558 | 漂移尾部收缩 |
| `near_boundary_rate_gap1` | 0.3378 | 0.4371 | 贴边比例提升 |
| `feasible_rate` | 0.3725 | 0.3540 | 可行率略降 |

配对 run 稳健性：

- `gap_ratio90`：16/18 个 run 改善。
- `gap_ratio50`：17/18 个 run 改善。
- `near_boundary_rate_gap1`：15/18 个 run 改善。
- `train_count` 与 `region_count`：18/18 个 run 增加。
- `feasible_rate`：只有 7/18 个 run 改善，不能作为本次改动的收益点。

上一代 `BMem` 的实际作用：

- `prev1_fair_union` 中上一代 `BMem` 平均候选约 207，平均存活约 196。
- 各问题上一代 `BMem` 存活率约 0.93 到 0.96。
- 这说明短记忆不是摆设，而是在 per-ref cap 和 gap cap 后大量进入新 `BMem`，直接缓解当前窗口边界覆盖不足。

因此当前主线仍更新为 `mb32 + CBS_RegionWGAN_GP + random_all_w + prev1_fair_union`，但 `prev1_fair_union` 的实现语义已从“整条 BMem row 晚合并”改成“上一代可行 anchor 配对前合并并用当前不可行解重配”。旧实验验证了短记忆有价值，但不能作为新合并时机的唯一最终证据；后续 `query_boundary`、Z 采样和五阶段图像实验已经补充了新的诊断依据。

## 2026-06-29 动态训练强度与 critic prescreen 实验

### A. `ganIter=150->30` 对固定 `ganIter=75`

对照目录：

- 固定 75：`Data/CBS_RegionGAN_compare/formal_round3_train_20260628_230707/round3/query_boundary_iter75`
- 动态 150->30：`Data/CBS_RegionGAN_compare/qw_plan_dynamic_z_prescreen_20260629_134917/dynamic_z_prescreen/linear_iter`

共同设置：

- `LIRCMOP5_BC` 到 `LIRCMOP10_BC`
- `runs=1:3`
- `N=100`
- `maxFE=100000`
- `queryMode=boundary_populated`
- `prevBMemMode=prev1_fair_union`
- `bmemBandMode=current`
- `nCritic=5`
- `zDim=6`
- `sampleZMode=random`
- `trainZMode=random`
- `sigma=1`
- 不画图

完成状态：

| 分支 | run_summary | stage_snapshots |
|---|---:|---:|
| `query_boundary_iter75` | 18/18 ok | 30 |
| `linear_iter` | 18/18 ok | 30 |

`linear_iter` 的阶段实际 `gan_iter_used`：

| target FE | gan_iter_used |
|---:|---:|
| 10000 | 138 |
| 30000 | 114 |
| 50000 | 90 |
| 70000 | 66 |
| 100000 | 30 |

run=3 stage snapshot 早期结果：

| FE=10000 指标 | fixed75 | 150->30 | 结论 |
|---|---:|---:|---|
| `gap_ratio50` 越低越好 | 5.546 | 2.125 | 动态更好 |
| `gap_ratio90` 越低越好 | 38.92 | 10.78 | 动态更好 |
| `near_boundary_rate_gap1` 越高越好 | 0.332 | 0.644 | 动态更好 |
| `feasible_rate` 越高越好 | 0.637 | 0.564 | 动态更差 |

18 个 run 的最后一次生成事件：

| 指标 | fixed75 | 150->30 | 配对胜场 |
|---|---:|---:|---:|
| `gap_ratio50` 越低越好 | 1.749 | 1.830 | 9/18 |
| `gap_ratio90` 越低越好 | 24.20 | 31.30 | 12/18 |
| `near_boundary_rate_gap1` 越高越好 | 0.430 | 0.427 | 8/18 |
| `feasible_rate` 越高越好 | 0.285 | 0.249 | 7/18 |

平均 runtime：

| 分支 | 平均秒/run |
|---|---:|
| fixed75 | 1645 |
| 150->30 | 2311 |

结论：

- `150->30` 明确改善早期 `FE=10000` 的生成点边界贴合，支持“早期训练强度不足”这个问题存在。
- `150->30` 没有形成最终整体改进；最终 `gap_ratio50`、`near_boundary_rate_gap1`、`feasible_rate` 都没有优于固定 75。
- `150->30` 平均 runtime 增加约 40%，不能作为主线替换固定 75。
- 不能据此否定动态训练强度本身；更合理的候选是较高下限，例如 `150->75` 或 `150->60`，但必须单独验证。

### B. `linear150->75 + prescreen8` 对固定 `ganIter=75` 的 run=1 图像实验

对照目录：

- 固定 75 图像：`Data/CBS_RegionGAN_compare/visual_run1_query_boundary_iter75_20260629_100927/query_boundary_iter75_run1_figures/domain_figures_all`
- `linear150->75 + prescreen8`：`Data/CBS_RegionGAN_compare/visual_run1_linear150_75_prescreen8_20260629_152622`

共同设置：

- `LIRCMOP5_BC` 到 `LIRCMOP10_BC`
- `run=1`
- `N=100`
- `maxFE=100000`
- 默认 `D`
- 5 个阶段图：`[10000 30000 50000 70000 100000]`
- 图中包含可行域、不可行域、训练集、GAN/WGAN 生成解。

`linear150->75 + prescreen8` 完成状态：

- `run_summary.csv`：6/6 ok
- `stage_snapshots_all.csv`：30 行
- `figure_manifest.csv`：30 行
- PNG：30 张
- 诊断表显示 `prescreen_multiplier=8`
- 阶段 `gan_iter_used`：`[142,127,112,97,75]`

阶段聚合指标：

| 阶段 | 指标 | fixed75 | linear150->75 + prescreen8 | 结论 |
|---:|---|---:|---:|---|
| FE=10000 | `gap_ratio50` | 8.531 | 0.933 | linear 更好 |
| FE=10000 | `gap_ratio90` | 17.81 | 4.898 | linear 更好 |
| FE=10000 | `near_boundary_rate_gap1` | 0.187 | 0.539 | linear 更好 |
| FE=50000 | `gap_ratio50` | 0.758 | 1.681 | fixed75 更好 |
| FE=50000 | `near_boundary_rate_gap1` | 0.585 | 0.415 | fixed75 更好 |
| FE=70000 | `gap_ratio50` | 0.814 | 2.148 | fixed75 更好 |
| FE=70000 | `near_boundary_rate_gap1` | 0.576 | 0.415 | fixed75 更好 |
| FE=100000 | `gap_ratio50` | 0.959 | 1.269 | fixed75 更好 |
| FE=100000 | `near_boundary_rate_gap1` | 0.564 | 0.424 | fixed75 更好 |
| FE=100000 | `feasible_rate` | 0.317 | 0.289 | fixed75 更好 |

视觉结论：

- 用户观察“iter75 的生成效果更好，linear150_75 反而没有拟合训练集，更偏移、离散”与图像和指标一致，尤其在 `FE>=50000` 后成立。
- `LIRCMOP7_BC`、`LIRCMOP8_BC` 的最终图中，`linear150->75 + prescreen8` 的红点明显更向训练带外侧扩散，并出现更高 `f2` 离群点。
- `LIRCMOP9_BC` 的 `gap90` 在 linear 分支上有改善，但红点变成几个 critic 偏好的离散簇；不能据此说明整体生成分布更贴边。

严格结论：

- `linear150->75 + prescreen8` 不是主线候选。
- 该实验同时改变了 `ganIter` 调度和 critic prescreen，因此不能把退化单独归因于 `150->75` 或单独归因于 prescreen。
- 但当前证据已经足够说明：直接用 WGAN critic 做 `8选1` 并不能保证更贴近目标空间可行/不可行边界，甚至可能放大 critic 偏差。
- critic 分数不是边界距离、可行性、连续性或训练集贴合度的等价替代；prescreen 继续使用前必须先做拆因子和相关性验证。

## 2026-06-30 Z 采样与 zDim 拆因子实验

### A. `trainSigma/sampleSigma` 拆分后验证采样 Z 方差

实验目标：训练仍保持随机标准 Z，只改变采样时的 Z 分布，隔离“采样 Z 方差过大导致后期离散/厚带”的影响。

共同设置：

- `LIRCMOP5_BC` 到 `LIRCMOP10_BC`
- `runs=1:3`
- `workers=10`
- `N=100`
- `D` 使用问题默认
- `maxFE=100000`
- 不画图
- `queryMode=boundary_populated`
- `prevBMemMode=prev1_fair_union`
- `bmemBandMode=current`
- `ganIter=50`
- `zDim=6`
- `trainZMode=random, trainSigma=1`
- `prescreenMultiplier=1`

实验目录：

- `Data/CBS_RegionGAN_compare/iter50_z_factors_20260629_195827/iter50_z_factors`
- `Data/CBS_RegionGAN_compare/iter50_sigma_tight_20260630_093920/iter50_sigma_tight`

完成状态：

| 分支 | sample 设置 | run_summary | stage_snapshots |
|---|---|---:|---:|
| `query_boundary_iter50` | `sampleSigma=1` | 18/18 ok | 30 |
| `query_boundary_iter50_z_zero` | `sampleZMode=zero` | 18/18 ok | 30 |
| `query_boundary_iter50_z_sigma025` | `sampleSigma=0.25` | 18/18 ok | 30 |
| `query_boundary_iter50_z_sigma05` | `sampleSigma=0.5` | 18/18 ok | 30 |
| `query_boundary_iter50_z_sigma03` | `sampleSigma=0.3` | 18/18 ok | 30 |
| `query_boundary_iter50_z_sigma015` | `sampleSigma=0.15` | 18/18 ok | 30 |

关键结论：

- `sampleSigma=1` 确认不是最佳采样尺度，后期生成点更容易形成厚带和离群。
- `sampleSigma=0.25/0.3/0.5` 都明显优于 `1` 的若干指标，说明“只缩小采样 Z 方差”是有效因素，而不是训练分布被一起改小造成的假象。
- `sampleSigma=0.15` 过小，没有进一步收益；`zero` 虽可降低一部分 tail，但会损失随机采样分布表达，不能作为主线。
- 综合全阶段、FE=100000 和最后生成事件，`sampleSigma=0.3` 是当前最均衡默认；`0.25` 在个别最终 tail 指标更强，但全阶段稳定性和折中性不如 `0.3`。

代表性聚合结果：

| 设置 | all-stage `gap50` | all-stage `gap90` | all-stage `near` | all-stage `feasible` |
|---|---:|---:|---:|---:|
| `sampleSigma=1` | 1.661 | 19.40 | 0.4745 | 0.3344 |
| `sampleSigma=0.5` | 1.540 | 11.02 | 0.5233 | 0.3919 |
| `sampleSigma=0.3` | 1.367 | 11.65 | 0.5197 | 0.3808 |
| `sampleSigma=0.25` | 1.677 | 35.50 | 0.5140 | 0.3862 |
| `sampleSigma=0.15` | 1.944 | 14.83 | 0.4919 | 0.3677 |
| `sampleZMode=zero` | 2.550 | 19.94 | 0.4583 | 0.4164 |

### B. 在 `sampleSigma=0.3` 上验证 zDim 范围

实验目标：在已确定的采样尺度上验证 `zDim` 是否需要从 6 调整。

实验目录：

- `Data/CBS_RegionGAN_compare/iter50_sigma03_zdim_range_20260630_113853/iter50_sigma03_zdim_range`

完成状态：

| 分支 | zDim | sampleSigma | run_summary | stage_snapshots |
|---|---:|---:|---:|---:|
| `query_boundary_iter50_sigma03_zdim4` | 4 | 0.3 | 18/18 ok | 30 |
| `query_boundary_iter50_sigma03_zdim8` | 8 | 0.3 | 18/18 ok | 30 |

对照采用 `query_boundary_iter50_z_sigma03` 的 `zDim=6,sampleSigma=0.3` 结果。

| 设置 | all-stage `gap50` | all-stage `gap90` | all-stage `near` | all-stage `feasible` |
|---|---:|---:|---:|---:|
| `zDim=6` | 1.367 | 11.65 | 0.5197 | 0.3808 |
| `zDim=4` | 2.063 | 13.07 | 0.4995 | 0.4381 |
| `zDim=8` | 1.394 | 13.59 | 0.4610 | 0.3730 |

结论：

- `zDim=4` 在 FE=100000 和最后生成事件的 `gap50/near` 上有局部信号，但 `gap90` 与全阶段稳定性较弱，说明压缩 latent 维度可能减少中位偏移，却更容易出现 tail 问题。
- `zDim=8` 没有提供稳定收益，尤其 `near` 和配对胜场不足。
- `zDim=2/3` 在上一轮 `iter50_z_factors` 中已表现为 gap tail 明显变差，不再作为主线候选。
- 当前保留 `zDim=6`。后续若再看 `zDim=4`，只能作为单独候选，不能和动态 `ganIter`、prescreen 同时叠加。

### C. `sampleSigma=0.3,zDim=6` 五阶段图像复跑

目的：在新采样尺度主线上重新查看“可行域、不可行域、训练集、CGAN/WGAN 生成解”的目标空间形态，特别关注生成点是否从厚带/离散点云收敛到窄边界带。

共同设置：

- `LIRCMOP5_BC` 到 `LIRCMOP10_BC`
- `runs=1`
- `workers=10`
- `N=100`
- `D` 使用问题默认
- `maxFE=100000`
- 五阶段：`[10000 30000 50000 70000 100000]`
- `queryMode=boundary_populated`
- `prevBMemMode=prev1_fair_union`
- `bmemBandMode=current`
- `zDim=6`
- `sampleSigma=0.3`
- `prescreenMultiplier=1`

图像目录：

- `ganIter=50`：`Data/CBS_RegionGAN_compare/visual_run1_query_boundary_sigma03_zdim6_20260630_155715/domain_figures_all`
- `ganIter=75`：`Data/CBS_RegionGAN_compare/visual_run1_query_boundary_sigma03_zdim6_iter75_20260630_162136/domain_figures_all`

完成状态：

- 两个目录均为 6/6 ok。
- 两个目录均有 `stage_snapshots_all.csv` 30 行。
- 两个目录均有 `domain_figure_manifest.csv` 30 行。
- 两个目录均有统一 `domain_figures_all` 图目录。

当前判定：

- 这两批图是视觉诊断，不替代 `runs=3` 指标结论。
- `ganIter=75` 已完成复跑，可用于和 `ganIter=50` 直接视觉对照。该段是 2026-06-30 的阶段性判定；2026-07-08 后，主线已由 true-boundary A/B/C/D 复核和本轮决策更新为 `ganIter=100,nCritic=2`。

## 2026-07-08 真实边界指标、C 主线与条件分支实验

### A. A/B/C/D 用新指标重算

实验目录：

- `Data/CBS_RegionGAN_compare/ncritic_true_boundary_20260708_105015`

共同设置：

- `LIRCMOP5_BC` 到 `LIRCMOP10_BC`
- `runs=1:3`
- `N=100`
- `maxFE=100000`
- `queryMode=random_all_w`
- `prevBMemMode=prev1_fair_union`
- `sampleSigma=0.3`
- `prescreenMultiplier=1`
- 主判据：先 `bdist50_true`，再 `bwidth90_10_true`

完成状态：

| 分支 | `ganIter` | `nCritic` | run_summary | stage_snapshots | event_count |
|---|---:|---:|---:|---:|---:|
| `A_current_50x5` | 50 | 5 | 18/18 ok | 30 | 8004 |
| `B_iter50_ncritic2` | 50 | 2 | 18/18 ok | 30 | 7991 |
| `C_iter100_ncritic2` | 100 | 2 | 18/18 ok | 30 | 8012 |
| `D_iter150_ncritic1` | 150 | 1 | 18/18 ok | 30 | 7996 |

18 个 problem/run 的事件内中位数汇总：

| 分支 | `bdist50_true` 中位数 | `bwidth90_10_true` 中位数 | `bcover_eps_true` 中位数 | `feasible_rate` 中位数 |
|---|---:|---:|---:|---:|
| `A_current_50x5` | 0.04084 | 0.30333 | 0.10000 | 0.41667 |
| `B_iter50_ncritic2` | 0.06791 | 0.43938 | 0.10000 | 0.41667 |
| `C_iter100_ncritic2` | 0.05025 | 0.30341 | 0.10000 | 0.39167 |
| `D_iter150_ncritic1` | 0.06315 | 0.29479 | 0.10000 | 0.40000 |

latest-valid 事件汇总：

| 分支 | `bdist50_true` 中位数 | `bwidth90_10_true` 中位数 | `bcover_eps_true` 中位数 | `feasible_rate` 中位数 |
|---|---:|---:|---:|---:|
| `A_current_50x5` | 0.02426 | 0.08335 | 0.12500 | 0.23333 |
| `B_iter50_ncritic2` | 0.03328 | 0.10559 | 0.10000 | 0.15000 |
| `C_iter100_ncritic2` | 0.02707 | 0.09112 | 0.15000 | 0.18333 |
| `D_iter150_ncritic1` | 0.03883 | 0.10687 | 0.15000 | 0.16667 |

判定：

- A 与 C 在不同聚合口径下各有优势，不能写成 C 在所有统计上绝对胜出。
- B 明显不是主线候选。
- D 的宽度局部有信号，但主距离和末段表现不足以替代 C。
- 本轮操作主线按决策更新为 `C_iter100_ncritic2`，并已反映到 `CBS_RegionWGAN_GP.mainlineDefaults()`：`ganIter=100,nCritic=2`。

### B. `s_local/rho` 条件分支对 C 基线

有效实验目录：

- 基线 C：`Data/CBS_RegionGAN_compare/ncritic_true_boundary_20260708_105015/C_iter100_ncritic2`
- 条件分支：`Data/CBS_RegionGAN_compare/condition_slocal_rho_20260708_153442`

无效或失败目录：

- `Data/CBS_RegionGAN_compare/condition_slocal_rho_20260708_153359`
- `Data/CBS_RegionGAN_compare/condition_true_boundary_20260708_153234`

完成状态：

| 分支 | condition mode | run_summary | stage_snapshots | event rows |
|---|---|---:|---:|---:|
| `C_baseline` | `region` | 18/18 ok | 30 | 8012 |
| `C_region_slocal` | `region_slocal` | 18/18 ok | 30 | 8004 |
| `C_region_rho` | `region_rho` | 18/18 ok | 30 | 7999 |

注意：`branch_summary.csv` 中 `ok_run_count=0` 是汇总字段 bug，不能作为实验失败证据；`run_summary.csv` 和 `run.log` 已确认 18/18 ok。

18 个 problem/run 的事件内中位数：

| 分支 | `bdist50_true` 中位数 | `bwidth90_10_true` 中位数 | 距离配对胜率 | 宽度配对胜率 |
|---|---:|---:|---:|---:|
| `C_baseline` | 0.05025 | 0.30341 | - | - |
| `C_region_slocal` | 0.05108 | 0.21276 | 9/18 | 8/18 |
| `C_region_rho` | 0.05120 | 0.31914 | 6/18 | 11/18 |

latest-valid 事件：

| 分支 | `bdist50_true` 中位数 | `bwidth90_10_true` 中位数 |
|---|---:|---:|
| `C_baseline` | 0.02707 | 0.09112 |
| `C_region_slocal` | 0.02746 | 0.09897 |
| `C_region_rho` | 0.03615 | 0.09879 |

固定 FE stage 结果仅含 `run=3`，只能作补充视角。该口径下基线更清楚领先：

| 分支 | `bdist50_true` 中位数 | `bwidth90_10_true` 中位数 |
|---|---:|---:|
| `C_baseline` | 0.05407 | 0.23531 |
| `C_region_slocal` | 0.06750 | 0.41632 |
| `C_region_rho` | 0.06101 | 0.25004 |

判定：

- `region_slocal` 有局部宽度信号，但主距离没有稳定改善；不能升主线。
- `region_rho` 有时能让宽度变窄，但主距离输得更多；应降低优先级。
- 两个分支都没有在 `bdist50_true -> bwidth90_10_true` 的主判定链上稳定击败 C。
- 条件实现是生效的，但条件描述的是 `BMem.y_b` cloud 内部坐标，不是真实边界法向距离或 pair-side 信息。
- `s_local/rho` 后续只应作为 pair-supported thin boundary 的辅助条件，不应再单独作为主线修复。

### C. 训练集方向的更新

本轮讨论后的训练集结论：

1. 使用 `Problem.GetPF()` 或真实 feasible-region grid 直接训练当前 CGAN 不成立，因为这些是真实 objective-space 边界，不含 decision-space `TrainX`。
2. 用户提出的实际方向是可行的：直接使用当前 `BMem` 已有的 `x_f/y_f` 与 `x_i/y_i` pair 及其对应决策变量，不需要外部 oracle。
3. 不建议把 `TrainX=[x_f;x_i]` 在无 `side/tau` 条件下直接混成一个 real distribution。这样会让生成器学习可行/不可行两侧混合带，可能增加不可行生成和带宽。
4. 推荐第一分支是 `C_pair_thin_slocal`：只改训练集构造，把 `TrainX` 从可行侧 cloud 改成 pair-supported thin feasible-side target，条件先用 `[W(ref,:),s_local]`。
5. 推荐第二分支才考虑 `C_pair_thin_slocal_gap` 或 two-side pair condition：在 `[W,s_local]` 基础上追加 `gap_norm`、`side` 或 `tau`，并且生成时只 query 可行侧。
6. 如果 pair-thin 分支仍不能降低 `bdist50_true` 和 `bwidth90_10_true`，再判断主要瓶颈是 generator/selection 闭环，而不是训练云厚度。

## 当前存在的问题

1. 生成目标仍没有收敛到“窄边界带”。当前 WGAN-GP 生成的仍是目标空间边界附近点云，很多阶段表现为厚带或离散簇，而不是稳定的一条窄边界。`linear150->75 + prescreen8` 在中后期更明显偏离训练集，`s_local/rho` 条件分支也没有稳定改善真实边界距离，说明增加训练强度、critic 筛选或单纯加 scalar 都不能自动解决这个核心问题。
2. critic 分数与目标空间边界质量脱钩。`prescreenMultiplier=8` 的设计会选 critic 分数最高的候选，但当前图像和指标表明 critic 高分样本不一定更贴近可行/不可行边界。`8选1` 还可能放大 critic 对某些离散模式或偏移区域的偏好。
3. 动态训练强度只解决早期问题，没有解决最终问题。`150->30` 在 `FE=10000` 明显改善，但最终 18-run 汇总不优于固定 75；`150->75 + prescreen8` 早期改善但中后期退化。训练步数不是可以单独加大的万能旋钮。
4. 训练集本身可能同时包含不同质量的边界片段。部分边界存档贴近当前更好边界，部分仍停在较差边界；生成器用这些样本学习时容易学成厚带或混合分布。2026-07-08 的结论更进一步：region 版当前只训练 `x_f` cloud，没有利用 `x_i/y_i` 作为 pair-side 监督。
5. `QueryC=random_all_w` 的 query 覆盖数没有变。短记忆收益主要来自训练 BMem 支撑变厚，而不是 QueryC 本身更聪明。对完全未见或极弱训练 region 的泛化能力仍未被单独证明。
6. 上一代 `BMem` 存活率很高，短期是优点，长期可能变成惯性。当前只验证了一代短记忆优于 current-only，尚未量化旧边界相对当前边界移动时是否会污染。
7. Z 采样方差已验证是有效因素，但不是根因闭环。`sampleSigma=0.3` 能缓解厚带和离群，但并不能保证生成“一条窄边界”；如果训练集本身是多层边界或厚带，缩小 Z 或加 `s_local/rho` 只能改变采样和条件划分，不会纠正训练目标。
8. `zDim=6` 当前最稳，但 latent 维度不是主要矛盾。`zDim=4` 的局部最终收益提示低维 latent 可能有助于收缩中位偏移，但 tail 风险仍在。
9. WGAN 事件/阶段汇总已修复，`event_summary_all.csv` 和 `stage_snapshots_all.csv` 已可直接用于分析。`train_history_all.csv` 对 WGAN 为空是当前实现没有记录逐 iter 训练历史，不再视为汇总 bug。

## 下一步需要验证的问题

优先级 1：实现并验证 pair-supported thin feasible-side 训练集。

- 新分支建议命名：`C_pair_thin_slocal`。
- 只改训练集构造，保持 C 主线其余设置：`CBS_RegionWGAN_GP + random_all_w + prev1_fair_union + ganIter=100 + nCritic=2 + zDim=6 + sampleSigma=0.3 + prescreen=1`。
- 训练目标从 `TrainX=BMem.x_b` 改成 pair-supported thin feasible-side target。优先复用旧 `UpdateBoundaryMemory_CBS` 的 `selectThinBoundaryTarget` 思路；若先做保守版本，则保留 gap 小、非支配、每 ref 少量 pair 的 `x_f`。
- `TrainC` 第一版用 `[W(ref,:),s_local]`，其中 `s_local` 应基于 pair thin target 或 pair representative objective 计算，不要分别按 `y_f/y_i` 混用。
- 不把 `x_i` 直接混入无 side 条件的 `TrainX`。

优先级 2：pair-thin 分支的正式实验设计。

- 对照：当前 C 基线 `C_iter100_ncritic2`。
- 新分支：`C_pair_thin_slocal`。
- 暂不同时叠加 `rho`、`gap_norm`、`side/tau`、dynamic `ganIter` 或 prescreen。
- 问题：`LIRCMOP5_BC` 到 `LIRCMOP10_BC`。
- `runs=1:3`，`N=100`，`maxFE=100000`。
- 主判据仍是 `bdist50_true`，其次 `bwidth90_10_true`；`bcover_eps_true` 只作旁证。
- 需要保留 event 级和 stage 级 true-boundary columns。当前 stage snapshot 只捕获单个 `captureRun`，如果要做固定 FE 的全 run 统计，需要扩展 runner 或主要依赖 event-level per-run median。

优先级 3：补充训练集和 condition 诊断。

- 对每个生成事件记录 pair-thin 前后的 `train_count`、每 ref pair 数、pair `gap` 分布、`x_f-x_i` 决策距离分布、thin target 到 `y_f->y_i` segment 的距离。
- 启用或补跑 `conditionDiagGap`，量化同一 `z` 下改变 condition 是否真的改变 decision/objective 输出。
- 保留 `query_count/query_sample_count/query_unique_ref_count`，确认新增条件没有只是在固定 `nGen=30` 下稀释采样。
- 继续记录 `offspringG_survival_rate` 和 `offspringG_feasible_survival_rate`。如果生成点真实边界指标变好但存活率仍为 0，中断点在 selection 闭环。

优先级 4：如果 pair-thin 有收益，再拆第二层条件。

- `C_pair_thin_slocal_gap`：在 `[W,s_local]` 基础上追加 `gap_norm`。
- `C_pair_twoside_slocal`：使用 `[x_f;x_i]` 训练时必须追加 `side` 或 `tau`，并且生成只 query 可行侧。
- `rho` 只作为第三优先级附加条件；上一轮 `region_rho` 已显示单独使用时主距离不稳。

优先级 5：只有在训练集目标对齐后，才回头拆 `dynamic ganIter` 和 `critic prescreen`。

- `150->30` 只证明早期训练强度不足，不证明最终主线要动态调度。
- `linear150->75 + prescreen8` 已证明组合退化；若要继续，必须单独验证固定 `ganIter` + prescreen，以及动态 `ganIter` + no prescreen。
- prescreen 继续前必须记录 raw candidates，并证明 critic score 与边界质量有稳定正相关。

当前不作为主线方向：

- 不做需要额外真实评估的后验边界筛选。
- 不做局部修复或约束投影。
- 不做多代历史累计 BMem。
- 不把 `nGen` 从 30 放大为主线。
- 不把普通 BCE `CBS_RegionCGAN` 作为主线。
- 不把 `Problem.GetPF()` 的真实 objective-space 边界作为正式训练 oracle。
- 不把 `[x_f;x_i]` 在无 `side/tau` 条件下直接混成同一 `TrainX`。
- 不把 `s_local/rho` 单独作为主线修复；它们只能作为 pair-thin 训练集的辅助条件。
- 不把 `150->30`、`150->75 + prescreen8`、`critic prescreen8`、`zDim=4/8`、`sampleSigma=0.15/0.25/0.5/zero` 作为主线，除非后续配对实验能证明它们稳定改善最终窄边界质量。

## 验证与产物核对记录

此前与本轮代码改动对应的验证记录：

```bash
matlab -batch "addpath(genpath('/Users/lanai/Code/Matlab/PlatEMO/PlatEMO')); test_CBS_region_gan_branches"
matlab -batch "addpath(genpath('/Users/lanai/Code/Matlab/PlatEMO/PlatEMO')); test_CBS_RegionCGAN_training_diagnostics_runner"
matlab -batch "addpath(genpath('/Users/lanai/Code/Matlab/PlatEMO/PlatEMO')); checkcode('Algorithms/Multi-objective optimization/CBS-CGAN/CBS_RegionGAN_Base.m','Algorithms/Multi-objective optimization/CBS-CGAN/RunRegionGAN_RC.m','Algorithms/Multi-objective optimization/CBS-CGAN/BoundaryWGAN_RC.m','Algorithms/Multi-objective optimization/CBS-CGAN/Support/run_CBS_RegionCGAN_training_diagnostics.m','Algorithms/Multi-objective optimization/CBS-CGAN/Support/run_CBS_RegionWGAN_GP_qw_plan_branches.m','Algorithms/Multi-objective optimization/CBS-CGAN/test_CBS_region_gan_branches.m')"
```

结果：

- `CBS region GAN branch regressions passed.`
- `test_CBS_RegionCGAN_training_diagnostics_runner` 通过。
- `checkcode` 对动态 `ganIter`、Z 控制、prescreen 和诊断 runner 相关 MATLAB 文件无警告输出。

本次文档更新重新核对的实验产物：

- `linear_iter`：`run_summary.csv` 为 18/18 ok；`stage_snapshots_all.csv` 为 30 行；`event_summary_all.csv` 为 8070 行；`train_history_all.csv` 为 0 行，符合 WGAN 当前不记录逐 iter 历史的实现语义。
- `linear150->75 + prescreen8` 可视化分支：`run_summary.csv` 为 6/6 ok；`stage_snapshots_all.csv` 为 30 行；`figure_manifest.csv` 为 30 行；PNG 为 30 张；阶段 `gan_iter_used=[142,127,112,97,75]`，且诊断表记录 `prescreen_multiplier=8`。
- `iter50_z_factors`：`branch_summary.csv` 记录 6 个分支，均为 18/18 ok，阶段快照均为 30 行。
- `iter50_sigma_tight`：`query_boundary_iter50_z_sigma03` 与 `query_boundary_iter50_z_sigma015` 均为 18/18 ok，阶段快照均为 30 行。
- `iter50_sigma03_zdim_range`：`zDim=4` 与 `zDim=8` 两个分支均为 18/18 ok，阶段快照均为 30 行。
- `visual_run1_query_boundary_sigma03_zdim6_20260630_155715`：`run_summary.csv` 为 6/6 ok；`stage_snapshots_all.csv` 为 30 行；`domain_figure_manifest.csv` 为 30 行。
- `visual_run1_query_boundary_sigma03_zdim6_iter75_20260630_162136`：`run_summary.csv` 为 6/6 ok；`stage_snapshots_all.csv` 为 30 行；`domain_figure_manifest.csv` 为 30 行。
- `ncritic_true_boundary_20260708_105015`：A/B/C/D 四支均为 18/18 ok，阶段快照均为 30 行，均有 true-boundary metrics。
- `condition_slocal_rho_20260708_153442`：`C_region_slocal` 与 `C_region_rho` 均为 18/18 ok；`branch_summary.csv` 的 `ok_run_count=0` 是汇总字段 bug，不能覆盖 `run_summary.csv` 与 `run.log` 的完成事实。
- 源码默认值已核对：`CBS_RegionWGAN_GP.mainlineDefaults()` 当前为 `queryMode=random_all_w, prevBMemMode=prev1_fair_union, bmemBandMode=current, ganIter=100, nCritic=2, zDim=6, sampleSigma=0.3, prescreenMultiplier` 未启用。
- 当前没有后台 MATLAB/parallel 实验进程继续运行；只看到 `fast-context-mcp` 辅助进程。

# CCMO-GAN-BDG fix.md C4-C7 实验记录

记录时间：2026-06-13。本文档记录 `fix.md` 要求的四个正式实验：C4-C7，均使用 `N=100`、`maxFE=100000`、`runs=1:3`、`workers=7`、不生成图片。

## 实现范围

本次只在 CGAN 训练目标与实验 runner 上加控制，不改变 C0 主线默认语义。

| 分支 | 控制字段 | 实验目的 |
|---|---|---|
| `C4_condFilter` | `trainFilterMode="condition_knn"` | 只在 CGAN 训练集中保留条件邻域内决策更一致的 AF-AI pair |
| `C5_aiDomOnlyTrain` | `trainFilterMode="ai_dom_only"` | 只训练 AI 支配 AF 的方向；不足时保留全量并下调 mutual-ND 权重 |
| `C6_refToken` | `conditionMode="yt_dt_ref"` | 在 `[y_t,d_t]` 条件后追加参考向量 token |
| `C7_boundaryQualityEval` | `targetRealLabelMode="boundary_quality_eval"` | 用已评价 AF 目标边界质量给 D 的 real label 加软权重 |

涉及文件：

- `Algorithms/Multi-objective optimization/CCMO-GAN-BDG/CCMO_GAN_BDG.m`
- `Algorithms/Multi-objective optimization/CCMO-GAN-BDG/BoundaryGAN_BDG.m`
- `Algorithms/Multi-objective optimization/CCMO-GAN-BDG/BoundaryQualityTarget_BDG.m`
- `Algorithms/Multi-objective optimization/CCMO-GAN-BDG/BuildBoundaryTargetTriples_BDG.m`
- `Algorithms/Multi-objective optimization/CCMO-GAN-BDG/FilterBoundaryTargetTriples_BDG.m`
- `Algorithms/Multi-objective optimization/CCMO-GAN-BDG/Support/run_CCMO_GAN_BDG_archive_pareto_filters_fullscope.m`
- `Algorithms/Multi-objective optimization/CCMO-GAN-BDG/Support/run_CCMO_GAN_BDG_fixmd_four_experiments.m`

## 数据目录

C0 基线使用已有正式 CSV：

- `Data/CCMO_GAN_BDG/cgan_next4_run3_n100_fe100000_7w_20260612_165839/C0_z4_iter50_g1_full`

本地背景 CSV 已查证：

- C0/C1/C2/C3 同批正式结果：`Data/CCMO_GAN_BDG/cgan_next4_run3_n100_fe100000_7w_20260612_165839/{C0_z4_iter50_g1_full,C1_z4_iter50_g3_full,C2_z4_iter50_g1_anchor,C3_z4_iter50_g1_yonly}`
- U4/C0 结构对照：`Data/CCMO_GAN_BDG/u4_nocap_dir_neighbor4_cgan_runs3_n100_fe100000_8w_20260612_100702/archive_pareto_filter_overall.csv`
- U4 archive 背景：`Data/CCMO_GAN_BDG/u4_archive_next4_runs3_n100_fe100000_8w_20260611_215155/archive_pareto_filter_overall.csv`

C4-C7 正式输出目录：

- `Data/CCMO_GAN_BDG/C4_condFilter_runs3_n100_fe100000_7w_20260613_021845`
- `Data/CCMO_GAN_BDG/C5_aiDomOnlyTrain_runs3_n100_fe100000_7w_20260613_021845`
- `Data/CCMO_GAN_BDG/C6_refToken_runs3_n100_fe100000_7w_20260613_021845`
- `Data/CCMO_GAN_BDG/C7_boundaryQualityEval_runs3_n100_fe100000_7w_20260613_021845`

汇总表：

- `Data/CCMO_GAN_BDG/fixmd_four_summary_20260613_021845/fixmd_overall_compare.csv`
- `Data/CCMO_GAN_BDG/fixmd_four_summary_20260613_021845/fixmd_by_problem_compare.csv`
- `Data/CCMO_GAN_BDG/fixmd_four_summary_20260613_021845/fixmd_focus_problem_compare.csv`

注意：正式 C6/C7 原始 CSV 生成时发现 `trainFilterMode="none"` 路径把 `target_triple_count` 诊断覆盖为 0；算法训练本身未受影响。代码已修正，并用 smoke 验证新输出会写出非零计数。汇总表保留 `target_triple_count_raw`，并额外给出 `target_triple_count_corrected`，C6/C7 用 `target_pair_count` 校正。

## 正式运行状态

四个实验均完成 30/30：

| 分支 | status |
|---|---:|
| `C4_condFilter` | 30 ok |
| `C5_aiDomOnlyTrain` | 30 ok |
| `C6_refToken` | 30 ok |
| `C7_boundaryQualityEval` | 30 ok |

## Overall 结果

| 分支 | BoundaryHit | ObjInterface | SegDist50 | SegDist90 | DecSeg50 | RawFeas | D_real | D_fake | D_mismatch | train_ref_cov | target_rows | runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 0.254 | 0.0474 | 0.871 | 2.130 | 0.0305 | 0.273 | 0.537 | 0.308 | 0.111 | 0.681 | 152.8 | 761.9 |
| C4 | 0.239 | 0.0392 | 0.667 | 1.758 | 0.0313 | 0.333 | 0.473 | 0.333 | 0.112 | 0.480 | 107.1 | 507.6 |
| C5 | 0.124 | 0.0113 | 1.256 | 2.665 | 0.0371 | 0.269 | 0.561 | 0.278 | 0.116 | 0.445 | 79.1 | 424.9 |
| C6 | 0.205 | 0.0329 | 2.090 | 4.037 | 0.0323 | 0.277 | 0.545 | 0.293 | 0.119 | 0.700 | 160.2 | 666.5 |
| C7 | 0.210 | 0.0367 | 0.854 | 2.141 | 0.0288 | 0.262 | 0.562 | 0.322 | 0.115 | 0.663 | 148.8 | 639.2 |

问题级胜场数，相对 C0：

| 分支 | BoundaryHit 更好 | ObjInterface 更好 | SegDist50 更好 | RawFeas 更好 |
|---|---:|---:|---:|---:|
| C4 | 6/10 | 6/10 | 7/10 | 7/10 |
| C5 | 3/10 | 2/10 | 3/10 | 4/10 |
| C6 | 2/10 | 2/10 | 3/10 | 3/10 |
| C7 | 4/10 | 2/10 | 6/10 | 4/10 |

## 重点问题

| 分支 | 问题 | BoundaryHit | ObjInterface | SegDist50 | DecSeg50 | RawFeas |
|---|---|---:|---:|---:|---:|---:|
| C0 | DASCMOP4 | 0.000 | 0.000000 | 4.696 | 0.0102 | 0.013 |
| C4 | DASCMOP4 | 0.040 | 0.000001 | 2.320 | 0.0065 | 0.100 |
| C5 | DASCMOP4 | 0.000 | 0.000000 | 4.625 | 0.0100 | 0.000 |
| C6 | DASCMOP4 | 0.000 | 0.000000 | 6.099 | 0.0105 | 0.000 |
| C7 | DASCMOP4 | 0.000 | 0.000000 | 5.074 | 0.0095 | 0.000 |
| C0 | DASCMOP5 | 0.007 | 0.000129 | 3.152 | 0.0072 | 0.013 |
| C4 | DASCMOP5 | 0.000 | 0.000006 | 2.770 | 0.0063 | 0.013 |
| C7 | DASCMOP5 | 0.013 | 0.000000 | 2.681 | 0.0067 | 0.033 |
| C0 | LIRCMOP9 | 0.653 | 0.306 | 0.018 | 0.0285 | 0.667 |
| C4 | LIRCMOP9 | 0.620 | 0.144 | 0.043 | 0.0181 | 0.540 |
| C6 | LIRCMOP9 | 0.653 | 0.261 | 0.012 | 0.0168 | 0.600 |
| C0 | LIRCMOP10 | 0.800 | 0.073 | 0.018 | 0.0136 | 0.747 |
| C4 | LIRCMOP10 | 0.987 | 0.211 | 0.002 | 0.0059 | 0.793 |
| C7 | LIRCMOP10 | 0.813 | 0.197 | 0.008 | 0.0109 | 0.647 |

## 假设判定

`C4_condFilter`：部分支持，置信度 Medium-High。总体 BoundaryHit 和 ObjInterface 仍低于 C0，但问题级改善最广，且 SegDist50/SegDist90、RawFeas、D_fake 接近 D_real 的方向都有改善。副作用是训练样本从 152.8 降到 107.1，`train_ref_cov` 从 0.681 降到 0.480。

`C5_aiDomOnlyTrain`：不支持，置信度 High。AI-dom hard filter 平均只保留约 45.3% 样本，`target_rows=79.1`、`train_ref_cov=0.445`，BoundaryHit 和 ObjInterface 明显下降。说明单纯要求 AI 支配 AF 会过度牺牲覆盖和局部多样性。

`C6_refToken`：不支持，置信度 High。条件维度从 4 增到 6，`train_ref_cov=0.700`、`target_rows=160.2` 并不缺覆盖，但 SegDist50/90 显著变差。参考向量 token 本身不能解决边界贴合，反而增加了条件学习难度。

`C7_boundaryQualityEval`：基本不支持，置信度 Medium-High。soft label 实际几乎饱和，`boundary_quality_label_mean=0.99985`，没有形成有效权重信号；指标接近但略弱于 C0。若继续该方向，需要重新设计能拉开差异的 label，而不是复用当前评估标签。

## 结论

当前边界贴合不足的主因不是“缺参考向量 token”，也不是“必须只保留 AI 支配 AF 的方向”。更可能的问题是：同一目标条件附近存在多个决策分支和噪声 AF-AI pair，直接把全量 pair 当作等权 real target 会让生成器学到偏宽、偏混合的边界分布。C4 的局部一致性过滤改善了最多问题，支持这个判断；C5 的退化说明硬方向过滤会破坏覆盖；C7 的标签饱和说明当前质量权重没有提供监督梯度。

下一条主线建议：以 C4 为基础做软化版 `condition-local quality weighting`，不再硬删到 70%，而是用条件邻域决策 spread 作为 sample weight，同时加每个 ref 的最低保留约束，避免 `train_ref_cov` 从 0.68 掉到 0.48。目标是保留 C4 的局部贴合收益，同时减少覆盖损失。

## 验证命令

已执行：

```bash
matlab -batch "addpath(genpath(pwd)); test_CCMO_GAN_BDG_target_conditioned"
matlab -batch "addpath(genpath(pwd)); Results=run_CCMO_GAN_BDG_fixmd_four_experiments(7,[],100,[],100000,1:3); disp(Results);"
matlab -batch "T=readtable('Data/CCMO_GAN_BDG/C7_boundaryQualityEval_runs3_n100_fe100000_7w_20260613_021845/archive_pareto_filter_run_summary.csv','TextType','string'); disp(groupsummary(T,'status'));"
```

诊断补丁 smoke：

```bash
matlab -batch "addpath(genpath(pwd)); ... run_CCMO_GAN_BDG_archive_pareto_filters_fullscope(...,'c6_reftoken',...); ... run_CCMO_GAN_BDG_archive_pareto_filters_fullscope(...,'c7_boundaryqualityeval',...);"
```

结果：C6 smoke `target_pair_count=6,target_triple_count=6`；C7 smoke `target_pair_count=12,target_triple_count=12`。

# C4 机制验证实验（2026-06-14）

## 目的

验证 `condition_knn` hard filter 的收益到底来自“删除 condition 冲突 triples”，还是只是来自“训练样本变少”。C0 与旧 C4_keep70 不重跑，直接复用已有正式结果；新增 C4_keep90/80/60 与 Rand_keep70/80。

## 新增实现

- `FilterBoundaryTargetTriples_BDG.m`：新增 `trainFilterMode="random_keep"`，按 seed 可复现随机保留指定比例。
- `CCMO_GAN_BDG.m`：暴露 `conditionKNNRetainRatio` 和 `trainFilterRandomSeed`，不再在训练集构造处硬编码 0.70。
- `run_CCMO_GAN_BDG_archive_pareto_filters_fullscope.m`：新增 `c4_keep90`、`c4_keep80`、`c4_keep60`、`rand_keep70`、`rand_keep80`、`c4_mechanism` variant set。
- `run_CCMO_GAN_BDG_c4_mechanism_experiments.m`：一键跑 5 个独立时间戳目录。
- `summarize_CCMO_GAN_BDG_c4_mechanism_results.m`：统一汇总 C0、C4_keep70 和 5 个新实验。

## 数据目录

复用：

- C0：`Data/CCMO_GAN_BDG/cgan_next4_run3_n100_fe100000_7w_20260612_165839/C0_z4_iter50_g1_full`
- C4_keep70：`Data/CCMO_GAN_BDG/C4_condFilter_runs3_n100_fe100000_7w_20260613_021845`

新增正式输出：

- `Data/CCMO_GAN_BDG/C4_keep90_runs3_n100_fe100000_7w_20260614_001027`
- `Data/CCMO_GAN_BDG/C4_keep80_runs3_n100_fe100000_7w_20260614_001027`
- `Data/CCMO_GAN_BDG/C4_keep60_runs3_n100_fe100000_7w_20260614_001027`
- `Data/CCMO_GAN_BDG/Rand_keep70_runs3_n100_fe100000_7w_20260614_001027`
- `Data/CCMO_GAN_BDG/Rand_keep80_runs3_n100_fe100000_7w_20260614_001027`

汇总：

- `Data/CCMO_GAN_BDG/c4_mechanism_summary_20260614_040649/c4_mechanism_overall_selected.csv`
- `Data/CCMO_GAN_BDG/c4_mechanism_summary_20260614_040649/c4_mechanism_diagnostic.csv`
- `Data/CCMO_GAN_BDG/c4_mechanism_summary_20260614_040649/c4_mechanism_by_problem_selected.csv`
- `Data/CCMO_GAN_BDG/c4_mechanism_summary_20260614_040649/c4_mechanism_focus_problems.csv`
- `Data/CCMO_GAN_BDG/c4_mechanism_summary_20260614_040649/c4_mechanism_pairwise_delta.csv`

## 正式运行状态

新增 5 个实验均完成 30/30：

| 分支 | status |
|---|---:|
| `C4_keep90` | 30 ok |
| `C4_keep80` | 30 ok |
| `C4_keep60` | 30 ok |
| `Rand_keep70` | 30 ok |
| `Rand_keep80` | 30 ok |

## Overall 结果

| 分支 | keep | BoundaryHit | ObjInterface | Seg50 | Seg90 | DecSeg50 | RawFeas | D_real | D_fake | D_mismatch | D_real_acc | D_fake_acc | D_mismatch_acc | train_ref_cov | AF_ref_cov | target_rows | runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 1.00 | 0.254 | 0.0474 | 0.871 | 2.130 | 0.0305 | 0.273 | 0.537 | 0.308 | 0.111 | 0.463 | 0.923 | 0.962 | 0.681 | 0.681 | 152.8 | 761.9 |
| C4_keep70 | 0.70 | 0.239 | 0.0392 | 0.667 | 1.758 | 0.0313 | 0.333 | 0.473 | 0.333 | 0.112 | 0.333 | 0.934 | 0.980 | 0.480 | 0.676 | 107.1 | 507.6 |
| C4_keep90 | 0.90 | 0.212 | 0.0255 | 1.306 | 2.399 | 0.0302 | 0.270 | 0.521 | 0.316 | 0.117 | 0.398 | 0.924 | 0.968 | 0.616 | 0.670 | 137.2 | 650.9 |
| C4_keep80 | 0.80 | 0.281 | 0.0513 | 1.310 | 2.481 | 0.0288 | 0.301 | 0.495 | 0.323 | 0.107 | 0.379 | 0.918 | 0.974 | 0.565 | 0.686 | 124.2 | 552.3 |
| C4_keep60 | 0.60 | 0.263 | 0.0348 | 0.616 | 1.411 | 0.0237 | 0.320 | 0.498 | 0.339 | 0.116 | 0.395 | 0.922 | 0.971 | 0.424 | 0.681 | 89.4 | 517.2 |
| Rand_keep70 | 0.70 | 0.257 | 0.0229 | 1.255 | 2.509 | 0.0316 | 0.291 | 0.514 | 0.303 | 0.106 | 0.399 | 0.938 | 0.971 | 0.577 | 0.676 | 107.3 | 518.7 |
| Rand_keep80 | 0.80 | 0.263 | 0.0457 | 1.450 | 3.011 | 0.0332 | 0.261 | 0.517 | 0.303 | 0.123 | 0.423 | 0.934 | 0.967 | 0.607 | 0.663 | 118.6 | 548.3 |

## 机制诊断

| 分支 | pre rows | post rows | actual keep | pre ref cov | post ref cov | spread mean | spread median | spread p90 | train_ref_cov |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | 0.681 |
| C4_keep70 | 152.3 | 107.1 | 0.704 | 0.676 | 0.480 | 0.380 | 0.302 | 0.687 | 0.480 |
| C4_keep90 | 152.0 | 137.2 | 0.905 | 0.670 | 0.616 | 0.386 | 0.301 | 0.700 | 0.616 |
| C4_keep80 | 154.8 | 124.2 | 0.805 | 0.686 | 0.565 | 0.388 | 0.303 | 0.718 | 0.565 |
| C4_keep60 | 148.5 | 89.4 | 0.603 | 0.681 | 0.424 | 0.372 | 0.289 | 0.720 | 0.424 |
| Rand_keep70 | 152.7 | 107.3 | 0.703 | 0.676 | 0.577 | n/a | n/a | n/a | 0.577 |
| Rand_keep80 | 147.6 | 118.6 | 0.804 | 0.663 | 0.607 | n/a | n/a | n/a | 0.607 |

## 关键差值

| 对比 | BoundaryHit | ObjInterface | Seg50 | Seg90 | DecSeg50 | RawFeas | train_ref_cov |
|---|---:|---:|---:|---:|---:|---:|---:|
| C4_keep80 - C4_keep70 | +0.041 | +0.012 | +0.643 | +0.722 | -0.0025 | -0.033 | +0.085 |
| C4_keep90 - C4_keep80 | -0.069 | -0.026 | -0.004 | -0.082 | +0.0014 | -0.031 | +0.051 |
| C4_keep60 - C4_keep70 | +0.023 | -0.004 | -0.051 | -0.347 | -0.0075 | -0.013 | -0.056 |
| C4_keep70 - Rand_keep70 | -0.018 | +0.016 | -0.588 | -0.751 | -0.0003 | +0.043 | -0.097 |
| C4_keep80 - Rand_keep80 | +0.017 | +0.006 | -0.140 | -0.531 | -0.0044 | +0.039 | -0.041 |

注：Seg/DecSeg 越低越好；BoundaryHit、ObjInterface、RawFeas、coverage 越高越好。

## 重点问题观察

- DASCMOP4：C4_keep80 与 Rand_keep80 的 BoundaryHit 都到 0.333，但 C4_keep80 的 Seg90=9.166 优于 Rand_keep80=10.056；C4_keep70 的 Seg90 最好 5.831，但 BoundaryHit 只有 0.040。
- DASCMOP5：Rand_keep70 的 BoundaryHit=0.333，但 Seg90=12.562 很差；C4_keep60 的 Seg90 最好 6.560，C4_keep70 次优 7.083。该问题不支持“更高 keep 一定更好”。
- LIRCMOP9：C4_keep60 最强，BoundaryHit=0.833、Seg90=0.0558；C4_keep80 也明显优于 Rand_keep80。说明强过滤在该问题上确实删除了有害冲突样本。
- LIRCMOP10：C4_keep70 最强，BoundaryHit=0.987、Seg90=0.0156；随机保留明显弱于 C4。说明 condition-kNN 信号在该问题上有效。

## 结论

`C4_keep70` 不是简单因为“样本变少”才有效。相同比例下，C4_keep70/80 都比 Rand_keep70/80 有更好的 Seg90、DecSeg50、RawFeas 与 ObjInterface，尤其 LIRCMOP9/10 明显支持 condition consistency 机制。这个机制成立，但不是全局无副作用。

70% keep 有过删风险：`train_ref_cov` 从 C0 的 0.681 掉到 0.480，明显低于 Rand_keep70 的 0.577；这说明 condition-kNN 会集中删掉部分 ref 区域，coverage 代价是真实存在的。

80% keep 更适合作下一主线候选：相比 70%，BoundaryHit 和 ObjInterface 更好，`train_ref_cov` 从 0.480 回升到 0.565，DecSeg50 更好；代价是 Seg50/Seg90 退化。它更像“覆盖友好版本”，但不是贴边距离最优版本。

90% keep 不够：coverage 更高，但 BoundaryHit、ObjInterface、RawFeas 都弱于 80%，Seg50/90 也没有优势，说明只删极端冲突样本不足以稳定收益。

60% keep 是强贴边但高风险：Seg50/90 和 DecSeg50 最好，RawFeas 也高，但 `train_ref_cov=0.424`、target_rows=89.4，覆盖损失比 70% 更重。适合作机制上限参考，不适合作主线。

下一步最有价值的是 `C4_keep80 + ref guard`，而不是立刻全面 soft weighting。原因是 hard keep80 已经显示出更好的 BoundaryHit/ObjInterface/coverage 折中，但 Seg 距离不如 70/60；先加每个 ref 最低保留约束，验证能否保住 70/60 的贴边优势同时避免 coverage 掉到 0.48/0.42。如果 ref guard 仍不足，再进入 spread-based soft weighting。

## 验证命令

已执行：

```bash
matlab -batch "addpath(genpath(pwd)); test_CCMO_GAN_BDG_target_conditioned"
matlab -batch "addpath(genpath(pwd)); Results=run_CCMO_GAN_BDG_c4_mechanism_experiments(7); disp(Results); assert(all(Results.status==\"ok\"));"
matlab -batch "addpath(genpath(pwd)); SummaryDir=summarize_CCMO_GAN_BDG_c4_mechanism_results(); disp(SummaryDir);"
```

smoke 覆盖：

- `conditionKNNRetainRatio=0.90/0.80/0.60` 能改变实际保留比例。
- `random_keep` 按 seed 可复现。
- fullscope `c4_mechanism` 1 问题 smoke 5/5 ok。
- no plot：smoke 未生成 `archive_objective_snapshots`。

  

你来根据ZIP文件内容向GPT提问，要求他来分析下面的问题、提出解决方法；（注意：你仅仅给出客观事实，不要加入主观猜测）；你要以最严格的标准要求GPT；

核心目的：边界解训练CGAN，CGAN生成边界解！！！（生成一条边界而非点云厚带）

遇到的问题：

1:比如图片展示，一部分边界存档在一个边界，而另一部分边界存档在更好的边界；

2:通过部分边界解的学习来学到当前边界的分布，然后具有直接生成还未探索到的当前边界的能力，此问题关系到我们算法的核心价值，请你仔细分析！

3:CGAN生成的解不够贴边，我们需要的是生成一条相对较窄的边界带而不是厚重的离散点云

4:{你来总结当起面临的其余问题}

核心约束：

1:核心创新点不能偏离（必须用CGAN，必须直接生成完整的解（决策变量），所谓的边界必须仅指目标空间的可行不可行边界）；

3:对于算法的设计来说：禁止随意添加不知道是否有效的参数或这模块，禁止做没有把握的猜测和修改，非必要不要随意修改我们的主线、机制和模块（除非你有明确的更好的替代）优先做统一和收敛，而不是分支和堆料；

4:你的重心不是分析现有的算法/实验，而是根据现有的算法/实验来分析上述问题，解决问题。
