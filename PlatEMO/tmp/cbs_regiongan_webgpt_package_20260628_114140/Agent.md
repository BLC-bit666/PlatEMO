# CBS-RegionGAN 当前主线状态（2026-06-27）

## 当前主线决策

Region-conditioned GAN 分支当前主线确定为：

- 算法：`CBS_RegionWGAN_GP`
- GAN 类型：`wgan-gp`
- `queryMode`：`random_all_w`
- `prevBMemMode`：`prev1_fair_union`，上一代 `BMem` 的可行 anchor 在配对前并入当前可行候选池，与当前 `[Population1, Offspring1, Population2, Offspring2]` 的可行解一起竞争
- `nGen`：30，每代 GAN 最多注入 30 个生成解
- `ganMiniBatch`：32
- `minGANTrainCount`：32，训练集少于 32 行时跳过本代 GAN 训练与生成
- `maxAnchorsPerRef`：5，在配对不可行解之前，每个 ref 最多保留 5 个按 `CalFitness_CBS` 排序最优的可行边界 anchor
- WGAN-GP 参数：`gpLambda=10`，`nCritic=5`

这条主线对应两步实验结论和一次机制修正：先由 `mb32 + CBS_RegionWGAN_GP` 确定 WGAN-GP 主分支，再由 6 问题、`runs=3` 的 `current_only` vs 旧版 `prev1_fair_union` 实验证明上一代 `BMem` 短记忆能提高训练覆盖；随后图像诊断暴露“整条 BMem row 晚合并”会沿用旧配对结构，因此当前主线已改为“上一代可行 anchor 配对前合并，且只与当前不可行解重新配对”。普通 `CBS_RegionCGAN` 保留为 BCE/sigmoid CGAN 对照分支，不再作为当前主线。

## 为什么形成这条主线

这条主线不是从“目标空间点 y 反推唯一决策变量 x”的逆映射思路来的，而是从传统条件生成模型的思想收敛来的：

- 早期 `ref_y` / `ref_y_tau` 方向把条件写成 `[参考方向, 归一化目标 y, 可选 tau]`，再用 `G(z,c)->x` 生成决策变量。这个设计在诊断上逐渐暴露出一个问题：当 `c` 里包含很具体的目标位置 y，且 `z=0` 或固定 z 经常用于重建时，模型很容易退化成“给定 y 找一个 x”的确定性逆映射。
- 逆映射不是当前目标。我们真正想要的是：在当前目标空间的可行/不可行边界附近，按粗区域学习一团决策变量分布 `p(x | region)`，再由随机 `z` 在该区域内采样不同的边界候选。也就是说，`c` 只负责告诉生成器“目标空间的大致区域/参考分区”，`z` 负责表达同一区域内的决策空间多样性。
- 这也是为什么后来改成 RegionCGAN/RegionWGAN：条件从精确 y 收缩为粗 region/ref 条件，训练目标从“重建某个 y 对应的 x”转为“拟合该 region 内当前边界决策云的分布”。这更接近传统 CGAN 的分布拟合逻辑。

这里的“传统 CGAN 思想”不等于必须坚持 BCE/sigmoid 损失。普通 BCE CGAN 是对照分支；主线采用 WGAN-GP，是因为它仍然学习条件分布 `p(x | region)`，但把对抗损失换成更适合小样本在线训练的 Wasserstein critic + gradient penalty。

关键设计取舍如下：

1. `z` 不再固定为主线用法。固定 z 只能诊断生成器是否能重建/回放条件，不能代表生成分布；主线采样必须使用随机 z。
2. 条件 `c` 不再承担精确目标坐标的逆映射职责，而是表示粗目标区域。当前主线用 `queryMode=random_all_w` 在所有 ref 上随机抽 QueryC，避免只在已有训练条件附近回放。
3. 不做无限跨代累计训练集。边界会随进化过程移动，长期历史会污染当前分布；当前只把上一代 `BMem` 的可行 anchor 作为短记忆，在配对前与当前可行候选公平竞争，过旧边界和旧不可行配对不进入主线。
4. 训练集也不能是每个 ref 一个点的薄骨架。当前用前二可行 PF + 每 ref 最多 5 个质量门控 anchor，目的是在不引入远端长尾污染的情况下保留区域内可学习的边界云。
5. 网络必须小。单次刷新样本量受当前种群窗口限制，不能靠跨代累计扩样；因此采用 `[32 32]` 小网络。
6. `zDim=6` 来自之前对密集边界云的 SVD 诊断：区域内决策云内在维大约 3-5，`z=2` 偏小，`z=8` 略宽，6 是保守折中。
7. WGAN-GP 优先于 BCE CGAN。BCE 诊断能看到 D/G 对抗过程，但在小样本、窄边界、易模式坍缩的设置下不够稳；WGAN-GP 的线性 critic 和梯度惩罚更适合作为主线。

因此，当前主线可以概括为：**用传统条件生成分布的思想学习 `p(x | coarse objective region)`，但用 WGAN-GP 训练而不是 BCE CGAN；避免确定性逆映射，避免长期历史累计污染，用上一代 BMem 可行 anchor 的短记忆缓解 ref 覆盖不足，并用当前不可行解重新定义边界配对。**

## 已落地代码

- `Algorithms/Multi-objective optimization/CBS-CGAN/CBS_RegionWGAN_GP.m`
  - 角色从 comparison branch 明确改为 mainline branch。
  - 新增 `CBS_RegionWGAN_GP.mainlineDefaults()`，集中固定当前主线默认参数。
  - `main()` 通过 `mainlineDefaults()` 注入默认参数，当前锁定 `queryMode=random_all_w` 与 `prevBMemMode=prev1_fair_union`，避免后续手改 `ParameterSet` 时不小心偏离主线。
- `Algorithms/Multi-objective optimization/CBS-CGAN/CBS_RegionGAN_Base.m`
  - `QueryMode` 默认值改为可从 `Config.queryMode` 读取。
  - `prevBMemMode` 可从 `Config.prevBMemMode` 或实验控制读取；若分支未提供，仍保持旧默认 `current_only`，因此 `CBS_RegionCGAN` 的对照语义不被强行改变。
- `Algorithms/Multi-objective optimization/CBS-CGAN/UpdateBoundaryMemory_RC.m`
  - `prev1_fair_union` 将上一代 `BMem` 的可行 anchor 在配对前注入可行候选池，与当前可行解一起做前 `frontDepth=2` 层筛选和每 ref 最多 5 个 anchor 的 cap。
  - 配对阶段只使用当前 `[Population1, Offspring1, Population2, Offspring2]` 中的不可行解；上一代 `BMem` 的旧 `x_i/y_i/gap` 不再整条沿用。
- `Algorithms/Multi-objective optimization/CBS-CGAN/Support/run_CBS_RegionCGAN_training_diagnostics.m`
  - 诊断 runner 在 `algorithmClass=CBS_RegionWGAN_GP` 且未显式覆盖时，默认跟随 `CBS_RegionWGAN_GP.mainlineDefaults()`。
- `Algorithms/Multi-objective optimization/CBS-CGAN/test_CBS_region_gan_branches.m`
  - 主线默认参数回归测试锁定 `wgan-gp + random_all_w + prev1_fair_union + ganMiniBatch=32 + minGANTrainCount=32 + nGen=30 + maxAnchorsPerRef=5 + gpLambda=10 + nCritic=5`。

## 训练集与生成流程现状

边界训练集仍按当前简化门控构造：

1. 从 `[Population1, Offspring1, Population2, Offspring2]` 中收集当前候选。
2. 先筛选可行解。
3. 对可行解做目标空间非支配排序，保留前 `frontDepth=2` 层。
4. 在配对不可行解之前，对每个 ref 按 `CalFitness_CBS` 升序最多保留 `maxAnchorsPerRef=5` 个可行 anchor。
5. 若启用 `prev1_fair_union`，把上一代 `BMem.x_f/y_f` 作为可行 anchor 加入当前可行候选池；不把上一代 `x_i/y_i/gap` 作为已有配对直接带入。
6. 在合并后的可行候选池上执行前二 PF 和每 ref 最多 `maxAnchorsPerRef=5` 的 anchor cap。
7. 将保留的可行 anchor 与当前窗口里的邻近不可行解重新配对，形成新边界 cloud，再执行 adaptive gap cap，得到新 `BMem` 与 `TrainX/TrainC`。
8. 若 `TrainX < minGANTrainCount=32`，本代跳过 WGAN-GP 训练和生成。
9. 若训练样本充足，则用 WGAN-GP 训练，并在所有 ref 上完全随机抽取 `QueryC=random_all_w`，总生成数受 `nGen=30` 限制。

注意：当前只保留上一代 `BMem` 这一层短记忆，不做多代历史累计，也不做需要额外真实评估的后验边界筛选或局部修复。

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

因此当前主线仍更新为 `mb32 + CBS_RegionWGAN_GP + random_all_w + prev1_fair_union`，但 `prev1_fair_union` 的实现语义已从“整条 BMem row 晚合并”改成“上一代可行 anchor 配对前合并并用当前不可行解重配”。旧实验验证了短记忆有价值，但不能作为新合并时机的最终验证；需要重跑 6 问题图像与指标。

## 当前存在的问题

1. LIRCMOP9 仍是主要困难点。`prev1_fair_union` 将 LIRCMOP9 的 `gap_ratio90` 从 91.4047 降到 60.6574，说明极端漂移被压住；但 `near_boundary_rate_gap1` 从 0.2767 降到 0.2566，说明局部贴边比例没有改善。
2. 生成解可行率略降。整体 `feasible_rate` 从 0.3725 降到 0.3540，18 个 run 中只有 7 个 run 改善。边界附近生成点跨到不可行侧并不必然是坏事，但当前实验没有保存最终种群，不能证明这些点会被环境选择完全淘汰。
3. `QueryC=random_all_w` 的 query 覆盖数没有变。两组 `query_unique_ref_count` 都约 22.7/30，所以收益主要来自训练 BMem 支撑变厚，而不是 QueryC 本身更聪明。对完全未见或极弱训练 region 的泛化能力仍未被单独证明。
4. 上一代 `BMem` 存活率很高，短期是优点，长期可能变成惯性。当前只验证了一代短记忆优于 current-only，尚未量化旧边界相对当前边界移动时是否会污染。
5. WGAN critic 分数仍不能说明生成器学好了。`better_than_random_gap_rate` 两组都接近饱和，`current_only=0.9947`、`prev1_fair_union=0.9960`，无法区分真实贴边质量。必须继续使用边界距离、贴边率、目标 region 命中等外部诊断。
6. `z` 多样性、网络容量、训练步数不足仍只是候选解释。当前实验证明覆盖短板很重要，但没有排除 `zDim`、`ganIter`、网络宽度、`nCritic` 对 LIRCMOP9 和个别 run 漂移的影响。
7. 诊断 runner 汇总存在字段兼容问题。本次 WGAN 结果实际保存在 `metric.region_gan_history` / `metric.region_wgan_gp_history`，但 `event_summary_all.csv`、`train_history_all.csv`、`stage_snapshots_all.csv` 为空，需要后处理 `metric.mat` 才能得到事件级表。这个问题不影响实验结论，但影响复现实验分析效率。

## 需要验证的问题

优先级 1：最终种群是否真的受益。下一次实验需要保存或导出环境选择前后的 `OffspringG` 存活标记、最终 Population 可行率、IGD/HV 或现有 PlatEMO metric。目的不是做后验筛选，而是验证“坏生成点会被环境选择淘汰”这个假设。

优先级 2：LIRCMOP9 为什么仍不贴边。建议只对 LIRCMOP9 加一个小诊断：比较 `BMem` 本身的 gap 分布、G 到目标 ref 邻域距离、G 到 segment 距离、generated feasible/infeasible 分布。如果训练 BMem 本身不贴边，就先修 BMem；如果 BMem 贴边但 G 不贴边，再看 GAN 训练。

优先级 3：验证 critic 分数和边界质量的脱钩。对每次生成事件同时记录 `score_real - score_fake`、`gap_ratio90`、`near_boundary_rate_gap1` 的相关性。现有事件级统计已经显示 critic 指标不能替代边界指标，但应在 runner 中正式输出。

优先级 4：快速 ablation 验证 `z` 多样性、容量、训练不足。不要大而全，先选 LIRCMOP6、LIRCMOP9 两个代表问题，`runs=2`，只比较：

- `zDim=3,6,8`
- `ganIter=25,50,100`
- generator/critic hidden `[32 32]` vs `[64 64]`

主指标仍是 `gap_ratio90`、`near_boundary_rate_gap1`、`feasible_rate`、生成触发率和运行时间。

优先级 5：修复诊断 runner 汇总字段。让 runner 对 WGAN 读取 `region_gan_history` 或 `region_wgan_gp_history`，并把 `prev_bmem_candidate_count`、`prev_bmem_survivor_count` 输出到事件汇总表。这样后续不用手工从 `metric.mat` 补导出。

当前不作为主线方向：

- 不做需要额外真实评估的后验边界筛选。
- 不做局部修复或约束投影。
- 不做多代历史累计 BMem。
- 不把 `nGen` 从 30 放大为主线。
- 不把普通 BCE `CBS_RegionCGAN` 作为主线。

## 本次更新验证

已执行：

```bash
matlab -batch "addpath(genpath(pwd)); test_CBS_region_gan_branches"
matlab -batch "addpath(genpath(pwd)); test_CBS_region_boundary_ref_cap"
matlab -batch "addpath(genpath(pwd)); test_CBS_region_random_query_boundary_runner"
matlab -batch "addpath(genpath(pwd)); checkcode('Algorithms/Multi-objective optimization/CBS-CGAN/CBS_RegionWGAN_GP.m','Algorithms/Multi-objective optimization/CBS-CGAN/CBS_RegionGAN_Base.m','Algorithms/Multi-objective optimization/CBS-CGAN/Support/run_CBS_RegionCGAN_training_diagnostics.m','Algorithms/Multi-objective optimization/CBS-CGAN/test_CBS_region_gan_branches.m')"
```

结果：

- `CBS region GAN branch regressions passed.`
- `CBS RegionGAN boundary ref-cap regressions passed.`
- `CBS RegionCGAN random-query boundary runner smoke passed.`
- `checkcode` 对四个相关 MATLAB 文件无警告输出。

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
