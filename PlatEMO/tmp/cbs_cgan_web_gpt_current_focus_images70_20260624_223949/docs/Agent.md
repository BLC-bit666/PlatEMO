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
