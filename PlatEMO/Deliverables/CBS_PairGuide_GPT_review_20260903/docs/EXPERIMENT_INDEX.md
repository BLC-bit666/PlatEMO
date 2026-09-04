# 实验索引与当前决策

## 当前有效实验

| 目录 | 核心问题 | 协议 | 关键资料 |
|---|---|---|---|
| `first_use_epoch_v4_20260901` | 首次训练 Epoch 对首次生成/使用形状和质量的影响 | 6 问题；Epoch 400/600/800/1000/1200；2 runs；首次使用后停止 | `analysis/*.csv`、`figures/`、`figure_index.csv` |
| `training_schedule_v1_20260901` | Critic:Generator 与后续 Epoch | 6 问题；30 个配对块；先比较 nCritic 1/2/4，再比较重训 5/10/20 | 两阶段 `analysis/`、约 20 MB 训练诊断 CSV、3×10 细节图 |
| `inference_sigma_epoch1000_v1_20260901` | 只改变推理噪声能否减薄点云 | 6 问题×2 runs；train sigma=1；use sigma=1/0.5 | `analysis/summary.txt`、配对 CSV、12 张对照图 |
| `train_use_sigma_threeway_epoch1000_v1_20260901` | 训练噪声也降至 0.5 是否继续减薄 | 6 问题×2 runs；三方案并列 | `analysis/three_way_summary.txt`、三路 CSV 和图片 |
| `mainline_vs_DE20_v1_20260902` | 20 个 PairGuide 名额是否优于普通 DE | 6 问题×10 runs；N=100、D=30、FE=100000、save=10；首次/后续 Epoch=500/10；5C:1G | 10 检查点 IGD、最终问题表 |
| `quota_sweep_v1_20260903` | PairGuide 名额 20/30/40/50 | 同上；20 使用前项基线，新增 30/40/50 共 180 runs | 600 组配对轨迹、60 个问题检查点、最终汇总 |

`DE20_pilot_v1_20260902` 是 runs=5 的先导实验，已被 runs=10 的 `mainline_vs_DE20_v1_20260902` 取代，只用于追踪实验演进。

## 历史 Epoch 摘要

- `first_use_epoch_v1_historical`：曾比较 10/25/50/100/200/400/800，5 runs，旧结论为 400。
- `first_use_epoch_v3_historical`：曾比较 200/400/700/1000，5 runs，旧结论为 700。
- v4 在更新后的范围 400–1200 上按既定综合规则推荐 1000。
- 当前研究决策把首次 Epoch 暂定为 500，原因是首次训练数据量较小；500 尚未在同一完整协议中单独完成最终验证。

历史结果对应不同代码快照或分析阶段，不能直接合并做单一统计检验。

## 当前人工锁定值与旧实验建议的区别

早期 `training_schedule_v1` 数值流程曾选择 `nCritic=4`、后续 `Epoch=20`。后续研究决策暂定为：

- `nCritic=5`，即 5C:1G；
- 首次 Epoch=500；
- 后续 Epoch=10；
- PairGuide 子代名额=20%。

最新 DE20 和 quota 实验均使用这组当前值，因此其内部比较公平，但不能反向证明 5C:1G 或 Epoch=500/10 本身最优。

## 已有结果要点

### PairGuide 对普通 DE

在 FE=100000 的六问题、10 runs 配对实验中，PairGuide 与 DE20 呈问题依赖差异，没有形成全局显著优势。LIRCMOP8 更支持 PairGuide；LIRCMOP10 更偏向 DE20；其余问题混合。

### 配额

- 最终六问题均值：20 在 LIRCMOP5/8/10 最好；30 在 LIRCMOP7 最好；50 在 LIRCMOP12 最好；40 在 LIRCMOP14 最好。
- 最终 60 个配对：30、40、50 相对 20 的胜负为 31:29、29:31、26:34。
- 高配额存在更明显尾部失败，没有单调收益；当前保留 20。

### 点云厚度

- train sigma=1 时，把 use sigma 从 1 降到 0.5：决策厚度比中位数 0.60293，目标厚度比 0.59731。
- 对比 train1/use0.5，把训练也改为 train0.5/use0.5：决策厚度比中位数 1.3212，目标厚度比 1.3104，通常更厚。
- 因而采样噪声能控制厚度，但当前证据没有证明点云中心位于正确约束分界面。

## 解释限制

- `mainline_vs_DE20` 和 `quota_sweep` 只以 IGD 作为算法结果指标。
- first-use 与 sigma 实验主要观察生成候选和引导子代几何/可行性，不是完整预算最终性能实验。
- 多个检查点共享同一 run，不能当成独立样本。
- PNG 显示的是二维或三维目标空间投影，不等价于 20/30 维决策空间的真实流形。

