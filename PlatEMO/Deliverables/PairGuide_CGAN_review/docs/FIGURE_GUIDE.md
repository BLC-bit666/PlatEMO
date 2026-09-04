# 图片阅读指南

## 图例

当前实验图通常使用：

- 浅蓝小点：500 个原始 CGAN 候选。算法内不评价；为了画图才离线计算目标与约束。
- 橙色菱形：由 PairGuide 真正产生并评价的引导子代。
- 紫色圆点：约束种群 P1。
- 青色正三角：无约束种群 P2。
- 绿色背景：可行目标区域。
- 粉红背景：不可行目标区域。
- 某些历史参考图中的粉色倒三角是另一类生成/采样点，不能与当前浅蓝 raw CGAN 点自动等同。

## 建议先看

1. `experiments/first_use_epoch_v4_20260901/figures/`
   - 每张图展示某个问题、首次 Epoch、run 在第一次实际使用时的 P1、P2、500 个原始候选和真正引导子代。
   - 本精简附件为每个问题保留 400/1000 Epoch 的 run01；完整统计范围仍见 `analysis/` 与 `figure_index.csv`。
   - LIRCMOP14 为三目标 3D 图，其余主要为二维目标图。
2. `experiments/training_schedule_v1_20260901/figures/training_schedule_detail_v1/`
   - 每张 3×10 图在同一放大范围比较三个训练配置和 10 个 FE 检查点。
   - `ratio_*` 比较 nCritic；`epoch_*` 比较后续重训 Epoch。
3. `experiments/inference_sigma_epoch1000_v1_20260901/figures/`
   - 同一个已训练模型的 use sigma=1 与 0.5 对照。
4. `experiments/train_use_sigma_threeway_epoch1000_v1_20260901/figures/`
   - train/use sigma 三方案并列，区分训练分布与推理采样范围。
5. `figures/user_references/`
   - 用户提供的显示范围和历史图示参考，不是本次最新实验结果。

## 重要限制

- “厚状点云”首先是目标空间中的视觉现象。生成器直接输出 D 维决策向量；约束映射和目标映射都可能改变投影厚度。
- 背景可行/不可行区域只在可绘制的目标维度上表达；同一目标位置可能对应不同决策及约束状态。
- 对原始候选的离线评价只用于诊断，不计搜索 FE、不更新档案、不参与选择。
- 判断是否学到分界线时，应同时检查决策空间 pair 法向坐标、到 `xf/xi` 线段或显式边界估计的距离，而不能只看二维散点图。
