# CBS-CGAN Web GPT Package

生成时间：2026-06-24 21:49:57 Asia/Shanghai

这个包用于把当前 CBS-CGAN 分支源码和实验结果交给网页版 GPT 复查。重点看：

- 当前算法源码：`Algorithms/Multi-objective optimization/CBS-CGAN/`
- 需求与讨论文档：`fix.md`、`fix1.md`、`fix2.md`
- 当前分支状态与差异：`metadata/git_status.txt`、`metadata/git_diff.patch`
- 实验结果：`Data/CBS_CGAN/`

重点实验目录：

- `Data/CBS_CGAN/A_ref_only_adv_LIR6_run1_nGen30_20260623_220259`
- `Data/CBS_CGAN/B_ref_only_adv_huber_LIR6_run1_nGen30_20260623_234001`
- `Data/CBS_CGAN/train_quality_confirm_epoch20_ABCD_LIR6_run1_20260624_122052`
- `Data/CBS_CGAN/train_quality_mechanism_sweep_LIR6_run1_20260624_094837`
- `Data/CBS_CGAN/train_quality_epoch_sweep_D_default_20260624_130858`
- `Data/CBS_CGAN/train_quality_pair6_vs_pair3_D_default_epoch50_20260624_153705`
- `Data/CBS_CGAN/train_quality_endpoint_vs_default_D_epoch50_20260624_181601`
- `Data/CBS_CGAN/train_quality_endpoint_yb_norm_vs_default_D_epoch50_20260624_192053`
- `Data/CBS_CGAN/single_stage_overfit_endpoint_yb_norm_20260624_203803`
- `Data/CBS_CGAN/single_stage_overfit_adv_only_no_huber_20260624_210752`

当前问题主线：

- CGAN 的训练阶段重构图中，红点经常没有贴回橙色训练点。
- 当前主线已转到 endpoint + `[ref, y_b_norm]` 条件；tau 已不再作为该条件模式的核心输入。
- 当前损失逻辑里，`B_ref_only_adv_huber` 和 D 系列实验包含 decision-space Huber；`A_ref_only_adv` 与纯 adversarial 过拟合诊断不含 Huber。
- 已观察到训练量增加能改善部分结果，但 adversarial BCE 对“红点回到橙点”的训练重构目标存在明显干扰；Huber 有帮助但仍不能保证目标空间视觉贴合。

包内包含 PNG/CSV/MAT 等实验结果。为了避免递归和噪音，已排除 `.DS_Store` 与 Data 下已有旧 zip。
