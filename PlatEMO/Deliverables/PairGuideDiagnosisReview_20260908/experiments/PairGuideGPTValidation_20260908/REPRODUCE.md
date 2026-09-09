**GPT 建议核验：实验与复现说明**

本目录比较模型专用方向尺度、可行性条件 s，以及候选使用方式。原算法的 P1/P2、严格存档资格、配对、激活、每代评价配额和损失均保持原有设置。新增参数是实验开关，当前生产默认仍为动态尺度并保留 s。

主线固定参数：LIRCMOP5/6/7/8_BC，N=100，默认 D=30，每次 100000 FE；G/D 隐藏层均为 [32,32]，学习率均为 0.001，nCritic=5，GP=10，batch=32，训练和生成噪声均为 0。首训为 1000 次 G 更新，后续每次为 20 次，持续保留权重和 Adam；字段名 Epoch 是历史命名，实际是更新步数。

实验分为四部分：

- 完整搜索初筛：种子 1–3，新增 stable_side 与 stable_no_side 各 12 次，复用此前原版、DE、pair-only 共 36 次。主要指标是 50000–100000 FE 的 IGD 梯形积分除以 50000，采用每 1000 FE 的 51 个已评价状态；跨题目比较采用配对比值的几何平均。不同批次并行负载不同，记录的耗时不用于算法运行速度优劣判断。
- 独立复核：初筛按 experiment_plan.json 选出 stable_no_side，参数锁定于 confirmation_lock.json。使用未参与选择的种子 11–15，对候选、原版、DE、pair-only 进行 80 次同 FE 搜索。不要把这批种子用于继续调参。
- 固定数据诊断：7/8 题、种子 1–3，使用原版晚期真实训练集及已保存的模型与 Adam。比较固定条件与仅重放已保存 span 变化，真实点、批次 ID 和事件 RNG 完全相同，追加 2000 次 G 更新。另有 6 个同结构新初始化模型，使用完全相同的固定数据、批次和预算；这只是离线历史状态对照，没有改为在线重置。
- 使用方式诊断：相同 50000/70000 FE 前缀、同一批 q，比较原样使用、中点 p+0.5(q-p)、DE、pair-only。每个分支只干预第一批，其后统一采用原算法 fallback 骨干运行 5000 FE。12 个原样分支第一步必须逐点重现原实验 P1、P2 和存档。中点运算发生在使用阶段，没有把父代坐标作为 G 的输入。

边界见证指标来自已真实评价的异侧短配对，使用单位决策空间的 RMS 距离。相对各运行 50000 FE 的存档，计算新的分离中点及 P1 的邻域覆盖；短窗口则相对共同前缀。分别报告 0.01/0.02/0.04 阈值。这是相对存档快照的代理量，不是“全局从未探索过的边界”的证明，也不应替代搜索性能。

现有结果文件不可覆盖。脚本会跳过已完成的结果；若需要真正复跑，应先为复现实验指定新的输出目录，同时保留原结果作为核对依据。完整搜索调用现有 run_PairGuide_validation；每次运行结束都有 FE=100000、CalObjRows=200000、CalConRows=100000 的断言。BC 约束实现内部会再计算一次目标，因此目标函数行数是完整评价行数的两倍。

在项目根目录，MATLAB 入口为：

```matlab
folder = fullfile(pwd,'Data','PairGuideGPTValidation_20260908');
addpath(folder);
run_search_arm(true,8);
addpath(folder); % run_search_arm 会恢复 MATLAB 默认路径。
run_search_arm(false,8);
addpath(folder);
run_confirmation(8);
```

固定数据和短窗口输入由 prepare_fixtures.m、prepare_suffix_fitness.m 从原始已完成实验导出。batch_schedules/ 保存本次实际使用且逐对核验相同的批次，不依赖 worker 的默认随机数生成器。每个训练事件仍显式使用固定的 twister RNG。固定集合的原始生成脚本保存在 before/run_fixed_conditions_initial.m。

汇总和核验入口：collect_search_metrics、audit_search_arm、audit_auxiliary、audit_fresh_conditions、collect_confirmation_evidence、summarize_tables.py、confirmation_comparisons.py。所有汇总均读取已评价结果，不调用真实目标或约束。collect_search_metrics('confirmation') 与 summarize_tables.py confirmation 汇总独立复核。plot_validation、plot_late_clouds 绘制科学图表；各完整搜索只有种子 1 保存生成分布图。

本次原始代码备份在 before/，实验源码冻结在 source_snapshot/，改动在 condition_experiment.patch，哈希在各 manifest JSON。两个作废短窗口批次分别保存在 suffix_pre_fitness_fix/ 与 suffix_pre_seed_fix/，合计消耗 422400 FE，均不用于结论；正式的 48 个短窗口在 suffix/。其他诊断差错说明见 fresh_schedule_note.md。

图中 RMSE 和判别器评分差仅是诊断量，RMSE 不是当前 WGAN 的损失，也不是候选价值的主判据。新方案的完整搜索前半程已经不同，所以后半程曲线变好可能包含较早获得的优势，不能单凭它证明后期跟踪机制已修复。

本轮全部实验与核验已完成，最终结论见 REPORT.md。统一报告的末尾汇总入口为 confirmation_secondary_summary.py 与 write_final_report.py；前者合并 40 个 CGAN 使用记录和 80 次搜索的边界代理记录，后者在完整结果与核验均齐全时生成报告。控制组旧 state.mat 保留当时被中止的冗余汇总状态；control_report_orchestration.json 记录统一报告已完成替代，原始实验数据未改写。
