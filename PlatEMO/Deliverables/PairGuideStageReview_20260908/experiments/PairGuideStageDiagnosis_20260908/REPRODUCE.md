本轮实验在 macOS、MATLAB R2025b、CPU 上完成；需要 Deep Learning Toolbox、Parallel Computing Toolbox、Statistics and Machine Learning Toolbox。未安装或修改全局环境。新脚本均位于本目录，生产算法没有本轮改动。

在 PlatEMO 根目录启动 MATLAB：

```matlab
addpath('Algorithms/Multi-objective optimization/CBS-CGAN/Support');
addCBSPaths(pwd);
addpath('Data/PairGuideStageDiagnosis_20260908');
```

原始输入是既有实验的真实搜索和重置结果：

- `Data/PairGuideSinglePoint_20260907/full_run/LIRCMOP{5..8}_BC_seed01_cgan.mat`：未修改算法的完整轨迹。
- `Data/PairGuideResetAt99800_20260908/fixtures/original_LIRCMOP{7,8}_BC.mat` 和对应 `runs/original_LIRCMOP{7,8}_BC_reset{0..3}.mat`：其中 -1 表示使用 fixture 内原始模型，实际不读取 reset-1 文件。用于复核 GPT 的 10 个静态状态。
- `Data/PairGuideGPTValidation_20260908/suffix_native_helpers.m`：既有原生 P1/P2 和 pair-only 机制副本。

这些 `.mat` 本地保留用于复现。CSV 和 PNG 可直接审阅；仅有 CSV 不足以重训网络。实验记录使用固定搜索 seed=1，每份训练快照另设 3 个训练随机流；这不等于 3 个独立搜索 run。

完整执行顺序如下。训练及快照脚本会跳过已完成结果；如需从头重跑，应在另行备份本目录结果后使用空结果目录，不要把重读已有结果称为新实验。

```matlab
% 先采样未修改算法的真实前/中/后期状态，另存首用与终期参照。
capture_stage_fixtures(1,4);
audit_stage_probe;
prepare_stage_baselines;
probe_adam_step_size;

% 4问题 × 3阶段 × 3训练流 × 2梯度。
run_gradient_comparison(6,{'early','middle','late'},5:8,1:3,.001,'runs');
audit_gradient_pairs;
analyze_stage_geometry('runs');

% 结果驱动的单变量追加：只减小 G 学习率。
run_gradient_comparison(6,{'early','middle','late'},7:8,1:3,.0001,'low_lr_runs');
audit_gradient_pairs('low_lr_runs',7:8);
analyze_stage_geometry('low_lr_runs');

% 接入原生搜索；每例只替换第一批候选，然后5000FE共同后续搜索。
prepare_stage_suffixes;
run_stage_suffixes(0);  % raw及adversarial20 smoke，结果包含在84例中
run_stage_suffixes(4);
audit_stage_suffixes;
export_capture_training_cost;
```

聚合 CSV，不进行训练或真实目标评估：

```sh
python3 Data/PairGuideStageDiagnosis_20260908/summarize_gradient_results.py
python3 Data/PairGuideStageDiagnosis_20260908/summarize_gradient_results.py low_lr_runs
python3 Data/PairGuideStageDiagnosis_20260908/summarize_followups.py
```

生成图，同样不调用目标函数：

```matlab
plot_gradient_stages(2000);
plot_gradient_stages(20);
plot_stage_followups;
```

`PairBoundaryWGAN_StageProbe.m` 是独立命名的诊断副本，没有覆盖或遮蔽生产版 `PairBoundaryWGAN_RC.m`。`gradient_probe.patch` 保存二者差异。A/B 都计算诊断梯度，因此二者的记录耗时不能直接代表去掉诊断后的生产训练成本；参数更新数和真实 FE 单独核算。

本轮修正的实验脚本问题均未改变算法：历史 query 缺少可选 `conditionScale` 字段，改从已保存的同阶段状态取得仅供采样元数据使用的尺度；44 个失败分支在真实评估前退出，修正后只补跑缺失文件。原始 72 个任务的元数据早于可配置学习率字段，审计兼容其缺省记录，原实验学习率由实验计划及当时脚本固定为 .001。原生 HV 为零时，比值记 NaN 并保留绝对差，不虚构有限比值。

完成聚合、生图与训练成本导出后，执行 `python3 Data/PairGuideStageDiagnosis_20260908/validate_deliverables.py` 检查本轮完整结果并生成指纹。

`final_validation.json` 与 `cost_ledger.json` 是最终完整性及成本记录。日志开头的大量历史失效路径警告来自既有 MATLAB 环境，本轮没有修改全局 path。
