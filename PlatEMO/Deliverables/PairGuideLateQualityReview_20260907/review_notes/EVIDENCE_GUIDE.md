# 事实、口径与证据定位

## 最新目标与已有结论的边界

最新用户要求：生成点不必可行，要帮助 P1 更快发现和逼近边界，尤其未探索区域。可以讨论删除可行性条件 s；不得把已有解坐标输入模型，不增加辅助网络，不大改搜索流程，优先一条统一主线。历史契约中固定 s 的文字只描述当前实现。

首用结果支持“当前选定配置能提供早期有用候选”，并未单独证明某个训练机制成功学习边界。1000 步配置相对同 FE 的普通 DE 为 11 胜 1 平，20 个点平均约 3.92 个进入 P1。这些案例参与过选择，尚无独立种子确认。

完整运行的最终 IGD 对 DE 为 5 胜 7 负，对真实配对采样为 3 胜 9 负。不能把“持续收益不稳定”写成“所有后续候选都完全无效”。真实配对对照更强说明它能利用已有存档信息，但不能据此证明 CGAN 必须恢复双端点输出，或将存档资格修正认定为已经独立验证的性能因果。

末期 rawUseful 约 0～0.7%，定义只是“可行且不被当前可行 P1 支配”，它遗漏不可行候选通过反馈与后续搜索产生的价值。标签准确率约 50% 说明当前 s 控制差；如果删去 s，该准确率不再是对应模型的评估目标。要判断有用性，必须核对实际消费、环境选择、反馈及后续搜索。

## 当前实现的因果链

入口：`code/Algorithms/Multi-objective optimization/PairGuide/PairGuide.m`。

主干：CBS-CGAN/Core/PairGuideCore.m。P1 约束搜索，P2 无约束搜索；环境选择分别使用 P1+O1+O2、P2+O1+O2。P1 的 GA/DE/引导份额为 25/55/20%，P2 为 25/75%。候选代末进入 Pending，下一代原样真实评价；合法去重导致缺额时由 DE 回填。真实消费后可进入种群或参与存档更新。

存档与条件：PairBoundaryArchive_RC.m。完整候选可行池统一分层，仅全局第一层具备端点资格；先排除被可行候选支配的不可行端点，再配对，允许一个不可行端复用。历史同筛，上限 500、允许不足或为空。每方向最多一条激活配对；激活端点独立训练，按自身真实目标关联参考方向，按 [x,s] 去重并平衡侧别/方向。

训练：PairBoundaryWGAN_RC.m。G([z,w,s]) 输出一个有界决策点，critic D([x,w,s]) 为标量。G 纯条件对抗损失，D 的 Wasserstein 差与决策梯度惩罚；当前没有端点重建、配对差向量损失或逆 gap 加权。数据中的 delta 仍用于存档/重训变化判定，不是恢复了重建损失。

当前统一参数：32×32；G/D 学习率 1e-3；Adam=(0,0.9)；GP=10；D:G=5:1；batch 上限 32；首训 1000 次 G 更新、续训 20 次。这里的更新次数不等同于遍历全数据集的 epoch。至少 8 对且内容变化达到 0.2 或间隔达到 10 代才续训。权重和 Adam 保留，已有模型至少一对即可服务。

使用：完整参考 W 上两侧各请求 250 行；sigma=0 时通常只有 200 个不同输出。原生合法性、去重、请求方向轮转后选至多 20 个，没有 critic 评分、真实目标或真实标签筛选，不再绑定已有配对。原生点无配对 ID，但实际评价后仍可按真实目标邻域反馈收紧存档。

## 优先核查的表格与图

均位于 `experiments/PairGuideSinglePoint_20260907/`：

| 内容 | 文件 | 能回答什么 |
|---|---|---|
| 首用选择 | first_use_all.csv、selected_budget_summary.csv、first_use_joint_scores.csv | 训练预算、候选直接留存、首用 IGD 与条件质量的取舍 |
| 正式对照 | full_run/interval_summary.csv、full_run/analysis/paired_comparisons.csv | 36 次搜索、实际引导消费与反馈、总成本、同种子胜负 |
| 搜索进程 | full_run/analysis/checkpoints_per_run.csv、各 `.mat.checkpoints.csv` | 相同 FE 下的种群质量与阶段进展 |
| 条件质量 | conditional_quality.csv、conditional_quality_mean.csv | 已见/未见条件、训练/生成尺度、方向误差、标签及 rawUseful |
| 固定数据 | holdout/selected1000_summary.csv、continued4000_summary.csv | 保持数据与模型状态续训、真实方向留出的变化 |
| 正确性 | full_verification.csv、first_use_verification.json、implementation/final_regression.log | FE、候选原样消费、初次查询复现、热启动与存档资格 |

图像均与实际事件对应。每题 run 1 有首次真实使用和 10000/30000/50000/70000/100000 FE 六阶段，各一张全景与 focus。左侧是所用模型实际训练时的数据，右侧是对应原生生成点，金色是实际消费点；不得把不同时间的种群和模型误当同步。LIRCMOP6 的末期有缺失生成事件，空白/缺测不等于生成了零值点。

两个有代表性的末期情况：LIRCMOP5 run 1 已见条件占比约 6.2%，未见方向偏移明显；LIRCMOP8 run 1 仍有约 96 条训练配对、方向误差约 5.1°，但目标位置仍偏移。因而“样本变少”不能解释全部问题；角度接近也不能证明学到有用边界位置。

固定数据续训有改善方向的案例，但未证明未知边界的搜索收益。当前未唯一分离损失、网络容量和优化方式；不要把“20 步一定不够”“噪声为零一定导致平均化”或“遗忘已被证实”作为既定结论。

## 可直接计算的数值证据

- `*_training_events.json`：12 次 CGAN 所有已记录训练尝试；`trained/reason/trigger/contentChange/updates/trainingPairs/preDiagnostics/postDiagnostics` 等。诊断值不是额外在线真实标签。
- `*_generation_counters.json`：逐代标量计数，包含请求/消费/回填、模型版本、档案活跃数、guided 与 ordinary 两类真实收紧次数。只导出标量及字符串，未保留每个后代的完整逐代血缘与布尔留存数组；延迟收益不能凭空推断。
- `*_population_counters.json`：每个观测的 FE、存档总数/激活数、可行 P1 数、参考归一化尺度。
- `*_plot_state.json`：四题六阶段的完整缓存绘图状态；`searchState` 是观测时刻，`trainingState` 是实际训练时刻，`generationState` 是生成时刻；三者的真实种群和存档坐标可供离线分析，不是被允许作为模型条件。
- `*_query.json`：23 个有效绘图节点的查询条件、原始决策、抽样参数、选择下标和 Pending。与对应 `*_raw_objectives.csv` 按行对应；原生 500 行保留重复，以便检查实际查询分布。
- `*_first_model.json` / `*_last_model.json`：run 1 四题的 G/D 数值参数、Adam 数值状态和模型元数据；网络结构见当前代码。`shape` 保留参数维度，值为从 MATLAB single/dlarray 导出的 double 表示。
- `*_held*.json`：8 个固定数据方向留出/续训模型，包括 D/Full、查询、真实 X/Y/C、已缓存角度、留出区间和 G/D 参数。它们不是正式搜索中的模型。

`lossless_compaction.json` 记录引用展开后的规范 JSON 哈希；`shared/` 保存重复的大数组/状态。`export_status.json` 明确本次仅导出数据，没有新增训练或 oracle 调用。原始全量 MAT 留在原工作区，不在包内。

## 来源与复现限制

当前完整搜索有冻结源码清单；搜索结束后仅修正两项离线汇总/绘图文件，`full_source` 和 `offline_source` 分别留存。`code/` 是当前工作树副本，包含离线修正；不能把它的整体哈希写成旧运行时哈希。

早期 44 个首用记录没有记录精确运行源码哈希，已有来源说明；其余 77 个保留真实哈希。不能事后声称所有早期臂都具备精确逐文件溯源。

旧诊断中的端点拟合、配对插值、逆 gap 权重、one-hot 变体只适用于旧配对网络。相关旧代码位于修正存档实验的 source_snapshot 和本轮 implementation/before。旧报告中的源码行号随版本变化，仅作定位提示，以对应冻结源码正文为准。
