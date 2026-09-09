本实验回答：在用户图中 FE99800 的同一份真实训练集上，把 G、D 和 Adam 全部重新初始化，异常生成形状是否还会出现。

六份数据对应 LIRCMOP7/8_BC、seed=1，以及 original、stable_side、stable_no_side 三个方案。每份数据比较一个原模型续训与三个全新初始化，总计 24 个模型。全新初始化种子为 600001/600002/600003；各组追加 10000 次 G 更新，记录 0/1000/2000/5000/10000 步。G/D 均为两层 [32,32]，lr=0.001，nCritic=5，GP=10，batch=32，训练和生成噪声为 0。

prepare_reset_fixtures 从原始结果提取最后一次训练、模型和查询，要求训练与查询时间均为 99800 FE，并逐点复现原始查询决策向量。所有真实端点、条件、模型方向尺度和两侧采样权重在实验中冻结。

run_reset_comparison 使用生产代码的原生端点批次采样器；不使用上一轮固定数据实验的自定义批次。每个训练事件给同一数据集的四个模型设置相同 Twister 状态，并保存事件结束后的完整随机数状态。audit_reset_comparison 比较这些轨迹，验证重置前后的数据、条件、采样随机性、G/D 更新数量和独立评价账目。

重置使用此前已核验的 fresh_model：重新调用 G/D 构造器，清空四个 Adam 矩估计和两个优化器计数。仅保留网络形状、固定条件尺度和训练触发元数据；ready=true 用于按统一的每事件 20 步执行，不意味着保留旧权重。在线算法没有改成重置网络。

所有新生成点都在独立问题对象上完整评价并计费，不回流 P1、P2、存档或训练。图的坐标范围一致，并标出范围外的生成点。这里研究生成偏差，不以可行率代替价值判断，也不据此声称完整搜索更好。

附加诊断均标为探索性：inspect_frozen_geometry 计算条件支持情况，并完整评价每个已见条件下的加权决策均值；这是诊断参照，没有把端点坐标输入模型，也不是提议使用查表生成。probe_generator_capacity 使用同一 G 和相同初始化做固定代表点的全批次 MSE 拟合，仅检查表示与优化难度；另在原版两题上试验 [128,128] 宽度。它们不是只改变损失的公平对照，不可直接推导出应更换 GAN 损失或扩宽生产网络。decompose_objective_offsets 使用已评价的 X/Y 分解目标偏差，不新增真实评价。

MATLAB 在项目根目录的入口：

```matlab
addpath(fullfile(pwd,'Data','PairGuideResetAt99800_20260908'));
prepare_reset_fixtures;
run_reset_comparison(8);
inspect_frozen_geometry;
probe_generator_capacity;
probe_generator_capacity([128 128]);
audit_reset_comparison;
decompose_objective_offsets;
```

随后运行 summarize_reset.py，再运行 plot_reset_comparison(1000)、plot_reset_comparison(2000)、plot_reset_comparison(5000)、plot_reset_comparison(10000)、plot_reset_metrics。已完成的结果会跳过；真正复跑需指定新的结果目录，避免覆盖现有证据。

解释边界：三个新初始化使用相同的冻结数据和训练采样轨迹，只衡量初始化敏感性；不能把它们当成三个独立搜索 seed。10000 更新是预定预算，不是数学上的收敛证明。目标距离、RMSE、同条件偏差与边界搜索收益是不同指标；最终结论还必须受完整搜索对照约束。

后续还增加了两个原方案固定数据对照，检验 D 的方向错配评分问题：run_reset_comparison(2,true) 只运行 7/8 题、fresh init=600001，使用独立的 PairBoundaryWGAN_MismatchProbe.m。唯一训练干预是将 D 的负样本混合为 50% 生成点和 50% 同侧错配方向的真实点，梯度惩罚也针对同一混合；不增加网络、随机数调用或真实评价。mismatch_critic.patch 给出相对生产训练器的差异。audit_mismatch_and_plot(10000) 核对初始权重与完整随机数轨迹；probe_critic_conditions('mismatch_runs',10000) 检查干预后的评分。该对照是探索性验证，没有修改生产训练器或推广为新算法。
