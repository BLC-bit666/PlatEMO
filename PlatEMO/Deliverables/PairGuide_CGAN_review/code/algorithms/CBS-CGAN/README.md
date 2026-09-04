# CBS-CGAN：生产主线与 PairGuide

当前只保留两个算法入口：

- `CBS_RegionWGAN_GP`：生产主线；
- `CBS_RegionWGAN_GP_PairGuide`：原子边界 pair、绝对端点 CGAN 与差分引导对比算法。

两者复用 `Core/CBS_RegionWGAN_GP_Core.m` 的双种群循环。A0/A1/A2、E0–E8、DE20、Random20、GA20、FullCGAN、Screening、MechanismAudit 等历史实验入口及 campaign 已删除。

## PairGuide 边界档案

每条记录为不可拆分的真实端点对：

```text
id, xf, yf, xi, yi, ref, gap, age, lastFE, active
```

- `xf` 为真实可行端点，`xi` 为真实不可行端点；当前 P1 的 `Fitness<1` 精英可创建 pair，但 retained `xf` 不要求继续属于 P1。
- 新 pair 使用当前 evaluated union 中的普通真实不可行解，按归一化决策距离做全局最近配对，不使用 reference 准入门控。
- 带 `matchedPairID` 的引导子代只更新该 pair；普通真实评价解按最近同侧端点唯一归属。
- 可行解仅在 `d_norm(zF,xi)` 严格缩小时更新 `xf/yf/ref`；不可行解仅在 `d_norm(xf,zI)` 严格缩小时更新 `xi/yi`。
- `ref` 只由 `yf` 决定，用于 CGAN 条件、查询/使用调度和容量分桶；不限制端点更新或新 pair 建立。
- 新建、严格收紧或当前 P1 精英的局部决策空间支持会激活 pair。无证据则 inactive；连续第 10 代仍无证据时删除。inactive pair 可被真实解收紧并复活，但不训练、不查询。
- 每参考方向最多 5 对，总容量为 `5*size(W,1)`。裁剪依次优先 active、小 gap、小 age、最近更新和较小 ID。
- 原始生成点 `G` 不进行 oracle 评价，不进入 Union，不更新档案。

## PairGuide CGAN

训练样本为绝对归一化决策端点：

```text
[xf_norm, w, s=1]
[xi_norm, w, s=0]
```

固定配置：隐藏层 `[32,32]`、`zDim=6`、学习率 `1e-4`、`GP=10`、`nCritic=5`、mini-batch 为 32 个完整 pair（64 个 endpoint）。

PairGuide 第三个公开参数为 `ganEpoch`，默认 500，用于首次训练；后续每次重训固定为 10 Epoch。每次训练事件执行完整 epoch：每条训练 pair 每 epoch 恰好访问一次，随机打乱、无放回、端点不拆分、尾批保留；每个 batch 执行 5 次 Critic 和 1 次 Generator 更新。生产主线的 `ganIter=100` 不变。

首次训练要求至少 32 条 active pair、4 个 active reference。训练使用全部 active pair，不划分训练集和验证集。之后 active 数据新增、收紧或移除 pair 合计至少 8 条，或新增 reference 至少 2 个时继续训练；训练结果立即采用并使用，不做 checkpoint 接受/拒绝判断。

## 生成与使用

1. 固定查询 `s=0`，按 active reference 均衡生成 500 个不可行侧候选 `G`。
2. Critic 不筛选、不排序候选。候选在同一/相邻 reference 中匹配最近 active `xi`，得到 `matchedPairID`。
3. 从当前 P1 精英选择局部支持 `xf` 的父代 `xc`，计算 `Fg=min(0.5,hc/(d+eps))` 与 `T=clip(xc+Fg*(G-xc))`。
4. 合法候选映射到 T-space 后，经参考方向空缺加权 maximin 选择本代 P1 子代名额的 20%。整数舍入余数由普通 DE 吸收，不会挤占该 20% 配额。
5. 仅评价 `OperatorDE(Problem,xc,T,xc,{1,1,1,20})` 的子代；缺失名额由普通 DE 补齐。

PairGuide 没有固定 50% FE 截止。是否生成和使用完全由 archive、训练触发条件与局部支持门控决定。

## 运行与验证

根目录 `test.m` 配置五个完整预算 LIRCMOP 问题、`runs=1:5`、10 个 process workers、`N=100`、`D=30`、`maxFE=200000`，结果写入 `Data/CBS_RegionWGAN_GP_PairGuide_v4`。

首次训练/首次使用 Epoch 实验使用 `run_CBS_PairGuide_epoch_first_use_campaign`；当前协议测试 Epoch 400/600/800/1000/1200、每组 2 runs、10 个 process workers，并在第一次实际使用后立即停止。每张图同时显示可行域、不可行域、当代约束种群 P1、当代无约束种群 P2、500 个原始 CGAN 候选和已真实评估的引导子代。结果写入 `Data/CBS_PairGuide_first_use_epoch_v4_20260901`，不覆盖旧实验。

最小回归：

```matlab
test_CBS_pair_guide
test_CBS_platemo_compliance
test_CBS_region_wgan_mainline
test_CBS_mainline_fingerprint
```
