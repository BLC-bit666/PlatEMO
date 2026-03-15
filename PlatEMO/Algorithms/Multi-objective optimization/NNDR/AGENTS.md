# NNDREA-MO-CNN Guide

## 1. 研究主线
本目录只保留一个方法学主线：

> 对超高维连续多目标优化，`theta -> z` 的二级压缩，能否在尽量小的低维搜索空间里保留共享 decoder 的有效搜索能力。

唯一方法学依据：
- [ChatGPT-超高维度压缩方法分析.md](/D:/MatlabCode/PlatEMO_/PlatEMO/Algorithms/Multi-objective optimization/NNDREA-MO-CNN/ChatGPT-超高维度压缩方法分析.md)

当前目录不再承载对照基线、teacher 固定验证、oracle 表示验证、在线 PCA、局部 PCA、数据采集、离线诊断或旧 CNN 路线。

## 2. 当前保留算法
只保留两类最贴近主思想的算法：

- `HDLatentMOO`
  - 当前 `Current-z` 主算法
  - `z -> FiLM -> decoder`
  - 带 sparse stage-2 refinement
- `HDLatentMOO_B3ZOnly`
  - 与 `HDLatentMOO` 共用同一 `Current-z` 表示
  - 不带 refinement
  - 作为“表示本身是否成立”的最小消融

说明：
- `HDLatentMOO` 是当前目录唯一的正式主入口。
- `HDLatentMOO_B3ZOnly` 是唯一保留的同思想消融入口。
- 其余实验矩阵与过程性算法已移除，避免目录继续扩散成验证工场。

## 3. 当前目录结构
根目录只放最小公共入口和主算法类：
- `HDLatentMOO.m`
- `HDLatentMOO_B3ZOnly.m`
- `HDEM_Base.m`
- `HDEM_EnsurePaths.m`

内部 helper 只保留主方法运行所需部分：
- `core/`
  - 共享状态与坐标特征
- `search/`
  - reduced-space 初始化、交叉变异、环境选择、历史维护
- `stage2/`
  - active set 选择与 residual 进入逻辑
- `representations/current_z/`
  - 当前 `z -> FiLM -> decoder` 路线

## 4. 明确边界
允许：
- 围绕 `Current-z` 主表示本身做最小必要修正。
- 在不改变 PlatEMO 入口约定的前提下，维护 refinement 与无 refinement 两个版本。
- 复用 `HDEM_Base`、搜索流程、环境选择与问题定义。

禁止：
- 把本目录重新扩展回 baseline、teacher、oracle、online/local PCA 或诊断矩阵集合。
- 回到 `CNNv1/CNN_v2` 的旧思路。
- 在没有证据前宣称 refinement 已成立。
- 通过增加无关实验脚本掩盖表示层问题。

## 5. 开发纪律
修改前必须确认：
- 是否仍然服务于“验证 `theta -> z` 二级压缩”。
- 是否复用了现有 `Current-z` decode/state 架构。
- 是否破坏 PlatEMO 入口、参数约定或 `Population/Y` 同步关系。

实现时必须遵守：
- 一个 `.m` 文件只放一个类或主函数。
- 优先复用已有 helper，不重复造轮子。
- 关键路径必须有边界检查与失败路径。
- 任何新增注释都应直接说明实验假设或关键机制。

## 6. 最小验证
每次改动后至少完成：
1. 入口可解析。
2. `Population.decs` 无 `NaN/Inf`。
3. reduced vector 与种群长度同步。
4. decode 后变量不越界。
5. `HDLatentMOO` 与 `HDLatentMOO_B3ZOnly` 的参数入口保持兼容。

推荐冒烟：
```matlab
platemo('algorithm',@HDLatentMOO,'problem',@LSMOP1,'N',100,'M',3,'D',1000,'maxFE',10000);
platemo('algorithm',@HDLatentMOO_B3ZOnly,'problem',@LSMOP1,'N',100,'M',3,'D',1000,'maxFE',10000);
```
