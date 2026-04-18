# PRBCCMO Loop Optimization Design

日期：2026-04-15

## 1. 任务目标

本任务不是重新实现 `fix.md` 的 plain-BCE 主线，而是：

1. 先验证当前 `PRBCCMO` 是否已经符合 `fix.md` 与 `算法思想和创新点.md`
2. 在该基线上补齐可重复运行的观测指标、套件脚本与汇总分析
3. 在 `DASCMOP_BC` 与 `LIRCMOP_BC` 上运行算法并分析结果
4. 结合运行结果、算法代码与联网检索提出修改意见
5. 将修改意见收缩到 `fix.md` 允许的三类改动，再实现并继续循环

循环终止条件固定为：

`0.2 <= IGD_ratio <= 5`

其中：

`IGD_ratio = 当前版本 IGD / 图片参考 IGD`

本任务中的“参考 IGD”只允许来自用户给出的两张图片中的 DRMCMO 与 NA-EMT 结果，不使用：

1. `Algorithms/Multi-objective optimization/NA-EMT-2025/NAEMT2025.m`
2. `Algorithms/Multi-objective optimization/NA-EMT-2025/benchmark_NAEMT2025_paper.m`

因为这两份文件是用户明确标注的失败复现版本。

## 2. 已确认事实

经代码调研与最小验证，当前仓库中的 `PRBCCMO` 主线已经满足 `fix.md` 的核心约束：

1. 使用 `B + RecentBoundaryOff` 作为训练核心
2. 采用 `trusted sector`
3. 采用 `bridge-gated` 的边界存档准入
4. 训练档案按扇区来源配额裁剪
5. `helper` 只取真实 opposite side
6. MLP 路径已收缩为 plain BCE
7. 不再使用 `BoundWeight`、校准集与温度缩放

并且以下验证已通过：

1. `test_PRBCCMO_semantics`
2. `test_PRBCCMO_t_smoke`
3. `test_PRBCCMO_t_metrics`

因此后续循环的起点是“合规基线上的性能与边界语义优化”，而不是“再次做 fix.md 清理”。

## 3. 核心设计原则

后续每一轮迭代都必须坚持以下原则：

1. 核心创新点始终是“MLP 学习与 PF 搜索相关的真实边界，并反过来驱动搜索”
2. 双种群、边界档案、扇区划分都只是服务于边界学习，不得喧宾夺主
3. 修改范围只允许落在：
   - 边界定义
   - `B` 的入选规则
   - 训练数据分布
4. 不新增：
   - 新损失函数
   - 新模型家族
   - calibration / temperature scaling
   - predicted-opposite helper
   - 与主线无关的复杂阶段切换
5. 每一轮修改都必须同时更新：
   - 主算法
   - traced 算法
   - 结果汇总脚本
   - 必要测试

## 4. 需要新增或修改的模块

### 4.1 套件执行模块

需要新增一套可重复运行的 benchmark suite：

1. 支持 `DASCMOP1_BC` 到 `DASCMOP9_BC`
2. 支持 `LIRCMOP1_BC` 到 `LIRCMOP14_BC`
3. 支持多次重复运行
4. 支持最多 6 个 MATLAB 进程并行
5. 输出每个问题每次运行的：
   - IGD
   - HV
   - Feasible_rate
   - runtime
   - traced 观测目录

### 4.2 参考值模块

需要把图片中的参考结果显式录入为机器可读表，而不是手工目测比较。

表中至少保存：

1. 问题名
2. 参考算法名
3. 参考均值 IGD
4. 参考标准差
5. 图片来源标记

其中：

1. `DASCMOP_BC` 参考来自图片中的 `NA-EMT`
2. `LIRCMOP_BC` 参考来自图片中的 `DRMCMO`

### 4.3 结果汇总与比较模块

需要新增或扩展脚本，对完整 suite 的结果做三层汇总：

1. run 级
2. problem 级
3. family 级

并计算：

1. `IGD_ratio`
2. 是否满足 `0.2 <= IGD_ratio <= 5`
3. 各家族满足比例
4. 最差任务
5. 与 traced 诊断的关联指标

### 4.4 失败模式分析模块

每轮分析必须同时看两类证据：

1. 性能证据
   - IGD
   - HV
   - Feasible_rate
2. 边界语义证据
   - `generation_summary.csv`
   - `boundary_event.csv`
   - `archive_members.csv`
   - `mlp_events.csv`

目标不是追求训练准确率，而是回答：

1. low-margin 是否更靠近真实边界
2. `B` 是否覆盖了更多 PF 相关边界扇区
3. MLP 是否在 trusted sector 中真正有用
4. helper / probe 是否在正确的局部边界上工作

## 5. 单轮循环流程

每轮固定执行以下步骤：

1. 以当前代码运行一轮完整或缩小版 suite
2. 汇总 IGD 与 traced 观测结果
3. 定位当前最主要失败模式
4. 用 `grok-search` 只针对该失败模式检索一手或高可信资料
5. 形成修改意见
6. 用 `fix.md` 与 `算法思想和创新点.md` 过滤修改意见
7. 实现通过过滤的意见
8. 跑相关测试与小规模回归
9. 再跑 suite 验证

## 6. 修改意见的过滤规则

只有同时满足以下条件的意见才允许进入实现：

1. 能解释当前运行结果中的明确失败模式
2. 不偏离“MLP 学边界并驱动边界采样”的主线
3. 最终落实到以下三类之一：
   - trusted-sector / bridge 的边界定义
   - `B` 的扇区内入选与保留规则
   - 训练档案的组成、补侧与配额规则

以下意见即使文献支持，也默认过滤掉：

1. 调 loss
2. 换网络结构
3. 加校准
4. 加 predicted helper
5. 加与主线无关的辅助档案或阶段机制

## 7. 停止条件

允许跳出循环必须满足：

1. 完整 suite 中，大多数问题的 `IGD_ratio` 落在 `[0.2, 5]`
2. 不存在某一整个家族系统性超出该范围
3. traced 诊断没有显示明显的边界语义崩坏

若未满足：

1. 继续下一轮
2. 但每轮必须保留证据链，不能无依据试参

## 8. 交付物

本任务最终需要维护和交付的内容包括：

1. `PRBCCMO.m`
2. `PRBCCMO_t.m`
3. suite runner 脚本
4. 参考值表
5. IGD 汇总与比较脚本
6. 必要测试脚本
7. 每轮运行后的比较结果表

## 9. 自检

已检查本设计：

1. 没有把主线重新扩展到 loss、模型家族或校准
2. 没有把用户明确排除的 `NAEMT2025.m` 纳入流程
3. 停止条件已从“同数量级”具体化为 `0.2 <= IGD_ratio <= 5`
4. 设计覆盖了验证、跑数、分析、检索、过滤和再实现的完整闭环
