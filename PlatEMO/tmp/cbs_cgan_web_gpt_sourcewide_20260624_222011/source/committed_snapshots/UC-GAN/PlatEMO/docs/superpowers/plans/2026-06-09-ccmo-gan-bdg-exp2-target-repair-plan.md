# CCMO-GAN-BDG Experiment-2 Target Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不进入 experiment 3-8 的前提下，继续在 experiment-2 D/G target 层寻找能让 raw GAN 生成可行边界解的候选。

**Architecture:** 保留当前 CCMO-GAN-BDG 主线、archive、condition semantics、condition sampling、fullscope runner 和 gate 统计格式。新增一个统一的 boundary-quality target，把 objective-space AF-AI segment proximity、可行性和标准目标范围惩罚合成单一标量监督信号；先验证 evaluated-sample D target，再验证 generated-sample soft label 增强。

**Tech Stack:** MATLAB / PlatEMO / Deep Learning Toolbox `dlnetwork` custom loop / existing CCMO-GAN-BDG support runners.

---

## Evidence Baseline

Current files and results used for this plan:

- `fix2.md`
- `Data/CCMO_GAN_BDG/fix_validation/experiment_0_2_report.md`
- `Data/CCMO_GAN_BDG/fix_validation/exp1_all10_failure_mode_summary.csv`
- `Data/CCMO_GAN_BDG/fix_validation/exp1_exp2_exp2h_all10_gate_counts.csv`
- `Algorithms/Multi-objective optimization/CCMO-GAN-BDG/CCMO_GAN_BDG.m`
- `Algorithms/Multi-objective optimization/CCMO-GAN-BDG/BoundaryGAN_BDG.m`
- `Algorithms/Multi-objective optimization/CCMO-GAN-BDG/Support/run_CCMO_GAN_BDG_archive_pareto_filters_fullscope.m`
- `Algorithms/Multi-objective optimization/CCMO-GAN-BDG/Support/test_CCMO_GAN_BDG_archive_pareto_filters.m`

Current confirmed gate facts:

- Experiment 1 clean baseline: 10 problems x runs 1:3, `30/30 ok`.
- Experiment 2 pair-boundary D target: 10 problems x runs 1:3, `30/30 ok`, `gate pass = 0/10`.
- Experiment 2h critic-regression G loss: 10 problems x runs 1:3, `30/30 ok`, `gate pass = 0/10`.
- Exp2 and exp2h both improve `SegmentDist90` on only `2/10` problems and improve `RawGANOutsideStandardRangeCount` on `0/10` problems.
- Experiments 3-5 are blocked because the experiment-2 D/G target gate failed.
- Experiment 6 is still gated even though tiny archive collapse is observed.
- Experiment 7 has raw/injected observability only; injected-strategy analysis has not legally started.
- Experiment 8 is blocked because no best repair chain exists.

Important correction to `fix2.md`:

- `fix2.md` recommends an "objective-space segment critic target".
- Current source already does objective-space segment labeling for evaluated critic samples in `BuildPairBoundaryCriticData_BDG`: labels are `exp(-dist/tau)` where `dist` is distance to normalized AF-AI objective-space segment.
- Therefore the next experiment must not simply duplicate current `pair_boundary` / exp2.
- The next target must be more specific: a unified feasible-boundary quality target that includes objective segment proximity, feasibility, and outside-range severity as one scalar target.

Hard constraints:

- Do not proceed to experiment 3-8 until a fullscope D/G target candidate passes the target-layer gate.
- Do not use DASCMOP4_BC / DASCMOP5_BC run=1 as final evidence.
- Do not use small-FE smoke as final evidence.
- Do not use `RawGANFeasibleRate` or `InjectedCount` alone as success criteria.
- Keep the innovation point: boundary-solution-trained GAN / related network generates feasible boundary solutions.
- Prefer unification, reduction, and convergence over branch proliferation.

## File Structure

Create:

- `Algorithms/Multi-objective optimization/CCMO-GAN-BDG/BoundaryQualityTarget_BDG.m`
  - One shared target builder for evaluated and generated samples.
  - Prevents duplicating target formulas across `CCMO_GAN_BDG.m` and `BoundaryGAN_BDG.m`.

Modify:

- `Algorithms/Multi-objective optimization/CCMO-GAN-BDG/CCMO_GAN_BDG.m`
  - Add `ganCriticMode` names.
  - Use `BoundaryQualityTarget_BDG` when building critic labels.
  - Pass objective and constraint evaluators to generated-label data.

- `Algorithms/Multi-objective optimization/CCMO-GAN-BDG/BoundaryGAN_BDG.m`
  - Accept the new critic modes.
  - Use the shared target builder for generated labels.
  - Preserve current `adversarial`, `critic_regression`, and `decision_segment` modes.

- `Algorithms/Multi-objective optimization/CCMO-GAN-BDG/Support/run_CCMO_GAN_BDG_archive_pareto_filters_fullscope.m`
  - Accept new `ganCriticMode` values in variant validation.

- `Algorithms/Multi-objective optimization/CCMO-GAN-BDG/Support/test_CCMO_GAN_BDG_archive_pareto_filters.m`
  - Add smoke tests for the new target family and generated-label diagnostics.

Do not modify in this phase:

- `generatorMode`
- condition sampling
- archive pair direction
- injection strategy
- experiment 3-8 scripts

## Target Definition

The new scalar target is:

```text
boundaryQuality = segmentScore * feasibleScore * outsideScore
```

Definitions:

- `segmentScore = exp(-objectiveSegmentDistance / tauSegment)`
- `objectiveSegmentDistance`: normalized objective-space distance from sample objective to nearest AF-AI segment.
- `feasibleScore = exp(-positiveConstraintViolation / tauCV)`
- `positiveConstraintViolation = max(max(cons, [], 2), 0)`
- `outsideScore = exp(-outsideSeverity / tauOutside)`
- `outsideSeverity = 0` when objective is inside standard objective bounds; otherwise normalized distance outside those bounds.

Rules:

- Invalid or non-finite objective/constraint values get target `0`.
- Do not hard-zero every outside generated sample unless it is non-finite. Use graded outside severity so generated labels do not collapse to all zero.
- `tauSegment` starts from current `nearTau`.
- `tauCV` is median positive constraint violation from the critic batch; fallback `1` when no positive violation exists.
- `tauOutside` is `0.05` in normalized objective units for the first candidate; do not sweep it in the first pass.

## Candidate Chain

### Candidate 2i: Boundary-Quality Evaluated Critic

Purpose:

- Replace evaluated-sample critic label from segment-only target to unified feasible-boundary quality target.

Variant:

```matlab
variants = table("A1_pair_cd_2i_boundary_quality", ...
    "global_af_nd", "ai_dominates_af", "pair_cd_conditioned", ...
    "pair_boundary_quality", "critic_regression", 0, ...
    'VariableNames', {'variant','archiveParetoFilterMode', ...
    'archivePairDirectionMode','generatorMode','ganCriticMode', ...
    'generatorLossMode','decisionSegmentWeight'});
```

Single intended change versus exp2h:

- D target changes from segment-only pair-boundary label to boundary-quality label.
- Keep `generatorLossMode="critic_regression"`.
- Keep `decisionSegmentWeight=0`.
- Keep archive, condition, sampling, `N`, `maxFE`, `runs`, and all runner parameters unchanged.

### Candidate 2j: Boundary-Quality Generated Soft Labels

Purpose:

- Test whether adding generated samples to D updates helps only after generated labels no longer collapse to all zero.

Variant:

```matlab
variants = table("A1_pair_cd_2j_boundary_quality_generated", ...
    "global_af_nd", "ai_dominates_af", "pair_cd_conditioned", ...
    "pair_boundary_quality_generated", "critic_regression", 0, ...
    'VariableNames', {'variant','archiveParetoFilterMode', ...
    'archivePairDirectionMode','generatorMode','ganCriticMode', ...
    'generatorLossMode','decisionSegmentWeight'});
```

Single intended change versus candidate 2i:

- Same boundary-quality target.
- Adds generated-sample soft labels to D updates.
- Generated labels use graded outside severity instead of hard all-zero outside labels.

### Deferred Candidate 2k: Generator Loss Recheck

Do not run this before 2i or 2j produces a nonzero fullscope signal.

Purpose:

- Test whether `adversarial` or `critic_regression` is better once D output semantics are boundary quality.

Allowed comparison only after candidate 2i or 2j passes pilot:

- Same `ganCriticMode`.
- Compare only `generatorLossMode="critic_regression"` against `generatorLossMode="adversarial"`.
- Do not add `decision_segment`.
- Do not add condition changes.

## Task 1: Implement Shared Boundary-Quality Target

**Files:**

- Create: `Algorithms/Multi-objective optimization/CCMO-GAN-BDG/BoundaryQualityTarget_BDG.m`
- Modify: none in this task.

- [ ] **Step 1: Create the target dispatcher**

Create a MATLAB function with these actions:

```matlab
function varargout = BoundaryQualityTarget_BDG(action,varargin)
% BoundaryQualityTarget_BDG - Unified scalar target for feasible boundary GAN.
    switch lower(string(action))
        case "builddata"
            varargout{1} = buildData(varargin{:});
        case "evaluatedlabels"
            [varargout{1:nargout}] = evaluatedLabels(varargin{:});
        case "generatedlabels"
            [varargout{1:nargout}] = generatedLabels(varargin{:});
        otherwise
            error('BoundaryQualityTarget_BDG:UnknownAction', ...
                'Unknown action "%s".',action);
    end
end
```

- [ ] **Step 2: Implement `buildData`**

Required fields:

```matlab
Data.AFObj
Data.AIObj
Data.zmin
Data.zmax
Data.tauSegment
Data.tauOutside
Data.objectiveStandardLower
Data.objectiveStandardUpper
Data.hasObjectiveStandardBounds
Data.objectiveFcn
Data.constraintFcn
Data.lower
Data.upper
```

Use existing logic from `PairObjectiveGap_BDG` and `ObjectiveStandardBounds_BDG` as the model, but keep this helper self-contained because local functions in `CCMO_GAN_BDG.m` are not externally visible.

- [ ] **Step 3: Implement `evaluatedLabels`**

Input:

```matlab
[Labels,pairIdx,Diag] = BoundaryQualityTarget_BDG( ...
    'evaluatedLabels',Objs,Cons,Data)
```

Output:

- `Labels`: column `single` vector in `[0,1]`.
- `pairIdx`: nearest AF-AI objective segment index.
- `Diag`: counts and distribution fields.

Diagnostic fields:

```matlab
Diag.boundary_quality_count
Diag.boundary_quality_mean
Diag.boundary_quality_p10
Diag.boundary_quality_p50
Diag.boundary_quality_p90
Diag.boundary_quality_zero_rate
Diag.boundary_quality_high_rate
Diag.boundary_quality_outside_rate
Diag.boundary_quality_positive_cv_rate
```

- [ ] **Step 4: Implement `generatedLabels`**

Input:

```matlab
[Labels,outside,Obj,Cons,Diag] = BoundaryQualityTarget_BDG( ...
    'generatedLabels',XScaled,pairIdx,Data)
```

Rules:

- Convert scaled decisions from `[-1,1]` to original decision bounds.
- Evaluate `Problem.CalObj(Problem.CalDec(X))`.
- Evaluate `Problem.CalCon(Problem.CalDec(X))`.
- Reuse the same boundary-quality formula as evaluated samples.
- Do not hard-zero outside samples unless objective is invalid or non-finite.

- [ ] **Step 5: Run syntax check**

Run:

```bash
matlab -batch "addpath(genpath(pwd)); which BoundaryQualityTarget_BDG"
```

Expected:

- MATLAB prints the path to `BoundaryQualityTarget_BDG.m`.

## Task 2: Wire Candidate 2i and 2j Modes

**Files:**

- Modify: `Algorithms/Multi-objective optimization/CCMO-GAN-BDG/CCMO_GAN_BDG.m`
- Modify: `Algorithms/Multi-objective optimization/CCMO-GAN-BDG/BoundaryGAN_BDG.m`
- Modify: `Algorithms/Multi-objective optimization/CCMO-GAN-BDG/Support/run_CCMO_GAN_BDG_archive_pareto_filters_fullscope.m`

- [ ] **Step 1: Add critic mode names**

Add these legal values everywhere `ganCriticMode` / `criticMode` is normalized:

```matlab
"pair_boundary_quality"
"pair_boundary_quality_generated"
```

- [ ] **Step 2: Add helper predicates**

In both main files, make pair-boundary checks include the new modes:

```matlab
IsPairBoundaryCriticMode_BDG("pair_boundary_quality") == true
IsPairBoundaryCriticMode_BDG("pair_boundary_quality_generated") == true
UsesGeneratedLabelCritic_BDG("pair_boundary_quality_generated") == true
UsesGeneratedLabelCritic_BDG("pair_boundary_quality") == false
```

- [ ] **Step 3: Wire evaluated labels in `CCMO_GAN_BDG.m`**

When `ganCriticMode == "pair_boundary_quality"` or `"pair_boundary_quality_generated"`:

- Build target data from `AFTrain`, `AITrain`, `Problem`, and `nearTau`.
- Call `BoundaryQualityTarget_BDG('evaluatedLabels', Samples.objs, Samples.cons, Data)`.
- Assign `CriticLabels` from that result.
- Preserve `CriticConditionData` logic from current pair-boundary mode.

- [ ] **Step 4: Wire generated labels in `BoundaryGAN_BDG.m`**

When mode is `"pair_boundary_quality_generated"`:

- Use `BoundaryQualityTarget_BDG('generatedLabels', XFake, pairIdx, LabelData)`.
- Store diagnostics in existing `critic_generated_*` columns where possible.
- Add boundary-quality-specific diagnostic columns only if existing columns cannot express the distribution.

- [ ] **Step 5: Update fullscope variant validation**

In `run_CCMO_GAN_BDG_archive_pareto_filters_fullscope.m`, accept:

```matlab
"pair_boundary_quality"
"pair_boundary_quality_generated"
```

Do not add new condition modes or sampling modes.

## Task 3: Regression Tests

**Files:**

- Modify: `Algorithms/Multi-objective optimization/CCMO-GAN-BDG/Support/test_CCMO_GAN_BDG_archive_pareto_filters.m`

- [ ] **Step 1: Add smoke for candidate 2i**

Add a test that runs:

```matlab
[Core,~,~,Meta] = runArchiveFilterSmoke("global_af_nd",718, ...
    "pair_boundary_quality","pair_cd_conditioned","critic_regression",0);
```

Expected assertions:

- `Meta.ganCriticMode` contains `"pair_boundary_quality"`.
- `Meta.generatorLossMode` contains `"critic_regression"`.
- `Meta.decisionSegmentWeight` contains `0`.
- critic label diagnostics are finite for at least one row.

- [ ] **Step 2: Add smoke for candidate 2j**

Add a test that runs:

```matlab
[Core,~,~,Meta] = runArchiveFilterSmoke("global_af_nd",719, ...
    "pair_boundary_quality_generated","pair_cd_conditioned", ...
    "critic_regression",0);
```

Expected assertions:

- `Meta.ganCriticMode` contains `"pair_boundary_quality_generated"`.
- generated critic label diagnostics have `critic_generated_label_count > 0`.
- generated label mean and score mean are finite.

- [ ] **Step 3: Preserve branch-retirement constraints**

Update `assertOnlyRetainedBranches` so the two new target modes are allowed, but do not allow retired condition/sampling tokens.

- [ ] **Step 4: Run tests**

Run:

```bash
matlab -batch "addpath(genpath(pwd)); test_CCMO_GAN_BDG_archive_pareto_filters"
```

Expected:

- No assertion failures.
- No PNG files are required.

## Task 4: Candidate Smoke Runs

**Purpose:** Validate wiring only. No repair conclusion is allowed from this task.

**Output directory:**

```text
Data/CCMO_GAN_BDG/fix_validation/exp2i_2j_boundary_quality_smoke_20260609
```

- [ ] **Step 1: Run 2i/2j on one problem with full runner settings but small task count**

Run:

```bash
matlab -batch "addpath(genpath(pwd)); outDir=fullfile(pwd,'Data','CCMO_GAN_BDG','fix_validation','exp2i_2j_boundary_quality_smoke_20260609'); problems={'DASCMOP1_BC'}; runs=1; variants=table([\"A1_pair_cd_2i_boundary_quality\";\"A1_pair_cd_2j_boundary_quality_generated\"], repmat(\"global_af_nd\",2,1), repmat(\"ai_dominates_af\",2,1), repmat(\"pair_cd_conditioned\",2,1), [\"pair_boundary_quality\";\"pair_boundary_quality_generated\"], repmat(\"critic_regression\",2,1), zeros(2,1), 'VariableNames',{'variant','archiveParetoFilterMode','archivePairDirectionMode','generatorMode','ganCriticMode','generatorLossMode','decisionSegmentWeight'}); params=struct('ganIter',256,'ganDPretrainIter',400,'ganDSteps',2,'ganGSteps',2,'stageProbeN',200,'probeRawN',0); plotOptions=struct('enable',false,'targets',[],'plotRuns',[],'plotVariants',[],'ganN',0); run_CCMO_GAN_BDG_archive_pareto_filters_fullscope(outDir,1,problems,100,[],100000,runs,variants,params,plotOptions);"
```

Expected:

- `archive_pareto_filter_run_summary.csv` has 2 ok rows.
- No PNG files are generated.
- `critic_generated_label_count > 0` for candidate 2j.
- Candidate 2j generated label diagnostics are not all invalid.

## Task 5: All-10 Run-1 Pilot

**Purpose:** Screen candidate 2i/2j across all requested problems without claiming final repair.

**Output directory:**

```text
Data/CCMO_GAN_BDG/fix_validation/exp2i_2j_boundary_quality_all10_run1_20260609
```

- [ ] **Step 1: Run all 10 problems, run 1**

Run:

```bash
matlab -batch "addpath(genpath(pwd)); outDir=fullfile(pwd,'Data','CCMO_GAN_BDG','fix_validation','exp2i_2j_boundary_quality_all10_run1_20260609'); problems={'DASCMOP1_BC','DASCMOP2_BC','DASCMOP4_BC','DASCMOP5_BC','LIRCMOP5_BC','LIRCMOP6_BC','LIRCMOP7_BC','LIRCMOP8_BC','LIRCMOP9_BC','LIRCMOP10_BC'}; runs=1; variants=table([\"A1_pair_cd_2i_boundary_quality\";\"A1_pair_cd_2j_boundary_quality_generated\"], repmat(\"global_af_nd\",2,1), repmat(\"ai_dominates_af\",2,1), repmat(\"pair_cd_conditioned\",2,1), [\"pair_boundary_quality\";\"pair_boundary_quality_generated\"], repmat(\"critic_regression\",2,1), zeros(2,1), 'VariableNames',{'variant','archiveParetoFilterMode','archivePairDirectionMode','generatorMode','ganCriticMode','generatorLossMode','decisionSegmentWeight'}); params=struct('ganIter',256,'ganDPretrainIter',400,'ganDSteps',2,'ganGSteps',2,'stageProbeN',200,'probeRawN',0); plotOptions=struct('enable',false,'targets',[],'plotRuns',[],'plotVariants',[],'ganN',0); run_CCMO_GAN_BDG_archive_pareto_filters_fullscope(outDir,8,problems,100,[],100000,runs,variants,params,plotOptions);"
```

Expected:

- 20/20 rows have `status=ok`.
- No PNG files are generated.
- All standard stage and run summary CSV files are written.

- [ ] **Step 2: Pilot screen**

Compare each candidate against the clean baseline and exp2/exp2h summaries.

Pilot can advance to fullscope only if all conditions hold:

- `RawGANOutsideStandardRangeCount` improves on at least 3/10 problems.
- `GAN_to_Segment_Dist90` improves on at least 4/10 problems.
- `AF_Cover_epsilon` is non-decreasing on at least 6/10 problems.
- `RawGANVarClipRate` is not increased.
- Candidate does not improve only DASCMOP4_BC / DASCMOP5_BC.
- Candidate does not improve only `RawGANFeasibleRate`.

If neither 2i nor 2j passes this screen, stop D/G target execution and inspect the target diagnostics before designing a new target.

## Task 6: Fullscope Gate Run

**Purpose:** Only run this if Task 5 passes the pilot screen. This is the first stage where a repair conclusion is allowed.

**Output directory:**

```text
Data/CCMO_GAN_BDG/fix_validation/exp2i_or_2j_boundary_quality_all10_runs1_3_20260609
```

- [ ] **Step 1: Run selected candidate on all 10 problems, runs 1:3**

Use only the best candidate from Task 5. Do not run both unless both passed pilot and the extra runtime is explicitly accepted.

Run template for candidate 2j:

```bash
matlab -batch "addpath(genpath(pwd)); outDir=fullfile(pwd,'Data','CCMO_GAN_BDG','fix_validation','exp2j_boundary_quality_generated_all10_runs1_3_20260609'); problems={'DASCMOP1_BC','DASCMOP2_BC','DASCMOP4_BC','DASCMOP5_BC','LIRCMOP5_BC','LIRCMOP6_BC','LIRCMOP7_BC','LIRCMOP8_BC','LIRCMOP9_BC','LIRCMOP10_BC'}; runs=1:3; variants=table(\"A1_pair_cd_2j_boundary_quality_generated\", \"global_af_nd\", \"ai_dominates_af\", \"pair_cd_conditioned\", \"pair_boundary_quality_generated\", \"critic_regression\", 0, 'VariableNames',{'variant','archiveParetoFilterMode','archivePairDirectionMode','generatorMode','ganCriticMode','generatorLossMode','decisionSegmentWeight'}); params=struct('ganIter',256,'ganDPretrainIter',400,'ganDSteps',2,'ganGSteps',2,'stageProbeN',200,'probeRawN',0); plotOptions=struct('enable',false,'targets',[],'plotRuns',[],'plotVariants',[],'ganN',0); run_CCMO_GAN_BDG_archive_pareto_filters_fullscope(outDir,8,problems,100,[],100000,runs,variants,params,plotOptions);"
```

Expected:

- 30/30 rows have `status=ok`.
- No PNG files are generated.

- [ ] **Step 2: Gate analysis**

Generate the same comparison CSV pattern already used for exp2 and exp2h:

- `*_all10_gate_counts.csv`
- `*_all10_gate_summary.csv`
- `*_failure_mode_summary.csv`

Gate acceptance for opening experiment 3 discussion:

- Same per-problem `GatePass` definition as current exp2/exp2h comparison.
- `sum_GatePass >= 5/10`.
- `sum_OutsideCountImproved >= 5/10`.
- `sum_SegmentDist90Improved >= 5/10`.
- `sum_AFCoverNonDecreasing >= 6/10`.
- `sum_VarClipNonIncreasing = 10/10`.

If these are not met, experiment 3 remains closed.

## Task 7: Report Update

**Files:**

- Modify: `Data/CCMO_GAN_BDG/fix_validation/experiment_0_2_report.md`

- [ ] **Step 1: Add an Experiment 2i/2j section**

Include:

- exact variant table
- exact output directory
- exact command
- run status count
- no-PNG statement
- gate counts
- problem-level verdict table
- whether experiment 3 remains closed

- [ ] **Step 2: Keep conclusions evidence-scoped**

Required wording:

- Pilot runs are screening only.
- Fullscope runs are required for repair conclusions.
- If fullscope gate fails, experiment 3-8 remain unopened.
- If fullscope gate passes, only then define the next condition-semantics experiment.

## Stop Conditions

Stop and do not proceed to experiment 3-8 if any of these occur:

- Smoke test fails.
- Pilot has status errors.
- Pilot only improves run=1 DASCMOP4_BC / DASCMOP5_BC.
- Pilot only improves `RawGANFeasibleRate`.
- Fullscope `gate pass` remains `0/10`.
- Fullscope `outside count improved` remains `0/10`.
- Fullscope improves `SegmentDist90` but collapses `AF_Cover`, `PairCover`, or `RefCov`.

## Self-Review

Spec coverage:

- Core innovation preserved: yes, all candidates train GAN-related target to generate feasible boundary solutions.
- Experiment 3-8 gate respected: yes, no downstream experiment is scheduled before fullscope D/G target pass.
- Unification / subtraction: yes, one shared boundary-quality target helper; no condition/sampling branch additions.
- No over-generalization: yes, pilot is all 10 problems run 1 and final is all 10 problems runs 1:3.
- Raw/injected separation: yes, primary gate is raw GAN metrics; injected count is auxiliary only.

Placeholder scan:

- No `TBD`.
- No `TODO`.
- No unbounded "add appropriate handling" steps.

Git note:

- Do not commit or push automatically. Use read-only git status/diff during execution; wait for explicit user instruction before any commit.
