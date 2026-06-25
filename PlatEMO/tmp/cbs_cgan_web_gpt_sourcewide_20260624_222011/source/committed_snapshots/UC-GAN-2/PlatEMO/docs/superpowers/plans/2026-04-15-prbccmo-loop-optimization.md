# PRBCCMO Loop Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repeatable optimization loop for `PRBCCMO` on `DASCMOP_BC` and `LIRCMOP_BC`, run the suite, analyze IGD and traced boundary diagnostics, and iterate only within the `fix.md`-allowed boundary-definition/archive/training-distribution space until the medium stop criterion is met.

**Architecture:** Treat the current `PRBCCMO` as the verified `fix.md` baseline. Add benchmark orchestration, reference-result tables, and machine-readable comparison scripts around it. Use `PRBCCMO_t` as the diagnostic baseline, and only modify `PRBCCMO` / `PRBCCMO_t` after the suite reveals a concrete failure mode that can be mapped back to trusted-sector semantics, boundary archive admission, or training archive composition.

**Tech Stack:** MATLAB, PlatEMO, `matlab -batch`, CSV diagnostics, shell parallelism, Grok MCP web search

**Repo Note:** This repository forbids unsolicited commits. During execution here, treat every commit step as a checkpoint to use only if the user explicitly asks for commits.

---

## File Structure

- Create: `Algorithms/Multi-objective optimization/PRBCCMO/benchmark_PRBCCMO_loop_suite.m`
  Responsibility: run one logical benchmark job, compute metrics, and write per-run result rows.
- Create: `Algorithms/Multi-objective optimization/PRBCCMO/run_PRBCCMO_loop_suite.sh`
  Responsibility: dispatch up to 6 MATLAB jobs across `DASCMOP_BC` and `LIRCMOP_BC`.
- Create: `Algorithms/Multi-objective optimization/PRBCCMO/load_PRBCCMO_reference_igd.m`
  Responsibility: expose image-derived reference IGD values for `DRMCMO` and `NA-EMT`.
- Create: `Algorithms/Multi-objective optimization/PRBCCMO/compare_PRBCCMO_suite_to_reference.m`
  Responsibility: compute `IGD_ratio`, pass/fail flags, family-level summaries, and stop-condition verdicts.
- Modify: `Algorithms/Multi-objective optimization/PRBCCMO/summarize_PRBCCMO_t_data.m`
  Responsibility: preserve traced summaries needed by the loop analysis.
- Modify: `Algorithms/Multi-objective optimization/PRBCCMO/analyze_PRBCCMO_t_suite.m`
  Responsibility: surface family/problem diagnostics used to explain failures.
- Modify: `Algorithms/Multi-objective optimization/PRBCCMO/PRBCCMO.m`
  Responsibility: apply loop-approved semantic changes.
- Modify: `Algorithms/Multi-objective optimization/PRBCCMO/PRBCCMO_t.m`
  Responsibility: mirror loop-approved semantic changes and diagnostics.
- Create: `Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_loop_suite.m`
  Responsibility: regression test for reference loading, benchmark output schema, and ratio comparison.

### Task 1: Lock the Verified Baseline and Add Loop-Level Regression Coverage

**Files:**
- Modify: `Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_semantics.m`
- Create: `Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_loop_suite.m`
- Test: `Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_semantics.m`
- Test: `Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_loop_suite.m`

- [ ] **Step 1: Re-run the baseline semantic guard**

Run:

```bash
matlab -batch "cd('/Users/lanai/Code/Matlab/PlatEMO/PlatEMO'); addpath(genpath(pwd)); cd('Algorithms/Multi-objective optimization/PRBCCMO'); test_PRBCCMO_semantics"
```

Expected: PASS with exit code `0`.

- [ ] **Step 2: Add a loop-suite regression test skeleton**

Create `test_PRBCCMO_loop_suite.m` with assertions for the planned helper functions and output schema:

```matlab
function test_PRBCCMO_loop_suite()
    Ref = load_PRBCCMO_reference_igd();
    assert(istable(Ref), 'test_PRBCCMO_loop_suite:ReferenceType', ...
        'Reference IGD loader must return a table.');
    assert(all(ismember({'problem','family','reference_algorithm','reference_igd_mean'}, Ref.Properties.VariableNames)), ...
        'test_PRBCCMO_loop_suite:ReferenceColumns', ...
        'Reference IGD table is missing required columns.');
    assert(~any(strcmp(string(Ref.reference_algorithm), "NAEMT2025")), ...
        'test_PRBCCMO_loop_suite:ForbiddenReference', ...
        'Loop reference table must not use the failed NAEMT2025 reproduction.');

    EmptyRuns = table('Size',[0 5], ...
        'VariableTypes',{'string','string','double','double','double'}, ...
        'VariableNames',{'problem','family','run','igd','feasible_rate'});
    [ProblemSummary,FamilySummary,StopSummary] = compare_PRBCCMO_suite_to_reference(EmptyRuns, Ref);
    assert(istable(ProblemSummary) && istable(FamilySummary) && istable(StopSummary), ...
        'test_PRBCCMO_loop_suite:CompareTypes', ...
        'Comparison helper must return tables.');
    assert(all(ismember({'igd_ratio','ratio_pass'}, ProblemSummary.Properties.VariableNames)), ...
        'test_PRBCCMO_loop_suite:ProblemSummaryColumns', ...
        'Problem summary must expose ratio fields.');
end
```

- [ ] **Step 3: Run the new loop-suite regression test and verify it fails before implementation**

Run:

```bash
matlab -batch "cd('/Users/lanai/Code/Matlab/PlatEMO/PlatEMO'); addpath(genpath(pwd)); cd('Algorithms/Multi-objective optimization/PRBCCMO'); test_PRBCCMO_loop_suite"
```

Expected: FAIL with missing-function errors for `load_PRBCCMO_reference_igd` or `compare_PRBCCMO_suite_to_reference`.

### Task 2: Implement the Reference IGD Table and Comparison Logic

**Files:**
- Create: `Algorithms/Multi-objective optimization/PRBCCMO/load_PRBCCMO_reference_igd.m`
- Create: `Algorithms/Multi-objective optimization/PRBCCMO/compare_PRBCCMO_suite_to_reference.m`
- Test: `Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_loop_suite.m`

- [ ] **Step 1: Implement the reference-value loader**

Create `load_PRBCCMO_reference_igd.m` as a static table whose rows come only from the user-provided images:

```matlab
function Ref = load_PRBCCMO_reference_igd()
    rows = {
        "DASCMOP1_BC","DASCMOP_BC","NA-EMT",3.3720e-3,2.68e-4,"image_table_1";
        "DASCMOP2_BC","DASCMOP_BC","NA-EMT",4.5321e-3,1.17e-4,"image_table_1";
        "DASCMOP3_BC","DASCMOP_BC","NA-EMT",2.0186e-2,9.08e-4,"image_table_1";
        "DASCMOP4_BC","DASCMOP_BC","NA-EMT",3.2022e-3,2.36e-3,"image_table_1";
        "DASCMOP5_BC","DASCMOP_BC","NA-EMT",4.4232e-3,1.09e-3,"image_table_1";
        "DASCMOP6_BC","DASCMOP_BC","NA-EMT",2.0045e-2,8.17e-4,"image_table_1";
        "DASCMOP7_BC","DASCMOP_BC","NA-EMT",4.4370e-2,5.20e-3,"image_table_1";
        "DASCMOP8_BC","DASCMOP_BC","NA-EMT",5.0510e-2,5.30e-3,"image_table_1";
        "DASCMOP9_BC","DASCMOP_BC","NA-EMT",4.1815e-2,1.27e-3,"image_table_1";
        "LIRCMOP1_BC","LIRCMOP_BC","DRMCMO",1.6979e-1,7.62e-2,"image_table_2";
        "LIRCMOP2_BC","LIRCMOP_BC","DRMCMO",1.0770e-1,4.37e-2,"image_table_2";
        "LIRCMOP3_BC","LIRCMOP_BC","DRMCMO",2.0470e-1,7.64e-2,"image_table_2";
        "LIRCMOP4_BC","LIRCMOP_BC","DRMCMO",1.5727e-1,2.78e-2,"image_table_2";
        "LIRCMOP5_BC","LIRCMOP_BC","DRMCMO",8.1548e-3,5.30e-4,"image_table_2";
        "LIRCMOP6_BC","LIRCMOP_BC","DRMCMO",7.5513e-3,6.57e-4,"image_table_2";
        "LIRCMOP7_BC","LIRCMOP_BC","DRMCMO",8.3228e-3,7.10e-4,"image_table_2";
        "LIRCMOP8_BC","LIRCMOP_BC","DRMCMO",8.4891e-3,9.45e-4,"image_table_2";
        "LIRCMOP9_BC","LIRCMOP_BC","DRMCMO",5.6304e-2,3.38e-2,"image_table_2";
        "LIRCMOP10_BC","LIRCMOP_BC","DRMCMO",7.0248e-3,8.71e-4,"image_table_2";
        "LIRCMOP11_BC","LIRCMOP_BC","DRMCMO",2.6527e-3,1.51e-4,"image_table_2";
        "LIRCMOP12_BC","LIRCMOP_BC","DRMCMO",3.2715e-3,3.39e-4,"image_table_2";
        "LIRCMOP13_BC","LIRCMOP_BC","DRMCMO",1.1655e-1,2.54e-3,"image_table_2";
        "LIRCMOP14_BC","LIRCMOP_BC","DRMCMO",1.0043e-1,1.93e-3,"image_table_2"
    };
    Ref = cell2table(rows, 'VariableNames', ...
        {'problem','family','reference_algorithm','reference_igd_mean','reference_igd_std','reference_source'});
end
```

- [ ] **Step 2: Implement the suite-to-reference comparator**

Create `compare_PRBCCMO_suite_to_reference.m`:

```matlab
function [ProblemSummary,FamilySummary,StopSummary] = compare_PRBCCMO_suite_to_reference(Runs, Ref)
    if isempty(Runs)
        ProblemSummary = table('Size',[0 8], ...
            'VariableTypes',{'string','string','double','double','double','double','logical','string'}, ...
            'VariableNames',{'problem','family','igd_mean','igd_std','reference_igd_mean','igd_ratio','ratio_pass','reference_algorithm'});
        FamilySummary = table('Size',[0 5], ...
            'VariableTypes',{'string','double','double','double','logical'}, ...
            'VariableNames',{'family','problem_count','pass_count','pass_ratio','family_pass'});
        StopSummary = table("overall",0,0,false,'VariableNames',{'scope','problem_count','pass_count','stop_ready'});
        return;
    end

    G = groupsummary(Runs, {'problem','family'}, {'mean','std'}, 'igd');
    ProblemSummary = outerjoin(G, Ref, 'Keys', 'problem', 'MergeKeys', true);
    ProblemSummary.igd_mean = ProblemSummary.mean_igd;
    ProblemSummary.igd_std = ProblemSummary.std_igd;
    ProblemSummary.igd_ratio = ProblemSummary.igd_mean ./ ProblemSummary.reference_igd_mean;
    ProblemSummary.ratio_pass = ProblemSummary.igd_ratio >= 0.2 & ProblemSummary.igd_ratio <= 5;
    ProblemSummary = ProblemSummary(:, {'problem','family_Runs','igd_mean','igd_std','reference_igd_mean','igd_ratio','ratio_pass','reference_algorithm'});
    ProblemSummary.Properties.VariableNames{'family_Runs'} = 'family';

    Families = unique(ProblemSummary.family, 'stable');
    Rows = cell(numel(Families), 5);
    for i = 1 : numel(Families)
        mask = ProblemSummary.family == Families(i);
        passRatio = mean(double(ProblemSummary.ratio_pass(mask)));
        Rows(i,:) = {Families(i), sum(mask), sum(ProblemSummary.ratio_pass(mask)), passRatio, passRatio >= 0.8};
    end
    FamilySummary = cell2table(Rows, 'VariableNames', ...
        {'family','problem_count','pass_count','pass_ratio','family_pass'});

    stopReady = ~isempty(FamilySummary) && all(FamilySummary.family_pass);
    StopSummary = table("overall", height(ProblemSummary), sum(ProblemSummary.ratio_pass), stopReady, ...
        'VariableNames', {'scope','problem_count','pass_count','stop_ready'});
end
```

- [ ] **Step 3: Run the loop-suite regression test and verify it passes**

Run:

```bash
matlab -batch "cd('/Users/lanai/Code/Matlab/PlatEMO/PlatEMO'); addpath(genpath(pwd)); cd('Algorithms/Multi-objective optimization/PRBCCMO'); test_PRBCCMO_loop_suite"
```

Expected: PASS with exit code `0`.

### Task 3: Implement the Benchmark Runner and 6-Job Suite Dispatcher

**Files:**
- Create: `Algorithms/Multi-objective optimization/PRBCCMO/benchmark_PRBCCMO_loop_suite.m`
- Create: `Algorithms/Multi-objective optimization/PRBCCMO/run_PRBCCMO_loop_suite.sh`
- Test: `Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_loop_suite.m`

- [ ] **Step 1: Implement the MATLAB benchmark runner**

Create `benchmark_PRBCCMO_loop_suite.m` to run one family slice:

```matlab
function Results = benchmark_PRBCCMO_loop_suite(problemNames, runs, outCsv, useTrace)
    if nargin < 4
        useTrace = true;
    end
    rows = cell(0, 8);
    row = 0;
    for p = 1 : numel(problemNames)
        problemName = problemNames{p};
        for r = 1 : runs
            rng(r, 'twister');
            if useTrace
                Algorithm = PRBCCMO_t('save', 0, 'run', r);
            else
                Algorithm = PRBCCMO('save', 0, 'run', r);
            end
            Problem = feval(problemName, 'N', 100, 'maxFE', 200000);
            Algorithm.Solve(Problem);
            row = row + 1;
            rows(row,:) = { ...
                string(problemName), ...
                familyOf(problemName), ...
                r, ...
                Algorithm.CalMetric('IGD'), ...
                Algorithm.CalMetric('HV'), ...
                Algorithm.CalMetric('Feasible_rate'), ...
                Algorithm.CalMetric('runtime'), ...
                string(getfield(Algorithm.metric, 'analysis_folder', ""))}; %#ok<GFLD>
        end
    end
    Results = cell2table(rows, 'VariableNames', ...
        {'problem','family','run','igd','hv','feasible_rate','runtime','analysis_folder'});
    Results.igd = cellfun(@(x) x(end), Results.igd);
    Results.hv = cellfun(@(x) x(end), Results.hv);
    Results.feasible_rate = cellfun(@(x) x(end), Results.feasible_rate);
    Results.runtime = cellfun(@(x) x(end), Results.runtime);
    if nargin >= 3 && ~isempty(outCsv)
        writetable(Results, outCsv);
    end
end

function family = familyOf(problemName)
    if startsWith(problemName, 'DASCMOP')
        family = "DASCMOP_BC";
    else
        family = "LIRCMOP_BC";
    end
end
```

- [ ] **Step 2: Implement the shell dispatcher with up to 6 jobs**

Create `run_PRBCCMO_loop_suite.sh`:

```bash
#!/bin/zsh
set -euo pipefail

ROOT="/Users/lanai/Code/Matlab/PlatEMO/PlatEMO"
OUT_DIR="$ROOT/Data/PRBCCMO_loop_suite"
mkdir -p "$OUT_DIR"

jobs=(
  "DASCMOP1_BC,DASCMOP2_BC,DASCMOP3_BC"
  "DASCMOP4_BC,DASCMOP5_BC,DASCMOP6_BC"
  "DASCMOP7_BC,DASCMOP8_BC,DASCMOP9_BC"
  "LIRCMOP1_BC,LIRCMOP2_BC,LIRCMOP3_BC,LIRCMOP4_BC,LIRCMOP5_BC"
  "LIRCMOP6_BC,LIRCMOP7_BC,LIRCMOP8_BC,LIRCMOP9_BC,LIRCMOP10_BC"
  "LIRCMOP11_BC,LIRCMOP12_BC,LIRCMOP13_BC,LIRCMOP14_BC"
)

for idx in {1..6}; do
  slice="${jobs[$idx]}"
  csv="$OUT_DIR/slice_${idx}.csv"
  matlab -batch "cd('$ROOT'); addpath(genpath(pwd)); cd('Algorithms/Multi-objective optimization/PRBCCMO'); problems = strsplit('$slice', ','); benchmark_PRBCCMO_loop_suite(problems, 3, '$csv', true);" &
done

wait
```

- [ ] **Step 3: Dry-run one small slice and verify the output schema**

Run:

```bash
matlab -batch "cd('/Users/lanai/Code/Matlab/PlatEMO/PlatEMO'); addpath(genpath(pwd)); cd('Algorithms/Multi-objective optimization/PRBCCMO'); benchmark_PRBCCMO_loop_suite({'DASCMOP1_BC'}, 1, fullfile(tempdir,'prbccmo_loop_smoke.csv'), true)"
```

Expected: PASS with a CSV containing `problem,family,run,igd,hv,feasible_rate,runtime,analysis_folder`.

### Task 4: Extend the Traced Summaries So They Can Explain Failures

**Files:**
- Modify: `Algorithms/Multi-objective optimization/PRBCCMO/summarize_PRBCCMO_t_data.m`
- Modify: `Algorithms/Multi-objective optimization/PRBCCMO/analyze_PRBCCMO_t_suite.m`
- Test: `Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_t_metrics.m`

- [ ] **Step 1: Keep the traced summarizer stable**

Run:

```bash
matlab -batch "cd('/Users/lanai/Code/Matlab/PlatEMO/PlatEMO'); addpath(genpath(pwd)); cd('Algorithms/Multi-objective optimization/PRBCCMO'); test_PRBCCMO_t_metrics"
```

Expected: PASS before further changes.

- [ ] **Step 2: Add loop-facing derived columns only if they come from existing traced CSVs**

If needed, extend `summarize_PRBCCMO_t_data.m` and `analyze_PRBCCMO_t_suite.m` to expose:

```matlab
'final_b_trusted_sector_coverage'
'final_b_trusted_lowmargin_count'
'final_train_minus_b_bal_gap'
'mean_helper_real_opp_ratio'
'mean_helper_skip_no_helper'
'mean_boundary_attempts'
```

Do not add synthetic columns that are not supported by existing traced CSV data.

- [ ] **Step 3: Re-run the traced regression**

Run:

```bash
matlab -batch "cd('/Users/lanai/Code/Matlab/PlatEMO/PlatEMO'); addpath(genpath(pwd)); cd('Algorithms/Multi-objective optimization/PRBCCMO'); test_PRBCCMO_t_metrics"
```

Expected: PASS with exit code `0`.

### Task 5: Run the Baseline Suite and Produce the First Gap Report

**Files:**
- Create/Modify: `Algorithms/Multi-objective optimization/PRBCCMO/run_PRBCCMO_loop_suite.sh`
- Create/Modify: `Algorithms/Multi-objective optimization/PRBCCMO/compare_PRBCCMO_suite_to_reference.m`

- [ ] **Step 1: Launch the 6-job baseline suite**

Run:

```bash
zsh "/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Algorithms/Multi-objective optimization/PRBCCMO/run_PRBCCMO_loop_suite.sh"
```

Expected: Six MATLAB processes complete and write slice CSV files under `Data/PRBCCMO_loop_suite`.

- [ ] **Step 2: Merge the slice CSVs and compute the first comparison report**

Run:

```bash
matlab -batch "cd('/Users/lanai/Code/Matlab/PlatEMO/PlatEMO'); addpath(genpath(pwd)); cd('Algorithms/Multi-objective optimization/PRBCCMO'); files = dir(fullfile(pwd,'..','..','..','Data','PRBCCMO_loop_suite','slice_*.csv')); Runs = cellfun(@(f) readtable(fullfile(files(1).folder,f)), {files.name}, 'UniformOutput', false); Runs = vertcat(Runs{:}); Ref = load_PRBCCMO_reference_igd(); [ProblemSummary,FamilySummary,StopSummary] = compare_PRBCCMO_suite_to_reference(Runs, Ref); writetable(ProblemSummary, fullfile(files(1).folder,'problem_summary.csv')); writetable(FamilySummary, fullfile(files(1).folder,'family_summary.csv')); writetable(StopSummary, fullfile(files(1).folder,'stop_summary.csv'));"
```

Expected: `problem_summary.csv`, `family_summary.csv`, and `stop_summary.csv` exist.

- [ ] **Step 3: Identify the first concrete failure mode**

Inspect:

```bash
matlab -batch "cd('/Users/lanai/Code/Matlab/PlatEMO/PlatEMO'); addpath(genpath(pwd)); cd('Algorithms/Multi-objective optimization/PRBCCMO'); analyze_PRBCCMO_t_suite(fullfile('/Users/lanai/Code/Matlab/PlatEMO/PlatEMO','Data','PRBCCMO_t'));"
```

Expected: family/problem ranked CSVs that explain whether failures correlate with poor trusted-sector coverage, weak archive balance, or helper/probe degeneracy.

### Task 6: Apply Evidence-Based Semantic Iterations

**Files:**
- Modify: `Algorithms/Multi-objective optimization/PRBCCMO/PRBCCMO.m`
- Modify: `Algorithms/Multi-objective optimization/PRBCCMO/PRBCCMO_t.m`
- Modify: `Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_semantics.m`
- Modify: `Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_t_metrics.m`

- [ ] **Step 1: Convert the first failure mode into one allowed modification**

Allowed categories only:

```text
trusted-sector / bridge semantics
boundary archive admission / retention
training archive composition / quota
```

If the proposed fix falls outside those categories, reject it and continue analysis instead of editing.

- [ ] **Step 2: Implement the minimal semantic change in `PRBCCMO.m` and mirror it in `PRBCCMO_t.m`**

Example shape:

```matlab
% If analysis shows trusted sectors are too sparse, adjust only the
% bridge qualification or local opposite-support rule, not the loss/model.
BridgeMask(i) = HasRealLocalOppositeSupport(Meta,i) && ...
                any(Local == 1-Meta.feasible(i));
```

or

```matlab
% If analysis shows the train archive over-represents one side, adjust only
% source quotas or missing-side anchor admission.
Quota = [2 2 1 1];
```

The exact code depends on the measured failure mode. Do not introduce new loss terms or model classes.

- [ ] **Step 3: Re-run the focused verification set**

Run:

```bash
matlab -batch "cd('/Users/lanai/Code/Matlab/PlatEMO/PlatEMO'); addpath(genpath(pwd)); cd('Algorithms/Multi-objective optimization/PRBCCMO'); test_PRBCCMO_semantics; test_PRBCCMO_t_smoke; test_PRBCCMO_t_metrics"
```

Expected: PASS with exit code `0`.

- [ ] **Step 4: Re-run a small diagnostic subset before the full suite**

Run:

```bash
matlab -batch "cd('/Users/lanai/Code/Matlab/PlatEMO/PlatEMO'); addpath(genpath(pwd)); cd('Algorithms/Multi-objective optimization/PRBCCMO'); benchmark_PRBCCMO_loop_suite({'DASCMOP1_BC','DASCMOP7_BC','LIRCMOP5_BC','LIRCMOP11_BC'}, 2, fullfile('/tmp','prbccmo_loop_subset.csv'), true)"
```

Expected: A subset CSV showing whether the targeted fix moved the relevant failure mode in the right direction.

### Task 7: Repeat Until the Medium Stop Criterion Is Met

**Files:**
- Reuse: all loop and traced-analysis files above

- [ ] **Step 1: Re-run the full suite after each accepted semantic change**

Run:

```bash
zsh "/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Algorithms/Multi-objective optimization/PRBCCMO/run_PRBCCMO_loop_suite.sh"
```

- [ ] **Step 2: Recompute the stop summary**

Run:

```bash
matlab -batch "cd('/Users/lanai/Code/Matlab/PlatEMO/PlatEMO'); addpath(genpath(pwd)); cd('Algorithms/Multi-objective optimization/PRBCCMO'); files = dir(fullfile('/Users/lanai/Code/Matlab/PlatEMO/PlatEMO','Data','PRBCCMO_loop_suite','slice_*.csv')); Runs = cellfun(@(f) readtable(fullfile(files(1).folder,f)), {files.name}, 'UniformOutput', false); Runs = vertcat(Runs{:}); Ref = load_PRBCCMO_reference_igd(); [~,~,StopSummary] = compare_PRBCCMO_suite_to_reference(Runs, Ref); disp(StopSummary);"
```

Expected: `StopSummary.stop_ready == true` only when the medium criterion is satisfied.

- [ ] **Step 3: Inspect the final diff**

Run:

```bash
git diff -- "Algorithms/Multi-objective optimization/PRBCCMO/PRBCCMO.m" "Algorithms/Multi-objective optimization/PRBCCMO/PRBCCMO_t.m" "Algorithms/Multi-objective optimization/PRBCCMO/benchmark_PRBCCMO_loop_suite.m" "Algorithms/Multi-objective optimization/PRBCCMO/run_PRBCCMO_loop_suite.sh" "Algorithms/Multi-objective optimization/PRBCCMO/load_PRBCCMO_reference_igd.m" "Algorithms/Multi-objective optimization/PRBCCMO/compare_PRBCCMO_suite_to_reference.m" "Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_loop_suite.m"
```

Expected: only loop-related files are changed, with no edits to `NAEMT2025.m` or its benchmark script.
