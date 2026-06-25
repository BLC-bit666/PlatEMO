# PRBCCMO Boundary-Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align `PRBCCMO` and `PRBCCMO_t` with `fix.md` by making boundary learning depend only on trusted-sector gating, bridge-qualified archive admission, sector-quota training archives, and plain BCE.

**Architecture:** Keep the existing dual-population and anchor-helper-probe workflow, but remove calibration and weighting mechanics from the MLP path. Push all boundary semantics into three places only: trusted-sector resolution, bridge-gated boundary archive admission, and sector-wise training archive quotas. Mirror the same semantics in the traced variant so CSV diagnostics, summarizers, and regression tests describe the same algorithm.

**Tech Stack:** MATLAB, PlatEMO, `matlab -batch`, CSV diagnostics

**Repo Note:** This repository forbids unsolicited commits. During execution here, treat every commit step as a checkpoint to use only if the user explicitly asks for commits.

---

## File Structure

- Modify: `Algorithms/Multi-objective optimization/PRBCCMO/PRBCCMO.m`
  Responsibility: main algorithm, boundary archive admission, training archive quotas, plain BCE training.
- Modify: `Algorithms/Multi-objective optimization/PRBCCMO/PRBCCMO_t.m`
  Responsibility: traced algorithm, CSV headers/rows, traced MLP diagnostics, summary-level metrics.
- Modify: `Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_semantics.m`
  Responsibility: text-based guard that the code no longer contains calibration/weighting mechanics and now contains bridge/quota mechanics.
- Modify: `Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_t_metrics.m`
  Responsibility: regression guard for traced CSV schema after the plain-BCE cleanup.
- Modify: `Algorithms/Multi-objective optimization/PRBCCMO/summarize_PRBCCMO_t_data.m`
  Responsibility: aggregate only the post-cleanup CSV columns.

### Task 1: Lock the New Semantics with Failing Regression Tests

**Files:**
- Modify: `Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_semantics.m`
- Test: `Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_semantics.m`

- [ ] **Step 1: Write the failing semantic assertions**

```matlab
function test_PRBCCMO_semantics()
% Regression guard for PRBCCMO boundary-learning semantics.

    rootDir = fileparts(mfilename('fullpath'));
    mainText = fileread(fullfile(rootDir,'PRBCCMO.m'));
    traceText = fileread(fullfile(rootDir,'PRBCCMO_t.m'));

    requireContains(mainText,'[Core,CoreSource] = BuildCoreTrainPopulation(B,RecentBoundaryOff);', ...
        'test_PRBCCMO_semantics:MainCoreTrain', ...
        'PRBCCMO.m must build the training core from B and RecentBoundaryOff.');
    requireContains(traceText,'[Core,CoreSource] = BuildCoreTrainPopulation(B,RecentBoundaryOff);', ...
        'test_PRBCCMO_semantics:TraceCoreTrain', ...
        'PRBCCMO_t.m must build the training core from B and RecentBoundaryOff.');

    requireContains(mainText,'ResolveBridgeBoundaryMask', ...
        'test_PRBCCMO_semantics:MainBridgeGate', ...
        'PRBCCMO.m must gate untrusted sectors through bridge-qualified admission.');
    requireContains(traceText,'ResolveBridgeBoundaryMask', ...
        'test_PRBCCMO_semantics:TraceBridgeGate', ...
        'PRBCCMO_t.m must gate untrusted sectors through bridge-qualified admission.');

    requireContains(mainText,'TrimTrainArchiveBySectorQuota', ...
        'test_PRBCCMO_semantics:MainSectorQuota', ...
        'PRBCCMO.m must trim the train archive by sector-wise quotas.');
    requireContains(traceText,'TrimTrainArchiveBySectorQuota', ...
        'test_PRBCCMO_semantics:TraceSectorQuota', ...
        'PRBCCMO_t.m must trim the train archive by sector-wise quotas.');

    requireNotContains(mainText,'CalibArchive', ...
        'test_PRBCCMO_semantics:MainNoCalibArchive', ...
        'PRBCCMO.m must not keep a calibration archive after the plain-BCE cleanup.');
    requireNotContains(traceText,'CalibArchive', ...
        'test_PRBCCMO_semantics:TraceNoCalibArchive', ...
        'PRBCCMO_t.m must not keep a calibration archive after the plain-BCE cleanup.');

    requireNotContains(mainText,'BoundWeight', ...
        'test_PRBCCMO_semantics:MainNoBoundWeight', ...
        'PRBCCMO.m must not use BoundWeight after the plain-BCE cleanup.');
    requireNotContains(traceText,'BoundWeight', ...
        'test_PRBCCMO_semantics:TraceNoBoundWeight', ...
        'PRBCCMO_t.m must not use BoundWeight after the plain-BCE cleanup.');

    requireNotContains(mainText,'FitTemperatureScaling', ...
        'test_PRBCCMO_semantics:MainNoTemperatureScaling', ...
        'PRBCCMO.m must not apply temperature scaling after the plain-BCE cleanup.');
    requireNotContains(traceText,'FitTemperatureScaling', ...
        'test_PRBCCMO_semantics:TraceNoTemperatureScaling', ...
        'PRBCCMO_t.m must not apply temperature scaling after the plain-BCE cleanup.');
end
```

- [ ] **Step 2: Run the semantic regression test and verify it fails**

Run:

```bash
matlab -batch "cd('/Users/lanai/Code/Matlab/PlatEMO/PlatEMO'); addpath(genpath(pwd)); test_PRBCCMO_semantics"
```

Expected: FAIL with one of the new identifiers such as `test_PRBCCMO_semantics:MainBridgeGate`, `test_PRBCCMO_semantics:MainSectorQuota`, or a `MainNoCalibArchive`/`MainNoBoundWeight` assertion because the implementation still contains the old mechanics.

- [ ] **Step 3: Prepare a commit checkpoint for the red state if the user explicitly requests commits**

```bash
git add "Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_semantics.m"
git commit -m "test(prbccmo): lock plain-bce semantics"
```

### Task 2: Refactor `PRBCCMO.m` to Trusted-Sector + Bridge-Gated Plain BCE

**Files:**
- Modify: `Algorithms/Multi-objective optimization/PRBCCMO/PRBCCMO.m`
- Test: `Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_semantics.m`

- [ ] **Step 1: Implement the main-loop and model-interface cleanup**

Update the top-level initialization and model-update calls so the main algorithm stops maintaining calibration state and trains only from the train archive.

```matlab
MaxTrain = max(6*Problem.N,4*size(W,1));
ProbeBeta = [0.25;0.50;0.75;1.05];

PopulationC = Problem.Initialization();
PopulationU = Problem.Initialization();
B           = PopulationC([]);
Model       = [];
RecentBoundaryOff = PopulationC([]);

SectorNeighbors = BuildSectorNeighbors(W,min(max(3,2*Problem.M),max(size(W,1)-1,0)));
TrainArchive    = InitTrainArchive(Problem.D);
TrainArchive    = UpdateTrainArchive( ...
    TrainArchive,B,PopulationC,PopulationU,RecentBoundaryOff, ...
    W,SectorNeighbors,MaxTrain,0);
Model           = UpdateBoundaryModelIfDue( ...
    Model,TrainArchive,minPos,minNeg,hidden,epoch,lr, ...
    ResolveModelUpdateGap(gapMin,gapMax,Problem.FE/max(Problem.maxFE,1)),0);
```

Later in the loop, keep only:

```matlab
RecentBoundaryOff = BoundaryOff;
TrainArchive = UpdateTrainArchive( ...
    TrainArchive,B,PopulationC,PopulationU,RecentBoundaryOff, ...
    W,SectorNeighbors,MaxTrain,Problem.FE);
Model = UpdateBoundaryModelIfDue( ...
    Model,TrainArchive,minPos,minNeg,hidden,epoch,lr, ...
    ResolveModelUpdateGap(gapMin,gapMax,Problem.FE/max(Problem.maxFE,1)), ...
    Problem.FE);
```

- [ ] **Step 2: Add bridge-gated boundary admission and enriched boundary meta**

Replace the current unconditional per-sector admission with trust-aware gating.

```matlab
function [B,Meta] = UpdateBoundaryArchive( ...
    CandidateB,W,Model,PopulationC,PopulationU,TrainArchive,SectorNeighbors,kappa)
    if isempty(CandidateB)
        B    = CandidateB;
        Meta = InitBoundaryMeta(0);
        return;
    end

    CandidateB = KeepUniquePopulation(CandidateB);
    Meta       = BuildBoundaryMeta( ...
        CandidateB,W,Model,PopulationC,PopulationU,TrainArchive,SectorNeighbors);
    BridgeMask = ResolveBridgeBoundaryMask(Meta,TrainArchive,SectorNeighbors,size(W,1));
    B          = SelectTopKPerSector(CandidateB,Meta,BridgeMask,kappa);
end

function BridgeMask = ResolveBridgeBoundaryMask(Meta,TrainArchive,SectorNeighbors,K)
    BridgeMask = Meta.trusted(:);
    CoreMask = IsCoreTrainSource(TrainArchive.Source);
    CoreLabel = TrainArchive.Label(CoreMask);
    CoreSector = TrainArchive.Sector(CoreMask);
    for i = 1 : numel(Meta.sector)
        if BridgeMask(i) || Meta.sector(i) <= 0
            continue;
        end
        Allowed = ResolveLocalSectorSet(Meta.sector(i),SectorNeighbors);
        Local = CoreLabel(ismember(CoreSector,Allowed));
        if isempty(Local)
            continue;
        end
        BridgeMask(i) = ~any(Local == Meta.feasible(i));
    end
end
```

Update `InitBoundaryMeta` / `BuildBoundaryMeta` / `ExtractBoundaryMetaRow` so they all carry `oppDist` and `feasible`.

- [ ] **Step 3: Replace boundary-weighted archive assembly with sector-wise quotas**

Shrink the train archive to plain data plus source/sector/time, and trim it by per-sector quotas.

```matlab
function TrainArchive = InitTrainArchive(D)
    TrainArchive = struct( ...
        'Dec',zeros(0,D), ...
        'Label',zeros(0,1), ...
        'Source',zeros(0,1), ...
        'Sector',zeros(0,1), ...
        'Time',zeros(0,1));
end

function TrainArchive = AppendTrainArchiveWithMeta(TrainArchive,Population,Meta)
    if isempty(Population)
        return;
    end

    TrainArchive.Dec    = [TrainArchive.Dec;Population.decs];
    TrainArchive.Label  = [TrainArchive.Label;double(all(Population.cons<=0,2))];
    TrainArchive.Source = [TrainArchive.Source;Meta.source(:)];
    TrainArchive.Sector = [TrainArchive.Sector;Meta.sector(:)];
    TrainArchive.Time   = [TrainArchive.Time;Meta.time(:)];

    Keep = KeepLatestDecisionRowsLocal(TrainArchive.Dec);
    TrainArchive.Dec    = TrainArchive.Dec(Keep,:);
    TrainArchive.Label  = TrainArchive.Label(Keep);
    TrainArchive.Source = TrainArchive.Source(Keep);
    TrainArchive.Sector = TrainArchive.Sector(Keep);
    TrainArchive.Time   = TrainArchive.Time(Keep);
end

function TrainArchive = TrimTrainArchiveBySectorQuota(TrainArchive,MaxTrain)
    Keep = SelectArchiveRowsBySectorQuota(TrainArchive,MaxTrain);
    TrainArchive.Dec    = TrainArchive.Dec(Keep,:);
    TrainArchive.Label  = TrainArchive.Label(Keep);
    TrainArchive.Source = TrainArchive.Source(Keep);
    TrainArchive.Sector = TrainArchive.Sector(Keep);
    TrainArchive.Time   = TrainArchive.Time(Keep);
end
```

Use the following quotas inside `SelectArchiveRowsBySectorQuota`:

```matlab
Quota = [2 2 1 1];   % source 1:B, 2:recent boundary, 3:feasible rep, 4:infeasible rep
```

Sort candidates by:

```matlab
Key = [-TrainArchive.Time(CandidateIdx), CandidateIdx];
```

Pick rows only while both:

```matlab
perSectorSourceCount(sec,src) < Quota(src)
numel(Pick) < MaxTrain
```

- [ ] **Step 4: Reduce the MLP path to plain BCE**

Delete `UpdateCalibrationArchive`, `BuildSampleWeights`, `ResolveSourceWeights`, `FitTemperatureScaling`, and `ComputeBoundaryWeights`. Train and predict with unweighted BCE only.

```matlab
function Model = UpdateBoundaryModelIfDue( ...
    Model,TrainArchive,minPos,minNeg,hidden,epoch,lr,Gap,Tick)
    if ~CanTrainBoundaryModel(TrainArchive,minPos,minNeg)
        return;
    end
    if isempty(Model) || mod(Tick,Gap) == 0
        Model = TrainBoundaryMLP(TrainArchive,hidden,epoch,lr,Model);
    end
end

function Model = TrainBoundaryMLP(TrainArchive,Hidden,Epoch,LR,PrevModel)
    if nargin < 5
        PrevModel = [];
    end
    Model = PrevModel;

    X = double(TrainArchive.Dec);
    Y = double(TrainArchive.Label(:) > 0);
    if isempty(X) || size(X,1) < 4
        return;
    end
    if numel(unique(Y)) < 2
        return;
    end

    Hidden    = max(2,round(Hidden));
    Epoch     = max(1,round(Epoch));
    LR        = max(double(LR),1e-4);
    [N,D]     = size(X);
    LambdaReg = 1e-4;

    Mu    = mean(X,1);
    Sigma = std(X,0,1);
    Sigma(Sigma < 1e-12) = 1;

    if ~isempty(PrevModel) && IsWarmStartCompatible(PrevModel,D,Hidden)
        W1 = PrevModel.W1;
        b1 = PrevModel.b1;
        W2 = PrevModel.W2;
        b2 = PrevModel.b2;
    else
        W1 = 0.1*randn(D,Hidden);
        b1 = zeros(1,Hidden);
        W2 = 0.1*randn(Hidden,1);
        b2 = 0;
    end

    Xn = (X-Mu)./Sigma;
    for e = 1 : Epoch
        H = tanh(Xn*W1 + repmat(b1,N,1));
        Z = H*W2 + b2;
        P = 1./(1+exp(-Z));

        Delta2 = (P-Y)/N;
        dW2 = H'*Delta2 + LambdaReg*W2;
        db2 = sum(Delta2);

        D1  = (Delta2*W2').*(1-H.^2);
        dW1 = Xn'*D1 + LambdaReg*W1;
        db1 = sum(D1,1);

        Step = LR/sqrt(e);
        W1 = W1 - Step*dW1;
        b1 = b1 - Step*db1;
        W2 = W2 - Step*dW2;
        b2 = b2 - Step*db2;
    end
    Model = struct('Mu',Mu,'Sigma',Sigma,'W1',W1,'b1',b1,'W2',W2,'b2',b2);
end

function [Prob,Stats] = PredictBoundaryMLP(Model,X)
    if nargin < 2 || isempty(X)
        Prob  = zeros(0,1);
        Stats = struct('logit',zeros(0,1),'rawLogit',zeros(0,1));
        return;
    end
    if isempty(Model) || ~isfield(Model,'Mu')
        Prob  = 0.5*ones(size(X,1),1);
        Stats = struct('logit',zeros(size(X,1),1),'rawLogit',zeros(size(X,1),1));
        return;
    end

    Xn = (double(X)-Model.Mu)./Model.Sigma;
    H  = tanh(Xn*Model.W1 + repmat(Model.b1,size(Xn,1),1));
    RawZ = H*Model.W2 + Model.b2;
    Prob = 1./(1+exp(-RawZ));
    Prob = min(max(Prob,1e-6),1-1e-6);
    Stats = struct('logit',RawZ(:),'rawLogit',RawZ(:));
end
```

- [ ] **Step 5: Run the semantic regression test and verify it passes**

Run:

```bash
matlab -batch "cd('/Users/lanai/Code/Matlab/PlatEMO/PlatEMO'); addpath(genpath(pwd)); test_PRBCCMO_semantics"
```

Expected: PASS with exit code `0`.

- [ ] **Step 6: Prepare a commit checkpoint for the main-algorithm cleanup if the user explicitly requests commits**

```bash
git add "Algorithms/Multi-objective optimization/PRBCCMO/PRBCCMO.m" "Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_semantics.m"
git commit -m "refactor(prbccmo): move boundary learning to plain bce data semantics"
```

### Task 3: Mirror the Cleanup in `PRBCCMO_t` and Its CSV Consumers

**Files:**
- Modify: `Algorithms/Multi-objective optimization/PRBCCMO/PRBCCMO_t.m`
- Modify: `Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_t_metrics.m`
- Modify: `Algorithms/Multi-objective optimization/PRBCCMO/summarize_PRBCCMO_t_data.m`
- Test: `Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_t_metrics.m`

- [ ] **Step 1: Write the failing traced-schema assertions**

Update the traced metrics test so it expects the new schema and rejects the removed columns.

```matlab
requireColumns(summary.Properties.VariableNames,{ ...
    'boundary_attempts', ...
    'helper_real_opp_ratio','helper_skip_no_helper', ...
    'seed_b_size','seed_b_sector_coverage','seed_b_mixed_sectors', ...
    'b_trusted_count','b_trusted_sector_coverage','b_trusted_lowmargin_count', ...
    'b_bal_acc','b_brier','train_minus_boundary_bal_gap','train_minus_b_bal_gap'}, ...
    'test_PRBCCMO_t_metrics:SummaryColumns');

requireAbsentColumns(summary.Properties.VariableNames,{ ...
    'helper_pred_opp_ratio','train_mean_bound_weight', ...
    'calib_brier_holdout','calib_bal_acc_holdout','temperature'}, ...
    'test_PRBCCMO_t_metrics:SummaryNoLegacyColumns');

requireColumns(mlp.Properties.VariableNames,{ ...
    'trained','warm_start','acc_after','bal_acc_after','brier_after','logloss_after'}, ...
    'test_PRBCCMO_t_metrics:MLPColumns');

requireAbsentColumns(mlp.Properties.VariableNames,{ ...
    'mean_bound_weight','temperature','calib_size','calib_bal_acc'}, ...
    'test_PRBCCMO_t_metrics:MLPNoLegacyColumns');
```

Add the helper:

```matlab
function requireAbsentColumns(names,forbidden,identifier)
    present = forbidden(ismember(forbidden,names));
    assert(isempty(present),identifier, ...
        'Unexpected legacy columns: %s', strjoin(present,', '));
end
```

- [ ] **Step 2: Run the traced metrics test and verify it fails**

Run:

```bash
matlab -batch "cd('/Users/lanai/Code/Matlab/PlatEMO/PlatEMO'); addpath(genpath(pwd)); test_PRBCCMO_t_metrics"
```

Expected: FAIL with `test_PRBCCMO_t_metrics:SummaryNoLegacyColumns` or `test_PRBCCMO_t_metrics:MLPNoLegacyColumns` because the traced implementation still emits legacy calibration and weight columns.

- [ ] **Step 3: Mirror the plain-BCE cleanup in `PRBCCMO_t.m`**

Mirror the main algorithm changes explicitly in the traced variant:

```matlab
TrainArchive = InitTrainArchive(Problem.D);
[TrainArchive,TrainDiag] = UpdateTrainArchiveWithDiagnostics( ...
    TrainArchive,B,PopulationC,PopulationU,RecentBoundaryOff, ...
    W,SectorNeighbors,MaxTrain,0);
[Model,MLPDiag] = UpdateBoundaryModelWithDiagnostics( ...
    Model,TrainArchive,minPos,minNeg,hidden,epoch,lr, ...
    ResolveModelUpdateGap(gapMin,gapMax,Problem.FE/max(Problem.maxFE,1)),0);
Observer = LogMLPEvent(Observer,MLPDiag);
```

and later in the loop:

```matlab
RecentBoundaryOff = BoundaryOff;
[TrainArchive,TrainDiag] = UpdateTrainArchiveWithDiagnostics( ...
    TrainArchive,B,PopulationC,PopulationU,RecentBoundaryOff, ...
    W,SectorNeighbors,MaxTrain,Problem.FE);
[Model,MLPDiag] = UpdateBoundaryModelWithDiagnostics( ...
    Model,TrainArchive,minPos,minNeg,hidden,epoch,lr, ...
    ResolveModelUpdateGap(gapMin,gapMax,Problem.FE/max(Problem.maxFE,1)), ...
    Problem.FE);
Observer = LogMLPEvent(Observer,MLPDiag);
```

Port `ResolveBridgeBoundaryMask`, `TrimTrainArchiveBySectorQuota`, and the same plain-BCE `TrainBoundaryMLP` / `PredictBoundaryMLP` bodies from Task 2 into `PRBCCMO_t.m`. Then change the traced diagnostic structures to:

```matlab
Diag = struct( ...
    'gap',Gap, ...
    'tick',Tick, ...
    'train_size',size(TrainArchive.Dec,1), ...
    'pos_count',sum(TrainArchive.Label == 1), ...
    'neg_count',sum(TrainArchive.Label == 0), ...
    'src_b',sum(TrainArchive.Source == 1), ...
    'src_recent_boundary',sum(TrainArchive.Source == 2), ...
    'src_rep_c',sum(TrainArchive.Source == 3), ...
    'src_rep_u',sum(TrainArchive.Source == 4), ...
    'can_train',double(CanTrainBoundaryModel(TrainArchive,minPos,minNeg)), ...
    'trained',0, ...
    'warm_start',0, ...
    'model_ready_before',double(HasBoundaryModel(Model)), ...
    'model_ready_after',double(HasBoundaryModel(Model)), ...
    'stats_before',EvaluateBinaryPredictions(Model,TrainArchive.Dec,TrainArchive.Label), ...
    'stats_after',EvaluateBinaryPredictions(Model,TrainArchive.Dec,TrainArchive.Label));
```

Update the boundary-event accumulator to remove predicted-opposite bookkeeping:

```matlab
Diag = struct( ...
    'generation',Generation, ...
    'fe',FE, ...
    'budget',Budget, ...
    'attempts',0, ...
    'selected',0, ...
    'feasible',0, ...
    'infeasible',0, ...
    'helper_real_opp',0, ...
    'skipped_no_helper',0, ...
    'skipped_duplicate',0, ...
    'events',{cell(0,20)});
```

Change the CSV headers to match:

```matlab
WriteCsvHeader(Observer.summary_file,{ ...
    'generation','fe','fe_ratio', ...
    'popc_feasible_ratio','popu_feasible_ratio', ...
    'offspringc_feasible_ratio','offspringu_feasible_ratio', ...
    'boundary_budget','boundary_attempts','boundary_selected','boundary_feasible_ratio', ...
    'boundary_survive_c','boundary_survive_u', ...
    'helper_real_opp_ratio','helper_skip_no_helper', ...
    'seed_b_size','seed_b_sector_coverage','seed_b_mixed_sectors', ...
    'b_size','b_feasible_ratio','b_sector_coverage','b_mixed_sectors', ...
    'b_trusted_count','b_trusted_sector_coverage','b_trusted_lowmargin_count', ...
    'b_mean_margin','b_mean_opp_support','b_mean_oppdist','b_mean_score', ...
    'lowmargin_count','lowmargin_feasible_ratio','lowmargin_mix_score','lowmargin_oppdist', ...
    'boundary_bal_acc','b_bal_acc','b_brier', ...
    'train_size','train_pos','train_neg', ...
    'train_src_b','train_src_recent_boundary','train_src_rep_c','train_src_rep_u', ...
    'model_ready_before','model_trained','model_ready_after', ...
    'train_bal_acc_after','train_minus_boundary_bal_gap','train_minus_b_bal_gap'});
```

And:

```matlab
WriteCsvHeader(Observer.mlp_file,{ ...
    'tick','gap','can_train','trained','warm_start', ...
    'model_ready_before','model_ready_after', ...
    'train_size','pos_count','neg_count', ...
    'src_b','src_recent_boundary','src_rep_c','src_rep_u', ...
    'acc_before','bal_acc_before','brier_before','logloss_before', ...
    'acc_after','bal_acc_after','brier_after','logloss_after'});
```

- [ ] **Step 4: Update the summarizer to consume only the new columns**

Use:

```matlab
requiredCols = { ...
    'seed_b_size','seed_b_sector_coverage','seed_b_mixed_sectors', ...
    'b_trusted_count','b_trusted_sector_coverage','b_trusted_lowmargin_count', ...
    'b_bal_acc','b_brier','train_minus_boundary_bal_gap','train_minus_b_bal_gap', ...
    'boundary_attempts'};
```

Build the row without any calibration or temperature fields:

```matlab
row = { ...
    string(meta.problem{1}), ...
    string(runFolder), ...
    runs(i).datenum, ...
    meta.N(1), ...
    meta.maxFE(1), ...
    last.fe, ...
    last.fe_ratio, ...
    last.popc_feasible_ratio, ...
    last.popu_feasible_ratio, ...
    last.seed_b_size, ...
    last.seed_b_sector_coverage, ...
    last.seed_b_mixed_sectors, ...
    last.b_size, ...
    last.b_sector_coverage, ...
    last.b_mixed_sectors, ...
    last.b_trusted_count, ...
    last.b_trusted_sector_coverage, ...
    last.b_trusted_lowmargin_count, ...
    last.lowmargin_count, ...
    last.lowmargin_feasible_ratio, ...
    last.lowmargin_mix_score, ...
    last.lowmargin_oppdist, ...
    LastNonMissing(gen.boundary_bal_acc), ...
    LastNonMissing(gen.b_bal_acc), ...
    LastNonMissing(gen.b_brier), ...
    LastNonMissing(gen.train_bal_acc_after), ...
    LastNonMissing(gen.train_minus_boundary_bal_gap), ...
    LastNonMissing(gen.train_minus_b_bal_gap), ...
    mean(gen.helper_real_opp_ratio,'omitnan'), ...
    mean(gen.boundary_attempts,'omitnan'), ...
    sum(mlp.trained > 0), ...
    sum(mlp.model_ready_after > 0)};
```

- [ ] **Step 5: Run the traced metrics regression test and verify it passes**

Run:

```bash
matlab -batch "cd('/Users/lanai/Code/Matlab/PlatEMO/PlatEMO'); addpath(genpath(pwd)); test_PRBCCMO_t_metrics"
```

Expected: PASS with exit code `0`.

- [ ] **Step 6: Prepare a commit checkpoint for the traced cleanup if the user explicitly requests commits**

```bash
git add "Algorithms/Multi-objective optimization/PRBCCMO/PRBCCMO_t.m" "Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_t_metrics.m" "Algorithms/Multi-objective optimization/PRBCCMO/summarize_PRBCCMO_t_data.m"
git commit -m "refactor(prbccmo): sync traced diagnostics with plain bce core"
```

### Task 4: Run the Full Verification Set

**Files:**
- Test: `Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_semantics.m`
- Test: `Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_t_smoke.m`
- Test: `Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_t_metrics.m`

- [ ] **Step 1: Run the semantic guard**

Run:

```bash
matlab -batch "cd('/Users/lanai/Code/Matlab/PlatEMO/PlatEMO'); addpath(genpath(pwd)); test_PRBCCMO_semantics"
```

Expected: PASS with exit code `0`.

- [ ] **Step 2: Run the traced smoke test**

Run:

```bash
matlab -batch "cd('/Users/lanai/Code/Matlab/PlatEMO/PlatEMO'); addpath(genpath(pwd)); test_PRBCCMO_t_smoke"
```

Expected: PASS with exit code `0`, plus a new `Data/PRBCCMO_t/PRBCCMO_t_LIRCMOP1_BC_*` folder containing:

```text
run_meta.csv
generation_summary.csv
boundary_event.csv
archive_members.csv
mlp_events.csv
```

- [ ] **Step 3: Run the traced metrics regression**

Run:

```bash
matlab -batch "cd('/Users/lanai/Code/Matlab/PlatEMO/PlatEMO'); addpath(genpath(pwd)); test_PRBCCMO_t_metrics"
```

Expected: PASS with exit code `0`.

- [ ] **Step 4: Inspect the final diff**

Run:

```bash
git diff -- "Algorithms/Multi-objective optimization/PRBCCMO/PRBCCMO.m" "Algorithms/Multi-objective optimization/PRBCCMO/PRBCCMO_t.m" "Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_semantics.m" "Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_t_metrics.m" "Algorithms/Multi-objective optimization/PRBCCMO/summarize_PRBCCMO_t_data.m"
```

Expected: only the planned plain-BCE, bridge-gating, sector-quota, and traced-schema changes appear.

- [ ] **Step 5: Prepare a verified commit checkpoint if the user explicitly requests commits**

```bash
git add "Algorithms/Multi-objective optimization/PRBCCMO/PRBCCMO.m" "Algorithms/Multi-objective optimization/PRBCCMO/PRBCCMO_t.m" "Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_semantics.m" "Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_t_metrics.m" "Algorithms/Multi-objective optimization/PRBCCMO/summarize_PRBCCMO_t_data.m"
git commit -m "test(prbccmo): verify boundary-data plain bce workflow"
```
