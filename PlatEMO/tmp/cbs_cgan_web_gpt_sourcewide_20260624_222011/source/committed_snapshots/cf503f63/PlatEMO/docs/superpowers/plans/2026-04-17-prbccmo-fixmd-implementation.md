# PRBCCMO fix.md Implementation Plan

> Superseded on 2026-04-18 by the current pair-centric `fix.md` implementation state.
> The old plan below still documents the previous `trusted sector / paired sector / bridgeGrowth` tightening path and should not be used as the active implementation checklist.
> Current enforced checkpoints are:
> 1. `ReserveAnchorPairs` instead of sector-side quota
> 2. `CanTrainBoundaryModel` based on `anchor_count` and `pair_count`
> 3. global helper/support eligibility instead of sector gating
> 4. traced metrics aligned to `anchor/pair` diagnostics

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `PRBCCMO.m` and `PRBCCMO_t.m` strictly conform to `fix.md` by applying only the four required semantic tightenings and protecting them with minimal MATLAB regression tests.

**Architecture:** Keep the current dual-population and anchor-helper-probe workflow intact, but tighten four interfaces only: model training gate, opposite-side support construction, boundary archive seed pool, and bridge admission for untrusted sectors. Mirror the same semantics in `PRBCCMO_t.m` so traced behavior and main behavior do not diverge.

**Tech Stack:** MATLAB, PlatEMO, `matlab -batch`

**Repo Note:** This repository forbids unsolicited commits. During execution here, treat every commit step as a checkpoint to use only if the user explicitly asks for commits.

---

## File Structure

- Modify: `Algorithms/Multi-objective optimization/PRBCCMO/PRBCCMO.m`
  Responsibility: main algorithm, strict train gate, core-only support, narrowed `SeedPool`, strict bridge admission.
- Modify: `Algorithms/Multi-objective optimization/PRBCCMO/PRBCCMO_t.m`
  Responsibility: traced algorithm with the same semantics as `PRBCCMO.m`, including traced model-update gating.
- Create: `Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_semantics.m`
  Responsibility: static MATLAB regression guard for the four `fix.md` semantics across both algorithm files.

## Task 1: Create the Failing Semantic Regression Test

**Files:**
- Create: `Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_semantics.m`
- Test: `Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_semantics.m`

- [ ] **Step 1: Write the failing semantic regression test**

Create `Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_semantics.m` with this content:

```matlab
function test_PRBCCMO_semantics()
% Static regression guard for the four fix.md semantic tightenings.

    rootDir = fileparts(mfilename('fullpath'));
    mainText = normalizeText(fileread(fullfile(rootDir,'PRBCCMO.m')));
    traceText = normalizeText(fileread(fullfile(rootDir,'PRBCCMO_t.m')));

    requireContains(mainText, ...
        'SeedPool = KeepUniquePopulation([B,OffspringC,OffspringU]);', ...
        'test_PRBCCMO_semantics:MainSeedPool', ...
        'PRBCCMO.m must narrow SeedPool to B plus fresh offspring only.');
    requireContains(traceText, ...
        'SeedPool = KeepUniquePopulation([B,OffspringC,OffspringU]);', ...
        'test_PRBCCMO_semantics:TraceSeedPool', ...
        'PRBCCMO_t.m must narrow SeedPool to B plus fresh offspring only.');

    requireContains(mainText, ...
        'Flag = nnz(Trusted) >= max(2,M) && PairCnt >= max(2,M);', ...
        'test_PRBCCMO_semantics:MainStrictTrainGate', ...
        'PRBCCMO.m must require trusted sectors and paired sectors before training.');
    requireContains(traceText, ...
        'Flag = nnz(Trusted) >= max(2,M) && PairCnt >= max(2,M);', ...
        'test_PRBCCMO_semantics:TraceStrictTrainGate', ...
        'PRBCCMO_t.m must require trusted sectors and paired sectors before training.');

    requireContains(mainText, ...
        'Support = BuildBoundarySupportSet(TrainArchive);', ...
        'test_PRBCCMO_semantics:MainCoreOnlySupport', ...
        'PRBCCMO.m must build archive support from TrainArchive core only.');
    requireContains(traceText, ...
        'Support = BuildBoundarySupportSet(TrainArchive);', ...
        'test_PRBCCMO_semantics:TraceCoreOnlySupport', ...
        'PRBCCMO_t.m must build archive support from TrainArchive core only.');

    requireContains(mainText, ...
        'Support = AppendSupportPair(Support,Anchor,AnchorMeta,Helper,HelperMeta);', ...
        'test_PRBCCMO_semantics:MainProbeSupportPair', ...
        'PRBCCMO.m must add only the real anchor/helper pair for probe support.');
    requireContains(traceText, ...
        'Support = AppendSupportPair(Support,Anchor,AnchorMeta,Helper,HelperMeta);', ...
        'test_PRBCCMO_semantics:TraceProbeSupportPair', ...
        'PRBCCMO_t.m must add only the real anchor/helper pair for probe support.');

    requireContains(mainText, ...
        'BridgeMask = Meta.trusted(:) | Meta.bridgeGrowth(:);', ...
        'test_PRBCCMO_semantics:MainBridgeMask', ...
        'PRBCCMO.m must gate archive admission by trusted || bridgeGrowth only.');
    requireContains(traceText, ...
        'BridgeMask = Meta.trusted(:) | Meta.bridgeGrowth(:);', ...
        'test_PRBCCMO_semantics:TraceBridgeMask', ...
        'PRBCCMO_t.m must gate archive admission by trusted || bridgeGrowth only.');

    requireNotContains(mainText, ...
        'Support = AppendSupportPopulation(Support,Candidates,W,RefObj);', ...
        'test_PRBCCMO_semantics:MainNoCandidateSupport', ...
        'PRBCCMO.m must not add Candidates into the opposite-support set.');
    requireNotContains(traceText, ...
        'Support = AppendSupportPopulation(Support,Candidates,W,RefObj);', ...
        'test_PRBCCMO_semantics:TraceNoCandidateSupport', ...
        'PRBCCMO_t.m must not add Candidates into the opposite-support set.');

    requireNotContains(mainText, ...
        'BridgeMask(i) = HasRealLocalOppositeSupport(Meta,i);', ...
        'test_PRBCCMO_semantics:MainNoWideBridge', ...
        'PRBCCMO.m must not admit untrusted sectors through finite oppDist alone.');
    requireNotContains(traceText, ...
        'BridgeMask(i) = HasRealLocalOppositeSupport(Meta,i);', ...
        'test_PRBCCMO_semantics:TraceNoWideBridge', ...
        'PRBCCMO_t.m must not admit untrusted sectors through finite oppDist alone.');
end

function requireContains(text,pattern,id,msg)
    assert(contains(text,pattern),id,msg);
end

function requireNotContains(text,pattern,id,msg)
    assert(~contains(text,pattern),id,msg);
end

function text = normalizeText(text)
    text = regexprep(text,'\s+',' ');
end
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
matlab -batch "cd('/Users/lanai/Code/Matlab/PlatEMO/PlatEMO'); addpath(genpath(pwd)); cd('Algorithms/Multi-objective optimization/PRBCCMO'); test_PRBCCMO_semantics"
```

Expected: FAIL with one of `test_PRBCCMO_semantics:MainSeedPool`, `MainStrictTrainGate`, `MainCoreOnlySupport`, `MainBridgeMask`, or the traced equivalents, because the current code still uses the old semantics.

- [ ] **Step 3: Prepare a red-state commit checkpoint only if the user explicitly asks for commits**

```bash
git add "Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_semantics.m"
git commit -m "test(prbccmo): lock fixmd semantics"
```

## Task 2: Refactor `PRBCCMO.m` to Match `fix.md`

**Files:**
- Modify: `Algorithms/Multi-objective optimization/PRBCCMO/PRBCCMO.m`
- Test: `Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_semantics.m`

- [ ] **Step 1: Tighten the model-update call sites and train gate**

Update the initialization-time and in-loop calls so the model gate receives `SectorNeighbors`, sector count, and objective count:

```matlab
Model = UpdateBoundaryModelIfDue( ...
    Model,TrainArchive,minPos,minNeg,hidden,epoch,lr, ...
    ResolveModelUpdateGap(gapMin,gapMax,Problem.FE/max(Problem.maxFE,1)), ...
    0,SectorNeighbors,size(W,1),Problem.M);
```

and later:

```matlab
Model = UpdateBoundaryModelIfDue( ...
    Model,TrainArchive,minPos,minNeg,hidden,epoch,lr, ...
    ResolveModelUpdateGap(gapMin,gapMax,Problem.FE/max(Problem.maxFE,1)), ...
    Problem.FE,SectorNeighbors,size(W,1),Problem.M);
```

Replace the gate helpers with:

```matlab
function Model = UpdateBoundaryModelIfDue( ...
    Model,TrainArchive,minPos,minNeg,hidden,epoch,lr,Gap,Tick, ...
    SectorNeighbors,K,M)
    if ~CanTrainBoundaryModel(TrainArchive,minPos,minNeg,SectorNeighbors,K,M)
        return;
    end

    if isempty(Model) || mod(Tick,Gap) == 0
        Model = TrainBoundaryMLP(TrainArchive,hidden,epoch,lr,Model);
    end
end

function Flag = CanTrainBoundaryModel(TrainArchive,minPos,minNeg,SectorNeighbors,K,M)
    if sum(TrainArchive.Label == 1) < minPos || ...
       sum(TrainArchive.Label == 0) < minNeg
        Flag = false;
        return;
    end

    Trusted = ResolveTrustedSectorsFromTrainArchive(TrainArchive,SectorNeighbors,K);
    PairCnt = CountCorePairedSectors(TrainArchive,SectorNeighbors,K);
    Flag = nnz(Trusted) >= max(2,M) && PairCnt >= max(2,M);
end

function Count = CountCorePairedSectors(TrainArchive,SectorNeighbors,K)
    Count = 0;
    CoreMask = IsCoreTrainSource(TrainArchive.Source);
    CoreLabel = TrainArchive.Label(CoreMask);
    CoreSector = TrainArchive.Sector(CoreMask);
    for s = 1 : K
        Allowed = ResolveLocalSectorSet(s,SectorNeighbors);
        Local = CoreLabel(ismember(CoreSector,Allowed));
        if any(Local == 0) && any(Local == 1)
            Count = Count + 1;
        end
    end
end
```

- [ ] **Step 2: Make archive support core-only and probe support pair-only**

In `BuildBoundaryMeta`, replace the existing support construction with:

```matlab
Support = BuildBoundarySupportSet(TrainArchive);
```

Replace the support helper block with:

```matlab
function Support = BuildBoundarySupportSet(TrainArchive)
    D = size(TrainArchive.Dec,2);
    Support = struct( ...
        'Dec',zeros(0,D), ...
        'Label',zeros(0,1), ...
        'Sector',zeros(0,1));
    if isempty(TrainArchive.Dec)
        return;
    end

    CoreMask = IsCoreTrainSource(TrainArchive.Source);
    Support.Dec    = TrainArchive.Dec(CoreMask,:);
    Support.Label  = TrainArchive.Label(CoreMask);
    Support.Sector = TrainArchive.Sector(CoreMask);
end

function Support = AppendSupportPair(Support,Anchor,AnchorMeta,Helper,HelperMeta)
    PairDec = [Anchor.decs;Helper.decs];
    PairLabel = [double(all(Anchor.cons<=0,2)); double(all(Helper.cons<=0,2))];
    PairSector = [AnchorMeta.sector;HelperMeta.sector];

    Support.Dec    = [Support.Dec;PairDec];
    Support.Label  = [Support.Label;PairLabel];
    Support.Sector = [Support.Sector;PairSector];

    Keep = KeepLatestDecisionRowsLocal(Support.Dec);
    Support.Dec    = Support.Dec(Keep,:);
    Support.Label  = Support.Label(Keep);
    Support.Sector = Support.Sector(Keep);
end
```

Then in `GenerateBoundaryOffspringModelDriven`, replace:

```matlab
Support = BuildBoundarySupportSet( ...
    B,PopulationC,PopulationU,TrainArchive,W,RefObj);
```

with:

```matlab
Support = BuildBoundarySupportSet(TrainArchive);
```

And inside the probe loop, just before `RankBoundaryProbeCandidates`, insert:

```matlab
ProbeSupport = AppendSupportPair(Support,Anchor,AnchorMeta,Helper,HelperMeta);
```

Then call:

```matlab
[~,Score] = RankBoundaryProbeCandidates( ...
    Model,ProbeDec,Beta,Anchor,AnchorMeta,Helper,HelperMeta, ...
    ProbeSupport,SectorNeighbors);
```

- [ ] **Step 3: Narrow the seed pool and tighten bridge admission**

Replace the main-loop seed pool line with:

```matlab
SeedPool = KeepUniquePopulation([B,OffspringC,OffspringU]);
```

Replace the bridge admission helper with:

```matlab
function BridgeMask = ResolveBridgeBoundaryMask(Meta,TrainArchive,SectorNeighbors,K)
    %#ok<INUSD>
    BridgeMask = Meta.trusted(:) | Meta.bridgeGrowth(:);
end
```

Leave `ResolveBridgeGrowthMask` unchanged, because `fix.md` explicitly keeps bridge-growth semantics.

- [ ] **Step 4: Run the semantic regression to verify `PRBCCMO.m` now matches the new text guard**

Run:

```bash
matlab -batch "cd('/Users/lanai/Code/Matlab/PlatEMO/PlatEMO'); addpath(genpath(pwd)); cd('Algorithms/Multi-objective optimization/PRBCCMO'); test_PRBCCMO_semantics"
```

Expected: The main-file assertions now pass. The overall run may still fail on traced-file assertions until Task 3 is complete.

- [ ] **Step 5: Verify that MATLAB can still parse the main class**

Run:

```bash
matlab -batch "cd('/Users/lanai/Code/Matlab/PlatEMO/PlatEMO'); addpath(genpath(pwd)); clear classes; mc = meta.class.fromName('PRBCCMO'); assert(~isempty(mc),'PRBCCMO class failed to load');"
```

Expected: PASS with exit code `0`.

- [ ] **Step 6: Prepare a main-file checkpoint only if the user explicitly asks for commits**

```bash
git add "Algorithms/Multi-objective optimization/PRBCCMO/PRBCCMO.m" "Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_semantics.m"
git commit -m "refactor(prbccmo): tighten fixmd boundary semantics"
```

## Task 3: Mirror the Same Semantics in `PRBCCMO_t.m`

**Files:**
- Modify: `Algorithms/Multi-objective optimization/PRBCCMO/PRBCCMO_t.m`
- Test: `Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_semantics.m`

- [ ] **Step 1: Tighten the traced model gate with the same signature and thresholds**

Update the traced initialization-time and in-loop model calls to pass `SectorNeighbors`, `size(W,1)`, and `Problem.M`:

```matlab
[Model,MLPDiag] = UpdateBoundaryModelWithDiagnostics( ...
    Model,TrainArchive,minPos,minNeg,hidden,epoch,lr, ...
    ResolveModelUpdateGap(gapMin,gapMax,Problem.FE/max(Problem.maxFE,1)), ...
    0,SectorNeighbors,size(W,1),Problem.M);
```

and later:

```matlab
[Model,MLPDiag] = UpdateBoundaryModelWithDiagnostics( ...
    Model,TrainArchive,minPos,minNeg,hidden,epoch,lr, ...
    ResolveModelUpdateGap(gapMin,gapMax,Problem.FE/max(Problem.maxFE,1)), ...
    Problem.FE,SectorNeighbors,size(W,1),Problem.M);
```

Replace the traced gate helpers with:

```matlab
function [Model,Diag] = UpdateBoundaryModelWithDiagnostics( ...
    Model,TrainArchive,minPos,minNeg,hidden,epoch,lr,Gap,Tick, ...
    SectorNeighbors,K,M)

    Diag = InitMLPDiag(Model,TrainArchive,Gap,Tick,minPos,minNeg);
    Diag.can_train = double(CanTrainBoundaryModel( ...
        TrainArchive,minPos,minNeg,SectorNeighbors,K,M));
    if ~Diag.can_train
        return;
    end

    if isempty(Model) || mod(Tick,Gap) == 0
        Diag.warm_start = double(HasBoundaryModel(Model));
        Model = TrainBoundaryMLP(TrainArchive,hidden,epoch,lr,Model);
        Diag.trained = 1;
        Diag.model_ready_after = double(HasBoundaryModel(Model));
        Diag.stats_after = EvaluateBinaryPredictions(Model,TrainArchive.Dec,TrainArchive.Label);
    end
end
```

Reuse the same `CanTrainBoundaryModel` and `CountCorePairedSectors` logic as in `PRBCCMO.m`.

- [ ] **Step 2: Make traced support construction match the main algorithm**

Apply the same support changes as Task 2 Step 2:

```matlab
Support = BuildBoundarySupportSet(TrainArchive);
```

Add the same helper:

```matlab
function Support = AppendSupportPair(Support,Anchor,AnchorMeta,Helper,HelperMeta)
    PairDec = [Anchor.decs;Helper.decs];
    PairLabel = [double(all(Anchor.cons<=0,2)); double(all(Helper.cons<=0,2))];
    PairSector = [AnchorMeta.sector;HelperMeta.sector];

    Support.Dec    = [Support.Dec;PairDec];
    Support.Label  = [Support.Label;PairLabel];
    Support.Sector = [Support.Sector;PairSector];

    Keep = KeepLatestDecisionRowsLocal(Support.Dec);
    Support.Dec    = Support.Dec(Keep,:);
    Support.Label  = Support.Label(Keep);
    Support.Sector = Support.Sector(Keep);
end
```

Inside the traced probe loop, insert:

```matlab
ProbeSupport = AppendSupportPair(Support,Anchor,AnchorMeta,Helper,HelperMeta);
```

and pass `ProbeSupport` into `RankBoundaryProbeCandidates`.

- [ ] **Step 3: Narrow traced `SeedPool` and tighten traced bridge admission**

Replace:

```matlab
SeedPool = KeepUniquePopulation([B,PopulationC,PopulationU,OffspringC,OffspringU]);
```

with:

```matlab
SeedPool = KeepUniquePopulation([B,OffspringC,OffspringU]);
```

Replace the traced bridge helper with:

```matlab
function BridgeMask = ResolveBridgeBoundaryMask(Meta,TrainArchive,SectorNeighbors,K)
    %#ok<INUSD>
    BridgeMask = Meta.trusted(:) | Meta.bridgeGrowth(:);
end
```

- [ ] **Step 4: Run the semantic regression and verify both files now pass**

Run:

```bash
matlab -batch "cd('/Users/lanai/Code/Matlab/PlatEMO/PlatEMO'); addpath(genpath(pwd)); cd('Algorithms/Multi-objective optimization/PRBCCMO'); test_PRBCCMO_semantics"
```

Expected: PASS with exit code `0`.

- [ ] **Step 5: Verify that MATLAB can still parse the traced class**

Run:

```bash
matlab -batch "cd('/Users/lanai/Code/Matlab/PlatEMO/PlatEMO'); addpath(genpath(pwd)); clear classes; mc = meta.class.fromName('PRBCCMO_t'); assert(~isempty(mc),'PRBCCMO_t class failed to load');"
```

Expected: PASS with exit code `0`.

- [ ] **Step 6: Prepare a traced-file checkpoint only if the user explicitly asks for commits**

```bash
git add "Algorithms/Multi-objective optimization/PRBCCMO/PRBCCMO_t.m" "Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_semantics.m"
git commit -m "refactor(prbccmo): sync traced fixmd semantics"
```

## Task 4: Final Verification and Diff Review

**Files:**
- Modify: `Algorithms/Multi-objective optimization/PRBCCMO/PRBCCMO.m`
- Modify: `Algorithms/Multi-objective optimization/PRBCCMO/PRBCCMO_t.m`
- Create: `Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_semantics.m`

- [ ] **Step 1: Run the full minimal verification set**

Run:

```bash
matlab -batch "cd('/Users/lanai/Code/Matlab/PlatEMO/PlatEMO'); addpath(genpath(pwd)); cd('Algorithms/Multi-objective optimization/PRBCCMO'); test_PRBCCMO_semantics"
matlab -batch "cd('/Users/lanai/Code/Matlab/PlatEMO/PlatEMO'); addpath(genpath(pwd)); clear classes; mc = meta.class.fromName('PRBCCMO'); assert(~isempty(mc),'PRBCCMO class failed to load');"
matlab -batch "cd('/Users/lanai/Code/Matlab/PlatEMO/PlatEMO'); addpath(genpath(pwd)); clear classes; mc = meta.class.fromName('PRBCCMO_t'); assert(~isempty(mc),'PRBCCMO_t class failed to load');"
```

Expected: All three commands PASS with exit code `0`.

- [ ] **Step 2: Inspect the final diff**

Run:

```bash
git diff -- "Algorithms/Multi-objective optimization/PRBCCMO/PRBCCMO.m" "Algorithms/Multi-objective optimization/PRBCCMO/PRBCCMO_t.m" "Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_semantics.m"
```

Expected: The diff shows only the four required semantic tightenings plus the new regression test.

- [ ] **Step 3: Prepare a final checkpoint only if the user explicitly asks for commits**

```bash
git add "Algorithms/Multi-objective optimization/PRBCCMO/PRBCCMO.m" "Algorithms/Multi-objective optimization/PRBCCMO/PRBCCMO_t.m" "Algorithms/Multi-objective optimization/PRBCCMO/test_PRBCCMO_semantics.m"
git commit -m "refactor(prbccmo): enforce fixmd semantics"
```
