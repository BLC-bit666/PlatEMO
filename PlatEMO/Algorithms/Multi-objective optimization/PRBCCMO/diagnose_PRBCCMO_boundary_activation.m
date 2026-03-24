function Results = diagnose_PRBCCMO_boundary_activation(varargin)
% Diagnose PRBCCMO boundary activation with in-memory generation traces only.
%
% Optional name-value pairs:
%   'Runs'        : number of runs, default 1
%   'RunSeeds'    : explicit seed list, default 1001:1000+Runs
%   'Population'  : population size N, default 50
%   'MaxFE'       : maximum function evaluations, default 10000
%   'ProblemNames': cell array of problem names, default {'DASCMOP1_BC'}
%   'Variant'     : raw | temperature | beta | auto_trust, default auto_trust
%   'Verbose'     : print per-run summary, default true

    Params = struct( ...
        'Runs',1, ...
        'RunSeeds',[], ...
        'Population',50, ...
        'MaxFE',10000, ...
        'ProblemNames',{{'DASCMOP1_BC'}}, ...
        'Variant','auto_trust', ...
        'Verbose',true);
    Params = ParseInputs(Params,varargin{:});
    Params = NormalizeRunControl(Params);

    Variant = ResolveDiagnosticVariant(Params.Variant);
    Tasks = BuildDiagnosticTasks(Params.ProblemNames,Params.RunSeeds);
    [RunResults,ExecutionInfo] = ExecuteDiagnosticTasks(Tasks,Params,Variant);

    Results = struct();
    Results.params = Params;
    Results.variant = Variant;
    Results.runResults = RunResults;
    Results.problemSummary = SummarizeProblemRuns(RunResults,Params.ProblemNames,Variant.name);
    Results.overallSummary = SummarizeOverallRuns(RunResults,Variant.name);
    Results.execution = ExecutionInfo;
end

function Tasks = BuildDiagnosticTasks(ProblemNames,RunSeeds)
    TaskCount = numel(ProblemNames)*numel(RunSeeds);
    Tasks = repmat(InitDiagnosticTask(),TaskCount,1);
    Row = 0;
    for p = 1 : numel(ProblemNames)
        for r = 1 : numel(RunSeeds)
            Row = Row + 1;
            Tasks(Row).problem = ProblemNames{p};
            Tasks(Row).run = r;
            Tasks(Row).seed = RunSeeds(r);
        end
    end
end

function Task = InitDiagnosticTask()
    Task = struct( ...
        'problem','', ...
        'run',0, ...
        'seed',0);
end

function [RunResults,ExecutionInfo] = ExecuteDiagnosticTasks(Tasks,Params,Variant)
    ProjectRoot = ResolveProjectRoot();
    ExecutionInfo = struct( ...
        'taskCount',numel(Tasks), ...
        'projectRoot',ProjectRoot, ...
        'scriptPath',mfilename('fullpath'));
    RunResults = repmat(InitActivationRunResult(),numel(Tasks),1);
    for t = 1 : numel(Tasks)
        RunResults(t,1) = RunDiagnosticTask( ...
            Tasks(t),Params.Population,Params.MaxFE,ProjectRoot,Variant);
        if Params.Verbose
            Summary = RunResults(t).summary;
            fprintf(['Task %03d/%03d | %s | run=%02d seed=%d | ' ...
                'maxRF=%d maxEA=%d maxFA=%d maxCand=%d maxSeed=%d maxBO=%d | blocker=%s\n'], ...
                t,numel(Tasks),Tasks(t).problem,Tasks(t).run,Tasks(t).seed, ...
                Summary.maxRegularFeasibleCount,Summary.maxExternalArchiveCount, ...
                Summary.maxFeasibleAnchorCount,Summary.maxCandidatePoolSize, ...
                Summary.maxBoundarySeedCount,Summary.maxBoundaryOffspringCount, ...
                Summary.blocker);
        end
    end
end

function Result = RunDiagnosticTask(Task,Population,MaxFE,ProjectRoot,Variant)
    EnsureDiagnosticPath(ProjectRoot);
    rng(Task.seed,'twister');

    Problem = feval(Task.problem,'N',Population,'maxFE',MaxFE);
    Algorithm = PRBCCMO('parameter',Variant.parameter,'save',0,'outputFcn',@SilentOutput);
    Algorithm.Solve(Problem);

    Trace = ResolveActivationTrace(Algorithm.metric);
    SelectionTrace = ResolveSelectionTrace(Algorithm.metric);
    BridgeTrace = ResolveBridgeTrace(Algorithm.metric);
    Summary = SummarizeActivationTrace(Trace,Task.problem,Variant.name,Task.run,Task.seed);
    Summary = AttachSelectionSummary(Summary,SelectionTrace);
    Summary = AttachBridgeSummary(Summary,BridgeTrace);

    Result = InitActivationRunResult();
    Result.problem = Task.problem;
    Result.variant = Variant.name;
    Result.run = Task.run;
    Result.seed = Task.seed;
    Result.activationTrace = Trace;
    Result.selectionTrace = SelectionTrace;
    Result.bridgeTrace = BridgeTrace;
    Result.summary = Summary;
end

function ProjectRoot = ResolveProjectRoot()
    ScriptPath = mfilename('fullpath');
    if isempty(ScriptPath)
        ProjectRoot = pwd;
        return;
    end
    ScriptDir = fileparts(ScriptPath);
    ProjectRoot = fileparts(fileparts(fileparts(ScriptDir)));
end

function EnsureDiagnosticPath(ProjectRoot)
    persistent PathReady CachedRoot
    if nargin < 1 || isempty(ProjectRoot)
        return;
    end
    if ~isempty(PathReady) && PathReady && isequal(CachedRoot,ProjectRoot)
        return;
    end
    cd(ProjectRoot);
    addpath(genpath(ProjectRoot));
    CachedRoot = ProjectRoot;
    PathReady = true;
end

function SilentOutput(~,~)
end

function Trace = ResolveActivationTrace(Metric)
    Trace = repmat(InitActivationTraceRow(),0,1);
    if ~isstruct(Metric) || ~isfield(Metric,'sectionB') || isempty(Metric.sectionB)
        return;
    end
    if ~isfield(Metric.sectionB,'activationTrace') || isempty(Metric.sectionB.activationTrace)
        return;
    end
    Trace = Metric.sectionB.activationTrace;
end

function Trace = ResolveSelectionTrace(Metric)
    Trace = repmat(InitSelectionTraceRow(),0,1);
    if ~isstruct(Metric) || ~isfield(Metric,'sectionB') || isempty(Metric.sectionB)
        return;
    end
    if ~isfield(Metric.sectionB,'selectionTrace') || isempty(Metric.sectionB.selectionTrace)
        return;
    end
    Trace = Metric.sectionB.selectionTrace;
end

function Trace = ResolveBridgeTrace(Metric)
    Trace = repmat(InitBridgeTraceRow(),0,1);
    if ~isstruct(Metric) || ~isfield(Metric,'sectionB') || isempty(Metric.sectionB)
        return;
    end
    if ~isfield(Metric.sectionB,'bridgeTrace') || isempty(Metric.sectionB.bridgeTrace)
        return;
    end
    Trace = Metric.sectionB.bridgeTrace;
end

function Row = InitActivationTraceRow()
    Row = struct( ...
        'generation',NaN, ...
        'FE',NaN, ...
        'regularFeasibleCount',0, ...
        'externalArchiveCount',0, ...
        'feasibleAnchorCount',0, ...
        'candidatePoolSize',0, ...
        'boundarySeedCount',0, ...
        'boundaryOffspringCount',0);
end

function Row = InitSelectionTraceRow()
    Row = struct( ...
        'generation',NaN, ...
        'FE',NaN, ...
        'budget',0, ...
        'selectionMode',1, ...
        'hasModel',false, ...
        'trustGate',false, ...
        'candidateCount',0, ...
        'eligibleCount',0, ...
        'ineligibleCount',0, ...
        'finiteScoreCount',0, ...
        'validCount',0, ...
        'selectedCount',0, ...
        'positiveParetoCount',0, ...
        'maxRankScore',NaN, ...
        'maxParetoValue',NaN, ...
        'maxQueryScore',NaN, ...
        'maxBoundaryTrust',NaN);
end

function Row = InitBridgeTraceRow()
    Row = struct( ...
        'generation',NaN, ...
        'FE',NaN, ...
        'feasibleAnchorCount',0, ...
        'infeasibleHelperCount',0, ...
        'feasibleSectorCount',0, ...
        'infeasibleSectorCount',0, ...
        'sharedSectorCount',0, ...
        'activeSectorCount',0, ...
        'strictActiveSectorCount',0, ...
        'weakActiveSectorCount',0, ...
        'usedWeakGate',false, ...
        'deltaG',0, ...
        'minRawMargin',NaN, ...
        'medianRawMargin',NaN, ...
        'maxRawMargin',NaN, ...
        'minActivationMargin',NaN, ...
        'medianActivationMargin',NaN, ...
        'maxActivationMargin',NaN);
end

function Result = InitActivationRunResult()
    Result = struct( ...
        'problem','', ...
        'variant','', ...
        'run',0, ...
        'seed',0, ...
        'activationTrace',repmat(InitActivationTraceRow(),0,1), ...
        'selectionTrace',repmat(InitSelectionTraceRow(),0,1), ...
        'bridgeTrace',repmat(InitBridgeTraceRow(),0,1), ...
        'summary',InitActivationRunSummary());
end

function Summary = InitActivationRunSummary()
    Summary = struct( ...
        'problem','', ...
        'variant','', ...
        'run',0, ...
        'seed',0, ...
        'steps',0, ...
        'lastGeneration',NaN, ...
        'lastFE',NaN, ...
        'maxRegularFeasibleCount',0, ...
        'maxExternalArchiveCount',0, ...
        'maxFeasibleAnchorCount',0, ...
        'maxCandidatePoolSize',0, ...
        'maxBoundarySeedCount',0, ...
        'maxBoundaryOffspringCount',0, ...
        'maxSharedSectorCount',0, ...
        'maxActiveSectorCount',0, ...
        'maxBridgeMargin',NaN, ...
        'minBridgeMargin',NaN, ...
        'maxEligibleCandidateCount',0, ...
        'maxFiniteRankScoreCount',0, ...
        'maxValidCandidateCount',0, ...
        'maxPositiveParetoCount',0, ...
        'regularFeasibleEver',false, ...
        'externalArchiveEver',false, ...
        'feasibleAnchorEver',false, ...
        'candidatePoolEver',false, ...
        'boundarySeedEver',false, ...
        'boundaryOffspringEver',false, ...
        'sharedSectorEver',false, ...
        'activeSectorEver',false, ...
        'eligibleCandidateEver',false, ...
        'validCandidateEver',false, ...
        'firstRegularFeasibleGeneration',NaN, ...
        'firstRegularFeasibleFE',NaN, ...
        'firstExternalArchiveGeneration',NaN, ...
        'firstExternalArchiveFE',NaN, ...
        'firstFeasibleAnchorGeneration',NaN, ...
        'firstFeasibleAnchorFE',NaN, ...
        'firstCandidateGeneration',NaN, ...
        'firstCandidateFE',NaN, ...
        'firstSharedSectorGeneration',NaN, ...
        'firstSharedSectorFE',NaN, ...
        'firstActiveSectorGeneration',NaN, ...
        'firstActiveSectorFE',NaN, ...
        'firstEligibleCandidateGeneration',NaN, ...
        'firstEligibleCandidateFE',NaN, ...
        'firstValidCandidateGeneration',NaN, ...
        'firstValidCandidateFE',NaN, ...
        'firstBoundarySeedGeneration',NaN, ...
        'firstBoundarySeedFE',NaN, ...
        'firstBoundaryOffspringGeneration',NaN, ...
        'firstBoundaryOffspringFE',NaN, ...
        'bridgeBlocker','no_bridge_trace', ...
        'selectionBlocker','no_selection_trace', ...
        'blocker','no_trace');
end

function Summary = SummarizeActivationTrace(Trace,ProblemName,VariantName,RunIndex,Seed)
    Summary = InitActivationRunSummary();
    Summary.problem = ProblemName;
    Summary.variant = VariantName;
    Summary.run = RunIndex;
    Summary.seed = Seed;
    Summary.steps = numel(Trace);
    if isempty(Trace)
        return;
    end

    Summary.lastGeneration = Trace(end).generation;
    Summary.lastFE = Trace(end).FE;
    Summary.maxRegularFeasibleCount = ResolveMaxTraceField(Trace,'regularFeasibleCount');
    Summary.maxExternalArchiveCount = ResolveMaxTraceField(Trace,'externalArchiveCount');
    Summary.maxFeasibleAnchorCount = ResolveMaxTraceField(Trace,'feasibleAnchorCount');
    Summary.maxCandidatePoolSize = ResolveMaxTraceField(Trace,'candidatePoolSize');
    Summary.maxBoundarySeedCount = ResolveMaxTraceField(Trace,'boundarySeedCount');
    Summary.maxBoundaryOffspringCount = ResolveMaxTraceField(Trace,'boundaryOffspringCount');

    Summary.regularFeasibleEver = Summary.maxRegularFeasibleCount > 0;
    Summary.externalArchiveEver = Summary.maxExternalArchiveCount > 0;
    Summary.feasibleAnchorEver = Summary.maxFeasibleAnchorCount > 0;
    Summary.candidatePoolEver = Summary.maxCandidatePoolSize > 0;
    Summary.boundarySeedEver = Summary.maxBoundarySeedCount > 0;
    Summary.boundaryOffspringEver = Summary.maxBoundaryOffspringCount > 0;

    [Summary.firstRegularFeasibleGeneration,Summary.firstRegularFeasibleFE] = ...
        ResolveFirstPositive(Trace,'regularFeasibleCount');
    [Summary.firstExternalArchiveGeneration,Summary.firstExternalArchiveFE] = ...
        ResolveFirstPositive(Trace,'externalArchiveCount');
    [Summary.firstFeasibleAnchorGeneration,Summary.firstFeasibleAnchorFE] = ...
        ResolveFirstPositive(Trace,'feasibleAnchorCount');
    [Summary.firstCandidateGeneration,Summary.firstCandidateFE] = ...
        ResolveFirstPositive(Trace,'candidatePoolSize');
    [Summary.firstBoundarySeedGeneration,Summary.firstBoundarySeedFE] = ...
        ResolveFirstPositive(Trace,'boundarySeedCount');
    [Summary.firstBoundaryOffspringGeneration,Summary.firstBoundaryOffspringFE] = ...
        ResolveFirstPositive(Trace,'boundaryOffspringCount');

    Summary.blocker = ResolveBlockingStage(Summary);
end

function Summary = AttachSelectionSummary(Summary,Trace)
    if isempty(Trace)
        return;
    end
    Summary.maxEligibleCandidateCount = ResolveMaxTraceField(Trace,'eligibleCount');
    Summary.maxFiniteRankScoreCount = ResolveMaxTraceField(Trace,'finiteScoreCount');
    Summary.maxValidCandidateCount = ResolveMaxTraceField(Trace,'validCount');
    Summary.maxPositiveParetoCount = ResolveMaxTraceField(Trace,'positiveParetoCount');
    Summary.eligibleCandidateEver = Summary.maxEligibleCandidateCount > 0;
    Summary.validCandidateEver = Summary.maxValidCandidateCount > 0;
    [Summary.firstEligibleCandidateGeneration,Summary.firstEligibleCandidateFE] = ...
        ResolveFirstPositive(Trace,'eligibleCount');
    [Summary.firstValidCandidateGeneration,Summary.firstValidCandidateFE] = ...
        ResolveFirstPositive(Trace,'validCount');
    Summary.selectionBlocker = ResolveSelectionBlockingStage(Summary,Trace);
end

function Summary = AttachBridgeSummary(Summary,Trace)
    if isempty(Trace)
        return;
    end
    Summary.maxSharedSectorCount = ResolveMaxTraceField(Trace,'sharedSectorCount');
    Summary.maxActiveSectorCount = ResolveMaxTraceField(Trace,'activeSectorCount');
    Summary.maxBridgeMargin = ResolveMaxTraceField(Trace,'maxActivationMargin');
    Summary.minBridgeMargin = ResolveMinTraceField(Trace,'minActivationMargin');
    Summary.sharedSectorEver = Summary.maxSharedSectorCount > 0;
    Summary.activeSectorEver = Summary.maxActiveSectorCount > 0;
    [Summary.firstSharedSectorGeneration,Summary.firstSharedSectorFE] = ...
        ResolveFirstPositive(Trace,'sharedSectorCount');
    [Summary.firstActiveSectorGeneration,Summary.firstActiveSectorFE] = ...
        ResolveFirstPositive(Trace,'activeSectorCount');
    Summary.bridgeBlocker = ResolveBridgeBlockingStage(Summary,Trace);
end

function Blocker = ResolveBridgeBlockingStage(Summary,Trace)
    if isempty(Trace)
        Blocker = 'no_bridge_trace';
    elseif ~Summary.feasibleAnchorEver
        Blocker = 'no_feasible_anchor';
    elseif ResolveMaxTraceField(Trace,'infeasibleHelperCount') == 0
        Blocker = 'no_infeasible_helper';
    elseif Summary.maxSharedSectorCount == 0
        Blocker = 'no_shared_sector';
    elseif Summary.maxActiveSectorCount == 0
        Blocker = 'activation_gap_not_met';
    else
        Blocker = 'generated';
    end
end

function Blocker = ResolveSelectionBlockingStage(Summary,Trace)
    if isempty(Trace)
        Blocker = 'no_selection_trace';
    elseif ~Summary.candidatePoolEver
        Blocker = 'no_candidate_pool';
    elseif Summary.maxEligibleCandidateCount == 0
        Blocker = 'all_ineligible';
    elseif Summary.maxFiniteRankScoreCount == 0
        Blocker = 'all_rankscore_nonfinite';
    elseif Summary.maxValidCandidateCount == 0
        Blocker = 'no_valid_candidate';
    elseif Summary.maxBoundarySeedCount == 0
        Blocker = 'selection_empty';
    else
        Blocker = 'selected';
    end
end

function Value = ResolveMaxTraceField(Trace,Field)
    if isempty(Trace)
        Value = 0;
        return;
    end
    Data = arrayfun(@(S) FieldOrDefault(S,Field,0),Trace);
    Data = Data(isfinite(Data));
    if isempty(Data)
        Value = 0;
    else
        Value = max(Data);
    end
end

function Value = ResolveMinTraceField(Trace,Field)
    if isempty(Trace)
        Value = NaN;
        return;
    end
    Data = arrayfun(@(S) FieldOrDefault(S,Field,NaN),Trace);
    Data = Data(isfinite(Data));
    if isempty(Data)
        Value = NaN;
    else
        Value = min(Data);
    end
end

function [Generation,FE] = ResolveFirstPositive(Trace,Field)
    Generation = NaN;
    FE = NaN;
    if isempty(Trace)
        return;
    end
    Values = arrayfun(@(S) FieldOrDefault(S,Field,0),Trace);
    Index = find(Values > 0,1,'first');
    if isempty(Index)
        return;
    end
    Generation = FieldOrDefault(Trace(Index),'generation',NaN);
    FE = FieldOrDefault(Trace(Index),'FE',NaN);
end

function Blocker = ResolveBlockingStage(Summary)
    if ~Summary.regularFeasibleEver
        Blocker = 'no_regular_feasible';
    elseif ~Summary.externalArchiveEver
        Blocker = 'no_external_archive';
    elseif ~Summary.feasibleAnchorEver
        Blocker = 'no_feasible_anchor';
    elseif ~Summary.candidatePoolEver
        Blocker = 'no_candidate_pool';
    elseif ~Summary.boundarySeedEver
        Blocker = 'no_boundary_seed';
    elseif ~Summary.boundaryOffspringEver
        Blocker = 'no_boundary_offspring';
    else
        Blocker = 'boundary_started';
    end
end

function Summary = SummarizeProblemRuns(RunResults,ProblemNames,VariantName)
    Summary = repmat(InitActivationProblemSummary(),numel(ProblemNames),1);
    for i = 1 : numel(ProblemNames)
        ProblemName = ProblemNames{i};
        Mask = strcmp({RunResults.problem},ProblemName);
        Summary(i,1) = SummarizeProblemSubset(RunResults(Mask),ProblemName,VariantName);
    end
end

function Summary = SummarizeOverallRuns(RunResults,VariantName)
    Summary = SummarizeProblemSubset(RunResults,'ALL',VariantName);
end

function Row = InitActivationProblemSummary()
    Row = struct( ...
        'problem','', ...
        'variant','', ...
        'runs',0, ...
        'regularFeasibleRuns',0, ...
        'externalArchiveRuns',0, ...
        'feasibleAnchorRuns',0, ...
        'candidatePoolRuns',0, ...
        'sharedSectorRuns',0, ...
        'activeSectorRuns',0, ...
        'eligibleCandidateRuns',0, ...
        'validCandidateRuns',0, ...
        'boundarySeedRuns',0, ...
        'boundaryOffspringRuns',0, ...
        'medianFirstRegularFeasibleGeneration',NaN, ...
        'medianFirstFeasibleAnchorGeneration',NaN, ...
        'medianFirstCandidateGeneration',NaN, ...
        'medianFirstSharedSectorGeneration',NaN, ...
        'medianFirstActiveSectorGeneration',NaN, ...
        'medianFirstEligibleCandidateGeneration',NaN, ...
        'medianFirstValidCandidateGeneration',NaN, ...
        'medianFirstBoundarySeedGeneration',NaN, ...
        'medianFirstBoundaryOffspringGeneration',NaN, ...
        'dominantBridgeBlocker','', ...
        'dominantSelectionBlocker','', ...
        'dominantBlocker','');
end

function Summary = SummarizeProblemSubset(RunResults,ProblemName,VariantName)
    Summary = InitActivationProblemSummary();
    Summary.problem = ProblemName;
    Summary.variant = VariantName;
    Summary.runs = numel(RunResults);
    if isempty(RunResults)
        Summary.dominantBlocker = 'no_runs';
        return;
    end

    S = [RunResults.summary];
    Summary.regularFeasibleRuns = sum([S.regularFeasibleEver]);
    Summary.externalArchiveRuns = sum([S.externalArchiveEver]);
    Summary.feasibleAnchorRuns = sum([S.feasibleAnchorEver]);
    Summary.candidatePoolRuns = sum([S.candidatePoolEver]);
    Summary.sharedSectorRuns = sum([S.sharedSectorEver]);
    Summary.activeSectorRuns = sum([S.activeSectorEver]);
    Summary.eligibleCandidateRuns = sum([S.eligibleCandidateEver]);
    Summary.validCandidateRuns = sum([S.validCandidateEver]);
    Summary.boundarySeedRuns = sum([S.boundarySeedEver]);
    Summary.boundaryOffspringRuns = sum([S.boundaryOffspringEver]);
    Summary.medianFirstRegularFeasibleGeneration = MedianFinite([S.firstRegularFeasibleGeneration]);
    Summary.medianFirstFeasibleAnchorGeneration = MedianFinite([S.firstFeasibleAnchorGeneration]);
    Summary.medianFirstCandidateGeneration = MedianFinite([S.firstCandidateGeneration]);
    Summary.medianFirstSharedSectorGeneration = MedianFinite([S.firstSharedSectorGeneration]);
    Summary.medianFirstActiveSectorGeneration = MedianFinite([S.firstActiveSectorGeneration]);
    Summary.medianFirstEligibleCandidateGeneration = MedianFinite([S.firstEligibleCandidateGeneration]);
    Summary.medianFirstValidCandidateGeneration = MedianFinite([S.firstValidCandidateGeneration]);
    Summary.medianFirstBoundarySeedGeneration = MedianFinite([S.firstBoundarySeedGeneration]);
    Summary.medianFirstBoundaryOffspringGeneration = MedianFinite([S.firstBoundaryOffspringGeneration]);
    Summary.dominantBridgeBlocker = ResolveDominantBlocker({S.bridgeBlocker});
    Summary.dominantSelectionBlocker = ResolveDominantBlocker({S.selectionBlocker});
    Summary.dominantBlocker = ResolveDominantBlocker({S.blocker});
end

function Value = MedianFinite(Data)
    Data = Data(isfinite(Data));
    if isempty(Data)
        Value = NaN;
    else
        Value = median(Data);
    end
end

function Blocker = ResolveDominantBlocker(Blockers)
    if isempty(Blockers)
        Blocker = 'no_runs';
        return;
    end
    [Names,~,Index] = unique(Blockers,'stable');
    Count = accumarray(Index(:),1);
    [~,Best] = max(Count);
    Blocker = Names{Best};
end

function Variant = ResolveDiagnosticVariant(Name)
    if nargin < 1 || isempty(Name)
        Name = 'auto_trust';
    end
    if iscell(Name)
        Name = Name{1};
    end
    if isstring(Name)
        Name = char(Name);
    end
    Name = lower(strtrim(Name));
    switch Name
        case 'raw'
            Variant = BuildDiagnosticVariant('raw','raw',1);
        case 'temperature'
            Variant = BuildDiagnosticVariant('temperature','temperature',4);
        case 'beta'
            Variant = BuildDiagnosticVariant('beta','beta',3);
        otherwise
            Variant = BuildDiagnosticVariant('auto_trust','online best-of-(temperature,beta) + trust gate',2);
    end
end

function Variant = BuildDiagnosticVariant(Name,Label,CalMode)
    Variant = struct();
    Variant.name = Name;
    Variant.label = Label;
    Variant.calMode = CalMode;
    % The last parameter keeps section B tracing enabled.
    Variant.parameter = {0.2,2,20,25,0.01,0.4,3,CalMode,1,0.05,1,1,1,1,1};
end

function Value = FieldOrDefault(S,Field,Default)
    if isstruct(S) && isfield(S,Field) && ~isempty(S.(Field))
        Value = S.(Field);
    else
        Value = Default;
    end
end

function Params = NormalizeRunControl(Params)
    Params.ProblemNames = NormalizeProblemNames(Params.ProblemNames);
    if isfield(Params,'RunSeeds') && ~isempty(Params.RunSeeds)
        Params.RunSeeds = reshape(double(Params.RunSeeds),1,[]);
        Params.Runs = numel(Params.RunSeeds);
    else
        Params.Runs = max(1,round(Params.Runs));
        Params.RunSeeds = 1000 + (1:Params.Runs);
    end
end

function ProblemNames = NormalizeProblemNames(ProblemNames)
    if isempty(ProblemNames)
        ProblemNames = {'DASCMOP1_BC'};
        return;
    end
    if ischar(ProblemNames)
        ProblemNames = {ProblemNames};
        return;
    end
    if isstring(ProblemNames)
        ProblemNames = cellstr(ProblemNames(:)');
    end
end

function Params = ParseInputs(Params,varargin)
    if isempty(varargin)
        return;
    end
    if mod(numel(varargin),2) ~= 0
        error('diagnose_PRBCCMO_boundary_activation:InvalidInput', ...
            'Inputs must be name-value pairs.');
    end
    for i = 1 : 2 : numel(varargin)
        Name = char(varargin{i});
        if ~isfield(Params,Name)
            error('diagnose_PRBCCMO_boundary_activation:UnknownOption', ...
                'Unknown option: %s',Name);
        end
        Params.(Name) = varargin{i+1};
    end
end
