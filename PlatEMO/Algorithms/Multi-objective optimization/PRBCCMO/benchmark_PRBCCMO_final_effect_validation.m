function Results = benchmark_PRBCCMO_final_effect_validation(varargin)
% Final-effect benchmark for PRBCCMO on BC problems.
%
% This script closes the third-layer validation requested by fix.md:
%   boundarycore_beta vs no_boundary vs external baselines
%   final HV / IGD+ / Feasible_rate / first-archive-entry FE
%   paired Wilcoxon signed-rank statistics
%
% Optional name-value pairs:
%   'Runs'          : independent runs, default 30
%   'RunSeeds'      : explicit random seeds, default 1001:1000+Runs
%   'Population'    : population size, default 100
%   'MaxFE'         : max function evaluations, default 200000
%   'ProblemNames'  : explicit BC problem list, default all 37 BC problems
%   'UseParallel'   : enable parfor execution, default true
%   'Workers'       : parallel worker count, default 6
%   'SaveSlots'     : saved population checkpoints per run, default 100
%   'PRBCCMOCalibrator' : {'beta','raw'}, default 'beta'
%   'BaselineAlgs'  : external baselines, default {'CCMO','NAEMT2025'}
%   'VariantSpecs'  : optional explicit variant spec struct array
%   'BaseVariant'   : variant used as paired-test reference, default 'boundarycore_beta'
%   'SavePrefix'    : output prefix without suffix, default
%                     'benchmark_PRBCCMO_final_effect_<problemCount>bc_r<runs>'
%   'SaveMat'       : whether to save MAT, default true
%   'Verbose'       : print serial progress, default true

    Params = struct( ...
        'Runs',30, ...
        'RunSeeds',[], ...
        'Population',100, ...
        'MaxFE',200000, ...
        'ProblemNames',{{}}, ...
        'UseParallel',true, ...
        'Workers',6, ...
        'SaveSlots',100, ...
        'PRBCCMOCalibrator','beta', ...
        'BaselineAlgs',{{'CCMO','NAEMT2025'}}, ...
        'VariantSpecs',repmat(struct('id','','label','','algorithm','', ...
            'parameter',[]),0,1), ...
        'BaseVariant','boundarycore_beta', ...
        'SavePrefix','', ...
        'SaveMat',true, ...
        'Verbose',true);
    Params = ParseInputs(Params,varargin{:});
    Params.ProblemNames = NormalizeProblemNames(Params.ProblemNames);
    Params.RunSeeds = NormalizeRunSeeds(Params.RunSeeds,Params.Runs);
    Params.SaveSlots = max(2,round(double(Params.SaveSlots)));
    Params.VariantSpecs = ResolveVariantSpecs(Params);
    Params.SavePrefix = ResolveDefaultSavePrefix(Params.SavePrefix, ...
        numel(Params.ProblemNames),numel(Params.RunSeeds));
    Params.Workers = ResolveParallelWorkers(Params.Workers, ...
        numel(Params.ProblemNames)*numel(Params.RunSeeds)*numel(Params.VariantSpecs));

    ProjectRoot = ResolveProjectRoot();
    EnsureProjectPath(ProjectRoot);
    Tasks = BuildTasks(Params.ProblemNames,Params.RunSeeds,Params.VariantSpecs);
    RunRowsCell = cell(numel(Tasks),1);

    if logical(Params.UseParallel) && numel(Tasks) > 1
        ConfigureParallelPool(Params.Workers);
        parfor t = 1 : numel(Tasks)
            RunRowsCell{t} = RunFinalEffectTask(Tasks(t),Params,ProjectRoot);
        end
    else
        for t = 1 : numel(Tasks)
            RunRowsCell{t} = RunFinalEffectTask(Tasks(t),Params,ProjectRoot);
            if logical(Params.Verbose)
                PrintRunSummary(RunRowsCell{t});
            end
        end
    end

    RunRows = MergeStructCells(RunRowsCell,InitRunRow());
    ProblemSummary = SummarizeScopedRuns(RunRows,'problem',Params.VariantSpecs);
    FamilySummary = SummarizeScopedRuns(RunRows,'family',Params.VariantSpecs);
    PooledSummary = SummarizeScopedRuns(RunRows,'pooled',Params.VariantSpecs);
    PairedSummary = BuildPairedSummary(RunRows,Params.VariantSpecs,Params.BaseVariant);
    VariantManifest = BuildVariantManifest(Params.VariantSpecs);

    Results = struct();
    Results.params = Params;
    Results.variantManifest = VariantManifest;
    Results.runSummary = RunRows;
    Results.problemSummary = ProblemSummary;
    Results.familySummary = FamilySummary;
    Results.pooledSummary = PooledSummary;
    Results.pairedSummary = PairedSummary;

    WriteOutputs(Results,Params.SavePrefix,Params.SaveMat);
end

function Params = ParseInputs(Params,varargin)
    if mod(numel(varargin),2) ~= 0
        error('benchmark_PRBCCMO_final_effect_validation:InvalidInput', ...
            'Inputs must be name-value pairs.');
    end
    for i = 1 : 2 : numel(varargin)
        Name = varargin{i};
        if ~(ischar(Name) || (isstring(Name) && isscalar(Name)))
            error('benchmark_PRBCCMO_final_effect_validation:InvalidInputName', ...
                'Input names must be character vectors or scalar strings.');
        end
        Name = char(Name);
        if ~isfield(Params,Name)
            error('benchmark_PRBCCMO_final_effect_validation:UnknownOption', ...
                'Unknown option ''%s''.',Name);
        end
        Params.(Name) = varargin{i+1};
    end
end

function ProblemNames = NormalizeProblemNames(ProblemNames)
    if isempty(ProblemNames)
        ProblemNames = ResolveAllBCProblems();
        return;
    end
    if ischar(ProblemNames) || (isstring(ProblemNames) && isscalar(ProblemNames))
        ProblemNames = {char(ProblemNames)};
    end
    ProblemNames = cellfun(@char,ProblemNames(:)','UniformOutput',false);
end

function ProblemNames = ResolveAllBCProblems()
    ProblemNames = [ ...
        arrayfun(@(i)sprintf('DASCMOP%d_BC',i),1:9,'UniformOutput',false), ...
        arrayfun(@(i)sprintf('LIRCMOP%d_BC',i),1:14,'UniformOutput',false), ...
        arrayfun(@(i)sprintf('MW%d_BC',i),1:14,'UniformOutput',false)];
end

function RunSeeds = NormalizeRunSeeds(RunSeeds,Runs)
    if isempty(RunSeeds)
        RunSeeds = 1000 + (1:Runs);
        return;
    end
    RunSeeds = round(double(RunSeeds(:)'));
    RunSeeds = RunSeeds(isfinite(RunSeeds));
    if isempty(RunSeeds)
        error('benchmark_PRBCCMO_final_effect_validation:InvalidRunSeeds', ...
            'RunSeeds must contain at least one finite integer.');
    end
end

function VariantSpecs = ResolveVariantSpecs(Params)
    if ~isempty(Params.VariantSpecs)
        VariantSpecs = NormalizeVariantSpecs(Params.VariantSpecs);
        return;
    end

    VariantSpecs = repmat(struct( ...
        'id','', ...
        'label','', ...
        'algorithm','', ...
        'parameter',[]),0,1);
    VariantSpecs(end+1) = struct( ...
        'id','boundarycore_beta', ...
        'label','PRBCCMO BoundaryCore mainline', ...
        'algorithm','PRBCCMO', ...
        'parameter',BuildPRBCCMOParameter(Params.PRBCCMOCalibrator,struct()));
    VariantSpecs(end+1) = struct( ...
        'id','no_boundary', ...
        'label','PRBCCMO without boundary budget', ...
        'algorithm','PRBCCMO', ...
        'parameter',BuildPRBCCMOParameter(Params.PRBCCMOCalibrator,struct('bRho',0)));

    BaselineAlgs = Params.BaselineAlgs;
    if ischar(BaselineAlgs) || (isstring(BaselineAlgs) && isscalar(BaselineAlgs))
        BaselineAlgs = {char(BaselineAlgs)};
    end
    for i = 1 : numel(BaselineAlgs)
        VariantSpecs(end+1) = ResolveBaselineVariant(BaselineAlgs{i}); %#ok<AGROW>
    end
end

function VariantSpecs = NormalizeVariantSpecs(VariantSpecs)
    Required = {'id','label','algorithm','parameter'};
    for i = 1 : numel(VariantSpecs)
        for j = 1 : numel(Required)
            if ~isfield(VariantSpecs(i),Required{j})
                error('benchmark_PRBCCMO_final_effect_validation:InvalidVariantSpec', ...
                    'VariantSpecs(%d) is missing field ''%s''.',i,Required{j});
            end
        end
    end
end

function Parameter = BuildPRBCCMOParameter(CalibratorName,Override)
    if nargin < 2 || ~isstruct(Override)
        Override = struct();
    end
    Parameter = Override;
    Parameter.calibratorCandidates = {lower(strtrim(char(CalibratorName)))};
end

function Variant = ResolveBaselineVariant(Name)
    Canonical = lower(strtrim(char(Name)));
    switch Canonical
        case 'ccmo'
            Variant = struct( ...
                'id','ccmo', ...
                'label','CCMO baseline', ...
                'algorithm','CCMO', ...
                'parameter',[]);
        case {'naemt2025','na-emt-2025','na-emt'}
            Variant = struct( ...
                'id','naemt2025', ...
                'label','NA-EMT-2025 baseline', ...
                'algorithm','NAEMT2025', ...
                'parameter',{{0.9,0.5,1000}});
        otherwise
            Variant = struct( ...
                'id',CanonicalizeVariantId(Name), ...
                'label',char(Name), ...
                'algorithm',char(Name), ...
                'parameter',[]);
    end
end

function Id = CanonicalizeVariantId(Name)
    Id = lower(regexprep(char(Name),'[^a-zA-Z0-9]+','_'));
    Id = regexprep(Id,'^_+|_+$','');
    if isempty(Id)
        Id = 'variant';
    end
end

function SavePrefix = ResolveDefaultSavePrefix(SavePrefix,ProblemCount,RunCount)
    if nargin >= 1 && ~isempty(SavePrefix)
        SavePrefix = char(SavePrefix);
        return;
    end
    SavePrefix = sprintf('benchmark_PRBCCMO_final_effect_%dbc_r%d', ...
        max(0,ProblemCount),max(0,RunCount));
end

function Workers = ResolveParallelWorkers(Workers,TaskCount)
    Workers = round(double(Workers));
    if ~isfinite(Workers) || Workers < 1
        Workers = 1;
    end
    Workers = min(Workers,max(1,TaskCount));
end

function ConfigureParallelPool(Workers)
    Pool = gcp('nocreate');
    if isempty(Pool)
        parpool('local',Workers);
    elseif Pool.NumWorkers ~= Workers
        delete(Pool);
        parpool('local',Workers);
    end
end

function ProjectRoot = ResolveProjectRoot()
    ProjectRoot = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
end

function EnsureProjectPath(ProjectRoot)
    persistent AddedRoot;
    if ~isempty(AddedRoot) && strcmp(AddedRoot,ProjectRoot)
        return;
    end
    addpath(ProjectRoot);
    addpath(genpath(fullfile(ProjectRoot,'Algorithms')));
    addpath(genpath(fullfile(ProjectRoot,'Problems')));
    addpath(genpath(fullfile(ProjectRoot,'Metrics')));
    AddedRoot = ProjectRoot;
end

function Tasks = BuildTasks(ProblemNames,RunSeeds,VariantSpecs)
    TaskCount = numel(ProblemNames) * numel(RunSeeds) * numel(VariantSpecs);
    Template = struct('problem','','family','','run',0,'seed',0,'variant',VariantSpecs(1));
    Tasks = repmat(Template,TaskCount,1);
    Row = 0;
    for p = 1 : numel(ProblemNames)
        ProblemName = ProblemNames{p};
        FamilyName = ResolveProblemFamily(ProblemName);
        for r = 1 : numel(RunSeeds)
            for v = 1 : numel(VariantSpecs)
                Row = Row + 1;
                Tasks(Row).problem = ProblemName;
                Tasks(Row).family = FamilyName;
                Tasks(Row).run = r;
                Tasks(Row).seed = RunSeeds(r);
                Tasks(Row).variant = VariantSpecs(v);
            end
        end
    end
end

function FamilyName = ResolveProblemFamily(ProblemName)
    if startsWith(ProblemName,'DASCMOP')
        FamilyName = 'DASCMOP_BC';
    elseif startsWith(ProblemName,'LIRCMOP')
        FamilyName = 'LIRCMOP_BC';
    elseif startsWith(ProblemName,'MW')
        FamilyName = 'MW_BC';
    else
        FamilyName = 'BC';
    end
end

function RunRow = RunFinalEffectTask(Task,Params,ProjectRoot)
    EnsureProjectPath(ProjectRoot);
    rng(Task.seed,'twister');
    Problem = feval(Task.problem,'N',Params.Population,'maxFE',Params.MaxFE);
    Algorithm = InstantiateAlgorithm(Task.variant,Params.SaveSlots);
    Algorithm.Solve(Problem);
    RunRow = SummarizeSingleRun(Algorithm,Problem,Task);
end

function Algorithm = InstantiateAlgorithm(Variant,SaveSlots)
    if isempty(Variant.parameter)
        Algorithm = feval(Variant.algorithm,'save',SaveSlots,'outputFcn',@NoOutput);
    else
        Algorithm = feval(Variant.algorithm,'parameter',Variant.parameter, ...
            'save',SaveSlots,'outputFcn',@NoOutput);
    end
end

function NoOutput(~,~)
end

function Row = SummarizeSingleRun(Algorithm,Problem,Task)
    Row = InitRunRow();
    Row.problem = Task.problem;
    Row.family = Task.family;
    Row.variant = Task.variant.id;
    Row.variantLabel = Task.variant.label;
    Row.algorithm = Task.variant.algorithm;
    Row.run = Task.run;
    Row.seed = Task.seed;

    FEHistory = ResolveResultFEHistory(Algorithm.result);
    Row.snapshotCount = numel(FEHistory);
    Row.finalFE = SafeLastFinite(FEHistory);

    HVHistory = ResolveMetricHistory(Algorithm,'HV',Row.snapshotCount);
    IGDpHistory = ResolveMetricHistory(Algorithm,'IGDp',Row.snapshotCount);
    FeasibleRateHistory = ResolveMetricHistory(Algorithm,'Feasible_rate',Row.snapshotCount);

    Row.finalHV = SafeLastFinite(HVHistory);
    Row.finalIGDp = SafeLastFinite(IGDpHistory);
    Row.finalFeasibleRate = SafeLastFinite(FeasibleRateHistory);
    Row.aucHV = ComputeAuc(FEHistory,HVHistory,Problem.maxFE);
    Row.firstFeasibleHitFE = ResolveFirstPositiveFE(FEHistory,FeasibleRateHistory);
    [Row.firstArchiveEntryFE,Row.firstArchiveEntrySource,Row.archiveEventCount] = ...
        ResolveFirstArchiveEntryFE(Algorithm.metric,Row.firstFeasibleHitFE);
    Row.boundaryGainTotal = ResolveBoundaryGainTotal(Algorithm.metric);
    Row.runtime = FieldOrDefault(Algorithm.metric,'runtime',NaN);
end

function FEHistory = ResolveResultFEHistory(ResultCell)
    FEHistory = zeros(0,1);
    if isempty(ResultCell)
        return;
    end
    FEHistory = cell2mat(ResultCell(:,1));
    FEHistory = double(FEHistory(:));
end

function Values = ResolveMetricHistory(Algorithm,MetricName,Count)
    Values = nan(max(Count,0),1);
    if Count <= 0
        return;
    end
    try
        Data = Algorithm.CalMetric(MetricName);
        Data = double(Data(:));
        Values(1:min(numel(Data),Count)) = Data(1:min(numel(Data),Count));
    catch
    end
end

function Value = SafeLastFinite(Data)
    Value = NaN;
    if isempty(Data)
        return;
    end
    Data = double(Data(:));
    Index = find(isfinite(Data),1,'last');
    if ~isempty(Index)
        Value = Data(Index);
    end
end

function Value = ComputeAuc(FEHistory,MetricHistory,MaxFE)
    Value = NaN;
    if isempty(FEHistory) || isempty(MetricHistory) || ~isfinite(MaxFE) || MaxFE <= 0
        return;
    end
    FEHistory = double(FEHistory(:));
    MetricHistory = double(MetricHistory(:));
    Mask = isfinite(FEHistory) & isfinite(MetricHistory);
    FEHistory = FEHistory(Mask);
    MetricHistory = MetricHistory(Mask);
    if isempty(FEHistory)
        return;
    end
    [FEHistory,UniqueIdx] = unique(FEHistory,'stable');
    MetricHistory = MetricHistory(UniqueIdx);
    if FEHistory(1) > 0
        FEHistory = [0; FEHistory];
        MetricHistory = [0; MetricHistory];
    end
    if isscalar(FEHistory)
        Value = MetricHistory(1) * FEHistory(1) / MaxFE;
        return;
    end
    Value = trapz(FEHistory,MetricHistory) / MaxFE;
end

function Value = ResolveFirstPositiveFE(FEHistory,MetricHistory)
    Value = NaN;
    if isempty(FEHistory) || isempty(MetricHistory)
        return;
    end
    Mask = isfinite(FEHistory) & isfinite(MetricHistory) & MetricHistory > 0;
    Index = find(Mask,1,'first');
    if ~isempty(Index)
        Value = FEHistory(Index);
    end
end

function [Value,Source,Count] = ResolveFirstArchiveEntryFE(Metric,FallbackFE)
    Value = NaN;
    Source = 'not_available';
    SectionB = FieldOrDefault(Metric,'sectionB',struct());
    ArchiveEvent = FieldOrDefault(SectionB,'archiveEvent',repmat(struct('FE',NaN),0,1));
    Count = numel(ArchiveEvent);
    if ~isempty(ArchiveEvent)
        ArchiveFE = [ArchiveEvent.FE];
        ArchiveFE = ArchiveFE(isfinite(ArchiveFE));
        if ~isempty(ArchiveFE)
            Value = ArchiveFE(1);
            Source = 'sectionB.archiveEvent';
            return;
        end
    end
    if nargin >= 2 && isfinite(FallbackFE)
        Value = FallbackFE;
        Source = 'population_feasible_hit_proxy';
    end
end

function Value = ResolveBoundaryGainTotal(Metric)
    SectionB = FieldOrDefault(Metric,'sectionB',struct());
    Value = FieldOrDefault(SectionB,'totalBoundaryGain',NaN);
    if isfinite(Value)
        return;
    end
    BoundaryGainTrace = FieldOrDefault(SectionB,'boundaryGainTrace',repmat(struct('boundaryGain',0),0,1));
    if isempty(BoundaryGainTrace)
        return;
    end
    Value = sum([BoundaryGainTrace.boundaryGain]);
end

function Row = InitRunRow()
    Row = struct( ...
        'problem','', ...
        'family','', ...
        'variant','', ...
        'variantLabel','', ...
        'algorithm','', ...
        'run',0, ...
        'seed',0, ...
        'snapshotCount',0, ...
        'finalFE',NaN, ...
        'finalHV',NaN, ...
        'finalIGDp',NaN, ...
        'finalFeasibleRate',NaN, ...
        'aucHV',NaN, ...
        'firstFeasibleHitFE',NaN, ...
        'firstArchiveEntryFE',NaN, ...
        'firstArchiveEntrySource','', ...
        'boundaryGainTotal',NaN, ...
        'archiveEventCount',0, ...
        'runtime',NaN);
end

function Rows = MergeStructCells(CellRows,Template)
    Valid = ~cellfun(@isempty,CellRows);
    if ~any(Valid)
        Rows = repmat(Template,0,1);
        return;
    end
    Rows = vertcat(CellRows{Valid});
end

function Rows = SummarizeScopedRuns(RunRows,Scope,VariantSpecs)
    Template = InitSummaryRow();
    if isempty(RunRows)
        Rows = repmat(Template,0,1);
        return;
    end

    switch Scope
        case 'problem'
            GroupNames = unique({RunRows.problem},'stable');
        case 'family'
            GroupNames = unique({RunRows.family},'stable');
        case 'pooled'
            GroupNames = {ResolvePooledSummaryName(RunRows)};
        otherwise
            error('benchmark_PRBCCMO_final_effect_validation:InvalidScope', ...
                'Unsupported summary scope ''%s''.',Scope);
    end

    Rows = repmat(Template,0,1);
    for g = 1 : numel(GroupNames)
        GroupName = GroupNames{g};
        for v = 1 : numel(VariantSpecs)
            VariantId = VariantSpecs(v).id;
            switch Scope
                case 'problem'
                    Mask = strcmp({RunRows.problem}',GroupName) & strcmp({RunRows.variant}',VariantId);
                case 'family'
                    Mask = strcmp({RunRows.family}',GroupName) & strcmp({RunRows.variant}',VariantId);
                otherwise
                    Mask = strcmp({RunRows.variant}',VariantId);
            end
            if ~any(Mask)
                continue;
            end
            Rows(end+1,1) = BuildSummaryRow(RunRows(Mask),Scope,GroupName,VariantSpecs(v)); %#ok<AGROW>
        end
    end
end

function Row = BuildSummaryRow(RunRows,Scope,Name,Variant)
    Row = InitSummaryRow();
    Row.scope = Scope;
    Row.name = Name;
    Row.family = ResolveSummaryFamily(RunRows,Scope);
    Row.variant = Variant.id;
    Row.variantLabel = Variant.label;
    Row.algorithm = Variant.algorithm;
    Row.problemCount = numel(unique({RunRows.problem}));
    Row.runCount = numel(RunRows);
    Row.finalHV_mean = MeanOrNaN([RunRows.finalHV]);
    Row.finalHV_median = MedianOrNaN([RunRows.finalHV]);
    Row.finalIGDp_mean = MeanOrNaN([RunRows.finalIGDp]);
    Row.finalIGDp_median = MedianOrNaN([RunRows.finalIGDp]);
    Row.finalFeasibleRate_mean = MeanOrNaN([RunRows.finalFeasibleRate]);
    Row.finalFeasibleRate_median = MedianOrNaN([RunRows.finalFeasibleRate]);
    Row.aucHV_mean = MeanOrNaN([RunRows.aucHV]);
    Row.aucHV_median = MedianOrNaN([RunRows.aucHV]);
    Row.firstFeasibleHitFE_mean = MeanOrNaN([RunRows.firstFeasibleHitFE]);
    Row.firstFeasibleHitFE_median = MedianOrNaN([RunRows.firstFeasibleHitFE]);
    Row.firstArchiveEntryFE_mean = MeanOrNaN([RunRows.firstArchiveEntryFE]);
    Row.firstArchiveEntryFE_median = MedianOrNaN([RunRows.firstArchiveEntryFE]);
    Row.boundaryGainTotal_mean = MeanOrNaN([RunRows.boundaryGainTotal]);
    Row.archiveEventCount_mean = MeanOrNaN([RunRows.archiveEventCount]);
    Row.runtime_mean = MeanOrNaN([RunRows.runtime]);
    Row.runtime_median = MedianOrNaN([RunRows.runtime]);
end

function Row = InitSummaryRow()
    Row = struct( ...
        'scope','', ...
        'name','', ...
        'family','', ...
        'variant','', ...
        'variantLabel','', ...
        'algorithm','', ...
        'problemCount',0, ...
        'runCount',0, ...
        'finalHV_mean',NaN, ...
        'finalHV_median',NaN, ...
        'finalIGDp_mean',NaN, ...
        'finalIGDp_median',NaN, ...
        'finalFeasibleRate_mean',NaN, ...
        'finalFeasibleRate_median',NaN, ...
        'aucHV_mean',NaN, ...
        'aucHV_median',NaN, ...
        'firstFeasibleHitFE_mean',NaN, ...
        'firstFeasibleHitFE_median',NaN, ...
        'firstArchiveEntryFE_mean',NaN, ...
        'firstArchiveEntryFE_median',NaN, ...
        'boundaryGainTotal_mean',NaN, ...
        'archiveEventCount_mean',NaN, ...
        'runtime_mean',NaN, ...
        'runtime_median',NaN);
end

function Family = ResolveSummaryFamily(RunRows,Scope)
    if strcmp(Scope,'problem') && ~isempty(RunRows)
        Family = RunRows(1).family;
    elseif strcmp(Scope,'family') && ~isempty(RunRows)
        Family = RunRows(1).family;
    else
        Family = 'ALL_BC';
    end
end

function Name = ResolvePooledSummaryName(RunRows)
    Name = sprintf('ALL_%d_BC',numel(unique({RunRows.problem})));
end

function Rows = BuildPairedSummary(RunRows,VariantSpecs,BaseVariant)
    Template = InitPairedRow();
    if isempty(RunRows)
        Rows = repmat(Template,0,1);
        return;
    end

    BaseMask = strcmp({RunRows.variant}',BaseVariant);
    if ~any(BaseMask)
        Rows = repmat(Template,0,1);
        return;
    end

    Metrics = { ...
        'finalHV','higher_better'; ...
        'finalIGDp','lower_better'; ...
        'finalFeasibleRate','higher_better'; ...
        'firstArchiveEntryFE','lower_better'; ...
        'aucHV','higher_better'};
    ScopeSpecs = [ ...
        struct('scope','problem','names',{unique({RunRows.problem},'stable')}), ...
        struct('scope','pooled','names',{{ResolvePooledSummaryName(RunRows)}})];

    Rows = repmat(Template,0,1);
    for s = 1 : numel(ScopeSpecs)
        ScopeName = ScopeSpecs(s).scope;
        GroupNames = ScopeSpecs(s).names;
        for g = 1 : numel(GroupNames)
            GroupName = GroupNames{g};
            for m = 1 : size(Metrics,1)
                MetricName = Metrics{m,1};
                Direction = Metrics{m,2};
                GroupRows = repmat(Template,0,1);
                for v = 1 : numel(VariantSpecs)
                    CompareVariant = VariantSpecs(v).id;
                    if strcmp(CompareVariant,BaseVariant)
                        continue;
                    end
                    [BaseData,CompareData,FamilyName] = ResolvePairedSamples( ...
                        RunRows,ScopeName,GroupName,BaseVariant,CompareVariant,MetricName);
                    if isempty(BaseData)
                        continue;
                    end
                    Row = InitPairedRow();
                    Row.scope = ScopeName;
                    Row.name = GroupName;
                    Row.family = FamilyName;
                    Row.metric = MetricName;
                    Row.baseVariant = BaseVariant;
                    Row.compareVariant = CompareVariant;
                    Row.betterDirection = Direction;
                    Row.pairCount = numel(BaseData);
                    Row.baseMedian = MedianOrNaN(BaseData);
                    Row.compareMedian = MedianOrNaN(CompareData);
                    Row.medianDiff = MedianOrNaN(CompareData - BaseData);
                    Row.signrankP = SafeSignrank(BaseData,CompareData);
                    GroupRows(end+1,1) = Row; %#ok<AGROW>
                end
                if isempty(GroupRows)
                    continue;
                end
                Adjusted = HolmAdjust([GroupRows.signrankP]);
                for i = 1 : numel(GroupRows)
                    GroupRows(i).holmAdjustedP = Adjusted(i);
                end
                Rows = [Rows; GroupRows]; %#ok<AGROW>
            end
        end
    end
end

function [BaseData,CompareData,FamilyName] = ResolvePairedSamples( ...
    RunRows,ScopeName,GroupName,BaseVariant,CompareVariant,MetricName)
    FamilyName = 'ALL_BC';
    switch ScopeName
        case 'problem'
            ScopeMask = strcmp({RunRows.problem}',GroupName);
        case 'pooled'
            ScopeMask = true(numel(RunRows),1);
        otherwise
            ScopeMask = false(numel(RunRows),1);
    end
    BaseRows = RunRows(ScopeMask & strcmp({RunRows.variant}',BaseVariant));
    CompareRows = RunRows(ScopeMask & strcmp({RunRows.variant}',CompareVariant));
    if strcmp(ScopeName,'problem') && ~isempty(BaseRows)
        FamilyName = BaseRows(1).family;
    end

    BaseData = zeros(0,1);
    CompareData = zeros(0,1);
    if isempty(BaseRows) || isempty(CompareRows)
        return;
    end

    BaseKeys = BuildPairKeys(BaseRows);
    CompareKeys = BuildPairKeys(CompareRows);
    [CommonKeys,BaseIdx,CompareIdx] = intersect(BaseKeys,CompareKeys,'stable');
    if isempty(CommonKeys)
        return;
    end
    BaseValues = [BaseRows(BaseIdx).(MetricName)];
    CompareValues = [CompareRows(CompareIdx).(MetricName)];
    Valid = isfinite(BaseValues) & isfinite(CompareValues);
    BaseData = double(BaseValues(Valid))';
    CompareData = double(CompareValues(Valid))';
end

function Keys = BuildPairKeys(Rows)
    Keys = arrayfun(@(R)sprintf('%s|%d|%d',R.problem,R.run,R.seed),Rows,'UniformOutput',false);
end

function Row = InitPairedRow()
    Row = struct( ...
        'scope','', ...
        'name','', ...
        'family','', ...
        'metric','', ...
        'baseVariant','', ...
        'compareVariant','', ...
        'betterDirection','', ...
        'pairCount',0, ...
        'baseMedian',NaN, ...
        'compareMedian',NaN, ...
        'medianDiff',NaN, ...
        'signrankP',NaN, ...
        'holmAdjustedP',NaN);
end

function Value = SafeSignrank(A,B)
    Value = NaN;
    if isempty(A) || isempty(B) || numel(A) ~= numel(B)
        return;
    end
    if exist('signrank','file') ~= 2
        return;
    end
    try
        Value = signrank(A,B);
    catch
        Value = NaN;
    end
end

function Adjusted = HolmAdjust(PValues)
    Adjusted = nan(size(PValues));
    if isempty(PValues)
        return;
    end
    Valid = isfinite(PValues);
    if ~any(Valid)
        return;
    end
    ValidP = PValues(Valid);
    [SortedP,Order] = sort(ValidP(:));
    Count = numel(SortedP);
    AdjustedSorted = zeros(Count,1);
    RunningMax = 0;
    for i = 1 : Count
        Candidate = (Count - i + 1) * SortedP(i);
        RunningMax = max(RunningMax,Candidate);
        AdjustedSorted(i) = min(RunningMax,1);
    end
    Temp = nan(size(ValidP(:)));
    Temp(Order) = AdjustedSorted;
    Adjusted(Valid) = Temp;
end

function Manifest = BuildVariantManifest(VariantSpecs)
    Count = numel(VariantSpecs);
    Manifest = table('Size',[Count 4], ...
        'VariableTypes',{'string','string','string','string'}, ...
        'VariableNames',{'variant','label','algorithm','parameter'});
    for i = 1 : Count
        Manifest.variant(i) = string(VariantSpecs(i).id);
        Manifest.label(i) = string(VariantSpecs(i).label);
        Manifest.algorithm(i) = string(VariantSpecs(i).algorithm);
        Manifest.parameter(i) = string(StructLikeToJson(VariantSpecs(i).parameter));
    end
end

function Text = StructLikeToJson(Value)
    if isempty(Value)
        Text = '[]';
        return;
    end
    Text = jsonencode(Value);
end

function WriteOutputs(Results,SavePrefix,SaveMat)
    if nargin < 2 || isempty(SavePrefix)
        return;
    end
    if nargin < 3 || isempty(SaveMat)
        SaveMat = true;
    end
    EnsureOutputDirectory(SavePrefix);
    if logical(SaveMat)
        save([SavePrefix,'.mat'],'Results','-v7.3');
    end
    writetable(Results.variantManifest,[SavePrefix,'_variant_manifest.csv']);
    writetable(struct2table(Results.runSummary,'AsArray',true),[SavePrefix,'_run_summary.csv']);
    writetable(struct2table(Results.problemSummary,'AsArray',true),[SavePrefix,'_problem_summary.csv']);
    writetable(struct2table(Results.familySummary,'AsArray',true),[SavePrefix,'_family_summary.csv']);
    writetable(struct2table(Results.pooledSummary,'AsArray',true),[SavePrefix,'_pooled_summary.csv']);
    writetable(struct2table(Results.pairedSummary,'AsArray',true),[SavePrefix,'_paired_summary.csv']);
end

function EnsureOutputDirectory(SavePrefix)
    OutDir = fileparts(SavePrefix);
    if isempty(OutDir) || exist(OutDir,'dir') == 7
        return;
    end
    mkdir(OutDir);
end

function PrintRunSummary(Row)
    fprintf(['[FinalEffect] %s %s run=%d seed=%d | HV=%.4e IGD+=%.4e Feasible=%.4f | ' ...
        'FHT=%.0f ArchiveFE=%.0f Runtime=%.2fs\n'], ...
        Row.problem,Row.variant,Row.run,Row.seed, ...
        Row.finalHV,Row.finalIGDp,Row.finalFeasibleRate, ...
        Row.firstFeasibleHitFE,Row.firstArchiveEntryFE,Row.runtime);
end

function Value = MeanOrNaN(Data)
    if isempty(Data)
        Value = NaN;
        return;
    end
    Value = mean(double(Data(:)),'omitnan');
    if isempty(Value) || ~isfinite(Value)
        Value = NaN;
    end
end

function Value = MedianOrNaN(Data)
    if isempty(Data)
        Value = NaN;
        return;
    end
    Value = median(double(Data(:)),'omitnan');
    if isempty(Value) || ~isfinite(Value)
        Value = NaN;
    end
end

function Value = FieldOrDefault(Data,Field,Default)
    Value = Default;
    if isstruct(Data) && isfield(Data,Field) && ~isempty(Data.(Field))
        Value = Data.(Field);
    end
end
