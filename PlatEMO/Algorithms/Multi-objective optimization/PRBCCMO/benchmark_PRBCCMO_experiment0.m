function Results = benchmark_PRBCCMO_experiment0(varargin)
% Formal experiment 0: calibration and trust-gate audit on DASCMOP-BC.
%
% Optional name-value pairs:
%   'Runs'       : number of independent runs, default 15
%   'RunSeeds'   : explicit seed list, default 1001:1015
%   'Population' : population size N, default 100
%   'MaxFE'      : maximum function evaluations, default 200000
%   'ProblemNames': cell array of problem names, default DASCMOP1_BC-9_BC
%   'UseParallel': whether to run tasks with parfor, default false
%   'Workers'    : number of parallel workers (0 = auto), default 0
%   'SavePath'   : MAT file path, default benchmark_PRBCCMO_experiment0.mat
%   'SaveSummaryOnly' : save only compact summaries to MAT, default true

    Params = struct( ...
        'Runs',15, ...
        'RunSeeds',[], ...
        'Population',100, ...
        'MaxFE',200000, ...
        'ProblemNames',{arrayfun(@(i)sprintf('DASCMOP%d_BC',i),1:9,'UniformOutput',false)}, ...
        'UseParallel',false, ...
        'Workers',0, ...
        'SavePath','benchmark_PRBCCMO_experiment0.mat', ...
        'SaveSummaryOnly',true);
    Params = ParseInputs(Params,varargin{:});
    Params = NormalizeRunControl(Params);

    Variants = ResolveExperimentVariants();
    Tasks = BuildExperimentTasks(Params.ProblemNames,Variants,Params.RunSeeds);
    [UpdateRows,RunSummary,ExecutionInfo] = ExecuteExperimentTasks(Tasks,Params);

    ProblemSummary = SummarizeProblems(UpdateRows,RunSummary,Variants,Params.ProblemNames);
    PooledSummary  = SummarizeVariants(UpdateRows,RunSummary,Variants);
    Criteria       = EvaluateSuccessCriteria(ProblemSummary,PooledSummary);

    Results = struct();
    Results.params = Params;
    Results.variants = Variants;
    Results.updateRows = UpdateRows;
    Results.runSummary = RunSummary;
    Results.problemSummary = ProblemSummary;
    Results.pooledSummary = PooledSummary;
    Results.criteria = Criteria;
    Results.execution = ExecutionInfo;

    if ~isempty(Params.SavePath)
        ResultOutput = Results;
        if Params.SaveSummaryOnly
            Results = BuildSavedResults(Results);
        end
        save(Params.SavePath,'Results','-v7.3');
        Results = ResultOutput;
    end
end

function Tasks = BuildExperimentTasks(ProblemNames,Variants,RunSeeds)
    TaskCount = numel(ProblemNames)*numel(Variants)*numel(RunSeeds);
    Tasks = repmat(InitExperimentTask(),TaskCount,1);
    Row = 0;
    for p = 1 : numel(ProblemNames)
        for v = 1 : numel(Variants)
            for r = 1 : numel(RunSeeds)
                Row = Row + 1;
                Tasks(Row).problem = ProblemNames{p};
                Tasks(Row).variantName = Variants(v).name;
                Tasks(Row).variantLabel = Variants(v).label;
                Tasks(Row).variantParameter = Variants(v).parameter;
                Tasks(Row).run = r;
                Tasks(Row).seed = RunSeeds(r);
            end
        end
    end
end

function Task = InitExperimentTask()
    Task = struct( ...
        'problem','', ...
        'variantName','', ...
        'variantLabel','', ...
        'variantParameter',{{}}, ...
        'run',0, ...
        'seed',0);
end

function [UpdateRows,RunSummary,ExecutionInfo] = ExecuteExperimentTasks(Tasks,Params)
    UpdateRowsCell = cell(numel(Tasks),1);
    RunSummaryCell = cell(numel(Tasks),1);
    ProjectRoot = pwd;
    Population = Params.Population;
    MaxFE = Params.MaxFE;
    ExecutionInfo = struct( ...
        'useParallel',logical(Params.UseParallel), ...
        'workers',1, ...
        'taskCount',numel(Tasks), ...
        'projectRoot',ProjectRoot);

    if Params.UseParallel && numel(Tasks) > 1
        Workers = ResolveParallelWorkers(Params.Workers,numel(Tasks));
        ExecutionInfo.workers = Workers;
        ConfigureParallelPool(Workers);
        parfor t = 1 : numel(Tasks)
            [UpdateRowsCell{t},RunSummaryCell{t}] = RunExperimentTask( ...
                Tasks(t),Population,MaxFE,ProjectRoot);
        end
    else
        for t = 1 : numel(Tasks)
            [UpdateRowsCell{t},RunSummaryCell{t}] = RunExperimentTask( ...
                Tasks(t),Population,MaxFE,ProjectRoot);
            Summary = RunSummaryCell{t};
            fprintf(['Task %03d/%03d | %s | %s | run=%02d seed=%d | ' ...
                'updates=%d pooledCount=%d invalid=%.4f singleClass=%.4f auditReady=%.4f coldStart=%.4f ' ...
                'pooledECE=%.4f coreGap=%.4f relaxedGap=%.4f TGP=%.4f\n'], ...
                t,numel(Tasks),Tasks(t).problem,Tasks(t).variantName,Tasks(t).run,Tasks(t).seed, ...
                Summary.updateCount,Summary.pooledCount,Summary.invalidUpdateRatio, ...
                Summary.singleClassUpdateRatio,Summary.auditReadyUpdateRatio,Summary.coldStartUpdateRatio, ...
                Summary.pooledECE,Summary.pooledCoreNearGap, ...
                Summary.pooledRelaxedNearGap,Summary.trustGatePassRate);
        end
    end

    UpdateRows = MergeUpdateRows(UpdateRowsCell);
    RunSummary = MergeRunSummary(RunSummaryCell);
end

function Workers = ResolveParallelWorkers(RequestedWorkers,TaskCount)
    if nargin < 1 || isempty(RequestedWorkers)
        RequestedWorkers = 0;
    end
    if nargin < 2 || isempty(TaskCount)
        TaskCount = inf;
    end

    if RequestedWorkers > 0
        Workers = min(max(1,round(RequestedWorkers)),TaskCount);
        return;
    end

    CoreCount = feature('numcores');
    Reserve = 4;
    Workers = max(1,CoreCount - Reserve);
    Workers = min([Workers,6,TaskCount]);
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

function [UpdateRows,RunSummary] = RunExperimentTask(Task,Population,MaxFE,ProjectRoot)
    EnsureBenchmarkPath(ProjectRoot);
    rng(Task.seed,'twister');

    Variant = struct('name',Task.variantName,'label',Task.variantLabel,'parameter',{Task.variantParameter});
    Problem = feval(Task.problem,'N',Population,'maxFE',MaxFE);
    Algorithm = PRBCCMO('parameter',Task.variantParameter,'save',0,'outputFcn',@SilentOutput);
    Algorithm.Solve(Problem);

    Trace = ResolveCalibrationTrace(Algorithm.metric);
    UpdateRows = ConvertTraceToRows(Trace,Task.problem,Variant,Task.run,Task.seed);
    RunSummary = SummarizeRun(Trace,Task.problem,Variant,Task.run,Task.seed);
end

function EnsureBenchmarkPath(ProjectRoot)
    persistent PathReady
    if ~isempty(PathReady) && PathReady
        return;
    end

    cd(ProjectRoot);
    addpath(genpath(ProjectRoot));
    PathReady = true;
end

function SilentOutput(~,~)
end

function UpdateRows = MergeUpdateRows(UpdateRowsCell)
    Counts = cellfun(@numel,UpdateRowsCell);
    Total = sum(Counts);
    if Total == 0
        UpdateRows = repmat(InitUpdateRow(),0,1);
        return;
    end

    UpdateRows = repmat(InitUpdateRow(),Total,1);
    Cursor = 0;
    for i = 1 : numel(UpdateRowsCell)
        Count = Counts(i);
        if Count == 0
            continue;
        end
        UpdateRows(Cursor + (1:Count),1) = UpdateRowsCell{i};
        Cursor = Cursor + Count;
    end
end

function RunSummary = MergeRunSummary(RunSummaryCell)
    Count = numel(RunSummaryCell);
    if Count == 0
        RunSummary = repmat(InitRunSummaryRow(),0,1);
        return;
    end

    RunSummary = repmat(InitRunSummaryRow(),Count,1);
    for i = 1 : Count
        RunSummary(i,1) = RunSummaryCell{i};
    end
end

function Variants = ResolveExperimentVariants()
    Variants = repmat(struct('name','','label','','calMode',0,'parameter',{{}}),4,1);
    Variants(1) = BuildVariant('raw','raw',1);
    Variants(2) = BuildVariant('temperature','temperature',4);
    Variants(3) = BuildVariant('beta','beta',3);
    Variants(4) = BuildVariant('auto_trust','online best-of-(temperature,beta) + trust gate',2);
end

function Variant = BuildVariant(Name,Label,CalMode)
    Variant = struct();
    Variant.name = Name;
    Variant.label = Label;
    Variant.calMode = CalMode;
    Variant.parameter = {0.2,2,20,25,0.01,0.4,3,CalMode,1,0.05,1,1,1,1,1};
end

function Trace = ResolveCalibrationTrace(Metric)
    Trace = repmat(InitTraceRow(),0,1);
    if ~isstruct(Metric) || ~isfield(Metric,'sectionB') || isempty(Metric.sectionB)
        return;
    end
    if ~isfield(Metric.sectionB,'calibrationTrace') || isempty(Metric.sectionB.calibrationTrace)
        return;
    end
    Trace = Metric.sectionB.calibrationTrace;
end

function Row = InitTraceRow()
    Row = struct( ...
        'generation',NaN, ...
        'FE',NaN, ...
        'count',0, ...
        'brier',NaN, ...
        'ece',NaN, ...
        'near_count',0, ...
        'near_gap',NaN, ...
        'core_near_count',0, ...
        'core_near_gap',NaN, ...
        'relaxed_near_count',0, ...
        'relaxed_near_gap',NaN, ...
        'valid',false, ...
        'invalid_reason','', ...
        'single_class',false, ...
        'class_count',0, ...
        'trust_gate',false, ...
        'calibrator','raw', ...
        'calibration_buffer_valid',false, ...
        'calibration_buffer_single_class',false, ...
        'calibration_buffer_class_count',0, ...
        'calibration_buffer_status','invalid_empty', ...
        'test_buffer_valid',false, ...
        'test_buffer_single_class',false, ...
        'test_buffer_class_count',0, ...
        'test_buffer_status','invalid_empty', ...
        'audit_ready',false, ...
        'audit_phase','not_yet_auditable', ...
        'coldstart_active',false, ...
        'coldstart_batch_count',0, ...
        'boundary_batch_count',0, ...
        'boundary_started',false, ...
        'ece_bin_count',zeros(1,10), ...
        'ece_bin_prob_sum',zeros(1,10), ...
        'ece_bin_label_sum',zeros(1,10), ...
        'prob',zeros(0,1), ...
        'label',zeros(0,1));
end

function Rows = ConvertTraceToRows(Trace,ProblemName,Variant,RunIndex,Seed)
    Rows = repmat(InitUpdateRow(),0,1);
    if isempty(Trace)
        return;
    end

    Rows = repmat(InitUpdateRow(),numel(Trace),1);
    for i = 1 : numel(Trace)
        Rows(i).problem = ProblemName;
        Rows(i).variant = Variant.name;
        Rows(i).run = RunIndex;
        Rows(i).seed = Seed;
        Rows(i).update = i;
        Rows(i).generation = FieldOrDefault(Trace(i),'generation',NaN);
        Rows(i).FE = FieldOrDefault(Trace(i),'FE',NaN);
        Rows(i).count = FieldOrDefault(Trace(i),'count',0);
        Rows(i).brier = FieldOrDefault(Trace(i),'brier',NaN);
        Rows(i).ece = FieldOrDefault(Trace(i),'ece',NaN);
        Rows(i).coreNearCount = ResolveTraceCoreCount(Trace(i));
        Rows(i).coreNearGap = ResolveTraceCoreGap(Trace(i));
        Rows(i).relaxedNearCount = FieldOrDefault(Trace(i),'relaxed_near_count',0);
        Rows(i).relaxedNearGap = FieldOrDefault(Trace(i),'relaxed_near_gap',NaN);
        Rows(i).valid = logical(FieldOrDefault(Trace(i),'valid',false));
        Rows(i).invalidReason = char(FieldOrDefault(Trace(i),'invalid_reason',''));
        Rows(i).singleClass = logical(FieldOrDefault(Trace(i),'single_class',false));
        Rows(i).classCount = FieldOrDefault(Trace(i),'class_count',0);
        Rows(i).trustGate = logical(FieldOrDefault(Trace(i),'trust_gate',false));
        Rows(i).calibrator = char(FieldOrDefault(Trace(i),'calibrator','raw'));
        Rows(i).calibrationBufferValid = logical(FieldOrDefault(Trace(i),'calibration_buffer_valid',false));
        Rows(i).calibrationBufferSingleClass = logical(FieldOrDefault(Trace(i),'calibration_buffer_single_class',false));
        Rows(i).testBufferValid = logical(FieldOrDefault(Trace(i),'test_buffer_valid',false));
        Rows(i).testBufferSingleClass = logical(FieldOrDefault(Trace(i),'test_buffer_single_class',false));
        Rows(i).auditReady = logical(FieldOrDefault(Trace(i),'audit_ready',false));
        Rows(i).auditPhase = char(FieldOrDefault(Trace(i),'audit_phase','not_yet_auditable'));
        Rows(i).coldStartActive = logical(FieldOrDefault(Trace(i),'coldstart_active',false));
        Rows(i).coldStartBatchCount = FieldOrDefault(Trace(i),'coldstart_batch_count',0);
        Rows(i).boundaryBatchCount = FieldOrDefault(Trace(i),'boundary_batch_count',0);
        Rows(i).boundaryStarted = logical(FieldOrDefault(Trace(i),'boundary_started',false));
    end
end

function Row = InitUpdateRow()
    Row = struct( ...
        'problem','', ...
        'variant','', ...
        'run',0, ...
        'seed',0, ...
        'update',0, ...
        'generation',NaN, ...
        'FE',NaN, ...
        'count',0, ...
        'brier',NaN, ...
        'ece',NaN, ...
        'coreNearCount',0, ...
        'coreNearGap',NaN, ...
        'relaxedNearCount',0, ...
        'relaxedNearGap',NaN, ...
        'valid',false, ...
        'invalidReason','', ...
        'singleClass',false, ...
        'classCount',0, ...
        'trustGate',false, ...
        'calibrator','raw', ...
        'calibrationBufferValid',false, ...
        'calibrationBufferSingleClass',false, ...
        'testBufferValid',false, ...
        'testBufferSingleClass',false, ...
        'auditReady',false, ...
        'auditPhase','not_yet_auditable', ...
        'coldStartActive',false, ...
        'coldStartBatchCount',0, ...
        'boundaryBatchCount',0, ...
        'boundaryStarted',false);
end

function Summary = SummarizeRun(Trace,ProblemName,Variant,RunIndex,Seed)
    Summary = InitRunSummaryRow();
    Summary.problem = ProblemName;
    Summary.variant = Variant.name;
    Summary.run = RunIndex;
    Summary.seed = Seed;
    Summary.updateCount = numel(Trace);
    Summary.validUpdateCount = sum(arrayfun(@(S)IsTraceAggregationReady(S),Trace));
    Summary.invalidUpdateCount = Summary.updateCount - Summary.validUpdateCount;
    Summary.auditReadyUpdateCount = sum(arrayfun(@(S)logical(FieldOrDefault(S,'audit_ready',false)),Trace));
    Summary.notYetAuditableUpdateCount = Summary.updateCount - Summary.auditReadyUpdateCount;
    Summary.coldStartUpdateCount = sum(arrayfun(@(S)logical(FieldOrDefault(S,'coldstart_active',false)),Trace));
    Summary.singleClassUpdateCount = sum(arrayfun(@(S)logical(FieldOrDefault(S,'single_class',false)),Trace));
    Summary.invalidUpdateRatio = SafeRatio(Summary.invalidUpdateCount,Summary.updateCount);
    Summary.auditReadyUpdateRatio = SafeRatio(Summary.auditReadyUpdateCount,Summary.updateCount);
    Summary.notYetAuditableUpdateRatio = SafeRatio(Summary.notYetAuditableUpdateCount,Summary.updateCount);
    Summary.coldStartUpdateRatio = SafeRatio(Summary.coldStartUpdateCount,Summary.updateCount);
    Summary.singleClassUpdateRatio = SafeRatio(Summary.singleClassUpdateCount,Summary.updateCount);
    Summary.meanBrier = MeanField(Trace,'brier',true);
    Summary.meanECE = MeanField(Trace,'ece',true);
    Summary.meanCoreNearGap = MeanCoreGap(Trace,true);
    Summary.meanRelaxedNearGap = MeanField(Trace,'relaxed_near_gap',true);
    Summary.trustGatePassRate = MeanTrustGate(Trace);
    Summary.trace = repmat(InitTraceRow(),0,1);
    if isempty(Trace)
        return;
    end

    Pooled = SummarizeTraceAggregation(Trace);
    Summary.pooledCount = Pooled.count;
    Summary.pooledBrier = Pooled.brier;
    Summary.pooledECE = Pooled.ece;
    Summary.pooledCoreNearCount = Pooled.coreNearCount;
    Summary.pooledCoreNearGap = Pooled.coreNearGap;
    Summary.pooledRelaxedNearCount = Pooled.relaxedNearCount;
    Summary.pooledRelaxedNearGap = Pooled.relaxedNearGap;
    Summary.pooledCoreNearPositiveCount = Pooled.coreNearPositiveCount;
    Summary.pooledRelaxedNearPositiveCount = Pooled.relaxedNearPositiveCount;
    Summary.eceBinCount = Pooled.eceBinCount;
    Summary.eceBinProbSum = Pooled.eceBinProbSum;
    Summary.eceBinLabelSum = Pooled.eceBinLabelSum;

    FinalTrace = ResolveFinalValidTrace(Trace);
    Summary.finalBrier = FieldOrDefault(FinalTrace,'brier',NaN);
    Summary.finalECE = FieldOrDefault(FinalTrace,'ece',NaN);
    Summary.finalCoreNearGap = ResolveTraceCoreGap(FinalTrace);
    Summary.finalRelaxedNearGap = FieldOrDefault(FinalTrace,'relaxed_near_gap',NaN);
end

function Row = InitRunSummaryRow()
    Row = struct( ...
        'problem','', ...
        'variant','', ...
        'run',0, ...
        'seed',0, ...
        'updateCount',0, ...
        'validUpdateCount',0, ...
        'invalidUpdateCount',0, ...
        'auditReadyUpdateCount',0, ...
        'notYetAuditableUpdateCount',0, ...
        'coldStartUpdateCount',0, ...
        'singleClassUpdateCount',0, ...
        'invalidUpdateRatio',NaN, ...
        'auditReadyUpdateRatio',NaN, ...
        'notYetAuditableUpdateRatio',NaN, ...
        'coldStartUpdateRatio',NaN, ...
        'singleClassUpdateRatio',NaN, ...
        'pooledCount',0, ...
        'pooledBrier',NaN, ...
        'pooledECE',NaN, ...
        'pooledCoreNearCount',0, ...
        'pooledCoreNearGap',NaN, ...
        'pooledCoreNearPositiveCount',0, ...
        'pooledRelaxedNearCount',0, ...
        'pooledRelaxedNearGap',NaN, ...
        'pooledRelaxedNearPositiveCount',0, ...
        'eceBinCount',zeros(1,10), ...
        'eceBinProbSum',zeros(1,10), ...
        'eceBinLabelSum',zeros(1,10), ...
        'meanBrier',NaN, ...
        'meanECE',NaN, ...
        'meanCoreNearGap',NaN, ...
        'meanRelaxedNearGap',NaN, ...
        'finalBrier',NaN, ...
        'finalECE',NaN, ...
        'finalCoreNearGap',NaN, ...
        'finalRelaxedNearGap',NaN, ...
        'trustGatePassRate',NaN, ...
        'trace',repmat(InitTraceRow(),0,1));
end

function Summary = SummarizeProblems(UpdateRows,RunSummary,Variants,ProblemNames)
    Summary = repmat(InitProblemSummaryRow(),numel(ProblemNames)*numel(Variants),1);
    Row = 0;
    for p = 1 : numel(ProblemNames)
        ProblemName = ProblemNames{p};
        for v = 1 : numel(Variants)
            VariantName = Variants(v).name;
            RunMask = strcmp({RunSummary.problem},ProblemName) & strcmp({RunSummary.variant},VariantName);
            UpdateMask = strcmp({UpdateRows.problem},ProblemName) & strcmp({UpdateRows.variant},VariantName);
            Row = Row + 1;
            Summary(Row,1) = SummarizeSubset( ...
                UpdateRows(UpdateMask),RunSummary(RunMask),ProblemName,VariantName);
        end
    end
end

function Summary = InitProblemSummaryRow()
    Summary = struct( ...
        'problem','', ...
        'variant','', ...
        'runs',0, ...
        'updates',0, ...
        'validUpdates',0, ...
        'invalidUpdates',0, ...
        'auditReadyUpdates',0, ...
        'notYetAuditableUpdates',0, ...
        'coldStartUpdates',0, ...
        'singleClassUpdates',0, ...
        'invalidUpdateRatio',NaN, ...
        'auditReadyUpdateRatio',NaN, ...
        'notYetAuditableUpdateRatio',NaN, ...
        'coldStartUpdateRatio',NaN, ...
        'singleClassUpdateRatio',NaN, ...
        'pooledCount',0, ...
        'pooledBrier',NaN, ...
        'pooledECE',NaN, ...
        'pooledCoreNearCount',0, ...
        'pooledCoreNearGap',NaN, ...
        'pooledCoreNearPositiveCount',0, ...
        'pooledRelaxedNearCount',0, ...
        'pooledRelaxedNearGap',NaN, ...
        'pooledRelaxedNearPositiveCount',0, ...
        'meanTrustGatePassRate',NaN, ...
        'medianTrustGatePassRate',NaN, ...
        'meanRunPooledECE',NaN, ...
        'meanRunPooledCoreNearGap',NaN);
end

function Summary = SummarizeVariants(UpdateRows,RunSummary,Variants)
    Summary = repmat(InitVariantSummaryRow(),numel(Variants),1);
    for v = 1 : numel(Variants)
        VariantName = Variants(v).name;
        RunMask = strcmp({RunSummary.variant},VariantName);
        UpdateMask = strcmp({UpdateRows.variant},VariantName);
        Summary(v,1) = SummarizeSubset( ...
            UpdateRows(UpdateMask),RunSummary(RunMask),'ALL',VariantName);
    end
end

function Summary = InitVariantSummaryRow()
    Summary = InitProblemSummaryRow();
end

function Summary = SummarizeSubset(UpdateRows,RunSummary,ProblemName,VariantName)
    Summary = InitProblemSummaryRow();
    Summary.problem = ProblemName;
    Summary.variant = VariantName;
    Summary.runs = numel(RunSummary);
    Summary.updates = numel(UpdateRows);
    Summary.validUpdates = sum([RunSummary.validUpdateCount]);
    Summary.invalidUpdates = sum([RunSummary.invalidUpdateCount]);
    Summary.auditReadyUpdates = sum([RunSummary.auditReadyUpdateCount]);
    Summary.notYetAuditableUpdates = sum([RunSummary.notYetAuditableUpdateCount]);
    Summary.coldStartUpdates = sum([RunSummary.coldStartUpdateCount]);
    Summary.singleClassUpdates = sum([RunSummary.singleClassUpdateCount]);
    Summary.invalidUpdateRatio = SafeRatio(Summary.invalidUpdates,Summary.updates);
    Summary.auditReadyUpdateRatio = SafeRatio(Summary.auditReadyUpdates,Summary.updates);
    Summary.notYetAuditableUpdateRatio = SafeRatio(Summary.notYetAuditableUpdates,Summary.updates);
    Summary.coldStartUpdateRatio = SafeRatio(Summary.coldStartUpdates,Summary.updates);
    Summary.singleClassUpdateRatio = SafeRatio(Summary.singleClassUpdates,Summary.updates);
    Summary.meanTrustGatePassRate = MeanScalarStructField(RunSummary,'trustGatePassRate');
    Summary.medianTrustGatePassRate = MedianScalarStructField(RunSummary,'trustGatePassRate');
    Summary.meanRunPooledECE = MeanScalarStructField(RunSummary,'pooledECE');
    Summary.meanRunPooledCoreNearGap = MeanScalarStructField(RunSummary,'pooledCoreNearGap');
    if isempty(UpdateRows)
        return;
    end

    Summary.pooledCount = sum([RunSummary.pooledCount]);
    Summary.pooledBrier = WeightedMeanScalarStructField(RunSummary,'pooledBrier','pooledCount');
    Summary.pooledCoreNearCount = sum([RunSummary.pooledCoreNearCount]);
    Summary.pooledCoreNearPositiveCount = sum([RunSummary.pooledCoreNearPositiveCount]);
    Summary.pooledCoreNearGap = ResolveNearGap( ...
        Summary.pooledCoreNearPositiveCount,Summary.pooledCoreNearCount);
    Summary.pooledRelaxedNearCount = sum([RunSummary.pooledRelaxedNearCount]);
    Summary.pooledRelaxedNearPositiveCount = sum([RunSummary.pooledRelaxedNearPositiveCount]);
    Summary.pooledRelaxedNearGap = ResolveNearGap( ...
        Summary.pooledRelaxedNearPositiveCount,Summary.pooledRelaxedNearCount);

    [Summary.pooledECE,~] = ResolveAggregatedECE(RunSummary);
end

function Criteria = EvaluateSuccessCriteria(ProblemSummary,PooledSummary)
    Criteria = struct();
    Criteria.primaryVariant = 'auto_trust';
    Criteria.pooledECEPass = false;
    Criteria.pooledCoreNearGapPass = false;
    Criteria.problemWinsVsRaw = 0;
    Criteria.problemWinsVsTemperature = 0;
    Criteria.requiredProblemWins = 7;
    Criteria.overallPass = false;

    PrimaryIdx = find(strcmp({PooledSummary.variant},Criteria.primaryVariant),1,'first');
    if ~isempty(PrimaryIdx)
        Criteria.pooledECEPass = PooledSummary(PrimaryIdx).pooledECE <= 0.05;
        Criteria.pooledCoreNearGapPass = PooledSummary(PrimaryIdx).pooledCoreNearGap <= 0.05;
    end

    Problems = unique({ProblemSummary.problem},'stable');
    Problems(strcmp(Problems,'ALL')) = [];
    for i = 1 : numel(Problems)
        ProblemName = Problems{i};
        AutoRow = ResolveSummaryRow(ProblemSummary,ProblemName,'auto_trust');
        RawRow  = ResolveSummaryRow(ProblemSummary,ProblemName,'raw');
        TempRow = ResolveSummaryRow(ProblemSummary,ProblemName,'temperature');
        if ~isempty(AutoRow) && ~isempty(RawRow) ...
                && AutoRow.meanTrustGatePassRate > RawRow.meanTrustGatePassRate
            Criteria.problemWinsVsRaw = Criteria.problemWinsVsRaw + 1;
        end
        if ~isempty(AutoRow) && ~isempty(TempRow) ...
                && AutoRow.meanTrustGatePassRate > TempRow.meanTrustGatePassRate
            Criteria.problemWinsVsTemperature = Criteria.problemWinsVsTemperature + 1;
        end
    end

    Criteria.overallPass = Criteria.pooledECEPass && Criteria.pooledCoreNearGapPass ...
        && Criteria.problemWinsVsRaw >= Criteria.requiredProblemWins ...
        && Criteria.problemWinsVsTemperature >= Criteria.requiredProblemWins;
end

function Row = ResolveSummaryRow(Summary,ProblemName,VariantName)
    Row = [];
    Match = find(strcmp({Summary.problem},ProblemName) & strcmp({Summary.variant},VariantName),1,'first');
    if isempty(Match)
        return;
    end
    Row = Summary(Match);
end

function Results = BuildSavedResults(FullResults)
    Results = struct();
    Results.params = FullResults.params;
    Results.variants = FullResults.variants;
    Results.runSummary = FullResults.runSummary;
    Results.problemSummary = FullResults.problemSummary;
    Results.pooledSummary = FullResults.pooledSummary;
    Results.criteria = FullResults.criteria;
    Results.execution = FullResults.execution;
end

function Value = MeanField(Trace,Field,ValidOnly)
    if nargin < 3
        ValidOnly = false;
    end
    if ValidOnly
        Trace = FilterValidTrace(Trace);
    end
    if isempty(Trace)
        Value = NaN;
        return;
    end
    Data = arrayfun(@(S)FieldOrDefault(S,Field,NaN),Trace);
    Value = MeanFinite(Data);
end

function Value = MeanCoreGap(Trace,ValidOnly)
    if nargin < 2
        ValidOnly = false;
    end
    if ValidOnly
        Trace = FilterValidTrace(Trace);
    end
    if isempty(Trace)
        Value = NaN;
        return;
    end
    Data = arrayfun(@(S)ResolveTraceCoreGap(S),Trace);
    Value = MeanFinite(Data);
end

function Value = MeanTrustGate(Trace)
    Trace = FilterAuditableTrace(Trace);
    if isempty(Trace)
        Value = NaN;
        return;
    end
    Data = arrayfun(@(S)double(FieldOrDefault(S,'trust_gate',false)),Trace);
    Value = mean(Data);
end

function Value = MeanScalarStructField(S,Field)
    if isempty(S)
        Value = NaN;
        return;
    end
    Data = arrayfun(@(Row)FieldOrDefault(Row,Field,NaN),S);
    Value = MeanFinite(Data);
end

function Value = MedianScalarStructField(S,Field)
    if isempty(S)
        Value = NaN;
        return;
    end
    Data = arrayfun(@(Row)FieldOrDefault(Row,Field,NaN),S);
    Data = Data(isfinite(Data));
    if isempty(Data)
        Value = NaN;
    else
        Value = median(Data);
    end
end

function Value = WeightedMeanScalarStructField(S,ValueField,WeightField)
    if isempty(S)
        Value = NaN;
        return;
    end
    Values = arrayfun(@(Row)FieldOrDefault(Row,ValueField,NaN),S);
    Weights = arrayfun(@(Row)FieldOrDefault(Row,WeightField,0),S);
    Value = WeightedMeanFinite(Values,Weights);
end

function Summary = SummarizeTraceAggregation(Trace)
    Summary = InitTraceAggregationSummary();
    Trace = FilterValidTrace(Trace);
    if isempty(Trace)
        return;
    end

    Counts = arrayfun(@(S)max(0,FieldOrDefault(S,'count',0)),Trace);
    Summary.count = sum(Counts);
    Brier = arrayfun(@(S)FieldOrDefault(S,'brier',NaN),Trace);
    Summary.brier = WeightedMeanFinite(Brier,Counts);

    CoreCount = arrayfun(@(S)max(0,FieldOrDefault(S,'core_near_count',0)),Trace);
    CoreRate  = arrayfun(@(S)FieldOrDefault(S,'core_near_feasible_rate',NaN),Trace);
    ValidCore = CoreCount > 0 & isfinite(CoreRate);
    Summary.coreNearCount = sum(CoreCount(ValidCore));
    Summary.coreNearPositiveCount = sum(CoreCount(ValidCore).*CoreRate(ValidCore));
    Summary.coreNearGap = ResolveNearGap( ...
        Summary.coreNearPositiveCount,Summary.coreNearCount);

    RelaxedCount = arrayfun(@(S)max(0,FieldOrDefault(S,'relaxed_near_count',0)),Trace);
    RelaxedRate  = arrayfun(@(S)FieldOrDefault(S,'relaxed_near_feasible_rate',NaN),Trace);
    ValidRelaxed = RelaxedCount > 0 & isfinite(RelaxedRate);
    Summary.relaxedNearCount = sum(RelaxedCount(ValidRelaxed));
    Summary.relaxedNearPositiveCount = sum(RelaxedCount(ValidRelaxed).*RelaxedRate(ValidRelaxed));
    Summary.relaxedNearGap = ResolveNearGap( ...
        Summary.relaxedNearPositiveCount,Summary.relaxedNearCount);

    [Summary.ece,BinStats] = ResolveAggregatedECE(Trace);
    Summary.eceBinCount = BinStats.count;
    Summary.eceBinProbSum = BinStats.probSum;
    Summary.eceBinLabelSum = BinStats.labelSum;
end

function Summary = InitTraceAggregationSummary()
    Summary = struct( ...
        'count',0, ...
        'brier',NaN, ...
        'ece',NaN, ...
        'coreNearCount',0, ...
        'coreNearPositiveCount',0, ...
        'coreNearGap',NaN, ...
        'relaxedNearCount',0, ...
        'relaxedNearPositiveCount',0, ...
        'relaxedNearGap',NaN, ...
        'eceBinCount',zeros(1,10), ...
        'eceBinProbSum',zeros(1,10), ...
        'eceBinLabelSum',zeros(1,10));
end

function [ECE,BinStats] = ResolveAggregatedECE(Source)
    [CountField,ProbField,LabelField] = ResolveECEFieldNames(Source);
    BinStats = InitECEBinStats(ResolveBinWidth(Source,CountField));
    if isempty(Source)
        ECE = NaN;
        return;
    end

    for i = 1 : numel(Source)
        BinStats.count = BinStats.count + NormalizeBinRow(FieldOrDefault(Source(i),CountField,[]),numel(BinStats.count));
        BinStats.probSum = BinStats.probSum + NormalizeBinRow(FieldOrDefault(Source(i),ProbField,[]),numel(BinStats.count));
        BinStats.labelSum = BinStats.labelSum + NormalizeBinRow(FieldOrDefault(Source(i),LabelField,[]),numel(BinStats.count));
    end

    Total = sum(BinStats.count);
    if Total <= 0
        ECE = NaN;
        return;
    end

    Valid = BinStats.count > 0;
    MeanProb = zeros(size(BinStats.count));
    FeasibleRate = zeros(size(BinStats.count));
    MeanProb(Valid) = BinStats.probSum(Valid)./BinStats.count(Valid);
    FeasibleRate(Valid) = BinStats.labelSum(Valid)./BinStats.count(Valid);
    ECE = sum((BinStats.count(Valid)./Total).*abs(FeasibleRate(Valid)-MeanProb(Valid)));
end

function [CountField,ProbField,LabelField] = ResolveECEFieldNames(Source)
    if ~isempty(Source) && isfield(Source,'ece_bin_count')
        CountField = 'ece_bin_count';
        ProbField = 'ece_bin_prob_sum';
        LabelField = 'ece_bin_label_sum';
    else
        CountField = 'eceBinCount';
        ProbField = 'eceBinProbSum';
        LabelField = 'eceBinLabelSum';
    end
end

function BinStats = InitECEBinStats(Width)
    Width = max(1,round(Width));
    BinStats = struct( ...
        'count',zeros(1,Width), ...
        'probSum',zeros(1,Width), ...
        'labelSum',zeros(1,Width));
end

function Width = ResolveBinWidth(Source,Field)
    Width = 10;
    if isempty(Source)
        return;
    end
    Sample = FieldOrDefault(Source(1),Field,zeros(1,10));
    if ~isempty(Sample)
        Width = numel(Sample);
    end
end

function Row = NormalizeBinRow(Data,Width)
    Row = zeros(1,Width);
    if isempty(Data)
        return;
    end
    Data = double(Data(:))';
    Take = min(numel(Data),Width);
    Row(1:Take) = Data(1:Take);
end

function Trace = FilterValidTrace(Trace)
    if isempty(Trace)
        return;
    end
    Mask = arrayfun(@(S)IsTraceAggregationReady(S),Trace);
    Trace = Trace(Mask);
end

function Trace = FilterAuditableTrace(Trace)
    if isempty(Trace)
        return;
    end
    Mask = arrayfun(@(S)logical(FieldOrDefault(S,'audit_ready',false)),Trace);
    Trace = Trace(Mask);
end

function Flag = IsTraceAggregationReady(Row)
    Flag = logical(FieldOrDefault(Row,'valid',false)) ...
        && logical(FieldOrDefault(Row,'audit_ready',false));
end

function Trace = ResolveFinalValidTrace(Trace)
    ValidTrace = FilterValidTrace(Trace);
    if ~isempty(ValidTrace)
        Trace = ValidTrace(end);
    elseif isempty(Trace)
        Trace = InitTraceRow();
    else
        Trace = Trace(end);
    end
end

function Value = WeightedMeanFinite(Data,Weights)
    Valid = isfinite(Data) & isfinite(Weights) & Weights > 0;
    if ~any(Valid)
        Value = NaN;
        return;
    end
    Value = sum(Data(Valid).*Weights(Valid))/sum(Weights(Valid));
end

function Value = ResolveNearGap(PositiveCount,TotalCount)
    if TotalCount <= 0
        Value = NaN;
        return;
    end
    Value = abs(PositiveCount/TotalCount - 0.5);
end

function Value = SafeRatio(Numerator,Denominator)
    if Denominator <= 0
        Value = NaN;
    else
        Value = Numerator/Denominator;
    end
end

function Value = MeanFinite(Data)
    Data = Data(isfinite(Data));
    if isempty(Data)
        Value = NaN;
    else
        Value = mean(Data);
    end
end

function Value = ResolveTraceCoreCount(Row)
    if isfield(Row,'core_near_count') && ~isempty(Row.core_near_count)
        Value = Row.core_near_count;
    else
        Value = FieldOrDefault(Row,'near_count',0);
    end
end

function Value = ResolveTraceCoreGap(Row)
    if isfield(Row,'core_near_gap') && ~isempty(Row.core_near_gap)
        Value = Row.core_near_gap;
    else
        Value = FieldOrDefault(Row,'near_gap',NaN);
    end
end

function Value = FieldOrDefault(S,Field,Default)
    if isstruct(S) && isfield(S,Field) && ~isempty(S.(Field))
        Value = S.(Field);
    else
        Value = Default;
    end
end

function Params = NormalizeRunControl(Params)
    if isfield(Params,'RunSeeds') && ~isempty(Params.RunSeeds)
        Params.RunSeeds = reshape(double(Params.RunSeeds),1,[]);
        Params.Runs = numel(Params.RunSeeds);
    else
        Params.Runs = max(1,round(Params.Runs));
        Params.RunSeeds = 1000 + (1:Params.Runs);
    end
end

function Params = ParseInputs(Params,varargin)
    if mod(numel(varargin),2) ~= 0
        error('benchmark_PRBCCMO_experiment0:InvalidInput','Inputs must be name-value pairs.');
    end
    for i = 1 : 2 : numel(varargin)
        Name = varargin{i};
        if ~isfield(Params,Name)
            error('benchmark_PRBCCMO_experiment0:UnknownOption','Unknown option: %s',Name);
        end
        Params.(Name) = varargin{i+1};
    end
end
