function [Summary,outDir,MetricsAll] = run_CCMO_GAN_BDG_boundary_diagnostics( ...
        outDir,workerCount,problemNames,N,D,maxFE,runIds,targets,diagOptions)
% Run no-plot CGAN boundary diagnostics for the current localMAD_nGen20 mainline.

    rootDir = fileparts(which('platemo'));
    if nargin < 1 || isempty(outDir)
        outDir = fullfile(rootDir,'Data','CCMO_GAN_BDG', ...
            ['GND_keep80_localMAD_nGen20_diag_', ...
            char(datetime('now','Format','yyyyMMdd_HHmmss'))]);
    end
    if nargin < 2 || isempty(workerCount)
        workerCount = 9;
    end
    if nargin < 3 || isempty(problemNames)
        problemNames = defaultDiagnosticProblemList_BDG();
    end
    if nargin < 4 || isempty(N)
        N = 100;
    end
    if nargin < 5
        D = [];
    end
    if nargin < 6 || isempty(maxFE)
        maxFE = 100000;
    end
    if nargin < 7 || isempty(runIds)
        runIds = 1 : 3;
    end
    if nargin < 8 || isempty(targets)
        targets = [30000 70000 100000];
    end
    if nargin < 9
        diagOptions = struct();
    end

    addpath(genpath(rootDir));
    workerCount = max(1,round(double(workerCount)));
    problemNames = cellstr(string(problemNames(:)));
    runIds = double(runIds(:)');
    targets = normalizeDiagnosticTargets_BDG(targets);
    diagOptions = normalizeDiagnosticOptions_BDG(diagOptions);
    algorithmParams = {1,20,4,5,0.20,50,0,200,0,1,1,1, ...
        "g2sl",64,0.9,1e-4,1e-4,"epoch"};

    if ~isfolder(outDir)
        mkdir(outDir);
    end
    fprintf('boundary_diagnostics outDir=%s\n',outDir);
    fprintf('tasks=%d problems x %d runs, N=%d, D=%s, maxFE=%d, workers=%d, targets=%s\n', ...
        numel(problemNames),numel(runIds),N,defaultDText_BDG(D),maxFE, ...
        workerCount,strjoin(string(targets),';'));

    Tasks = buildDiagnosticTasks_BDG(problemNames,runIds);
    Rows = repmat(emptyDiagnosticRunRow_BDG(),height(Tasks),1);
    if workerCount > 1
        ensureDiagnosticParallelPool_BDG(workerCount);
        parfor task = 1 : height(Tasks)
            Rows(task) = completeDiagnosticRunRow_BDG(runOneDiagnosticTask_BDG( ...
                Tasks(task,:),outDir,N,D,maxFE,targets,diagOptions, ...
                algorithmParams));
        end
    else
        for task = 1 : height(Tasks)
            Rows(task) = completeDiagnosticRunRow_BDG(runOneDiagnosticTask_BDG( ...
                Tasks(task,:),outDir,N,D,maxFE,targets,diagOptions, ...
                algorithmParams));
            fprintf('[%d/%d] %s run=%g status=%s\n',task,height(Tasks), ...
                Tasks.problem(task),Tasks.run(task),Rows(task).status);
        end
    end

    Summary = struct2table(Rows);
    writetable(Summary,fullfile(outDir,'boundary_diagnostic_run_summary.csv'));
    MetricsAll = collectDiagnosticMetrics_BDG(Summary);
    writetable(MetricsAll,fullfile(outDir,'gan_diagnostic_metrics_all.csv'));
    writeDiagnosticMetricSummaries_BDG(MetricsAll,outDir);
end

function Tasks = buildDiagnosticTasks_BDG(problemNames,runIds)
    count = numel(problemNames) * numel(runIds);
    Rows = repmat(struct('problem',"",'run',NaN),count,1);
    k = 0;
    for p = 1 : numel(problemNames)
        for r = 1 : numel(runIds)
            k = k + 1;
            Rows(k).problem = string(problemNames{p});
            Rows(k).run = double(runIds(r));
        end
    end
    Tasks = struct2table(Rows);
end

function Row = runOneDiagnosticTask_BDG(Task,outDir,N,D,maxFE,targets, ...
        diagOptions,algorithmParams)
    Row = emptyDiagnosticRunRow_BDG();
    Row.variant = "GND_keep80_localMAD_nGen20";
    Row.problem = string(Task.problem);
    Row.family = diagnosticProblemFamily_BDG(Row.problem);
    Row.run = double(Task.run);
    Row.seed = Row.run;
    Row.N = double(N);
    Row.maxFE = double(maxFE);
    Row.target_text = strjoin(string(targets),';');
    Row.diagnostic_condition_count = diagOptions.conditionCount;
    Row.diagnostic_z_per_condition = diagOptions.zPerCondition;
    Row.diagnostic_normal_n = diagOptions.normalN;
    try
        rootDir = fileparts(which('platemo'));
        addpath(genpath(rootDir));
        rng(Row.seed,'twister');
        ProblemConstructor = str2func(char(Row.problem));
        if isempty(D)
            Problem = ProblemConstructor('N',N,'maxFE',maxFE);
        else
            Problem = ProblemConstructor('N',N,'D',D,'maxFE',maxFE);
        end
        Row.D = double(Problem.D);
        taskName = sprintf('%s_run%d',char(Row.problem),round(Row.run));
        snapshotFolder = string(fullfile(outDir,'snapshots',taskName));
        diagnosticFolder = string(fullfile(outDir,'diagnostics',taskName));
        if ~isfolder(char(snapshotFolder))
            mkdir(char(snapshotFolder));
        end
        if ~isfolder(char(diagnosticFolder))
            mkdir(char(diagnosticFolder));
        end
        Row.snapshot_folder = snapshotFolder;
        Row.snapshot_manifest_file = string(fullfile(char(snapshotFolder), ...
            'archive_snapshot_manifest.csv'));
        Row.diagnostic_folder = diagnosticFolder;
        Row.diagnostic_manifest_file = string(fullfile(char(diagnosticFolder), ...
            'gan_diagnostic_manifest.csv'));

        Control = struct( ...
            'variant',Row.variant, ...
            'archiveGap',1, ...
            'archiveParetoFilterMode',"global_af_nd", ...
            'archivePairDirectionMode',"af_not_dominates_ai", ...
            'archiveSourceCapMode',"none", ...
            'archivePairRefMode',"neighbor4", ...
            'trainFilterMode',"local_mad_weight", ...
            'conditionKNNRetainRatio',0.80, ...
            'cganTrainMinRefCov',0.00, ...
            'cganTrainMinTargetTriples',0, ...
            'trainFilterRandomSeed',Row.seed, ...
            'conditionMode',"yt_dt", ...
            'targetRealLabelMode',"binary", ...
            'archiveSnapshotTargets',targets, ...
            'archiveSnapshotFolder',snapshotFolder, ...
            'archiveSnapshotTag',Row.variant, ...
            'archiveSnapshotGanN',0, ...
            'archiveDiagnosticEnabled',true, ...
            'archiveDiagnosticFolder',diagnosticFolder, ...
            'archiveDiagnosticConditionCount',diagOptions.conditionCount, ...
            'archiveDiagnosticZPerCondition',diagOptions.zPerCondition, ...
            'archiveDiagnosticNormalN',diagOptions.normalN, ...
            'archiveDiagnosticSeed',diagOptions.seed, ...
            'generatorMode',"objective_target_conditioned", ...
            'generatorLossMode',"conditional_adversarial", ...
            'ganCriticMode',"target_conditioned");
        setappdata(0,'CCMO_GAN_BDG_ExperimentControl',Control);
        cleanup = onCleanup(@()removeDiagnosticAppdata_BDG()); %#ok<NASGU>

        Algorithm = CCMO_GAN_BDG( ...
            'save',0, ...
            'run',Row.run, ...
            'metName',{'IGD','HV','Feasible_rate'}, ...
            'outputFcn',@(varargin)[], ...
            'parameter',algorithmParams);
        Algorithm.Solve(Problem);

        Row.runtime = double(Algorithm.metric.runtime);
        Row.core_metrics_file = string(Algorithm.metric.analysis_core_csv);
        Row.stage_metrics_file = string(Algorithm.metric.analysis_stage_csv);
        Row.IGD = lastDiagnosticValue_BDG(Algorithm.CalMetric('IGD'));
        Row.HV = lastDiagnosticValue_BDG(Algorithm.CalMetric('HV'));
        Row.Feasible_rate = lastDiagnosticValue_BDG( ...
            Algorithm.CalMetric('Feasible_rate'));
        Row.finalFE = double(Algorithm.result{end,1});
        if isfile(Row.diagnostic_manifest_file)
            Manifest = readtable(Row.diagnostic_manifest_file, ...
                'TextType','string','Delimiter',',', ...
                'ReadVariableNames',true,'VariableNamingRule','preserve');
            Row.diagnostic_snapshot_count = height(Manifest);
        end
        if isfile(Row.snapshot_manifest_file)
            Manifest = readtable(Row.snapshot_manifest_file, ...
                'TextType','string','Delimiter',',', ...
                'ReadVariableNames',true,'VariableNamingRule','preserve');
            Row.snapshot_count = height(Manifest);
        end
        Row.status = "ok";
    catch err
        Row.status = "failed";
        Row.error_message = string(getReport(err,'extended', ...
            'hyperlinks','off'));
    end
end

function Row = completeDiagnosticRunRow_BDG(Source)
    Row = emptyDiagnosticRunRow_BDG();
    names = intersect(fieldnames(Source),fieldnames(Row),'stable');
    for i = 1 : numel(names)
        Row.(names{i}) = Source.(names{i});
    end
end

function Row = emptyDiagnosticRunRow_BDG()
    Row = struct( ...
        'variant',"", ...
        'problem',"", ...
        'family',"", ...
        'run',NaN, ...
        'seed',NaN, ...
        'N',NaN, ...
        'D',NaN, ...
        'maxFE',NaN, ...
        'target_text',"", ...
        'diagnostic_condition_count',NaN, ...
        'diagnostic_z_per_condition',NaN, ...
        'diagnostic_normal_n',NaN, ...
        'runtime',NaN, ...
        'IGD',NaN, ...
        'HV',NaN, ...
        'Feasible_rate',NaN, ...
        'finalFE',NaN, ...
        'snapshot_count',0, ...
        'diagnostic_snapshot_count',0, ...
        'core_metrics_file',"", ...
        'stage_metrics_file',"", ...
        'snapshot_folder',"", ...
        'snapshot_manifest_file',"", ...
        'diagnostic_folder',"", ...
        'diagnostic_manifest_file',"", ...
        'status',"pending", ...
        'error_message',"");
end

function MetricsAll = collectDiagnosticMetrics_BDG(Summary)
    MetricsAll = table();
    if isempty(Summary)
        return;
    end
    for i = 1 : height(Summary)
        if string(Summary.status(i)) ~= "ok"
            continue;
        end
        manifestFile = string(Summary.diagnostic_manifest_file(i));
        if strlength(manifestFile) == 0 || ~isfile(manifestFile)
            continue;
        end
        Manifest = readtable(manifestFile,'TextType','string', ...
            'Delimiter',',','ReadVariableNames',true, ...
            'VariableNamingRule','preserve');
        for j = 1 : height(Manifest)
            metricFile = string(Manifest.("metric_file")(j));
            if strlength(metricFile) == 0 || ~isfile(metricFile)
                continue;
            end
            T = readtable(metricFile,'TextType','string', ...
                'Delimiter',',','ReadVariableNames',true, ...
                'VariableNamingRule','preserve');
            if isempty(T)
                continue;
            end
            T.variant = repmat(string(Summary.variant(i)),height(T),1);
            T.problem_name = repmat(string(Summary.problem(i)),height(T),1);
            T.family = repmat(string(Summary.family(i)),height(T),1);
            T.run_id = repmat(double(Summary.run(i)),height(T),1);
            T.metric_file = repmat(metricFile,height(T),1);
            MetricsAll = [MetricsAll;T]; %#ok<AGROW>
        end
    end
end

function writeDiagnosticMetricSummaries_BDG(MetricsAll,outDir)
    if isempty(MetricsAll)
        writetable(table(),fullfile(outDir,'gan_diagnostic_metrics_by_fe.csv'));
        writetable(table(),fullfile(outDir, ...
            'gan_diagnostic_metrics_by_problem_fe.csv'));
        return;
    end
    numericMask = varfun(@isnumeric,MetricsAll,'OutputFormat','uniform');
    numericVars = string(MetricsAll.Properties.VariableNames(numericMask));
    metricVars = setdiff(numericVars, ...
        ["run","run_id","target_FE","actual_FE","gen"],'stable');
    ByFE = groupsummary(MetricsAll,{'variant','target_FE'}, ...
        'mean',cellstr(metricVars));
    ByProblemFE = groupsummary(MetricsAll, ...
        {'variant','problem_name','target_FE'},'mean',cellstr(metricVars));
    writetable(ByFE,fullfile(outDir,'gan_diagnostic_metrics_by_fe.csv'));
    writetable(ByProblemFE,fullfile(outDir, ...
        'gan_diagnostic_metrics_by_problem_fe.csv'));
end

function Options = normalizeDiagnosticOptions_BDG(Options)
    Default = struct('conditionCount',30,'zPerCondition',10, ...
        'normalN',200,'seed',20260614);
    if isempty(Options)
        Options = Default;
    elseif isstruct(Options)
        names = intersect(fieldnames(Options),fieldnames(Default),'stable');
        for i = 1 : numel(names)
            Default.(names{i}) = Options.(names{i});
        end
        Options = Default;
    else
        error('run_CCMO_GAN_BDG_boundary_diagnostics:BadOptions', ...
            'diagOptions must be a struct or empty.');
    end
    Options.conditionCount = max(0,round(double(Options.conditionCount)));
    Options.zPerCondition = max(0,round(double(Options.zPerCondition)));
    Options.normalN = max(0,round(double(Options.normalN)));
    Options.seed = max(0,round(double(Options.seed)));
end

function targets = normalizeDiagnosticTargets_BDG(targets)
    targets = unique(round(double(targets(:)')),'stable');
    targets = targets(isfinite(targets) & targets > 0);
end

function value = lastDiagnosticValue_BDG(x)
    if isempty(x)
        value = NaN;
    else
        value = double(x(end));
    end
end

function removeDiagnosticAppdata_BDG()
    if isappdata(0,'CCMO_GAN_BDG_ExperimentControl')
        rmappdata(0,'CCMO_GAN_BDG_ExperimentControl');
    end
end

function ensureDiagnosticParallelPool_BDG(workerCount)
    pool = gcp('nocreate');
    if isempty(pool)
        parpool('local',workerCount);
    elseif pool.NumWorkers ~= workerCount
        delete(pool);
        parpool('local',workerCount);
    end
end

function names = defaultDiagnosticProblemList_BDG()
    names = { ...
        'DASCMOP1_BC'; ...
        'DASCMOP2_BC'; ...
        'DASCMOP4_BC'; ...
        'DASCMOP5_BC'; ...
        'LIRCMOP5_BC'; ...
        'LIRCMOP6_BC'; ...
        'LIRCMOP7_BC'; ...
        'LIRCMOP8_BC'; ...
        'LIRCMOP9_BC'; ...
        'LIRCMOP10_BC'};
end

function family = diagnosticProblemFamily_BDG(problemName)
    if startsWith(string(problemName),"DASCMOP")
        family = "DASCMOP_BC";
    elseif startsWith(string(problemName),"LIRCMOP")
        family = "LIRCMOP_BC";
    else
        family = "Other";
    end
end

function text = defaultDText_BDG(D)
    if isempty(D)
        text = 'problem-default';
    else
        text = char(string(D));
    end
end
