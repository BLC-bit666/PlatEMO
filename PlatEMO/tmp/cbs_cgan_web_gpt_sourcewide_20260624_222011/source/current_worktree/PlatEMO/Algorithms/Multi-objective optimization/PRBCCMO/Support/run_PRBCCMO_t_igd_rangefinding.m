function [ConfigSummary,RawResults,ConfigTable] = run_PRBCCMO_t_igd_rangefinding(runs,workerCount,N,maxFE,outDir,problemNames,customConfigTable)
% Range-finding tuner for PRBCCMO_t using IGD only.

    if nargin < 1 || isempty(runs)
        runs = 5;
    end
    if nargin < 2 || isempty(workerCount)
        workerCount = 10;
    end
    if nargin < 3 || isempty(N)
        N = 100;
    end
    if nargin < 4 || isempty(maxFE)
        maxFE = 200000;
    end
    if nargin < 5 || isempty(outDir)
        rootDir = fileparts(which('platemo'));
        outDir = fullfile(rootDir,'Data','PRBCCMO_t', ...
            ['igd_rangefinding_',char(datetime('now','Format','yyyyMMdd_HHmmss'))]);
    end
    if nargin < 6 || isempty(problemNames)
        problemNames = representativeProblems();
    end
    if nargin < 7 || isempty(customConfigTable)
        ConfigTable = makeConfigTable();
    else
        ConfigTable = normalizeConfigTable(customConfigTable);
    end

    rootDir = fileparts(which('platemo'));
    addpath(genpath(rootDir));
    if ~isfolder(outDir)
        mkdir(outDir);
    end

    problemNames = cellstr(string(problemNames(:)));
    runIds = 1 : runs;
    writetable(ConfigTable,fullfile(outDir,'configs.csv'));

    workerCount = max(1,round(double(workerCount)));
    ensureParallelPool(workerCount,rootDir);

    fprintf('PRBCCMO_t IGD range finding: %d configs, %d problems, %d runs, N=%d, maxFE=%d, workers=%d\n', ...
        height(ConfigTable),numel(problemNames),numel(runIds),N,maxFE,workerCount);
    fprintf('Output: %s\n',outDir);

    TaskCount = numel(problemNames)*numel(runIds);
    AllRows = repmat(emptyRawRow(),height(ConfigTable)*TaskCount,1);
    rowOffset = 0;
    for c = 1 : height(ConfigTable)
        fprintf('[config %02d/%02d] %s\n',c,height(ConfigTable),char(ConfigTable.name(c)));
        Rows = repmat(emptyRawRow(),TaskCount,1);
        Config = ConfigTable(c,:);
        parfor task = 1 : TaskCount
            [pIdx,rIdx] = ind2sub([numel(problemNames),numel(runIds)],task);
            Rows(task) = runOneTask(rootDir,Config,problemNames{pIdx},runIds(rIdx),N,maxFE);
        end
        AllRows(rowOffset+1:rowOffset+TaskCount) = Rows;
        rowOffset = rowOffset + TaskCount;
        RawResults = struct2table(AllRows(1:rowOffset));
        writetable(RawResults,fullfile(outDir,'raw_results.csv'));
        ConfigSummary = summarizeConfigs(RawResults,ConfigTable);
        writetable(ConfigSummary,fullfile(outDir,'config_summary.csv'));
    end

    RawResults = struct2table(AllRows(1:rowOffset));
    ConfigSummary = summarizeConfigs(RawResults,ConfigTable);
    writetable(RawResults,fullfile(outDir,'raw_results.csv'));
    writetable(ConfigSummary,fullfile(outDir,'config_summary.csv'));
end

function problemNames = representativeProblems()
    problemNames = { ...
        'DASCMOP1_BC','DASCMOP3_BC','DASCMOP5_BC','DASCMOP7_BC','DASCMOP9_BC', ...
        'LIRCMOP1_BC','LIRCMOP3_BC','LIRCMOP7_BC','LIRCMOP10_BC','LIRCMOP13_BC'};
end

function ConfigTable = makeConfigTable()
    Baseline = table( ...
        1,"baseline",64,200,0.001,3.0,0.10,10,0, ...
        'VariableNames',configColumnNames());

    rng(20260511,'twister');
    Samples = makeLatinHypercube(24,6);
    hidden = roundToStep(scaleLinear(Samples(:,1),20,160),1);
    epoch = roundToStep(scaleLinear(Samples(:,2),20,200),1);
    lr = scaleLog(Samples(:,3),0.001,0.1);
    betaB = roundToStep(scaleLinear(Samples(:,4),2,12),0.25);
    etaB = roundToStep(scaleLinear(Samples(:,5),0.05,1.0),0.01);
    Tretrain = max(1,roundToStep(scaleLinear(Samples(:,6),5,80),1));
    Gstart = zeros(size(Tretrain));

    ConfigTable = table( ...
        (2:25)',compose("lhs_%02d",(1:24)'), ...
        hidden,epoch,lr,betaB,etaB,Tretrain,Gstart, ...
        'VariableNames',configColumnNames());
    ConfigTable = [Baseline;ConfigTable];
end

function Names = configColumnNames()
    Names = {'config_id','name','hidden','epoch','lr','betaB','etaB','Tretrain','Gstart'};
end

function ConfigTable = normalizeConfigTable(ConfigTable)
    assert(istable(ConfigTable), ...
        'run_PRBCCMO_t_igd_rangefinding:InvalidConfigTable', ...
        'customConfigTable must be a table.');
    Required = configColumnNames();
    Missing = setdiff(Required,ConfigTable.Properties.VariableNames);
    assert(isempty(Missing), ...
        'run_PRBCCMO_t_igd_rangefinding:MissingConfigColumns', ...
        'customConfigTable is missing columns: %s.', strjoin(Missing,', '));

    ConfigTable = ConfigTable(:,Required);
    ConfigTable.name = string(ConfigTable.name);
    NumericNames = setdiff(Required,{'name'});
    for i = 1 : numel(NumericNames)
        ConfigTable.(NumericNames{i}) = double(ConfigTable.(NumericNames{i}));
    end
    assert(numel(unique(ConfigTable.config_id)) == height(ConfigTable), ...
        'run_PRBCCMO_t_igd_rangefinding:DuplicateConfigIds', ...
        'customConfigTable.config_id values must be unique.');
end

function X = makeLatinHypercube(n,d)
    X = zeros(n,d);
    for j = 1 : d
        X(:,j) = (randperm(n)' - rand(n,1))./n;
    end
end

function Value = scaleLinear(U,Low,High)
    Value = Low + U.*(High-Low);
end

function Value = scaleLog(U,Low,High)
    Value = exp(log(Low) + U.*(log(High)-log(Low)));
end

function Value = roundToStep(Value,Step)
    Value = round(Value./Step).*Step;
end

function ensureParallelPool(workerCount,rootDir)
    pool = gcp('nocreate');
    if ~isempty(pool) && pool.NumWorkers ~= workerCount
        delete(pool);
        pool = [];
    end
    if isempty(pool)
        parpool('Processes',workerCount);
    end
    escapedRoot = strrep(rootDir,'''','''''');
    pctRunOnAll(sprintf('addpath(genpath(''%s''));',escapedRoot));
end

function Row = runOneTask(rootDir,Config,problemName,runId,N,maxFE)
    addpath(genpath(rootDir));
    Row = emptyRawRow();
    Row.config_id = double(Config.config_id);
    Row.config_name = string(Config.name);
    Row.problem = string(problemName);
    Row.family = problemFamily(problemName);
    Row.run = double(runId);
    Row.N = double(N);
    Row.maxFE = double(maxFE);
    Row.hidden = double(Config.hidden);
    Row.epoch = double(Config.epoch);
    Row.lr = double(Config.lr);
    Row.betaB = double(Config.betaB);
    Row.etaB = double(Config.etaB);
    Row.Tretrain = double(Config.Tretrain);
    Row.Gstart = double(Config.Gstart);

    try
        rng(double(runId),'twister');
        Algorithm = PRBCCMO_t( ...
            'save',0, ...
            'run',double(runId), ...
            'outputFcn',@(varargin)[], ...
            'parameter',{Row.hidden,Row.epoch,Row.lr,Row.betaB,Row.etaB,Row.Tretrain,Row.Gstart,0,1});
        ProblemConstructor = str2func(problemName);
        Problem = ProblemConstructor('N',N,'maxFE',maxFE);
        Algorithm.Solve(Problem);
        Row.igd = scalarMetric(Algorithm,'IGD');
        Row.runtime = double(Algorithm.metric.runtime);
        Row.analysis_folder = string(Algorithm.metric.analysis_folder);
        Trace = summarize_PRBCCMO_t_run(Algorithm.metric.analysis_folder);
        Row.final_fe = Trace.final_fe;
        Row.final_b_mean_pair_gap = Trace.final_b_mean_pair_gap;
        Row.final_b_p90_pair_gap = Trace.final_b_p90_pair_gap;
        Row.final_b_mean_mid_scalar = Trace.final_b_mean_mid_scalar;
        Row.final_train_size = Trace.final_train_size;
        Row.inf_ranked_total = Trace.inf_ranked_total;
        Row.inf_selected_total = Trace.inf_selected_total;
        Row.status = "ok";
    catch err
        Row.status = "failed";
        Row.error_message = string(err.message);
    end
end

function Value = scalarMetric(Algorithm,metricName)
    Value = NaN;
    try
        Scores = Algorithm.CalMetric(metricName);
        if ~isempty(Scores)
            Value = double(Scores(end));
        end
    catch
        Value = NaN;
    end
end

function Family = problemFamily(problemName)
    if startsWith(string(problemName),"DASCMOP")
        Family = "DASCMOP_BC";
    else
        Family = "LIRCMOP_BC";
    end
end

function Summary = summarizeConfigs(RawResults,ConfigTable)
    Summary = ConfigTable;
    Summary.completed = zeros(height(ConfigTable),1);
    Summary.failed = zeros(height(ConfigTable),1);
    Summary.mean_igd = NaN(height(ConfigTable),1);
    Summary.median_igd = NaN(height(ConfigTable),1);
    Summary.mean_problem_rank = NaN(height(ConfigTable),1);
    Summary.mean_runtime = NaN(height(ConfigTable),1);

    for c = 1 : height(ConfigTable)
        idx = RawResults.config_id == ConfigTable.config_id(c);
        Data = RawResults(idx,:);
        OK = Data.status == "ok" & isfinite(Data.igd);
        Summary.completed(c) = sum(OK);
        Summary.failed(c) = sum(Data.status ~= "ok");
        Summary.mean_igd(c) = PRBCCMOUtils.meanFinite(Data.igd(OK));
        Summary.median_igd(c) = PRBCCMOUtils.medianFinite(Data.igd(OK));
        Summary.mean_runtime(c) = PRBCCMOUtils.meanFinite(Data.runtime(Data.status == "ok"));
    end

    Summary.mean_problem_rank = configMeanProblemRanks(RawResults,ConfigTable.config_id);
    Summary = sortrows(Summary,{'mean_problem_rank','mean_igd','failed'},{'ascend','ascend','ascend'});
end

function MeanRanks = configMeanProblemRanks(RawResults,ConfigIds)
    MeanRanks = NaN(numel(ConfigIds),1);
    Problems = unique(RawResults.problem);
    Runs = unique(RawResults.run);
    MaxRanksPerConfig = max(1,numel(Problems)*numel(Runs));
    Accum = NaN(numel(ConfigIds),MaxRanksPerConfig);
    Counts = zeros(numel(ConfigIds),1);
    for p = 1 : numel(Problems)
        for r = 1 : numel(Runs)
            idx = RawResults.problem == Problems(p) & RawResults.run == Runs(r) & ...
                RawResults.status == "ok" & isfinite(RawResults.igd);
            if ~any(idx)
                continue;
            end
            Config = RawResults.config_id(idx);
            IGD = RawResults.igd(idx);
            [~,ord] = sort(IGD,'ascend');
            Rank = NaN(size(IGD));
            Rank(ord) = 1:numel(ord);
            for i = 1 : numel(Config)
                c = find(ConfigIds == Config(i),1);
                Counts(c) = Counts(c) + 1;
                Accum(c,Counts(c)) = Rank(i);
            end
        end
    end
    for c = 1 : numel(ConfigIds)
        MeanRanks(c) = PRBCCMOUtils.meanFinite(Accum(c,1:Counts(c)));
    end
end

function Row = emptyRawRow()
    Row = struct( ...
        'config_id',NaN, ...
        'config_name',"", ...
        'problem',"", ...
        'family',"", ...
        'run',NaN, ...
        'N',NaN, ...
        'maxFE',NaN, ...
        'hidden',NaN, ...
        'epoch',NaN, ...
        'lr',NaN, ...
        'betaB',NaN, ...
        'etaB',NaN, ...
        'Tretrain',NaN, ...
        'Gstart',NaN, ...
        'igd',NaN, ...
        'runtime',NaN, ...
        'final_fe',NaN, ...
        'final_b_mean_pair_gap',NaN, ...
        'final_b_p90_pair_gap',NaN, ...
        'final_b_mean_mid_scalar',NaN, ...
        'final_train_size',NaN, ...
        'inf_ranked_total',NaN, ...
        'inf_selected_total',NaN, ...
        'analysis_folder',"", ...
        'status',"pending", ...
        'error_message',"");
end
