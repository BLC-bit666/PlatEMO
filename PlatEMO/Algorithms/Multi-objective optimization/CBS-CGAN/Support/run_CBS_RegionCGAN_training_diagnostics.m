function [Summary,EventSummaryAll,TrainHistoryAll,outDir, ...
    FigureManifest,StageSnapshotsAll] = ...
    run_CBS_RegionCGAN_training_diagnostics(outDir,workerCount, ...
    problemNames,N,D,maxFE,runIds,Options)
%RUN_CBS_REGIONCGAN_TRAINING_DIAGNOSTICS Run CGAN D/G training diagnostics.

    rootDir = locateRootDir();
    addpath(genpath(rootDir));
    if nargin < 1 || isempty(outDir)
        outDir = fullfile(rootDir,'Data','CBS_RegionCGAN', ...
            ['training_diagnostics_', ...
            char(datetime('now','Format','yyyyMMdd_HHmmss'))]);
    end
    if nargin < 2 || isempty(workerCount)
        workerCount = 10;
    end
    if nargin < 3 || isempty(problemNames)
        problemNames = ["LIRCMOP5_BC";"LIRCMOP6_BC";"LIRCMOP7_BC"; ...
            "LIRCMOP8_BC";"LIRCMOP9_BC";"LIRCMOP10_BC"];
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
    if nargin < 8 || isempty(Options)
        Options = struct();
    end
    Options = normalizeDiagnosticOptions(Options);

    workerCount = max(1,round(double(workerCount)));
    if ischar(problemNames)
        problemNames = string(problemNames);
    else
        problemNames = string(problemNames(:));
    end
    runIds = double(runIds(:)');
    N = max(1,round(double(N)));
    maxFE = max(1,round(double(maxFE)));

    if ~isfolder(outDir)
        mkdir(outDir);
    end
    Tasks = buildDiagnosticTasks(problemNames,runIds);
    taskCount = height(Tasks);
    taskProblems = Tasks.problem;
    taskRuns = Tasks.run;
    Rows = repmat(emptyRunRow(),taskCount,1);

    if workerCount > 1
        ensureDiagnosticParallelPool(workerCount);
        doneCount = 0;
        progressQueue = parallel.pool.DataQueue;
        afterEach(progressQueue,@reportProgress);
        parfor task = 1 : taskCount
            Rows(task) = runOneDiagnosticTask(taskProblems(task), ...
                taskRuns(task),outDir,N,D,maxFE,Options);
            send(progressQueue,Rows(task));
        end
    else
        for task = 1 : taskCount
            Rows(task) = runOneDiagnosticTask(taskProblems(task), ...
                taskRuns(task),outDir,N,D,maxFE,Options);
            fprintf('[%d/%d] %s run=%d status=%s\n',task,taskCount, ...
                char(Rows(task).problem),Rows(task).run,char(Rows(task).status));
        end
    end

    Summary = struct2table(Rows);
    writetable(Summary,fullfile(outDir,'run_summary.csv'));
    EventSummaryAll = collectDiagnosticTables( ...
        Summary.event_summary_file,emptyEventRow());
    writetable(EventSummaryAll,fullfile(outDir,'event_summary_all.csv'));
    TrainHistoryAll = collectDiagnosticTables( ...
        Summary.train_history_file,emptyTrainRow());
    writetable(TrainHistoryAll,fullfile(outDir,'train_history_all.csv'));
    StageSnapshotsAll = collectDiagnosticTables( ...
        Summary.stage_snapshot_file,emptyStageSnapshotRow());
    writetable(StageSnapshotsAll,fullfile(outDir,'stage_snapshots_all.csv'));
    FigureManifest = collectDiagnosticTables( ...
        Summary.figure_manifest_file,emptyFigureRow());
    writetable(FigureManifest,fullfile(outDir,'figure_manifest.csv'));

    function reportProgress(Row)
        doneCount = doneCount + 1;
        fprintf('[%d/%d] %s run=%d status=%s\n',doneCount,taskCount, ...
            char(Row.problem),Row.run,char(Row.status));
    end
end

function rootDir = locateRootDir()
    rootDir = fileparts(which('platemo'));
    if isempty(rootDir)
        rootDir = pwd;
    end
end

function Options = normalizeDiagnosticOptions(Options)
    if ~isfield(Options,'algorithmClass') || isempty(Options.algorithmClass)
        Options.algorithmClass = "CBS_RegionCGAN";
    end
    Options.algorithmClass = string(Options.algorithmClass);
    Defaults = regionDiagnosticAlgorithmDefaults(Options.algorithmClass);
    if ~isfield(Options,'algorithmParams')
        Options.algorithmParams = {};
    end
    if ~isfield(Options,'queryMode') || isempty(Options.queryMode)
        Options.queryMode = Defaults.queryMode;
    end
    if ~isfield(Options,'conditionMode') || isempty(Options.conditionMode)
        Options.conditionMode = Defaults.conditionMode;
    end
    if ~isfield(Options,'prevBMemMode') || isempty(Options.prevBMemMode)
        Options.prevBMemMode = Defaults.prevBMemMode;
    end
    if ~isfield(Options,'bmemBandMode') || isempty(Options.bmemBandMode)
        Options.bmemBandMode = Defaults.bmemBandMode;
    end
    if ~isfield(Options,'bandMaxAnchorsPerRef')
        Options.bandMaxAnchorsPerRef = Defaults.bandMaxAnchorsPerRef;
    end
    if ~isfield(Options,'plotRun') || isempty(Options.plotRun)
        Options.plotRun = 0;
    end
    if ~isfield(Options,'captureRun') || isempty(Options.captureRun)
        Options.captureRun = Options.plotRun;
    end
    if ~isfield(Options,'drawFigures') || isempty(Options.drawFigures)
        Options.drawFigures = Options.plotRun > 0;
    end
    if ~isfield(Options,'stageTargets') || isempty(Options.stageTargets)
        Options.stageTargets = [];
    end
    if ~isfield(Options,'sampleZMode')
        Options.sampleZMode = [];
    end
    if ~isfield(Options,'trainZMode')
        Options.trainZMode = [];
    end
    if ~isfield(Options,'sigma')
        Options.sigma = [];
    end
    if ~isfield(Options,'trainSigma')
        Options.trainSigma = [];
    end
    if ~isfield(Options,'sampleSigma')
        Options.sampleSigma = [];
    end
    if ~isfield(Options,'ganIterSchedule')
        Options.ganIterSchedule = [];
    end
    if ~isfield(Options,'ganIterStart')
        Options.ganIterStart = [];
    end
    if ~isfield(Options,'ganIterEnd')
        Options.ganIterEnd = [];
    end
    if ~isfield(Options,'prescreenMultiplier')
        Options.prescreenMultiplier = [];
    end
    if ~isfield(Options,'trainTriggerMode')
        Options.trainTriggerMode = [];
    end
    if ~isfield(Options,'trainTriggerDelta')
        Options.trainTriggerDelta = [];
    end
    Options.queryMode = lower(strtrim(string(Options.queryMode)));
    Options.conditionMode = lower(strtrim(string(Options.conditionMode)));
    Options.prevBMemMode = lower(strtrim(string(Options.prevBMemMode)));
    Options.bmemBandMode = lower(strtrim(string(Options.bmemBandMode)));
    if ~isempty(Options.bandMaxAnchorsPerRef)
        Options.bandMaxAnchorsPerRef = ...
            max(1,round(double(Options.bandMaxAnchorsPerRef)));
    end
    if ~isempty(Options.sampleZMode)
        Options.sampleZMode = lower(strtrim(string(Options.sampleZMode)));
    end
    if ~isempty(Options.trainZMode)
        Options.trainZMode = lower(strtrim(string(Options.trainZMode)));
    end
    if ~isempty(Options.sigma)
        Options.sigma = double(Options.sigma);
    end
    if ~isempty(Options.trainSigma)
        Options.trainSigma = double(Options.trainSigma);
    end
    if ~isempty(Options.sampleSigma)
        Options.sampleSigma = double(Options.sampleSigma);
    end
    if ~isempty(Options.ganIterSchedule)
        Options.ganIterSchedule = lower(strtrim(string( ...
            Options.ganIterSchedule)));
    end
    if ~isempty(Options.ganIterStart)
        Options.ganIterStart = double(Options.ganIterStart);
    end
    if ~isempty(Options.ganIterEnd)
        Options.ganIterEnd = double(Options.ganIterEnd);
    end
    if ~isempty(Options.prescreenMultiplier)
        Options.prescreenMultiplier = max(1,round(double( ...
            Options.prescreenMultiplier)));
    end
    if ~isempty(Options.trainTriggerMode)
        Options.trainTriggerMode = lower(strtrim(string( ...
            Options.trainTriggerMode)));
    end
    if ~isempty(Options.trainTriggerDelta)
        Options.trainTriggerDelta = double(Options.trainTriggerDelta);
    end
    Options.plotRun = round(double(Options.plotRun));
    Options.captureRun = round(double(Options.captureRun));
    Options.drawFigures = logical(Options.drawFigures);
    Options.stageTargets = double(Options.stageTargets(:)');
end

function Defaults = regionDiagnosticAlgorithmDefaults(algorithmClass)
    Defaults = struct( ...
        'queryMode',"boundary_populated", ...
        'conditionMode',"region", ...
        'prevBMemMode',"current_only", ...
        'bmemBandMode',"current", ...
        'bandMaxAnchorsPerRef',[]);
    if string(algorithmClass) == "CBS_RegionWGAN_GP"
        WGANDefaults = CBS_RegionWGAN_GP.mainlineDefaults();
        Defaults.queryMode = WGANDefaults.queryMode;
        Defaults.prevBMemMode = WGANDefaults.prevBMemMode;
        Defaults.bmemBandMode = WGANDefaults.bmemBandMode;
        Defaults.bandMaxAnchorsPerRef = WGANDefaults.bandMaxAnchorsPerRef;
    end
end

function Tasks = buildDiagnosticTasks(problemNames,runIds)
    Rows = repmat(struct('problem',"",'run',NaN), ...
        numel(problemNames)*numel(runIds),1);
    row = 0;
    for p = 1 : numel(problemNames)
        for r = 1 : numel(runIds)
            row = row + 1;
            Rows(row).problem = problemNames(p);
            Rows(row).run = runIds(r);
        end
    end
    Tasks = struct2table(Rows);
end

function Row = runOneDiagnosticTask(problemName,runId,outDir,N,D,maxFE,Options)
    Row = emptyRunRow();
    Row.problem = string(problemName);
    Row.run = double(runId);
    Row.seed = double(runId);
    Row.N = double(N);
    Row.maxFE = double(maxFE);
    try
        rootDir = locateRootDir();
        addpath(genpath(rootDir));
        try
            maxNumCompThreads(1);
        catch
        end
        rng(Row.seed,'twister');
        ProblemConstructor = str2func(char(Row.problem));
        if isempty(D)
            Problem = ProblemConstructor('N',N,'maxFE',maxFE);
        else
            Problem = ProblemConstructor('N',N,'D',D,'maxFE',maxFE);
        end
        Row.D = double(Problem.D);
        runFolder = fullfile(outDir,sprintf('%s_run%d', ...
            char(Row.problem),round(Row.run)));
        if ~isfolder(runFolder)
            mkdir(runFolder);
        end
        Row.run_folder = string(runFolder);
        Row.metric_file = string(fullfile(runFolder,'metric.mat'));
        Row.event_summary_file = string(fullfile(runFolder, ...
            'event_summary.csv'));
        Row.train_history_file = string(fullfile(runFolder, ...
            'train_history.csv'));
        Row.stage_snapshot_file = string(fullfile(runFolder, ...
            'stage_snapshots.csv'));
        Row.figure_manifest_file = string(fullfile(runFolder, ...
            'figure_manifest.csv'));

        AlgorithmConstructor = str2func(char(Options.algorithmClass));
        Control = struct( ...
            'queryMode',Options.queryMode, ...
            'conditionMode',Options.conditionMode, ...
            'prevBMemMode',Options.prevBMemMode, ...
            'bmemBandMode',Options.bmemBandMode, ...
            'bandMaxAnchorsPerRef',Options.bandMaxAnchorsPerRef, ...
            'stageTargets',[], ...
            'sampleZMode',Options.sampleZMode, ...
            'trainZMode',Options.trainZMode, ...
            'sigma',Options.sigma, ...
            'trainSigma',Options.trainSigma, ...
            'sampleSigma',Options.sampleSigma, ...
            'ganIterSchedule',Options.ganIterSchedule, ...
            'ganIterStart',Options.ganIterStart, ...
            'ganIterEnd',Options.ganIterEnd, ...
            'prescreenMultiplier',Options.prescreenMultiplier, ...
            'trainTriggerMode',Options.trainTriggerMode, ...
            'trainTriggerDelta',Options.trainTriggerDelta);
        if Row.run == Options.captureRun
            Control.stageTargets = Options.stageTargets;
        end
        setappdata(0,'CBS_RegionGAN_ExperimentControl',Control);
        cleanupControl = onCleanup(@()removeRegionGANExperimentControl());
        if isempty(Options.algorithmParams)
            Algorithm = AlgorithmConstructor('save',0,'run',Row.run, ...
                'outputFcn',@(varargin)[]);
        else
            Algorithm = AlgorithmConstructor('save',0,'run',Row.run, ...
                'outputFcn',@(varargin)[], ...
                'parameter',Options.algorithmParams);
        end
        Algorithm.Solve(Problem);
        clear cleanupControl

        Row.runtime = double(Algorithm.metric.runtime);
        Row.finalFE = double(Algorithm.result{end,1});
        metric = Algorithm.metric;
        problem = Row.problem;
        run = Row.run;
        seed = Row.seed;
        save(Row.metric_file,'metric','problem','run','seed','N','maxFE');

        if isfield(Algorithm.metric,'region_gan_history')
            History = Algorithm.metric.region_gan_history;
        elseif isfield(Algorithm.metric,'region_wgan_gp_history')
            History = Algorithm.metric.region_wgan_gp_history;
        elseif isfield(Algorithm.metric,'region_cgan_history')
            History = Algorithm.metric.region_cgan_history;
        else
            History = struct([]);
        end
        [EventSummary,TrainHistory] = buildDiagnosticTables(History,Row);
        writetable(EventSummary,Row.event_summary_file);
        writetable(TrainHistory,Row.train_history_file);
        Row.event_count = height(EventSummary);
        Row.train_row_count = height(TrainHistory);
        if isfield(Algorithm.metric,'region_gan_stage_snapshots')
            Snapshots = Algorithm.metric.region_gan_stage_snapshots;
        else
            Snapshots = emptyRegionStageSnapshotStruct();
        end
        StageSnapshots = regionStageSnapshotTable(Snapshots,Row);
        writetable(StageSnapshots,Row.stage_snapshot_file);
        if Options.drawFigures && Row.run == Options.captureRun
            Figures = plotRegionStageFigures(Problem,Snapshots, ...
                Row.problem,Row.run,runFolder);
        else
            Figures = struct2table(repmat(emptyFigureRow(),0,1));
        end
        writetable(Figures,Row.figure_manifest_file);
        Row.figure_count = height(Figures);
        Row.status = "ok";
    catch err
        removeRegionGANExperimentControl();
        Row.status = "failed";
        Row.error_message = string(getReport(err,'extended', ...
            'hyperlinks','off'));
    end
end

function removeRegionGANExperimentControl()
    if isappdata(0,'CBS_RegionGAN_ExperimentControl')
        rmappdata(0,'CBS_RegionGAN_ExperimentControl');
    end
end

function [EventSummary,TrainHistory] = buildDiagnosticTables(History,RunRow)
    trainRowTotal = 0;
    eventRowTotal = 0;
    for eventIndex = 1 : numel(History)
        Event = History(eventIndex);
        if isfield(Event,'generation') && ~isempty(Event.generation)
            eventRowTotal = eventRowTotal + 1;
        end
        if isfield(Event,'gan_train_history') && ...
                ~isempty(Event.gan_train_history)
            trainRowTotal = trainRowTotal + numel(Event.gan_train_history);
        end
    end
    EventRows = repmat(emptyEventRow(),eventRowTotal,1);
    TrainRows = repmat(emptyTrainRow(),trainRowTotal,1);
    eventRow = 0;
    trainRow = 0;
    for eventIndex = 1 : numel(History)
        Event = History(eventIndex);
        H = struct([]);
        if ~isfield(Event,'gan_train_history') || ...
                isempty(Event.gan_train_history)
            if ~isfield(Event,'generation') || isempty(Event.generation)
                continue;
            end
        else
            H = Event.gan_train_history;
        end
        eventRow = eventRow + 1;
        EventRows(eventRow,1) = summarizeTrainingEvent( ...
            H,Event,RunRow,eventIndex);
        for h = 1 : numel(H)
            trainRow = trainRow + 1;
            TrainRows(trainRow,1) = summarizeTrainingStep( ...
                H(h),Event,RunRow,eventIndex);
        end
    end
    EventRows = EventRows(1:eventRow);
    TrainRows = TrainRows(1:trainRow);
    EventSummary = struct2table(EventRows);
    TrainHistory = struct2table(TrainRows);
end

function Row = summarizeTrainingEvent(H,Event,RunRow,eventIndex)
    Row = emptyEventRow();
    Row.problem = RunRow.problem;
    Row.run = RunRow.run;
    Row.seed = RunRow.seed;
    Row.event_index = double(eventIndex);
    Row.generation = getNumericField(Event,'generation');
    Row.train_count = getNumericField(Event,'train_count');
    Row.condition_mode = getStringField(Event,'condition_mode');
    Row.query_mode = getStringField(Event,'query_mode');
    Row.gan_iter_used = getNumericField(Event,'gan_iter_used');
    Row.gan_iter_schedule = getStringField(Event,'gan_iter_schedule');
    Row.train_trigger_mode = getStringField(Event,'train_trigger_mode');
    Row.trigger_reason = getStringField(Event,'trigger_reason');
    Row.trigger_train_now = getNumericField(Event,'trigger_train_now');
    Row.trigger_train_count_delta = getNumericField(Event, ...
        'trigger_train_count_delta');
    Row.sample_z_mode = getStringField(Event,'sample_z_mode');
    Row.train_z_mode = getStringField(Event,'train_z_mode');
    Row.train_z_sigma = getNumericField(Event,'train_z_sigma');
    Row.sample_z_sigma = getNumericField(Event,'sample_z_sigma');
    Row.prescreen_multiplier = getNumericField(Event, ...
        'prescreen_multiplier');
    Row.prescreen_candidate_count = getNumericField(Event, ...
        'prescreen_candidate_count');
    Row.prescreen_selected_count = getNumericField(Event, ...
        'prescreen_selected_count');
    Row.prescreen_score_min = getNumericField(Event, ...
        'prescreen_score_min');
    Row.prescreen_score_max = getNumericField(Event, ...
        'prescreen_score_max');
    Row.prescreen_score_mean = getNumericField(Event, ...
        'prescreen_score_mean');
    Row.query_count = getNumericField(Event,'query_count');
    Row.query_sample_count = getNumericField(Event,'query_sample_count');
    Row.query_unique_ref_count = getNumericField(Event, ...
        'query_unique_ref_count');
    Row.raw_generated_count = getNumericField(Event,'raw_generated_count');
    Row.feasible_rate = getNumericField(Event,'feasible_rate');
    Row.history_rows = double(numel(H));
    if isempty(H)
        Row.final_loss_d = firstFiniteField(Event, ...
            {'last_discriminator_loss','last_critic_loss'});
        Row.final_loss_g = getNumericField(Event,'last_generator_loss');
        Row.final_score_real_mean = getNumericField(Event,'last_score_real');
        Row.final_score_fake_mean = getNumericField(Event,'last_score_fake');
        Row.final_score_random_mean = getNumericField(Event, ...
            'last_score_random');
    else
        First = H(1);
        Last = H(end);
        Row.start_d_bal_acc = double(First.d_bal_acc);
        Row.final_d_bal_acc = double(Last.d_bal_acc);
        Row.delta_d_bal_acc = Row.final_d_bal_acc - Row.start_d_bal_acc;
        Row.start_d_fake_acc = double(First.d_fake_acc);
        Row.final_d_fake_acc = double(Last.d_fake_acc);
        Row.delta_d_fake_acc = Row.final_d_fake_acc - Row.start_d_fake_acc;
        Row.start_g_fool_rate = double(First.g_fool_rate);
        Row.final_g_fool_rate = double(Last.g_fool_rate);
        Row.delta_g_fool_rate = Row.final_g_fool_rate - Row.start_g_fool_rate;
        Row.final_loss_d = double(Last.loss_d);
        Row.final_loss_g = double(Last.loss_g);
        Row.final_score_real_mean = double(Last.score_real_mean);
        Row.final_score_fake_mean = double(Last.score_fake_mean);
        Row.final_score_random_mean = getHistoryField(Last, ...
            'score_random_mean');
    end
    Row.score_gap = Row.final_score_real_mean - Row.final_score_fake_mean;
    Row.real_random_gap = Row.final_score_real_mean - ...
        Row.final_score_random_mean;
    Row.fake_random_gap = Row.final_score_fake_mean - ...
        Row.final_score_random_mean;
    boundaryFields = generatedBoundaryDiagnosticFields();
    for i = 1 : numel(boundaryFields)
        Row.(boundaryFields{i}) = getNumericField(Event,boundaryFields{i});
    end
    diagFields = conditionDiagnosticFields();
    for i = 1 : numel(diagFields)
        Row.(diagFields{i}) = getNumericField(Event,diagFields{i});
    end
    fixFields = fixmdDiagnosticFields();
    for i = 1 : numel(fixFields)
        Row.(fixFields{i}) = getNumericField(Event,fixFields{i});
    end
end

function value = firstFiniteField(S,names)
    value = NaN;
    for i = 1 : numel(names)
        candidate = getNumericField(S,names{i});
        if isfinite(candidate)
            value = candidate;
            return;
        end
    end
end

function Row = summarizeTrainingStep(H,Event,RunRow,eventIndex)
    Row = emptyTrainRow();
    Row.problem = RunRow.problem;
    Row.run = RunRow.run;
    Row.seed = RunRow.seed;
    Row.event_index = double(eventIndex);
    Row.generation = getNumericField(Event,'generation');
    Row.step = double(H.step);
    Row.loss_d = double(H.loss_d);
    Row.loss_g = double(H.loss_g);
    Row.d_real_acc = double(H.d_real_acc);
    Row.d_fake_acc = double(H.d_fake_acc);
    Row.d_bal_acc = double(H.d_bal_acc);
    Row.g_fool_rate = double(H.g_fool_rate);
    Row.score_real_mean = double(H.score_real_mean);
    Row.score_fake_mean = double(H.score_fake_mean);
    Row.score_random_mean = getHistoryField(H,'score_random_mean');
    Row.random_as_fake_rate = getHistoryField(H,'random_as_fake_rate');
end

function value = getNumericField(S,name)
    if isstruct(S) && isfield(S,name) && ~isempty(S.(name))
        value = double(S.(name));
    else
        value = NaN;
    end
end

function value = getHistoryField(S,name)
    if isstruct(S) && isfield(S,name) && ~isempty(S.(name))
        value = double(S.(name));
    else
        value = NaN;
    end
end

function value = getStringField(S,name)
    if isstruct(S) && isfield(S,name) && ~isempty(S.(name))
        value = string(S.(name));
    else
        value = "";
    end
end

function T = collectDiagnosticTables(files,emptyRow)
    files = string(files(:));
    Schema = struct2table(repmat(emptyRow,0,1));
    Tables = {};
    for i = 1 : numel(files)
        if strlength(files(i)) > 0 && isfile(files(i))
            Ti = readtable(files(i),'TextType','string', ...
                'Delimiter',',','ReadVariableNames',true, ...
                'VariableNamingRule','preserve');
            Tables{end+1,1} = alignTableToSchema(Ti,Schema,emptyRow); %#ok<AGROW>
        end
    end
    if isempty(Tables)
        T = Schema;
    else
        T = vertcat(Tables{:});
    end
end

function T = alignTableToSchema(T,Schema,emptyRow)
    schemaNames = string(Schema.Properties.VariableNames);
    if height(T) == 0
        T = Schema;
        return;
    end
    currentNames = string(T.Properties.VariableNames);
    for i = 1 : numel(schemaNames)
        name = schemaNames(i);
        if ~ismember(name,currentNames)
            T.(name) = defaultTableColumn(emptyRow.(name),height(T));
        end
    end
    extra = setdiff(string(T.Properties.VariableNames),schemaNames, ...
        'stable');
    if ~isempty(extra)
        T(:,cellstr(extra)) = [];
    end
    T = T(:,cellstr(schemaNames));
end

function col = defaultTableColumn(value,n)
    if isstring(value)
        col = repmat(string(value),n,1);
    elseif ischar(value)
        col = repmat(string(value),n,1);
    elseif islogical(value)
        col = false(n,1);
    else
        col = NaN(n,1);
    end
end

function ensureDiagnosticParallelPool(workerCount)
    pool = gcp('nocreate');
    if ~isempty(pool) && pool.NumWorkers ~= workerCount
        delete(pool);
        pool = [];
    end
    if isempty(pool)
        parpool('local',workerCount);
    end
end

function Row = emptyRunRow()
    Row = struct( ...
        'problem',"", ...
        'run',NaN, ...
        'seed',NaN, ...
        'N',NaN, ...
        'D',NaN, ...
        'maxFE',NaN, ...
        'finalFE',NaN, ...
        'runtime',NaN, ...
        'event_count',0, ...
        'train_row_count',0, ...
        'status',"pending", ...
        'error_message',"", ...
        'run_folder',"", ...
        'metric_file',"", ...
        'event_summary_file',"", ...
        'train_history_file',"", ...
        'stage_snapshot_file',"", ...
        'figure_manifest_file',"", ...
        'figure_count',0);
end

function Row = emptyEventRow()
    Row = struct( ...
        'problem',"", ...
        'run',NaN, ...
        'seed',NaN, ...
        'event_index',NaN, ...
        'generation',NaN, ...
        'train_count',NaN, ...
        'condition_mode',"", ...
        'query_mode',"", ...
        'gan_iter_used',NaN, ...
        'gan_iter_schedule',"", ...
        'train_trigger_mode',"", ...
        'trigger_reason',"", ...
        'trigger_train_now',NaN, ...
        'trigger_train_count_delta',NaN, ...
        'sample_z_mode',"", ...
        'train_z_mode',"", ...
        'train_z_sigma',NaN, ...
        'sample_z_sigma',NaN, ...
        'prescreen_multiplier',NaN, ...
        'prescreen_candidate_count',NaN, ...
        'prescreen_selected_count',NaN, ...
        'prescreen_score_min',NaN, ...
        'prescreen_score_max',NaN, ...
        'prescreen_score_mean',NaN, ...
        'query_count',NaN, ...
        'query_sample_count',NaN, ...
        'query_unique_ref_count',NaN, ...
        'raw_generated_count',NaN, ...
        'feasible_rate',NaN, ...
        'history_rows',NaN, ...
        'start_d_bal_acc',NaN, ...
        'final_d_bal_acc',NaN, ...
        'delta_d_bal_acc',NaN, ...
        'start_d_fake_acc',NaN, ...
        'final_d_fake_acc',NaN, ...
        'delta_d_fake_acc',NaN, ...
        'start_g_fool_rate',NaN, ...
        'final_g_fool_rate',NaN, ...
        'delta_g_fool_rate',NaN, ...
        'final_loss_d',NaN, ...
        'final_loss_g',NaN, ...
        'final_score_real_mean',NaN, ...
        'final_score_fake_mean',NaN, ...
        'final_score_random_mean',NaN, ...
        'score_gap',NaN, ...
        'real_random_gap',NaN, ...
        'fake_random_gap',NaN);
    boundaryFields = generatedBoundaryDiagnosticFields();
    for i = 1 : numel(boundaryFields)
        Row.(boundaryFields{i}) = NaN;
    end
    fields = conditionDiagnosticFields();
    for i = 1 : numel(fields)
        Row.(fields{i}) = NaN;
    end
    fields = fixmdDiagnosticFields();
    for i = 1 : numel(fields)
        Row.(fields{i}) = NaN;
    end
end

function Row = emptyTrainRow()
    Row = struct( ...
        'problem',"", ...
        'run',NaN, ...
        'seed',NaN, ...
        'event_index',NaN, ...
        'generation',NaN, ...
        'step',NaN, ...
        'loss_d',NaN, ...
        'loss_g',NaN, ...
        'd_real_acc',NaN, ...
        'd_fake_acc',NaN, ...
        'd_bal_acc',NaN, ...
        'g_fool_rate',NaN, ...
        'score_real_mean',NaN, ...
        'score_fake_mean',NaN, ...
        'score_random_mean',NaN, ...
        'random_as_fake_rate',NaN);
end

function fields = generatedBoundaryDiagnosticFields()
    fields = {'dist_to_bmem50', ...
        'dist_to_bmem90', ...
        'dist_to_target_pm2_bmem50', ...
        'dist_to_target_pm2_bmem90', ...
        'segment_dist50', ...
        'segment_dist90', ...
        'gap_ratio50', ...
        'gap_ratio90', ...
        'near_boundary_rate_gap1', ...
        'near_boundary_feasible_rate_gap1', ...
        'random_gap_ratio50', ...
        'random_gap_ratio90', ...
        'random_near_boundary_rate_gap1', ...
        'random_feasible_rate', ...
        'better_than_random_gap_rate', ...
        'bdist50_true', ...
        'bwidth90_10_true', ...
        'bcover_eps_true'};
end

function fields = fixmdDiagnosticFields()
    fields = {'train_width50', ...
        'train_width90', ...
        'train_width_count', ...
        'gen_width50', ...
        'gen_width90', ...
        'gen_width_count', ...
        'gen_to_train_dist50', ...
        'gen_to_train_dist90', ...
        'gen_to_train_dist_count', ...
        'gen_to_train_dec_dist50', ...
        'gen_to_train_dec_dist90', ...
        'offspringG_count', ...
        'offspringG_survive_count', ...
        'offspringG_survival_rate', ...
        'offspringG_feasible_count', ...
        'offspringG_feasible_survive_count', ...
        'offspringG_feasible_survival_rate', ...
        'critic_train_gap', ...
        'critic_holdout_gap', ...
        'critic_train_real_score_mean', ...
        'critic_train_fake_score_mean', ...
        'critic_holdout_real_score_mean', ...
        'critic_holdout_fake_score_mean', ...
        'critic_train_diag_count', ...
        'critic_holdout_count', ...
        'generated_critic_score_count', ...
        'generated_critic_score_min', ...
        'generated_critic_score_max', ...
        'generated_critic_score_mean'};
end

function fields = conditionDiagnosticFields()
    fields = {'condition_diag_condition_count', ...
        'condition_diag_z_count', ...
        'same_z_diff_c_dec_median', ...
        'same_z_diff_c_obj_median', ...
        'same_z_diff_c_ref_unique_rate', ...
        'same_c_diff_z_dec_median', ...
        'same_c_diff_z_obj_median', ...
        'same_c_diff_z_ref_leak_rate', ...
        'same_c_diff_z_collapse_rate', ...
        'condition_effect_ratio_dec', ...
        'condition_effect_ratio_obj', ...
        'query_generated_count', ...
        'query_exact_ref_match_rate', ...
        'query_neighbor_ref_match_rate', ...
        'query_target_ref_rank_median', ...
        'query_target_ref_rank_mean', ...
        'query_feasible_rate_probe', ...
        'query_shuffled_exact_match_rate', ...
        'all_w_condition_count', ...
        'all_w_z_per_ref', ...
        'all_w_query_generated_count', ...
        'all_w_exact_ref_match_rate', ...
        'all_w_pm2_ref_match_rate', ...
        'all_w_shuffled_pm2_ref_match_rate', ...
        'all_w_seen_count', ...
        'all_w_unseen_count', ...
        'all_w_seen_pm2_ref_match_rate', ...
        'all_w_unseen_pm2_ref_match_rate', ...
        'all_w_target_ref_rank_median', ...
        'all_w_target_ref_rank_mean', ...
        'all_w_feasible_rate_probe'};
end

function T = regionStageSnapshotTable(Snapshots,RunRow)
    if isempty(Snapshots)
        T = struct2table(repmat(emptyStageSnapshotRow(),0,1));
        return;
    end
    Rows = repmat(emptyStageSnapshotRow(),numel(Snapshots),1);
    for i = 1 : numel(Snapshots)
        S = Snapshots(i);
        Rows(i).problem = RunRow.problem;
        Rows(i).run = RunRow.run;
        Rows(i).seed = RunRow.seed;
        Rows(i).target_FE = getSnapshotNumeric(S,'target_FE');
        Rows(i).actual_FE = getSnapshotNumeric(S,'actual_FE');
        Rows(i).generation = getSnapshotNumeric(S,'generation');
        Rows(i).condition_mode = getSnapshotString(S,'condition_mode');
        Rows(i).query_mode = getSnapshotString(S,'query_mode');
        Rows(i).gan_iter_used = getSnapshotNumeric(S,'gan_iter_used');
        Rows(i).gan_iter_schedule = getSnapshotString(S, ...
            'gan_iter_schedule');
        Rows(i).train_trigger_mode = getSnapshotString(S, ...
            'train_trigger_mode');
        Rows(i).trigger_reason = getSnapshotString(S,'trigger_reason');
        Rows(i).trigger_train_now = getSnapshotNumeric(S, ...
            'trigger_train_now');
        Rows(i).trigger_train_count_delta = getSnapshotNumeric(S, ...
            'trigger_train_count_delta');
        Rows(i).sample_z_mode = getSnapshotString(S,'sample_z_mode');
        Rows(i).train_z_mode = getSnapshotString(S,'train_z_mode');
        Rows(i).train_z_sigma = getSnapshotNumeric(S, ...
            'train_z_sigma');
        Rows(i).sample_z_sigma = getSnapshotNumeric(S, ...
            'sample_z_sigma');
        Rows(i).prescreen_multiplier = getSnapshotNumeric(S, ...
            'prescreen_multiplier');
        Rows(i).prescreen_candidate_count = getSnapshotNumeric(S, ...
            'prescreen_candidate_count');
        Rows(i).prescreen_selected_count = getSnapshotNumeric(S, ...
            'prescreen_selected_count');
        Rows(i).train_count = getSnapshotNumeric(S,'train_count');
        Rows(i).prev_bmem_candidate_count = getSnapshotNumeric(S, ...
            'prev_bmem_candidate_count');
        Rows(i).prev_bmem_survivor_count = getSnapshotNumeric(S, ...
            'prev_bmem_survivor_count');
        Rows(i).raw_generated_count = getSnapshotNumeric(S, ...
            'raw_generated_count');
        Rows(i).feasible_rate = getSnapshotNumeric(S,'feasible_rate');
        Rows(i).gap_ratio50 = getSnapshotNumeric(S,'gap_ratio50');
        Rows(i).gap_ratio90 = getSnapshotNumeric(S,'gap_ratio90');
        Rows(i).near_boundary_rate_gap1 = getSnapshotNumeric(S, ...
            'near_boundary_rate_gap1');
        Rows(i).better_than_random_gap_rate = getSnapshotNumeric(S, ...
            'better_than_random_gap_rate');
        Rows(i).bdist50_true = getSnapshotNumeric(S,'bdist50_true');
        Rows(i).bwidth90_10_true = getSnapshotNumeric(S, ...
            'bwidth90_10_true');
        Rows(i).bcover_eps_true = getSnapshotNumeric(S, ...
            'bcover_eps_true');
        Rows(i).train_width50 = getSnapshotNumeric(S,'train_width50');
        Rows(i).train_width90 = getSnapshotNumeric(S,'train_width90');
        Rows(i).gen_width50 = getSnapshotNumeric(S,'gen_width50');
        Rows(i).gen_width90 = getSnapshotNumeric(S,'gen_width90');
        Rows(i).gen_to_train_dist50 = getSnapshotNumeric(S, ...
            'gen_to_train_dist50');
        Rows(i).gen_to_train_dist90 = getSnapshotNumeric(S, ...
            'gen_to_train_dist90');
        Rows(i).offspringG_survival_rate = getSnapshotNumeric(S, ...
            'offspringG_survival_rate');
        Rows(i).critic_train_gap = getSnapshotNumeric(S, ...
            'critic_train_gap');
        Rows(i).critic_holdout_gap = getSnapshotNumeric(S, ...
            'critic_holdout_gap');
        Rows(i).generated_critic_score_mean = getSnapshotNumeric(S, ...
            'generated_critic_score_mean');
        Rows(i).sample_feasible_count = sum(logicalVectorField(S, ...
            'sample_feasible'));
        Rows(i).sample_infeasible_count = sizeField(S,'sample_objs') - ...
            Rows(i).sample_feasible_count;
    end
    T = struct2table(Rows);
end

function Figures = plotRegionStageFigures(Problem,Snapshots,problemName, ...
        runId,runFolder)
    if isempty(Snapshots)
        Figures = struct2table(repmat(emptyFigureRow(),0,1));
        return;
    end
    figureFolder = fullfile(runFolder,'figures');
    if ~isfolder(figureFolder)
        mkdir(figureFolder);
    end
    Rows = repmat(emptyFigureRow(),numel(Snapshots),1);
    for i = 1 : numel(Snapshots)
        S = Snapshots(i);
        figureFile = fullfile(figureFolder,sprintf( ...
            '%s_run%d_FE%06d_domain_boundary.png',char(problemName), ...
            round(runId),round(S.target_FE)));
        plotSingleRegionStageFigure(Problem,S,problemName,runId, ...
            figureFile);
        Rows(i).problem = string(problemName);
        Rows(i).run = double(runId);
        Rows(i).target_FE = double(S.target_FE);
        Rows(i).actual_FE = double(S.actual_FE);
        Rows(i).generation = double(S.generation);
        Rows(i).figure_file = string(figureFile);
    end
    Figures = struct2table(Rows);
end

function plotSingleRegionStageFigure(Problem,S,problemName,runId,figureFile)
    [xLimits,yLimits] = regionObjectivePlotLimits(Problem,S);
    fig = figure('Visible','off','Color','w','Position',[100 100 1120 760]);
    ax = axes(fig);
    hold(ax,'on');
    set(ax,'FontName','Times New Roman','FontSize',14,'Box','on', ...
        'Layer','top','View',[0 90],'Color',[0.98 0.94 0.88]);
    plotRegionFeasibleInfeasibleDomain(ax,Problem);
    plotRegionTrainingSet(ax,S.train_objs);
    plotRegionGenerated(ax,S.generated_objs);
    xlim(ax,xLimits);
    ylim(ax,yLimits);
    xlabel(ax,'f_1');
    ylabel(ax,'f_2');
    title(ax,sprintf('%s run=%d targetFE=%d actualFE=%d gen=%d', ...
        char(problemName),round(runId),round(S.target_FE), ...
        round(S.actual_FE),round(S.generation)), ...
        'Interpreter','none','FontWeight','normal');
    legend(ax,'Location','northeastoutside','Box','off');
    hold(ax,'off');
    exportgraphics(fig,figureFile,'Resolution',300);
    close(fig);
end

function plotRegionFeasibleInfeasibleDomain(ax,Problem)
    plot(ax,nan,nan,'s','MarkerSize',10, ...
        'MarkerFaceColor',[0.98 0.94 0.88], ...
        'MarkerEdgeColor',[0.78 0.63 0.50], ...
        'DisplayName','Infeasible domain');
    PF = Problem.PF;
    if iscell(PF) && numel(PF) >= 3
        surf(ax,PF{1},PF{2},PF{3},'EdgeColor','none', ...
            'FaceColor',[0.82 0.92 0.82],'FaceAlpha',0.72, ...
            'DisplayName','Feasible domain');
    elseif isnumeric(PF) && size(PF,2) >= 2
        plot(ax,PF(:,1),PF(:,2),'-','Color',[0.20 0.55 0.20], ...
            'LineWidth',1.5,'DisplayName','Feasible domain');
    else
        plot(ax,nan,nan,'s','MarkerSize',10, ...
            'MarkerFaceColor',[0.82 0.92 0.82], ...
            'MarkerEdgeColor','none','DisplayName','Feasible domain');
    end
end

function plotRegionTrainingSet(ax,TrainObj)
    Obj = twoColumnObj(TrainObj);
    if isempty(Obj)
        scatter(ax,nan,nan,42,'s','MarkerFaceColor',[1.00 0.68 0.20], ...
            'MarkerEdgeColor',[0.28 0.18 0.05], ...
            'DisplayName','Training set');
        return;
    end
    scatter(ax,Obj(:,1),Obj(:,2),42,'s', ...
        'MarkerFaceColor',[1.00 0.68 0.20], ...
        'MarkerEdgeColor',[0.28 0.18 0.05], ...
        'LineWidth',0.5,'MarkerFaceAlpha',0.82, ...
        'DisplayName','Training set');
end

function plotRegionGenerated(ax,GeneratedObj)
    Obj = twoColumnObj(GeneratedObj);
    if isempty(Obj)
        scatter(ax,nan,nan,50,'o','MarkerFaceColor',[0.88 0.16 0.18], ...
            'MarkerEdgeColor','none','DisplayName','GAN generated');
        return;
    end
    scatter(ax,Obj(:,1),Obj(:,2),54,'o', ...
        'MarkerFaceColor',[0.88 0.16 0.18], ...
        'MarkerEdgeColor','none','LineWidth',0.5, ...
        'MarkerFaceAlpha',0.78,'DisplayName','GAN generated');
end

function [xLimits,yLimits] = regionObjectivePlotLimits(Problem,S)
    Obj = [regionDomainPFPoints(Problem);twoColumnObj(S.train_objs); ...
        twoColumnObj(S.generated_objs)];
    if isempty(Obj)
        PF = Problem.PF;
        if isnumeric(PF) && size(PF,2) >= 2
            Obj = PF(:,1:2);
        end
    end
    Obj = Obj(all(isfinite(Obj),2),:);
    if isempty(Obj)
        xLimits = [0 1];
        yLimits = [0 1];
        return;
    end
    xLimits = [min(Obj(:,1)),max(Obj(:,1))];
    yLimits = [min(Obj(:,2)),max(Obj(:,2))];
    xPad = max(0.05,0.04*max(diff(xLimits),eps));
    yPad = max(0.05,0.04*max(diff(yLimits),eps));
    xLimits = xLimits + [-xPad xPad];
    yLimits = yLimits + [-yPad yPad];
end

function Obj = regionDomainPFPoints(Problem)
    PF = Problem.PF;
    if iscell(PF) && numel(PF) >= 2
        Obj = [PF{1}(:),PF{2}(:)];
    elseif isnumeric(PF) && size(PF,2) >= 2
        Obj = PF(:,1:2);
    else
        Obj = zeros(0,2);
    end
    Obj = Obj(all(isfinite(Obj),2),:);
end

function Obj = twoColumnObj(Obj)
    if isempty(Obj) || size(Obj,2) < 2
        Obj = zeros(0,2);
    else
        Obj = double(Obj(:,1:2));
        Obj = Obj(all(isfinite(Obj),2),:);
    end
end

function values = logicalVectorField(S,name)
    if isstruct(S) && isfield(S,name) && ~isempty(S.(name))
        values = logical(S.(name)(:));
    else
        values = false(0,1);
    end
end

function n = sizeField(S,name)
    if isstruct(S) && isfield(S,name) && ~isempty(S.(name))
        n = size(S.(name),1);
    else
        n = 0;
    end
end

function value = getSnapshotNumeric(S,name)
    if isstruct(S) && isfield(S,name) && ~isempty(S.(name))
        value = double(S.(name));
    else
        value = NaN;
    end
end

function value = getSnapshotString(S,name)
    if isstruct(S) && isfield(S,name) && ~isempty(S.(name))
        value = string(S.(name));
    else
        value = "";
    end
end

function S = emptyRegionStageSnapshotStruct()
    S = struct([]);
end

function Row = emptyStageSnapshotRow()
    Row = struct( ...
        'problem',"", ...
        'run',NaN, ...
        'seed',NaN, ...
        'target_FE',NaN, ...
        'actual_FE',NaN, ...
        'generation',NaN, ...
        'condition_mode',"", ...
        'query_mode',"", ...
        'gan_iter_used',NaN, ...
        'gan_iter_schedule',"", ...
        'train_trigger_mode',"", ...
        'trigger_reason',"", ...
        'trigger_train_now',NaN, ...
        'trigger_train_count_delta',NaN, ...
        'sample_z_mode',"", ...
        'train_z_mode',"", ...
        'train_z_sigma',NaN, ...
        'sample_z_sigma',NaN, ...
        'prescreen_multiplier',NaN, ...
        'prescreen_candidate_count',NaN, ...
        'prescreen_selected_count',NaN, ...
        'train_count',NaN, ...
        'prev_bmem_candidate_count',NaN, ...
        'prev_bmem_survivor_count',NaN, ...
        'raw_generated_count',NaN, ...
        'feasible_rate',NaN, ...
        'gap_ratio50',NaN, ...
        'gap_ratio90',NaN, ...
        'near_boundary_rate_gap1',NaN, ...
        'better_than_random_gap_rate',NaN, ...
        'bdist50_true',NaN, ...
        'bwidth90_10_true',NaN, ...
        'bcover_eps_true',NaN, ...
        'train_width50',NaN, ...
        'train_width90',NaN, ...
        'gen_width50',NaN, ...
        'gen_width90',NaN, ...
        'gen_to_train_dist50',NaN, ...
        'gen_to_train_dist90',NaN, ...
        'offspringG_survival_rate',NaN, ...
        'critic_train_gap',NaN, ...
        'critic_holdout_gap',NaN, ...
        'generated_critic_score_mean',NaN, ...
        'sample_feasible_count',NaN, ...
        'sample_infeasible_count',NaN);
end

function Row = emptyFigureRow()
    Row = struct( ...
        'problem',"", ...
        'run',NaN, ...
        'target_FE',NaN, ...
        'actual_FE',NaN, ...
        'generation',NaN, ...
        'figure_file',"");
end
