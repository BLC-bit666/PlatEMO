function [Summary,outDir,MetricsAll,FigureManifest] = ...
    run_CBS_CGAN_boundary_quality_experiments(outDir,workerCount, ...
    problemNames,N,D,maxFE,runIds,targets,Options)
%RUN_CBS_CGAN_BOUNDARY_QUALITY_EXPERIMENTS Run CBS-CGAN boundary diagnostics.
%
% The default configuration matches the current CBS-CGAN boundary-quality
% experiment: 10 problems, runs=1:3, N=100, default D, maxFE=100000, and
% stage targets 10000/30000/50000/70000/100000. Only plotRun is rendered.

    rootDir = fileparts(which('platemo'));
    if isempty(rootDir)
        rootDir = pwd;
    end
    addpath(genpath(rootDir));
    if nargin < 1 || isempty(outDir)
        outDir = fullfile(rootDir,'Data','CBS_CGAN', ...
            ['boundary_quality_', ...
            char(datetime('now','Format','yyyyMMdd_HHmmss'))]);
    end
    if nargin < 2 || isempty(workerCount)
        workerCount = 8;
    end
    if nargin < 3 || isempty(problemNames)
        problemNames = defaultCBSProblemList();
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
        targets = [10000 30000 50000 70000 100000];
    end
    if nargin < 9 || isempty(Options)
        Options = struct();
    end
    Options = normalizeCBSRunnerOptions(Options);
    workerCount = max(1,round(double(workerCount)));
    problemNames = string(problemNames(:));
    runIds = double(runIds(:)');
    targets = double(targets(:)');
    targets = unique(targets(isfinite(targets) & targets > 0),'stable');
    N = max(1,round(double(N)));
    maxFE = max(1,round(double(maxFE)));

    if ~isfolder(outDir)
        mkdir(outDir);
    end
    Tasks = buildCBSTasks(problemNames,runIds);
    Rows = repmat(emptyCBSRunRow(),height(Tasks),1);

    if workerCount > 1
        ensureCBSParallelPool(workerCount);
        parfor task = 1 : height(Tasks)
            Rows(task) = runOneCBSTask(Tasks(task,:),outDir,N,D,maxFE, ...
                targets,Options);
        end
    else
        for task = 1 : height(Tasks)
            Rows(task) = runOneCBSTask(Tasks(task,:),outDir,N,D,maxFE, ...
                targets,Options);
            fprintf('[%d/%d] %s run=%d status=%s\n',task,height(Tasks), ...
                Tasks.problem(task),Tasks.run(task),Rows(task).status);
        end
    end

    Summary = struct2table(Rows);
    writetable(Summary,fullfile(outDir,'run_summary.csv'));
    MetricsAll = collectCBSStageMetrics(Summary);
    writetable(MetricsAll,fullfile(outDir,'stage_metrics_all.csv'));
    HistoryAll = collectCBSHistoryMetrics(Summary);
    writetable(HistoryAll,fullfile(outDir,'history_metrics_all.csv'));
    FigureManifest = collectCBSFigureManifest(Summary);
    writetable(FigureManifest,fullfile(outDir,'figure_manifest.csv'));
end

function problemNames = defaultCBSProblemList()
    problemNames = ["DASCMOP1_BC";"DASCMOP2_BC"; ...
        "DASCMOP4_BC";"DASCMOP5_BC"; ...
        "LIRCMOP5_BC";"LIRCMOP6_BC"; ...
        "LIRCMOP7_BC";"LIRCMOP8_BC"; ...
        "LIRCMOP9_BC";"LIRCMOP10_BC"];
end

function Options = normalizeCBSRunnerOptions(Options)
    if ~isfield(Options,'plotRun') || isempty(Options.plotRun)
        Options.plotRun = 3;
    end
    if ~isfield(Options,'algorithmParams') || isempty(Options.algorithmParams)
        Options.algorithmParams = {1,1,20,2,50,32,1e-4,2e-4, ...
            0,1,1,4,1,2,1,0.10,1};
    end
    if ~isfield(Options,'algorithmClass') || isempty(Options.algorithmClass)
        Options.algorithmClass = "CBS_CGAN";
    end
    if ~isfield(Options,'conditionMode') || isempty(Options.conditionMode)
        Options.conditionMode = "ref_tau";
    end
    if ~isfield(Options,'visualDiagnostics') || isempty(Options.visualDiagnostics)
        Options.visualDiagnostics = false;
    end
    if ~isfield(Options,'advWeight')
        Options.advWeight = [];
    end
    if ~isfield(Options,'reconstructionWeight')
        Options.reconstructionWeight = [];
    end
    if ~isfield(Options,'trainZMode')
        Options.trainZMode = [];
    end
    if ~isfield(Options,'trainMode')
        Options.trainMode = [];
    end
    if ~isfield(Options,'sampleZMode')
        Options.sampleZMode = [];
    end
    if ~isfield(Options,'generatorHidden')
        Options.generatorHidden = [];
    end
    if ~isfield(Options,'discriminatorHidden')
        Options.discriminatorHidden = [];
    end
    if ~isfield(Options,'plotDiagnosticTrends') || ...
            isempty(Options.plotDiagnosticTrends)
        Options.plotDiagnosticTrends = true;
    end
    if ~isfield(Options,'figureKinds') || isempty(Options.figureKinds)
        Options.figureKinds = ["stage","visual_z_diagnostic", ...
            "train_reconstruction","condition_space_coverage", ...
            "diagnostic_trends"];
    end
    Options.plotRun = round(double(Options.plotRun));
    Options.algorithmClass = string(Options.algorithmClass);
    Options.conditionMode = lower(strtrim(string(Options.conditionMode)));
    Options.visualDiagnostics = logical(Options.visualDiagnostics);
    Options.plotDiagnosticTrends = logical(Options.plotDiagnosticTrends);
    Options.figureKinds = normalizeCBSFigureKinds(Options.figureKinds);
end

function Tasks = buildCBSTasks(problemNames,runIds)
    Rows = repmat(struct('problem',"",'run',NaN), ...
        numel(problemNames)*numel(runIds),1);
    k = 0;
    for p = 1 : numel(problemNames)
        for r = 1 : numel(runIds)
            k = k + 1;
            Rows(k).problem = problemNames(p);
            Rows(k).run = runIds(r);
        end
    end
    Tasks = struct2table(Rows);
end

function Row = runOneCBSTask(Task,outDir,N,D,maxFE,targets,Options)
    Row = emptyCBSRunRow();
    Row.problem = string(Task.problem);
    Row.run = double(Task.run);
    Row.seed = Row.run;
    Row.N = double(N);
    Row.maxFE = double(maxFE);
    Row.condition_mode = string(Options.conditionMode);
    Row.target_text = strjoin(string(targets),';');
    Row.plot_enabled = Row.run == Options.plotRun;
    try
        rootDir = fileparts(which('platemo'));
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
        Row.stage_metrics_file = string(fullfile(runFolder, ...
            'stage_metrics.csv'));
        Row.history_metrics_file = string(fullfile(runFolder, ...
            'history_metrics.csv'));
        Row.figure_manifest_file = string(fullfile(runFolder, ...
            'figure_manifest.csv'));

        Control = struct( ...
            'stageTargets',targets, ...
            'conditionMode',Options.conditionMode, ...
            'visualDiagnostics',Options.visualDiagnostics);
        Control = copyOptionalRunnerControl(Control,Options);
        setappdata(0,'CBS_CGAN_ExperimentControl',Control);
        cleanup = onCleanup(@()removeCBSExperimentControl());

        AlgorithmConstructor = str2func(char(Options.algorithmClass));
        Algorithm = AlgorithmConstructor('save',0,'run',Row.run, ...
            'outputFcn',@(varargin)[], ...
            'parameter',Options.algorithmParams);
        Algorithm.Solve(Problem);

        Row.runtime = double(Algorithm.metric.runtime);
        Row.finalFE = double(Algorithm.result{end,1});
        HistoryMetrics = cbsHistoryMetricsTable( ...
            Algorithm.metric.cbs_cgan_history,Row.problem,Row.run);
        writetable(HistoryMetrics,Row.history_metrics_file);
        Row.history_count = height(HistoryMetrics);
        Snapshots = Algorithm.metric.cbs_cgan_stage_snapshots;
        StageMetrics = cbsStageMetricsTable(Snapshots);
        writetable(StageMetrics,Row.stage_metrics_file);
        Row.stage_count = height(StageMetrics);
        if Row.plot_enabled
            Figures = plotCBSStageFigures(Problem,Snapshots,Row.problem, ...
                Row.run,runFolder,Options.figureKinds);
            if Options.plotDiagnosticTrends && ...
                    cbsFigureKindEnabled(Options.figureKinds,"diagnostic_trends")
                DiagnosticFigures = plotCBSDiagnosticFigures(HistoryMetrics, ...
                    Row.problem,Row.run,runFolder);
            else
                DiagnosticFigures = cbsFigureTable([]);
            end
            Figures = [Figures;DiagnosticFigures];
        else
            Figures = cbsFigureTable([]);
        end
        writetable(Figures,Row.figure_manifest_file);
        Row.figure_count = height(Figures);
        Row.status = "ok";
    catch err
        Row.status = "failed";
        Row.error_message = string(getReport(err,'extended', ...
            'hyperlinks','off'));
    end
end

function removeCBSExperimentControl()
    if isappdata(0,'CBS_CGAN_ExperimentControl')
        rmappdata(0,'CBS_CGAN_ExperimentControl');
    end
end

function Control = copyOptionalRunnerControl(Control,Options)
    names = {'advWeight','reconstructionWeight','trainZMode', ...
        'trainMode','sampleZMode','generatorHidden', ...
        'discriminatorHidden','boundaryTargetMode'};
    for i = 1 : numel(names)
        if isfield(Options,names{i}) && ~isempty(Options.(names{i}))
            Control.(names{i}) = Options.(names{i});
        end
    end
end

function kinds = normalizeCBSFigureKinds(kinds)
    if ischar(kinds)
        kinds = string({kinds});
    else
        kinds = string(kinds);
    end
    kinds = lower(strtrim(kinds(:)));
    kinds = kinds(strlength(kinds) > 0);
    if isempty(kinds) || any(kinds == "all")
        kinds = ["stage";"visual_z_diagnostic";"train_reconstruction"; ...
            "condition_space_coverage";"diagnostic_trends"];
    end
    aliases = ["visual","visual_z";"visual_z_diagnostic","visual_z_diagnostic"; ...
        "train","train_reconstruction";"coverage","condition_space_coverage"; ...
        "diagnostic","diagnostic_trends"];
    for i = 1 : size(aliases,1)
        kinds(kinds == aliases(i,1)) = aliases(i,2);
    end
    kinds = unique(kinds,'stable');
end

function enabled = cbsFigureKindEnabled(kinds,kind)
    enabled = any(string(kinds(:)) == string(kind));
end

function Row = emptyCBSRunRow()
    Row = struct( ...
        'problem',"", ...
        'run',NaN, ...
        'seed',NaN, ...
        'N',NaN, ...
        'D',NaN, ...
        'maxFE',NaN, ...
        'condition_mode',"", ...
        'target_text',"", ...
        'plot_enabled',false, ...
        'runtime',NaN, ...
        'finalFE',NaN, ...
        'stage_count',0, ...
        'history_count',0, ...
        'figure_count',0, ...
        'run_folder',"", ...
        'stage_metrics_file',"", ...
        'history_metrics_file',"", ...
        'figure_manifest_file',"", ...
        'status',"pending", ...
        'error_message',"");
end

function T = cbsStageMetricsTable(Snapshots)
    Rows = repmat(emptyCBSStageRow(),numel(Snapshots),1);
    for i = 1 : numel(Snapshots)
        S = Snapshots(i);
        Rows(i).problem = string(S.problem);
        Rows(i).run = double(S.run);
        Rows(i).condition_mode = string(S.condition_mode);
        Rows(i).target_FE = double(S.target_FE);
        Rows(i).actual_FE = double(S.actual_FE);
        Rows(i).gen = double(S.gen);
        Rows(i).bmem_count = double(S.bmem_count);
        Rows(i).boundary_count = double(S.boundary_count);
        Rows(i).bmem_ref_coverage = double(S.bmem_ref_coverage);
        Rows(i).finite_gap_count = double(S.finite_gap_count);
        Rows(i).inf_gap_count = double(S.inf_gap_count);
        Rows(i).median_gap = double(S.median_gap);
        Rows(i).max_gap = double(S.max_gap);
        Rows(i).train_count = double(S.train_count);
        Rows(i).query_count = double(S.query_count);
        Rows(i).condition_dim = double(S.condition_dim);
        Rows(i).train_param_ratio = double(S.train_param_ratio);
        Rows(i).gan_sample_reuse = double(S.gan_sample_reuse);
        Rows(i).train_pair_dec_dist50 = double(S.train_pair_dec_dist50);
        Rows(i).train_pair_dec_dist90 = double(S.train_pair_dec_dist90);
        Rows(i).train_tau_iqr = double(S.train_tau_iqr);
        Rows(i).train_tau_range = double(S.train_tau_range);
        Rows(i).train_tau_nonzero_rate = double(S.train_tau_nonzero_rate);
        Rows(i).train_x_rec90 = double(S.train_x_rec90);
        Rows(i).train_y_rec90 = double(S.train_y_rec90);
        Rows(i).sample_z_mode = string(S.sample_z_mode);
        Rows(i).missing_ref_query_count = double(S.missing_ref_query_count);
        Rows(i).large_gap_query_count = double(S.large_gap_query_count);
        Rows(i).raw_generated_count = double(S.raw_generated_count);
        Rows(i).feasible_generated_count = ...
            double(S.feasible_generated_count);
        Rows(i).generated_per_train = double(S.generated_per_train);
        Rows(i).boundary_dist50 = double(S.boundary_dist50);
        Rows(i).boundary_dist90 = double(S.boundary_dist90);
        Rows(i).query_width90 = double(S.query_width90);
        Rows(i).segment_width90 = double(S.segment_width90);
        Rows(i).segment_width90_ratio = double(S.segment_width90_ratio);
        Rows(i).side_rate = double(S.side_rate);
        Rows(i).pair_margin50 = double(S.pair_margin50);
        Rows(i).ref_cover = double(S.ref_cover);
        Rows(i).query_obj_dist50 = double(S.query_obj_dist50);
        Rows(i).query_obj_dist90 = double(S.query_obj_dist90);
        Rows(i).missing_ref_query_obj_dist90 = ...
            double(S.missing_ref_query_obj_dist90);
        Rows(i).large_gap_query_obj_dist90 = ...
            double(S.large_gap_query_obj_dist90);
        Rows(i).feasible_rate = double(S.feasible_rate);
        Rows(i) = copyPipelineMetrics(Rows(i),S);
    end
    T = struct2table(Rows);
end

function Row = emptyCBSStageRow()
    Row = struct( ...
        'problem',"", ...
        'run',NaN, ...
        'condition_mode',"ref_tau", ...
        'target_FE',NaN, ...
        'actual_FE',NaN, ...
        'gen',NaN, ...
        'bmem_count',0, ...
        'boundary_count',0, ...
        'bmem_ref_coverage',NaN, ...
        'finite_gap_count',0, ...
        'inf_gap_count',0, ...
        'median_gap',NaN, ...
        'max_gap',NaN, ...
        'train_count',0, ...
        'query_count',0, ...
        'condition_dim',NaN, ...
        'train_param_ratio',NaN, ...
        'gan_sample_reuse',NaN, ...
        'train_pair_dec_dist50',NaN, ...
        'train_pair_dec_dist90',NaN, ...
        'train_tau_iqr',NaN, ...
        'train_tau_range',NaN, ...
        'train_tau_nonzero_rate',NaN, ...
        'train_x_rec90',NaN, ...
        'train_y_rec90',NaN, ...
        'sample_z_mode',"zero", ...
        'missing_ref_query_count',0, ...
        'large_gap_query_count',0, ...
        'raw_generated_count',0, ...
        'feasible_generated_count',0, ...
        'generated_per_train',NaN, ...
        'boundary_dist50',NaN, ...
        'boundary_dist90',NaN, ...
        'query_width90',NaN, ...
        'segment_width90',NaN, ...
        'segment_width90_ratio',NaN, ...
        'side_rate',NaN, ...
        'pair_margin50',NaN, ...
        'ref_cover',NaN, ...
        'query_obj_dist50',NaN, ...
        'query_obj_dist90',NaN, ...
        'missing_ref_query_obj_dist90',NaN, ...
        'large_gap_query_obj_dist90',NaN, ...
        'feasible_rate',NaN);
    Row = addDefaultPipelineMetrics(Row);
end

function T = cbsHistoryMetricsTable(History,problemName,runId)
    Rows = repmat(emptyCBSHistoryRow(),numel(History),1);
    for i = 1 : numel(History)
        S = History(i);
        Rows(i).problem = string(problemName);
        Rows(i).run = double(runId);
        Rows(i).condition_mode = string(metricValue( ...
            S,'condition_mode',"ref_tau"));
        Rows(i).gen = double(metricValue(S,'generation',i));
        Rows(i).bmem_count = double(metricValue(S,'bmem_count',0));
        Rows(i).boundary_count = double(metricValue(S,'boundary_count',0));
        Rows(i).bmem_ref_coverage = double(metricValue( ...
            S,'bmem_ref_coverage',NaN));
        Rows(i).finite_gap_count = double(metricValue( ...
            S,'finite_gap_count',0));
        Rows(i).inf_gap_count = double(metricValue(S,'inf_gap_count',0));
        Rows(i).median_gap = double(metricValue(S,'median_gap',NaN));
        Rows(i).max_gap = double(metricValue(S,'max_gap',NaN));
        Rows(i).train_count = double(metricValue(S,'train_count',0));
        Rows(i).query_count = double(metricValue(S,'query_count',0));
        Rows(i).condition_dim = double(metricValue(S,'condition_dim',NaN));
        Rows(i).train_param_ratio = double(metricValue( ...
            S,'train_param_ratio',NaN));
        Rows(i).gan_sample_reuse = double(metricValue( ...
            S,'gan_sample_reuse',NaN));
        Rows(i).train_pair_dec_dist50 = double(metricValue( ...
            S,'train_pair_dec_dist50',NaN));
        Rows(i).train_pair_dec_dist90 = double(metricValue( ...
            S,'train_pair_dec_dist90',NaN));
        Rows(i).train_tau_iqr = double(metricValue(S,'train_tau_iqr',NaN));
        Rows(i).train_tau_range = double(metricValue( ...
            S,'train_tau_range',NaN));
        Rows(i).train_tau_nonzero_rate = double(metricValue( ...
            S,'train_tau_nonzero_rate',NaN));
        Rows(i).train_x_rec90 = double(metricValue(S,'train_x_rec90',NaN));
        Rows(i).train_y_rec90 = double(metricValue(S,'train_y_rec90',NaN));
        Rows(i).sample_z_mode = string(metricValue(S, ...
            'sample_z_mode',"zero"));
        Rows(i).missing_ref_query_count = double(metricValue( ...
            S,'missing_ref_query_count',0));
        Rows(i).large_gap_query_count = double(metricValue( ...
            S,'large_gap_query_count',0));
        Rows(i).raw_generated_count = double(metricValue( ...
            S,'raw_generated_count',0));
        Rows(i).feasible_generated_count = double(metricValue( ...
            S,'feasible_generated_count',0));
        Rows(i).generated_per_train = double(metricValue( ...
            S,'generated_per_train',NaN));
        Rows(i).boundary_dist50 = double(metricValue( ...
            S,'boundary_dist50',NaN));
        Rows(i).boundary_dist90 = double(metricValue( ...
            S,'boundary_dist90',NaN));
        Rows(i).query_width90 = double(metricValue(S,'query_width90',NaN));
        Rows(i).segment_width90 = double(metricValue( ...
            S,'segment_width90',NaN));
        Rows(i).segment_width90_ratio = double(metricValue( ...
            S,'segment_width90_ratio',NaN));
        Rows(i).side_rate = double(metricValue(S,'side_rate',NaN));
        Rows(i).pair_margin50 = double(metricValue(S,'pair_margin50',NaN));
        Rows(i).ref_cover = double(metricValue(S,'ref_cover',NaN));
        Rows(i).query_obj_dist50 = double(metricValue( ...
            S,'query_obj_dist50',NaN));
        Rows(i).query_obj_dist90 = double(metricValue( ...
            S,'query_obj_dist90',NaN));
        Rows(i).missing_ref_query_obj_dist90 = double(metricValue( ...
            S,'missing_ref_query_obj_dist90',NaN));
        Rows(i).large_gap_query_obj_dist90 = double(metricValue( ...
            S,'large_gap_query_obj_dist90',NaN));
        Rows(i).feasible_rate = double(metricValue(S,'feasible_rate',NaN));
        Rows(i) = copyPipelineMetrics(Rows(i),S);
    end
    T = struct2table(Rows);
end

function Row = emptyCBSHistoryRow()
    Stage = emptyCBSStageRow();
    Stage = rmfield(Stage,{'target_FE','actual_FE'});
    Row = Stage;
end

function value = metricValue(S,name,defaultValue)
    if isstruct(S) && isfield(S,name) && ~isempty(S.(name))
        value = S.(name);
    else
        value = defaultValue;
    end
end

function Row = copyPipelineMetrics(Row,S)
    names = cbsPipelineMetricFields();
    for i = 1 : numel(names)
        Row.(names{i}) = double(metricValue(S,names{i},0));
    end
end

function Row = addDefaultPipelineMetrics(Row)
    names = cbsPipelineMetricFields();
    for i = 1 : numel(names)
        Row.(names{i}) = 0;
    end
end

function names = cbsPipelineMetricFields()
    names = {'boundary_sample_current_count', ...
        'boundary_sample_prev_pair_count', ...
        'boundary_sample_merged_count', ...
        'boundary_sample_finite_count', ...
        'boundary_feasible_count', ...
        'boundary_infeasible_count', ...
        'boundary_main_feasible_count', ...
        'boundary_ref_with_main_feasible_count', ...
        'boundary_ref_with_neighbor_infeasible_count', ...
        'boundary_ref_pairable_count', ...
        'boundary_pair_distance_count', ...
        'boundary_pair_dominated_skip_count', ...
        'boundary_pair_appended_count', ...
        'boundary_candidate_raw_count', ...
        'boundary_candidate_finite_count', ...
        'boundary_candidate_pareto_count', ...
        'boundary_candidate_gap_count', ...
        'boundary_candidate_dedup_count', ...
        'boundary_candidate_final_count', ...
        'dataset_bmem_input_count', ...
        'dataset_valid_train_count', ...
        'dataset_invalid_train_count', ...
        'dataset_query_count'};
end

function Figures = plotCBSStageFigures(Problem,Snapshots,problemName, ...
    runId,runFolder,figureKinds)
    Rows = repmat(emptyCBSFigureRow(),0,1);
    figureFolder = fullfile(runFolder,'figures');
    if ~isfolder(figureFolder)
        mkdir(figureFolder);
    end
    for i = 1 : numel(Snapshots)
        S = Snapshots(i);
        if cbsFigureKindEnabled(figureKinds,"stage")
            figureFile = fullfile(figureFolder,sprintf( ...
                '%s_run%d_targetFE%06d.png',char(problemName), ...
                round(runId),round(S.target_FE)));
            plotSingleCBSStageFigure(Problem,S,problemName,runId,figureFile);
            Rows(end+1) = makeCBSFigureRow(problemName,runId,S,figureFile); %#ok<AGROW>
        end
        if hasCBSVisualDiagnostics(S)
            if cbsFigureKindEnabled(figureKinds,"visual_z_diagnostic")
                visualFile = fullfile(figureFolder,sprintf( ...
                    '%s_run%d_targetFE%06d_visual_z_diagnostic.png', ...
                    char(problemName),round(runId),round(S.target_FE)));
                plotSingleCBSVisualDiagnosticFigure(Problem,S,problemName, ...
                    runId,visualFile);
                Rows(end+1) = makeCBSFigureRow(problemName,runId,S,visualFile); %#ok<AGROW>
            end
            if cbsFigureKindEnabled(figureKinds,"train_reconstruction")
                trainFile = fullfile(figureFolder,sprintf( ...
                    '%s_run%d_targetFE%06d_train_reconstruction.png', ...
                    char(problemName),round(runId),round(S.target_FE)));
                plotSingleCBSTrainReconstructionFigure(Problem,S,problemName, ...
                    runId,trainFile);
                Rows(end+1) = makeCBSFigureRow(problemName,runId,S,trainFile); %#ok<AGROW>
            end
            if cbsFigureKindEnabled(figureKinds,"condition_space_coverage")
                coverageFile = fullfile(figureFolder,sprintf( ...
                    '%s_run%d_targetFE%06d_condition_space_coverage.png', ...
                    char(problemName),round(runId),round(S.target_FE)));
                plotSingleCBSConditionSpaceCoverageFigure(S,problemName, ...
                    runId,coverageFile);
                Rows(end+1) = makeCBSFigureRow(problemName,runId,S,coverageFile); %#ok<AGROW>
            end
        end
    end
    Figures = cbsFigureTable(Rows);
end

function Row = makeCBSFigureRow(problemName,runId,S,figureFile)
    Row = emptyCBSFigureRow();
    Row.problem = string(problemName);
    Row.run = double(runId);
    Row.target_FE = double(S.target_FE);
    Row.actual_FE = double(S.actual_FE);
    Row.gen = double(S.gen);
    Row.figure_file = string(figureFile);
end

function hasVisual = hasCBSVisualDiagnostics(S)
    hasVisual = isfield(S,'visual_query_zero_objs') && ...
        (~isempty(S.visual_query_zero_objs) || ...
        ~isempty(S.visual_train_zero_objs));
end

function Figures = plotCBSDiagnosticFigures(HistoryMetrics,problemName, ...
    runId,runFolder)
    if isempty(HistoryMetrics) || height(HistoryMetrics) == 0
        Figures = cbsFigureTable([]);
        return;
    end
    figureFolder = fullfile(runFolder,'figures');
    if ~isfolder(figureFolder)
        mkdir(figureFolder);
    end
    figureFile = fullfile(figureFolder,sprintf( ...
        '%s_run%d_diagnostic_trends.png',char(problemName),round(runId)));
    plotSingleCBSDiagnosticFigure(HistoryMetrics,problemName,runId, ...
        figureFile);
    Row = emptyCBSFigureRow();
    Row.problem = string(problemName);
    Row.run = double(runId);
    Row.target_FE = NaN;
    Row.actual_FE = NaN;
    Row.gen = double(HistoryMetrics.gen(end));
    Row.figure_file = string(figureFile);
    Figures = cbsFigureTable(Row);
end

function plotSingleCBSDiagnosticFigure(T,problemName,runId,figureFile)
    fig = figure('Visible','off','Color','w','Position',[100 100 1220 860]);
    layout = tiledlayout(fig,3,2,'TileSpacing','compact', ...
        'Padding','compact');
    title(layout,sprintf('%s run=%d CBS-CGAN diagnostics', ...
        char(problemName),round(runId)), ...
        'Interpreter','none','FontWeight','normal');
    plotCBSMetricTile(nexttile(layout),T,'train_count', ...
        'Train count');
    plotCBSMetricTile(nexttile(layout),T,'bmem_ref_coverage', ...
        'BMem ref coverage');
    plotCBSMetricTile(nexttile(layout),T,'gan_sample_reuse', ...
        'GAN sample reuse');
    plotCBSMetricTile(nexttile(layout),T,'train_param_ratio', ...
        'Train / parameter ratio');
    plotCBSMetricTile(nexttile(layout),T,'feasible_rate', ...
        'Generated feasible rate');
    plotCBSMetricTile(nexttile(layout),T,'boundary_dist90', ...
        'Boundary distance p90');
    exportgraphics(fig,figureFile,'Resolution',300);
    close(fig);
end

function plotCBSMetricTile(ax,T,fieldName,labelText)
    x = double(T.gen);
    y = double(T.(fieldName));
    valid = isfinite(x) & isfinite(y);
    plot(ax,x(valid),y(valid),'-o','LineWidth',1.2,'MarkerSize',3, ...
        'Color',[0.16 0.36 0.62]);
    grid(ax,'on');
    box(ax,'on');
    xlabel(ax,'generation');
    ylabel(ax,labelText,'Interpreter','none');
    title(ax,labelText,'Interpreter','none','FontWeight','normal');
    if ~any(valid)
        text(ax,0.5,0.5,'no finite data','Units','normalized', ...
            'HorizontalAlignment','center','Color',[0.45 0.45 0.45]);
    end
end

function T = cbsFigureTable(Rows)
    if isempty(Rows)
        Rows = repmat(emptyCBSFigureRow(),0,1);
    end
    T = struct2table(Rows);
end

function Row = emptyCBSFigureRow()
    Row = struct( ...
        'problem',"", ...
        'run',NaN, ...
        'target_FE',NaN, ...
        'actual_FE',NaN, ...
        'gen',NaN, ...
        'figure_file',"");
end

function plotSingleCBSStageFigure(Problem,S,problemName,runId,figureFile)
    [xLimits,yLimits] = cbsObjectiveLimitsFromPF(Problem);
    fig = figure('Visible','off','Color','w','Position',[100 100 1120 760]);
    ax = axes(fig);
    hold(ax,'on');
    set(ax,'FontName','Times New Roman','FontSize',16,'Box','on', ...
        'Layer','top','View',[0 90],'Color',[0.98 0.94 0.88]);
    plotCBSFeasibleRegion(ax,Problem);
    plotCBSTrainingSet(ax,S.train_objs);
    plotCBSGenerated(ax,S.generated_objs);
    xlim(ax,xLimits);
    ylim(ax,yLimits);
    xlabel(ax,'f_1');
    ylabel(ax,'f_2');
    title(ax,sprintf('%s run=%d targetFE=%d actualFE=%d gen=%d', ...
        char(problemName),round(runId),round(S.target_FE), ...
        round(S.actual_FE),round(S.gen)), ...
        'Interpreter','none','FontWeight','normal','FontSize',18);
    legend(ax,'Location','northeastoutside','Box','off','FontSize',16);
    hold(ax,'off');
    exportgraphics(fig,figureFile,'Resolution',300);
    close(fig);
end

function plotSingleCBSVisualDiagnosticFigure(Problem,S,problemName,runId, ...
    figureFile)
    [xLimits,yLimits] = cbsObjectiveLimitsFromPF(Problem);
    fig = figure('Visible','off','Color','w','Position',[100 100 1380 520]);
    layout = tiledlayout(fig,1,3,'TileSpacing','compact','Padding','compact');
    title(layout,sprintf('%s run=%d targetFE=%d visual z diagnostic', ...
        char(problemName),round(runId),round(S.target_FE)), ...
        'Interpreter','none','FontWeight','normal','FontSize',18);

    plotCBSVisualTile(nexttile(layout),Problem,S,S.generated_objs, ...
        formalQueryTileTitle(S),xLimits,yLimits,true);
    plotCBSVisualTile(nexttile(layout),Problem,S,S.visual_query_zero_objs, ...
        'QueryC z=0',xLimits,yLimits,true);
    plotCBSVisualTile(nexttile(layout),Problem,S,S.visual_train_zero_objs, ...
        'TrainC z=0',xLimits,yLimits,false);

    exportgraphics(fig,figureFile,'Resolution',220);
    close(fig);
end

function plotSingleCBSTrainReconstructionFigure(Problem,S,problemName,runId, ...
    figureFile)
    [xLimits,yLimits] = cbsObjectiveLimitsFromPF(Problem);
    fig = figure('Visible','off','Color','w','Position',[100 100 760 620]);
    layout = tiledlayout(fig,1,1,'TileSpacing','compact','Padding','compact');
    title(layout,sprintf('%s run=%d targetFE=%d TrainC reconstruction', ...
        char(problemName),round(runId),round(S.target_FE)), ...
        'Interpreter','none','FontWeight','normal','FontSize',18);
    plotCBSVisualTile(nexttile(layout),Problem,S,S.visual_train_zero_objs, ...
        'TrainC z=0',xLimits,yLimits,false);
    exportgraphics(fig,figureFile,'Resolution',240);
    close(fig);
end

function plotSingleCBSConditionSpaceCoverageFigure(S,problemName,runId, ...
    figureFile)
    fig = figure('Visible','off','Color','w','Position',[100 100 940 650]);
    ax = axes(fig);
    hold(ax,'on');
    set(ax,'FontName','Times New Roman','FontSize',14,'Box','on', ...
        'Layer','top');
    hasData = false;
    hasData = plotCBSConditionPoints(ax,S.train_ref,S.train_tau, ...
        's',[0.89 0.45 0.10],[0.89 0.45 0.10], ...
        'Training condition') || hasData;
    hasData = plotCBSConditionPoints(ax,S.query_ref,S.query_tau, ...
        'd',[0.12 0.38 0.72],'none','Query condition') || hasData;
    [genRef,genTau] = generatedConditionPairs(S);
    hasData = plotCBSConditionPoints(ax,genRef,genTau, ...
        'o',[0.72 0.14 0.12],[0.72 0.14 0.12], ...
        'Feasible generated') || hasData;
    grid(ax,'on');
    xlabel(ax,'reference index');
    ylabel(ax,'tau');
    title(ax,sprintf('%s run=%d targetFE=%d condition-space coverage', ...
        char(problemName),round(runId),round(S.target_FE)), ...
        'Interpreter','none','FontWeight','normal');
    ylim(ax,[-0.05 1.05]);
    refs = [double(S.train_ref(:));double(S.query_ref(:));genRef(:)];
    refs = refs(isfinite(refs));
    if isempty(refs)
        xlim(ax,[0 1]);
    else
        xlim(ax,[max(0.5,min(refs)-0.5),max(refs)+0.5]);
    end
    if hasData
        legend(ax,'Location','best','Box','off');
    else
        text(ax,0.5,0.5,'no condition data','Units','normalized', ...
            'HorizontalAlignment','center','Color',[0.45 0.45 0.45]);
    end
    hold(ax,'off');
    exportgraphics(fig,figureFile,'Resolution',240);
    close(fig);
end

function hasData = plotCBSConditionPoints(ax,ref,tau,marker,edgeColor, ...
    faceColor,labelText)
    ref = double(ref(:));
    tau = double(tau(:));
    n = min(numel(ref),numel(tau));
    ref = ref(1:n);
    tau = tau(1:n);
    valid = isfinite(ref) & isfinite(tau);
    hasData = any(valid);
    if ~hasData
        return;
    end
    tau = max(0,min(1,tau));
    scatter(ax,ref(valid),tau(valid),48, ...
        'Marker',marker, ...
        'MarkerEdgeColor',edgeColor, ...
        'MarkerFaceColor',faceColor, ...
        'LineWidth',1.1, ...
        'DisplayName',labelText);
end

function [ref,tau] = generatedConditionPairs(S)
    ref = zeros(0,1);
    tau = zeros(0,1);
    if ~isfield(S,'query_index') || ~isfield(S,'query_ref') || ...
            ~isfield(S,'query_tau') || isempty(S.query_index)
        return;
    end
    idx = round(double(S.query_index(:)));
    keep = true(size(idx));
    if isfield(S,'generated_feasible') && ...
            numel(S.generated_feasible) == numel(idx)
        keep = logical(S.generated_feasible(:));
    end
    idx = idx(keep);
    valid = isfinite(idx) & idx >= 1 & idx <= numel(S.query_ref) & ...
        idx <= numel(S.query_tau);
    idx = idx(valid);
    ref = double(S.query_ref(idx));
    tau = double(S.query_tau(idx));
end

function titleText = formalQueryTileTitle(S)
    mode = "generated";
    if isfield(S,'sample_z_mode') && ~isempty(S.sample_z_mode)
        mode = lower(strtrim(string(S.sample_z_mode)));
    end
    switch mode
        case "zero"
            titleText = 'QueryC z=0';
        case "random"
            titleText = 'QueryC random z';
        otherwise
            titleText = 'QueryC generated';
    end
end

function plotCBSVisualTile(ax,Problem,S,GeneratedObj,titleText, ...
    xLimits,yLimits,showQuery)
    [tileXLimits,tileYLimits] = cbsVisualTileLimits( ...
        xLimits,yLimits,S,GeneratedObj,showQuery);
    hold(ax,'on');
    set(ax,'FontName','Times New Roman','FontSize',11,'Box','on', ...
        'Layer','top','View',[0 90],'Color',[0.98 0.94 0.88]);
    plotCBSFeasibleRegion(ax,Problem);
    plotCBSTrainingSet(ax,S.train_objs);
    if showQuery
        plotCBSQueryTargets(ax,S.query_objs);
    end
    plotCBSGenerated(ax,GeneratedObj);
    xlim(ax,tileXLimits);
    ylim(ax,tileYLimits);
    xlabel(ax,'f_1');
    ylabel(ax,'f_2');
    title(ax,titleText,'Interpreter','none','FontWeight','normal');
    hold(ax,'off');
end

function [xLimits,yLimits] = cbsVisualTileLimits( ...
    baseXLimits,baseYLimits,S,GeneratedObj,showQuery)
    Obj = zeros(0,2);
    if isfield(S,'train_objs') && size(S.train_objs,2) >= 2
        Obj = [Obj;S.train_objs(:,1:2)];
    end
    if showQuery && isfield(S,'query_objs') && size(S.query_objs,2) >= 2
        Obj = [Obj;S.query_objs(:,1:2)];
    end
    if ~isempty(GeneratedObj) && size(GeneratedObj,2) >= 2
        Obj = [Obj;GeneratedObj(:,1:2)];
    end
    valid = all(isfinite(Obj),2);
    Obj = Obj(valid,:);
    if isempty(Obj)
        xLimits = baseXLimits;
        yLimits = baseYLimits;
        return;
    end
    xLimits = [min([baseXLimits(1);Obj(:,1)]), ...
        max([baseXLimits(2);Obj(:,1)])];
    yLimits = [min([baseYLimits(1);Obj(:,2)]), ...
        max([baseYLimits(2);Obj(:,2)])];
    xPad = max(0.05,0.04*diff(xLimits));
    yPad = max(0.05,0.04*diff(yLimits));
    xLimits = xLimits + [-xPad xPad];
    yLimits = yLimits + [-yPad yPad];
end

function plotCBSFeasibleRegion(ax,Problem)
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
    end
end

function plotCBSTrainingSet(ax,TrainObj)
    if isempty(TrainObj) || size(TrainObj,2) < 2
        plot(ax,nan,nan,'s','MarkerSize',7, ...
            'MarkerFaceColor',[1.00 0.68 0.20], ...
            'MarkerEdgeColor',[0.28 0.18 0.05], ...
            'DisplayName','Training set');
        return;
    end
    scatter(ax,TrainObj(:,1),TrainObj(:,2),38,'s', ...
        'MarkerFaceColor',[1.00 0.68 0.20], ...
        'MarkerEdgeColor',[0.28 0.18 0.05], ...
        'LineWidth',0.6,'DisplayName','Training set');
end

function plotCBSQueryTargets(ax,QueryObj)
    if isempty(QueryObj) || size(QueryObj,2) < 2
        plot(ax,nan,nan,'d','MarkerSize',7, ...
            'MarkerFaceColor',[0.20 0.45 0.95], ...
            'MarkerEdgeColor','none','DisplayName','Query target');
        return;
    end
    scatter(ax,QueryObj(:,1),QueryObj(:,2),34,'d', ...
        'MarkerFaceColor',[0.20 0.45 0.95], ...
        'MarkerEdgeColor','none','MarkerFaceAlpha',0.60, ...
        'DisplayName','Query target');
end

function plotCBSGenerated(ax,GeneratedObj)
    if isempty(GeneratedObj) || size(GeneratedObj,2) < 2
        plot(ax,nan,nan,'o','MarkerSize',7, ...
            'MarkerFaceColor',[0.90 0.18 0.20], ...
            'MarkerEdgeColor','none','DisplayName','CGAN generated');
        return;
    end
    scatter(ax,GeneratedObj(:,1),GeneratedObj(:,2),36,'o', ...
        'MarkerFaceColor',[0.90 0.18 0.20], ...
        'MarkerEdgeColor','none','MarkerFaceAlpha',0.68, ...
        'DisplayName','CGAN generated');
end

function [xLimits,yLimits] = cbsObjectiveLimitsFromPF(Problem)
    PF = Problem.PF;
    if iscell(PF) && numel(PF) >= 2
        xLimits = finiteCBSLimits(PF{1}(:));
        yLimits = finiteCBSLimits(PF{2}(:));
    elseif isnumeric(PF) && size(PF,2) >= 2
        xLimits = finiteCBSLimits(PF(:,1));
        yLimits = finiteCBSLimits(PF(:,2));
    else
        xLimits = [0 1];
        yLimits = [0 1];
    end
end

function limits = finiteCBSLimits(values)
    values = double(values(:));
    values = values(isfinite(values));
    if isempty(values)
        limits = [0 1];
        return;
    end
    lo = min(values);
    hi = max(values);
    if lo == hi
        pad = max(1,abs(lo))*0.05;
    else
        pad = (hi - lo)*0.06;
    end
    limits = [lo - pad,hi + pad];
end

function MetricsAll = collectCBSStageMetrics(Summary)
    MetricsAll = table();
    for i = 1 : height(Summary)
        file = string(Summary.stage_metrics_file(i));
        if strlength(file) > 0 && isfile(file)
            T = readtable(file,'TextType','string','Delimiter',',', ...
                'VariableNamingRule','preserve');
            MetricsAll = [MetricsAll;T]; %#ok<AGROW>
        end
    end
    if isempty(MetricsAll)
        MetricsAll = struct2table(repmat(emptyCBSStageRow(),0,1));
    end
end

function HistoryAll = collectCBSHistoryMetrics(Summary)
    HistoryAll = table();
    for i = 1 : height(Summary)
        file = string(Summary.history_metrics_file(i));
        if strlength(file) > 0 && isfile(file)
            T = readtable(file,'TextType','string','Delimiter',',', ...
                'VariableNamingRule','preserve');
            HistoryAll = [HistoryAll;T]; %#ok<AGROW>
        end
    end
    if isempty(HistoryAll)
        HistoryAll = struct2table(repmat(emptyCBSHistoryRow(),0,1));
    end
end

function FigureManifest = collectCBSFigureManifest(Summary)
    FigureManifest = table();
    for i = 1 : height(Summary)
        file = string(Summary.figure_manifest_file(i));
        if strlength(file) > 0 && isfile(file)
            T = readtable(file,'TextType','string','Delimiter',',', ...
                'VariableNamingRule','preserve');
            FigureManifest = [FigureManifest;T]; %#ok<AGROW>
        end
    end
    if isempty(FigureManifest)
        FigureManifest = struct2table(repmat(emptyCBSFigureRow(),0,1));
    end
end

function ensureCBSParallelPool(workerCount)
    pool = gcp('nocreate');
    if isempty(pool) || pool.NumWorkers ~= workerCount
        if ~isempty(pool)
            delete(pool);
        end
        parpool('local',workerCount);
    end
end
