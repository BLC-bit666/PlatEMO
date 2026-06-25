function [Summary,outDir,StageMetrics,FigureManifest,VariantManifest, ...
        HistoryMetrics,PairedDeltas] = ...
        run_CBS_CGAN_query_condition_ablation(outDir,Options)
%RUN_CBS_CGAN_QUERY_CONDITION_ABLATION Compare CBS-CGAN query conditions.
%
% Numeric-only default experiment:
%   K = 3, condition variants A/B/C, runs = 1:3, plotRun = 0.
% Each variant is delegated to run_CBS_CGAN_boundary_quality_experiments.

    rootDir = fileparts(which('platemo'));
    if isempty(rootDir)
        rootDir = pwd;
    end
    addpath(genpath(rootDir));
    if nargin < 1 || isempty(outDir)
        outDir = fullfile(rootDir,'Data','CBS_CGAN', ...
            ['query_condition_ablation_K3_runs3_noplot_', ...
            char(datetime('now','Format','yyyyMMdd_HHmmss'))]);
    end
    if nargin < 2 || isempty(Options)
        Options = struct();
    end
    Options = normalizeQueryConditionAblationOptions(Options);

    if ~isfolder(outDir)
        mkdir(outDir);
    end

    Summary = table();
    StageMetrics = table();
    HistoryMetrics = table();
    FigureManifest = table();
    Variants = queryConditionVariants();
    VariantRows = repmat(emptyConditionVariantRow(),numel(Variants),1);

    for i = 1 : numel(Variants)
        variant = Variants(i).variant;
        conditionMode = Variants(i).conditionMode;
        variantDir = fullfile(outDir,char(variant));
        algorithmParams = setQueryConditionPairCount( ...
            Options.baseAlgorithmParams,Options.pairCount);
        RunnerOptions = struct( ...
            'plotRun',Options.plotRun, ...
            'conditionMode',conditionMode, ...
            'algorithmParams',{algorithmParams}, ...
            'algorithmClass',Options.algorithmClass);

        [RunSummary,~,RunStage,RunFigures] = ...
            run_CBS_CGAN_boundary_quality_experiments(variantDir, ...
            Options.workerCount,Options.problemNames,Options.N,Options.D, ...
            Options.maxFE,Options.runIds,Options.targets,RunnerOptions);
        RunHistory = readOptionalConditionTable(fullfile(variantDir, ...
            'history_metrics_all.csv'));

        RunSummary = addConditionVariantColumns( ...
            RunSummary,variant,conditionMode,Options.pairCount);
        RunStage = addConditionVariantColumns( ...
            RunStage,variant,conditionMode,Options.pairCount);
        RunHistory = addConditionVariantColumns( ...
            RunHistory,variant,conditionMode,Options.pairCount);
        RunFigures = addConditionVariantColumns( ...
            RunFigures,variant,conditionMode,Options.pairCount);

        Summary = appendConditionTable(Summary,RunSummary);
        StageMetrics = appendConditionTable(StageMetrics,RunStage);
        HistoryMetrics = appendConditionTable(HistoryMetrics,RunHistory);
        FigureManifest = appendConditionTable(FigureManifest,RunFigures);

        VariantRows(i) = buildConditionVariantRow(variant,conditionMode, ...
            Options.pairCount,variantDir,algorithmParams,RunSummary, ...
            RunStage,RunHistory,RunFigures);
    end

    VariantManifest = struct2table(VariantRows);
    PairedDeltas = buildPairedVariantDeltas(StageMetrics);
    ProblemSummary = buildProblemConditionSummary(StageMetrics);

    writetable(VariantManifest,fullfile(outDir, ...
        'comparison_variant_manifest.csv'));
    writetable(Summary,fullfile(outDir,'comparison_summary.csv'));
    writetable(StageMetrics,fullfile(outDir, ...
        'comparison_stage_metrics_all.csv'));
    writetable(HistoryMetrics,fullfile(outDir, ...
        'comparison_history_metrics_all.csv'));
    writetable(FigureManifest,fullfile(outDir, ...
        'comparison_figure_manifest.csv'));
    writetable(PairedDeltas,fullfile(outDir,'paired_variant_deltas.csv'));
    writetable(ProblemSummary,fullfile(outDir, ...
        'comparison_stage_metrics_by_problem.csv'));
end

function Options = normalizeQueryConditionAblationOptions(Options)
    Options = ensureConditionField(Options,'workerCount',7);
    Options = ensureConditionField(Options,'problemNames', ...
        defaultCBSConditionProblemList());
    Options = ensureConditionField(Options,'N',100);
    Options = ensureConditionField(Options,'D',[]);
    Options = ensureConditionField(Options,'maxFE',100000);
    Options = ensureConditionField(Options,'runIds',1:3);
    Options = ensureConditionField(Options,'targets', ...
        [10000 30000 50000 70000 100000]);
    Options = ensureConditionField(Options,'pairCount',3);
    Options = ensureConditionField(Options,'plotRun',0);
    Options = ensureConditionField(Options,'algorithmClass',"CBS_CGAN");
    Options = ensureConditionField(Options,'baseAlgorithmParams', ...
        {1,1,20,2,50,32,1e-4,2e-4,0,1,1,4,1,2,1,0.10,1});

    Options.workerCount = max(1,round(double(Options.workerCount)));
    Options.problemNames = string(Options.problemNames(:));
    Options.N = max(1,round(double(Options.N)));
    if ~isempty(Options.D)
        Options.D = max(1,round(double(Options.D)));
    end
    Options.maxFE = max(1,round(double(Options.maxFE)));
    Options.runIds = double(Options.runIds(:)');
    Options.runIds = unique(Options.runIds(isfinite(Options.runIds)), ...
        'stable');
    Options.targets = double(Options.targets(:)');
    Options.targets = unique(Options.targets(isfinite(Options.targets) & ...
        Options.targets > 0),'stable');
    Options.pairCount = max(1,round(double(Options.pairCount)));
    Options.plotRun = round(double(Options.plotRun));
    Options.algorithmClass = string(Options.algorithmClass);
    Options.baseAlgorithmParams = normalizeConditionAlgorithmParams( ...
        Options.baseAlgorithmParams);

    assert(~isempty(Options.problemNames), ...
        'CBSConditionAblation:EmptyProblemList', ...
        'Options.problemNames must contain at least one problem.');
    assert(~isempty(Options.runIds), ...
        'CBSConditionAblation:EmptyRuns', ...
        'Options.runIds must contain at least one run.');
    assert(~isempty(Options.targets), ...
        'CBSConditionAblation:EmptyTargets', ...
        'Options.targets must contain at least one FE target.');
end

function Variants = queryConditionVariants()
    Variants = struct( ...
        'variant',{"A_ref_tau","B_ref_y_tau","C_ref_y"}, ...
        'conditionMode',{"ref_tau","ref_y_tau","ref_y"});
end

function problemNames = defaultCBSConditionProblemList()
    problemNames = ["DASCMOP1_BC";"DASCMOP2_BC"; ...
        "DASCMOP4_BC";"DASCMOP5_BC"; ...
        "LIRCMOP5_BC";"LIRCMOP6_BC"; ...
        "LIRCMOP7_BC";"LIRCMOP8_BC"; ...
        "LIRCMOP9_BC";"LIRCMOP10_BC"];
end

function S = ensureConditionField(S,name,value)
    if ~isfield(S,name) || isempty(S.(name))
        S.(name) = value;
    end
end

function params = normalizeConditionAlgorithmParams(params)
    if isnumeric(params)
        params = num2cell(double(params(:)'));
    end
    assert(iscell(params), ...
        'CBSConditionAblation:BadAlgorithmParams', ...
        'Options.baseAlgorithmParams must be a cell array or numeric row.');
    assert(numel(params) >= 17, ...
        'CBSConditionAblation:ShortAlgorithmParams', ...
        ['CBS_CGAN parameter vector must contain at least 17 values; ', ...
        'parameter 13 is maxCandidatePairsPerRef.']);
end

function params = setQueryConditionPairCount(baseParams,pairCount)
    params = normalizeConditionAlgorithmParams(baseParams);
    params{13} = max(1,round(double(pairCount)));
end

function T = addConditionVariantColumns(T,variant,conditionMode,pairCount)
    if isempty(T)
        T = table();
    end
    n = height(T);
    T.variant = repmat(string(variant),n,1);
    if ~ismember('condition_mode',T.Properties.VariableNames)
        T.condition_mode = repmat(string(conditionMode),n,1);
    end
    T.maxCandidatePairsPerRef = repmat(double(pairCount),n,1);
    leading = intersect({'variant','condition_mode', ...
        'maxCandidatePairsPerRef'},T.Properties.VariableNames,'stable');
    if ~isempty(leading) && width(T) > numel(leading)
        T = movevars(T,leading,'Before',1);
    end
end

function A = appendConditionTable(A,B)
    if isempty(A) || width(A) == 0
        A = B;
    elseif ~isempty(B)
        A = [A;B]; %#ok<AGROW>
    end
end

function T = readOptionalConditionTable(fileName)
    if isfile(fileName)
        T = readtable(fileName,'TextType','string','Delimiter',',', ...
            'VariableNamingRule','preserve');
    else
        T = table();
    end
end

function Row = emptyConditionVariantRow()
    Row = struct( ...
        'variant',"", ...
        'condition_mode',"", ...
        'maxCandidatePairsPerRef',NaN, ...
        'variant_folder',"", ...
        'algorithm_params',"", ...
        'run_count',0, ...
        'ok_count',0, ...
        'failed_count',0, ...
        'stage_count',0, ...
        'history_count',0, ...
        'figure_count',0, ...
        'summary_file',"", ...
        'stage_metrics_file',"", ...
        'history_metrics_file',"", ...
        'figure_manifest_file',"");
end

function Row = buildConditionVariantRow(variant,conditionMode,pairCount, ...
        variantDir,algorithmParams,Summary,StageMetrics,HistoryMetrics, ...
        FigureManifest)
    Row = emptyConditionVariantRow();
    Row.variant = string(variant);
    Row.condition_mode = string(conditionMode);
    Row.maxCandidatePairsPerRef = double(pairCount);
    Row.variant_folder = string(variantDir);
    Row.algorithm_params = strjoin(string(cellfun(@conditionParamToText, ...
        algorithmParams,'UniformOutput',false)),',');
    Row.run_count = height(Summary);
    if height(Summary) > 0 && ismember('status',Summary.Properties.VariableNames)
        Row.ok_count = sum(string(Summary.status) == "ok");
        Row.failed_count = sum(string(Summary.status) ~= "ok");
    end
    Row.stage_count = height(StageMetrics);
    Row.history_count = height(HistoryMetrics);
    Row.figure_count = height(FigureManifest);
    Row.summary_file = string(fullfile(variantDir,'run_summary.csv'));
    Row.stage_metrics_file = string(fullfile(variantDir, ...
        'stage_metrics_all.csv'));
    Row.history_metrics_file = string(fullfile(variantDir, ...
        'history_metrics_all.csv'));
    Row.figure_manifest_file = string(fullfile(variantDir, ...
        'figure_manifest.csv'));
end

function text = conditionParamToText(value)
    if isnumeric(value) || islogical(value)
        text = char(string(double(value)));
    elseif isstring(value) || ischar(value)
        text = char(string(value));
    else
        text = char(string(class(value)));
    end
end

function Deltas = buildPairedVariantDeltas(StageMetrics)
    if isempty(StageMetrics) || width(StageMetrics) == 0 || ...
            height(StageMetrics) == 0
        Deltas = struct2table(repmat(emptyDeltaRow(),0,1));
        return;
    end
    keys = StageMetrics(:,{'problem','run','target_FE'});
    [GroupKeys,~,groupIndex] = unique(keys,'rows','stable');
    pairs = ["B_ref_y_tau","A_ref_tau"; ...
        "C_ref_y","B_ref_y_tau"; ...
        "C_ref_y","A_ref_tau"];
    Rows = repmat(emptyDeltaRow(),height(GroupKeys)*size(pairs,1),1);
    count = 0;
    for i = 1 : height(GroupKeys)
        G = StageMetrics(groupIndex == i,:);
        for p = 1 : size(pairs,1)
            count = count + 1;
            Rows(count) = makeDeltaRow(GroupKeys(i,:),G,pairs(p,1), ...
                pairs(p,2));
        end
    end
    Rows = Rows(1:count);
    Deltas = struct2table(Rows);
end

function Row = emptyDeltaRow()
    Row = struct( ...
        'problem',"", ...
        'run',NaN, ...
        'target_FE',NaN, ...
        'variant_to',"", ...
        'variant_from',"", ...
        'delta_query_obj_dist90',NaN, ...
        'delta_boundary_dist90',NaN, ...
        'delta_ref_cover',NaN, ...
        'delta_feasible_rate',NaN, ...
        'delta_missing_ref_query_obj_dist90',NaN, ...
        'delta_large_gap_query_obj_dist90',NaN, ...
        'delta_condition_dim',NaN);
end

function Row = makeDeltaRow(Key,G,toVariant,fromVariant)
    Row = emptyDeltaRow();
    Row.problem = string(Key.problem);
    Row.run = double(Key.run);
    Row.target_FE = double(Key.target_FE);
    Row.variant_to = string(toVariant);
    Row.variant_from = string(fromVariant);
    Row.delta_query_obj_dist90 = deltaMetric(G,toVariant,fromVariant, ...
        'query_obj_dist90');
    Row.delta_boundary_dist90 = deltaMetric(G,toVariant,fromVariant, ...
        'boundary_dist90');
    Row.delta_ref_cover = deltaMetric(G,toVariant,fromVariant, ...
        'ref_cover');
    Row.delta_feasible_rate = deltaMetric(G,toVariant,fromVariant, ...
        'feasible_rate');
    Row.delta_missing_ref_query_obj_dist90 = deltaMetric( ...
        G,toVariant,fromVariant,'missing_ref_query_obj_dist90');
    Row.delta_large_gap_query_obj_dist90 = deltaMetric( ...
        G,toVariant,fromVariant,'large_gap_query_obj_dist90');
    Row.delta_condition_dim = deltaMetric(G,toVariant,fromVariant, ...
        'condition_dim');
end

function value = deltaMetric(T,toVariant,fromVariant,metricName)
    value = variantMetric(T,toVariant,metricName) - ...
        variantMetric(T,fromVariant,metricName);
end

function value = variantMetric(T,variant,metricName)
    value = NaN;
    if ~ismember(metricName,T.Properties.VariableNames)
        return;
    end
    idx = string(T.variant) == string(variant);
    if any(idx)
        values = double(T.(metricName)(idx));
        value = values(1);
    end
end

function Summary = buildProblemConditionSummary(StageMetrics)
    if isempty(StageMetrics) || width(StageMetrics) == 0 || ...
            height(StageMetrics) == 0
        Summary = table();
        return;
    end
    metrics = {'train_count','query_count','feasible_rate', ...
        'boundary_dist90','query_obj_dist90','ref_cover', ...
        'missing_ref_query_obj_dist90','large_gap_query_obj_dist90'};
    metrics = metrics(ismember(metrics,StageMetrics.Properties.VariableNames));
    Summary = groupsummary(StageMetrics, ...
        {'variant','condition_mode','problem'},'median',metrics);
end
