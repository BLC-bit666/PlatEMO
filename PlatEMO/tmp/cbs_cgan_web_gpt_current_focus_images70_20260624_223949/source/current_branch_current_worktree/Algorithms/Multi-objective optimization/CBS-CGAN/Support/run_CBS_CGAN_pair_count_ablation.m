function [Summary,outDir,StageMetrics,FigureManifest,VariantManifest, ...
        HistoryMetrics,AnalysisSummary] = ...
        run_CBS_CGAN_pair_count_ablation(outDir,Options)
%RUN_CBS_CGAN_PAIR_COUNT_ABLATION Compare per-ref boundary-pair counts.
%
% Default experiment:
%   maxCandidatePairsPerRef = [1 3 5], runs = 1:3, plotRun = 3.
% Each variant is delegated to run_CBS_CGAN_boundary_quality_experiments.

    rootDir = fileparts(which('platemo'));
    if isempty(rootDir)
        rootDir = pwd;
    end
    addpath(genpath(rootDir));
    if nargin < 1 || isempty(outDir)
        outDir = fullfile(rootDir,'Data','CBS_CGAN', ...
            ['pair_count_ablation_runs3_plotrun3_', ...
            char(datetime('now','Format','yyyyMMdd_HHmmss'))]);
    end
    if nargin < 2 || isempty(Options)
        Options = struct();
    end
    Options = normalizePairCountAblationOptions(Options);

    if ~isfolder(outDir)
        mkdir(outDir);
    end

    Summary = table();
    StageMetrics = table();
    HistoryMetrics = table();
    FigureManifest = table();
    VariantRows = repmat(emptyVariantRow(),numel(Options.pairCounts),1);

    for i = 1 : numel(Options.pairCounts)
        pairCount = Options.pairCounts(i);
        variant = sprintf('pair%d',pairCount);
        variantDir = fullfile(outDir,variant);
        algorithmParams = setMaxCandidatePairsPerRef( ...
            Options.baseAlgorithmParams,pairCount);
        RunnerOptions = struct( ...
            'plotRun',Options.plotRun, ...
            'algorithmParams',{algorithmParams}, ...
            'algorithmClass',Options.algorithmClass);

        [RunSummary,~,RunStage,RunFigures] = ...
            run_CBS_CGAN_boundary_quality_experiments(variantDir, ...
            Options.workerCount,Options.problemNames,Options.N,Options.D, ...
            Options.maxFE,Options.runIds,Options.targets,RunnerOptions);
        RunHistory = readOptionalTable(fullfile(variantDir, ...
            'history_metrics_all.csv'));

        RunSummary = addVariantColumns(RunSummary,variant,pairCount);
        RunStage = addVariantColumns(RunStage,variant,pairCount);
        RunHistory = addVariantColumns(RunHistory,variant,pairCount);
        RunFigures = addVariantColumns(RunFigures,variant,pairCount);

        Summary = appendTable(Summary,RunSummary);
        StageMetrics = appendTable(StageMetrics,RunStage);
        HistoryMetrics = appendTable(HistoryMetrics,RunHistory);
        FigureManifest = appendTable(FigureManifest,RunFigures);

        VariantRows(i) = buildVariantRow(variant,pairCount,variantDir, ...
            algorithmParams,RunSummary,RunStage,RunHistory,RunFigures);
    end

    VariantManifest = struct2table(VariantRows);
    AnalysisSummary = buildComparisonAnalysisSummary(StageMetrics, ...
        FigureManifest);

    writetable(VariantManifest,fullfile(outDir, ...
        'comparison_variant_manifest.csv'));
    writetable(Summary,fullfile(outDir,'comparison_summary.csv'));
    writetable(StageMetrics,fullfile(outDir, ...
        'comparison_stage_metrics_all.csv'));
    writetable(HistoryMetrics,fullfile(outDir, ...
        'comparison_history_metrics_all.csv'));
    writetable(FigureManifest,fullfile(outDir, ...
        'comparison_figure_manifest.csv'));
    writetable(AnalysisSummary,fullfile(outDir, ...
        'comparison_analysis_summary.csv'));
end

function Options = normalizePairCountAblationOptions(Options)
    Options = ensureField(Options,'workerCount',8);
    Options = ensureField(Options,'problemNames',defaultCBSProblemListLocal());
    Options = ensureField(Options,'N',100);
    Options = ensureField(Options,'D',[]);
    Options = ensureField(Options,'maxFE',100000);
    Options = ensureField(Options,'runIds',1:3);
    Options = ensureField(Options,'targets', ...
        [10000 30000 50000 70000 100000]);
    Options = ensureField(Options,'pairCounts',[1 3 5]);
    Options = ensureField(Options,'plotRun',3);
    Options = ensureField(Options,'algorithmClass',"CBS_CGAN");
    Options = ensureField(Options,'baseAlgorithmParams', ...
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
    Options.pairCounts = double(Options.pairCounts(:)');
    Options.pairCounts = unique(max(1,round( ...
        Options.pairCounts(isfinite(Options.pairCounts)))),'stable');
    Options.plotRun = round(double(Options.plotRun));
    Options.algorithmClass = string(Options.algorithmClass);
    Options.baseAlgorithmParams = normalizeAlgorithmParams( ...
        Options.baseAlgorithmParams);

    assert(~isempty(Options.problemNames), ...
        'CBSPairAblation:EmptyProblemList', ...
        'Options.problemNames must contain at least one problem.');
    assert(~isempty(Options.runIds), ...
        'CBSPairAblation:EmptyRuns', ...
        'Options.runIds must contain at least one run.');
    assert(~isempty(Options.targets), ...
        'CBSPairAblation:EmptyTargets', ...
        'Options.targets must contain at least one FE target.');
    assert(~isempty(Options.pairCounts), ...
        'CBSPairAblation:EmptyPairCounts', ...
        'Options.pairCounts must contain at least one positive value.');
end

function problemNames = defaultCBSProblemListLocal()
    problemNames = ["DASCMOP1_BC";"DASCMOP2_BC"; ...
        "DASCMOP4_BC";"DASCMOP5_BC"; ...
        "LIRCMOP5_BC";"LIRCMOP6_BC"; ...
        "LIRCMOP7_BC";"LIRCMOP8_BC"; ...
        "LIRCMOP9_BC";"LIRCMOP10_BC"];
end

function S = ensureField(S,name,value)
    if ~isfield(S,name) || isempty(S.(name))
        S.(name) = value;
    end
end

function params = normalizeAlgorithmParams(params)
    if isnumeric(params)
        params = num2cell(double(params(:)'));
    end
    assert(iscell(params), ...
        'CBSPairAblation:BadAlgorithmParams', ...
        'Options.baseAlgorithmParams must be a cell array or numeric row.');
    assert(numel(params) >= 17, ...
        'CBSPairAblation:ShortAlgorithmParams', ...
        ['CBS_CGAN parameter vector must contain at least 17 values; ', ...
        'parameter 13 is maxCandidatePairsPerRef.']);
end

function params = setMaxCandidatePairsPerRef(baseParams,pairCount)
    params = normalizeAlgorithmParams(baseParams);
    params{13} = max(1,round(double(pairCount)));
end

function T = addVariantColumns(T,variant,pairCount)
    if isempty(T)
        T = table();
    end
    n = height(T);
    T.variant = repmat(string(variant),n,1);
    T.maxCandidatePairsPerRef = repmat(double(pairCount),n,1);
    if width(T) > 2
        T = movevars(T,{'variant','maxCandidatePairsPerRef'}, ...
            'Before',1);
    end
end

function A = appendTable(A,B)
    if isempty(A) || width(A) == 0
        A = B;
    elseif ~isempty(B)
        A = [A;B]; %#ok<AGROW>
    end
end

function T = readOptionalTable(fileName)
    if isfile(fileName)
        T = readtable(fileName,'TextType','string','Delimiter',',', ...
            'VariableNamingRule','preserve');
    else
        T = table();
    end
end

function Row = emptyVariantRow()
    Row = struct( ...
        'variant',"", ...
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

function Row = buildVariantRow(variant,pairCount,variantDir,algorithmParams, ...
        Summary,StageMetrics,HistoryMetrics,FigureManifest)
    Row = emptyVariantRow();
    Row.variant = string(variant);
    Row.maxCandidatePairsPerRef = double(pairCount);
    Row.variant_folder = string(variantDir);
    Row.algorithm_params = strjoin(string(cellfun(@paramToText, ...
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

function text = paramToText(value)
    if isnumeric(value) || islogical(value)
        text = char(string(double(value)));
    elseif isstring(value) || ischar(value)
        text = char(string(value));
    else
        text = char(string(class(value)));
    end
end

function Analysis = buildComparisonAnalysisSummary(StageMetrics,FigureManifest)
    if isempty(StageMetrics) || width(StageMetrics) == 0 || ...
            height(StageMetrics) == 0
        Analysis = struct2table(repmat(emptyAnalysisRow(),0,1));
        return;
    end
    [GroupKeys,~,groupIndex] = unique( ...
        StageMetrics(:,{'variant','maxCandidatePairsPerRef','problem'}), ...
        'rows','stable');
    Rows = repmat(emptyAnalysisRow(),height(GroupKeys),1);
    for i = 1 : height(GroupKeys)
        idx = groupIndex == i;
        T = StageMetrics(idx,:);
        Rows(i).variant = string(GroupKeys.variant(i));
        Rows(i).maxCandidatePairsPerRef = ...
            double(GroupKeys.maxCandidatePairsPerRef(i));
        Rows(i).problem = string(GroupKeys.problem(i));
        Rows(i).stage_count = height(T);
        Rows(i).train_count_med = finiteMedian(T.train_count);
        Rows(i).train_count_p90 = finitePercentile(T.train_count,90);
        Rows(i).train_count_max = finiteMax(T.train_count);
        Rows(i).ref_cov_med = finiteMedian(T.bmem_ref_coverage);
        Rows(i).param_ratio_med = finiteMedian(T.train_param_ratio);
        Rows(i).sample_reuse_med = finiteMedian(T.gan_sample_reuse);
        Rows(i).raw_generated_med = finiteMedian(T.raw_generated_count);
        Rows(i).feasible_rate_med = finiteMedian(T.feasible_rate);
        Rows(i).boundary_dist90_med = finiteMedian(T.boundary_dist90);
        Rows(i).pair_margin50_med = finiteMedian(T.pair_margin50);
        Rows(i).missing_ref_query_total = finiteSum( ...
            T.missing_ref_query_count);
        Rows(i).large_gap_query_total = finiteSum(T.large_gap_query_count);
        Rows(i).figure_count = figureCountForGroup(FigureManifest, ...
            Rows(i).variant,Rows(i).problem);
    end
    Analysis = struct2table(Rows);
end

function Row = emptyAnalysisRow()
    Row = struct( ...
        'variant',"", ...
        'maxCandidatePairsPerRef',NaN, ...
        'problem',"", ...
        'stage_count',0, ...
        'train_count_med',NaN, ...
        'train_count_p90',NaN, ...
        'train_count_max',NaN, ...
        'ref_cov_med',NaN, ...
        'param_ratio_med',NaN, ...
        'sample_reuse_med',NaN, ...
        'raw_generated_med',NaN, ...
        'feasible_rate_med',NaN, ...
        'boundary_dist90_med',NaN, ...
        'pair_margin50_med',NaN, ...
        'missing_ref_query_total',0, ...
        'large_gap_query_total',0, ...
        'figure_count',0);
end

function value = finiteMedian(values)
    values = finiteValues(values);
    if isempty(values)
        value = NaN;
    else
        value = median(values);
    end
end

function value = finitePercentile(values,p)
    values = sort(finiteValues(values));
    if isempty(values)
        value = NaN;
        return;
    end
    rank = 1 + (numel(values) - 1) * double(p) / 100;
    lo = floor(rank);
    hi = ceil(rank);
    if lo == hi
        value = values(lo);
    else
        value = values(lo) * (hi - rank) + values(hi) * (rank - lo);
    end
end

function value = finiteMax(values)
    values = finiteValues(values);
    if isempty(values)
        value = NaN;
    else
        value = max(values);
    end
end

function value = finiteSum(values)
    values = finiteValues(values);
    if isempty(values)
        value = 0;
    else
        value = sum(values);
    end
end

function values = finiteValues(values)
    values = double(values(:));
    values = values(isfinite(values));
end

function count = figureCountForGroup(FigureManifest,variant,problem)
    count = 0;
    if isempty(FigureManifest) || width(FigureManifest) == 0 || ...
            height(FigureManifest) == 0
        return;
    end
    if ~all(ismember(["variant","problem"], ...
            string(FigureManifest.Properties.VariableNames)))
        return;
    end
    count = sum(string(FigureManifest.variant) == string(variant) & ...
        string(FigureManifest.problem) == string(problem));
end
