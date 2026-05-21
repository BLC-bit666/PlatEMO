function [Benchmark,FigureSummary] = run_PRBCCMO_t_motivation_boundary_figures( ...
    problemNames,runIds,N,maxFE,outDir,reuseBenchmark,targetFEs)
% Prepare figures for the three PRBCCMO_t core observations.

    if nargin < 1 || isempty(problemNames)
        problemNames = {'LIRCMOP3_BC','LIRCMOP4_BC','DASCMOP1_BC','DASCMOP3_BC'};
    end
    if nargin < 2 || isempty(runIds)
        runIds = 1 : 5;
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
            ['motivation_core_metrics_',char(datetime('now','Format','yyyyMMdd_HHmmss'))]);
    end
    if nargin < 6 || isempty(reuseBenchmark)
        reuseBenchmark = false;
    end
    if nargin < 7 || isempty(targetFEs)
        targetFEs = [0.25 0.50 0.75 1.00]*maxFE;
    end

    targetFEs = unique(round(double(targetFEs(:)')));
    targetFEs = targetFEs(targetFEs > 0 & targetFEs <= maxFE);
    if isempty(targetFEs)
        targetFEs = maxFE;
    end
    if ~isfolder(outDir)
        mkdir(outDir);
    end

    benchCsv = fullfile(outDir,'motivation_benchmark.csv');
    if reuseBenchmark && isfile(benchCsv)
        Benchmark = readtable(benchCsv,'TextType','string');
    else
        Benchmark = benchmark_PRBCCMO_t_suite(problemNames,runIds,benchCsv,N,maxFE);
    end

    rows = repmat(emptyFigureRow(),height(Benchmark)*numel(targetFEs),1);
    row = 0;
    for i = 1 : height(Benchmark)
        analysisFolder = char(string(Benchmark.analysis_folder(i)));
        coreFile = fullfile(analysisFolder,'core_metrics.csv');
        if ~isfile(coreFile)
            continue;
        end
        Core = PRBCCMOUtils.numericizeTable(readtable(coreFile,'TextType','string'));
        plotDir = fullfile(outDir,'figures',char(string(Benchmark.problem(i))),sprintf('run%d',double(Benchmark.run(i))));
        for targetFE = targetFEs
            [generation,actualFE,idx] = nearestGenerationAtFE(Core,targetFE);
            if isnan(generation)
                continue;
            end
            pngFile = plot_PRBCCMO_t_generation_region_evidence(analysisFolder,plotDir,generation);
            row = row + 1;
            rows(row) = composeFigureRow(Benchmark(i,:),Core,idx,targetFE,actualFE,generation,pngFile);
        end
    end

    FigureSummary = struct2table(rows(1:row));
    writetable(FigureSummary,fullfile(outDir,'motivation_figure_summary.csv'));
end

function Row = composeFigureRow(BenchmarkRow,Core,idx,targetFE,actualFE,generation,pngFile)
    Row = emptyFigureRow();
    Row.problem = string(BenchmarkRow.problem);
    Row.run = double(BenchmarkRow.run);
    Row.variant = PRBCCMOUtils.valueAtString(BenchmarkRow,'variant',"");
    Row.useMLP = PRBCCMOUtils.valueAt(BenchmarkRow,'useMLP',1);
    Row.igd = PRBCCMOUtils.valueAt(BenchmarkRow,'igd',1);
    Row.target_fe = double(targetFE);
    Row.actual_fe = double(actualFE);
    Row.generation = double(generation);
    Row.b_mean_pair_gap = PRBCCMOUtils.valueAt(Core,'b_mean_pair_gap',idx);
    Row.b_p90_pair_gap = PRBCCMOUtils.valueAt(Core,'b_p90_pair_gap',idx);
    Row.b_mean_mid_scalar = PRBCCMOUtils.valueAt(Core,'b_mean_mid_scalar',idx);
    Row.b_p90_mid_scalar = PRBCCMOUtils.valueAt(Core,'b_p90_mid_scalar',idx);
    Row.mlp_trained = PRBCCMOUtils.valueAt(Core,'mlp_trained',idx);
    Row.train_size = PRBCCMOUtils.valueAt(Core,'train_size',idx);
    Row.train_b_core_count = PRBCCMOUtils.valueAt(Core,'train_b_core_count',idx);
    Row.train_refinement_count = PRBCCMOUtils.valueAt(Core,'train_refinement_count',idx);
    Row.train_current_pop_count = PRBCCMOUtils.valueAt(Core,'train_current_pop_count',idx);
    Row.mlp_prob_curr_feasible = PRBCCMOUtils.valueAt(Core,'mlp_prob_curr_feasible',idx);
    Row.mlp_prob_curr_infeasible = PRBCCMOUtils.valueAt(Core,'mlp_prob_curr_infeasible',idx);
    Row.prob_accepted_refinement = PRBCCMOUtils.valueAt(Core,'prob_accepted_refinement',idx);
    Row.prob_rejected_refinement = PRBCCMOUtils.valueAt(Core,'prob_rejected_refinement',idx);
    Row.mlp_two_inf_tournament_rate = PRBCCMOUtils.valueAt(Core,'mlp_two_inf_tournament_rate',idx);
    Row.mlp_effective_win_rate = PRBCCMOUtils.valueAt(Core,'mlp_effective_win_rate',idx);
    Row.mlp_primary_resolution_rate = PRBCCMOUtils.valueAt(Core,'mlp_primary_resolution_rate',idx);
    Row.b_fallback_rate = PRBCCMOUtils.valueAt(Core,'b_fallback_rate',idx);
    Row.inf_ranked_count = PRBCCMOUtils.valueAt(Core,'inf_ranked_count',idx);
    Row.inf_selected = PRBCCMOUtils.valueAt(Core,'inf_selected',idx);
    Row.inf_utility_gain = PRBCCMOUtils.valueAt(Core,'inf_utility_gain',idx);
    Row.png_file = string(pngFile);
end

function Row = emptyFigureRow()
    Row = struct( ...
        'problem',"", ...
        'run',NaN, ...
        'variant',"", ...
        'useMLP',NaN, ...
        'igd',NaN, ...
        'target_fe',NaN, ...
        'actual_fe',NaN, ...
        'generation',NaN, ...
        'b_mean_pair_gap',NaN, ...
        'b_p90_pair_gap',NaN, ...
        'b_mean_mid_scalar',NaN, ...
        'b_p90_mid_scalar',NaN, ...
        'mlp_trained',NaN, ...
        'train_size',NaN, ...
        'train_b_core_count',NaN, ...
        'train_refinement_count',NaN, ...
        'train_current_pop_count',NaN, ...
        'mlp_prob_curr_feasible',NaN, ...
        'mlp_prob_curr_infeasible',NaN, ...
        'prob_accepted_refinement',NaN, ...
        'prob_rejected_refinement',NaN, ...
        'mlp_two_inf_tournament_rate',NaN, ...
        'mlp_effective_win_rate',NaN, ...
        'mlp_primary_resolution_rate',NaN, ...
        'b_fallback_rate',NaN, ...
        'inf_ranked_count',NaN, ...
        'inf_selected',NaN, ...
        'inf_utility_gain',NaN, ...
        'png_file',"");
end

function [generation,actualFE,idx] = nearestGenerationAtFE(T,targetFE)
    generation = NaN;
    actualFE = NaN;
    idx = NaN;
    if isempty(T) || ~PRBCCMOUtils.hasColumns(T,{'generation','fe'})
        return;
    end
    [~,idx] = min(abs(double(T.fe) - double(targetFE)));
    generation = double(T.generation(idx));
    actualFE = double(T.fe(idx));
end
