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
        Core = numericizeTable(readtable(coreFile,'TextType','string'));
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
    Row.target_fe = double(targetFE);
    Row.actual_fe = double(actualFE);
    Row.generation = double(generation);
    Row.b_mean_pair_gap = valueAt(Core,'b_mean_pair_gap',idx);
    Row.b_p90_pair_gap = valueAt(Core,'b_p90_pair_gap',idx);
    Row.mlp_train_acc = valueAt(Core,'mlp_train_acc',idx);
    Row.train_size = valueAt(Core,'train_size',idx);
    Row.train_src_refinement = valueAt(Core,'train_src_refinement',idx);
    Row.train_src_boundary_band = valueAt(Core,'train_src_boundary_band',idx);
    Row.inf_selected = valueAt(Core,'inf_selected',idx);
    Row.inf_prob_gain = valueAt(Core,'inf_prob_gain',idx);
    Row.inf_carry_selected = valueAt(Core,'inf_carry_selected',idx);
    Row.inf_carry_prob_gain = valueAt(Core,'inf_carry_prob_gain',idx);
    Row.png_file = string(pngFile);
end

function Row = emptyFigureRow()
    Row = struct( ...
        'problem',"", ...
        'run',NaN, ...
        'target_fe',NaN, ...
        'actual_fe',NaN, ...
        'generation',NaN, ...
        'b_mean_pair_gap',NaN, ...
        'b_p90_pair_gap',NaN, ...
        'mlp_train_acc',NaN, ...
        'train_size',NaN, ...
        'train_src_refinement',NaN, ...
        'train_src_boundary_band',NaN, ...
        'inf_selected',NaN, ...
        'inf_prob_gain',NaN, ...
        'inf_carry_selected',NaN, ...
        'inf_carry_prob_gain',NaN, ...
        'png_file',"");
end

function [generation,actualFE,idx] = nearestGenerationAtFE(T,targetFE)
    generation = NaN;
    actualFE = NaN;
    idx = NaN;
    if isempty(T) || ~hasColumns(T,{'generation','fe'})
        return;
    end
    [~,idx] = min(abs(double(T.fe) - double(targetFE)));
    generation = double(T.generation(idx));
    actualFE = double(T.fe(idx));
end

function value = valueAt(T,Name,idx)
    if isnan(idx) || ~hasColumns(T,{Name})
        value = NaN;
    else
        value = double(T.(Name)(idx));
    end
end

function Flag = hasColumns(T,Names)
    Flag = all(ismember(string(Names),string(T.Properties.VariableNames)));
end

function T = numericizeTable(T)
    Names = T.Properties.VariableNames;
    for i = 1 : numel(Names)
        Value = T.(Names{i});
        if iscell(Value) || isstring(Value)
            Num = str2double(string(Value));
            if any(~isnan(Num))
                T.(Names{i}) = Num;
            end
        elseif islogical(Value)
            T.(Names{i}) = double(Value);
        end
    end
end
