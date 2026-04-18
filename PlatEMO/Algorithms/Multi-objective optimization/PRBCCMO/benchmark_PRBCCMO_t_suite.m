function Results = benchmark_PRBCCMO_t_suite(problemNames,runs,outCsv,N,maxFE)
% Run traced PRBCCMO_t benchmarks and emit a run-level CSV summary.

    if nargin < 1 || isempty(problemNames)
        problemNames = defaultProblemList();
    end
    if nargin < 2 || isempty(runs)
        runs = 6;
    end
    if nargin < 3 || isempty(outCsv)
        rootDir = fileparts(which('platemo'));
        outCsv = fullfile(rootDir,'Data','PRBCCMO_t','benchmark_runs.csv');
    end
    if nargin < 4 || isempty(N)
        N = 100;
    end
    if nargin < 5 || isempty(maxFE)
        maxFE = 200000;
    end

    problemNames = cellstr(string(problemNames(:)));
    outDir = fileparts(outCsv);
    if ~isempty(outDir) && ~isfolder(outDir)
        mkdir(outDir);
    end

    Rows = cell(numel(problemNames)*runs,18);
    row = 0;
    for p = 1 : numel(problemNames)
        problemName = problemNames{p};
        family = familyOfProblem(problemName);
        for r = 1 : runs
            rng(r,'twister');
            Algorithm = PRBCCMO_t('save',0,'run',r);
            Problem   = feval(problemName,'N',N,'maxFE',maxFE);
            Algorithm.Solve(Problem);
            Trace = summarize_PRBCCMO_t_run(Algorithm.metric.analysis_folder);

            row = row + 1;
            Rows(row,:) = { ...
                string(problemName),string(family),r, ...
                metricLast(Algorithm,'IGD'),metricLast(Algorithm,'HV'), ...
                metricLast(Algorithm,'Feasible_rate'),Algorithm.metric.runtime, ...
                string(Algorithm.metric.analysis_folder), ...
                Trace.final_boundary_selected,Trace.final_boundary_sector_coverage, ...
                Trace.final_boundary_lowmargin_pair_hit,Trace.final_boundary_seg_cross_dist, ...
                Trace.final_b_size,Trace.final_b_sector_coverage, ...
                Trace.final_train_pair_count,Trace.final_train_bal_acc, ...
                Trace.first_trained_generation,Trace.stable3_generation};
        end
    end

    Results = cell2table(Rows(1:row,:), ...
        'VariableNames',{ ...
        'problem','family','run','igd','hv','feasible_rate','runtime', ...
        'analysis_folder','final_boundary_selected','final_boundary_sector_coverage', ...
        'final_boundary_lowmargin_pair_hit','final_boundary_seg_cross_dist','final_b_size','final_b_sector_coverage', ...
        'final_train_pair_count','final_train_bal_acc','first_trained_generation','stable3_generation'});
    writetable(Results,outCsv);
end

function value = metricLast(Algorithm,metricName)
    value = Algorithm.CalMetric(metricName);
    if iscell(value)
        value = value{end};
    elseif numel(value) > 1
        value = value(end);
    end
    value = double(value);
end

function names = defaultProblemList()
    rootDir = fileparts(which('platemo'));
    specs = { ...
        fullfile(rootDir,'Problems','Multi-objective optimization','DAS-CMOP_BC'), ...
        fullfile(rootDir,'Problems','Multi-objective optimization','LIR-CMOP_BC')};
    names = {};
    for i = 1 : numel(specs)
        files = dir(fullfile(specs{i},'*_BC.m'));
        fileNames = erase({files.name},'.m');
        [~,ord] = sort(problemOrder(fileNames));
        names = [names; reshape(fileNames(ord),[],1)]; %#ok<AGROW>
    end
end

function order = problemOrder(names)
    order = inf(numel(names),1);
    for i = 1 : numel(names)
        tokens = regexp(names{i},'(\d+)','tokens','once');
        if ~isempty(tokens)
            order(i) = str2double(tokens{1});
        end
    end
end

function family = familyOfProblem(problem)
    if startsWith(problem,"DASCMOP")
        family = "DASCMOP_BC";
    elseif startsWith(problem,"LIRCMOP")
        family = "LIRCMOP_BC";
    else
        family = "OTHER";
    end
end
