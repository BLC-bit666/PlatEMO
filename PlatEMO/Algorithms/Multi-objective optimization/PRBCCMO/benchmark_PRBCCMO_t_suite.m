function Results = benchmark_PRBCCMO_t_suite(problemNames,runIds,outCsv,N,maxFE,workerCount)
% Run PRBCCMO_t and write the minimal core-observation summary.

    if nargin < 1 || isempty(problemNames)
        problemNames = defaultProblemList();
    end
    if nargin < 2 || isempty(runIds)
        runIds = 1 : 5;
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
    if nargin < 6 || isempty(workerCount)
        workerCount = 1;
    end

    problemNames = cellstr(string(problemNames(:)));
    runIds = double(runIds(:)');
    outDir = fileparts(outCsv);
    if ~isempty(outDir) && ~isfolder(outDir)
        mkdir(outDir);
    end

    [TaskProblem,TaskRun] = ndgrid(1:numel(problemNames),runIds);
    TaskProblem = TaskProblem(:);
    TaskRun = TaskRun(:);
    TaskProblemNames = problemNames(TaskProblem);
    Rows = repmat(emptyBenchmarkRow(),numel(TaskProblem),1);

    workerCount = max(1,round(double(workerCount)));
    if workerCount > 1
        ensureParallelPool(workerCount);
        parfor task = 1 : numel(TaskProblem)
            Rows(task) = runOneBenchmark(TaskProblemNames{task},TaskRun(task),N,maxFE);
        end
    else
        for task = 1 : numel(TaskProblem)
            Rows(task) = runOneBenchmark(TaskProblemNames{task},TaskRun(task),N,maxFE);
        end
    end

    Results = struct2table(Rows);
    writetable(Results,outCsv);
end

function Row = runOneBenchmark(problemName,runId,N,maxFE)
    rng(runId,'twister');
    Algorithm = PRBCCMO_t('save',0,'run',runId,'outputFcn',@(varargin)[]);
    ProblemConstructor = str2func(problemName);
    Problem = ProblemConstructor('N',N,'maxFE',maxFE);
    Algorithm.Solve(Problem);
    Trace = summarize_PRBCCMO_t_run(Algorithm.metric.analysis_folder);
    Row = composeBenchmarkRow(Algorithm,Trace);
end

function ensureParallelPool(workerCount)
    pool = gcp('nocreate');
    if ~isempty(pool) && pool.NumWorkers ~= workerCount
        delete(pool);
        pool = [];
    end
    if isempty(pool)
        parpool('local',workerCount);
    end
end

function Row = composeBenchmarkRow(Algorithm,Trace)
    Row = emptyBenchmarkRow();
    Row.problem = Trace.problem;
    Row.family = Trace.family;
    Row.run = Trace.run;
    Row.runtime = double(Algorithm.metric.runtime);
    Row.analysis_folder = Trace.analysis_folder;
    Row.analysis_core_csv = Trace.analysis_core_csv;
    Row.final_fe = Trace.final_fe;
    Row.final_b_pair_count = Trace.final_b_pair_count;
    Row.final_b_mean_pair_gap = Trace.final_b_mean_pair_gap;
    Row.mean_b_mean_pair_gap = Trace.mean_b_mean_pair_gap;
    Row.p90_b_mean_pair_gap = Trace.p90_b_mean_pair_gap;
    Row.final_b_p90_pair_gap = Trace.final_b_p90_pair_gap;
    Row.final_b_mean_infeasible_boundary_dist = Trace.final_b_mean_infeasible_boundary_dist;
    Row.mean_b_mean_infeasible_boundary_dist = Trace.mean_b_mean_infeasible_boundary_dist;
    Row.final_b_far_infeasible_boundary_ratio = Trace.final_b_far_infeasible_boundary_ratio;
    Row.mean_b_far_infeasible_boundary_ratio = Trace.mean_b_far_infeasible_boundary_ratio;
    Row.final_b_inf_unconstrained_src_ratio = Trace.final_b_inf_unconstrained_src_ratio;
    Row.mean_b_inf_unconstrained_src_ratio = Trace.mean_b_inf_unconstrained_src_ratio;
    Row.final_b_overlap_popc_ratio = Trace.final_b_overlap_popc_ratio;
    Row.mean_b_overlap_popc_ratio = Trace.mean_b_overlap_popc_ratio;
    Row.final_b_overlap_popu_ratio = Trace.final_b_overlap_popu_ratio;
    Row.mean_b_overlap_popu_ratio = Trace.mean_b_overlap_popu_ratio;
    Row.final_b_inf_src_popu = Trace.final_b_inf_src_popu;
    Row.final_b_inf_src_offu = Trace.final_b_inf_src_offu;
    Row.final_b_inf_src_refinement = Trace.final_b_inf_src_refinement;
    Row.final_b_active_sector_ratio = Trace.final_b_active_sector_ratio;
    Row.mean_b_active_sector_ratio = Trace.mean_b_active_sector_ratio;
    Row.mean_b_changed_pair_count = Trace.mean_b_changed_pair_count;
    Row.mean_b_changed_pair_ratio = Trace.mean_b_changed_pair_ratio;
    Row.mean_b_replaced_pair_count = Trace.mean_b_replaced_pair_count;
    Row.mean_b_contracted_pair_count = Trace.mean_b_contracted_pair_count;
    Row.mean_b_global_change_pair_ratio = Trace.mean_b_global_change_pair_ratio;
    Row.mean_b_contraction_pair_ratio = Trace.mean_b_contraction_pair_ratio;
    Row.mlp_due_events = Trace.mlp_due_events;
    Row.mlp_can_train_events = Trace.mlp_can_train_events;
    Row.mlp_trained_events = Trace.mlp_trained_events;
    Row.mlp_degraded_events = Trace.mlp_degraded_events;
    Row.final_mlp_train_acc = Trace.final_mlp_train_acc;
    Row.mean_mlp_train_acc = Trace.mean_mlp_train_acc;
    Row.final_train_size = Trace.final_train_size;
    Row.mean_train_size = Trace.mean_train_size;
    Row.final_train_pos = Trace.final_train_pos;
    Row.final_train_neg = Trace.final_train_neg;
    Row.final_train_src_b = Trace.final_train_src_b;
    Row.final_train_src_refinement = Trace.final_train_src_refinement;
    Row.final_train_src_boundary_band = Trace.final_train_src_boundary_band;
    Row.inf_selection_generations = Trace.inf_selection_generations;
    Row.inf_selected_total = Trace.inf_selected_total;
    Row.mean_inf_pool_mean_prob = Trace.mean_inf_pool_mean_prob;
    Row.mean_inf_selected_mean_prob = Trace.mean_inf_selected_mean_prob;
    Row.mean_inf_prob_gain = Trace.mean_inf_prob_gain;
    Row.pos_rate_inf_prob_gain = Trace.pos_rate_inf_prob_gain;
    Row.inf_carry_generations = Trace.inf_carry_generations;
    Row.inf_carry_selected_total = Trace.inf_carry_selected_total;
    Row.mean_inf_carry_pool_mean_prob = Trace.mean_inf_carry_pool_mean_prob;
    Row.mean_inf_carry_selected_mean_prob = Trace.mean_inf_carry_selected_mean_prob;
    Row.mean_inf_carry_prob_gain = Trace.mean_inf_carry_prob_gain;
    Row.pos_rate_inf_carry_prob_gain = Trace.pos_rate_inf_carry_prob_gain;
end

function Row = emptyBenchmarkRow()
    Row = struct( ...
        'problem',"", ...
        'family',"", ...
        'run',NaN, ...
        'runtime',NaN, ...
        'analysis_folder',"", ...
        'analysis_core_csv',"", ...
        'final_fe',NaN, ...
        'final_b_pair_count',NaN, ...
        'final_b_mean_pair_gap',NaN, ...
        'mean_b_mean_pair_gap',NaN, ...
        'p90_b_mean_pair_gap',NaN, ...
        'final_b_p90_pair_gap',NaN, ...
        'final_b_mean_infeasible_boundary_dist',NaN, ...
        'mean_b_mean_infeasible_boundary_dist',NaN, ...
        'final_b_far_infeasible_boundary_ratio',NaN, ...
        'mean_b_far_infeasible_boundary_ratio',NaN, ...
        'final_b_inf_unconstrained_src_ratio',NaN, ...
        'mean_b_inf_unconstrained_src_ratio',NaN, ...
        'final_b_overlap_popc_ratio',NaN, ...
        'mean_b_overlap_popc_ratio',NaN, ...
        'final_b_overlap_popu_ratio',NaN, ...
        'mean_b_overlap_popu_ratio',NaN, ...
        'final_b_inf_src_popu',NaN, ...
        'final_b_inf_src_offu',NaN, ...
        'final_b_inf_src_refinement',NaN, ...
        'final_b_active_sector_ratio',NaN, ...
        'mean_b_active_sector_ratio',NaN, ...
        'mean_b_changed_pair_count',NaN, ...
        'mean_b_changed_pair_ratio',NaN, ...
        'mean_b_replaced_pair_count',NaN, ...
        'mean_b_contracted_pair_count',NaN, ...
        'mean_b_global_change_pair_ratio',NaN, ...
        'mean_b_contraction_pair_ratio',NaN, ...
        'mlp_due_events',NaN, ...
        'mlp_can_train_events',NaN, ...
        'mlp_trained_events',NaN, ...
        'mlp_degraded_events',NaN, ...
        'final_mlp_train_acc',NaN, ...
        'mean_mlp_train_acc',NaN, ...
        'final_train_size',NaN, ...
        'mean_train_size',NaN, ...
        'final_train_pos',NaN, ...
        'final_train_neg',NaN, ...
        'final_train_src_b',NaN, ...
        'final_train_src_refinement',NaN, ...
        'final_train_src_boundary_band',NaN, ...
        'inf_selection_generations',NaN, ...
        'inf_selected_total',NaN, ...
        'mean_inf_pool_mean_prob',NaN, ...
        'mean_inf_selected_mean_prob',NaN, ...
        'mean_inf_prob_gain',NaN, ...
        'pos_rate_inf_prob_gain',NaN, ...
        'inf_carry_generations',NaN, ...
        'inf_carry_selected_total',NaN, ...
        'mean_inf_carry_pool_mean_prob',NaN, ...
        'mean_inf_carry_selected_mean_prob',NaN, ...
        'mean_inf_carry_prob_gain',NaN, ...
        'pos_rate_inf_carry_prob_gain',NaN);
end

function names = defaultProblemList()
    rootDir = fileparts(which('platemo'));
    specs = { ...
        fullfile(rootDir,'Problems','Multi-objective optimization','DAS-CMOP_BC'), ...
        fullfile(rootDir,'Problems','Multi-objective optimization','LIR-CMOP_BC')};
    namesBySpec = cell(numel(specs),1);
    for i = 1 : numel(specs)
        files = dir(fullfile(specs{i},'*_BC.m'));
        fileNames = erase({files.name},'.m');
        [~,ord] = sort(problemOrder(fileNames));
        namesBySpec{i} = reshape(fileNames(ord),[],1);
    end
    names = vertcat(namesBySpec{:});
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
