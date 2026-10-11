clc; clear;

%% Run the selected algorithm; GANCMO uses the fixed 8/8 mainline.
nWorker = 10;
popSize = 100;
maxFE   = 150000;
runs    = 1:30;
algName = 'GANCMO';
proNames = {
    'LIRCMOP1_BC','LIRCMOP2_BC','LIRCMOP3_BC','LIRCMOP4_BC', ...
    'LIRCMOP5_BC','LIRCMOP6_BC','LIRCMOP7_BC','LIRCMOP8_BC', ...
    'LIRCMOP9_BC','LIRCMOP10_BC','LIRCMOP11_BC','LIRCMOP12_BC', ...
    'LIRCMOP13_BC','LIRCMOP14_BC'};

rootPath = fileparts(mfilename('fullpath'));
runExperiment(rootPath,nWorker,popSize,maxFE,runs,algName,proNames);

function runExperiment(rootPath,nWorker,popSize,maxFE,runs,algName,proNames)
    cd(rootPath);
    addRuntimePaths(rootPath);

    % Flatten problem x run tasks and resume completed runs.
    tasks = cell(0,2);
    for p = 1:numel(proNames)
        for r = runs
            tasks(end+1,:) = {proNames{p},r}; %#ok<SAGROW>
        end
    end
    dataDir = fullfile(rootPath,'Data',algName);
    [~,~] = mkdir(dataDir);
    todo = true(size(tasks,1),1);
    for t = 1:size(tasks,1)
        files = dir(fullfile(dataDir,sprintf('%s_%s_M*_D30_%d.mat', ...
            algName,tasks{t,1},tasks{t,2})));
        todo(t) = isempty(files);
    end
    tasks = tasks(todo,:);
    fprintf('%s remaining tasks: %d\n',algName,size(tasks,1));
    if isempty(tasks)
        fprintf('ALL %s TASKS DONE\n',algName);
        return;
    end

    pool = gcp('nocreate');
    if isempty(pool)
        pool = parpool("Processes",nWorker);
        poolCleanup = onCleanup(@()delete(pool)); %#ok<NASGU>
    elseif pool.NumWorkers ~= nWorker
        error('Experiment:WorkerCount', ...
            'Existing pool must contain exactly %d workers.',nWorker);
    end

    nTask = size(tasks,1);
    taskProblems = tasks(:,1);
    taskRuns = cell2mat(tasks(:,2));
    parfor t = 1:nTask
        addRuntimePaths(rootPath);
        cd(rootPath);
        rng(taskRuns(t),'twister');
        problem = str2func(taskProblems{t});
        algorithm = str2func(algName);
        fprintf('Running %s on %s run %d\n', ...
            algName,taskProblems{t},taskRuns(t));
        Problem = problem('N',popSize,'D',30,'maxFE',maxFE);
        Algorithm = algorithm( ...
            'save',0,'run',taskRuns(t), ...
            'metName',{'IGD','HV','Feasible_rate'});
        Algorithm.Solve(Problem);
        Algorithm.CalMetric('IGD');
        Algorithm.CalMetric('HV');
        Algorithm.CalMetric('Feasible_rate');
        saveResult(dataDir,Algorithm,Problem,taskRuns(t));
    end

    fprintf('ALL %s TASKS DONE\n',algName);
end

function addRuntimePaths(rootPath)
    for folder = {'Algorithms','Problems','Metrics'}
        addpath(genpath(fullfile(rootPath,folder{1})),'-begin');
    end
end

function saveResult(dataDir,Algorithm,Problem,run)
%SAVERESULT Save the final population and metrics under the algorithm name.

    result = Algorithm.result;
    metric = Algorithm.metric;
    file = fullfile(dataDir,sprintf('%s_%s_M%d_D%d_%d.mat', ...
        class(Algorithm),class(Problem),Problem.M,Problem.D,run));
    save(file,'result','metric');
end
