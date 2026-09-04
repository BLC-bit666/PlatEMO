clc; clear;

%% ===== PairGuide mainline campaign: five LIRCMOP problems x five runs =====
nWorker = 10;
popSize = 100;
maxFE   = 2e5;
saveNum = 2;
runs    = 1:5;
algName = 'PairGuide';
proNames = {
    'LIRCMOP5_BC','LIRCMOP7_BC','LIRCMOP8_BC', ...
    'LIRCMOP10_BC','LIRCMOP14_BC'};

rootPath = fileparts(mfilename('fullpath'));
cd(rootPath);
supportPath = fullfile(rootPath,'Algorithms','Multi-objective optimization', ...
    'CBS-CGAN','Support');
addpath(supportPath,'-begin');
addCBSPaths(rootPath);

%% ===== Flatten problem x run tasks and resume completed runs =====
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
fprintf('PairGuide remaining tasks: %d\n',size(tasks,1));
if isempty(tasks)
    disp('ALL PAIR-GUIDE TASKS DONE');
    return;
end

pool = gcp('nocreate');
ownsPool = isempty(pool);
if ownsPool
    pool = parpool("Processes",nWorker);
elseif pool.NumWorkers ~= nWorker
    error('CBSRegionGAN:WorkerCount', ...
        'Existing pool must contain exactly %d workers.',nWorker);
end
poolCleanup = onCleanup(@()closeOwnedPool(ownsPool));

nTask = size(tasks,1);
taskProblems = tasks(:,1);
taskRuns = cell2mat(tasks(:,2));
parfor t = 1:nTask
    addCBSPaths(rootPath);
    cd(rootPath);
    rng(taskRuns(t),'twister');
    problem = str2func(taskProblems{t});
    fprintf('Running %s on %s run %d\n', ...
        algName,taskProblems{t},taskRuns(t));
    Problem = problem('N',popSize,'D',30,'maxFE',maxFE);
    Algorithm = PairGuide( ...
        'save',-saveNum,'run',taskRuns(t), ...
        'metName',{'IGD','HV','Feasible_rate'});
    Algorithm.Solve(Problem);
    Algorithm.CalMetric('IGD');
    Algorithm.CalMetric('HV');
    Algorithm.CalMetric('Feasible_rate');
    savePairGuideResult(dataDir,Algorithm,Problem,taskRuns(t));
end

disp('ALL PAIR-GUIDE TASKS DONE');

function closeOwnedPool(ownsPool)
    if ownsPool
        pool = gcp('nocreate');
        if ~isempty(pool)
            delete(pool);
        end
    end
end

function savePairGuideResult(dataDir,Algorithm,Problem,run)
%SAVEPAIRGUIDERESULT Save PairGuide results under its own name.

    result = Algorithm.result;
    metric = Algorithm.metric;
    file = fullfile(dataDir,sprintf('%s_%s_M%d_D%d_%d.mat', ...
        class(Algorithm),class(Problem),Problem.M,Problem.D,run));
    save(file,'result','metric');
end
