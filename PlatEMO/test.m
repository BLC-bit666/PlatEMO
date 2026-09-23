clc; clear;

%% GANCMO coverage-0.04 mainline; run this campaign only when requested.
nWorker = 10;
popSize = 100;
maxFE   = 2e5;
runs    = 151:200;
algName = 'GANCMO';
proNames = {
    'LIRCMOP1_BC','LIRCMOP2_BC','LIRCMOP3_BC','LIRCMOP4_BC', ...
    'LIRCMOP5_BC','LIRCMOP6_BC','LIRCMOP7_BC','LIRCMOP8_BC', ...
    'LIRCMOP9_BC','LIRCMOP10_BC','LIRCMOP11_BC','LIRCMOP12_BC', ...
    'LIRCMOP13_BC','LIRCMOP14_BC'};

rootPath = fileparts(mfilename('fullpath'));
cd(rootPath);
addRuntimePaths(rootPath);

%% ===== Flatten problem x run tasks and resume completed runs =====
tasks = cell(0,2);
for p = 1:numel(proNames)
    for r = runs
        tasks(end+1,:) = {proNames{p},r}; %#ok<SAGROW>
    end
end
% Keep prior fixed-quota results separate from the selected adaptive mainline.
dataDir = fullfile(rootPath,'Data','PairGuideCoverage');
[~,~] = mkdir(dataDir);
todo = true(size(tasks,1),1);
for t = 1:size(tasks,1)
    files = dir(fullfile(dataDir,sprintf('%s_%s_M*_D30_%d.mat', ...
        algName,tasks{t,1},tasks{t,2})));
    todo(t) = isempty(files);
end
tasks = tasks(todo,:);
fprintf('GANCMO remaining tasks: %d\n',size(tasks,1));
if isempty(tasks)
    disp('ALL GANCMO TASKS DONE');
    return;
end

pool = gcp('nocreate');
ownsPool = isempty(pool);
if ownsPool
    pool = parpool("Processes",nWorker);
elseif pool.NumWorkers ~= nWorker
    error('GANCMO:WorkerCount', ...
        'Existing pool must contain exactly %d workers.',nWorker);
end
poolCleanup = onCleanup(@()closeOwnedPool(ownsPool));

nTask = size(tasks,1);
taskProblems = tasks(:,1);
taskRuns = cell2mat(tasks(:,2));
parfor t = 1:nTask
    addRuntimePaths(rootPath);
    cd(rootPath);
    rng(taskRuns(t),'twister');
    problem = str2func(taskProblems{t});
    fprintf('Running %s on %s run %d\n', ...
        algName,taskProblems{t},taskRuns(t));
    Problem = problem('N',popSize,'D',30,'maxFE',maxFE);
    Algorithm = GANCMO( ...
        'save',0,'run',taskRuns(t), ...
        'metName',{'IGD','HV','Feasible_rate'});
    Algorithm.Solve(Problem);
    Algorithm.CalMetric('IGD');
    Algorithm.CalMetric('HV');
    Algorithm.CalMetric('Feasible_rate');
    saveGANCMOResult(dataDir,Algorithm,Problem,taskRuns(t));
end

disp('ALL GANCMO TASKS DONE');

function addRuntimePaths(rootPath)
    for folder = {'Algorithms','Problems','Metrics'}
        addpath(genpath(fullfile(rootPath,folder{1})),'-begin');
    end
end

function closeOwnedPool(ownsPool)
    if ownsPool
        pool = gcp('nocreate');
        if ~isempty(pool)
            delete(pool);
        end
    end
end

function saveGANCMOResult(dataDir,Algorithm,Problem,run)
%SAVEGANCMORESULT Save GANCMO results under its own name.

    result = Algorithm.result;
    metric = Algorithm.metric;
    file = fullfile(dataDir,sprintf('%s_%s_M%d_D%d_%d.mat', ...
        class(Algorithm),class(Problem),Problem.M,Problem.D,run));
    save(file,'result','metric');
end
