clc; clear;

%% Run the selected algorithms using each problem's default dimensions.
nWorker = 10;
popSize = 100;
maxFE   = 150000;
runs    = 1:10;
% 每行：算法名称、type（1=GA，2=DE）；取消注释即可启用。
algNames = {
    'GANCMO',       1;
    % 'DRMCMO',     1;
    % 'IMTCMO',     1;
    % 'MCCMO',      1;
    % 'MTCMO',      1;
    % 'NAEMT',      1;
    % 'EADMMMOEAD', 1;
    % 'GANCMO_V1',  1;
    % 'GANCMO_V2',  1;
    % 'GANCMO_V3',  1;
    % 'GANCMO_V4',  1;
    % 'GANCMO_V5',  1;
};
% 全部64题：DASCMOP(9)、LIRCMOP(14)、FCP(5)、DOC(9)、SDC(15)、LSCM(12)。
proNames = { ...
    'DASCMOP1_BC','DASCMOP2_BC','DASCMOP3_BC', ...
    'DASCMOP4_BC','DASCMOP5_BC','DASCMOP6_BC', ...
    'DASCMOP7_BC','DASCMOP8_BC','DASCMOP9_BC', ...
    'LIRCMOP1_BC','LIRCMOP2_BC','LIRCMOP3_BC', ...
    'LIRCMOP4_BC','LIRCMOP5_BC','LIRCMOP6_BC', ...
    'LIRCMOP7_BC','LIRCMOP8_BC','LIRCMOP9_BC', ...
    'LIRCMOP10_BC','LIRCMOP11_BC','LIRCMOP12_BC', ...
    'LIRCMOP13_BC','LIRCMOP14_BC', ...
    'FCP1_BC','FCP2_BC','FCP3_BC', ...
    'FCP4_BC','FCP5_BC', ...
    'DOC1_BC','DOC2_BC','DOC3_BC', ...
    'DOC4_BC','DOC5_BC','DOC6_BC', ...
    'DOC7_BC','DOC8_BC','DOC9_BC', ...
    'SDC1_BC','SDC2_BC','SDC3_BC', ...
    'SDC4_BC','SDC5_BC','SDC6_BC', ...
    'SDC7_BC','SDC8_BC','SDC9_BC', ...
    'SDC10_BC','SDC11_BC','SDC12_BC', ...
    'SDC13_BC','SDC14_BC','SDC15_BC', ...
    'LSCM1_BC','LSCM2_BC','LSCM3_BC', ...
    'LSCM4_BC','LSCM5_BC','LSCM6_BC', ...
    'LSCM7_BC','LSCM8_BC','LSCM9_BC', ...
    'LSCM10_BC','LSCM11_BC','LSCM12_BC'
};

rootPath = fileparts(mfilename('fullpath'));
assert(size(algNames,2)==2,'Experiment:AlgorithmConfig','Each row must contain an algorithm name and type.');
for a = 1:size(algNames,1)
    runExperiment(rootPath,nWorker,popSize,maxFE,runs,algNames{a,1},algNames{a,2},proNames);
end

function runExperiment(rootPath,nWorker,popSize,maxFE,runs,algName,type,proNames)
    assert(isnumeric(type) && isscalar(type) && isreal(type) && ismember(type,[1 2]), ...
        'Experiment:BadOperatorType','type must be 1 (GA) or 2 (DE).');
    cd(rootPath);
    addRuntimePaths(rootPath);

    % Flatten problem x run tasks and resume completed runs.
    tasks = cell(numel(proNames)*numel(runs),4);
    index = 0;
    for p = 1:numel(proNames)
        dimensions = feval(proNames{p},'N',popSize,'maxFE',maxFE);
        for r = runs
            index = index+1;
            tasks(index,:) = {proNames{p},r,dimensions.M,dimensions.D};
        end
    end

    dataDir = fullfile(rootPath,'Data',algName);
    [~,~] = mkdir(dataDir);
    todo = true(size(tasks,1),1);
    for t = 1:size(tasks,1)
        file = fullfile(dataDir,sprintf('%s_%s_M%d_D%d_%d.mat', ...
            algName,tasks{t,1},tasks{t,3},tasks{t,4},tasks{t,2}));
        todo(t) = ~isfile(file);
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
    algorithmParameters = {type};
    parfor t = 1:nTask
        addRuntimePaths(rootPath);
        cd(rootPath);
        rng(taskRuns(t),'twister');
        problem = str2func(taskProblems{t});
        algorithm = str2func(algName);
        fprintf('Running %s type=%d on %s run %d\n', ...
            algName,type,taskProblems{t},taskRuns(t));
        Problem = problem('N',popSize,'maxFE',maxFE);
        Algorithm = algorithm( ...
            'parameter',algorithmParameters,'save',0,'run',taskRuns(t), ...
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
