clc; clear;

%% ===== 正式确认实验 =====

nWorker = 8;
popSize = 100;
maxFE   = 2e5;
saveNum = 1;
runs    = 1:30;
%% algName = 'DRMCMO';

algName = 'NAEMT';

problemSets = {
    'DASCMOP',  9
    'LIRCMOP', 14
};

proNames = {};
for s = 1:size(problemSets,1)
    for i = 1:problemSets{s,2}
        proNames{end+1} = sprintf('%s%d_BC',problemSets{s,1},i); %#ok<SAGROW>
    end
end

rootPath = fileparts(mfilename('fullpath'));
cd(rootPath);
addpath(genpath(rootPath));

%% ===== 任务清单（跳过已完成） =====
tasks = cell(0,2);
for p = 1:numel(proNames)
    for r = runs
        tasks(end+1,:) = {proNames{p},r}; %#ok<SAGROW>
    end
end
dataDir = fullfile(rootPath,'Data',algName);
todo = true(size(tasks,1),1);
for t = 1:size(tasks,1)
    f = dir(fullfile(dataDir,sprintf('%s_%s_M*_D*_%d.mat', ...
        algName,tasks{t,1},tasks{t,2})));
    todo(t) = isempty(f);
end
tasks = tasks(todo,:);
fprintf('Confirmation remaining tasks: %d\n',size(tasks,1));
if isempty(tasks)
    disp('ALL CONFIRMATION TASKS DONE');
    return;
end

%% ===== 清理旧并行池和旧 Jobs =====
delete(gcp('nocreate'));
try
    c = parcluster('Processes');
    delete(c.Jobs);
catch
end

%% ===== 并行池 =====
parpool("Processes",nWorker);
pctRunOnAll addpath(genpath(rootPath));

%% ===== 扁平化并行（问题 x run 一起排队，吃满 worker） =====
nTask = size(tasks,1);
parfor t = 1:nTask
    proHandle = str2func(tasks{t,1});
    fprintf('Running %s on %s run %d\n',algName,tasks{t,1},tasks{t,2});
    platemo( ...
        'algorithm',@NAEMT, ...
        'problem',  proHandle, ...
        'N',        popSize, ...
        'maxFE',    maxFE, ...
        'save',     saveNum, ...
        'run',      tasks{t,2} ...
    );
end

delete(gcp('nocreate'));
disp('ALL CONFIRMATION TASKS DONE');
