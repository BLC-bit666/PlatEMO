clc; clear;

%% ===== 基本设置 =====
nRun    = 10;
nWorker = 8;

popSize = 100;
maxFE   = 2e5;
saveNum = 2;

%% ===== 算法列表 =====
% 旧算法列表（已停用，保留为注释）
% algs = {@IMTCMO,@MCCMO,@MTCMO,@CPCMO,{@DRMCMO,2}};

algs = {@CBS_RegionWGAN_GP,@CBS_RegionWGAN_GP_NoCGAN};

%% ===== 问题列表 =====
problemSets = {
    'DASCMOP',  9
    'LIRCMOP', 14
    'MW',      14
    'CF',      10
    'DOC',      9
    'FCP',      5
    'SDC',     15
};

proNames = {};
for s = 1:size(problemSets,1)
    for i = 1:problemSets{s,2}
        proNames{end+1} = sprintf('%s%d_BC',problemSets{s,1},i); %#ok<SAGROW>
    end
end

%% ===== 清理旧并行池和旧 Jobs =====
delete(gcp('nocreate'));

try
    c = parcluster('Processes');
    delete(c.Jobs);
catch
end

%% ===== 打开并行池 =====
parpool("Processes",nWorker);

%% ===== 确保所有 worker 都有当前路径 =====
rootPath = fileparts(mfilename('fullpath'));
cd(rootPath);
addpath(genpath(rootPath));
pctRunOnAll addpath(genpath(rootPath));

%% ===== 批量实验 =====
for a = 1:numel(algs)
    algHandle = algs{a};
    algName   = func2str(algHandle);

    for p = 1:numel(proNames)
        proName   = proNames{p};
        proHandle = str2func(proName);

        fprintf('\nRunning %s on %s\n',algName,proName);

        parfor run = 1:nRun
            platemo( ...
                'algorithm',algHandle, ...
                'problem',  proHandle, ...
                'N',        popSize, ...
                'maxFE',    maxFE, ...
                'save',     saveNum, ...
                'run',      run ...
            );
        end
    end
end

%% ===== 关闭并行池 =====
delete(gcp('nocreate'));

disp('All CBS-CGAN experiments finished.');
