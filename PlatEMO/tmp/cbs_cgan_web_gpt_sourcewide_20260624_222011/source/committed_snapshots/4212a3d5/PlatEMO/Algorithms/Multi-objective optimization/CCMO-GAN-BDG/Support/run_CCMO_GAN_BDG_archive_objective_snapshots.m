function [Summary,outDir,ImageManifest] = run_CCMO_GAN_BDG_archive_objective_snapshots( ...
        outDir,workerCount,problemNames,N,D,maxFE,runId,targets, ...
        algorithmParams,variantSet)
% Compatibility wrapper: run the unified experiment entry with plotting on.

    rootDir = fileparts(which('platemo'));
    if nargin < 1 || isempty(outDir)
        outDir = fullfile(rootDir,'Data','CCMO_GAN_BDG', ...
            ['archive_objective_snapshots_', ...
            char(datetime('now','Format','yyyyMMdd_HHmmss'))]);
    end
    if nargin < 2 || isempty(workerCount)
        workerCount = 6;
    end
    if nargin < 3
        problemNames = [];
    end
    if nargin < 4 || isempty(N)
        N = 100;
    end
    if nargin < 5
        D = [];
    end
    if nargin < 6 || isempty(maxFE)
        maxFE = 100000;
    end
    if nargin < 7 || isempty(runId)
        runId = 1;
    end
    if nargin < 8 || isempty(targets)
        targets = [10000 30000 50000 70000 100000];
    end
    if nargin < 9
        algorithmParams = [];
    end
    if nargin < 10 || isempty(variantSet)
        variantSet = "default";
    end

    plotOptions = struct( ...
        'enable',true, ...
        'targets',targets, ...
        'snapshotOutDir',outDir);
    fullscopeOutDir = fullfile(outDir,'fullscope');
    [Summary,~,ImageManifest] = ...
        run_CCMO_GAN_BDG_archive_pareto_filters_fullscope( ...
        fullscopeOutDir,workerCount,problemNames,N,D,maxFE,runId, ...
        variantSet,algorithmParams,plotOptions);
end
