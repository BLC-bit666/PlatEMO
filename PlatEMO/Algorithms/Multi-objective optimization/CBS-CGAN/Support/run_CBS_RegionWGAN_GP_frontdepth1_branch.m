function BranchSummary = run_CBS_RegionWGAN_GP_frontdepth1_branch(rootOutDir,workerCount)
%RUN_CBS_REGIONWGAN_GP_FRONTDEPTH1_BRANCH Run C mainline with frontDepth=1.
%   Single-branch diagnostic experiment. All settings follow the current
%   CBS_RegionWGAN_GP mainline C defaults except frontDepth is set to 1.

    repoRoot = locateFrontDepthRepoRoot();
    addpath(genpath(repoRoot));
    if nargin < 1 || isempty(rootOutDir)
        rootOutDir = fullfile(repoRoot,'Data','CBS_RegionGAN_compare', ...
            ['c_frontdepth1_true_boundary_', ...
            char(datetime('now','Format','yyyyMMdd_HHmmss'))]);
    end
    if nargin < 2 || isempty(workerCount)
        workerCount = 10;
    end

    problems = ["LIRCMOP5_BC";"LIRCMOP6_BC";"LIRCMOP7_BC"; ...
        "LIRCMOP8_BC";"LIRCMOP9_BC";"LIRCMOP10_BC"];
    stageTargets = [10000 30000 50000 70000 100000];
    runIds = 1 : 3;
    branchName = "C_frontDepth1";
    outDir = fullfile(rootOutDir,char(branchName));

    if ~isfolder(rootOutDir)
        mkdir(rootOutDir);
    end

    Row = emptyFrontDepthRow();
    Row.branch = branchName;
    Row.out_dir = string(outDir);
    Row.front_depth = 1;
    Row.query_mode = "random_all_w";
    Row.prev_bmem_mode = "prev1_fair_union";
    Row.condition_mode = "region";
    Row.prescreen_multiplier = 1;
    Defaults = CBS_RegionWGAN_GP.mainlineDefaults();
    Row.gan_iter = double(Defaults.ganIter);
    Row.n_critic = double(Defaults.nCritic);
    Row.sample_sigma = double(Defaults.sampleSigma);

    try
        Options = struct( ...
            'algorithmClass',"CBS_RegionWGAN_GP", ...
            'queryMode',Defaults.queryMode, ...
            'conditionMode',"region", ...
            'prevBMemMode',Defaults.prevBMemMode, ...
            'bmemBandMode',Defaults.bmemBandMode, ...
            'bandMaxAnchorsPerRef',[], ...
            'captureRun',3, ...
            'plotRun',0, ...
            'drawFigures',false, ...
            'stageTargets',stageTargets, ...
            'algorithmParams',{wganParamsFrontDepth1(Defaults)}, ...
            'sampleZMode',"random", ...
            'trainZMode',"random", ...
            'sigma',1.0, ...
            'trainSigma',1.0, ...
            'sampleSigma',Defaults.sampleSigma, ...
            'prescreenMultiplier',1);
        run_CBS_RegionCGAN_training_diagnostics(outDir,workerCount, ...
            problems,100,[],100000,runIds,Options);
        Row.status = "ok";
        Row = fillFrontDepthCounts(Row,outDir);
    catch err
        Row.status = "failed";
        Row.error_message = string(getReport(err,'extended', ...
            'hyperlinks','off'));
        fprintf('%s failed:\n%s\n',char(branchName),char(Row.error_message));
    end

    BranchSummary = struct2table(Row);
    writetable(BranchSummary,fullfile(rootOutDir,'branch_summary.csv'));
    fprintf('FrontDepth1 branch written to: %s\n',rootOutDir);
end

function Params = wganParamsFrontDepth1(Defaults)
    Params = { ...
        Defaults.trainGap, ...
        Defaults.archiveGap, ...
        Defaults.nGen, ...
        Defaults.zDim, ...
        Defaults.ganIter, ...
        Defaults.ganMiniBatch, ...
        Defaults.ganLrD, ...
        Defaults.ganLrG, ...
        1, ...
        Defaults.pairNeighborRefRadius, ...
        Defaults.refDivisor, ...
        Defaults.minBoundaryLength, ...
        Defaults.queryPerCondition, ...
        Defaults.gpLambda, ...
        Defaults.nCritic, ...
        Defaults.maxAnchorsPerRef, ...
        Defaults.minGANTrainCount};
end

function repoRoot = locateFrontDepthRepoRoot()
    repoRoot = fileparts(mfilename('fullpath'));
    while ~isempty(repoRoot) && ~isfile(fullfile(repoRoot,'platemo.m'))
        parent = fileparts(repoRoot);
        if strcmp(parent,repoRoot)
            break;
        end
        repoRoot = parent;
    end
    if ~isfile(fullfile(repoRoot,'platemo.m'))
        error('CBSRegionWGAN:FrontDepthRootNotFound', ...
            'Unable to locate PlatEMO root.');
    end
end

function Row = fillFrontDepthCounts(Row,outDir)
    Row.run_count = countFrontDepthRows(fullfile(outDir,'run_summary.csv'));
    Row.ok_run_count = countFrontDepthOkRuns(fullfile(outDir,'run_summary.csv'));
    Row.stage_snapshot_count = countFrontDepthRows(fullfile(outDir, ...
        'stage_snapshots_all.csv'));
    Row.event_count = countFrontDepthRows(fullfile(outDir, ...
        'event_summary_all.csv'));
    Row.has_true_boundary_metrics = hasFrontDepthTrueBoundaryMetrics(outDir);
end

function yes = hasFrontDepthTrueBoundaryMetrics(outDir)
    yes = false;
    file = fullfile(outDir,'event_summary_all.csv');
    if ~isfile(file)
        return;
    end
    fid = fopen(file,'r');
    if fid < 0
        return;
    end
    cleanup = onCleanup(@()fclose(fid));
    header = fgetl(fid);
    if ~ischar(header)
        return;
    end
    vars = split(string(header),",");
    yes = all(ismember(["bdist50_true","bwidth90_10_true", ...
        "bcover_eps_true"],vars));
end

function n = countFrontDepthRows(file)
    n = 0;
    if ~isfile(file)
        return;
    end
    fid = fopen(file,'r');
    if fid < 0
        return;
    end
    cleanup = onCleanup(@()fclose(fid));
    while true
        line = fgetl(fid);
        if ~ischar(line)
            break;
        end
        if strlength(string(strtrim(line))) > 0
            n = n + 1;
        end
    end
    n = max(0,n - 1);
end

function n = countFrontDepthOkRuns(file)
    n = 0;
    if ~isfile(file)
        return;
    end
    fid = fopen(file,'r');
    if fid < 0
        return;
    end
    cleanup = onCleanup(@()fclose(fid));
    header = fgetl(fid);
    if ~ischar(header)
        return;
    end
    names = split(string(header),",");
    statusIdx = find(strcmpi(names,"status"),1);
    if isempty(statusIdx)
        return;
    end
    while true
        line = fgetl(fid);
        if ~ischar(line)
            break;
        end
        parts = split(string(line),",");
        if numel(parts) >= statusIdx && parts(statusIdx) == "ok"
            n = n + 1;
        end
    end
end

function Row = emptyFrontDepthRow()
    Row = struct( ...
        'branch',"", ...
        'status',"pending", ...
        'out_dir',"", ...
        'front_depth',NaN, ...
        'query_mode',"", ...
        'prev_bmem_mode',"", ...
        'condition_mode',"", ...
        'gan_iter',NaN, ...
        'n_critic',NaN, ...
        'sample_sigma',NaN, ...
        'prescreen_multiplier',NaN, ...
        'run_count',0, ...
        'ok_run_count',0, ...
        'stage_snapshot_count',0, ...
        'event_count',0, ...
        'has_true_boundary_metrics',false, ...
        'error_message',"");
end
