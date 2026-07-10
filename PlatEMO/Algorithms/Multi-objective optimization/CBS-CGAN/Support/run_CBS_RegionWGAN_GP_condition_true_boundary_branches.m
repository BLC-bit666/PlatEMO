function BranchSummary = run_CBS_RegionWGAN_GP_condition_true_boundary_branches( ...
        rootOutDir,workerCount,problemNames,N,D,maxFE,runIds,Overrides)
%RUN_CBS_REGIONWGAN_GP_CONDITION_TRUE_BOUNDARY_BRANCHES Compare region conditions.
%   Runs the promoted C WGAN-GP training dose while changing only the region
%   condition mode: W-only control, W+s_local, and W+rho.

    repoRoot = locateRepoRoot();
    addpath(genpath(repoRoot));
    if nargin < 1 || isempty(rootOutDir)
        rootOutDir = fullfile(repoRoot,'Data','CBS_RegionGAN_compare', ...
            ['condition_true_boundary_', ...
            char(datetime('now','Format','yyyyMMdd_HHmmss'))]);
    end
    if nargin < 2 || isempty(workerCount)
        workerCount = 8;
    end
    if nargin < 3 || isempty(problemNames)
        problemNames = ["LIRCMOP5_BC";"LIRCMOP6_BC";"LIRCMOP7_BC"; ...
            "LIRCMOP8_BC";"LIRCMOP9_BC";"LIRCMOP10_BC"];
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
    if nargin < 7 || isempty(runIds)
        runIds = 1 : 3;
    end
    if nargin < 8 || isempty(Overrides)
        Overrides = struct();
    end

    problemNames = string(problemNames(:));
    stageTargets = overrideValue(Overrides,'stageTargets', ...
        [10000 30000 50000 70000 100000]);
    stageTargets = double(stageTargets(:)');
    captureRun = overrideValue(Overrides,'captureRun',3);
    plotRun = overrideValue(Overrides,'plotRun',0);
    drawFigures = logical(overrideValue(Overrides,'drawFigures',false));
    if ~isfolder(rootOutDir)
        mkdir(rootOutDir);
    end

    Branches = branchSpecs(Overrides);
    Rows = repmat(emptyBranchRow(),numel(Branches),1);
    for i = 1 : numel(Branches)
        Branch = Branches(i);
        outDir = fullfile(rootOutDir,char(Branch.name));
        Rows(i) = rowFromBranch(Branch,outDir);
        if branchComplete(outDir,numel(problemNames),numel(runIds), ...
                numel(stageTargets),Branch.conditionMode)
            Rows(i).status = "skipped_complete";
            Rows(i) = fillCounts(Rows(i),outDir);
            fprintf('[%d/%d] %s already complete, skipped.\n', ...
                i,numel(Branches),char(Branch.name));
            BranchSummary = writeSummary(Rows,rootOutDir);
            continue;
        end

        fprintf('[%d/%d] running %s -> %s\n',i,numel(Branches), ...
            char(Branch.name),outDir);
        try
            Options = struct( ...
                'algorithmClass',"CBS_RegionWGAN_GP", ...
                'queryMode',"random_all_w", ...
                'conditionMode',Branch.conditionMode, ...
                'prevBMemMode',"prev1_fair_union", ...
                'bmemBandMode',"current", ...
                'bandMaxAnchorsPerRef',[], ...
                'captureRun',captureRun, ...
                'plotRun',plotRun, ...
                'drawFigures',drawFigures, ...
                'stageTargets',stageTargets, ...
                'algorithmParams',{wganParams(Overrides)}, ...
                'sampleZMode',"random", ...
                'trainZMode',"random", ...
                'sigma',1.0, ...
                'trainSigma',1.0, ...
                'sampleSigma',0.3, ...
                'ganIterSchedule',"fixed", ...
                'ganIterStart',[], ...
                'ganIterEnd',[], ...
                'prescreenMultiplier',1, ...
                'trainTriggerMode',"off", ...
                'trainTriggerDelta',0.2);
            run_CBS_RegionCGAN_training_diagnostics(outDir,workerCount, ...
                problemNames,N,D,maxFE,runIds,Options);
            Rows(i).status = "ok";
            Rows(i) = fillCounts(Rows(i),outDir);
        catch err
            Rows(i).status = "failed";
            Rows(i).error_message = string(getReport(err,'extended', ...
                'hyperlinks','off'));
            fprintf('[%d/%d] %s failed:\n%s\n',i,numel(Branches), ...
                char(Branch.name),char(Rows(i).error_message));
        end
        BranchSummary = writeSummary(Rows,rootOutDir);
    end
    BranchSummary = writeSummary(Rows,rootOutDir);
    fprintf('condition true-boundary branches written to: %s\n',rootOutDir);
end

function repoRoot = locateRepoRoot()
    repoRoot = fileparts(which('platemo'));
    if ~isempty(repoRoot)
        return;
    end
    repoRoot = fileparts(mfilename('fullpath'));
    while ~isempty(repoRoot) && ~isfile(fullfile(repoRoot,'platemo.m'))
        parent = fileparts(repoRoot);
        if strcmp(parent,repoRoot)
            break;
        end
        repoRoot = parent;
    end
    if ~isfile(fullfile(repoRoot,'platemo.m'))
        error('CBSRegionWGAN:RepoRootNotFound', ...
            'Unable to locate PlatEMO root.');
    end
end

function Branches = branchSpecs(Overrides)
    Branches = [ ...
        makeBranch("C_control_region","region"); ...
        makeBranch("C_region_slocal","region_slocal"); ...
        makeBranch("C_region_rho","region_rho")];
    if isstruct(Overrides) && isfield(Overrides,'branchNames') && ...
            ~isempty(Overrides.branchNames)
        wanted = string(Overrides.branchNames(:));
        keep = false(numel(Branches),1);
        for i = 1 : numel(Branches)
            keep(i) = any(Branches(i).name == wanted);
        end
        Branches = Branches(keep);
    end
end

function Branch = makeBranch(name,conditionMode)
    Branch = struct( ...
        'name',string(name), ...
        'conditionMode',string(conditionMode));
end

function Params = wganParams(Overrides)
    if isstruct(Overrides) && isfield(Overrides,'algorithmParams') && ...
            ~isempty(Overrides.algorithmParams)
        Params = Overrides.algorithmParams;
        return;
    end
    Defaults = CBS_RegionWGAN_GP.mainlineDefaults();
    Params = { ...
        Defaults.trainGap, ...
        Defaults.archiveGap, ...
        Defaults.nGen, ...
        Defaults.zDim, ...
        Defaults.ganIter, ...
        Defaults.ganMiniBatch, ...
        Defaults.ganLrD, ...
        Defaults.ganLrG, ...
        Defaults.frontDepth, ...
        Defaults.pairNeighborRefRadius, ...
        Defaults.refDivisor, ...
        Defaults.minBoundaryLength, ...
        Defaults.queryPerCondition, ...
        Defaults.gpLambda, ...
        Defaults.nCritic, ...
        Defaults.maxAnchorsPerRef, ...
        Defaults.minGANTrainCount};
end

function value = overrideValue(Overrides,name,defaultValue)
    if isstruct(Overrides) && isfield(Overrides,name) && ...
            ~isempty(Overrides.(name))
        value = Overrides.(name);
    else
        value = defaultValue;
    end
end

function Row = rowFromBranch(Branch,outDir)
    Defaults = CBS_RegionWGAN_GP.mainlineDefaults();
    Row = emptyBranchRow();
    Row.branch = Branch.name;
    Row.status = "pending";
    Row.out_dir = string(outDir);
    Row.query_mode = Defaults.queryMode;
    Row.condition_mode = Branch.conditionMode;
    Row.gan_iter = double(Defaults.ganIter);
    Row.n_critic = double(Defaults.nCritic);
    Row.critic_updates = double(Defaults.ganIter*Defaults.nCritic);
    Row.generator_updates = double(Defaults.ganIter);
end

function BranchSummary = writeSummary(Rows,rootOutDir)
    BranchSummary = struct2table(Rows);
    writetable(BranchSummary,fullfile(rootOutDir,'branch_summary.csv'));
end

function yes = branchComplete(outDir,problemCount,runCount,stageCount, ...
        conditionMode)
    summaryFile = fullfile(outDir,'run_summary.csv');
    stageFile = fullfile(outDir,'stage_snapshots_all.csv');
    eventFile = fullfile(outDir,'event_summary_all.csv');
    yes = isfile(summaryFile) && isfile(stageFile) && isfile(eventFile) && ...
        countTableRows(summaryFile) == problemCount*runCount && ...
        countOkRunRows(summaryFile) == problemCount*runCount && ...
        countTableRows(stageFile) == problemCount*stageCount && ...
        hasTrueBoundaryColumns(stageFile) && ...
        hasTrueBoundaryColumns(eventFile) && ...
        hasConditionMode(stageFile,conditionMode) && ...
        hasConditionMode(eventFile,conditionMode);
end

function Row = fillCounts(Row,outDir)
    Row.run_count = countTableRows(fullfile(outDir,'run_summary.csv'));
    Row.ok_run_count = countOkRunRows(fullfile(outDir,'run_summary.csv'));
    Row.stage_snapshot_count = countTableRows(fullfile(outDir, ...
        'stage_snapshots_all.csv'));
    Row.event_count = countTableRows(fullfile(outDir, ...
        'event_summary_all.csv'));
    Row.has_true_boundary_metrics = hasTrueBoundaryColumns( ...
        fullfile(outDir,'stage_snapshots_all.csv')) && ...
        hasTrueBoundaryColumns(fullfile(outDir,'event_summary_all.csv'));
end

function n = countTableRows(file)
    n = 0;
    if ~isfile(file)
        return;
    end
    fid = fopen(file,'r');
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

function n = countOkRunRows(file)
    n = 0;
    if ~isfile(file)
        return;
    end
    try
        T = readtable(file,'TextType','string');
        if ismember("status",string(T.Properties.VariableNames))
            n = sum(string(T.status) == "ok");
        end
    catch
        n = 0;
    end
end

function yes = hasTrueBoundaryColumns(file)
    yes = false;
    if ~isfile(file)
        return;
    end
    try
        opts = detectImportOptions(file);
        vars = string(opts.VariableNames);
        yes = all(ismember(["bdist50_true","bwidth90_10_true", ...
            "bcover_eps_true"],vars));
    catch
        yes = false;
    end
end

function yes = hasConditionMode(file,conditionMode)
    yes = false;
    if ~isfile(file)
        return;
    end
    try
        T = readtable(file,'TextType','string');
        if ismember("condition_mode",string(T.Properties.VariableNames))
            values = string(T.condition_mode);
            values = values(strlength(values) > 0);
            yes = ~isempty(values) && all(values == string(conditionMode));
        end
    catch
        yes = false;
    end
end

function Row = emptyBranchRow()
    Row = struct( ...
        'branch',"", ...
        'status',"pending", ...
        'out_dir',"", ...
        'query_mode',"", ...
        'condition_mode',"", ...
        'gan_iter',NaN, ...
        'n_critic',NaN, ...
        'critic_updates',NaN, ...
        'generator_updates',NaN, ...
        'run_count',0, ...
        'ok_run_count',0, ...
        'stage_snapshot_count',0, ...
        'event_count',0, ...
        'has_true_boundary_metrics',false, ...
        'error_message',"");
end
