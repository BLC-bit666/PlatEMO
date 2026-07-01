function BranchSummary = run_CBS_RegionWGAN_GP_ablation_branches(rootOutDir,workerCount)
%RUN_CBS_REGIONWGAN_GP_ABLATION_BRANCHES Run mainline WGAN-GP ablations.
%   Uses the current baseline experiment specification:
%   six LIRCMOP*_BC problems, runs=1:3, N=100, default D, maxFE=100000,
%   plot only run=3 at FE=[10000 30000 50000 70000 100000].

    repoRoot = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
    addpath(genpath(repoRoot));
    if nargin < 1 || isempty(rootOutDir)
        rootOutDir = fullfile(repoRoot,'Data','CBS_RegionGAN_compare', ...
            ['ablation_z_condition_wgan_', ...
            char(datetime('now','Format','yyyyMMdd_HHmmss'))]);
    end
    if nargin < 2 || isempty(workerCount)
        workerCount = 10;
    end

    problems = ["LIRCMOP5_BC";"LIRCMOP6_BC";"LIRCMOP7_BC"; ...
        "LIRCMOP8_BC";"LIRCMOP9_BC";"LIRCMOP10_BC"];
    stageTargets = [10000 30000 50000 70000 100000];
    runIds = 1 : 3;
    N = 100;
    D = [];
    maxFE = 100000;

    if ~isfolder(rootOutDir)
        mkdir(rootOutDir);
    end

    Branches = makeBranches();
    Rows = repmat(emptyBranchRow(),numel(Branches),1);
    for b = 1 : numel(Branches)
        Branch = Branches(b);
        outDir = fullfile(rootOutDir,char(Branch.name));
        Rows(b).branch = Branch.name;
        Rows(b).out_dir = string(outDir);
        Rows(b).query_mode = Branch.queryMode;
        Rows(b).sample_z_mode = Branch.sampleZMode;
        Rows(b).train_z_mode = Branch.trainZMode;
        Rows(b).sigma = doubleOrNaN(Branch.sigma);
        Rows(b).gan_iter = double(Branch.ganIter);
        Rows(b).n_critic = double(Branch.nCritic);

        if branchComplete(outDir,numel(problems),numel(runIds), ...
                numel(stageTargets))
            Rows(b).status = "skipped_complete";
            Rows(b) = fillBranchCounts(Rows(b),outDir);
            fprintf('[branch %d/%d] %s already complete, skipped.\n', ...
                b,numel(Branches),char(Branch.name));
            continue;
        end

        fprintf('[branch %d/%d] running %s -> %s\n',b,numel(Branches), ...
            char(Branch.name),outDir);
        try
            Options = struct( ...
                'algorithmClass',"CBS_RegionWGAN_GP", ...
                'queryMode',Branch.queryMode, ...
                'prevBMemMode',"prev1_fair_union", ...
                'plotRun',3, ...
                'stageTargets',stageTargets, ...
                'algorithmParams',{wganParams(Branch)}, ...
                'sampleZMode',Branch.sampleZMode, ...
                'trainZMode',Branch.trainZMode, ...
                'sigma',Branch.sigma);
            run_CBS_RegionCGAN_training_diagnostics(outDir,workerCount, ...
                problems,N,D,maxFE,runIds,Options);
            redraw_CBS_RegionCGAN_domain_figures(outDir, ...
                fullfile(outDir,'domain_figures_all'));
            Rows(b).status = "ok";
            Rows(b) = fillBranchCounts(Rows(b),outDir);
        catch err
            Rows(b).status = "failed";
            Rows(b).error_message = string(getReport(err,'extended', ...
                'hyperlinks','off'));
            fprintf('[branch %d/%d] %s failed:\n%s\n',b,numel(Branches), ...
                char(Branch.name),char(Rows(b).error_message));
        end
        BranchSummary = struct2table(Rows);
        writetable(BranchSummary,fullfile(rootOutDir,'branch_summary.csv'));
    end

    BranchSummary = struct2table(Rows);
    writetable(BranchSummary,fullfile(rootOutDir,'branch_summary.csv'));
    fprintf('WGAN-GP ablation branches written to: %s\n',rootOutDir);
end

function Branches = makeBranches()
    Defaults = CBS_RegionWGAN_GP.mainlineDefaults();
    Base = struct( ...
        'queryMode',Defaults.queryMode, ...
        'sampleZMode',"random", ...
        'trainZMode',"random", ...
        'sigma',1.0, ...
        'ganIter',Defaults.ganIter, ...
        'nCritic',Defaults.nCritic);

    Branches = repmat(Base,4,1);
    Branches(1).name = "query_boundary_populated";
    Branches(1).queryMode = "boundary_populated";

    Branches(2).name = "z_zero_sample";
    Branches(2).sampleZMode = "zero";

    Branches(3).name = "z_sigma_025";
    Branches(3).sigma = 0.25;

    Branches(4).name = "wgan_iter100";
    Branches(4).ganIter = 100;
end

function Params = wganParams(Branch)
    Defaults = CBS_RegionWGAN_GP.mainlineDefaults();
    Params = { ...
        Defaults.trainGap, ...
        Defaults.archiveGap, ...
        Defaults.nGen, ...
        Defaults.zDim, ...
        Branch.ganIter, ...
        Defaults.ganMiniBatch, ...
        Defaults.ganLrD, ...
        Defaults.ganLrG, ...
        Defaults.frontDepth, ...
        Defaults.pairNeighborRefRadius, ...
        Defaults.refDivisor, ...
        Defaults.minBoundaryLength, ...
        Defaults.queryPerCondition, ...
        Defaults.gpLambda, ...
        Branch.nCritic, ...
        Defaults.maxAnchorsPerRef, ...
        Defaults.minGANTrainCount};
end

function yes = branchComplete(outDir,problemCount,runCount,stageCount)
    yes = false;
    summaryFile = fullfile(outDir,'run_summary.csv');
    stageFile = fullfile(outDir,'stage_snapshots_all.csv');
    figureDir = fullfile(outDir,'domain_figures_all');
    if ~isfile(summaryFile) || ~isfile(stageFile) || ~isfolder(figureDir)
        return;
    end
    try
        figs = dir(fullfile(figureDir,'*domain_boundary.png'));
        yes = countTableRows(summaryFile) == problemCount*runCount && ...
            countOkRunRows(summaryFile) == problemCount*runCount && ...
            countTableRows(stageFile) == problemCount*stageCount && ...
            numel(figs) == problemCount*stageCount;
    catch
        yes = false;
    end
end

function Row = fillBranchCounts(Row,outDir)
    Row.run_count = countTableRows(fullfile(outDir,'run_summary.csv'));
    Row.stage_snapshot_count = countTableRows(fullfile(outDir, ...
        'stage_snapshots_all.csv'));
    Row.figure_count = numel(dir(fullfile(outDir,'domain_figures_all', ...
        '*domain_boundary.png')));
    summaryFile = fullfile(outDir,'run_summary.csv');
    if isfile(summaryFile)
        Row.ok_run_count = countOkRunRows(summaryFile);
    end
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
    fid = fopen(file,'r');
    cleanup = onCleanup(@()fclose(fid));
    header = fgetl(fid);
    if ~ischar(header)
        return;
    end
    names = split(string(header),",");
    statusIdx = find(strcmpi(names,"status"),1);
    if isempty(statusIdx)
        statusIdx = 11;
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

function value = doubleOrNaN(x)
    if isempty(x)
        value = NaN;
    else
        value = double(x);
    end
end

function Row = emptyBranchRow()
    Row = struct( ...
        'branch',"", ...
        'status',"pending", ...
        'out_dir',"", ...
        'query_mode',"", ...
        'sample_z_mode',"", ...
        'train_z_mode',"", ...
        'sigma',NaN, ...
        'gan_iter',NaN, ...
        'n_critic',NaN, ...
        'run_count',0, ...
        'ok_run_count',0, ...
        'stage_snapshot_count',0, ...
        'figure_count',0, ...
        'error_message',"");
end
