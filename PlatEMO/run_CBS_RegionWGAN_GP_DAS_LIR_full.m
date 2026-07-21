%% Run the CBS RegionWGAN-GP campaign with its two module-level ablations
% This launcher executes three native PlatEMO campaigns on the complete
% DAS-CMOP_BC plus LIR-CMOP_BC sets (23 problems, ten seeds each):
%
%   Phase 1  CBS_RegionWGAN_GP_A00  module-removal ablation (no boundary
%            memory training, no guides, no calibration; plain 50/50
%            GA+DE backbone, 200 evaluations per generation) - fast
%   Phase 2  CBS_RegionWGAN_GP_CNB  learning-necessity ablation (the
%            trained guide generator is replaced by the trivial
%            copy-noise sampler; everything else identical) - fast
%   Phase 3  CBS_RegionWGAN_GP      the fixed GD20 guide mainline with
%            full-run boundary calibration feeding the boundary memory
%
% The cheap ablation phases run first so wiring faults surface within
% minutes instead of after the nine-hour mainline phase. Every run is
% saved through the native PlatEMO mechanism as
% Data/<Algorithm>/<Algorithm>_<problem>_M<M>_D<D>_<run>.mat containing
% exactly the variables result and metric; save=2 makes metric.IGD hold
% the ~100k-FE and the final values. Valid existing files are reused, so
% each output directory must not contain results from another
% configuration (earlier campaigns are archived as
% Data/CBS_RegionWGAN_GP-7-18 and Data/CBS_RegionWGAN_GP-7-19).
%
% To validate the manifest without opening a pool or running a task:
%   setenv('CBS_MAINLINE_DRY_RUN','1');
%   run_CBS_RegionWGAN_GP_DAS_LIR_full
%   setenv('CBS_MAINLINE_DRY_RUN','');

%% Fixed experiment configuration
rootDir = fileparts(mfilename('fullpath'));
addpath(genpath(rootDir));

dasProblems = "DASCMOP" + string((1:9)') + "_BC";
lirProblems = "LIRCMOP" + string((1:14)') + "_BC";
problemNames = [dasProblems; lirProblems];
runIds = 1:10;
workerCount = 10;
N = 100;
maxFE = 200000;
campaignNames = ["CBS_RegionWGAN_GP_A00"; ...
    "CBS_RegionWGAN_GP_CNB"; ...
    "CBS_RegionWGAN_GP"];

%% Preflight validation
runnerPath = which('run_CBS_RegionWGAN_GP_mainline');
if isempty(runnerPath)
    error('CBSRegionGAN:MissingMainlineRunner', ...
        'Cannot find run_CBS_RegionWGAN_GP_mainline below %s.',rootDir);
end
Probe = CBS_RegionWGAN_GP('save',0,'outputFcn',@(varargin)[]);
if Probe.effectiveOperatorMode() ~= "ga_de_half" || ...
        Probe.effectiveBoundarySearch() ~= "on" || ...
        Probe.effectiveScoutMode() ~= "off" || ...
        Probe.effectiveGeneratorMode() ~= "wgan" || ...
        Probe.effectiveGuideMode() ~= "on" || ...
        Probe.effectiveGuideShare() ~= 0.2 || ...
        Probe.effectiveGuideCarve() ~= "sym" || ...
        Probe.effectiveGuideWindow() ~= "half" || ...
        Probe.effectiveBlsWindow() ~= "full" || ...
        Probe.effectiveBlsFeed() ~= "on"
    error('CBSRegionGAN:MainlineDefaultsDrift', ...
        ['Algorithm defaults are not the fixed GD20 guide mainline ', ...
        '(ga_de_half + full-run calibration + 40/40/20 guided DE).']);
end
ProbeA00 = CBS_RegionWGAN_GP_A00('save',0,'outputFcn',@(varargin)[]);
if ProbeA00.effectiveGuideMode() ~= "off" || ...
        ProbeA00.effectiveBoundarySearch() ~= "off" || ...
        ProbeA00.effectiveScoutMode() ~= "off" || ...
        ProbeA00.effectiveGeneratorMode() ~= "wgan"
    error('CBSRegionGAN:AblationDefaultsDrift', ...
        'CBS_RegionWGAN_GP_A00 is not the module-removal configuration.');
end
ProbeCNB = CBS_RegionWGAN_GP_CNB('save',0,'outputFcn',@(varargin)[]);
if ProbeCNB.effectiveGeneratorMode() ~= "copynoise" || ...
        ProbeCNB.effectiveGuideMode() ~= "on" || ...
        ProbeCNB.effectiveGuideShare() ~= 0.2 || ...
        ProbeCNB.effectiveBoundarySearch() ~= "on" || ...
        ProbeCNB.effectiveBlsWindow() ~= "full" || ...
        ProbeCNB.effectiveBlsFeed() ~= "on"
    error('CBSRegionGAN:AblationDefaultsDrift', ...
        'CBS_RegionWGAN_GP_CNB is not the learning-necessity control.');
end
problemFiles = strings(size(problemNames));
for problem = 1 : numel(problemNames)
    problemFiles(problem) = string(which(char(problemNames(problem))));
end
missingProblems = problemNames(strlength(problemFiles) == 0);
if ~isempty(missingProblems)
    error('CBSRegionGAN:MissingBenchmarkProblem', ...
        'Cannot find benchmark class %s.',missingProblems(1));
end
if ~license('test','Distrib_Computing_Toolbox')
    error('CBSRegionGAN:MissingParallelToolbox', ...
        'Parallel Computing Toolbox is required for the ten-worker run.');
end
localCluster = parcluster('local');
if localCluster.NumWorkers < workerCount
    error('CBSRegionGAN:InsufficientLocalWorkers', ...
        'The local profile supports %d workers; this experiment requires %d.', ...
        localCluster.NumWorkers,workerCount);
end

Defaults = CBS_RegionWGAN_GP.mainlineDefaults();
ganStopFE = Defaults.ganStopFraction*maxFE;
taskCount = numel(problemNames)*numel(runIds);
fprintf(['CBS RegionWGAN-GP campaign + ablation manifest\n', ...
    '  Problems  : %d (DAS-CMOP_BC: %d, LIR-CMOP_BC: %d)\n', ...
    '  Runs      : %d per problem (%d tasks per campaign)\n', ...
    '  Campaigns : %s\n', ...
    '  Settings  : N=%d, maxFE=%d, CGAN stop FE=%d, save=2 (IGD@100k+final)\n', ...
    '  Pool      : local process pool, %d workers\n'], ...
    numel(problemNames),numel(dasProblems),numel(lirProblems), ...
    numel(runIds),taskCount,strjoin(campaignNames,' -> '), ...
    N,maxFE,ganStopFE,workerCount);

dryRunValue = lower(strtrim(string(getenv('CBS_MAINLINE_DRY_RUN'))));
dryRun = any(dryRunValue == ["1","true","yes"]);
if dryRun
    fprintf('Dry-run validation passed; no pool was opened and no task ran.\n');
    return;
end

%% Execute the three campaigns (cheap ablations first, mainline last)
for campaign = 1 : numel(campaignNames)
    campaignName = campaignNames(campaign);
    campaignDir = fullfile(rootDir,'Data',char(campaignName));
    fprintf('=== Campaign %d/%d: %s ===\n', ...
        campaign,numel(campaignNames),campaignName);
    Options = struct('resume',true,'algorithm',campaignName);
    [Summary,campaignDir] = run_CBS_RegionWGAN_GP_mainline( ...
        campaignDir,workerCount,problemNames,N,[],maxFE,runIds,Options);
    fprintf(['Campaign %s completed: %d tasks, %d finite IGD, ', ...
        '%d NaN IGD.\nNative PlatEMO results: %s\n'],campaignName, ...
        height(Summary),sum(isfinite(Summary.IGD)), ...
        sum(isnan(Summary.IGD)),campaignDir);
end
fprintf('All campaigns finished.\n');
