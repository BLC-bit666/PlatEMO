%% Run CBS RegionWGAN-GP on the complete DAS-CMOP_BC and LIR-CMOP_BC sets
% This launcher runs 23 problems with ten independent seeds per problem
% using the fixed S2+BLS mainline (half SBX+PM half DE offspring, CGAN
% active in the first 50% FE, boundary line search afterwards).
% Every problem uses the decision-space dimension declared by its class.
% It uses a ten-worker local process pool and writes native PlatEMO files
% containing result and metric. Valid existing files are reused, so the
% output directory must not contain results from another configuration
% (the previous pure-DE campaign is archived at Data/CBS_RegionWGAN_GP-7-18).
%
% To validate the manifest without opening a pool or running an algorithm:
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
Options = struct('resume',true);
outDir = fullfile(rootDir,'Data','CBS_RegionWGAN_GP');

%% Preflight validation
runnerPath = which('run_CBS_RegionWGAN_GP_mainline');
if isempty(runnerPath)
    error('CBSRegionGAN:MissingMainlineRunner', ...
        'Cannot find run_CBS_RegionWGAN_GP_mainline below %s.',rootDir);
end
Probe = CBS_RegionWGAN_GP('save',0,'outputFcn',@(varargin)[]);
if Probe.effectiveOperatorMode() ~= "ga_de_half" || ...
        Probe.effectiveBoundarySearch() ~= "on"
    error('CBSRegionGAN:MainlineDefaultsDrift', ...
        ['Algorithm defaults are not the fixed S2+BLS mainline ', ...
        '(ga_de_half + boundary line search).']);
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
fprintf(['CBS RegionWGAN-GP full benchmark manifest\n', ...
    '  Problems : %d (DAS-CMOP_BC: %d, LIR-CMOP_BC: %d)\n', ...
    '  Runs     : %d per problem (%d tasks total)\n', ...
    '  Settings : N=%d, maxFE=%d, CGAN stop FE=%d\n', ...
    '  Pool     : local process pool, %d workers\n', ...
    '  Output   : %s\n'], ...
    numel(problemNames),numel(dasProblems),numel(lirProblems), ...
    numel(runIds),taskCount,N,maxFE,ganStopFE,workerCount,outDir);

dryRunValue = lower(strtrim(string(getenv('CBS_MAINLINE_DRY_RUN'))));
dryRun = any(dryRunValue == ["1","true","yes"]);
if dryRun
    fprintf('Dry-run validation passed; no pool was opened and no task ran.\n');
    return;
end

%% Execute and save native PlatEMO results
[Summary,outDir] = run_CBS_RegionWGAN_GP_mainline( ...
    outDir,workerCount,problemNames,N,[],maxFE,runIds,Options);
fprintf(['All %d tasks completed: %d finite IGD, %d NaN IGD.\n', ...
    'Native PlatEMO results: %s\n'],height(Summary), ...
    sum(isfinite(Summary.IGD)),sum(isnan(Summary.IGD)),outDir);
