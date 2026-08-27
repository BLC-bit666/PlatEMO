function [RunSummary,Effects,Pairwise,outRoot] = ...
        run_CBS_CGAN_factor_experiment(outRoot,workerCount,problemNames, ...
        N,D,maxFE,runIds,arms)
%RUN_CBS_CGAN_FACTOR_EXPERIMENT Superseded three-arm development runner.
%   The formal matrix reuses existing A0 data and runs only A1/A2. Use the
%   root-level generic test.m script instead.

    if formalScriptSupersedesRunner()
        error('CBSRegionGAN:SupersededFactorRunner', ...
            ['Do not rerun A0. Run the root-level generic test.m ', ...
             'script for the formal A1/A2 matrix.']);
    end

    repoRoot = fileparts(which('platemo'));
    if isempty(repoRoot); repoRoot = pwd; end
    algorithmRoot = fileparts(fileparts(mfilename('fullpath')));
    addpath(algorithmRoot,'-begin');
    addCBSPaths(repoRoot);
    if nargin < 1 || isempty(outRoot)
        stamp = char(datetime('now','Format','yyyyMMdd_HHmmss'));
        outRoot = fullfile(repoRoot,'Data',['CBS_CGAN_factor_',stamp]);
    end
    if nargin < 2 || isempty(workerCount); workerCount = 10; end
    if nargin < 3 || isempty(problemNames)
        problemNames = ["DASCMOP1_BC";"DASCMOP3_BC";"DASCMOP6_BC"; ...
            "DASCMOP7_BC";"DASCMOP9_BC";"LIRCMOP5_BC"; ...
            "LIRCMOP7_BC";"LIRCMOP8_BC";"LIRCMOP11_BC"; ...
            "LIRCMOP12_BC";"LIRCMOP14_BC";"CF2_BC";"CF4_BC"; ...
            "CF5_BC";"CF6_BC"];
    end
    if nargin < 4 || isempty(N); N = 100; end
    if nargin < 5; D = []; end
    if nargin < 6 || isempty(maxFE); maxFE = 200000; end
    if nargin < 7 || isempty(runIds); runIds = 201:205; end
    if nargin < 8 || isempty(arms); arms = 0:2; end

    workerCount = positiveInteger(workerCount,'workerCount');
    if workerCount ~= 1 && workerCount ~= 10
        error('CBSRegionGAN:FactorWorkerCount', ...
            'Use one worker for validation or exactly ten for formal runs.');
    end
    N = positiveInteger(N,'N');
    maxFE = positiveInteger(maxFE,'maxFE');
    if maxFE ~= 200000
        error('CBSRegionGAN:FactorMaxFE', ...
            'The formal attribution contract requires maxFE=200000.');
    end
    problemNames = string(problemNames(:));
    if isempty(problemNames) || any(ismissing(problemNames) | ...
            strlength(problemNames) == 0)
        error('CBSRegionGAN:FactorProblemNames', ...
            'problemNames must contain nonempty class names.');
    end
    runIds = double(runIds(:)');
    if isempty(runIds) || any(~isfinite(runIds) | runIds < 1 | ...
            runIds ~= fix(runIds)) || numel(unique(runIds)) ~= numel(runIds)
        error('CBSRegionGAN:FactorRunIds', ...
            'runIds must be unique positive integers.');
    end
    arms = double(arms(:)');
    if ~isequal(arms,0:2)
        error('CBSRegionGAN:FactorArms', ...
            'Attribution requires all three ordered arms [0 1 2].');
    end
    useDefaultD = isempty(D);
    if ~useDefaultD; D = positiveInteger(D,'D'); end
    outRoot = freshOutputRoot(outRoot);
    createSourceSnapshot(repoRoot,algorithmRoot,outRoot);

    [dimensions,objectives] = resolveSettings( ...
        problemNames,N,D,maxFE,useDefaultD);
    Tasks = buildTasks(problemNames,runIds,arms,dimensions,objectives);
    rows = repmat(emptyRunRow(),height(Tasks),1);
    for arm = arms
        mkdir(fullfile(outRoot,sprintf('A%d',arm)));
    end
    if workerCount == 1
        for task = 1 : height(Tasks)
            rows(task) = runTask(Tasks(task,:),repoRoot,outRoot,N, ...
                maxFE,useDefaultD);
            reportProgress(task,height(Tasks),rows(task));
        end
    else
        ensurePool(workerCount);
        taskRows = table2struct(Tasks);
        parfor task = 1 : height(Tasks)
            rows(task) = runTask(struct2table(taskRows(task)), ...
                repoRoot,outRoot,N,maxFE,useDefaultD);
        end
    end
    RunSummary = struct2table(rows);
    save(fullfile(outRoot,'RunSummary.mat'),'RunSummary');
    writetable(RunSummary,fullfile(outRoot,'RunSummary.csv'));
    failed = RunSummary.status ~= "ok";
    if any(failed)
        first = find(failed,1);
        error('CBSRegionGAN:FactorTasksFailed', ...
            '%d tasks failed; first is A%d %s run %d: %s.', ...
            sum(failed),RunSummary.arm(first),RunSummary.problem(first), ...
            RunSummary.run(first),RunSummary.error_message(first));
    end
    [Effects,Pairwise] = analyze_CBS_CGAN_factor_experiment(RunSummary);
    save(fullfile(outRoot,'Attribution.mat'),'Effects','Pairwise');
    writetable(Effects,fullfile(outRoot,'Effects.csv'));
    writetable(Pairwise,fullfile(outRoot,'Pairwise.csv'));
end

function Row = runTask(Task,repoRoot,outRoot,N,maxFE,useDefaultD)
    Row = emptyRunRow();
    Row.arm = Task.arm;
    Row.problem = Task.problem;
    Row.run = Task.run;
    Row.seed = Task.run;
    Row.N = N;
    Row.D = Task.D;
    Row.M = Task.M;
    armRoot = fullfile(outRoot,sprintf('A%d',Row.arm));
    stem = sprintf('A%d_%s_M%d_D%d_run%d',Row.arm,char(Row.problem), ...
        Row.M,Row.D,Row.run);
    Row.result_file = string(fullfile(armRoot,[stem,'.mat']));
    Row.diagnostic_file = string(fullfile(armRoot,[stem,'_guide.mat']));
    if isfile(Row.result_file) || isfile(Row.diagnostic_file)
        Row.status = "failed";
        Row.error_identifier = "CBSRegionGAN:FactorExistingRun";
        Row.error_message = "Fresh-run file already exists.";
        return;
    end
    try
        addCBSPaths(repoRoot);
        limitWorkerThreads();
        rng(Row.seed,'twister');
        Constructor = str2func(char(Row.problem));
        if useDefaultD
            Problem = Constructor('N',N,'maxFE',maxFE);
        else
            Problem = Constructor('N',N,'D',Row.D,'maxFE',maxFE);
        end
        Algorithm = CBS_RegionWGAN_GP_Experiment( ...
            'parameter',{Row.arm},'save',2,'run',Row.run, ...
            'outputFcn',@(varargin)[]);
        started = tic;
        Algorithm.Solve(Problem);
        Row.runtime = toc(started);
        result = Algorithm.result;
        metric = Algorithm.metric;
        metric.IGD = Algorithm.CalMetric('IGD');
        metric.HV = Algorithm.CalMetric('HV');
        metric.Feasible_rate = Algorithm.CalMetric('Feasible_rate');
        guideSnapshot = Algorithm.guideExperimentSnapshot();
        experimentMetadata = struct('arm',Row.arm,'seed',Row.seed, ...
            'problem',Row.problem,'N',Row.N,'D',Row.D,'M',Row.M, ...
            'maxFE',maxFE);
        save(Row.result_file,'result','metric');
        save(Row.diagnostic_file,'guideSnapshot','experimentMetadata');
        Row.FE100K = double(result{1,1});
        Row.IGD100K = double(metric.IGD(1));
        Row.FE200K = double(result{end,1});
        Row.IGD200K = double(metric.IGD(end));
        Row.HV200K = double(metric.HV(end));
        Row.feasibleRate200K = double(metric.Feasible_rate(end));
        Row.generationEvents = guideSnapshot.generationEvents;
        Row.rawCandidates = guideSnapshot.rawCandidates;
        Row.criticKept = guideSnapshot.criticKept;
        Row.meanKeptConditions = guideSnapshot.meanKeptConditions;
        Row.selectionRate = guideSnapshot.selectionRate;
        Row.fallbackRate = guideSnapshot.fallbackRate;
        Row.meanAlpha = guideSnapshot.meanAlpha;
        Row.meanCenterStep = guideSnapshot.meanCenterStep;
        Row.meanActualStep = guideSnapshot.meanActualStep;
        Row.guidedFeasibleRate = guideSnapshot.guidedFeasibleRate;
        Row.guidedDominanceRate = guideSnapshot.guidedDominanceRate;
        Row.guidedSurvivalRate = guideSnapshot.guidedSurvivalRate;
        Row.status = "ok";
    catch Error
        Row.status = "failed";
        Row.error_identifier = string(Error.identifier);
        Row.error_message = string(Error.message);
    end
end

function [dimensions,objectives] = resolveSettings( ...
        problemNames,N,D,maxFE,useDefaultD)
    dimensions = zeros(numel(problemNames),1);
    objectives = zeros(numel(problemNames),1);
    savedRNG = rng;
    cleanup = onCleanup(@()rng(savedRNG));
    for i = 1 : numel(problemNames)
        Constructor = str2func(char(problemNames(i)));
        if useDefaultD
            Problem = Constructor('N',N,'maxFE',maxFE);
        else
            Problem = Constructor('N',N,'D',D,'maxFE',maxFE);
        end
        dimensions(i) = Problem.D;
        objectives(i) = Problem.M;
    end
    clear cleanup
end

function Tasks = buildTasks(problemNames,runIds,arms,D,M)
    [armGrid,problemGrid,runGrid] = ndgrid(arms,problemNames,runIds);
    dimensionGrid = repmat(reshape(D,1,[],1),numel(arms),1,numel(runIds));
    objectiveGrid = repmat(reshape(M,1,[],1),numel(arms),1,numel(runIds));
    Tasks = table(armGrid(:),problemGrid(:),runGrid(:), ...
        dimensionGrid(:),objectiveGrid(:), ...
        'VariableNames',{'arm','problem','run','D','M'});
end

function outRoot = freshOutputRoot(outRoot)
    if ~(ischar(outRoot) || (isstring(outRoot) && isscalar(outRoot)))
        error('CBSRegionGAN:FactorOutputRoot','outRoot must be scalar text.');
    end
    outRoot = char(outRoot);
    if isfolder(outRoot) && numel(dir(outRoot)) > 2
        error('CBSRegionGAN:FactorFreshRoot', ...
            'outRoot must be new or empty; resume is intentionally disabled.');
    end
    if ~isfolder(outRoot); mkdir(outRoot); end
end

function createSourceSnapshot(repoRoot,algorithmRoot,outRoot)
    snapshotRoot = fullfile(outRoot,'source_snapshot');
    mkdir(snapshotRoot);
    copyfile(algorithmRoot,fullfile(snapshotRoot,'CBS-CGAN'));
    copyfile(fullfile(repoRoot,'Agent.md'),fullfile(snapshotRoot,'Agent.md'));
    files = [dir(fullfile(snapshotRoot,'**','*.m')); ...
        dir(fullfile(snapshotRoot,'**','*.md')); ...
        dir(fullfile(snapshotRoot,'Agent.md'))];
    relativePath = strings(numel(files),1);
    sha256 = strings(numel(files),1);
    for i = 1 : numel(files)
        file = fullfile(files(i).folder,files(i).name);
        relativePath(i) = erase(string(file),string(snapshotRoot)+filesep);
        sha256(i) = fileSHA256(file);
    end
    Manifest = table(relativePath,sha256);
    save(fullfile(snapshotRoot,'Manifest.mat'),'Manifest');
    writetable(Manifest,fullfile(snapshotRoot,'Manifest.csv'));
end

function hash = fileSHA256(file)
    command = sprintf('/usr/bin/shasum -a 256 "%s"', ...
        strrep(file,'"','\"'));
    [status,out] = system(command);
    if status ~= 0
        error('CBSRegionGAN:FactorHashFailed', ...
            'Unable to hash source file %s.',file);
    end
    parts = split(strtrim(string(out)));
    hash = parts(1);
end

function ensurePool(workerCount)
    setThreadEnvironment();
    pool = gcp('nocreate');
    if ~isempty(pool) && pool.NumWorkers ~= workerCount
        delete(pool);
        pool = [];
    end
    if isempty(pool); parpool('local',workerCount); end
end

function limitWorkerThreads()
    setThreadEnvironment();
    try
        maxNumCompThreads(1);
    catch
    end
end

function setThreadEnvironment()
    names = {'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS', ...
        'MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'};
    for i = 1 : numel(names); setenv(names{i},'1'); end
end

function value = positiveInteger(value,name)
    if ~(isnumeric(value) && isscalar(value) && isfinite(value) && ...
            value >= 1 && value == fix(value))
        error('CBSRegionGAN:FactorPositiveInteger', ...
            '%s must be a positive integer.',name);
    end
    value = double(value);
end

function reportProgress(done,total,Row)
    fprintf('[%d/%d] A%d %s run=%d status=%s IGD200K=%.8g\n', ...
        done,total,Row.arm,Row.problem,Row.run,Row.status,Row.IGD200K);
end

function Row = emptyRunRow()
    Row = struct('arm',NaN,'problem',"",'run',NaN,'seed',NaN, ...
        'N',NaN,'D',NaN,'M',NaN,'FE100K',NaN,'IGD100K',NaN, ...
        'FE200K',NaN,'IGD200K',NaN,'HV200K',NaN, ...
        'feasibleRate200K',NaN,'runtime',NaN,'generationEvents',NaN, ...
        'rawCandidates',NaN,'criticKept',NaN,'meanKeptConditions',NaN, ...
        'selectionRate',NaN,'fallbackRate',NaN,'meanAlpha',NaN, ...
        'meanCenterStep',NaN,'meanActualStep',NaN, ...
        'guidedFeasibleRate',NaN,'guidedDominanceRate',NaN, ...
        'guidedSurvivalRate',NaN,'status',"pending", ...
        'result_file',"",'diagnostic_file',"", ...
        'error_identifier',"",'error_message',"");
end

function superseded = formalScriptSupersedesRunner()
    superseded = true;
end
