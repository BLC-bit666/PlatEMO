function [Summary,outDir] = run_CBS_operator_triage(outDir,workerCount,Options)
%RUN_CBS_OPERATOR_TRIAGE Compare offspring-operator modes by final IGD.
%   Arms S1 (imtcmo_de) and S2 (ga_de_half) run on ten representative
%   problems with three seeds each. The A1 baseline is joined at analysis
%   time from the existing native mainline results, which is valid because
%   the default "de" path is byte-identical (fixed-seed regression test).

    rootDir = fileparts(which('platemo'));
    if isempty(rootDir); rootDir = pwd; end
    addpath(genpath(rootDir));
    if nargin < 1 || isempty(outDir)
        outDir = fullfile(rootDir,'Data','CBS_RegionGAN_compare', ...
            'operator_triage_v1');
    end
    if nargin < 2 || isempty(workerCount); workerCount = 10; end
    if nargin < 3 || isempty(Options); Options = struct(); end
    allowed = {'resume','problems','seeds','maxFE','N','arms'};
    unexpected = setdiff(fieldnames(Options),allowed);
    if ~isempty(unexpected)
        error('CBSRegionGAN:BadTriageOptions', ...
            'Unsupported option(s): %s.',strjoin(unexpected,', '));
    end
    Options = defaultField(Options,'resume',true);
    Options = defaultField(Options,'problems',[ ...
        "DASCMOP1_BC";"DASCMOP4_BC";"DASCMOP5_BC";"DASCMOP6_BC"; ...
        "DASCMOP7_BC";"DASCMOP8_BC";"DASCMOP9_BC"; ...
        "LIRCMOP6_BC";"LIRCMOP10_BC";"LIRCMOP13_BC"]);
    Options = defaultField(Options,'seeds',1:3);
    Options = defaultField(Options,'maxFE',200000);
    Options = defaultField(Options,'N',100);
    Options = defaultField(Options,'arms',["S1","S2"]);

    workerCount = max(1,round(double(workerCount)));
    if workerCount ~= 1 && workerCount ~= 10
        error('CBSRegionGAN:TriageWorkerCount', ...
            'Use one worker for tests or exactly ten workers formally.');
    end
    arms = string(Options.arms(:));
    modes = strings(size(arms));
    bls = strings(size(arms));
    for i = 1 : numel(arms)
        switch arms(i)
            case "S1";  modes(i) = "imtcmo_de";  bls(i) = "off";
            case "S2";  modes(i) = "ga_de_half"; bls(i) = "off";
            case "S2B"; modes(i) = "ga_de_half"; bls(i) = "on";
            case "A1";  modes(i) = "de";         bls(i) = "off";
            otherwise
                error('CBSRegionGAN:BadTriageArm', ...
                    'Arms must be selected from A1, S1, S2, and S2B.');
        end
    end
    problems = string(Options.problems(:));
    seeds = unique(round(double(Options.seeds(:)')),'stable');
    maxFE = max(1,round(double(Options.maxFE)));
    N = max(2,round(double(Options.N)));
    if ~isfolder(outDir); mkdir(outDir); end

    [armGrid,problemGrid,seedGrid] = ndgrid(1:numel(arms),problems,seeds);
    Tasks = table(armGrid(:),problemGrid(:),seedGrid(:), ...
        'VariableNames',{'arm_index','problem','seed'});
    rows = repmat(emptyRunRow(),height(Tasks),1);

    if workerCount == 1
        for task = 1 : height(Tasks)
            rows(task) = runTask(arms(Tasks.arm_index(task)), ...
                modes(Tasks.arm_index(task)),bls(Tasks.arm_index(task)), ...
                Tasks.problem(task),Tasks.seed(task),outDir,N,maxFE, ...
                Options.resume);
            reportProgress(task,height(Tasks),rows(task));
        end
    else
        ensurePool(workerCount);
        done = 0;
        queue = parallel.pool.DataQueue;
        afterEach(queue,@onFinished);
        taskArm = Tasks.arm_index;
        taskProblem = Tasks.problem;
        taskSeed = Tasks.seed;
        parfor task = 1 : height(Tasks)
            rows(task) = runTask(arms(taskArm(task)), ...
                modes(taskArm(task)),bls(taskArm(task)), ...
                taskProblem(task),taskSeed(task),outDir,N,maxFE, ...
                Options.resume);
            send(queue,rows(task));
        end
    end

    Summary = struct2table(rows);
    writetable(Summary,fullfile(outDir,'operator_triage_summary.csv'));
    failed = Summary.status ~= "ok";
    if any(failed)
        first = find(failed,1);
        error('CBSRegionGAN:TriageTasksFailed', ...
            '%d of %d tasks failed. First: %s %s run %d: %s', ...
            sum(failed),height(Summary),Summary.arm(first), ...
            Summary.problem(first),Summary.seed(first), ...
            Summary.error_message(first));
    end

    function onFinished(Row)
        done = done + 1;
        reportProgress(done,height(Tasks),Row);
    end
end

function Row = runTask(armId,mode,blsState,problemName,seedValue,outDir, ...
        N,maxFE,resume)
%RUNTASK Execute or reuse one arm/problem/seed task.

    Row = emptyRunRow();
    Row.arm = string(armId);
    Row.mode = string(mode);
    Row.bls = string(blsState);
    Row.problem = string(problemName);
    Row.seed = double(seedValue);
    Row.N = double(N);
    Row.maxFE = double(maxFE);
    Row.task_file = string(fullfile(outDir,sprintf('%s_%s_run%d.mat', ...
        char(armId),char(problemName),round(seedValue))));
    if resume && isfile(Row.task_file)
        try
            Loaded = load(Row.task_file,'TaskRow');
            Candidate = Loaded.TaskRow;
            if ~isfield(Candidate,'bls'); Candidate.bls = "off"; end
            if string(Candidate.status) == "ok" && ...
                    string(Candidate.mode) == Row.mode && ...
                    string(Candidate.bls) == Row.bls && ...
                    string(Candidate.problem) == Row.problem && ...
                    double(Candidate.seed) == Row.seed && ...
                    double(Candidate.maxFE) == Row.maxFE && ...
                    double(Candidate.N) == Row.N && ...
                    double(Candidate.finalFE) == Row.maxFE
                Row = orderfields(Candidate,Row);
                Row.reused = 1;
                return;
            end
        catch
        end
    end

    wallTimer = tic;
    try
        limitWorkerThreads();
        rng(Row.seed,'twister');
        Constructor = str2func(char(Row.problem));
        Problem = Constructor('N',N,'maxFE',maxFE);
        Algorithm = CBS_RegionWGAN_GP('save',0, ...
            'outputFcn',@quietOutput,'operatorMode',char(mode), ...
            'boundarySearch',char(blsState));
        if Algorithm.effectiveOperatorMode() ~= Row.mode || ...
                Algorithm.effectiveBoundarySearch() ~= Row.bls
            error('CBSRegionGAN:OperatorModeNotApplied', ...
                'Arm %s switches were not applied.',char(armId));
        end
        Algorithm.Solve(Problem);
        Row.finalFE = double(Problem.FE);
        Row.D = double(Problem.D);
        Row.M = double(Problem.M);
        Row.IGD = double(Problem.CalMetric('IGD',Algorithm.result{end,2}));
        Row.wall_seconds = toc(wallTimer);
        if Row.finalFE ~= Row.maxFE
            error('CBSRegionGAN:IncompleteTriageRun', ...
                'Expected finalFE=%d, got %d.',Row.maxFE,Row.finalFE);
        end
        Row.status = "ok";
    catch Error
        Row.wall_seconds = toc(wallTimer);
        Row.status = "failed";
        Row.error_identifier = string(Error.identifier);
        Row.error_message = string(Error.message);
    end
    TaskRow = Row; %#ok<NASGU>
    save(Row.task_file,'TaskRow');
end

function S = defaultField(S,name,value)
    if ~isfield(S,name) || isempty(S.(name)); S.(name) = value; end
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

function quietOutput(~,~)
end

function reportProgress(done,total,Row)
    fprintf('[%d/%d] %s %s run=%d status=%s IGD=%.8g reused=%d\n', ...
        done,total,char(Row.arm),char(Row.problem),Row.seed, ...
        char(Row.status),Row.IGD,Row.reused);
end

function Row = emptyRunRow()
    Row = struct( ...
        'arm',"",'mode',"",'bls',"off",'problem',"",'seed',NaN,'N',NaN,'D',NaN, ...
        'M',NaN,'maxFE',NaN,'finalFE',NaN,'IGD',NaN, ...
        'wall_seconds',NaN,'status',"pending",'reused',0, ...
        'task_file',"",'error_identifier',"",'error_message',"");
end
