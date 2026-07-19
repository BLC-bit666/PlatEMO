function [Summary,outDir] = run_CBS_RegionWGAN_GP_mainline( ...
        outDir,workerCount,problemNames,N,D,maxFE,runIds,Options)
%RUN_CBS_REGIONWGAN_GP_MAINLINE Run and save native PlatEMO result files.
%   Formal runs use ten workers. Every completed run is saved as
%   Data/CBS_RegionWGAN_GP/CBS_RegionWGAN_GP_<problem>_M<M>_D<D>_<run>.mat
%   with exactly the native PlatEMO variables result and metric. An empty D
%   uses the default dimension declared by each problem class.

    rootDir = fileparts(which('platemo'));
    if isempty(rootDir)
        rootDir = pwd;
    end
    addpath(genpath(rootDir));
    if nargin < 1 || isempty(outDir)
        outDir = fullfile(rootDir,'Data','CBS_RegionWGAN_GP');
    end
    [outDir,runRoot] = validateOutputDirectory(outDir);
    if nargin < 2 || isempty(workerCount); workerCount = 10; end
    if nargin < 3 || isempty(problemNames)
        problemNames = "LIRCMOP" + string((5:10)') + "_BC";
    end
    if nargin < 4 || isempty(N); N = 100; end
    if nargin < 5; D = []; end
    if nargin < 6 || isempty(maxFE); maxFE = 200000; end
    if nargin < 7 || isempty(runIds); runIds = 1:3; end
    if nargin < 8 || isempty(Options); Options = struct(); end
    Options = normalizeOptions(Options);

    workerCount = validatePositiveInteger(workerCount,'workerCount');
    if workerCount ~= 1 && workerCount ~= 10
        error('CBSRegionGAN:MainlineWorkerCount', ...
            'Use one worker for tests or exactly ten workers for formal runs.');
    end
    if ischar(problemNames)
        problemNames = string(cellstr(problemNames));
    else
        problemNames = string(problemNames(:));
    end
    problemNames = problemNames(:);
    if isempty(problemNames) || any(ismissing(problemNames) | ...
            strlength(problemNames) == 0)
        error('CBSRegionGAN:BadProblemNames', ...
            'problemNames must contain nonempty MATLAB class names.');
    end
    N = validatePositiveInteger(N,'N');
    maxFE = validatePositiveInteger(maxFE,'maxFE');
    runIds = double(runIds(:)');
    if isempty(runIds) || any(~isfinite(runIds) | runIds < 1 | ...
            runIds ~= round(runIds)) || numel(unique(runIds)) ~= numel(runIds)
        error('CBSRegionGAN:BadRunIds', ...
            'runIds must contain unique positive integers.');
    end
    useProblemDefaultD = isempty(D);
    if ~useProblemDefaultD
        D = validatePositiveInteger(D,'D');
    end
    [problemDimensions,problemObjectives] = resolveProblemSettings( ...
        problemNames,N,D,maxFE,useProblemDefaultD);
    Tasks = buildTasks(problemNames,runIds,problemDimensions, ...
        problemObjectives);
    rows = repmat(emptyRunRow(),height(Tasks),1);

    if ~isfolder(runRoot); mkdir(runRoot); end
    if workerCount == 1
        for task = 1 : height(Tasks)
            rows(task) = runTask(Tasks.problem(task),Tasks.run(task), ...
                Tasks.D(task),Tasks.M(task),runRoot,outDir,N,maxFE, ...
                useProblemDefaultD,Options);
            reportProgress(task,height(Tasks),rows(task));
        end
    else
        ensurePool(workerCount);
        done = 0;
        queue = parallel.pool.DataQueue;
        afterEach(queue,@onFinished);
        taskProblems = Tasks.problem;
        taskRuns = Tasks.run;
        taskDimensions = Tasks.D;
        taskObjectives = Tasks.M;
        parfor task = 1 : height(Tasks)
            rows(task) = runTask(taskProblems(task),taskRuns(task), ...
                taskDimensions(task),taskObjectives(task),runRoot,outDir, ...
                N,maxFE,useProblemDefaultD,Options);
            send(queue,rows(task));
        end
    end

    Summary = struct2table(rows);
    failed = Summary.status ~= "ok";
    if any(failed)
        first = find(failed,1);
        error('CBSRegionGAN:MainlineTasksFailed', ...
            ['%d of %d tasks failed. First failure: %s run %d: %s. ', ...
            'No nonstandard task files were written.'], ...
            sum(failed),height(Summary),Summary.problem(first), ...
            Summary.run(first),Summary.error_message(first));
    end

    function onFinished(Row)
        done = done + 1;
        reportProgress(done,height(Tasks),Row);
    end
end

function Options = normalizeOptions(Options)
    unexpected = setdiff(fieldnames(Options),{'resume'});
    if ~isempty(unexpected)
        error('CBSRegionGAN:BadMainlineOptions', ...
            'The only supported runner option is resume.');
    end
    if ~isfield(Options,'resume') || isempty(Options.resume)
        Options.resume = true;
    end
    Options.resume = logical(Options.resume);
    if ~isscalar(Options.resume)
        error('CBSRegionGAN:BadResume','Options.resume must be scalar.');
    end
end

function value = validatePositiveInteger(value,name)
    if ~(isnumeric(value) && isscalar(value) && isreal(value) && ...
            isfinite(value) && value >= 1 && value == round(value))
        error('CBSRegionGAN:BadPositiveInteger', ...
            '%s must be a positive integer.',name);
    end
    value = double(value);
end

function [outDir,runRoot] = validateOutputDirectory(outDir)
    if ~(ischar(outDir) || (isstring(outDir) && isscalar(outDir)))
        error('CBSRegionGAN:BadOutputDirectory', ...
            'outDir must be a character vector or string scalar.');
    end
    outDir = char(outDir);
    while numel(outDir) > 1 && ismember(outDir(end),['/','\'])
        outDir(end) = [];
    end
    [dataDir,algorithmFolder] = fileparts(outDir);
    [runRoot,dataFolder] = fileparts(dataDir);
    if string(algorithmFolder) ~= "CBS_RegionWGAN_GP" || ...
            string(dataFolder) ~= "Data" || isempty(runRoot)
        error('CBSRegionGAN:NonstandardOutputDirectory', ...
            ['outDir must follow the native PlatEMO path ', ...
            '<root>/Data/CBS_RegionWGAN_GP.']);
    end
end

function [dimensions,objectives] = resolveProblemSettings( ...
        problemNames,N,D,maxFE,useProblemDefaultD)
    dimensions = zeros(numel(problemNames),1);
    objectives = zeros(numel(problemNames),1);
    savedRNG = rng;
    cleanup = onCleanup(@()rng(savedRNG));
    for problem = 1 : numel(problemNames)
        Constructor = str2func(char(problemNames(problem)));
        if useProblemDefaultD
            Problem = Constructor('N',N,'maxFE',maxFE);
        else
            Problem = Constructor('N',N,'D',D,'maxFE',maxFE);
        end
        dimensions(problem) = double(Problem.D);
        objectives(problem) = double(Problem.M);
        if any(~isfinite([dimensions(problem),objectives(problem)])) || ...
                any([dimensions(problem),objectives(problem)] < 1)
            error('CBSRegionGAN:BadProblemSetting', ...
                'Problem %s did not declare valid M and D values.', ...
                problemNames(problem));
        end
    end
    clear cleanup
end

function Tasks = buildTasks(problemNames,runIds,dimensions,objectives)
    [problemGrid,runGrid] = ndgrid(problemNames,runIds);
    dimensionGrid = repmat(dimensions(:),1,numel(runIds));
    objectiveGrid = repmat(objectives(:),1,numel(runIds));
    Tasks = table(problemGrid(:),runGrid(:),dimensionGrid(:), ...
        objectiveGrid(:),'VariableNames',{'problem','run','D','M'});
end

function Row = runTask(problemName,runId,D,M,runRoot,outDir,N,maxFE, ...
        useProblemDefaultD,Options)
    Row = emptyRunRow();
    Row.problem = string(problemName);
    Row.run = double(runId);
    Row.seed = double(runId);
    Row.N = double(N);
    Row.D = double(D);
    Row.M = double(M);
    Row.maxFE = double(maxFE);
    Row.result_file = string(standardResultFile(outDir,Row));
    if Options.resume
        [Row,reused] = readStandardResult(Row);
        if reused
            Row.reused = 1;
            return;
        end
    end

    try
        limitWorkerThreads();
        previousFolder = pwd;
        folderCleanup = onCleanup(@()cd(previousFolder));
        cd(runRoot);
        rng(Row.seed,'twister');
        Constructor = str2func(char(Row.problem));
        if useProblemDefaultD
            Problem = Constructor('N',N,'maxFE',maxFE);
        else
            Problem = Constructor('N',N,'D',D,'maxFE',maxFE);
        end
        if double(Problem.D) ~= Row.D || double(Problem.M) ~= Row.M
            error('CBSRegionGAN:ProblemSettingMismatch', ...
                'Resolved and constructed M/D values differ for %s.', ...
                Row.problem);
        end
        Algorithm = CBS_RegionWGAN_GP('save',1,'run',round(Row.run), ...
            'metName',{'IGD'}); %#ok<NASGU>
        evalc('Algorithm.Solve(Problem);');
        clear folderCleanup
        [Row,valid] = readStandardResult(Row);
        if ~valid
            error('CBSRegionGAN:InvalidNativeResult', ...
                'PlatEMO did not create a valid native result file: %s', ...
                Row.result_file);
        end
    catch Error
        Row.status = "failed";
        Row.error_identifier = string(Error.identifier);
        Row.error_message = string(Error.message);
    end
end

function filePath = standardResultFile(outDir,Row)
    filePath = fullfile(outDir,sprintf( ...
        'CBS_RegionWGAN_GP_%s_M%d_D%d_%d.mat',char(Row.problem), ...
        round(Row.M),round(Row.D),round(Row.run)));
end

function [Row,valid] = readStandardResult(Row)
    valid = false;
    filePath = char(Row.result_file);
    if ~isfile(filePath); return; end
    try
        Variables = whos('-file',filePath);
        if ~isequal(sort(string({Variables.name})),["metric","result"])
            return;
        end
        Saved = load(filePath,'result','metric');
        if ~iscell(Saved.result) || isempty(Saved.result) || ...
                size(Saved.result,2) ~= 2 || ...
                ~isstruct(Saved.metric) || ...
                ~all(isfield(Saved.metric,{'runtime','IGD'}))
            return;
        end
        finalFE = double(Saved.result{end,1});
        Population = Saved.result{end,2};
        if finalFE ~= Row.maxFE || ~isa(Population,'SOLUTION') || ...
                numel(Population) ~= Row.N || ...
                size(Population.decs,2) ~= Row.D || ...
                size(Population.objs,2) ~= Row.M
            return;
        end
        Row.finalFE = finalFE;
        Row.IGD = double(Saved.metric.IGD(end));
        Row.runtime_seconds = double(Saved.metric.runtime(end));
        Row.status = "ok";
        Row.error_identifier = "";
        Row.error_message = "";
        valid = true;
    catch
        valid = false;
    end
end

function ensurePool(workerCount)
    setThreadEnvironment();
    pool = gcp('nocreate');
    if ~isempty(pool) && pool.NumWorkers ~= workerCount
        delete(pool);
        pool = [];
    end
    if isempty(pool)
        parpool('local',workerCount);
    end
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

function reportProgress(done,total,Row)
    fprintf('[%d/%d] %s run=%d status=%s IGD=%.8g reused=%d file=%s\n', ...
        done,total,char(Row.problem),Row.run,char(Row.status), ...
        Row.IGD,Row.reused,char(Row.result_file));
end

function Row = emptyRunRow()
    Row = struct( ...
        'problem',"",'run',NaN,'seed',NaN,'N',NaN,'D',NaN,'M',NaN, ...
        'maxFE',NaN,'finalFE',NaN,'IGD',NaN,'runtime_seconds',NaN, ...
        'status',"pending",'reused',0,'result_file',"", ...
        'error_identifier',"",'error_message',"");
end
