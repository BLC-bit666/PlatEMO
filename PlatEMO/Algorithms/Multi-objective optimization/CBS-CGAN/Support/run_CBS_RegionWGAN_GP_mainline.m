function [Summary,outDir] = run_CBS_RegionWGAN_GP_mainline( ...
        outDir,workerCount,problemNames,N,D,maxFE,runIds,Options)
%RUN_CBS_REGIONWGAN_GP_MAINLINE Run the fixed mainline and save final IGD.
%   Parallel formal runs use exactly nine workers. Each task is immutable,
%   resumable, and stores only the final IGD plus operational metadata.

    rootDir = fileparts(which('platemo'));
    if isempty(rootDir)
        rootDir = pwd;
    end
    addpath(genpath(rootDir));
    if nargin < 1 || isempty(outDir)
        outDir = fullfile(rootDir,'Data','CBS_RegionGAN_compare', ...
            ['mainline_igd_',char(datetime('now', ...
            'Format','yyyyMMdd_HHmmss'))]);
    end
    if nargin < 2 || isempty(workerCount); workerCount = 9; end
    if nargin < 3 || isempty(problemNames)
        problemNames = "LIRCMOP" + string((5:10)') + "_BC";
    end
    if nargin < 4 || isempty(N); N = 100; end
    if nargin < 5 || isempty(D); D = 30; end
    if nargin < 6 || isempty(maxFE); maxFE = 100000; end
    if nargin < 7 || isempty(runIds); runIds = 1:3; end
    if nargin < 8 || isempty(Options); Options = struct(); end
    Options = normalizeOptions(Options);
    workerCount = max(1,round(double(workerCount)));
    if workerCount ~= 1 && workerCount ~= 9
        error('CBSRegionGAN:MainlineWorkerCount', ...
            'Use one worker for tests or exactly nine workers for formal runs.');
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
    runIds = double(runIds(:)');
    N = max(1,round(double(N)));
    D = max(1,round(double(D)));
    maxFE = max(1,round(double(maxFE)));

    if ~isfolder(outDir); mkdir(outDir); end
    Provenance = CBS_RegionGAN_Provenance(rootDir,Options,workerCount);
    writeRootArtifacts(outDir,Provenance,N,D,maxFE,problemNames,runIds);
    Tasks = buildTasks(problemNames,runIds);
    rows = repmat(emptyRunRow(),height(Tasks),1);

    if workerCount == 1
        for task = 1 : height(Tasks)
            rows(task) = runTask(Tasks.problem(task),Tasks.run(task), ...
                outDir,N,D,maxFE,Options,Provenance);
            reportProgress(task,height(Tasks),rows(task));
        end
    else
        ensurePool(workerCount);
        done = 0;
        queue = parallel.pool.DataQueue;
        afterEach(queue,@onFinished);
        taskProblems = Tasks.problem;
        taskRuns = Tasks.run;
        parfor task = 1 : height(Tasks)
            rows(task) = runTask(taskProblems(task),taskRuns(task), ...
                outDir,N,D,maxFE,Options,Provenance);
            send(queue,rows(task));
        end
    end

    Summary = struct2table(rows);
    writetable(Summary,fullfile(outDir,'run_summary.csv'));
    failed = Summary.status ~= "ok";
    if any(failed)
        error('CBSRegionGAN:MainlineTasksFailed', ...
            ['%d of %d mainline tasks failed. Inspect run_summary.csv; ', ...
            'after diagnosis, rerun the same manifest to execute only ', ...
            'failed tasks.'],sum(failed),height(Summary));
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
    Options.schemaVersion = "cbs_region_wgan_igd_mainline_v1";
    Options.resume = logical(Options.resume);
    if ~isscalar(Options.resume)
        error('CBSRegionGAN:BadResume','Options.resume must be scalar.');
    end
end

function Tasks = buildTasks(problemNames,runIds)
    [problemGrid,runGrid] = ndgrid(problemNames,runIds);
    Tasks = table(problemGrid(:),runGrid(:), ...
        'VariableNames',{'problem','run'});
end

function Row = runTask(problemName,runId,outDir,N,D,maxFE, ...
        Options,Provenance)
    Row = emptyRunRow();
    Row.problem = string(problemName);
    Row.run = double(runId);
    Row.seed = double(runId);
    Row.N = double(N);
    Row.D = double(D);
    Row.maxFE = double(maxFE);
    Row.source_tree_sha256 = string(Provenance.source_tree_sha256);
    Row.task_signature = taskSignature(Row,Options);
    taskRoot = fullfile(outDir,sprintf('%s_run%d', ...
        char(Row.problem),round(Row.run)));
    if ~isfolder(taskRoot); mkdir(taskRoot); end
    if Options.resume
        [Row,reused] = reusableTask(taskRoot,Row);
        if reused
            Row.reused = 1;
            return;
        end
    end

    attemptDir = fullfile(taskRoot,nextAttemptName(taskRoot));
    mkdir(attemptDir);
    Row.attempt_folder = string(attemptDir);
    Row.task_result_file = string(fullfile(attemptDir,'task_result.mat'));
    wallTimer = tic;
    try
        limitWorkerThreads();
        rng(Row.seed,'twister');
        Constructor = str2func(char(Row.problem));
        Problem = Constructor('N',N,'D',D,'maxFE',maxFE);
        Algorithm = CBS_RegionWGAN_GP('save',0, ...
            'outputFcn',@quietOutput);
        Algorithm.Solve(Problem);
        Row.wall_seconds = toc(wallTimer);
        Row.runtime_seconds = double(Algorithm.metric.runtime);
        Row.finalFE = double(Algorithm.result{end,1});
        Population = Algorithm.result{end,2};
        Row.M = double(Problem.M);
        Row.IGD = double(Problem.CalMetric('IGD',Population));
        if Row.finalFE ~= Row.maxFE || ~isfinite(Row.IGD)
            error('CBSRegionGAN:IncompleteMainlineRun', ...
                'Expected finalFE=%d and finite IGD, got %d and %.17g.', ...
                Row.maxFE,Row.finalFE,Row.IGD);
        end
        Row.status = "ok";
        TaskResult = struct('row',Row);
    catch Error
        Row.wall_seconds = toc(wallTimer);
        Row.status = "failed";
        Row.error_identifier = string(Error.identifier);
        Row.error_message = string(Error.message);
        TaskResult = struct('row',Row);
    end
    save(Row.task_result_file,'TaskResult');
end

function [Row,found] = reusableTask(taskRoot,Expected)
    Row = Expected;
    found = false;
    files = dir(fullfile(taskRoot,'attempt_*','task_result.mat'));
    if isempty(files); return; end
    [~,order] = sort(string({files.folder}),'descend');
    for i = order
        try
            Loaded = load(fullfile(files(i).folder,files(i).name), ...
                'TaskResult');
            Candidate = Loaded.TaskResult.row;
            valid = string(Candidate.status) == "ok" && ...
                string(Candidate.task_signature) == Expected.task_signature && ...
                string(Candidate.source_tree_sha256) == ...
                    Expected.source_tree_sha256 && ...
                double(Candidate.finalFE) == Expected.maxFE && ...
                isfinite(double(Candidate.IGD));
            if valid
                Row = Candidate;
                found = true;
                return;
            end
        catch
        end
    end
end

function name = nextAttemptName(taskRoot)
    attempts = dir(fullfile(taskRoot,'attempt_*'));
    numbers = zeros(0,1);
    for i = 1 : numel(attempts)
        token = regexp(attempts(i).name,'^attempt_(\d+)$','tokens','once');
        if ~isempty(token); numbers(end+1,1) = str2double(token{1}); end %#ok<AGROW>
    end
    if isempty(numbers); number = 1; else; number = max(numbers)+1; end
    name = sprintf('attempt_%03d',number);
end

function signature = taskSignature(Row,Options)
    payload = struct( ...
        'problem',Row.problem,'run',Row.run,'seed',Row.seed, ...
        'N',Row.N,'D',Row.D,'maxFE',Row.maxFE, ...
        'source_tree_sha256',Row.source_tree_sha256, ...
        'schema_version',Options.schemaVersion);
    signature = sha256Text(string(jsonencode(payload)));
end

function value = sha256Text(value)
    digest = java.security.MessageDigest.getInstance('SHA-256');
    digest.update(unicode2native(char(value),'UTF-8'));
    value = lower(string(reshape(dec2hex(typecast( ...
        digest.digest(),'uint8'),2).',1,[])));
end

function writeRootArtifacts(outDir,P,N,D,maxFE,problemNames,runIds)
    Config = studyConfig(P,N,D,maxFE,problemNames,runIds);
    provenanceFile = fullfile(outDir,'provenance.csv');
    configFile = fullfile(outDir,'mainline_config.json');
    if isfile(provenanceFile)
        Existing = readtable(provenanceFile,'TextType','string');
        if ~ismember('source_tree_sha256',Existing.Properties.VariableNames) || ...
                string(Existing.source_tree_sha256(1)) ~= P.source_tree_sha256
            error('CBSRegionGAN:OutputProvenanceMismatch', ...
                'The output directory belongs to another source tree.');
        end
        if ~isfile(configFile)
            error('CBSRegionGAN:OutputConfigurationMismatch', ...
                'Existing output directory has no mainline_config.json.');
        end
        ExistingConfig = jsondecode(fileread(configFile));
        if ~isfield(ExistingConfig,'studySignature') || ...
                string(ExistingConfig.studySignature) ~= Config.studySignature
            error('CBSRegionGAN:OutputConfigurationMismatch', ...
                ['The output directory belongs to another experiment ', ...
                'configuration. Use a new output directory.']);
        end
    else
        Scalar = table(P.schema_version,P.git_sha,P.git_branch, ...
            double(P.git_dirty),P.matlab_release,P.host,P.worker_count, ...
            P.options_json,P.source_tree_sha256, ...
            'VariableNames',{'schema_version','git_sha','git_branch', ...
            'git_dirty','matlab_release','host','worker_count', ...
            'options_json','source_tree_sha256'});
        writetable(Scalar,provenanceFile);
        writetable(P.source_manifest,fullfile(outDir,'source_manifest.csv'));
        fid = fopen(configFile,'w');
        if fid < 0
            error('CBSRegionGAN:ConfigWriteFailed', ...
                'Cannot create %s.',configFile);
        end
        cleanup = onCleanup(@()fclose(fid));
        fwrite(fid,jsonencode(Config,'PrettyPrint',true),'char');
    end
end

function Config = studyConfig(P,N,D,maxFE,problemNames,runIds)
    Config = CBS_RegionWGAN_GP.mainlineDefaults();
    Config.N = N;
    Config.D = D;
    Config.maxFE = maxFE;
    Config.problems = problemNames;
    Config.runs = runIds;
    Config.workerCount = double(P.worker_count);
    Config.schemaVersion = string(P.schema_version);
    Config.sourceTreeSHA256 = string(P.source_tree_sha256);
    SignaturePayload = struct();
    SignaturePayload.N = N;
    SignaturePayload.D = D;
    SignaturePayload.maxFE = maxFE;
    SignaturePayload.problems = problemNames;
    SignaturePayload.runs = runIds;
    SignaturePayload.workerCount = double(P.worker_count);
    SignaturePayload.schemaVersion = string(P.schema_version);
    SignaturePayload.sourceTreeSHA256 = string(P.source_tree_sha256);
    Config.studySignature = sha256Text(string(jsonencode(SignaturePayload)));
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

function quietOutput(~,~)
end

function reportProgress(done,total,Row)
    fprintf('[%d/%d] %s run=%d status=%s IGD=%.8g reused=%d\n', ...
        done,total,char(Row.problem),Row.run,char(Row.status), ...
        Row.IGD,Row.reused);
end

function Row = emptyRunRow()
    Row = struct( ...
        'problem',"",'run',NaN,'seed',NaN, ...
        'N',NaN,'D',NaN,'M',NaN,'maxFE',NaN,'finalFE',NaN, ...
        'IGD',NaN,'runtime_seconds',NaN,'wall_seconds',NaN, ...
        'status',"pending",'reused',0, ...
        'error_identifier',"",'error_message',"", ...
        'source_tree_sha256',"",'task_signature',"", ...
        'attempt_folder',"",'task_result_file',"");
end
