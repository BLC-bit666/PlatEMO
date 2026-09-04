function State = run_CBS_CGAN_objective_space_campaign(rootPath,nWorker)
%RUN_CBS_CGAN_OBJECTIVE_SPACE_CAMPAIGN Run the fixed 10-problem figure run.

    if nargin < 1 || isempty(rootPath)
        rootPath = fileparts(which('platemo'));
    end
    if nargin < 2 || isempty(nWorker)
        nWorker = 10;
    end
    rootPath = char(rootPath);
    addCBSPaths(rootPath);
    Protocol = objectiveSpaceProtocol(rootPath,nWorker);
    validateProtocol(Protocol);
    campaignDir = fullfile(rootPath,'Data',Protocol.campaignName);
    resultDir = fullfile(campaignDir,'results');
    failureDir = fullfile(campaignDir,'failures');
    figureDir = fullfile(campaignDir,'figures');
    ensureFolder(campaignDir);
    ensureFolder(resultDir);
    ensureFolder(failureDir);
    ensureFolder(figureDir);

    Tasks = buildTasks(Protocol,resultDir);
    manifestFile = fullfile(campaignDir,'campaign_manifest.mat');
    if exist(manifestFile,'file') ~= 2
        save(manifestFile,'Protocol','Tasks','-v7.3');
    else
        Existing = load(manifestFile,'Protocol','Tasks');
        if ~protocolsMatch(Existing.Protocol,Protocol) || ...
                ~isequaln(Existing.Tasks,Tasks)
            error('CBSRegionGAN:ObjectiveSpaceManifestConflict', ...
                'The existing campaign manifest uses another protocol.');
        end
    end

    State = struct('schemaVersion',Protocol.schemaVersion, ...
        'status',"running",'startedAt',string(datetime('now')), ...
        'finishedAt',"",'totalTasks',numel(Tasks), ...
        'completeTasks',0,'remainingTasks',numel(Tasks), ...
        'attempt',0,'error',"");
    State = updateState(State,Tasks,Protocol,campaignDir);
    if State.remainingTasks == 0
        State.status = "rendering";
        saveState(State,Protocol,campaignDir);
        render_CBS_CGAN_objective_space_figures(rootPath, ...
            Protocol.campaignName);
        State.status = "complete";
        State.finishedAt = string(datetime('now'));
        saveState(State,Protocol,campaignDir);
        return;
    end

    threadVariables = {'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS', ...
        'MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'};
    for i = 1 : numel(threadVariables)
        setenv(threadVariables{i},'1');
    end
    pool = gcp('nocreate');
    ownsPool = isempty(pool);
    if ownsPool
        pool = parpool("Processes",Protocol.nWorker);
    elseif pool.NumWorkers ~= Protocol.nWorker || ...
            contains(class(pool),'ThreadPool')
        error('CBSRegionGAN:ObjectiveSpaceParallelPool', ...
            'The existing pool must contain %d process workers.', ...
            Protocol.nWorker);
    end
    cleanup = onCleanup(@()closeOwnedPool(pool,ownsPool));

    try
        pending = Tasks(~tasksComplete(Tasks,Protocol));
        for attempt = 1 : Protocol.maxAttempts
            if isempty(pending)
                break;
            end
            State.attempt = attempt;
            saveState(State,Protocol,campaignDir);
            fprintf('CBS objective-space: %d tasks, attempt %d/%d\n', ...
                numel(pending),attempt,Protocol.maxAttempts);
            options = parforOptions(pool,'RangePartitionMethod','fixed', ...
                'SubrangeSize',1);
            parfor (t = 1:numel(pending),options)
                Task = pending(t);
                try
                    runTask(rootPath,Task,Protocol);
                catch err
                    writeFailure(failureDir,Task,attempt,err);
                end
            end
            pending = pending(~tasksComplete(pending,Protocol));
            State = updateState(State,Tasks,Protocol,campaignDir);
            saveState(State,Protocol,campaignDir);
        end
        if ~isempty(pending)
            error('CBSRegionGAN:ObjectiveSpaceTasksFailed', ...
                '%d tasks failed after %d attempts.', ...
                numel(pending),Protocol.maxAttempts);
        end
        State.status = "rendering";
        State = updateState(State,Tasks,Protocol,campaignDir);
        saveState(State,Protocol,campaignDir);
        render_CBS_CGAN_objective_space_figures(rootPath, ...
            Protocol.campaignName);
        State.status = "complete";
        State.finishedAt = string(datetime('now'));
        State = updateState(State,Tasks,Protocol,campaignDir);
        saveState(State,Protocol,campaignDir);
        writelines("COMPLETE",fullfile(campaignDir,'COMPLETE.txt'));
    catch err
        State.status = "failed";
        State.finishedAt = string(datetime('now'));
        State.error = string(getReport(err,'extended','hyperlinks','off'));
        State = updateState(State,Tasks,Protocol,campaignDir);
        saveState(State,Protocol,campaignDir);
        writelines(State.error,fullfile(campaignDir,'FAILED.txt'));
        rethrow(err);
    end
    clear cleanup;
    closeOwnedPool(pool,ownsPool);
end

function Protocol = objectiveSpaceProtocol(rootPath,nWorker)
    Protocol = struct( ...
        'schemaVersion',"CBS-CGAN-objective-space-v1", ...
        'campaignName',"CBS_CGAN_objective_space_run1_10problems", ...
        'rootPath',string(rootPath), ...
        'algorithm',"PairGuide", ...
        'problems',["DASCMOP1_BC","DASCMOP2_BC", ...
            "DASCMOP4_BC","DASCMOP5_BC","DASCMOP9_BC", ...
            "LIRCMOP5_BC","LIRCMOP7_BC","LIRCMOP8_BC", ...
            "LIRCMOP10_BC","LIRCMOP14_BC"], ...
        'seed',1,'run',1,'popSize',100,'dimension',30, ...
        'maxFE',200000,'targetFE',10000:10000:100000, ...
        'expectedRawCount',500,'expectedGuidedCount',20, ...
        'nWorker',double(nWorker),'maxAttempts',3);
end

function validateProtocol(P)
    if P.algorithm ~= "PairGuide" || ...
            numel(P.problems) ~= 10 || P.seed ~= 1 || P.run ~= 1 || ...
            P.popSize ~= 100 || P.dimension ~= 30 || ...
            P.maxFE ~= 200000 || ...
            ~isequal(P.targetFE,10000:10000:100000) || ...
            P.expectedRawCount ~= 500 || ...
            P.expectedGuidedCount ~= 20 || ...
            P.nWorker < 1 || P.nWorker > 10
        error('CBSRegionGAN:BadObjectiveSpaceProtocol', ...
            'The fixed objective-space protocol was changed.');
    end
end

function same = protocolsMatch(A,B)
    fields = {'schemaVersion','campaignName','rootPath','algorithm', ...
        'problems','seed','run','popSize','dimension','maxFE', ...
        'targetFE','expectedRawCount','expectedGuidedCount'};
    same = all(isfield(A,fields)) && all(isfield(B,fields));
    for i = 1 : numel(fields)
        same = same && isequaln(A.(fields{i}),B.(fields{i}));
    end
end

function Tasks = buildTasks(P,resultDir)
    template = struct('problem',"",'outputFile',"");
    Tasks = repmat(template,numel(P.problems),1);
    for i = 1 : numel(P.problems)
        Tasks(i).problem = P.problems(i);
        Tasks(i).outputFile = string(fullfile(resultDir, ...
            P.problems(i)+"_run01.mat"));
    end
end

function OutputFile = runTask(rootPath,Task,P)
    cd(rootPath);
    addCBSPaths(rootPath);
    try
        maxNumCompThreads(1);
    catch
    end
    if exist(Task.outputFile,'file') == 2
        if resultComplete(Task,P)
            OutputFile = char(Task.outputFile);
            return;
        end
        error('CBSRegionGAN:ExistingObjectiveSpaceResult', ...
            'An invalid result already exists and will not be overwritten.');
    end

    rng(P.seed,'twister');
    problemConstructor = str2func(char(Task.problem));
    Problem = problemConstructor('N',P.popSize,'D',P.dimension, ...
        'maxFE',P.maxFE,'maxRuntime',Inf);
    algorithmConstructor = str2func(char(P.algorithm));
    Algorithm = algorithmConstructor('save',1,'run',P.run, ...
        'outputFcn',@silentOutput);
    Algorithm.configureObjectiveSpaceSnapshots(struct( ...
        'enabled',true,'targetFE',P.targetFE, ...
        'expectedRawCount',P.expectedRawCount, ...
        'expectedGuidedCount',P.expectedGuidedCount));

    started = datetime('now','TimeZone','local');
    wallClock = tic;
    Algorithm.Solve(Problem);
    elapsed = toc(wallClock);
    terminalRNG = rng;
    Snapshots = Algorithm.objectiveSpaceSnapshots();
    validateSnapshots(Task,P,Problem,Snapshots);
    Record = struct('schemaVersion',P.schemaVersion,'status',"ok", ...
        'algorithm',P.algorithm,'problem',Task.problem, ...
        'seed',P.seed,'run',P.run,'N',P.popSize,'D',P.dimension, ...
        'M',Problem.M,'maxFE',P.maxFE,'targetFE',P.targetFE, ...
        'startedAt',string(started), ...
        'finishedAt',string(datetime('now')), ...
        'wallClockSeconds',elapsed, ...
        'algorithmRuntimeSeconds',double(Algorithm.metric.runtime), ...
        'terminalRNG',terminalRNG);

    OutputFile = char(Task.outputFile);
    partialFile = [OutputFile,'.partial.mat'];
    save(partialFile,'Record','Snapshots','-v7.3');
    PartialTask = Task;
    PartialTask.outputFile = string(partialFile);
    if ~resultComplete(PartialTask,P)
        error('CBSRegionGAN:InvalidObjectiveSpacePartial', ...
            'The saved partial result failed validation.');
    end
    [moved,message] = movefile(partialFile,OutputFile);
    if ~moved
        error('CBSRegionGAN:ObjectiveSpaceResultMoveFailed','%s',message);
    end
end

function validateSnapshots(Task,P,Problem,S)
    if string(class(Problem)) ~= Task.problem || Problem.FE ~= P.maxFE || ...
            numel(S) ~= numel(P.targetFE) || ...
            ~isequal([S.targetFE],P.targetFE) || ...
            any([S.rawCount] ~= P.expectedRawCount) || ...
            any([S.guidedCount] ~= P.expectedGuidedCount) || ...
            any(~isfinite([S.actualFE])) || any(~isfinite([S.poolFE]))
        error('CBSRegionGAN:BadObjectiveSpaceSnapshots', ...
            'Snapshot identity, budget, FE, or point counts are invalid.');
    end
    for i = 1 : numel(S)
        if size(S(i).rawObjs,1) ~= P.expectedRawCount || ...
                size(S(i).guidedObjs,1) ~= P.expectedGuidedCount || ...
                size(S(i).rawObjs,2) ~= Problem.M || ...
                size(S(i).guidedObjs,2) ~= Problem.M || ...
                size(S(i).rawCons,1) ~= P.expectedRawCount || ...
                size(S(i).guidedCons,1) ~= P.expectedGuidedCount
            error('CBSRegionGAN:BadObjectiveSpaceSnapshotShape', ...
                'Snapshot %d has an invalid objective/constraint shape.',i);
        end
    end
    actualFE = [S.actualFE];
    poolFE = [S.poolFE];
    if any(diff(actualFE) < 0) || any(poolFE >= actualFE) || ...
            any(actualFE > 0.5*P.maxFE+2*P.popSize+20)
        error('CBSRegionGAN:BadObjectiveSpaceSnapshotFE', ...
            'Snapshot event FE values are unordered or outside the CGAN phase.');
    end
end

function mask = tasksComplete(Tasks,P)
    mask = false(numel(Tasks),1);
    for i = 1 : numel(Tasks)
        mask(i) = resultComplete(Tasks(i),P);
    end
end

function complete = resultComplete(Task,P)
    complete = false;
    if exist(Task.outputFile,'file') ~= 2
        return;
    end
    try
        Data = load(Task.outputFile,'Record','Snapshots');
        R = Data.Record;
        S = Data.Snapshots;
        complete = string(R.schemaVersion) == P.schemaVersion && ...
            string(R.status) == "ok" && string(R.algorithm) == P.algorithm && ...
            string(R.problem) == Task.problem && R.seed == P.seed && ...
            R.run == P.run && R.N == P.popSize && R.D == P.dimension && ...
            R.maxFE == P.maxFE && numel(S) == numel(P.targetFE) && ...
            isequal([S.targetFE],P.targetFE) && ...
            all([S.rawCount] == P.expectedRawCount) && ...
            all([S.guidedCount] == P.expectedGuidedCount) && ...
            all(isfinite([S.actualFE])) && all(isfinite([S.poolFE]));
        if complete
            for i = 1 : numel(S)
                complete = complete && ...
                    size(S(i).rawObjs,1) == P.expectedRawCount && ...
                    size(S(i).guidedObjs,1) == P.expectedGuidedCount && ...
                    size(S(i).rawObjs,2) == R.M && ...
                    size(S(i).guidedObjs,2) == R.M;
            end
            complete = complete && all(diff([S.actualFE]) >= 0) && ...
                all([S.poolFE] < [S.actualFE]) && ...
                all([S.actualFE] <= 0.5*P.maxFE+2*P.popSize+20);
        end
    catch
        complete = false;
    end
end

function State = updateState(State,Tasks,P,campaignDir)
    complete = tasksComplete(Tasks,P);
    State.completeTasks = sum(complete);
    State.remainingTasks = numel(Tasks)-State.completeTasks;
    saveState(State,P,campaignDir);
end

function saveState(State,Protocol,campaignDir)
    save(fullfile(campaignDir,'campaign_state.mat'),'State','Protocol');
    lines = ["schemaVersion="+State.schemaVersion; ...
        "status="+State.status; ...
        "completeTasks="+State.completeTasks; ...
        "remainingTasks="+State.remainingTasks; ...
        "attempt="+State.attempt];
    writelines(lines,fullfile(campaignDir,'campaign_state.txt'));
end

function writeFailure(failureDir,Task,attempt,err)
    ensureFolder(failureDir);
    name = sprintf('%s_attempt%d.txt',Task.problem,attempt);
    report = string(getReport(err,'extended','hyperlinks','off'));
    writelines(report,fullfile(failureDir,name));
end

function ensureFolder(folder)
    if ~isfolder(folder)
        mkdir(folder);
    end
end

function closeOwnedPool(pool,ownsPool)
    if ownsPool && ~isempty(pool) && isvalid(pool)
        delete(pool);
    end
end

function silentOutput(~,~)
end
