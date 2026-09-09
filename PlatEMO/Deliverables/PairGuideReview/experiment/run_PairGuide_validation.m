function State = run_PairGuide_validation(rootPath,nWorker)
%RUN_PAIRGUIDE_VALIDATION Run the fixed first PairGuide validation campaign.
%   Five LIR-CMOP_BC problems are run five times at 100000 FE. The problem
%   classes choose their default decision dimension; D is never supplied.
%   Run 1 stores objective-space snapshots, while every run stores the
%   final population, 10000-FE metrics, and the complete PairGuide audit.

    if nargin < 1 || isempty(rootPath)
        rootPath = fileparts(which('platemo'));
    end
    if nargin < 2 || isempty(nWorker)
        nWorker = 10;
    end
    rootPath = char(rootPath);
    addCBSPaths(rootPath);
    Protocol = validationProtocol(rootPath,nWorker);
    validateProtocol(Protocol);

    campaignDir = fullfile(rootPath,'Data',char(Protocol.campaignName));
    resultDir = fullfile(campaignDir,'results');
    failureDir = fullfile(campaignDir,'failures');
    ensureFolder(campaignDir);
    ensureFolder(resultDir);
    ensureFolder(failureDir);
    Tasks = buildTasks(Protocol,resultDir);
    ensureManifest(campaignDir,Protocol,Tasks);

    State = struct('schemaVersion',Protocol.schemaVersion, ...
        'campaignName',Protocol.campaignName,'status',"running", ...
        'startedAt',string(datetime('now')),'finishedAt',"", ...
        'totalTasks',numel(Tasks),'completeTasks',0, ...
        'remainingTasks',numel(Tasks),'attempt',0,'error',"");
    removeMarker(fullfile(campaignDir,'COMPLETE.txt'));
    removeMarker(fullfile(campaignDir,'FAILED.txt'));
    State = updateState(State,Tasks,Protocol,campaignDir);
    pending = Tasks(~tasksComplete(Tasks,Protocol));
    if isempty(pending)
        State = finishCampaign(State,Tasks,Protocol,campaignDir);
        return;
    end

    setProcessThreadLimits();
    [pool,ownsPool] = exactProcessPool(Protocol.nWorker);
    cleanup = onCleanup(@()closeOwnedPool(pool,ownsPool));
    try
        for attempt = 1 : Protocol.maxAttempts
            if isempty(pending)
                break;
            end
            State.attempt = attempt;
            saveState(State,Protocol,campaignDir);
            fprintf('PairGuide validation: %d tasks, attempt %d/%d\n', ...
                numel(pending),attempt,Protocol.maxAttempts);
            options = parforOptions(pool,'RangePartitionMethod','fixed', ...
                'SubrangeSize',1);
            parfor (i = 1:numel(pending),options)
                Task = pending(i);
                try
                    runTask(rootPath,Task,Protocol,failureDir);
                catch err
                    writeFailure(failureDir,Task,attempt,err);
                end
            end
            pending = Tasks(~tasksComplete(Tasks,Protocol));
            State = updateState(State,Tasks,Protocol,campaignDir);
        end
        if ~isempty(pending)
            error('PairGuide:ValidationTasksFailed', ...
                '%d tasks failed after %d attempts.', ...
                numel(pending),Protocol.maxAttempts);
        end
        State = finishCampaign(State,Tasks,Protocol,campaignDir);
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

function P = validationProtocol(rootPath,nWorker)
%VALIDATIONPROTOCOL Single source of truth for the approved first study.

    P = struct( ...
        'schemaVersion',"PairGuide-ball", ...
        'campaignName',"PairGuideBall", ...
        'rootPath',string(rootPath),'algorithm',"PairGuide", ...
        'problems',["LIRCMOP5_BC","LIRCMOP7_BC", ...
            "LIRCMOP8_BC","LIRCMOP10_BC","LIRCMOP12_BC"], ...
        'runs',1:5,'N',100,'maxFE',100000, ...
        'metricTargetFE',10000:10000:100000, ...
        'snapshotTargetFE',[1,10000:10000:100000], ...
        'snapshotRun',1,'expectedDefaultD',30, ...
        'expectedRawCount',500,'expectedRequestedCount',20, ...
        'expectedInitialEpoch',500,'expectedRetrainEpoch',20, ...
        'expectedNCritic',5,'expectedTrainingSigma',1, ...
        'expectedSampleSigma',0.3, ...
        'nWorker',double(nWorker),'maxAttempts',3);
end

function validateProtocol(P)
    expectedProblems = ["LIRCMOP5_BC","LIRCMOP7_BC", ...
        "LIRCMOP8_BC","LIRCMOP10_BC","LIRCMOP12_BC"];
    if P.algorithm ~= "PairGuide" || ...
            ~isequal(P.problems,expectedProblems) || ...
            ~isequal(P.runs,1:5) || P.N ~= 100 || P.maxFE ~= 100000 || ...
            ~isequal(P.metricTargetFE,10000:10000:100000) || ...
            ~isequal(P.snapshotTargetFE,[1,10000:10000:100000]) || ...
            P.snapshotRun ~= 1 || P.expectedDefaultD ~= 30 || ...
            P.expectedRawCount ~= 500 || ...
            P.expectedRequestedCount ~= 20 || ...
            P.expectedInitialEpoch ~= 500 || ...
            P.expectedRetrainEpoch ~= 20 || P.expectedNCritic ~= 5 || ...
            P.expectedTrainingSigma ~= 1 || P.expectedSampleSigma ~= 0.3 || ...
            P.nWorker ~= 10 || ...
            P.maxAttempts ~= 3
        error('PairGuide:BadValidationProtocol', ...
            'The fixed PairGuide validation protocol was changed.');
    end
end

function Tasks = buildTasks(P,resultDir)
    template = struct('problem',"",'run',0,'seed',0, ...
        'snapshotsEnabled',false,'outputFile',"");
    Tasks = repmat(template,numel(P.problems)*numel(P.runs),1);
    row = 0;
    % Run 1 is deliberately first and is the only plotted run.
    for run = P.runs
        for problem = 1 : numel(P.problems)
            row = row+1;
            Tasks(row).problem = P.problems(problem);
            Tasks(row).run = run;
            Tasks(row).seed = run;
            Tasks(row).snapshotsEnabled = run == P.snapshotRun;
            Tasks(row).outputFile = string(fullfile(resultDir,sprintf( ...
                '%s_run%02d.mat',P.problems(problem),run)));
        end
    end
end

function runTask(rootPath,Task,P,failureDir)
    cd(rootPath);
    addCBSPaths(rootPath);
    setProcessThreadLimits();
    try
        maxNumCompThreads(1);
    catch
    end

    outputFile = char(Task.outputFile);
    partialFile = [outputFile,'.partial.mat'];
    if isfile(outputFile)
        if resultComplete(Task,P)
            return;
        end
        quarantineFile(outputFile,failureDir,Task,"invalid_result");
    end
    if isfile(partialFile)
        PartialTask = Task;
        PartialTask.outputFile = string(partialFile);
        if resultComplete(PartialTask,P)
            atomicMove(partialFile,outputFile, ...
                'PairGuide:ValidationPartialRecoveryFailed');
            return;
        end
        quarantineFile(partialFile,failureDir,Task,"invalid_partial");
    end

    rng(Task.seed,'twister');
    constructor = str2func(char(Task.problem));
    % D intentionally omitted: each problem class supplies its default.
    Problem = constructor('N',P.N,'maxFE',P.maxFE,'maxRuntime',Inf);
    if Problem.D ~= P.expectedDefaultD
        error('PairGuide:UnexpectedDefaultDimension', ...
            '%s default D is %d, expected %d.', ...
            Task.problem,Problem.D,P.expectedDefaultD);
    end

    Trajectory = emptyTrajectory(P.metricTargetFE);
    Algorithm = PairGuide('save',1,'run',Task.run, ...
        'outputFcn',@captureTrajectory);
    % Same mainline values; this only enables fixed-probe train diagnostics.
    Algorithm.configurePairGuideTrainingExperiment(struct( ...
        'initialEpoch',P.expectedInitialEpoch, ...
        'retrainEpoch',P.expectedRetrainEpoch, ...
        'nCritic',P.expectedNCritic));
    Algorithm.configureCutoffDiagnostics(struct( ...
        'enabled',false,'stopAtCGANEnd',false, ...
        'disableOracleAudit',~Task.snapshotsEnabled));
    if Task.snapshotsEnabled
        Algorithm.configureObjectiveSpaceSnapshots(struct( ...
            'enabled',true,'targetFE',P.snapshotTargetFE, ...
            'expectedRawCount',P.expectedRawCount, ...
            'expectedGuidedCount',P.expectedRequestedCount));
    end

    started = datetime('now','TimeZone','local');
    wallClock = tic;
    Algorithm.Solve(Problem);
    elapsed = toc(wallClock);
    terminalRNG = rng;
    Audit = Algorithm.guideExperimentSnapshot();
    Snapshots = Algorithm.objectiveSpaceSnapshots();
    Population = Algorithm.result{end,2};
    FinalPopulation = populationData(Population);
    FinalMetrics = populationMetrics(Problem,Population);
    Record = struct( ...
        'schemaVersion',P.schemaVersion,'status',"ok", ...
        'campaignName',P.campaignName,'algorithm',P.algorithm, ...
        'problem',Task.problem,'run',Task.run,'seed',Task.seed, ...
        'N',P.N,'D',double(Problem.D),'M',double(Problem.M), ...
        'problemDefaultD',true,'maxFE',P.maxFE, ...
        'trainingSigma',double(Audit.trainingSigma), ...
        'sampleSigma',double(Audit.sampleSigma), ...
        'finalFE',double(Problem.FE), ...
        'metricTargetFE',P.metricTargetFE, ...
        'snapshotsEnabled',logical(Task.snapshotsEnabled), ...
        'snapshotTargetFE',snapshotTargets(Task,P), ...
        'snapshotCapturedCount',snapshotCapturedCount(Snapshots), ...
        'startedAt',string(started), ...
        'finishedAt',string(datetime('now')), ...
        'wallClockSeconds',double(elapsed), ...
        'algorithmRuntimeSeconds',double(Algorithm.metric.runtime), ...
        'terminalRNG',terminalRNG);
    validateResultData(Task,P,Record,Trajectory,Snapshots,Audit, ...
        FinalPopulation,FinalMetrics);

    save(partialFile,'Record','Trajectory','Snapshots','Audit', ...
        'FinalPopulation','FinalMetrics','-v7.3');
    PartialTask = Task;
    PartialTask.outputFile = string(partialFile);
    if ~resultComplete(PartialTask,P)
        error('PairGuide:InvalidValidationPartial', ...
            'Saved partial failed validation: %s',partialFile);
    end
    atomicMove(partialFile,outputFile, ...
        'PairGuide:ValidationResultMoveFailed');

    function captureTrajectory(CurrentAlgorithm,CurrentProblem)
    %CAPTURETRAJECTORY Sample metrics only when a 10000-FE target is crossed.
        if isempty(CurrentAlgorithm.result)
            error('PairGuide:MissingValidationPopulation', ...
                'Output callback received no population.');
        end
        fe = double(CurrentProblem.FE);
        rows = find(isnan(Trajectory.actualFE) & ...
            Trajectory.targetFE <= fe);
        if isempty(rows)
            return;
        end
        CurrentPopulation = CurrentAlgorithm.result{end,2};
        Metrics = populationMetrics(CurrentProblem,CurrentPopulation);
        Trajectory.actualFE(rows) = fe;
        Trajectory.IGD(rows) = Metrics.IGD;
        Trajectory.HV(rows) = Metrics.HV;
        Trajectory.feasibleCount(rows) = Metrics.feasibleCount;
        Trajectory.nondominatedFeasibleCount(rows) = ...
            Metrics.nondominatedFeasibleCount;
    end
end

function T = emptyTrajectory(targetFE)
    count = numel(targetFE);
    T = struct('targetFE',double(targetFE(:)), ...
        'actualFE',nan(count,1),'IGD',nan(count,1),'HV',nan(count,1), ...
        'feasibleCount',nan(count,1), ...
        'nondominatedFeasibleCount',nan(count,1));
end

function Data = populationData(Population)
    Data = struct('decs',double(Population.decs), ...
        'objs',double(Population.objs),'cons',double(Population.cons));
end

function Metrics = populationMetrics(Problem,Population)
%POPULATIONMETRICS Behavior-neutral metrics from already evaluated points.

    savedFE = Problem.FE;
    savedRNG = rng;
    cleanup = onCleanup(@()restoreMetricState(Problem,savedFE,savedRNG));
    Metrics = struct('IGD',safeMetric(Problem,'IGD',Population), ...
        'HV',safeMetric(Problem,'HV',Population), ...
        'feasibleCount',0,'nondominatedFeasibleCount',0);
    constraints = double(Population.cons);
    if isempty(constraints)
        feasible = true(numel(Population),1);
    else
        feasible = all(constraints <= 0,2);
    end
    Metrics.feasibleCount = nnz(feasible);
    if any(feasible)
        Metrics.nondominatedFeasibleCount = numel(Population(feasible).best);
    end
    clear cleanup;
    restoreMetricState(Problem,savedFE,savedRNG);
end

function value = safeMetric(Problem,name,Population)
    try
        value = double(Problem.CalMetric(name,Population));
        if ~isscalar(value) || ~isreal(value) || value < 0 || ...
                (~isfinite(value) && ~isnan(value))
            value = NaN;
        end
    catch
        value = NaN;
    end
end

function restoreMetricState(Problem,fe,rngState)
    Problem.FE = fe;
    rng(rngState);
end

function targets = snapshotTargets(Task,P)
    if Task.snapshotsEnabled
        targets = P.snapshotTargetFE;
    else
        targets = zeros(1,0);
    end
end

function count = snapshotCapturedCount(Snapshots)
    if isempty(Snapshots) || ~isfield(Snapshots,'actualFE')
        count = 0;
    else
        count = nnz(isfinite(double([Snapshots.actualFE])));
    end
end

function complete = resultComplete(Task,P)
    complete = false;
    file = char(Task.outputFile);
    if ~isfile(file)
        return;
    end
    try
        variables = string({whos('-file',file).name});
        required = {'Record','Trajectory','Snapshots','Audit', ...
            'FinalPopulation','FinalMetrics'};
        if any(~ismember(string(required),variables))
            return;
        end
        Data = load(file,required{:});
        validateResultData(Task,P,Data.Record,Data.Trajectory, ...
            Data.Snapshots,Data.Audit,Data.FinalPopulation, ...
            Data.FinalMetrics);
        complete = true;
    catch
        complete = false;
    end
end

function validateResultData(Task,P,R,T,S,A,F,M)
    requiredRecord = {'schemaVersion','status','campaignName','algorithm', ...
        'problem','run','seed','N','D','M','problemDefaultD','maxFE', ...
        'trainingSigma','sampleSigma','finalFE','metricTargetFE','snapshotsEnabled', ...
        'snapshotTargetFE','snapshotCapturedCount'};
    exact = isstruct(R) && isscalar(R) && ...
        all(isfield(R,requiredRecord)) && ...
        string(R.schemaVersion) == P.schemaVersion && ...
        string(R.status) == "ok" && ...
        string(R.campaignName) == P.campaignName && ...
        string(R.algorithm) == P.algorithm && ...
        string(R.problem) == Task.problem && ...
        double(R.run) == Task.run && double(R.seed) == Task.seed && ...
        double(R.N) == P.N && double(R.D) == P.expectedDefaultD && ...
        isscalar(R.M) && isfinite(double(R.M)) && double(R.M) >= 2 && ...
        logical(R.problemDefaultD) && double(R.maxFE) == P.maxFE && ...
        double(R.trainingSigma) == P.expectedTrainingSigma && ...
        double(R.sampleSigma) == P.expectedSampleSigma && ...
        double(R.finalFE) == P.maxFE && ...
        isequal(double(R.metricTargetFE(:)'),P.metricTargetFE) && ...
        logical(R.snapshotsEnabled) == Task.snapshotsEnabled && ...
        isequal(double(R.snapshotTargetFE(:)'),snapshotTargets(Task,P));
    if ~exact
        error('PairGuide:ValidationIdentityMismatch', ...
            'Saved result identity, default dimension, or FE budget differs.');
    end
    validateTrajectory(T,R,P);
    validateSnapshots(S,Task,P,R);
    validateAudit(A,Task,P);
    validateFinal(F,M,R,P);
end

function validateTrajectory(T,R,P)
    required = {'targetFE','actualFE','IGD','HV','feasibleCount', ...
        'nondominatedFeasibleCount'};
    if ~isstruct(T) || ~isscalar(T) || ~all(isfield(T,required))
        error('PairGuide:BadValidationTrajectory', ...
            'Trajectory fields are missing.');
    end
    target = double(T.targetFE(:));
    actual = double(T.actualFE(:));
    igd = double(T.IGD(:));
    hv = double(T.HV(:));
    feasible = double(T.feasibleCount(:));
    nondominated = double(T.nondominatedFeasibleCount(:));
    if ~isequal(target,P.metricTargetFE(:)) || ...
            numel(actual) ~= numel(target) || ...
            any(~isfinite(actual) | actual ~= round(actual)) || ...
            any(actual < target) || any(diff(actual) < 0) || ...
            actual(end) ~= P.maxFE || ...
            numel(igd) ~= numel(target) || numel(hv) ~= numel(target) || ...
            any(igd(isfinite(igd)) < 0) || any(hv(isfinite(hv)) < 0) || ...
            any(isinf(igd) | isinf(hv)) || ...
            any(~isfinite(feasible) | feasible < 0 | feasible > P.N | ...
                feasible ~= round(feasible)) || ...
            any(~isfinite(nondominated) | nondominated < 0 | ...
                nondominated > feasible | nondominated ~= round(nondominated)) || ...
            double(R.finalFE) ~= actual(end)
        error('PairGuide:InvalidValidationTrajectory', ...
            'Trajectory targets, ordering, metrics, or counts are invalid.');
    end
end

function validateSnapshots(S,Task,P,R)
    if ~Task.snapshotsEnabled
        if ~isempty(S) || double(R.snapshotCapturedCount) ~= 0
            error('PairGuide:UnexpectedValidationSnapshots', ...
                'Only run 1 may store point snapshots.');
        end
        return;
    end
    required = {'targetFE','actualFE','poolFE','rawCount', ...
        'requestedCount','guidedCount','fallbackCount','rawObjs', ...
        'rawCons','guidedObjs','guidedCons','population1Objs', ...
        'population1Cons','population2Objs','population2Cons'};
    if ~isstruct(S) || numel(S) ~= numel(P.snapshotTargetFE) || ...
            ~all(isfield(S,required)) || ...
            ~isequal(double([S.targetFE]),P.snapshotTargetFE)
        error('PairGuide:BadValidationSnapshots', ...
            'Run 1 snapshot targets or fields are invalid.');
    end
    captured = isfinite(double([S.actualFE]));
    if double(R.snapshotCapturedCount) ~= nnz(captured)
        error('PairGuide:BadValidationSnapshotCount', ...
            'Snapshot captured count differs from stored data.');
    end
    if ~any(captured)
        return;
    end
    if ~all(captured) || any(diff([S.actualFE]) < 0) || ...
            any(~isfinite([S.poolFE])) || any([S.poolFE] >= [S.actualFE])
        error('PairGuide:BadValidationSnapshotFE', ...
            'Captured snapshot FE values are invalid.');
    end
    for i = 1 : numel(S)
        if S(i).rawCount ~= P.expectedRawCount || ...
                S(i).requestedCount ~= P.expectedRequestedCount || ...
                S(i).guidedCount+S(i).fallbackCount ~= ...
                    P.expectedRequestedCount || ...
                size(S(i).rawObjs,1) ~= P.expectedRawCount || ...
                size(S(i).rawObjs,2) ~= R.M || ...
                size(S(i).rawCons,1) ~= P.expectedRawCount || ...
                size(S(i).guidedObjs,1) ~= S(i).guidedCount || ...
                size(S(i).guidedObjs,2) ~= R.M || ...
                size(S(i).guidedCons,1) ~= S(i).guidedCount || ...
                size(S(i).population1Objs,1) ~= P.N || ...
                size(S(i).population2Objs,1) ~= P.N
            error('PairGuide:BadValidationSnapshotShape', ...
                'Snapshot %d has invalid counts or matrix sizes.',i);
        end
    end
end

function validateAudit(A,Task,P)
    required = {'generationMode','useMode','nCritic', ...
        'trainingSigma','sampleSigma','pairInitialEpoch', ...
        'pairRetrainEpoch','diagnosticsEnabled', ...
        'oracleAuditDisabled','rawOracleCount','checkpointTargets', ...
        'checkpointFE','checkpointIGD','checkpointHV', ...
        'checkpointFeasibleCount','checkpointNondominatedFeasibleCount', ...
        'pairTrainingEvents','pairTrainingLog','ObjFE', ...
        'PairGuideFullFE'};
    if ~isstruct(A) || ~isscalar(A) || ~all(isfield(A,required)) || ...
            string(A.generationMode) ~= "pair_guide" || ...
            string(A.useMode) ~= "pair_guide" || ...
            double(A.nCritic) ~= P.expectedNCritic || ...
            double(A.trainingSigma) ~= P.expectedTrainingSigma || ...
            double(A.sampleSigma) ~= P.expectedSampleSigma || ...
            double(A.pairInitialEpoch) ~= P.expectedInitialEpoch || ...
            double(A.pairRetrainEpoch) ~= P.expectedRetrainEpoch || ...
            logical(A.diagnosticsEnabled) || ...
            logical(A.oracleAuditDisabled) ~= ~Task.snapshotsEnabled || ...
            double(A.rawOracleCount) ~= 0 || double(A.ObjFE) < 0 || ...
            double(A.PairGuideFullFE) < 0
        error('PairGuide:BadValidationAudit', ...
            'Audit does not confirm the approved PairGuide configuration.');
    end
    names = {'checkpointTargets','checkpointFE','checkpointIGD', ...
        'checkpointHV','checkpointFeasibleCount', ...
        'checkpointNondominatedFeasibleCount'};
    count = numel(A.checkpointTargets);
    if count < 1 || any(cellfun(@(name)numel(A.(name)) ~= count,names))
        error('PairGuide:BadValidationAuditCheckpoints', ...
            'Audit checkpoint arrays have inconsistent lengths.');
    end
    Log = A.pairTrainingLog;
    if ~isstruct(Log) || double(A.pairTrainingEvents) ~= numel(Log)
        error('PairGuide:BadValidationTrainingLog', ...
            'Training event count differs from training log.');
    end
    if isempty(Log)
        return;
    end
    logRequired = {'fe','kind','nCritic','epochs','trainingPairs', ...
        'updates','pairVisits','batchesPerEpoch'};
    if ~all(isfield(Log,logRequired))
        error('PairGuide:BadValidationTrainingLog', ...
            'Training log fields are missing.');
    end
    kinds = string({Log.kind});
    epochs = double([Log.epochs]);
    pairs = double([Log.trainingPairs]);
    updates = double([Log.updates]);
    visits = double([Log.pairVisits]);
    batches = double([Log.batchesPerEpoch]);
    fe = double([Log.fe]);
    if kinds(1) ~= "initial" || epochs(1) ~= P.expectedInitialEpoch || ...
            any(kinds(2:end) ~= "retrain") || ...
            any(epochs(2:end) ~= P.expectedRetrainEpoch) || ...
            any(double([Log.nCritic]) ~= P.expectedNCritic) || ...
            any(~isfinite(fe)) || any(diff(fe) <= 0) || ...
            any(pairs < 32 | pairs ~= round(pairs)) || ...
            any(updates ~= epochs.*batches) || any(visits ~= epochs.*pairs)
        error('PairGuide:InvalidValidationTrainingLog', ...
            'Training schedule or complete-pair visit count is invalid.');
    end
end

function validateFinal(F,M,R,P)
    if ~isstruct(F) || ~isscalar(F) || ...
            ~all(isfield(F,{'decs','objs','cons'})) || ...
            size(F.decs,1) ~= P.N || size(F.decs,2) ~= R.D || ...
            size(F.objs,1) ~= P.N || size(F.objs,2) ~= R.M || ...
            size(F.cons,1) ~= P.N || ...
            any(~isfinite(F.decs),'all') || any(~isfinite(F.objs),'all') || ...
            any(~isfinite(F.cons),'all') || ...
            ~isstruct(M) || ~isscalar(M) || ...
            ~all(isfield(M,{'IGD','HV','feasibleCount', ...
                'nondominatedFeasibleCount'})) || ...
            any(isinf(double([M.IGD,M.HV]))) || ...
            ~isfinite(double(M.feasibleCount)) || ...
            M.feasibleCount < 0 || M.feasibleCount > P.N || ...
            M.feasibleCount ~= round(M.feasibleCount) || ...
            ~isfinite(double(M.nondominatedFeasibleCount)) || ...
            M.nondominatedFeasibleCount < 0 || ...
            M.nondominatedFeasibleCount > M.feasibleCount || ...
            M.nondominatedFeasibleCount ~= ...
                round(M.nondominatedFeasibleCount)
        error('PairGuide:BadValidationFinalResult', ...
            'Final population or metrics are invalid.');
    end
end

function mask = tasksComplete(Tasks,P)
    mask = false(numel(Tasks),1);
    for i = 1 : numel(Tasks)
        mask(i) = resultComplete(Tasks(i),P);
    end
end

function ensureManifest(campaignDir,Protocol,Tasks)
    file = fullfile(campaignDir,'campaign_manifest.mat');
    if isfile(file)
        Existing = load(file,'Protocol','Tasks');
        if ~isequaln(Existing.Protocol,Protocol) || ...
                ~isequaln(Existing.Tasks,Tasks)
            error('PairGuide:ValidationManifestConflict', ...
                'Existing campaign manifest uses another protocol.');
        end
        return;
    end
    partial = [file,'.partial.mat'];
    save(partial,'Protocol','Tasks','-v7.3');
    atomicMove(partial,file,'PairGuide:ValidationManifestMoveFailed');
end

function State = updateState(State,Tasks,P,campaignDir)
    complete = tasksComplete(Tasks,P);
    State.completeTasks = nnz(complete);
    State.remainingTasks = numel(Tasks)-State.completeTasks;
    saveState(State,P,campaignDir);
end

function State = finishCampaign(State,Tasks,P,campaignDir)
    Rows = repmat(emptySummaryRow(),numel(Tasks),1);
    for i = 1 : numel(Tasks)
        Data = load(Tasks(i).outputFile,'Record','FinalMetrics','Audit');
        Rows(i) = summaryRow(Data.Record,Data.FinalMetrics,Data.Audit, ...
            Tasks(i).outputFile);
    end
    Summary = struct2table(Rows);
    summaryFile = fullfile(campaignDir,'campaign_summary.mat');
    partial = [summaryFile,'.partial.mat'];
    save(partial,'Summary','P','Tasks','-v7.3');
    atomicMove(partial,summaryFile,'PairGuide:ValidationSummaryMoveFailed');
    State.status = "complete";
    State.finishedAt = string(datetime('now'));
    State.error = "";
    State = updateState(State,Tasks,P,campaignDir);
    saveState(State,P,campaignDir);
    writelines("COMPLETE",fullfile(campaignDir,'COMPLETE.txt'));
end

function Row = summaryRow(R,M,A,file)
    Row = emptySummaryRow();
    Row.problem = string(R.problem);
    Row.run = double(R.run);
    Row.seed = double(R.seed);
    Row.D = double(R.D);
    Row.M = double(R.M);
    Row.finalIGD = double(M.IGD);
    Row.finalHV = double(M.HV);
    Row.finalFeasibleCount = double(M.feasibleCount);
    Row.finalNondominatedFeasibleCount = ...
        double(M.nondominatedFeasibleCount);
    Row.firstTrainingFE = auditScalar(A,'firstTrainingFE',Inf);
    Row.firstGuidedUseFE = auditScalar(A,'firstGuidedUseFE',Inf);
    Row.trainingEvents = auditScalar(A,'pairTrainingEvents',0);
    Row.guideUseEvents = auditScalar(A,'useEvents',0);
    Row.guidedSelected = auditScalar(A,'guidedSelected',0);
    Row.guidedFallback = auditScalar(A,'guidedFallback',0);
    Row.ObjFE = auditScalar(A,'ObjFE',0);
    Row.PairGuideFullFE = auditScalar(A,'PairGuideFullFE',0);
    Row.wallClockSeconds = double(R.wallClockSeconds);
    Row.resultFile = string(file);
end

function value = auditScalar(A,name,default)
    value = default;
    if isfield(A,name) && isnumeric(A.(name)) && ...
            isscalar(A.(name)) && isreal(A.(name))
        value = double(A.(name));
    end
end

function Row = emptySummaryRow()
    Row = struct('problem',"",'run',NaN,'seed',NaN,'D',NaN,'M',NaN, ...
        'finalIGD',NaN,'finalHV',NaN,'finalFeasibleCount',NaN, ...
        'finalNondominatedFeasibleCount',NaN,'firstTrainingFE',Inf, ...
        'firstGuidedUseFE',Inf,'trainingEvents',0,'guideUseEvents',0, ...
        'guidedSelected',0,'guidedFallback',0,'ObjFE',0, ...
        'PairGuideFullFE',0,'wallClockSeconds',NaN,'resultFile',"");
end

function saveState(State,Protocol,campaignDir)
    file = fullfile(campaignDir,'campaign_state.mat');
    partial = [file,'.partial.mat'];
    save(partial,'State','Protocol');
    atomicReplace(partial,file,'PairGuide:ValidationStateMoveFailed');
    lines = ["schemaVersion="+State.schemaVersion; ...
        "status="+State.status; ...
        "completeTasks="+State.completeTasks; ...
        "remainingTasks="+State.remainingTasks; ...
        "attempt="+State.attempt];
    textFile = fullfile(campaignDir,'campaign_state.txt');
    textPartial = [textFile,'.partial.txt'];
    writelines(lines,textPartial);
    atomicReplace(textPartial,textFile, ...
        'PairGuide:ValidationStateTextMoveFailed');
end

function writeFailure(failureDir,Task,attempt,err)
    Failure = struct('problem',Task.problem,'run',Task.run, ...
        'seed',Task.seed,'attempt',attempt,'identifier',string(err.identifier), ...
        'message',string(err.message), ...
        'report',string(getReport(err,'extended','hyperlinks','off')), ...
        'failedAt',string(datetime('now')));
    stem = sprintf('%s_run%02d_attempt%d_%s',Task.problem,Task.run, ...
        attempt,timestampToken());
    save(fullfile(failureDir,[stem,'.mat']),'Failure');
    writelines(Failure.report,fullfile(failureDir,[stem,'.txt']));
end

function quarantineFile(file,failureDir,Task,reason)
    [~,name,extension] = fileparts(file);
    target = fullfile(failureDir,sprintf('%s_run%02d_%s_%s%s', ...
        name,Task.run,reason,timestampToken(),extension));
    [moved,message] = movefile(file,target);
    if ~moved
        error('PairGuide:ValidationQuarantineFailed','%s',message);
    end
end

function atomicMove(source,target,identifier)
    [moved,message] = movefile(source,target);
    if ~moved
        error(identifier,'%s',message);
    end
end

function atomicReplace(source,target,identifier)
    [moved,message] = movefile(source,target,'f');
    if ~moved
        error(identifier,'%s',message);
    end
end

function ensureFolder(folder)
    if ~isfolder(folder)
        mkdir(folder);
    end
end

function removeMarker(file)
    if isfile(file)
        delete(file);
    end
end

function stamp = timestampToken()
    stamp = char(datetime('now','Format','yyyyMMdd_HHmmss_SSS'));
end

function setProcessThreadLimits()
    names = {'OMP_NUM_THREADS','OPENBLAS_NUM_THREADS', ...
        'MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'};
    for i = 1 : numel(names)
        setenv(names{i},'1');
    end
end

function [pool,ownsPool] = exactProcessPool(nWorker)
    pool = gcp('nocreate');
    ownsPool = isempty(pool);
    if ownsPool
        pool = parpool("Processes",nWorker);
    elseif pool.NumWorkers ~= nWorker || contains(class(pool),'ThreadPool')
        error('PairGuide:ValidationParallelPool', ...
            'Existing pool must contain exactly %d process workers.',nWorker);
    end
end

function closeOwnedPool(pool,ownsPool)
    if ownsPool && ~isempty(pool) && isvalid(pool)
        delete(pool);
    end
end
