function State = run_CBS_CGAN_cutoff_diagnostics(rootPath,nWorker,phase)
%RUN_CBS_CGAN_CUTOFF_DIAGNOSTICS Run resumable exact-cutoff experiments.

    if nargin < 1 || isempty(rootPath)
        rootPath = fileparts(which('platemo'));
    end
    if nargin < 2 || isempty(nWorker)
        nWorker = 10;
    end
    if nargin < 3 || isempty(phase)
        phase = "phase1";
    end
    phase = lower(string(phase));
    Protocol = CBS_CGAN_cutoff_diagnostic_protocol(rootPath,nWorker);
    validateProtocol(Protocol,phase);
    rootPath = Protocol.rootPath;
    addCBSPaths(rootPath);
    campaignDir = fullfile(rootPath,'Data',Protocol.campaignName);
    failureDir = fullfile(campaignDir,'failures');
    analysisDir = fullfile(campaignDir,'analysis');
    mkdir(campaignDir);
    mkdir(failureDir);
    mkdir(analysisDir);
    freezeAndVerifySource(rootPath,campaignDir);
    Tasks = buildTasks(Protocol,phase,campaignDir);
    save(fullfile(campaignDir,"campaign_manifest_"+phase+".mat"), ...
        'Protocol','Tasks','-v7.3');

    State = struct('schemaVersion',Protocol.schemaVersion, ...
        'phase',phase,'status',"running", ...
        'startedAt',string(datetime('now')),'finishedAt',"", ...
        'totalTasks',numel(Tasks),'completeTasks',0, ...
        'remainingTasks',numel(Tasks),'attempt',0,'error',"");
    State = updateState(State,Tasks,Protocol,campaignDir);
    saveState(State,Protocol,campaignDir);
    if State.remainingTasks == 0
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
        error('CBSRegionGAN:IncompatibleParallelPool', ...
            ['An existing pool is incompatible. Close it explicitly or ', ...
             'rerun with its process-worker count.']);
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
            fprintf('CBS cutoff %s: %d tasks, attempt %d/%d\n', ...
                phase,numel(pending),attempt,Protocol.maxAttempts);
            options = parforOptions(pool,'RangePartitionMethod','fixed', ...
                'SubrangeSize',1);
            parfor (t = 1:numel(pending),options)
                Task = pending(t);
                try
                    run_CBS_CGAN_cutoff_diagnostic_task( ...
                        rootPath,Task,Protocol);
                catch err
                    writeFailure(failureDir,Task,attempt,err);
                end
            end
            pending = pending(~tasksComplete(pending,Protocol));
            State = updateState(State,Tasks,Protocol,campaignDir);
            saveState(State,Protocol,campaignDir);
        end
        if ~isempty(pending)
            error('CBSRegionGAN:CutoffTasksFailed', ...
                '%d cutoff tasks failed after %d attempts.', ...
                numel(pending),Protocol.maxAttempts);
        end
        State.status = "complete";
        State.finishedAt = string(datetime('now'));
        State = updateState(State,Tasks,Protocol,campaignDir);
        saveState(State,Protocol,campaignDir);
        writelines("COMPLETE "+phase, ...
            fullfile(campaignDir,"COMPLETE_"+phase+".txt"));
        analyze_CBS_CGAN_cutoff_diagnostics( ...
            rootPath,Protocol.campaignName,phase);
    catch err
        State.status = "failed";
        State.finishedAt = string(datetime('now'));
        State.error = string(getReport(err,'extended','hyperlinks','off'));
        State = updateState(State,Tasks,Protocol,campaignDir);
        saveState(State,Protocol,campaignDir);
        writelines(State.error, ...
            fullfile(campaignDir,"FAILED_"+phase+".txt"));
        rethrow(err);
    end
    clear cleanup;
    closeOwnedPool(pool,ownsPool);
end

function Tasks = buildTasks(Protocol,phase,campaignDir)
    if phase == "phase1"
        Arms = Protocol.phase1Arms;
    else
        Arms = Protocol.phase2Arms;
    end
    template = struct('phase',phase,'arm',"",'className',"", ...
        'parameters',{{}},'problem',"",'seed',0,'outputFile',"");
    Tasks = repmat(template,numel(Arms)*numel(Protocol.problems)* ...
        numel(Protocol.seeds),1);
    next = 0;
    for a = 1 : numel(Arms)
        for problem = Protocol.problems
            for seed = Protocol.seeds
                next = next+1;
                Tasks(next).phase = phase;
                Tasks(next).arm = Arms(a).label;
                Tasks(next).className = Arms(a).className;
                Tasks(next).parameters = Arms(a).parameters;
                Tasks(next).problem = problem;
                Tasks(next).seed = seed;
                name = sprintf('%s_%s_seed%02d.mat', ...
                    Arms(a).label,problem,seed);
                Tasks(next).outputFile = string(fullfile(campaignDir, ...
                    char(phase),'runs',char(Arms(a).label),name));
            end
        end
    end
end

function mask = tasksComplete(Tasks,Protocol)
    mask = false(numel(Tasks),1);
    for i = 1 : numel(Tasks)
        mask(i) = resultComplete(Tasks(i),Protocol);
    end
end

function complete = resultComplete(Task,Protocol)
    complete = false;
    file = char(Task.outputFile);
    if exist(file,'file') ~= 2
        return;
    end
    try
        Data = load(file,'Record','Audit','CutoffPopulation');
        if ~all(isfield(Data,{'Record','Audit','CutoffPopulation'}))
            return;
        end
        R = Data.Record;
        A = Data.Audit;
        P = Data.CutoffPopulation;
        complete = string(R.schemaVersion) == Protocol.schemaVersion && ...
            string(R.status) == "ok" && ...
            string(R.phase) == string(Task.phase) && ...
            string(R.arm) == string(Task.arm) && ...
            string(R.className) == string(Task.className) && ...
            string(R.problem) == string(Task.problem) && ...
            double(R.seed) == double(Task.seed) && ...
            double(R.N) == Protocol.popSize && ...
            double(R.D) == Protocol.dimension && ...
            double(R.maxFE) == Protocol.maxFE && ...
            all(isfield(A,Protocol.requiredAuditFields)) && ...
            double(A.cganEndFE) >= 0.5*Protocol.maxFE && ...
            double(A.cganEndFE) < Protocol.maxFE && ...
            size(P.decs,1) == Protocol.popSize && ...
            size(P.decs,2) == Protocol.dimension && ...
            isequal(double(P.decs),double(A.cganEndDecs)) && ...
            isequal(double(P.objs),double(A.cganEndObjs)) && ...
            isequal(double(P.cons),double(A.cganEndCons)) && ...
            A.rawOracleCount == 0 && A.selectedTargetCount == 0;
    catch
        complete = false;
    end
end

function State = updateState(State,Tasks,Protocol,campaignDir)
    complete = tasksComplete(Tasks,Protocol);
    State.completeTasks = sum(complete);
    State.remainingTasks = numel(Tasks)-State.completeTasks;
    lines = [ ...
        "schemaVersion="+State.schemaVersion; ...
        "phase="+State.phase; ...
        "status="+State.status; ...
        "completeTasks="+State.completeTasks; ...
        "remainingTasks="+State.remainingTasks; ...
        "attempt="+State.attempt];
    writelines(lines,fullfile(campaignDir,"campaign_state_"+ ...
        State.phase+".txt"));
end

function saveState(State,Protocol,campaignDir)
    file = fullfile(campaignDir,"campaign_state_"+State.phase+".mat");
    save(file,'State','Protocol');
end

function validateProtocol(Protocol,phase)
    if ~ismember(phase,["phase1","phase2"]) || ...
            Protocol.nWorker < 1 || Protocol.nWorker > 10 || ...
            Protocol.popSize ~= 100 || Protocol.dimension ~= 30 || ...
            Protocol.maxFE ~= 200000 || ...
            ~isequal(Protocol.seeds,1:5) || ...
            numel(Protocol.problems) ~= 5 || ...
            numel(Protocol.phase1Arms) ~= 7 || ...
            numel(Protocol.phase2Arms) ~= 4
        error('CBSRegionGAN:BadCutoffProtocol', ...
            'The frozen exact-cutoff protocol was changed.');
    end
    for name = [Protocol.problems, ...
            [Protocol.phase1Arms.className], ...
            [Protocol.phase2Arms.className]]
        if isempty(which(char(name)))
            error('CBSRegionGAN:MissingCampaignClass', ...
                'Cannot resolve %s.',name);
        end
    end
end

function freezeAndVerifySource(rootPath,campaignDir)
    manifestFile = fullfile(campaignDir,'source_manifest.mat');
    current = sourceManifest(rootPath);
    if exist(manifestFile,'file') == 2
        saved = load(manifestFile,'SourceManifest');
        if ~isfield(saved,'SourceManifest') || ...
                ~isequal(saved.SourceManifest,current)
            error('CBSRegionGAN:CampaignSourceChanged', ...
                ['CBS-CGAN source changed after this campaign began. ', ...
                 'Use a new campaign directory.']);
        end
        return;
    end
    SourceManifest = current;
    snapshot = fullfile(campaignDir,'source_snapshot');
    mkdir(snapshot);
    source = fullfile(rootPath,'Algorithms','Multi-objective optimization', ...
        'CBS-CGAN');
    copyfile(source,fullfile(snapshot,'CBS-CGAN'));
    save(manifestFile,'SourceManifest');
end

function Manifest = sourceManifest(rootPath)
    source = fullfile(rootPath,'Algorithms','Multi-objective optimization', ...
        'CBS-CGAN');
    files = [dir(fullfile(source,'**','*.m')); ...
        dir(fullfile(source,'**','*.md'))];
    fullNames = string(fullfile({files.folder},{files.name}))';
    relative = erase(fullNames,string(source)+filesep);
    [relative,order] = sort(relative);
    bytes = reshape(double([files(order).bytes]),[],1);
    modified = reshape(double([files(order).datenum]),[],1);
    Manifest = table(relative,bytes,modified, ...
        'VariableNames',{'path','bytes','datenum'});
end

function writeFailure(failureDir,Task,attempt,err)
    name = sprintf('%s_%s_seed%02d_attempt%d.txt', ...
        Task.arm,Task.problem,Task.seed,attempt);
    report = string(getReport(err,'extended','hyperlinks','off'));
    writelines(report,fullfile(failureDir,name));
end

function closeOwnedPool(pool,ownsPool)
    if ~ownsPool
        return;
    end
    try
        if ~isempty(pool) && isvalid(pool)
            delete(pool);
        end
    catch
    end
end
