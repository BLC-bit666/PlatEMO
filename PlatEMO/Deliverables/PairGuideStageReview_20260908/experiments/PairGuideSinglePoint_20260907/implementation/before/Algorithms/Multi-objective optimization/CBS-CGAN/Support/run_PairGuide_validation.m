function State = run_PairGuide_validation(rootPath,nWorker,Options)
%RUN_PAIRGUIDE_VALIDATION Same-seed CGAN, fallback-only and pair-only controls.
%   Defaults: five problems, five seeds, N=100, 100000 FE, default D.
%   Results use a NEW directory. Frozen PairGuideBall results remain evidence.
    if nargin < 1 || isempty(rootPath); rootPath = fileparts(which('platemo')); end
    if nargin < 2 || isempty(nWorker); nWorker = 10; end
    if nargin < 3; Options = struct(); end
    addCBSPaths(rootPath);
    Defaults = struct('problems',["LIRCMOP5_BC","LIRCMOP7_BC","LIRCMOP8_BC", ...
        "LIRCMOP10_BC","LIRCMOP12_BC"],'seeds',1:5,'N',100,'maxFE',100000, ...
        'modes',["cgan","fallback_only","pair_only"], ...
        'checkpointFE',[10000 30000 50000 70000 100000], ...
        'distributionLayout',"training_pair", ...
        'outputDir',string(fullfile(rootPath,'Data','PairGuideNative')));
    for field = string(fieldnames(Defaults))'
        if ~isfield(Options,field); Options.(field) = Defaults.(field); end
    end
    assert(all(ismember(Options.modes,Defaults.modes)) && ...
        nWorker >= 1 && nWorker == fix(nWorker));
    assert(Options.maxFE >= 2*Options.N && Options.N >= 4);
    assert(isnumeric(Options.checkpointFE) && all(isfinite(Options.checkpointFE)) && ...
        all(Options.checkpointFE>0) && ismember(Options.distributionLayout,["training_pair","legacy_three"]));
    Options.checkpointFE = unique(reshape(Options.checkpointFE,1,[]));
    sources = ["PairGuide","PairGuideCore","PairBoundaryArchive_RC","PairBoundaryWGAN_RC", ...
        "PairGuideCost_RC","run_PairGuide_validation", ...
        "render_PairGuide_interval_distribution","analyze_PairGuide_interval_validation", ...
        "report_PairGuide_interval_validation","PairGuide_checkpoint_metrics", ...
        "draw_CBS_CGAN_objective_region", ...
        Options.problems];
    text = "";
    for name = sources; text = text+string(fileread(which(name))); end
    digest = java.security.MessageDigest.getInstance('SHA-256');
    digest.update(unicode2native(char(text),'UTF-8'));
    sourceHash = lower(string(reshape(dec2hex(typecast(digest.digest(),'uint8'),2)',1,[])));
    Tasks = struct('problem',{},'seed',{},'mode',{},'outputFile',{});
    folder = char(Options.outputDir);
    if ~isfolder(folder); mkdir(folder); end
    for seed = Options.seeds
        for problem = Options.problems
            for mode = Options.modes
                file = fullfile(folder,sprintf('%s_seed%02d_%s.mat',problem,seed,mode));
                Tasks(end+1) = struct('problem',problem,'seed',seed,'mode',mode, ...
                    'outputFile',string(file)); %#ok<AGROW>
            end
        end
    end
    Protocol = struct('schema',"PairGuide-native-v2",'options',Options, ...
        'sourceHash',sourceHash,'createdAt',string(datetime('now')), ...
        'workers',nWorker,'threadsPerWorker',1,'matlabVersion',string(version), ...
        'distributionMode',"cgan",'distributionSeed',1, ...
        'algorithmDefaults',PairGuide.mainlineDefaults(), ...
        'distributionStages',"first true use and specified checkpointFE; offline oracle bill");
    manifest = fullfile(folder,'protocol.mat');
    if isfile(manifest)
        old = load(manifest,'Protocol');
        assert(isequal(old.Protocol.options,Options) && ...
            old.Protocol.sourceHash == sourceHash, ...
            'Existing results use another protocol/source. Choose a fresh outputDir.');
    else
        save(manifest,'Protocol');
    end
    pool = [];
    owned = false;
    if nWorker > 1
        pool = gcp('nocreate');
        if isempty(pool); pool = parpool('Processes',nWorker); owned = true; end
        assert(pool.NumWorkers == nWorker);
    end
    cleanup = onCleanup(@()closePool(pool,owned));
    State = struct('status',"running",'tasks',numel(Tasks), ...
        'outputDir',string(folder),'sourceHash',sourceHash);
    save(fullfile(folder,'state.mat'),'State');
    errors = strings(numel(Tasks),1);
    if nWorker == 1
        for k=1:numel(Tasks); errors(k)=runTaskSafely(Tasks(k),Options,sourceHash); end
    else
        parallelOptions=parforOptions(pool,'RangePartitionMethod','fixed','SubrangeSize',1);
        parfor (k=1:numel(Tasks),parallelOptions)
            errors(k)=runTaskSafely(Tasks(k),Options,sourceHash);
        end
    end
    State.failed = nnz(strlength(errors)>0);
    State.completed = numel(Tasks)-State.failed;
    State.errors = errors;
    State.status = "reporting";
    if State.failed > 0; State.status="failed"; end
    save(fullfile(folder,'state.mat'),'State');
    clear cleanup;
    if State.failed == 0
        report_PairGuide_interval_validation(folder);
        State.status = "complete";
        save(fullfile(folder,'state.mat'),'State');
    else
        error('PairGuide:CampaignFailures','%d tasks failed; see state.mat and task logs.',State.failed);
    end
end

function message = runTaskSafely(Task,O,sourceHash)
    message = "";
    try
        maxNumCompThreads(1);
        fprintf('START %s seed=%d mode=%s\n',Task.problem,Task.seed,Task.mode);
        runTask(Task,O,sourceHash);
        fprintf('DONE %s seed=%d mode=%s\n',Task.problem,Task.seed,Task.mode);
    catch err
        message = string(getReport(err,'extended','hyperlinks','off'));
        fid=fopen([char(Task.outputFile),'.failure.txt'],'w');
        if fid>=0; fprintf(fid,'%s',message); fclose(fid); end
        fprintf('FAILED %s seed=%d mode=%s: %s\n',Task.problem,Task.seed,Task.mode,err.message);
    end
end

function runTask(Task,O,sourceHash)
    file = char(Task.outputFile);
    if isfile(file)
        old = load(file,'Record');
        assert(old.Record.sourceHash == sourceHash && old.Record.finalFE == O.maxFE);
        if Task.mode == "cgan" && Task.seed == 1
            render_PairGuide_interval_distribution(file);
        end
        return;
    end
    rng(Task.seed,'twister');
    constructor = str2func(char(Task.problem));
    Problem = constructor('N',O.N,'maxFE',O.maxFE,'maxRuntime',Inf); % D omitted.
    lastProgress = -Inf;
    lastCheckpoint = -Inf; firstCheckpoint = false;
    Algorithm = PairGuide('save',1,'run',Task.seed,'outputFcn',@progress);
    Algorithm.configureComparison(Task.mode);
    Algorithm.configureObjectiveSpaceSnapshots(struct('enabled',Task.mode == "cgan" && Task.seed == 1, ...
        'targetFE',O.checkpointFE(O.checkpointFE<=O.maxFE),'expectedRawCount',500, ...
        'expectedGuidedCount',round(0.2*O.N)));
    timer = tic;
    Algorithm.Solve(Problem);
    wallSeconds = toc(timer);
    Audit = Algorithm.guideExperimentSnapshot();
    assert(Problem.FE == O.maxFE && Audit.ObjFE == 0);
    assert(Audit.evidence.oracleCalls.instrumented && ...
        Audit.evidence.oracleCalls.CalObjRows == 2*O.maxFE && ...
        Audit.evidence.oracleCalls.CalConRows == O.maxFE);
    if Task.mode == "fallback_only"
        assert(isempty(Audit.evidence.queries) && isempty(Audit.evidence.training));
    elseif Task.mode == "pair_only"
        assert(isempty(Audit.evidence.training) && Audit.evidence.networkEndpointRows == 0);
    end
    Snapshots = Algorithm.objectiveSpaceSnapshots();
    Trajectory = metricTrajectory(Problem,Audit.evidence);
    FinalPopulation = Audit.evidence.population{end};
    Record = struct('schema',"PairGuide-native-v2",'algorithm',"PairGuide", ...
        'mode',Task.mode,'problem',Task.problem,'seed',Task.seed, ...
        'N',O.N,'D',Problem.D,'defaultDimension',true,'finalFE',Problem.FE, ...
        'sourceHash',sourceHash,'wallSeconds',wallSeconds, ...
        'checkpointFE',O.checkpointFE,'distributionLayout',O.distributionLayout);
    Checkpoints = PairGuide_checkpoint_metrics(Problem,Audit.evidence,O.checkpointFE,Record);
    writetable(Checkpoints,[file,'.checkpoints.csv']);
    save(file,'Record','Audit','Trajectory','Snapshots','FinalPopulation','Checkpoints','-v7.3');
    clear Algorithm Audit Trajectory Snapshots FinalPopulation;
    if Task.mode == "cgan" && Task.seed == 1
        render_PairGuide_interval_distribution(file);
    end

    function progress(CurrentAlgorithm,CurrentProblem)
        Current = CurrentAlgorithm.guideExperimentSnapshot(); E = Current.evidence;
        reached = nnz(O.checkpointFE<=CurrentProblem.FE);
        firstUsed = any(cellfun(@(g)g.use.selected>0,E.generations));
        if reached>lastCheckpoint || (firstUsed && ~firstCheckpoint)
            checkpoint = PairGuide_checkpoint_metrics(CurrentProblem,E,O.checkpointFE,Task);
            writetable(checkpoint,[file,'.checkpoints.csv']);
            lastCheckpoint = reached; firstCheckpoint = firstUsed;
        end
        block=floor(CurrentProblem.FE/10000);
        if block>lastProgress || CurrentProblem.FE>=CurrentProblem.maxFE
            lastProgress=block;
            fid=fopen([file,'.progress.txt'],'w');
            if fid>=0
                fprintf(fid,'%s seed=%d mode=%s FE=%d/%d\n', ...
                    Task.problem,Task.seed,Task.mode,CurrentProblem.FE,CurrentProblem.maxFE);
                fclose(fid);
            end
            fprintf('PROGRESS %s seed=%d mode=%s FE=%d/%d\n', ...
                Task.problem,Task.seed,Task.mode,CurrentProblem.FE,CurrentProblem.maxFE);
        end
    end
end

function T = metricTrajectory(Problem,E)
%METRICTRAJECTORY Use only the latest completed state at or before each target.
    targets = unique([1000:1000:Problem.maxFE,Problem.maxFE]);
    times = cellfun(@(p)p.observationFE,E.population);
    T = table(targets',nan(numel(targets),1),nan(numel(targets),1), ...
        nan(numel(targets),1),'VariableNames',{'targetFE','actualFE','IGD','HV'});
    saved = rng; cleanup = onCleanup(@()rng(saved));
    rng(314159,'twister');
    for k=1:numel(targets)
        row = find(times <= targets(k),1,'last');
        if isempty(row); continue; end
        P = E.population{row};
        Population = SOLUTION(P.p1Decs,P.p1Objs,P.p1Cons);
        T.actualFE(k) = times(row);
        T.IGD(k) = Problem.CalMetric('IGD',Population);
        T.HV(k) = Problem.CalMetric('HV',Population);
    end
    clear cleanup;
end

function closePool(pool,owned)
    if owned && ~isempty(pool) && isvalid(pool); delete(pool); end
end
