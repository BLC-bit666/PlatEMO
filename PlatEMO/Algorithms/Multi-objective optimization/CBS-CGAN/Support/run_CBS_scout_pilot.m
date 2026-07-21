function [Summary,outDir] = run_CBS_scout_pilot(outDir,workerCount,Options)
%RUN_CBS_SCOUT_PILOT CGAN scout-mode pilot with checkpoints and metrics.
%   Five arms on ten representative problems, three seeds, maxFE=200k:
%     SC    scoutMode=on         generator=wgan       (candidate mainline)
%     CNSC  scoutMode=on         generator=copynoise  (learning control)
%     SCNP  scoutMode=noprotect  generator=wgan       (queries only)
%     SCNF  scoutMode=nofrontier generator=wgan       (protection only)
%     A0    nGen=0                                    (no CGAN at all)
%   Every task records IGD at {25%,32.5%,40%,50%}*maxFE plus the final
%   value, and the scout study counters (metricsMode=on) are stored in the
%   task file. Cheap arms should be launched before the WGAN-bearing arms
%   so wiring assertions fail within minutes, not hours.

    rootDir = fileparts(which('platemo'));
    if isempty(rootDir); rootDir = pwd; end
    addpath(genpath(rootDir));
    if nargin < 1 || isempty(outDir)
        outDir = fullfile(rootDir,'Data','CBS_RegionGAN_compare', ...
            'scout_pilot_v1');
    end
    if nargin < 2 || isempty(workerCount); workerCount = 10; end
    if nargin < 3 || isempty(Options); Options = struct(); end
    allowed = {'resume','problems','seeds','maxFE','N','arms'};
    unexpected = setdiff(fieldnames(Options),allowed);
    if ~isempty(unexpected)
        error('CBSRegionGAN:BadScoutPilotOptions', ...
            'Unsupported option(s): %s.',strjoin(unexpected,', '));
    end
    Options = defaultField(Options,'resume',true);
    Options = defaultField(Options,'problems',[ ...
        "LIRCMOP5_BC";"LIRCMOP6_BC";"LIRCMOP9_BC";"LIRCMOP10_BC"; ...
        "LIRCMOP11_BC";"LIRCMOP13_BC"; ...
        "DASCMOP1_BC";"DASCMOP5_BC";"DASCMOP6_BC";"DASCMOP9_BC"]);
    Options = defaultField(Options,'seeds',1:3);
    Options = defaultField(Options,'maxFE',200000);
    Options = defaultField(Options,'N',100);
    Options = defaultField(Options,'arms',["A0","CNSC","SC","SCNP","SCNF"]);

    workerCount = max(1,round(double(workerCount)));
    if ~ismember(workerCount,[1,10,12])
        error('CBSRegionGAN:ScoutPilotWorkerCount', ...
            'Use 1 worker for tests, or 10/12 workers formally.');
    end
    arms = string(Options.arms(:));
    scout = strings(size(arms));
    gen = strings(size(arms));
    nGenValue = zeros(size(arms));
    sigmaValue = zeros(size(arms));
    guide = strings(size(arms));
    guideShareValue = zeros(size(arms));
    guideWindowValue = strings(size(arms));
    guideCarveValue = strings(size(arms));
    blsWindowValue = strings(size(arms));
    blsFeedValue = strings(size(arms));
    for i = 1 : numel(arms)
        scout(i) = "on"; gen(i) = "wgan"; nGenValue(i) = 20;
        sigmaValue(i) = 0.3; guide(i) = "off";
        guideShareValue(i) = 0.3; guideWindowValue(i) = "half";
        guideCarveValue(i) = "sym";
        blsWindowValue(i) = "late"; blsFeedValue(i) = "off";
        switch arms(i)
            case "SC"    % candidate mainline: full scout mode
            case "CNSC"; gen(i) = "copynoise";
            case "SCNP"; scout(i) = "noprotect";
            case "SCNF"; scout(i) = "nofrontier";
            case "JX"    % learned landing-zone jump, 17:3 origins
                scout(i) = "nofrontier"; gen(i) = "jump";
                sigmaValue(i) = 1.0;
            case "JX50"  % learned jump, half frontier-origin dose
                scout(i) = "halffrontier"; gen(i) = "jump";
                sigmaValue(i) = 1.0;
            case "TJ"    % trivial jump control: sample landing rows
                scout(i) = "nofrontier"; gen(i) = "jumptrivial";
            case "GD"    % unevaluated guides steer 30% of P1 offspring
                scout(i) = "off"; guide(i) = "on";
            case "MIX"   % ratio control: same GA/DE split, no guides
                scout(i) = "off"; guide(i) = "mix"; nGenValue(i) = 0;
            case "CN"    % learning control: copynoise guides, same wiring
                scout(i) = "off"; guide(i) = "on"; gen(i) = "copynoise";
            case "GD20"  % dose arm: 40/40/20
                scout(i) = "off"; guide(i) = "on";
                guideShareValue(i) = 0.2;
            case "GD40"  % dose arm: 30/30/40
                scout(i) = "off"; guide(i) = "on";
                guideShareValue(i) = 0.4;
            case "GDA20" % GA-preserving carve: 50 GA + 30 DE + 20 guided
                scout(i) = "off"; guide(i) = "on";
                guideShareValue(i) = 0.2; guideCarveValue(i) = "de";
            case "BFF"   % module fusion: GD20 + full-run BLS feeding BMem
                scout(i) = "off"; guide(i) = "on";
                guideShareValue(i) = 0.2;
                blsWindowValue(i) = "full"; blsFeedValue(i) = "on";
            case "GDW"   % winner config + guides supplied to the end
                scout(i) = "off"; guide(i) = "on";
                guideShareValue(i) = 0.2; guideCarveValue(i) = "de";
                guideWindowValue(i) = "full";
            case "A0";   scout(i) = "off"; nGenValue(i) = 0;
            otherwise
                error('CBSRegionGAN:BadScoutPilotArm', ...
                    ['Arms must be from SC, CNSC, SCNP, SCNF, ' ...
                    'JX, JX50, TJ, GD, MIX, CN, GD20, GD40, GDA20, GDW, BFF, and A0.']);
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
    [~,order] = sort(Tasks.arm_index);
    Tasks = Tasks(order,:);
    rows = repmat(emptyRunRow(),height(Tasks),1);

    if workerCount == 1
        for task = 1 : height(Tasks)
            a = Tasks.arm_index(task);
            rows(task) = runTask(arms(a),scout(a),gen(a),nGenValue(a), ...
                sigmaValue(a),guide(a),guideShareValue(a), ...
                guideWindowValue(a),guideCarveValue(a), ...
                blsWindowValue(a),blsFeedValue(a), ...
                Tasks.problem(task),Tasks.seed(task), ...
                outDir,N,maxFE,Options.resume);
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
            a = taskArm(task);
            rows(task) = runTask(arms(a),scout(a),gen(a),nGenValue(a), ...
                sigmaValue(a),guide(a),guideShareValue(a), ...
                guideWindowValue(a),guideCarveValue(a), ...
                blsWindowValue(a),blsFeedValue(a), ...
                taskProblem(task),taskSeed(task), ...
                outDir,N,maxFE,Options.resume);
            send(queue,rows(task));
        end
    end

    Summary = struct2table(rows);
    writetable(Summary,fullfile(outDir,'scout_pilot_summary.csv'));
    failed = Summary.status ~= "ok";
    if any(failed)
        first = find(failed,1);
        error('CBSRegionGAN:ScoutPilotTasksFailed', ...
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

function Row = runTask(armId,scoutState,genMode,nGenValue, ...
        sigmaValue,guideState,guideShareValue,guideWindowValue, ...
        guideCarveValue,blsWindowValue,blsFeedValue, ...
        problemName,seedValue,outDir,N,maxFE,resume)
%RUNTASK Execute or reuse one arm/problem/seed task.

    Row = emptyRunRow();
    Row.arm = string(armId);
    Row.scout = string(scoutState);
    Row.gen = string(genMode);
    Row.nGen = double(nGenValue);
    Row.sigma = double(sigmaValue);
    Row.guide = string(guideState);
    Row.guide_share = double(guideShareValue);
    Row.guide_window = string(guideWindowValue);
    Row.guide_carve = string(guideCarveValue);
    Row.bls_window = string(blsWindowValue);
    Row.bls_feed = string(blsFeedValue);
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
            if ~isfield(Candidate,'guide_carve')
                Candidate.guide_carve = "sym";
            end
            if ~isfield(Candidate,'bls_window')
                Candidate.bls_window = "late";
            end
            if ~isfield(Candidate,'bls_feed')
                Candidate.bls_feed = "off";
            end
            if string(Candidate.status) == "ok" && ...
                    string(Candidate.scout) == Row.scout && ...
                    string(Candidate.gen) == Row.gen && ...
                    double(Candidate.nGen) == Row.nGen && ...
                    double(Candidate.sigma) == Row.sigma && ...
                    string(Candidate.guide) == Row.guide && ...
                    double(Candidate.guide_share) == Row.guide_share && ...
                    string(Candidate.guide_window) == ...
                        Row.guide_window && ...
                    string(Candidate.guide_carve) == ...
                        Row.guide_carve && ...
                    string(Candidate.bls_window) == Row.bls_window && ...
                    string(Candidate.bls_feed) == Row.bls_feed && ...
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
        Defaults = CBS_RegionWGAN_GP.mainlineDefaults();
        args = {'save',40,'outputFcn',@quietOutput, ...
            'operatorMode','ga_de_half','boundarySearch','on', ...
            'generatorMode',char(genMode), ...
            'scoutMode',char(scoutState),'metricsMode','on', ...
            'guideMode',char(guideState), ...
            'guideShare',guideShareValue, ...
            'guideWindow',char(guideWindowValue), ...
            'guideCarve',char(guideCarveValue), ...
            'blsWindow',char(blsWindowValue), ...
            'blsFeed',char(blsFeedValue), ...
            'parameter',{nGenValue,Defaults.zDim,Defaults.ganIter, ...
            Defaults.ganMiniBatch,Defaults.nCritic, ...
            Defaults.minGANTrainCount,sigmaValue}};
        Algorithm = CBS_RegionWGAN_GP(args{:});
        if Algorithm.effectiveOperatorMode() ~= "ga_de_half" || ...
                Algorithm.effectiveBoundarySearch() ~= "on" || ...
                Algorithm.effectiveGeneratorMode() ~= Row.gen || ...
                Algorithm.effectiveScoutMode() ~= Row.scout || ...
                Algorithm.effectiveGuideMode() ~= Row.guide || ...
                Algorithm.effectiveGuideShare() ~= Row.guide_share || ...
                Algorithm.effectiveGuideWindow() ~= Row.guide_window || ...
                Algorithm.effectiveGuideCarve() ~= Row.guide_carve || ...
                Algorithm.effectiveBlsWindow() ~= Row.bls_window || ...
                Algorithm.effectiveBlsFeed() ~= Row.bls_feed || ...
                Algorithm.effectiveMetricsMode() ~= "on"
            error('CBSRegionGAN:ScoutSwitchNotApplied', ...
                'Arm %s switches were not applied.',char(armId));
        end
        Algorithm.Solve(Problem);
        Row.finalFE = double(Problem.FE);
        Row.D = double(Problem.D);
        Row.M = double(Problem.M);
        Row.wall_seconds = toc(wallTimer);
        if Row.finalFE ~= Row.maxFE
            error('CBSRegionGAN:IncompleteScoutPilotRun', ...
                'Expected finalFE=%d, got %d.',Row.maxFE,Row.finalFE);
        end

        Row.IGD = double(Problem.CalMetric('IGD',Algorithm.result{end,2}));
        checkpoints = [0.25,0.325,0.4,0.5]*Row.maxFE;
        resultFE = cellfun(@double,Algorithm.result(:,1));
        cpFE = nan(1,numel(checkpoints));
        cpIGD = nan(1,numel(checkpoints));
        for c = 1 : numel(checkpoints)
            [~,idx] = min(abs(resultFE-checkpoints(c)));
            cpFE(c) = resultFE(idx);
            cpIGD(c) = double(Problem.CalMetric('IGD', ...
                Algorithm.result{idx,2}));
        end
        Row.IGD_50k = cpIGD(1);  Row.FE_50k = cpFE(1);
        Row.IGD_65k = cpIGD(2);  Row.FE_65k = cpFE(2);
        Row.IGD_80k = cpIGD(3);  Row.FE_80k = cpFE(3);
        Row.IGD_100k = cpIGD(4); Row.FE_100k = cpFE(4);

        Metrics = Algorithm.collectedScoutMetrics();
        Row = flattenMetrics(Row,Metrics,checkpoints);
        assertArmWiring(Row,Metrics);
        Row.status = "ok";
        TaskRow = Row;
        save(Row.task_file,'TaskRow','Metrics');
        return;
    catch Error
        Row.wall_seconds = toc(wallTimer);
        Row.status = "failed";
        Row.error_identifier = string(Error.identifier);
        Row.error_message = string(Error.message);
    end
    TaskRow = Row; %#ok<NASGU>
    save(Row.task_file,'TaskRow');
end

function Row = flattenMetrics(Row,Mx,checkpoints)
%FLATTENMETRICS Copy headline window-2 counters into summary columns.

    if isempty(Mx)
        return;
    end
    Row.gan_events = sum(Mx.ganEvents);
    Row.front_gen2 = Mx.ganFrontGen(2);
    Row.front_feas2 = Mx.ganFrontFeas(2);
    Row.front_p12 = Mx.ganFrontP1(2);
    Row.front_empty2 = Mx.ganFrontEmpty(2);
    Row.pop_gen2 = Mx.ganPopGen(2);
    Row.pop_feas2 = Mx.ganPopFeas(2);
    Row.pop_p12 = Mx.ganPopP1(2);
    if Mx.ctrlTot(2) > 0
        Row.ctrl_rate2 = Mx.ctrlHit(2)/Mx.ctrlTot(2);
    end
    Row.prot_used2 = Mx.protUsed(2);
    Row.guide_events = sum(Mx.guideEvents);
    Row.gd_gen2 = Mx.gdGen(2);
    Row.gd_feas2 = Mx.gdFeas(2);
    Row.gd_p12 = Mx.gdP1(2);
    Row.gd_empty2 = Mx.gdEmpty(2);
    Row.gdF_p1_1 = sum(Mx.gdFP1(:,1));
    Row.gdF_p1_2 = sum(Mx.gdFP1(:,2));
    Row.gdF_p1_3 = sum(Mx.gdFP1(:,3));
    Row.gdF_gen_1 = sum(Mx.gdFGen(:,1));
    Row.gdF_gen_2 = sum(Mx.gdFGen(:,2));
    Row.gdF_gen_3 = sum(Mx.gdFGen(:,3));
    Row.sc_gen2 = Mx.scGen(2);
    Row.sc_feas2 = Mx.scFeas(2);
    Row.sc_p12 = Mx.scP1(2);
    Row.sc_empty2 = Mx.scEmpty(2);
    Row.o1_gen2 = Mx.o1Gen(2);
    Row.o1_feas2 = Mx.o1Feas(2);
    Row.o1_p12 = Mx.o1P1(2);
    if ~isempty(Mx.covFE)
        for c = [1,4]
            [~,idx] = min(abs(Mx.covFE-checkpoints(c)));
            if c == 1
                Row.cov_50k = Mx.covCount(idx);
            else
                Row.cov_100k = Mx.covCount(idx);
            end
        end
    end
end

function assertArmWiring(Row,Mx)
%ASSERTARMWIRING Fail fast when a switch silently did not take effect.

    if isempty(Mx)
        error('CBSRegionGAN:ScoutMetricsMissing', ...
            'Arm %s produced no metrics struct.',char(Row.arm));
    end
    events = sum(Mx.ganEvents);
    frontTotal = sum(Mx.ganFrontGen);
    popTotal = sum(Mx.ganPopGen);
    protTotal = sum(Mx.protUsed);
    hop2Total = sum(Mx.ganHop2Gen);
    front1 = Mx.ganFrontGen(1);
    pop1 = Mx.ganPopGen(1);
    earlyEvents = Mx.ganEvents(1) > 0;
    if earlyEvents
        scoutShareOk = front1 > pop1;
        legacyShareOk = front1 < pop1;
    else
        scoutShareOk = frontTotal > popTotal;
        legacyShareOk = frontTotal < popTotal;
    end
    % On three-objective problems the feasible front can cover every
    % reference direction early, legitimately emptying the frontier pool
    % (allocation then reflows to populated by design), so the share
    % checks are only decisive on two-objective problems.
    shareDecisive = double(Row.M) == 2;
    switch Row.arm
        case "A0"
            if events > 0 || frontTotal + popTotal > 0
                error('CBSRegionGAN:ScoutA0NotClean', ...
                    'A0 must not run any CGAN event.');
            end
        case {"SC","CNSC"}
            if events > 0 && protTotal == 0
                error('CBSRegionGAN:ScoutProtectionNotApplied', ...
                    'Arm %s expected protected injection.',char(Row.arm));
            end
            if events > 0 && shareDecisive && ~scoutShareOk
                error('CBSRegionGAN:ScoutQueriesNotApplied', ...
                    'Arm %s expected frontier-majority queries.', ...
                    char(Row.arm));
            end
        case {"JX","TJ"}
            if events > 0 && protTotal == 0
                error('CBSRegionGAN:ScoutProtectionNotApplied', ...
                    'Arm %s expected protected injection.',char(Row.arm));
            end
            if hop2Total > 0
                error('CBSRegionGAN:ScoutLegacyQueriesExpected', ...
                    'Arm %s must use the legacy one-hop allocation.', ...
                    char(Row.arm));
            end
            if events > 0 && shareDecisive && ~legacyShareOk
                error('CBSRegionGAN:ScoutLegacyQueriesExpected', ...
                    'Arm %s expected the one-sixth allocation.', ...
                    char(Row.arm));
            end
        case "JX50"
            if events > 0 && protTotal == 0
                error('CBSRegionGAN:ScoutProtectionNotApplied', ...
                    'JX50 expected protected injection.');
            end
            if hop2Total > 0
                error('CBSRegionGAN:ScoutLegacyQueriesExpected', ...
                    'JX50 must use the one-hop half allocation.');
            end
        case "MIX"
            if sum(Mx.guideEvents) > 0 || sum(Mx.gdGen) > 0 || ...
                    sum(Mx.ganEvents) > 0
                error('CBSRegionGAN:MixMustBeGuideFree', ...
                    'MIX must run without guides or CGAN events.');
            end
        case {"GD","CN","GD20","GD40","GDW"}
            if sum(Mx.guideEvents) > 0 && sum(Mx.gdGen) == 0
                error('CBSRegionGAN:GuideChildrenMissing', ...
                    'GD produced guides but no guided offspring.');
            end
            if frontTotal + popTotal > 0
                error('CBSRegionGAN:GuideMustNotEvaluate', ...
                    'GD must never evaluate raw CGAN solutions.');
            end
            if protTotal > 0
                error('CBSRegionGAN:GuideMustNotProtect', ...
                    'GD must not use protected injection.');
            end
        case "SCNP"
            if events > 0 && protTotal > 0
                error('CBSRegionGAN:ScoutProtectionUnexpected', ...
                    'SCNP must not protect scouts.');
            end
            if events > 0 && shareDecisive && ~scoutShareOk
                error('CBSRegionGAN:ScoutQueriesNotApplied', ...
                    'SCNP expected frontier-majority queries.');
            end
        case "SCNF"
            if events > 0 && protTotal == 0
                error('CBSRegionGAN:ScoutProtectionNotApplied', ...
                    'SCNF expected protected injection.');
            end
            if hop2Total > 0
                error('CBSRegionGAN:ScoutLegacyQueriesExpected', ...
                    'SCNF must use the legacy one-hop allocation.');
            end
            if events > 0 && shareDecisive && ~legacyShareOk
                error('CBSRegionGAN:ScoutLegacyQueriesExpected', ...
                    'SCNF expected the legacy one-sixth allocation.');
            end
    end
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
    if isempty(pool)
        cluster = parcluster('local');
        if cluster.NumWorkers < workerCount
            cluster.NumWorkers = workerCount;
            saveProfile(cluster);
        end
        parpool(cluster,workerCount);
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
    fprintf(['[%d/%d] %s %s run=%d status=%s IGD=%.8g IGD100k=%.8g ' ...
        'reused=%d\n'],done,total,char(Row.arm),char(Row.problem), ...
        Row.seed,char(Row.status),Row.IGD,Row.IGD_100k,Row.reused);
end

function Row = emptyRunRow()
    Row = struct( ...
        'arm',"",'scout',"off",'gen',"wgan",'nGen',NaN,'sigma',NaN, ...
        'guide',"off",'guide_share',NaN,'guide_window',"half", ...
        'guide_carve',"sym",'bls_window',"late",'bls_feed',"off", ...
        'problem',"",'seed',NaN,'N',NaN,'D',NaN,'M',NaN, ...
        'maxFE',NaN,'finalFE',NaN, ...
        'IGD',NaN,'IGD_50k',NaN,'IGD_65k',NaN,'IGD_80k',NaN, ...
        'IGD_100k',NaN,'FE_50k',NaN,'FE_65k',NaN,'FE_80k',NaN, ...
        'FE_100k',NaN, ...
        'gan_events',NaN,'front_gen2',NaN,'front_feas2',NaN, ...
        'front_p12',NaN,'front_empty2',NaN,'pop_gen2',NaN, ...
        'pop_feas2',NaN,'pop_p12',NaN,'ctrl_rate2',NaN, ...
        'prot_used2',NaN,'guide_events',NaN,'gd_gen2',NaN, ...
        'gd_feas2',NaN,'gd_p12',NaN,'gd_empty2',NaN, ...
        'gdF_p1_1',NaN,'gdF_p1_2',NaN,'gdF_p1_3',NaN, ...
        'gdF_gen_1',NaN,'gdF_gen_2',NaN,'gdF_gen_3',NaN, ...
        'sc_gen2',NaN,'sc_feas2',NaN,'sc_p12',NaN, ...
        'sc_empty2',NaN,'o1_gen2',NaN,'o1_feas2',NaN,'o1_p12',NaN, ...
        'cov_50k',NaN,'cov_100k',NaN, ...
        'wall_seconds',NaN,'status',"pending",'reused',0, ...
        'task_file',"",'error_identifier',"",'error_message',"");
end
