function State = run_CBS_CGAN_screening_campaign(rootPath,nWorker)
%RUN_CBS_CGAN_SCREENING_CAMPAIGN Run the unattended sequential campaign.
%   Each stage reuses the selected control and adds only the next candidate.
%   Independent task queues mirror root test.m: problem x run tasks, one
%   platemo call per fixed parfor subrange, and ten process workers.

    if nargin < 1 || isempty(rootPath)
        rootPath = fileparts(which('platemo'));
    end
    if nargin < 2 || isempty(nWorker)
        nWorker = 10;
    end
    Protocol = CBS_CGAN_screening_protocol(rootPath,nWorker);
    validateProtocol(Protocol);
    cd(Protocol.rootPath);
    addCBSPaths(Protocol.rootPath);
    campaignDir = fullfile(Protocol.rootPath,'Data',Protocol.campaignName);
    analysisDir = fullfile(campaignDir,'analysis');
    failureDir = fullfile(campaignDir,'failures');
    mkdir(analysisDir);
    mkdir(failureDir);
    freezeSourceOnce(Protocol.rootPath,campaignDir);
    State = initialState(Protocol);
    saveState(campaignDir,State,Protocol);

    delete(gcp('nocreate'));
    try
        cluster = parcluster('Processes');
        delete(cluster.Jobs);
    catch
    end
    pool = parpool("Processes",Protocol.nWorker);
    cleanup = onCleanup(@()deletePool(pool));
    try
        %% E0/E1: keep or drop unpaired true-feasible anchors
        base = arm("CBS_RegionWGAN_GP_E0_Base",Protocol.baseParameters);
        keep = arm("CBS_RegionWGAN_GP_E1_KeepAnchor", ...
            setParameter(Protocol.baseParameters,1,1));
        runQueue(pool,[base keep],Protocol,failureDir);
        [Summary,Problems,Runs] = analyzeStage( ...
            "E1_keep_anchor",base,keep,Protocol,analysisDir);
        current = selectWinner(base,keep,Summary);
        State = recordStage(State,"E1_keep_anchor",base,keep, ...
            current,Summary,Problems,Runs);
        saveState(campaignDir,State,Protocol);

        %% E2: K5 versus K10 versus unrestricted pairing
        k10 = arm("CBS_RegionWGAN_GP_E2_K10", ...
            setParameter(current.parameters,2,10));
        kall = arm("CBS_RegionWGAN_GP_E2_KAll", ...
            setParameter(current.parameters,2,Inf));
        runQueue(pool,[k10 kall],Protocol,failureDir);
        [Summary,Problems,Runs] = analyzeStage( ...
            "E2_pair_directions",current,[k10 kall],Protocol,analysisDir);
        current = selectWinner(current,[k10 kall],Summary);
        State = recordStage(State,"E2_pair_directions", ...
            State.current,[k10 kall],current,Summary,Problems,Runs);
        saveState(campaignDir,State,Protocol);

        %% E3: all true-feasible anchors, only when front filtering matters
        if auditTrigger(current,Protocol,"frontDropRate", ...
                Protocol.frontTrigger)
            candidate = arm("CBS_RegionWGAN_GP_E3_AllFeasible", ...
                setParameter(current.parameters,3,Inf));
            [State,current] = runBinaryStage(pool,State,current,candidate, ...
                "E3_all_feasible",Protocol,analysisDir,failureDir,campaignDir);
        else
            State = recordSkip(State,"E3_all_feasible",current, ...
                "front opportunity trigger was false");
            saveState(campaignDir,State,Protocol);
        end

        %% E4: cap 5 -> 10 -> unrestricted, stopping when cap is inactive
        if auditTrigger(current,Protocol,"capDropRate",Protocol.capTrigger)
            candidate = arm("CBS_RegionWGAN_GP_E4_Cap10", ...
                setParameter(current.parameters,4,10));
            [State,current] = runBinaryStage(pool,State,current,candidate, ...
                "E4_cap10",Protocol,analysisDir,failureDir,campaignDir);
            if current.algorithm == candidate.algorithm && ...
                    auditTrigger(current,Protocol,"capDropRate", ...
                    Protocol.capTrigger)
                candidate = arm("CBS_RegionWGAN_GP_E4_CapAll", ...
                    setParameter(current.parameters,4,Inf));
                [State,current] = runBinaryStage(pool,State,current,candidate, ...
                    "E4_cap_all",Protocol,analysisDir,failureDir,campaignDir);
            else
                State = recordSkip(State,"E4_cap_all",current, ...
                    "Cap10 was rejected or removed the cap bottleneck");
                saveState(campaignDir,State,Protocol);
            end
        else
            State = recordSkip(State,"E4_cap10",current, ...
                "anchor cap trigger was false");
            State = recordSkip(State,"E4_cap_all",current, ...
                "anchor cap trigger was false");
            saveState(campaignDir,State,Protocol);
        end

        %% E5: split class/coverage gate
        gateTriggered = auditTrigger(current,Protocol, ...
            "unsafeTrainingEventRate",Protocol.unsafeGateTrigger);
        if gateTriggered
            candidate = arm("CBS_RegionWGAN_GP_E5_SplitGate", ...
                setParameter(current.parameters,5,1));
            [State,current] = runBinaryStage(pool,State,current,candidate, ...
                "E5_split_gate",Protocol,analysisDir,failureDir,campaignDir);
            if current.algorithm == candidate.algorithm && ...
                    current.parameters(1) == 0
                candidate = arm("CBS_RegionWGAN_GP_E5b_KeepWithSplit", ...
                    setParameter(current.parameters,1,1));
                [State,current] = runBinaryStage(pool,State,current,candidate, ...
                    "E5b_keep_with_split",Protocol,analysisDir, ...
                    failureDir,campaignDir);
            else
                State = recordSkip(State,"E5b_keep_with_split",current, ...
                    "split gate was rejected or unpaired anchors were already retained");
                saveState(campaignDir,State,Protocol);
            end
        else
            State = recordSkip(State,"E5_split_gate",current, ...
                "unsafe training trigger was false");
            State = recordSkip(State,"E5b_keep_with_split",current, ...
                "split gate was not active");
            saveState(campaignDir,State,Protocol);
        end

        %% E6: balanced pairflag batches
        if auditTrigger(current,Protocol,"imbalancedTrainingEventRate", ...
                Protocol.imbalanceTrigger)
            candidate = arm("CBS_RegionWGAN_GP_E6_BalancedBatch", ...
                setParameter(current.parameters,6,1));
            [State,current] = runBinaryStage(pool,State,current,candidate, ...
                "E6_balanced_batch",Protocol,analysisDir,failureDir,campaignDir);
        else
            State = recordSkip(State,"E6_balanced_batch",current, ...
                "pairflag imbalance trigger was false");
            saveState(campaignDir,State,Protocol);
        end

        %% E7: supplement only an unusable current feasible parent set
        if auditTrigger(current,Protocol,"parentFallbackRate", ...
                Protocol.parentFallbackTrigger)
            candidate = arm("CBS_RegionWGAN_GP_E7_MemoryParent", ...
                setParameter(current.parameters,7,1));
            [State,current] = runBinaryStage(pool,State,current,candidate, ...
                "E7_memory_parent",Protocol,analysisDir,failureDir,campaignDir);
        else
            State = recordSkip(State,"E7_memory_parent",current, ...
                "feasible-parent fallback trigger was false");
            saveState(campaignDir,State,Protocol);
        end

        %% E8: old 20-slot pool versus all-W 500-query critic pool
        candidate = arm("CBS_RegionWGAN_GP_E8_GlobalCritic", ...
            setParameter(current.parameters,8,1));
        [State,current] = runBinaryStage(pool,State,current,candidate, ...
            "E8_global_critic",Protocol,analysisDir,failureDir,campaignDir);

        State.current = current;
        State.status = "complete";
        State.completedAt = string(datetime('now'));
        saveState(campaignDir,State,Protocol);
        writelines("ALL SCREENING STAGES COMPLETE", ...
            fullfile(campaignDir,'COMPLETE.txt'));
    catch err
        State.status = "failed";
        State.error = string(getReport(err,'extended','hyperlinks','off'));
        saveState(campaignDir,State,Protocol);
        writelines(State.error,fullfile(campaignDir,'FAILED.txt'));
        rethrow(err);
    end
    clear cleanup;
    deletePool(pool);
end

function [State,current] = runBinaryStage(pool,State,current,candidate, ...
        stage,Protocol,analysisDir,failureDir,campaignDir)
    control = current;
    runQueue(pool,candidate,Protocol,failureDir);
    [Summary,Problems,Runs] = analyzeStage( ...
        stage,control,candidate,Protocol,analysisDir);
    current = selectWinner(control,candidate,Summary);
    State = recordStage(State,stage,control,candidate, ...
        current,Summary,Problems,Runs);
    saveState(campaignDir,State,Protocol);
end

function runQueue(pool,Arms,Protocol,failureDir)
    tasks = buildTasks(Arms,Protocol);
    todo = false(numel(tasks),1);
    for t = 1 : numel(tasks)
        todo(t) = ~resultComplete(Protocol.rootPath,tasks(t),Protocol.maxFE);
    end
    tasks = tasks(todo);
    for attempt = 1 : 3
        if isempty(tasks)
            return;
        end
        fprintf('Screening queue: %d tasks, attempt %d\n', ...
            numel(tasks),attempt);
        rootPath = Protocol.rootPath;
        popSize = Protocol.popSize;
        maxFE = Protocol.maxFE;
        saveNum = Protocol.saveNum;
        options = parforOptions(pool,'RangePartitionMethod','fixed', ...
            'SubrangeSize',1);
        parfor (t = 1:numel(tasks),options)
            task = tasks(t);
            try
                addCBSPaths(rootPath);
                algorithm = str2func(char(task.algorithm));
                problem = str2func(char(task.problem));
                specification = [{algorithm},num2cell(task.parameters)];
                platemo('algorithm',specification,'problem',problem, ...
                    'N',popSize,'maxFE',maxFE, ...
                    'save',saveNum,'run',task.run, ...
                    'metName',{'IGD'});
                addCBSPaths(rootPath);
            catch err
                report = string(getReport( ...
                    err,'extended','hyperlinks','off'));
                name = sprintf('%s_%s_%d_attempt%d.txt', ...
                    task.algorithm,task.problem,task.run,attempt);
                writelines(report,fullfile(failureDir,name));
            end
        end
        missing = false(numel(tasks),1);
        for t = 1 : numel(tasks)
            missing(t) = ~resultComplete( ...
                Protocol.rootPath,tasks(t),Protocol.maxFE);
        end
        tasks = tasks(missing);
    end
    if ~isempty(tasks)
        error('CBSRegionGAN:ScreeningTasksFailed', ...
            '%d screening tasks failed after three attempts.',numel(tasks));
    end
end

function tasks = buildTasks(Arms,Protocol)
    template = struct('algorithm',"",'parameters',zeros(1,9), ...
        'problem',"",'run',0);
    tasks = repmat(template,numel(Arms)*numel(Protocol.problems)* ...
        numel(Protocol.runs),1);
    next = 0;
    for a = 1 : numel(Arms)
        for problem = Protocol.problems
            for run = Protocol.runs
                next = next+1;
                tasks(next).algorithm = Arms(a).algorithm;
                tasks(next).parameters = Arms(a).parameters;
                tasks(next).problem = problem;
                tasks(next).run = run;
            end
        end
    end
end

function complete = resultComplete(rootPath,task,maxFE)
    pattern = sprintf('%s_%s_M*_D*_%d.mat', ...
        task.algorithm,task.problem,task.run);
    files = dir(fullfile(rootPath,'Data',task.algorithm,pattern));
    complete = false;
    if numel(files) ~= 1
        return;
    end
    try
        Data = load(fullfile(files(1).folder,files(1).name), ...
            'result','metric');
        complete = isfield(Data,'result') && ~isempty(Data.result) && ...
            Data.result{end,1} >= maxFE && isfield(Data,'metric') && ...
            isfield(Data.metric,'IGD') && ...
            isfield(Data.metric,'CBSAudit');
    catch
        complete = false;
    end
end

function [Summary,Problems,Runs] = analyzeStage( ...
        stage,control,candidates,Protocol,analysisDir)
    [Summary,Problems,Runs] = analyze_CBS_CGAN_screening_stage( ...
        Protocol.rootPath,control.algorithm,[candidates.algorithm], ...
        Protocol.problems,Protocol.runs);
    writetable(Summary,fullfile(analysisDir,stage+"_candidates.csv"));
    writetable(Problems,fullfile(analysisDir,stage+"_problems.csv"));
    auditless = removevars(Runs,'audit');
    writetable(auditless,fullfile(analysisDir,stage+"_runs.csv"));
    save(fullfile(analysisDir,stage+".mat"), ...
        'Summary','Problems','Runs','control','candidates');
end

function winner = selectWinner(control,candidates,Summary)
    winner = control;
    passed = find(Summary.pass);
    if isempty(passed)
        return;
    end
    [~,order] = sort(Summary.gmeanRatio200K(passed),'ascend');
    best = passed(order(1));
    if numel(passed) > 1
        first = passed(1);
        if Summary.gmeanRatio200K(first) <= ...
                1.02*Summary.gmeanRatio200K(best)
            best = first;
        end
    end
    name = Summary.candidate(best);
    winner = candidates(find([candidates.algorithm] == name,1));
end

function triggered = auditTrigger(Arm,Protocol,field,threshold)
    problemRates = nan(numel(Protocol.problems),1);
    for p = 1 : numel(Protocol.problems)
        values = nan(numel(Protocol.runs),1);
        for r = 1 : numel(Protocol.runs)
            task = struct('algorithm',Arm.algorithm, ...
                'problem',Protocol.problems(p),'run',Protocol.runs(r));
            pattern = sprintf('%s_%s_M*_D*_%d.mat', ...
                task.algorithm,task.problem,task.run);
            files = dir(fullfile(Protocol.rootPath,'Data', ...
                task.algorithm,pattern));
            if isscalar(files)
                Data = load(fullfile(files(1).folder,files(1).name),'metric');
                if isfield(Data.metric,'CBSAudit') && ...
                        isfield(Data.metric.CBSAudit,field)
                    values(r) = double(Data.metric.CBSAudit.(field));
                end
            end
        end
        values = values(isfinite(values));
        if ~isempty(values)
            problemRates(p) = mean(values);
        end
    end
    triggered = sum(problemRates >= threshold) >= ...
        Protocol.triggerProblemCount;
end

function State = recordStage(State,stage,control,candidates,winner, ...
        Summary,Problems,Runs)
    Entry = struct('stage',string(stage),'status',"complete", ...
        'control',control,'candidates',candidates,'winner',winner, ...
        'reason',"performance gate",'candidateSummary',Summary, ...
        'problemSummary',Problems,'runSummary',Runs);
    State.stages{end+1,1} = Entry;
    State.current = winner;
end

function State = recordSkip(State,stage,current,reason)
    Entry = struct('stage',string(stage),'status',"skipped", ...
        'control',current,'candidates',repmat(arm("",zeros(1,9)),0,1), ...
        'winner',current,'reason',string(reason), ...
        'candidateSummary',table(),'problemSummary',table(), ...
        'runSummary',table());
    State.stages{end+1,1} = Entry;
    State.current = current;
end

function State = initialState(Protocol)
    State = struct('status',"running", ...
        'startedAt',string(datetime('now')),'completedAt',"", ...
        'error',"",'current',arm("CBS_RegionWGAN_GP_E0_Base", ...
        Protocol.baseParameters),'stages',{{}});
end

function saveState(campaignDir,State,Protocol)
    save(fullfile(campaignDir,'campaign_state.mat'),'State','Protocol');
    lines = strings(numel(State.stages)+3,1);
    lines(1) = "status="+State.status;
    lines(2) = "current="+State.current.algorithm;
    lines(3) = "parameters="+join(string(State.current.parameters),",");
    for i = 1 : numel(State.stages)
        stage = State.stages{i};
        lines(i+3) = stage.stage+":"+stage.status+":"+ ...
            stage.winner.algorithm+":"+stage.reason;
    end
    writelines(lines,fullfile(campaignDir,'campaign_state.txt'));
end

function freezeSourceOnce(rootPath,campaignDir)
    snapshot = fullfile(campaignDir,'source_snapshot');
    if isfolder(snapshot)
        return;
    end
    mkdir(snapshot);
    source = fullfile(rootPath,'Algorithms','Multi-objective optimization', ...
        'CBS-CGAN');
    copyfile(source,fullfile(snapshot,'CBS-CGAN'));
    copyfile(fullfile(rootPath,'test.m'),fullfile(snapshot,'test.m'));
end

function validateProtocol(Protocol)
    if Protocol.nWorker ~= 10 || Protocol.popSize ~= 100 || ...
            Protocol.maxFE ~= 2e5 || ~isequal(Protocol.runs,1:5) || ...
            numel(Protocol.problems) ~= 5
        error('CBSRegionGAN:BadScreeningProtocol', ...
            ['The formal screening protocol must remain 10 workers, N=100, ', ...
             'maxFE=200000, five problems, and runs 1:5.']);
    end
end

function output = setParameter(parameters,index,value)
    output = parameters;
    output(index) = value;
end

function Arm = arm(algorithm,parameters)
    Arm = struct('algorithm',string(algorithm), ...
        'parameters',double(parameters(:)'));
end

function deletePool(pool)
    try
        if ~isempty(pool) && isvalid(pool)
            delete(pool);
        end
    catch
    end
end
