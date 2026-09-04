function Summary = run_CBS_PairGuide_DE20_ablation(rootPath,nWorker)
%RUN_CBS_PAIRGUIDE_DE20_ABLATION Run paired 20%-ordinary-DE ablation.

    if nargin < 1 || isempty(rootPath)
        rootPath = fileparts(which('platemo'));
    end
    if nargin < 2 || isempty(nWorker)
        nWorker = 10;
    end
    rootPath = char(rootPath);
    addCBSPaths(rootPath);
    problems = ["LIRCMOP5_BC","LIRCMOP7_BC","LIRCMOP8_BC", ...
        "LIRCMOP10_BC","LIRCMOP12_BC","LIRCMOP14_BC"];
    runs = 1:5;
    expectedFE = 10000:10000:100000;
    outputDir = fullfile(rootPath,'Data', ...
        'CBS_PairGuide_DE20_ablation_v1_20260902');
    resultDir = fullfile(outputDir,'results');
    if ~isfolder(resultDir), mkdir(resultDir); end

    smokeTest();
    tasks = buildTasks(problems,runs,resultDir);
    setenv('OMP_NUM_THREADS','1');
    setenv('MKL_NUM_THREADS','1');
    setenv('OPENBLAS_NUM_THREADS','1');
    [pool,ownsPool] = exactProcessPool(nWorker);
    cleanup = onCleanup(@()closeOwnedPool(pool,ownsPool));
    parfor i = 1 : numel(tasks)
        runOne(tasks(i),expectedFE);
    end

    Summary = summarizeResults(rootPath,outputDir,tasks,expectedFE);
    save(fullfile(outputDir,'summary.mat'),'Summary','-v7.3');
    writelines(["status=complete"; ...
        "taskCount="+numel(tasks); ...
        "saveCount="+numel(expectedFE); ...
        "finishedAt="+string(datetime('now'))], ...
        fullfile(outputDir,'COMPLETE.txt'));
end

function Tasks = buildTasks(problems,runs,resultDir)
    template = struct('problem',"",'run',0,'seed',0,'outputFile',"");
    Tasks = repmat(template,numel(problems)*numel(runs),1);
    row = 0;
    for problem = problems
        for run = runs
            row = row+1;
            Tasks(row).problem = problem;
            Tasks(row).run = run;
            Tasks(row).seed = run;
            Tasks(row).outputFile = string(fullfile(resultDir,sprintf( ...
                '%s_DE20_run%02d.mat',problem,run)));
        end
    end
end

function runOne(Task,expectedFE)
    if resultComplete(Task,expectedFE)
        return;
    end
    if isfile(Task.outputFile)
        error('CBSPairGuide:InvalidDE20Result', ...
            'Refusing to overwrite an invalid result: %s',Task.outputFile);
    end
    rng(Task.seed,'twister');
    constructor = str2func(char(Task.problem));
    Problem = constructor('N',100,'D',30,'maxFE',100000, ...
        'maxRuntime',Inf);
    Algorithm = CBS_RegionWGAN_GP_PairGuide_DE20( ...
        'save',10,'run',Task.run,'outputFcn',@(varargin)[]);
    startedAt = string(datetime('now'));
    timer = tic;
    Algorithm.Solve(Problem);
    wallClockSeconds = toc(timer);
    FE = cell2mat(Algorithm.result(:,1));
    IGD = reshape(double(Algorithm.CalMetric('IGD')),[],1);
    Audit = Algorithm.guideExperimentSnapshot();
    if ~isequal(reshape(double(FE),1,[]),expectedFE) || ...
            numel(IGD) ~= 10 || any(~isfinite(IGD)) || ...
            string(Audit.generationMode) ~= "traditional_de" || ...
            Audit.rawCandidates ~= 0 || Audit.pairTrainingEvents ~= 0 || ...
            Audit.guidedRequested ~= 20*499 || ...
            Audit.guidedSelected ~= Audit.guidedRequested || ...
            Audit.guidedFallback ~= 0
        error('CBSPairGuide:BadDE20Run', ...
            'DE20 run violated its FE, metric, or mechanism contract.');
    end
    Record = struct( ...
        'schemaVersion',"CBS-PairGuide-DE20-ablation-v1", ...
        'algorithm',"CBS_RegionWGAN_GP_PairGuide_DE20", ...
        'problem',Task.problem,'run',Task.run,'seed',Task.seed, ...
        'N',100,'D',30,'maxFE',100000,'finalFE',double(Problem.FE), ...
        'save',10,'metric',"IGD",'replacement',"ordinary_de", ...
        'quotaPerGeneration',20,'generationCount',499, ...
        'quotaRequested',double(Audit.guidedRequested), ...
        'quotaReplaced',double(Audit.guidedSelected), ...
        'startedAt',startedAt,'finishedAt',string(datetime('now')), ...
        'wallClockSeconds',double(wallClockSeconds), ...
        'algorithmRuntimeSeconds',double(Algorithm.metric.runtime));
    partial = char(Task.outputFile+".partial.mat");
    save(partial,'Record','FE','IGD','-v7.3');
    [ok,message] = movefile(partial,Task.outputFile,'f');
    if ~ok
        error('CBSPairGuide:DE20MoveFailed','%s',message);
    end
end

function complete = resultComplete(Task,expectedFE)
    complete = false;
    if ~isfile(Task.outputFile), return; end
    try
        Data = load(Task.outputFile,'Record','FE','IGD');
        R = Data.Record;
        complete = R.schemaVersion == ...
            "CBS-PairGuide-DE20-ablation-v1" && ...
            string(R.problem) == Task.problem && R.run == Task.run && ...
            R.seed == Task.seed && R.N == 100 && R.D == 30 && ...
            R.maxFE == 100000 && R.finalFE == 100000 && R.save == 10 && ...
            R.metric == "IGD" && R.replacement == "ordinary_de" && ...
            R.quotaRequested == 20*499 && ...
            R.quotaReplaced == R.quotaRequested && ...
            isequal(reshape(double(Data.FE),1,[]),expectedFE) && ...
            numel(Data.IGD) == 10 && all(isfinite(Data.IGD));
    catch
        complete = false;
    end
end

function Summary = summarizeResults(rootPath,outputDir,Tasks,expectedFE)
    baselineDir = fullfile(rootPath,'Data', ...
        'CBS_PairGuide_training_schedule_v1_20260901','ratio','results');
    rows = repmat(struct('problem',"",'run',0,'seed',0,'FE',0, ...
        'DE20IGD',NaN,'CGANIGD',NaN,'deltaDE20MinusCGAN',NaN), ...
        numel(Tasks)*numel(expectedFE),1);
    taskRows = repmat(struct('problem',"",'run',0,'seed',0, ...
        'finalDE20IGD',NaN,'finalCGANIGD',NaN, ...
        'deltaDE20MinusCGAN',NaN,'wallClockSeconds',NaN),numel(Tasks),1);
    row = 0;
    for i = 1 : numel(Tasks)
        Data = load(Tasks(i).outputFile,'Record','FE','IGD');
        baselineFile = fullfile(baselineDir,sprintf( ...
            '%s_nCritic04_run%02d.mat',Tasks(i).problem,Tasks(i).run));
        Base = load(baselineFile,'Record','Trajectory');
        if Base.Record.nCritic ~= 4 || Base.Record.retrainEpoch ~= 10 || ...
                Base.Record.finalFE ~= 100000
            error('CBSPairGuide:BadDE20Baseline', ...
                'Historical CGAN baseline contract mismatch.');
        end
        [present,baseRows] = ismember(expectedFE,double(Base.Trajectory.FE));
        if ~all(present)
            error('CBSPairGuide:MissingDE20BaselineFE', ...
                'Historical baseline lacks one requested FE checkpoint.');
        end
        baseIGD = double(Base.Trajectory.IGD(baseRows));
        for checkpoint = 1 : numel(expectedFE)
            row = row+1;
            rows(row).problem = Tasks(i).problem;
            rows(row).run = Tasks(i).run;
            rows(row).seed = Tasks(i).seed;
            rows(row).FE = expectedFE(checkpoint);
            rows(row).DE20IGD = Data.IGD(checkpoint);
            rows(row).CGANIGD = baseIGD(checkpoint);
            rows(row).deltaDE20MinusCGAN = ...
                Data.IGD(checkpoint)-baseIGD(checkpoint);
        end
        taskRows(i).problem = Tasks(i).problem;
        taskRows(i).run = Tasks(i).run;
        taskRows(i).seed = Tasks(i).seed;
        taskRows(i).finalDE20IGD = Data.IGD(end);
        taskRows(i).finalCGANIGD = baseIGD(end);
        taskRows(i).deltaDE20MinusCGAN = Data.IGD(end)-baseIGD(end);
        taskRows(i).wallClockSeconds = Data.Record.wallClockSeconds;
    end
    PairedTrajectory = struct2table(rows);
    TaskSummary = struct2table(taskRows);
    CheckpointSummary = groupedSummary(PairedTrajectory,expectedFE);
    FinalSummary = CheckpointSummary(CheckpointSummary.FE == 100000,:);
    writetable(PairedTrajectory,fullfile(outputDir,'paired_igd_trajectory.csv'));
    writetable(TaskSummary,fullfile(outputDir,'task_final_igd.csv'));
    writetable(CheckpointSummary, ...
        fullfile(outputDir,'problem_checkpoint_summary.csv'));
    writetable(FinalSummary,fullfile(outputDir,'problem_final_summary.csv'));
    Summary = struct('outputDir',string(outputDir), ...
        'PairedTrajectory',PairedTrajectory,'TaskSummary',TaskSummary, ...
        'CheckpointSummary',CheckpointSummary,'FinalSummary',FinalSummary);
end

function Table = groupedSummary(PairedTrajectory,expectedFE)
    problems = unique(PairedTrajectory.problem,'stable');
    template = struct('problem',"",'FE',0,'DE20Mean',NaN, ...
        'DE20Std',NaN,'DE20Median',NaN,'CGANMean',NaN,'CGANStd',NaN, ...
        'CGANMedian',NaN,'meanDeltaDE20MinusCGAN',NaN, ...
        'medianDeltaDE20MinusCGAN',NaN,'CGANWins',0,'DE20Wins',0, ...
        'ties',0);
    rows = repmat(template,numel(problems)*numel(expectedFE),1);
    row = 0;
    for problem = reshape(problems,1,[])
        for fe = expectedFE
            row = row+1;
            keep = PairedTrajectory.problem == problem & ...
                PairedTrajectory.FE == fe;
            de = PairedTrajectory.DE20IGD(keep);
            cg = PairedTrajectory.CGANIGD(keep);
            delta = de-cg;
            rows(row).problem = problem;
            rows(row).FE = fe;
            rows(row).DE20Mean = mean(de);
            rows(row).DE20Std = std(de);
            rows(row).DE20Median = median(de);
            rows(row).CGANMean = mean(cg);
            rows(row).CGANStd = std(cg);
            rows(row).CGANMedian = median(cg);
            rows(row).meanDeltaDE20MinusCGAN = mean(delta);
            rows(row).medianDeltaDE20MinusCGAN = median(delta);
            rows(row).CGANWins = nnz(delta > 0);
            rows(row).DE20Wins = nnz(delta < 0);
            rows(row).ties = nnz(delta == 0);
        end
    end
    Table = struct2table(rows);
end

function smokeTest()
    rng(1,'twister');
    Algorithm = CBS_RegionWGAN_GP_PairGuide_DE20( ...
        'save',1,'outputFcn',@(varargin)[]);
    Problem = DASCMOP1_BC('N',10,'D',5,'maxFE',40);
    Algorithm.Solve(Problem);
    Audit = Algorithm.guideExperimentSnapshot();
    assert(Audit.generationMode == "traditional_de" && ...
        Audit.useMode == "traditional_de" && ...
        Audit.guidedRequested == 2 && Audit.guidedSelected == 2 && ...
        Audit.guidedFallback == 0 && Audit.rawCandidates == 0 && ...
        Audit.pairTrainingEvents == 0);
end

function [pool,ownsPool] = exactProcessPool(nWorker)
    pool = gcp('nocreate');
    ownsPool = isempty(pool);
    if ownsPool
        pool = parpool("Processes",nWorker);
    elseif pool.NumWorkers ~= nWorker || contains(class(pool),'ThreadPool')
        error('CBSPairGuide:BadDE20Pool', ...
            'Experiment requires exactly %d process workers.',nWorker);
    end
end

function closeOwnedPool(pool,ownsPool)
    if ownsPool && ~isempty(pool) && isvalid(pool)
        delete(pool);
    end
end
