function [Summary,outDir,ByProblem,ByVariant,Raw,Manifest] = ...
    run_CBS_CGAN_loss_ablation_lir_offline(outDir,Options)
%RUN_CBS_CGAN_LOSS_ABLATION_LIR_OFFLINE Offline CBS-CGAN loss ablation.
%
% The offline layer freezes one boundary dataset snapshot per problem and
% trains loss variants on exactly the same TrainX/TrainC/QueryC data.

    rootDir = fileparts(which('platemo'));
    if isempty(rootDir)
        rootDir = pwd;
    end
    addpath(genpath(rootDir));
    if nargin < 1 || isempty(outDir)
        outDir = fullfile(rootDir,'Data','CBS_CGAN', ...
            ['loss_ablation_lir_offline_', ...
            char(datetime('now','Format','yyyyMMdd_HHmmss'))]);
    end
    if nargin < 2 || isempty(Options)
        Options = struct();
    end
    Options = normalizeLossAblationOptions(Options);
    if ~isfolder(outDir)
        mkdir(outDir);
    end

    Variants = lossAblationVariants();
    [SnapshotManifest,snapshotFiles] = prepareLossSnapshots(outDir,Options);
    Tasks = buildLossTasks(snapshotFiles,Variants,Options.ganSeeds);
    Rows = repmat(emptyLossRawRow(),height(Tasks),1);

    if Options.workerCount > 1
        ensureLossParallelPool(Options.workerCount);
        parfor task = 1 : height(Tasks)
            Rows(task) = runOneLossTask(Tasks(task,:),Options);
        end
    else
        for task = 1 : height(Tasks)
            Rows(task) = runOneLossTask(Tasks(task,:),Options);
            fprintf('[%d/%d] %s %s seed=%d status=%s\n', ...
                task,height(Tasks),Rows(task).problem,Rows(task).variant, ...
                Rows(task).gan_seed,Rows(task).status);
        end
    end

    Raw = struct2table(Rows);
    ByProblem = aggregateLossMetrics(Raw,{'variant','problem'});
    ByVariant = aggregateLossMetrics(Raw,{'variant'});
    Summary = buildLossComparisonSummary(Raw);
    Manifest = buildLossManifest(Variants,Options,SnapshotManifest);

    writetable(Summary,fullfile(outDir,'offline_loss_ablation_summary.csv'));
    writetable(ByProblem,fullfile(outDir, ...
        'offline_loss_ablation_by_problem.csv'));
    writetable(ByVariant,fullfile(outDir, ...
        'offline_loss_ablation_by_variant.csv'));
    writetable(Raw,fullfile(outDir,'offline_loss_ablation_raw.csv'));
    writetable(Manifest,fullfile(outDir, ...
        'offline_loss_ablation_manifest.csv'));
    writetable(SnapshotManifest,fullfile(outDir,'snapshot_manifest.csv'));
end

function Options = normalizeLossAblationOptions(Options)
    Options = ensureLossField(Options,'workerCount',10);
    Options = ensureLossField(Options,'problemNames',defaultLossProblemList());
    Options = ensureLossField(Options,'snapshotRunIds',1);
    Options = ensureLossField(Options,'ganSeeds',1:3);
    Options = ensureLossField(Options,'N',100);
    Options = ensureLossField(Options,'D',[]);
    Options = ensureLossField(Options,'maxFE',100000);
    Options = ensureLossField(Options,'snapshotFE',Options.maxFE);
    Options = ensureLossField(Options,'conditionMode',"ref_y");
    Options = ensureLossField(Options,'pairCount',3);
    Options = ensureLossField(Options,'queryPerCondition',1);
    Options = ensureLossField(Options,'nGen',20);
    Options = ensureLossField(Options,'zDim',2);
    Options = ensureLossField(Options,'ganIter',50);
    Options = ensureLossField(Options,'ganMiniBatch',32);
    Options = ensureLossField(Options,'ganLrD',1e-4);
    Options = ensureLossField(Options,'ganLrG',2e-4);
    Options = ensureLossField(Options,'ganDPretrainIter',0);
    Options = ensureLossField(Options,'ganDSteps',1);
    Options = ensureLossField(Options,'ganGSteps',1);
    Options = ensureLossField(Options,'pairNeighborRefRadius',4);
    Options = ensureLossField(Options,'minBoundaryLength',2);
    Options = ensureLossField(Options,'reconstructionHuberDelta',0.10);
    Options = ensureLossField(Options,'pairMargin',0.05);
    Options = ensureLossField(Options,'sigma',0.05);
    Options = ensureLossField(Options,'trainZMode',"zero");
    Options = ensureLossField(Options,'coverageTolerance',0.05);
    Options = ensureLossField(Options,'outlierThreshold',0.10);
    Options = ensureLossField(Options,'plotEnabled',false);
    Options = ensureLossField(Options,'snapshotFiles',strings(0,1));

    Options.workerCount = max(1,round(double(Options.workerCount)));
    Options.problemNames = string(Options.problemNames(:));
    Options.snapshotRunIds = unique(double(Options.snapshotRunIds(:)'),'stable');
    Options.ganSeeds = unique(double(Options.ganSeeds(:)'),'stable');
    Options.N = max(1,round(double(Options.N)));
    if ~isempty(Options.D)
        Options.D = max(1,round(double(Options.D)));
    end
    Options.maxFE = max(1,round(double(Options.maxFE)));
    Options.snapshotFE = max(1,round(double(Options.snapshotFE)));
    Options.conditionMode = lower(strtrim(string(Options.conditionMode)));
    Options.pairCount = max(1,round(double(Options.pairCount)));
    Options.queryPerCondition = max(1,round(double(Options.queryPerCondition)));
    Options.nGen = max(0,round(double(Options.nGen)));
    Options.zDim = max(0,round(double(Options.zDim)));
    Options.ganIter = max(0,round(double(Options.ganIter)));
    Options.ganMiniBatch = max(1,round(double(Options.ganMiniBatch)));
    Options.ganDPretrainIter = max(0,round(double(Options.ganDPretrainIter)));
    Options.ganDSteps = max(1,round(double(Options.ganDSteps)));
    Options.ganGSteps = max(1,round(double(Options.ganGSteps)));
    Options.pairNeighborRefRadius = max(0,round(double( ...
        Options.pairNeighborRefRadius)));
    Options.minBoundaryLength = max(1,round(double( ...
        Options.minBoundaryLength)));
    Options.trainZMode = lower(strtrim(string(Options.trainZMode)));
    Options.coverageTolerance = max(0,double(Options.coverageTolerance));
    Options.outlierThreshold = max(0,double(Options.outlierThreshold));
    Options.plotEnabled = logical(Options.plotEnabled);
    if ischar(Options.snapshotFiles)
        Options.snapshotFiles = string({Options.snapshotFiles});
    elseif iscellstr(Options.snapshotFiles)
        Options.snapshotFiles = string(Options.snapshotFiles(:));
    else
        Options.snapshotFiles = string(Options.snapshotFiles(:));
    end

    assert(~isempty(Options.problemNames), ...
        'CBSCGANLossAblation:EmptyProblemList', ...
        'Options.problemNames must not be empty.');
    assert(~isempty(Options.ganSeeds), ...
        'CBSCGANLossAblation:EmptySeeds', ...
        'Options.ganSeeds must not be empty.');
end

function S = ensureLossField(S,name,value)
    if ~isfield(S,name) || isempty(S.(name))
        S.(name) = value;
    end
end

function problemNames = defaultLossProblemList()
    problemNames = ["LIRCMOP5_BC";"LIRCMOP6_BC";"LIRCMOP7_BC"; ...
        "LIRCMOP8_BC";"LIRCMOP9_BC";"LIRCMOP10_BC"];
end

function Variants = lossAblationVariants()
    Variants = struct( ...
        'variant',{"V0_Pure_CGAN","V1_CGAN_mismatchD", ...
            "V2_Adv_Huber","V3_Adv_Pair","V4_Full_current", ...
            "V5_Huber_Pair_no_adv"}, ...
        'description',{"D real/fake, G adversarial only", ...
            "D real/fake/mismatch, G adversarial only", ...
            "Adversarial plus Huber reconstruction", ...
            "Adversarial plus pair-margin", ...
            "Adversarial plus Huber plus pair-margin", ...
            "Huber plus pair-margin, no adversarial"}, ...
        'advWeight',{1,1,1,1,1,0}, ...
        'reconstructionWeight',{0,0,1,0,1,1}, ...
        'pairMarginWeight',{0,0,0,1,1,1}, ...
        'useMismatchD',{false,true,true,true,true,false});
end

function [SnapshotManifest,snapshotFiles] = prepareLossSnapshots(outDir,Options)
    if ~isempty(Options.snapshotFiles)
        snapshotFiles = Options.snapshotFiles;
        Rows = repmat(emptySnapshotRow(),numel(snapshotFiles),1);
        for i = 1 : numel(snapshotFiles)
            Snapshot = loadLossSnapshot(snapshotFiles(i));
            Rows(i) = snapshotManifestRow(Snapshot,snapshotFiles(i),"provided");
        end
        SnapshotManifest = struct2table(Rows);
        return;
    end

    snapshotDir = fullfile(outDir,'snapshots');
    if ~isfolder(snapshotDir)
        mkdir(snapshotDir);
    end
    Tasks = buildSnapshotTasks(Options.problemNames,Options.snapshotRunIds);
    Rows = repmat(emptySnapshotRow(),height(Tasks),1);
    if Options.workerCount > 1
        ensureLossParallelPool(Options.workerCount);
        parfor task = 1 : height(Tasks)
            Rows(task) = captureOneSnapshot(Tasks(task,:),snapshotDir,Options);
        end
    else
        for task = 1 : height(Tasks)
            Rows(task) = captureOneSnapshot(Tasks(task,:),snapshotDir,Options);
            fprintf('[snapshot %d/%d] %s run=%d status=%s\n', ...
                task,height(Tasks),Rows(task).problem,Rows(task).run, ...
                Rows(task).status);
        end
    end
    SnapshotManifest = struct2table(Rows);
    ok = string(SnapshotManifest.status) == "ok";
    snapshotFiles = string(SnapshotManifest.snapshot_file(ok));
end

function Tasks = buildSnapshotTasks(problemNames,runIds)
    Rows = repmat(struct('problem',"",'run',NaN), ...
        numel(problemNames)*numel(runIds),1);
    k = 0;
    for p = 1 : numel(problemNames)
        for r = 1 : numel(runIds)
            k = k + 1;
            Rows(k).problem = problemNames(p);
            Rows(k).run = runIds(r);
        end
    end
    Tasks = struct2table(Rows);
end

function Row = captureOneSnapshot(Task,snapshotDir,Options)
    Row = emptySnapshotRow();
    Row.problem = string(Task.problem);
    Row.run = double(Task.run);
    Row.seed = Row.run;
    Row.condition_mode = string(Options.conditionMode);
    Row.snapshot_source = "generated";
    try
        rootDir = fileparts(which('platemo'));
        addpath(genpath(rootDir));
        try
            maxNumCompThreads(1);
        catch
        end
        rng(Row.seed,'twister');
        Problem = makeLossProblem(Row.problem,Options.N,Options.D, ...
            Options.snapshotFE);
        Snapshot = captureDatasetSnapshot(Problem,Row.problem,Row.run, ...
            Options);
        if isempty(Snapshot.TrainX) || isempty(Snapshot.QueryC)
            Row.status = "empty_snapshot";
            return;
        end
        fileName = fullfile(snapshotDir,sprintf('%s_run%d_snapshot.mat', ...
            char(Row.problem),round(Row.run)));
        save(fileName,'Snapshot','-v7.3');
        Row = snapshotManifestRow(Snapshot,fileName,"generated");
        Row.status = "ok";
    catch err
        Row.status = "failed";
        Row.error_message = string(getReport(err,'extended', ...
            'hyperlinks','off'));
    end
end

function Snapshot = captureDatasetSnapshot(Problem,problemName,runId,Options)
    [W,~] = UniformPoint(Problem.N,Problem.M);
    BoundaryOptions = struct( ...
        'pairNeighborRefRadius',Options.pairNeighborRefRadius, ...
        'maxCandidatePairsPerRef',Options.pairCount, ...
        'maxCanonicalPairsPerRef',Options.pairCount, ...
        'minBoundaryLength',Options.minBoundaryLength);
    DatasetOptions = struct( ...
        'pairNeighborRefRadius',Options.pairNeighborRefRadius, ...
        'queryConditionBudget',max(1,ceil(max(Options.nGen,1)/ ...
            Options.queryPerCondition)), ...
        'conditionMode',Options.conditionMode);
    GANOptions = baseGANOptions(Options);

    Population1 = Problem.Initialization();
    Population2 = Problem.Initialization();
    DatasetOptions.conditionScale = initializeLossConditionScale( ...
        [Population1,Population2],Problem);
    Fitness1 = CalFitness_CBS(Population1.objs,Population1.cons);
    Fitness2 = CalFitness_CBS(Population2.objs);
    BMem = [];
    GAN = [];
    gen = 0;
    Snapshot = emptyLossSnapshot(problemName,runId,Problem,W,Options);

    while Problem.FE < Options.snapshotFE
        gen = gen + 1;
        MatingPool1 = TournamentSelection(2,2*Problem.N,Fitness1);
        MatingPool2 = TournamentSelection(2,2*Problem.N,Fitness2);
        Offspring1 = OperatorDE(Problem,Population1, ...
            Population1(MatingPool1(1:end/2)), ...
            Population1(MatingPool1(end/2+1:end)));
        Offspring2 = OperatorDE(Problem,Population2, ...
            Population2(MatingPool2(1:end/2)), ...
            Population2(MatingPool2(end/2+1:end)));

        if mod(gen,1) == 0
            [BMem,~] = UpdateBoundaryMemory_CBS(BMem,Population1, ...
                Offspring1,Population2,Offspring2,W,BoundaryOptions);
        end

        OffspringG = Offspring1([]);
        if ~isempty(BMem) && Options.nGen > 0
            [TrainX,TrainC,QueryC,BMem,DatasetInfo] = ...
                BuildBoundaryDataset_CBS(BMem, ...
                [Population1,Offspring1,Population2,Offspring2], ...
                W,Problem,DatasetOptions);
            if size(TrainX,1) >= Options.minBoundaryLength && ...
                    ~isempty(TrainC) && ~isempty(QueryC)
                TrainGANOptions = GANOptions;
                TrainGANOptions.trainXf = DatasetInfo.trainXf;
                TrainGANOptions.trainXi = DatasetInfo.trainXi;
                GAN = BoundaryCGAN_CBS('train',GAN,TrainX,TrainC, ...
                    Problem,TrainGANOptions);
                SampleOptions = GANOptions;
                nSample = size(QueryC,1)*Options.queryPerCondition;
                SampleOptions.sampleZ = zeros(nSample,Options.zDim);
                [RawDec,~] = BoundaryCGAN_CBS('samplebycondition',GAN, ...
                    QueryC,Options.queryPerCondition,SampleOptions);
                if size(RawDec,1) > Options.nGen
                    RawDec = RawDec(1:Options.nGen,:);
                end
                if ~isempty(RawDec)
                    OffspringG = Problem.Evaluation(RawDec);
                end
                Snapshot = makeLossSnapshot(problemName,runId,Problem,W, ...
                    BMem,TrainX,TrainC,QueryC,DatasetInfo,GANOptions, ...
                    gen,Options);
            end
        end

        UnionPopulation = [Population1,Population2,Offspring1, ...
            Offspring2,OffspringG];
        [Population1,Fitness1] = EnvironmentalSelection_CBS( ...
            UnionPopulation,Problem.N,true);
        [Population2,Fitness2] = EnvironmentalSelection_CBS( ...
            UnionPopulation,Problem.N,false);
    end
end

function Snapshot = makeLossSnapshot(problemName,runId,Problem,W,BMem, ...
        TrainX,TrainC,QueryC,DatasetInfo,GANOptions,gen,Options)
    Snapshot = emptyLossSnapshot(problemName,runId,Problem,W,Options);
    Snapshot.actualFE = double(Problem.FE);
    Snapshot.gen = double(gen);
    Snapshot.W = W;
    Snapshot.BMem = BMem;
    Snapshot.TrainX = TrainX;
    Snapshot.TrainC = TrainC;
    Snapshot.QueryC = QueryC;
    Snapshot.trainObjs = DatasetInfo.trainObjs;
    Snapshot.queryObjs = DatasetInfo.queryObjs;
    Snapshot.trainXf = DatasetInfo.trainXf;
    Snapshot.trainXi = DatasetInfo.trainXi;
    Snapshot.trainYf = DatasetInfo.trainYf;
    Snapshot.trainYi = DatasetInfo.trainYi;
    Snapshot.queryMeta = DatasetInfo.queryMeta;
    Snapshot.objMin = DatasetInfo.objMin;
    Snapshot.objSpan = DatasetInfo.objSpan;
    Snapshot.GANOptions = GANOptions;
end

function Snapshot = emptyLossSnapshot(problemName,runId,Problem,W,Options)
    Snapshot = struct( ...
        'problem',string(problemName), ...
        'run',double(runId), ...
        'seed',double(runId), ...
        'N',double(Problem.N), ...
        'D',double(Problem.D), ...
        'maxFE',double(Options.snapshotFE), ...
        'actualFE',double(Problem.FE), ...
        'gen',0, ...
        'conditionMode',string(Options.conditionMode), ...
        'W',W, ...
        'BMem',[], ...
        'TrainX',zeros(0,Problem.D), ...
        'TrainC',zeros(0,0), ...
        'QueryC',zeros(0,0), ...
        'trainObjs',zeros(0,Problem.M), ...
        'queryObjs',zeros(0,Problem.M), ...
        'trainXf',zeros(0,Problem.D), ...
        'trainXi',zeros(0,Problem.D), ...
        'trainYf',zeros(0,Problem.M), ...
        'trainYi',zeros(0,Problem.M), ...
        'queryMeta',struct(), ...
        'objMin',zeros(1,Problem.M), ...
        'objSpan',ones(1,Problem.M), ...
        'GANOptions',baseGANOptions(Options));
end

function Options = baseGANOptions(Options)
    Options = struct( ...
        'zDim',Options.zDim, ...
        'iter',Options.ganIter, ...
        'miniBatch',Options.ganMiniBatch, ...
        'lrD',double(Options.ganLrD), ...
        'lrG',double(Options.ganLrG), ...
        'dPretrainIter',Options.ganDPretrainIter, ...
        'dSteps',Options.ganDSteps, ...
        'gSteps',Options.ganGSteps, ...
        'sigma',double(Options.sigma), ...
        'reconstructionHuberDelta',double( ...
            Options.reconstructionHuberDelta), ...
        'reconstructionWeight',1, ...
        'pairMargin',double(Options.pairMargin), ...
        'pairMarginWeight',1, ...
        'advWeight',1, ...
        'useMismatchD',true, ...
        'trainZMode',Options.trainZMode);
end

function Scale = initializeLossConditionScale(Population,Problem)
    if isempty(Population)
        Obj = zeros(0,Problem.M);
    else
        Obj = Population.objs;
    end
    Obj = double(Obj);
    Obj = Obj(all(isfinite(Obj),2),:);
    if isempty(Obj)
        objMin = zeros(1,Problem.M);
        objSpan = ones(1,Problem.M);
    else
        objMin = min(Obj,[],1);
        objMax = max(Obj,[],1);
        objSpan = objMax - objMin;
        objSpan(objSpan <= eps) = 1;
    end
    Scale = struct('objMin',objMin,'objSpan',objSpan);
end

function Row = snapshotManifestRow(Snapshot,fileName,source)
    Row = emptySnapshotRow();
    Row.problem = string(Snapshot.problem);
    Row.run = double(Snapshot.run);
    Row.seed = double(getSnapshotField(Snapshot,'seed',Snapshot.run));
    Row.condition_mode = string(getSnapshotField(Snapshot, ...
        'conditionMode',""));
    Row.snapshot_source = string(source);
    Row.snapshot_file = string(fileName);
    Row.N = double(getSnapshotField(Snapshot,'N',NaN));
    Row.D = double(getSnapshotField(Snapshot,'D',NaN));
    Row.maxFE = double(getSnapshotField(Snapshot,'maxFE',NaN));
    Row.actualFE = double(getSnapshotField(Snapshot,'actualFE',NaN));
    Row.gen = double(getSnapshotField(Snapshot,'gen',NaN));
    Row.train_count = size(Snapshot.TrainX,1);
    Row.query_count = size(Snapshot.QueryC,1);
    Row.condition_dim = size(Snapshot.TrainC,2);
    if isfield(Snapshot,'BMem') && isstruct(Snapshot.BMem) && ...
            isfield(Snapshot.BMem,'y_b')
        Row.bmem_count = size(Snapshot.BMem.y_b,1);
    end
    Row.status = "ok";
end

function Row = emptySnapshotRow()
    Row = struct( ...
        'problem',"", ...
        'run',NaN, ...
        'seed',NaN, ...
        'condition_mode',"", ...
        'snapshot_source',"", ...
        'snapshot_file',"", ...
        'N',NaN, ...
        'D',NaN, ...
        'maxFE',NaN, ...
        'actualFE',NaN, ...
        'gen',NaN, ...
        'train_count',0, ...
        'query_count',0, ...
        'condition_dim',0, ...
        'bmem_count',0, ...
        'status',"pending", ...
        'error_message',"");
end

function Tasks = buildLossTasks(snapshotFiles,Variants,ganSeeds)
    Rows = repmat(struct('snapshot_file',"",'variant_index',NaN, ...
        'gan_seed',NaN),numel(snapshotFiles)*numel(Variants)* ...
        numel(ganSeeds),1);
    k = 0;
    for s = 1 : numel(snapshotFiles)
        for v = 1 : numel(Variants)
            for r = 1 : numel(ganSeeds)
                k = k + 1;
                Rows(k).snapshot_file = snapshotFiles(s);
                Rows(k).variant_index = v;
                Rows(k).gan_seed = ganSeeds(r);
            end
        end
    end
    Tasks = struct2table(Rows);
end

function Row = runOneLossTask(Task,Options)
    Variants = lossAblationVariants();
    variant = Variants(round(double(Task.variant_index)));
    Row = emptyLossRawRow();
    Row.snapshot_file = string(Task.snapshot_file);
    Row.variant = string(variant.variant);
    Row.variant_description = string(variant.description);
    Row.gan_seed = double(Task.gan_seed);
    Row.advWeight = double(variant.advWeight);
    Row.reconstructionWeight = double(variant.reconstructionWeight);
    Row.pairMarginWeight = double(variant.pairMarginWeight);
    Row.useMismatchD = logical(variant.useMismatchD);
    try
        rootDir = fileparts(which('platemo'));
        addpath(genpath(rootDir));
        try
            maxNumCompThreads(1);
        catch
        end
        Snapshot = loadLossSnapshot(Row.snapshot_file);
        Row.problem = string(Snapshot.problem);
        Row.snapshot_run = double(Snapshot.run);
        Row.condition_mode = string(getSnapshotField(Snapshot, ...
            'conditionMode',Options.conditionMode));
        Row.train_count = size(Snapshot.TrainX,1);
        Row.query_count = size(Snapshot.QueryC,1);
        Row.condition_dim = size(Snapshot.TrainC,2);
        Row.D = double(getSnapshotField(Snapshot,'D',size(Snapshot.TrainX,2)));
        if isempty(Snapshot.TrainX) || isempty(Snapshot.TrainC) || ...
                isempty(Snapshot.QueryC)
            Row.status = "empty_snapshot";
            return;
        end
        Problem = makeLossProblem(Row.problem, ...
            double(getSnapshotField(Snapshot,'N',Options.N)), ...
            Row.D,double(getSnapshotField(Snapshot,'maxFE',Options.maxFE)));
        rng(Row.gan_seed,'twister');
        GANOptions = makeVariantGANOptions(Snapshot,Options,variant);
        tStart = tic;
        GAN = BoundaryCGAN_CBS('train',[],Snapshot.TrainX, ...
            Snapshot.TrainC,Problem,GANOptions);
        Row.runtime = toc(tStart);
        Metrics = evaluateOfflineGAN(GAN,Snapshot,Problem,GANOptions, ...
            Options);
        Row = copyLossMetrics(Row,Metrics);
        Row.status = "ok";
    catch err
        Row.status = "failed";
        Row.error_message = string(getReport(err,'extended', ...
            'hyperlinks','off'));
    end
end

function Options = makeVariantGANOptions(Snapshot,RunnerOptions,variant)
    if isfield(Snapshot,'GANOptions') && isstruct(Snapshot.GANOptions)
        Options = Snapshot.GANOptions;
    else
        Options = baseGANOptions(RunnerOptions);
    end
    Options.iter = RunnerOptions.ganIter;
    Options.miniBatch = RunnerOptions.ganMiniBatch;
    Options.zDim = RunnerOptions.zDim;
    Options.sigma = RunnerOptions.sigma;
    Options.reconstructionHuberDelta = RunnerOptions.reconstructionHuberDelta;
    Options.pairMargin = RunnerOptions.pairMargin;
    Options.trainZMode = RunnerOptions.trainZMode;
    Options.advWeight = double(variant.advWeight);
    Options.reconstructionWeight = double(variant.reconstructionWeight);
    Options.pairMarginWeight = double(variant.pairMarginWeight);
    Options.useMismatchD = logical(variant.useMismatchD);
    Options.trainXf = Snapshot.trainXf;
    Options.trainXi = Snapshot.trainXi;
end

function Metrics = evaluateOfflineGAN(GAN,Snapshot,Problem,GANOptions, ...
        Options)
    Metrics = emptyLossMetrics();
    zDim = GAN.zDim;
    trainN = size(Snapshot.TrainC,1);
    queryN = size(Snapshot.QueryC,1);
    TrainOptions = GANOptions;
    TrainOptions.sampleZ = zeros(trainN,zDim);
    QueryOptions = GANOptions;
    QueryOptions.sampleZ = zeros(queryN,zDim);
    [TrainDec,~] = BoundaryCGAN_CBS('samplebycondition',GAN, ...
        Snapshot.TrainC,1,TrainOptions);
    [QueryDec,QueryInfo] = BoundaryCGAN_CBS('samplebycondition',GAN, ...
        Snapshot.QueryC,1,QueryOptions);
    RandomOptions = GANOptions;
    RandomOptions = rmfieldIfPresent(RandomOptions,'sampleZ');
    [RandomDec,~] = BoundaryCGAN_CBS('samplebycondition',GAN, ...
        Snapshot.QueryC,1,RandomOptions);

    [TrainObj,~] = EvaluateDecisions_CBS(Problem,TrainDec);
    [QueryObj,QueryCon] = EvaluateDecisions_CBS(Problem,QueryDec);
    [RandomObj,~] = EvaluateDecisions_CBS(Problem,RandomDec);

    Metrics.train_x_rec50 = percentileFinite(rowNorm( ...
        TrainDec - Snapshot.TrainX),50);
    Metrics.train_x_rec90 = percentileFinite(rowNorm( ...
        TrainDec - Snapshot.TrainX),90);
    trainObjDist = objectiveDistance(TrainObj,Snapshot.trainObjs, ...
        Snapshot.objMin,Snapshot.objSpan);
    Metrics.train_y_rec50 = percentileFinite(trainObjDist,50);
    Metrics.train_y_rec90 = percentileFinite(trainObjDist,90);
    queryObjDist = objectiveDistance(QueryObj,Snapshot.queryObjs, ...
        Snapshot.objMin,Snapshot.objSpan);
    Metrics.query_y_err50 = percentileFinite(queryObjDist,50);
    Metrics.query_y_err90 = percentileFinite(queryObjDist,90);

    boundaryDist = generatedBoundaryDistancesOffline(QueryObj,Snapshot);
    Metrics.boundary_dist50 = percentileFinite(boundaryDist,50);
    Metrics.boundary_dist90 = percentileFinite(boundaryDist,90);
    PairStats = pairSideStatsOffline(QueryDec,QueryInfo.query_index, ...
        Snapshot,GANOptions.pairMargin);
    Metrics.pair_violation_rate = PairStats.violation_rate;
    Metrics.pair_margin50 = PairStats.margin50;
    Metrics.pair_margin90 = PairStats.margin90;
    Metrics.side_rate = PairStats.side_rate;
    Metrics.feasible_rate = feasibleRate(QueryCon);
    Metrics.segment_width90 = generatedSegmentDistance90(QueryObj, ...
        QueryInfo.query_index,Snapshot);
    Metrics.coverage_by_ref = coverageByRef(queryObjDist,Snapshot,Options);
    Metrics.outlier_rate = meanFinite(queryObjDist > Options.outlierThreshold);
    Metrics.collapse_rate = collapseRate(QueryDec);
    Metrics.z_sensitivity = zSensitivity(QueryObj,RandomObj,Snapshot);
end

function Metrics = emptyLossMetrics()
    names = lossMetricNames();
    for i = 1 : numel(names)
        Metrics.(names{i}) = NaN;
    end
end

function Row = copyLossMetrics(Row,Metrics)
    names = lossMetricNames();
    for i = 1 : numel(names)
        Row.(names{i}) = double(Metrics.(names{i}));
    end
end

function names = lossMetricNames()
    names = {'train_x_rec50','train_x_rec90','train_y_rec50', ...
        'train_y_rec90','query_y_err50','query_y_err90', ...
        'boundary_dist50','boundary_dist90','pair_violation_rate', ...
        'pair_margin50','pair_margin90','side_rate','feasible_rate', ...
        'segment_width90','coverage_by_ref','outlier_rate', ...
        'collapse_rate','z_sensitivity'};
end

function Row = emptyLossRawRow()
    Row = struct( ...
        'problem',"", ...
        'snapshot_run',NaN, ...
        'gan_seed',NaN, ...
        'variant',"", ...
        'variant_description',"", ...
        'condition_mode',"", ...
        'snapshot_file',"", ...
        'train_count',0, ...
        'query_count',0, ...
        'condition_dim',0, ...
        'D',NaN, ...
        'advWeight',NaN, ...
        'reconstructionWeight',NaN, ...
        'pairMarginWeight',NaN, ...
        'useMismatchD',false, ...
        'runtime',NaN, ...
        'status',"pending", ...
        'error_message',"");
    Row = copyLossMetrics(Row,emptyLossMetrics());
end

function Snapshot = loadLossSnapshot(fileName)
    Loaded = load(fileName);
    if isfield(Loaded,'Snapshot')
        Snapshot = Loaded.Snapshot;
    elseif isfield(Loaded,'Data')
        Snapshot = snapshotFromCapturedData(Loaded.Data,fileName);
    else
        error('CBSCGANLossAblation:BadSnapshot', ...
            'Snapshot file must contain Snapshot or Data: %s',fileName);
    end
end

function Snapshot = snapshotFromCapturedData(Data,fileName)
    [problemName,runId] = parseProblemRunFromPath(fileName);
    BMem = Data.BMem;
    Problem = makeLossProblem(problemName,100,size(BMem.x_b,2),100000);
    W = Data.W;
    Options = struct( ...
        'conditionMode',"ref_tau", ...
        'queryConditionBudget',20, ...
        'conditionScale',Data.DatasetInfo);
    [TrainX,TrainC,QueryC,BMem,Info] = BuildBoundaryDataset_CBS( ...
        BMem,SOLUTION(BMem.x_b,BMem.y_b,zeros(size(BMem.y_b,1),1)), ...
        W,Problem,Options);
    Snapshot = makeLossSnapshot(problemName,runId,Problem,W,BMem,TrainX, ...
        TrainC,QueryC,Info,Data.GANOptions,NaN,struct( ...
        'snapshotFE',100000,'conditionMode',"ref_tau"));
end

function [problemName,runId] = parseProblemRunFromPath(fileName)
    [folder,~] = fileparts(fileName);
    [~,leaf] = fileparts(folder);
    token = regexp(leaf,'^(.*)_run(\d+)$','tokens','once');
    if isempty(token)
        problemName = "";
        runId = NaN;
    else
        problemName = string(token{1});
        runId = str2double(token{2});
    end
end

function value = getSnapshotField(S,name,defaultValue)
    if isstruct(S) && isfield(S,name) && ~isempty(S.(name))
        value = S.(name);
    else
        value = defaultValue;
    end
end

function Problem = makeLossProblem(problemName,N,D,maxFE)
    Constructor = str2func(char(problemName));
    if isempty(D) || ~isfinite(D)
        Problem = Constructor('N',N,'maxFE',maxFE);
    else
        Problem = Constructor('N',N,'D',D,'maxFE',maxFE);
    end
end

function X = rmfieldIfPresent(X,name)
    if isfield(X,name)
        X = rmfield(X,name);
    end
end

function d = rowNorm(X)
    if isempty(X)
        d = NaN(0,1);
    else
        d = sqrt(sum(double(X).^2,2));
    end
end

function Dist = objectiveDistance(Obj,Target,MinV,SpanV)
    n = min(size(Obj,1),size(Target,1));
    if n <= 0
        Dist = NaN(0,1);
        return;
    end
    ObjN = normalizeWithScale(Obj(1:n,:),MinV,SpanV);
    TargetN = normalizeWithScale(Target(1:n,:),MinV,SpanV);
    Dist = rowNorm(ObjN - TargetN);
end

function Xn = normalizeWithScale(X,MinV,SpanV)
    SpanV = double(SpanV(:)');
    MinV = double(MinV(:)');
    SpanV(SpanV <= eps) = 1;
    Xn = (double(X) - MinV)./SpanV;
    Xn(~isfinite(Xn)) = 0;
end

function Dist = generatedBoundaryDistancesOffline(GeneratedObj,Snapshot)
    BMem = Snapshot.BMem;
    if isempty(GeneratedObj) || isempty(BMem) || ~isfield(BMem,'y_b') || ...
            isempty(BMem.y_b)
        Dist = NaN(0,1);
        return;
    end
    Y = normalizeWithScale(GeneratedObj,Snapshot.objMin,Snapshot.objSpan);
    BY = normalizeWithScale(BMem.y_b,Snapshot.objMin,Snapshot.objSpan);
    Dist = inf(size(Y,1),1);
    [~,ord] = sort(BMem.ref(:));
    idx = ord(:)';
    if numel(idx) < 2
        Dist = min(pointDistance(Y,BY(idx,:)),[],2);
    else
        for k = 1 : numel(idx)-1
            Dist = min(Dist,pointSegmentDistanceRows( ...
                Y,BY(idx(k),:),BY(idx(k+1),:)));
        end
    end
end

function Stats = pairSideStatsOffline(GeneratedDec,QueryIndex,Snapshot,margin)
    Stats = struct('violation_rate',NaN,'margin50',NaN, ...
        'margin90',NaN,'side_rate',NaN);
    if isempty(GeneratedDec)
        return;
    end
    [XF,XI] = pairSupportForQueries(QueryIndex,Snapshot);
    n = min([size(GeneratedDec,1),size(XF,1),size(XI,1)]);
    if n <= 0
        return;
    end
    distF = rowNorm(GeneratedDec(1:n,:) - XF(1:n,:));
    distI = rowNorm(GeneratedDec(1:n,:) - XI(1:n,:));
    pairMargin = distI - distF;
    violation = distF - distI + double(margin) > 0;
    Stats.violation_rate = meanFinite(violation);
    Stats.margin50 = percentileFinite(pairMargin,50);
    Stats.margin90 = percentileFinite(pairMargin,90);
    Stats.side_rate = meanFinite(distF < distI);
end

function [XF,XI] = pairSupportForQueries(QueryIndex,Snapshot)
    Meta = Snapshot.queryMeta;
    if isstruct(Meta) && isfield(Meta,'x_f') && isfield(Meta,'x_i') && ...
            ~isempty(Meta.x_f) && ~isempty(QueryIndex)
        q = round(double(QueryIndex(:)));
        valid = isfinite(q) & q >= 1 & q <= size(Meta.x_f,1);
        XF = NaN(numel(q),size(Meta.x_f,2));
        XI = NaN(numel(q),size(Meta.x_i,2));
        XF(valid,:) = Meta.x_f(q(valid),:);
        XI(valid,:) = Meta.x_i(q(valid),:);
        keep = all(isfinite(XF),2) & all(isfinite(XI),2);
        XF = XF(keep,:);
        XI = XI(keep,:);
    else
        XF = Snapshot.trainXf;
        XI = Snapshot.trainXi;
    end
end

function Width = generatedSegmentDistance90(GeneratedObj,QueryIndex,Snapshot)
    Width = NaN;
    BMem = Snapshot.BMem;
    Meta = Snapshot.queryMeta;
    if isempty(GeneratedObj) || isempty(QueryIndex) || isempty(BMem) || ...
            ~isfield(BMem,'ref') || ~isfield(BMem,'y_b') || ...
            ~isstruct(Meta) || ~isfield(Meta,'source_interval')
        return;
    end
    ObjN = normalizeWithScale(GeneratedObj,Snapshot.objMin, ...
        Snapshot.objSpan);
    BObjN = normalizeWithScale(BMem.y_b,Snapshot.objMin, ...
        Snapshot.objSpan);
    QueryIndex = round(double(QueryIndex(:)));
    n = min(size(ObjN,1),numel(QueryIndex));
    dist = NaN(n,1);
    for row = 1 : n
        q = QueryIndex(row);
        if ~isfinite(q) || q < 1 || q > size(Meta.source_interval,1)
            continue;
        end
        interval = Meta.source_interval(q,:);
        [a,b] = sourceSegmentRows(BMem,interval);
        if isempty(a)
            continue;
        end
        if isempty(b)
            dist(row) = min(pointDistance(ObjN(row,:),BObjN(a,:)),[],2);
        else
            dist(row) = pointSegmentDistanceRows(ObjN(row,:), ...
                BObjN(a,:),BObjN(b,:));
        end
    end
    Width = percentileFinite(dist,90);
end

function [a,b] = sourceSegmentRows(BMem,interval)
    a = [];
    b = [];
    if isempty(interval) || numel(interval) < 2 || any(~isfinite(interval))
        return;
    end
    r1 = round(double(interval(1)));
    r2 = round(double(interval(2)));
    a = find(BMem.ref(:) == r1,1,'first');
    b = find(BMem.ref(:) == r2,1,'first');
    if isempty(a) && ~isempty(BMem.ref)
        [~,a] = min(abs(double(BMem.ref(:)) - r1));
    end
    if isempty(b) && ~isempty(BMem.ref)
        [~,b] = min(abs(double(BMem.ref(:)) - r2));
    end
end

function Cover = coverageByRef(queryObjDist,Snapshot,Options)
    Meta = Snapshot.queryMeta;
    if isempty(queryObjDist) || ~isstruct(Meta) || ~isfield(Meta,'ref') || ...
            isempty(Meta.ref)
        Cover = NaN;
        return;
    end
    refs = round(double(Meta.ref(:)));
    n = min(numel(queryObjDist),numel(refs));
    refs = refs(1:n);
    success = isfinite(queryObjDist(1:n)) & ...
        queryObjDist(1:n) <= Options.coverageTolerance;
    validRefs = unique(refs(isfinite(refs) & refs > 0));
    if isempty(validRefs)
        Cover = NaN;
    else
        coveredRefs = unique(refs(success & isfinite(refs) & refs > 0));
        Cover = numel(coveredRefs)/numel(validRefs);
    end
end

function Rate = collapseRate(Dec)
    if isempty(Dec)
        Rate = NaN;
        return;
    end
    Rounded = round(double(Dec)*1e6)/1e6;
    uniqueCount = size(unique(Rounded,'rows'),1);
    Rate = 1 - uniqueCount/size(Dec,1);
end

function value = zSensitivity(QueryObj,RandomObj,Snapshot)
    n = min(size(QueryObj,1),size(RandomObj,1));
    if n <= 0
        value = NaN;
        return;
    end
    Q = normalizeWithScale(QueryObj(1:n,:),Snapshot.objMin, ...
        Snapshot.objSpan);
    R = normalizeWithScale(RandomObj(1:n,:),Snapshot.objMin, ...
        Snapshot.objSpan);
    value = percentileFinite(rowNorm(R - Q),50);
end

function Rate = feasibleRate(Con)
    if isempty(Con)
        Rate = 1;
    else
        Rate = meanFinite(sum(max(0,Con),2) <= 0);
    end
end

function value = meanFinite(x)
    x = double(x(:));
    x = x(isfinite(x));
    if isempty(x)
        value = NaN;
    else
        value = mean(x);
    end
end

function q = percentileFinite(X,p)
    X = double(X(:));
    X = X(isfinite(X));
    if isempty(X)
        q = NaN;
    else
        q = prctile(X,p);
    end
end

function D = pointDistance(A,B)
    if isempty(A) || isempty(B)
        D = NaN(size(A,1),size(B,1));
        return;
    end
    AA = sum(A.^2,2);
    BB = sum(B.^2,2)';
    D = sqrt(max(AA + BB - 2*(A*B'),0));
end

function D = pointSegmentDistanceRows(P,A,B)
    AB = B - A;
    denom = sum(AB.^2);
    if denom <= eps
        D = rowNorm(P - A);
        return;
    end
    T = ((P - A)*AB')/denom;
    T = max(0,min(1,T));
    Projection = A + T.*AB;
    D = rowNorm(P - Projection);
end

function T = aggregateLossMetrics(Raw,groupNames)
    if isempty(Raw) || height(Raw) == 0
        T = table();
        return;
    end
    Ok = Raw(string(Raw.status) == "ok",:);
    if isempty(Ok) || height(Ok) == 0
        T = table();
        return;
    end
    metrics = lossMetricNames();
    metrics = metrics(ismember(metrics,Ok.Properties.VariableNames));
    T = groupsummary(Ok,groupNames,'median',metrics);
end

function Summary = buildLossComparisonSummary(Raw)
    if isempty(Raw) || height(Raw) == 0
        Summary = table();
        return;
    end
    Ok = Raw(string(Raw.status) == "ok",:);
    if isempty(Ok) || height(Ok) == 0
        Summary = table();
        return;
    end
    comparisons = ["V2_Adv_Huber","V1_CGAN_mismatchD","Huber_vs_mismatch"; ...
        "V3_Adv_Pair","V1_CGAN_mismatchD","Pair_vs_mismatch"; ...
        "V4_Full_current","V2_Adv_Huber","Full_vs_Huber"; ...
        "V4_Full_current","V3_Adv_Pair","Full_vs_Pair"; ...
        "V4_Full_current","V5_Huber_Pair_no_adv","Adversarial_effect"; ...
        "V1_CGAN_mismatchD","V0_Pure_CGAN","MismatchD_effect"];
    metrics = ["train_x_rec90","train_y_rec90","query_y_err90", ...
        "boundary_dist90","pair_violation_rate","side_rate", ...
        "feasible_rate","z_sensitivity"];
    Rows = repmat(struct('comparison',"",'variant_to',"", ...
        'variant_from',"",'metric',"",'matched_count',0, ...
        'median_to',NaN,'median_from',NaN,'median_delta',NaN), ...
        size(comparisons,1)*numel(metrics),1);
    row = 0;
    for c = 1 : size(comparisons,1)
        for m = 1 : numel(metrics)
            row = row + 1;
            Rows(row) = pairedMetricDelta(Ok,comparisons(c,1), ...
                comparisons(c,2),comparisons(c,3),metrics(m));
        end
    end
    Summary = struct2table(Rows);
end

function Row = pairedMetricDelta(T,toVariant,fromVariant,label,metric)
    Row = struct('comparison',string(label),'variant_to',string(toVariant), ...
        'variant_from',string(fromVariant),'metric',string(metric), ...
        'matched_count',0,'median_to',NaN,'median_from',NaN, ...
        'median_delta',NaN);
    keys = unique(T(:,{'problem','snapshot_run','gan_seed'}),'rows','stable');
    toVals = NaN(height(keys),1);
    fromVals = NaN(height(keys),1);
    keep = false(height(keys),1);
    for i = 1 : height(keys)
        same = string(T.problem) == string(keys.problem(i)) & ...
            double(T.snapshot_run) == double(keys.snapshot_run(i)) & ...
            double(T.gan_seed) == double(keys.gan_seed(i));
        toIdx = same & string(T.variant) == string(toVariant);
        fromIdx = same & string(T.variant) == string(fromVariant);
        if any(toIdx) && any(fromIdx)
            toVals(i) = double(T.(metric)(find(toIdx,1,'first')));
            fromVals(i) = double(T.(metric)(find(fromIdx,1,'first')));
            keep(i) = isfinite(toVals(i)) && isfinite(fromVals(i));
        end
    end
    toVals = toVals(keep);
    fromVals = fromVals(keep);
    Row.matched_count = numel(toVals);
    Row.median_to = percentileFinite(toVals,50);
    Row.median_from = percentileFinite(fromVals,50);
    Row.median_delta = percentileFinite(toVals - fromVals,50);
end

function Manifest = buildLossManifest(Variants,Options,SnapshotManifest)
    Rows = repmat(struct('entry_type',"variant",'variant',"", ...
        'description',"",'advWeight',NaN,'reconstructionWeight',NaN, ...
        'pairMarginWeight',NaN,'useMismatchD',false, ...
        'trainZMode',Options.trainZMode,'condition_mode', ...
        Options.conditionMode,'workerCount',Options.workerCount, ...
        'problem_count',numel(Options.problemNames), ...
        'snapshot_count',height(SnapshotManifest), ...
        'snapshot_file',""),numel(Variants),1);
    for i = 1 : numel(Variants)
        Rows(i).variant = string(Variants(i).variant);
        Rows(i).description = string(Variants(i).description);
        Rows(i).advWeight = double(Variants(i).advWeight);
        Rows(i).reconstructionWeight = double(Variants(i).reconstructionWeight);
        Rows(i).pairMarginWeight = double(Variants(i).pairMarginWeight);
        Rows(i).useMismatchD = logical(Variants(i).useMismatchD);
    end
    Manifest = struct2table(Rows);
end

function ensureLossParallelPool(workerCount)
    pool = gcp('nocreate');
    if isempty(pool) || pool.NumWorkers ~= workerCount
        if ~isempty(pool)
            delete(pool);
        end
        parpool('local',workerCount);
    end
end
