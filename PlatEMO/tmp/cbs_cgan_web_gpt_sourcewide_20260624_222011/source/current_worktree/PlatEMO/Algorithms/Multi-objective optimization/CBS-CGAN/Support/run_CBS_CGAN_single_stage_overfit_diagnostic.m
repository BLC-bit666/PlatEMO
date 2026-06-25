function [outDir,Metrics,Snapshot,Trace] = ...
    run_CBS_CGAN_single_stage_overfit_diagnostic(outDir,Options)
%RUN_CBS_CGAN_SINGLE_STAGE_OVERFIT_DIAGNOSTIC Diagnose one-stage train fit.
%   Captures one CBS-CGAN boundary dataset snapshot, then freezes TrainX/TrainC
%   and trains G(0,C)->X offline with adversarial loss disabled.

    rootDir = fileparts(which('platemo'));
    if isempty(rootDir)
        rootDir = pwd;
    end
    addpath(genpath(rootDir));
    if nargin < 1 || isempty(outDir)
        outDir = fullfile(rootDir,'Data','CBS_CGAN', ...
            ['single_stage_overfit_endpoint_yb_norm_', ...
            char(datetime('now','Format','yyyyMMdd_HHmmss'))]);
    end
    if nargin < 2 || isempty(Options)
        Options = struct();
    end
    Options = normalizeOptions(Options);
    if ~isfolder(outDir)
        mkdir(outDir);
    end

    rng(Options.snapshotSeed,'twister');
    Problem = makeProblem(Options.problemName,Options.N,Options.D, ...
        Options.maxFE);
    Snapshot = captureSnapshot(Problem,Options);
    if isempty(Snapshot.TrainX) || isempty(Snapshot.TrainC)
        error('CBSSingleStageOverfit:EmptySnapshot', ...
            'Captured snapshot has empty TrainX or TrainC.');
    end

    snapshotFile = fullfile(outDir,sprintf('%s_run%d_targetFE%06d_snapshot.mat', ...
        char(Options.problemName),Options.runId,Options.snapshotFE));
    save(snapshotFile,'Snapshot','Options','-v7.3');

    Variants = overfitVariants(Options);
    Rows = repmat(emptyMetricRow(),numel(Variants),1);
    TraceRows = emptyMetricRow();
    TraceRows = TraceRows([]);
    Recon = cell(numel(Variants),1);
    for i = 1 : numel(Variants)
        rng(Options.ganSeed,'twister');
        [Rows(i),Recon{i},VariantTrace] = runVariant( ...
            Snapshot,Variants(i),Options, ...
            snapshotFile);
        TraceRows = [TraceRows;VariantTrace(:)]; %#ok<AGROW>
        fprintf('[overfit %d/%d] %s status=%s train_y_rec90_norm=%.6g train_x_rec90=%.6g\n', ...
            i,numel(Variants),Rows(i).variant,Rows(i).status, ...
            Rows(i).train_y_rec90_norm,Rows(i).train_x_rec90);
    end
    Metrics = struct2table(Rows);
    Trace = struct2table(TraceRows);
    metricsFile = fullfile(outDir,'single_stage_overfit_metrics.csv');
    writetable(Metrics,metricsFile);
    writetable(Trace,fullfile(outDir,'single_stage_overfit_trace.csv'));
    writeSnapshotSummary(outDir,Snapshot,Options,snapshotFile);
    writeTraceFigure(outDir,Trace,Options);
    writeReconstructionFigure(outDir,Snapshot,Variants,Recon,Options);
end

function Options = normalizeOptions(Options)
    Options = ensureField(Options,'problemName',"LIRCMOP7_BC");
    Options = ensureField(Options,'runId',1);
    Options = ensureField(Options,'snapshotSeed',Options.runId);
    Options = ensureField(Options,'ganSeed',1);
    Options = ensureField(Options,'N',100);
    Options = ensureField(Options,'D',[]);
    Options = ensureField(Options,'maxFE',100000);
    Options = ensureField(Options,'snapshotFE',50000);
    Options = ensureField(Options,'conditionMode',"ref_y");
    Options = ensureField(Options,'boundaryTargetMode',"feasible_endpoint");
    Options = ensureField(Options,'pairNeighborRefRadius',4);
    Options = ensureField(Options,'maxCandidatePairsPerRef',3);
    Options = ensureField(Options,'minBoundaryLength',2);
    Options = ensureField(Options,'nGen',30);
    Options = ensureField(Options,'queryPerCondition',1);
    Options = ensureField(Options,'zDim',2);
    Options = ensureField(Options,'ganMiniBatch',32);
    Options = ensureField(Options,'ganLrD',1e-4);
    Options = ensureField(Options,'ganLrG',2e-4);
    Options = ensureField(Options,'ganDPretrainIter',0);
    Options = ensureField(Options,'ganDSteps',1);
    Options = ensureField(Options,'ganGSteps',1);
    Options = ensureField(Options,'sigma',0.05);
    Options = ensureField(Options,'trainZMode',"zero");
    Options = ensureField(Options,'sampleZMode',"zero");
    Options = ensureField(Options,'onlineGanIter',50);
    Options = ensureField(Options,'onlineTrainMode',"epoch");
    Options = ensureField(Options,'onlineAdvWeight',1);
    Options = ensureField(Options,'onlineReconstructionWeight',1);
    Options = ensureField(Options,'reconstructionHuberDelta',0.10);
    Options = ensureField(Options,'networkPreset',"default");
    Options = ensureField(Options,'generatorHidden',[64 64 64]);
    Options = ensureField(Options,'discriminatorHidden',[64 64 32]);
    Options = ensureField(Options,'overfitMode',"standard");
    Options = ensureField(Options,'traceCheckpoints', ...
        [1 2 5 10 20 50 100 200 500 1000]);

    Options.problemName = string(Options.problemName);
    Options.runId = round(double(Options.runId));
    Options.snapshotSeed = round(double(Options.snapshotSeed));
    Options.ganSeed = round(double(Options.ganSeed));
    Options.N = max(1,round(double(Options.N)));
    if ~isempty(Options.D)
        Options.D = max(1,round(double(Options.D)));
    end
    Options.maxFE = max(1,round(double(Options.maxFE)));
    Options.snapshotFE = max(1,round(double(Options.snapshotFE)));
    Options.conditionMode = lower(strtrim(string(Options.conditionMode)));
    Options.boundaryTargetMode = lower(strtrim(string(Options.boundaryTargetMode)));
    Options.pairNeighborRefRadius = max(0,round(double( ...
        Options.pairNeighborRefRadius)));
    Options.maxCandidatePairsPerRef = max(1,round(double( ...
        Options.maxCandidatePairsPerRef)));
    Options.minBoundaryLength = max(1,round(double(Options.minBoundaryLength)));
    Options.nGen = max(0,round(double(Options.nGen)));
    Options.queryPerCondition = max(1,round(double(Options.queryPerCondition)));
    Options.zDim = max(0,round(double(Options.zDim)));
    Options.ganMiniBatch = max(1,round(double(Options.ganMiniBatch)));
    Options.ganDPretrainIter = max(0,round(double(Options.ganDPretrainIter)));
    Options.ganDSteps = max(1,round(double(Options.ganDSteps)));
    Options.ganGSteps = max(1,round(double(Options.ganGSteps)));
    Options.trainZMode = lower(strtrim(string(Options.trainZMode)));
    Options.sampleZMode = lower(strtrim(string(Options.sampleZMode)));
    Options.onlineGanIter = max(0,round(double(Options.onlineGanIter)));
    Options.onlineTrainMode = lower(strtrim(string(Options.onlineTrainMode)));
    Options.overfitMode = lower(strtrim(string(Options.overfitMode)));
    Options.onlineAdvWeight = max(0,double(Options.onlineAdvWeight));
    Options.onlineReconstructionWeight = max(0,double( ...
        Options.onlineReconstructionWeight));
    Options.generatorHidden = double(Options.generatorHidden(:)');
    Options.discriminatorHidden = double(Options.discriminatorHidden(:)');
    Options.traceCheckpoints = unique(max(1,round(double( ...
        Options.traceCheckpoints(:)'))),'stable');
end

function S = ensureField(S,name,value)
    if ~isfield(S,name) || isempty(S.(name))
        S.(name) = value;
    end
end

function Problem = makeProblem(problemName,N,D,maxFE)
    Constructor = str2func(char(problemName));
    if isempty(D)
        Problem = Constructor('N',N,'maxFE',maxFE);
    else
        Problem = Constructor('N',N,'D',D,'maxFE',maxFE);
    end
end

function Snapshot = captureSnapshot(Problem,Options)
    [W,~] = UniformPoint(Problem.N,Problem.M);
    BoundaryOptions = struct( ...
        'pairNeighborRefRadius',Options.pairNeighborRefRadius, ...
        'maxCandidatePairsPerRef',Options.maxCandidatePairsPerRef, ...
        'maxCanonicalPairsPerRef',Options.maxCandidatePairsPerRef, ...
        'minBoundaryLength',Options.minBoundaryLength, ...
        'boundaryTargetMode',Options.boundaryTargetMode);
    DatasetOptions = struct( ...
        'pairNeighborRefRadius',Options.pairNeighborRefRadius, ...
        'queryConditionBudget',max(1,ceil(max(Options.nGen,1)/ ...
            Options.queryPerCondition)), ...
        'conditionMode',Options.conditionMode);
    GANOptions = baseGANOptions(Options,Options.onlineGanIter, ...
        Options.onlineAdvWeight,Options.onlineReconstructionWeight, ...
        Options.onlineTrainMode);

    Population1 = Problem.Initialization();
    Population2 = Problem.Initialization();
    DatasetOptions.conditionScale = initializeConditionScale( ...
        [Population1,Population2],Problem);
    Fitness1 = CalFitness_CBS(Population1.objs,Population1.cons);
    Fitness2 = CalFitness_CBS(Population2.objs);
    BMem = [];
    GAN = [];
    gen = 0;
    Snapshot = emptySnapshot(Problem,W,Options);

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

        [BMem,BoundaryDiag] = UpdateBoundaryMemory_CBS(BMem, ...
            Population1,Offspring1,Population2,Offspring2,W, ...
            BoundaryOptions);
        OffspringG = Offspring1([]);
        DatasetInfo = struct();
        TrainX = zeros(0,Problem.D);
        TrainC = zeros(0,0);
        QueryC = zeros(0,0);
        if ~isempty(BMem) && Options.nGen > 0
            [TrainX,TrainC,QueryC,BMem,DatasetInfo] = ...
                BuildBoundaryDataset_CBS(BMem, ...
                [Population1,Offspring1,Population2,Offspring2], ...
                W,Problem,DatasetOptions);
            if size(TrainX,1) >= Options.minBoundaryLength && ...
                    ~isempty(TrainC) && ~isempty(QueryC)
                GAN = BoundaryCGAN_CBS('train',GAN,TrainX,TrainC, ...
                    Problem,GANOptions);
                SampleOptions = sampleOptions(GANOptions, ...
                    size(QueryC,1)*Options.queryPerCondition);
                [RawDec,~] = BoundaryCGAN_CBS('samplebycondition',GAN, ...
                    QueryC,Options.queryPerCondition,SampleOptions);
                if size(RawDec,1) > Options.nGen
                    RawDec = RawDec(1:Options.nGen,:);
                end
                if ~isempty(RawDec)
                    OffspringG = Problem.Evaluation(RawDec);
                end
                Snapshot = makeSnapshot(Problem,W,BMem,TrainX,TrainC, ...
                    QueryC,DatasetInfo,GANOptions,gen,BoundaryDiag,Options);
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

function Options = baseGANOptions(RunnerOptions,iter,advWeight,recWeight,trainMode)
    Options = struct( ...
        'zDim',RunnerOptions.zDim, ...
        'iter',iter, ...
        'miniBatch',RunnerOptions.ganMiniBatch, ...
        'lrD',double(RunnerOptions.ganLrD), ...
        'lrG',double(RunnerOptions.ganLrG), ...
        'dPretrainIter',RunnerOptions.ganDPretrainIter, ...
        'dSteps',RunnerOptions.ganDSteps, ...
        'gSteps',RunnerOptions.ganGSteps, ...
        'sigma',double(RunnerOptions.sigma), ...
        'reconstructionHuberDelta',double( ...
            RunnerOptions.reconstructionHuberDelta), ...
        'reconstructionWeight',double(recWeight), ...
        'advWeight',double(advWeight), ...
        'trainZMode',RunnerOptions.trainZMode, ...
        'sampleZMode',RunnerOptions.sampleZMode, ...
        'trainMode',trainMode, ...
        'generatorHidden',RunnerOptions.generatorHidden, ...
        'discriminatorHidden',RunnerOptions.discriminatorHidden);
end

function Options = sampleOptions(Options,n)
    Options.sampleZMode = "zero";
    Options.sampleZ = zeros(max(0,round(double(n))),Options.zDim);
end

function Scale = initializeConditionScale(Population,Problem)
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

function Snapshot = emptySnapshot(Problem,W,Options)
    Snapshot = struct( ...
        'problem',Options.problemName, ...
        'run',double(Options.runId), ...
        'seed',double(Options.snapshotSeed), ...
        'N',double(Problem.N), ...
        'D',double(Problem.D), ...
        'M',double(Problem.M), ...
        'maxFE',double(Options.maxFE), ...
        'targetFE',double(Options.snapshotFE), ...
        'actualFE',double(Problem.FE), ...
        'gen',0, ...
        'conditionMode',Options.conditionMode, ...
        'boundaryTargetMode',Options.boundaryTargetMode, ...
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
        'trainRef',zeros(0,1), ...
        'trainTau',zeros(0,1), ...
        'queryMeta',struct(), ...
        'objMin',zeros(1,Problem.M), ...
        'objSpan',ones(1,Problem.M), ...
        'GANOptions',baseGANOptions(Options,Options.onlineGanIter, ...
            Options.onlineAdvWeight,Options.onlineReconstructionWeight, ...
            Options.onlineTrainMode), ...
        'BoundaryDiag',struct());
end

function Snapshot = makeSnapshot(Problem,W,BMem,TrainX,TrainC,QueryC, ...
        DatasetInfo,GANOptions,gen,BoundaryDiag,Options)
    Snapshot = emptySnapshot(Problem,W,Options);
    Snapshot.actualFE = double(Problem.FE);
    Snapshot.gen = double(gen);
    Snapshot.BMem = BMem;
    Snapshot.TrainX = TrainX;
    Snapshot.TrainC = TrainC;
    Snapshot.QueryC = QueryC;
    Snapshot.GANOptions = GANOptions;
    Snapshot.BoundaryDiag = BoundaryDiag;
    if isfield(DatasetInfo,'trainObjs'); Snapshot.trainObjs = DatasetInfo.trainObjs; end
    if isfield(DatasetInfo,'queryObjs'); Snapshot.queryObjs = DatasetInfo.queryObjs; end
    if isfield(DatasetInfo,'trainXf'); Snapshot.trainXf = DatasetInfo.trainXf; end
    if isfield(DatasetInfo,'trainXi'); Snapshot.trainXi = DatasetInfo.trainXi; end
    if isfield(DatasetInfo,'trainYf'); Snapshot.trainYf = DatasetInfo.trainYf; end
    if isfield(DatasetInfo,'trainYi'); Snapshot.trainYi = DatasetInfo.trainYi; end
    if isfield(DatasetInfo,'trainRef'); Snapshot.trainRef = DatasetInfo.trainRef; end
    if isfield(DatasetInfo,'trainTau'); Snapshot.trainTau = DatasetInfo.trainTau; end
    if isfield(DatasetInfo,'queryMeta'); Snapshot.queryMeta = DatasetInfo.queryMeta; end
    if isfield(DatasetInfo,'objMin'); Snapshot.objMin = DatasetInfo.objMin; end
    if isfield(DatasetInfo,'objSpan'); Snapshot.objSpan = DatasetInfo.objSpan; end
end

function Variants = overfitVariants(Options)
    switch Options.overfitMode
        case "adv_only"
            maxEpoch = max(Options.traceCheckpoints);
            Variants = struct( ...
                'variant',sprintf("adv_only_epoch%d_no_huber",maxEpoch), ...
                'advWeight',1, ...
                'reconstructionWeight',0, ...
                'iter',maxEpoch, ...
                'trainMode',"epoch", ...
                'huberDelta',Options.reconstructionHuberDelta);
            return;
        case "standard"
        otherwise
            error('CBSSingleStageOverfit:BadOverfitMode', ...
                'Unsupported overfitMode: %s.',Options.overfitMode);
    end
    Variants = struct( ...
        'variant',{"online_like_adv_huber_epoch50", ...
            "supervised_huber_epoch50", ...
            "supervised_huber_epoch200", ...
            "supervised_huber_epoch1000"}, ...
        'advWeight',{1,0,0,0}, ...
        'reconstructionWeight',{1,1,1,1}, ...
        'iter',{50,50,200,1000}, ...
        'trainMode',{"epoch","epoch","epoch","epoch"}, ...
        'huberDelta',{Options.reconstructionHuberDelta, ...
            Options.reconstructionHuberDelta, ...
            Options.reconstructionHuberDelta, ...
            Options.reconstructionHuberDelta});
end

function [Row,Recon,TraceRows] = runVariant( ...
        Snapshot,Variant,RunnerOptions,snapshotFile)
    Row = emptyMetricRow();
    Row.problem = string(Snapshot.problem);
    Row.run = double(Snapshot.run);
    Row.targetFE = double(Snapshot.targetFE);
    Row.actualFE = double(Snapshot.actualFE);
    Row.gen = double(Snapshot.gen);
    Row.variant = string(Variant.variant);
    Row.condition_mode = string(Snapshot.conditionMode);
    Row.boundaryTargetMode = string(Snapshot.boundaryTargetMode);
    Row.snapshot_file = string(snapshotFile);
    Row.train_count = size(Snapshot.TrainX,1);
    Row.query_count = size(Snapshot.QueryC,1);
    Row.condition_dim = size(Snapshot.TrainC,2);
    Row.advWeight = double(Variant.advWeight);
    Row.reconstructionWeight = double(Variant.reconstructionWeight);
    Row.huberDelta = double(Variant.huberDelta);
    Row.iter = double(Variant.iter);
    Row.checkpoint_epoch = double(Variant.iter);
    Row.chunk_epoch = double(Variant.iter);
    Row.trainMode = string(Variant.trainMode);
    Row.miniBatch = double(RunnerOptions.ganMiniBatch);
    Row.g_updates = updateCount(Row.train_count,Row.miniBatch, ...
        Row.iter,RunnerOptions.ganGSteps,Row.trainMode);
    if Row.advWeight > 0
        Row.d_updates = updateCount(Row.train_count,Row.miniBatch, ...
            Row.iter,RunnerOptions.ganDSteps,Row.trainMode) + ...
            RunnerOptions.ganDPretrainIter;
    else
        Row.d_updates = 0;
    end
    Row.condition_unique_count = uniqueRowCount(Snapshot.TrainC,1e-12);
    Row.condition_duplicate_count = Row.train_count - ...
        Row.condition_unique_count;
    Row.condition_nn_dist50 = conditionNearestDistance(Snapshot.TrainC,50);
    Row.condition_nn_dist90 = conditionNearestDistance(Snapshot.TrainC,90);
    Row.train_tau_iqr = iqrFinite(Snapshot.trainTau);
    Row.train_tau_range = rangeFinite(Snapshot.trainTau);
    Row.train_tau_nonzero_rate = mean(abs(Snapshot.trainTau) > 1e-12);
    Row.train_pair_dec_dist50 = percentileFinite(rowNorm( ...
        Snapshot.trainXf - Snapshot.trainXi),50);
    Row.train_pair_dec_dist90 = percentileFinite(rowNorm( ...
        Snapshot.trainXf - Snapshot.trainXi),90);

    TraceRows = emptyMetricRow();
    TraceRows = TraceRows([]);
    try
        Problem = makeProblem(Snapshot.problem,Snapshot.N,Snapshot.D, ...
            Snapshot.maxFE);
        checkpoints = overfitCheckpoints(Variant.iter, ...
            RunnerOptions.traceCheckpoints);
        TraceRows = repmat(emptyMetricRow(),numel(checkpoints),1);
        GAN = [];
        cumulativeEpoch = 0;
        tStart = tic;
        for c = 1 : numel(checkpoints)
            chunkEpoch = checkpoints(c) - cumulativeEpoch;
            GANOptions = baseGANOptions(RunnerOptions,chunkEpoch, ...
                Variant.advWeight,Variant.reconstructionWeight, ...
                Variant.trainMode);
            GANOptions.reconstructionHuberDelta = double(Variant.huberDelta);
            if chunkEpoch > 0
                GAN = BoundaryCGAN_CBS('train',GAN,Snapshot.TrainX, ...
                    Snapshot.TrainC,Problem,GANOptions);
            end
            cumulativeEpoch = checkpoints(c);
            TrainOptions = sampleOptions(GANOptions,size(Snapshot.TrainC,1));
            [TrainDec,~] = BoundaryCGAN_CBS('samplebycondition',GAN, ...
                Snapshot.TrainC,1,TrainOptions);
            [TrainObj,~] = EvaluateDecisions_CBS(Problem,TrainDec);

            TraceRows(c) = Row;
            TraceRows(c).iter = double(cumulativeEpoch);
            TraceRows(c).checkpoint_epoch = double(cumulativeEpoch);
            TraceRows(c).chunk_epoch = double(chunkEpoch);
            TraceRows(c).g_updates = updateCount(Row.train_count, ...
                Row.miniBatch,cumulativeEpoch,RunnerOptions.ganGSteps, ...
                Row.trainMode);
            if Row.advWeight > 0
                TraceRows(c).d_updates = updateCount(Row.train_count, ...
                    Row.miniBatch,cumulativeEpoch, ...
                    RunnerOptions.ganDSteps,Row.trainMode) + ...
                    RunnerOptions.ganDPretrainIter;
            else
                TraceRows(c).d_updates = 0;
            end
            TraceRows(c).runtime_sec = toc(tStart);
            TraceRows(c) = addFitMetrics(TraceRows(c),Snapshot, ...
                TrainDec,TrainObj);
            TraceRows(c).status = "ok";
        end
        Row = TraceRows(end);
        Recon = struct('dec',TrainDec,'obj',TrainObj);
    catch err
        Recon = struct('dec',zeros(0,size(Snapshot.TrainX,2)), ...
            'obj',zeros(0,size(Snapshot.trainObjs,2)));
        Row.status = "failed";
        Row.error_message = string(getReport(err,'extended', ...
            'hyperlinks','off'));
        if isempty(TraceRows)
            TraceRows = Row;
        end
    end
end

function checkpoints = overfitCheckpoints(maxIter,requested)
    maxIter = max(1,round(double(maxIter)));
    requested = unique(max(1,round(double(requested(:)'))),'stable');
    checkpoints = requested(requested <= maxIter);
    checkpoints = unique([checkpoints,maxIter],'stable');
end

function Row = addFitMetrics(Row,Snapshot,TrainDec,TrainObj)
    xDist = rowNorm(TrainDec - Snapshot.TrainX);
    yRawDist = rowNorm(TrainObj - Snapshot.trainObjs);
    yNormDist = objectiveDistance(TrainObj,Snapshot.trainObjs, ...
        Snapshot.objMin,Snapshot.objSpan);
    Row.train_x_rec50 = percentileFinite(xDist,50);
    Row.train_x_rec90 = percentileFinite(xDist,90);
    Row.train_x_recmax = maxFinite(xDist);
    Row.train_y_rec50_raw = percentileFinite(yRawDist,50);
    Row.train_y_rec90_raw = percentileFinite(yRawDist,90);
    Row.train_y_recmax_raw = maxFinite(yRawDist);
    Row.train_y_rec50_norm = percentileFinite(yNormDist,50);
    Row.train_y_rec90_norm = percentileFinite(yNormDist,90);
    Row.train_y_recmax_norm = maxFinite(yNormDist);
end

function n = updateCount(trainCount,batchSize,iter,steps,mode)
    if trainCount <= 0
        n = 0;
        return;
    end
    if string(mode) == "epoch"
        batches = ceil(double(trainCount)/min(double(batchSize), ...
            double(trainCount)));
        n = batches*double(iter)*double(steps);
    else
        n = double(iter)*double(steps);
    end
end

function n = uniqueRowCount(X,tol)
    if isempty(X)
        n = 0;
        return;
    end
    Q = round(double(X)./tol).*tol;
    n = size(unique(Q,'rows'),1);
end

function q = conditionNearestDistance(C,p)
    if size(C,1) < 2
        q = NaN;
        return;
    end
    D = pairDistance(double(C),double(C));
    D(1:size(D,1)+1:end) = inf;
    q = percentileFinite(min(D,[],2),p);
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

function D = pairDistance(A,B)
    if isempty(A) || isempty(B)
        D = NaN(size(A,1),size(B,1));
        return;
    end
    AA = sum(A.^2,2);
    BB = sum(B.^2,2)';
    D = sqrt(max(AA + BB - 2*(A*B'),0));
end

function d = rowNorm(X)
    if isempty(X)
        d = NaN(0,1);
    else
        d = sqrt(sum(double(X).^2,2));
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

function v = maxFinite(X)
    X = double(X(:));
    X = X(isfinite(X));
    if isempty(X)
        v = NaN;
    else
        v = max(X);
    end
end

function v = iqrFinite(X)
    X = double(X(:));
    X = X(isfinite(X));
    if isempty(X)
        v = NaN;
    else
        v = iqr(X);
    end
end

function v = rangeFinite(X)
    X = double(X(:));
    X = X(isfinite(X));
    if isempty(X)
        v = NaN;
    else
        v = max(X) - min(X);
    end
end

function v = minOrNaN(X)
    X = double(X(:));
    X = X(isfinite(X));
    if isempty(X)
        v = NaN;
    else
        v = min(X);
    end
end

function v = maxOrNaN(X)
    X = double(X(:));
    X = X(isfinite(X));
    if isempty(X)
        v = NaN;
    else
        v = max(X);
    end
end

function r = safeRatio(num,den)
    den = double(den);
    if den <= 0 || ~isfinite(den)
        r = NaN;
    else
        r = double(num)./den;
    end
end

function V = rangeRows(X)
    X = double(X);
    if isempty(X)
        V = NaN(0,1);
        return;
    end
    V = NaN(1,size(X,2));
    for j = 1 : size(X,2)
        Col = X(:,j);
        Col = Col(isfinite(Col));
        if ~isempty(Col)
            V(j) = max(Col) - min(Col);
        end
    end
    V = V(:);
end

function v = columnSpan(X,col)
    X = double(X);
    if isempty(X) || col > size(X,2)
        v = NaN;
        return;
    end
    v = rangeFinite(X(:,col));
end

function v = columnMin(X,col)
    X = double(X);
    if isempty(X) || col > size(X,2)
        v = NaN;
        return;
    end
    v = minOrNaN(X(:,col));
end

function v = columnMax(X,col)
    X = double(X);
    if isempty(X) || col > size(X,2)
        v = NaN;
        return;
    end
    v = maxOrNaN(X(:,col));
end

function q = nearestRowDistance(X,p)
    if size(X,1) < 2
        q = NaN;
        return;
    end
    X = double(X);
    valid = all(isfinite(X),2);
    X = X(valid,:);
    if size(X,1) < 2
        q = NaN;
        return;
    end
    D = pairDistance(X,X);
    D(1:size(D,1)+1:end) = inf;
    q = percentileFinite(min(D,[],2),p);
end

function n = bmemCount(BMem)
    if isempty(BMem) || ~isstruct(BMem) || ~isfield(BMem,'x_b')
        n = 0;
    else
        n = size(BMem.x_b,1);
    end
end

function S = conditionGroupMetrics(C,X,Y,objMin,objSpan,tol)
    S = struct( ...
        'duplicateGroupCount',0, ...
        'maxGroupCount',0, ...
        'xSpread50',NaN, ...
        'xSpread90',NaN, ...
        'xSpreadMax',NaN, ...
        'ySpread50Raw',NaN, ...
        'ySpread90Raw',NaN, ...
        'ySpreadMaxRaw',NaN, ...
        'ySpread50Norm',NaN, ...
        'ySpread90Norm',NaN, ...
        'ySpreadMaxNorm',NaN);
    if isempty(C)
        return;
    end
    Q = round(double(C)./tol).*tol;
    [~,~,Group] = unique(Q,'rows','stable');
    Counts = accumarray(Group,1);
    S.maxGroupCount = max(Counts);
    duplicate = find(Counts > 1);
    S.duplicateGroupCount = numel(duplicate);
    if isempty(duplicate)
        return;
    end

    XSpread = NaN(numel(duplicate),1);
    YSpreadRaw = NaN(numel(duplicate),1);
    YSpreadNorm = NaN(numel(duplicate),1);
    Yn = normalizeWithScale(Y,objMin,objSpan);
    for i = 1 : numel(duplicate)
        idx = find(Group == duplicate(i));
        XSpread(i) = maxPairDistance(X(idx,:));
        YSpreadRaw(i) = maxPairDistance(Y(idx,:));
        YSpreadNorm(i) = maxPairDistance(Yn(idx,:));
    end
    S.xSpread50 = percentileFinite(XSpread,50);
    S.xSpread90 = percentileFinite(XSpread,90);
    S.xSpreadMax = maxFinite(XSpread);
    S.ySpread50Raw = percentileFinite(YSpreadRaw,50);
    S.ySpread90Raw = percentileFinite(YSpreadRaw,90);
    S.ySpreadMaxRaw = maxFinite(YSpreadRaw);
    S.ySpread50Norm = percentileFinite(YSpreadNorm,50);
    S.ySpread90Norm = percentileFinite(YSpreadNorm,90);
    S.ySpreadMaxNorm = maxFinite(YSpreadNorm);
end

function d = maxPairDistance(X)
    X = double(X);
    X = X(all(isfinite(X),2),:);
    if size(X,1) < 2
        d = 0;
        return;
    end
    D = pairDistance(X,X);
    d = maxFinite(D(:));
end

function writeSnapshotSummary(outDir,Snapshot,Options,snapshotFile)
    Row = stageProfileRow(Snapshot,Options,snapshotFile);
    T = struct2table(Row);
    writetable(T,fullfile(outDir,'snapshot_summary.csv'));
    writetable(T,fullfile(outDir,'stage_trainset_profile.csv'));
    writeStageRefProfile(outDir,Snapshot);
end

function Row = stageProfileRow(Snapshot,Options,snapshotFile)
    trainX = double(Snapshot.TrainX);
    trainY = double(Snapshot.trainObjs);
    trainC = double(Snapshot.TrainC);
    trainRef = double(Snapshot.trainRef(:));
    if isempty(trainRef)
        trainRef = nan(size(trainX,1),1);
    end
    uniqueRefs = unique(trainRef(isfinite(trainRef)),'stable');
    GroupStats = conditionGroupMetrics(trainC,trainX,trainY, ...
        Snapshot.objMin,Snapshot.objSpan,1e-12);
    ySpan = rangeRows(trainY);
    xSpan = rangeRows(trainX);
    Row = struct( ...
        'problem',string(Snapshot.problem), ...
        'run',double(Snapshot.run), ...
        'seed',double(Snapshot.seed), ...
        'targetFE',double(Snapshot.targetFE), ...
        'actualFE',double(Snapshot.actualFE), ...
        'gen',double(Snapshot.gen), ...
        'condition_mode',string(Snapshot.conditionMode), ...
        'boundaryTargetMode',string(Snapshot.boundaryTargetMode), ...
        'train_count',size(Snapshot.TrainX,1), ...
        'query_count',size(Snapshot.QueryC,1), ...
        'condition_dim',size(Snapshot.TrainC,2), ...
        'condition_unique_count',uniqueRowCount(trainC,1e-12), ...
        'condition_duplicate_count',size(trainC,1) - ...
            uniqueRowCount(trainC,1e-12), ...
        'duplicate_condition_group_count',GroupStats.duplicateGroupCount, ...
        'condition_group_size_max',GroupStats.maxGroupCount, ...
        'duplicate_condition_x_spread50',GroupStats.xSpread50, ...
        'duplicate_condition_x_spread90',GroupStats.xSpread90, ...
        'duplicate_condition_x_spreadmax',GroupStats.xSpreadMax, ...
        'duplicate_condition_y_spread50_raw',GroupStats.ySpread50Raw, ...
        'duplicate_condition_y_spread90_raw',GroupStats.ySpread90Raw, ...
        'duplicate_condition_y_spreadmax_raw',GroupStats.ySpreadMaxRaw, ...
        'duplicate_condition_y_spread50_norm',GroupStats.ySpread50Norm, ...
        'duplicate_condition_y_spread90_norm',GroupStats.ySpread90Norm, ...
        'duplicate_condition_y_spreadmax_norm',GroupStats.ySpreadMaxNorm, ...
        'condition_nn_dist50',conditionNearestDistance(trainC,50), ...
        'condition_nn_dist90',conditionNearestDistance(trainC,90), ...
        'train_ref_unique_count',numel(uniqueRefs), ...
        'train_ref_min',minOrNaN(uniqueRefs), ...
        'train_ref_max',maxOrNaN(uniqueRefs), ...
        'train_ref_coverage',safeRatio(numel(uniqueRefs),size(Snapshot.W,1)), ...
        'train_tau_iqr',iqrFinite(Snapshot.trainTau), ...
        'train_tau_range',rangeFinite(Snapshot.trainTau), ...
        'train_tau_nonzero_rate',mean(abs(Snapshot.trainTau) > 1e-12), ...
        'train_x_nn_dist50',nearestRowDistance(trainX,50), ...
        'train_x_nn_dist90',nearestRowDistance(trainX,90), ...
        'train_y_nn_dist50_raw',nearestRowDistance(trainY,50), ...
        'train_y_nn_dist90_raw',nearestRowDistance(trainY,90), ...
        'train_y_nn_dist50_norm',nearestRowDistance( ...
            normalizeWithScale(trainY,Snapshot.objMin,Snapshot.objSpan),50), ...
        'train_y_nn_dist90_norm',nearestRowDistance( ...
            normalizeWithScale(trainY,Snapshot.objMin,Snapshot.objSpan),90), ...
        'train_x_span50',percentileFinite(xSpan,50), ...
        'train_x_span90',percentileFinite(xSpan,90), ...
        'train_y_span_f1_raw',columnSpan(trainY,1), ...
        'train_y_span_f2_raw',columnSpan(trainY,2), ...
        'train_y_min_f1_raw',columnMin(trainY,1), ...
        'train_y_max_f1_raw',columnMax(trainY,1), ...
        'train_y_min_f2_raw',columnMin(trainY,2), ...
        'train_y_max_f2_raw',columnMax(trainY,2), ...
        'train_y_span50_norm',percentileFinite( ...
            rangeRows(normalizeWithScale(trainY,Snapshot.objMin, ...
            Snapshot.objSpan)),50), ...
        'train_pair_dec_dist50',percentileFinite(rowNorm( ...
            Snapshot.trainXf - Snapshot.trainXi),50), ...
        'train_pair_dec_dist90',percentileFinite(rowNorm( ...
            Snapshot.trainXf - Snapshot.trainXi),90), ...
        'train_pair_obj_dist50_norm',percentileFinite( ...
            objectiveDistance(Snapshot.trainYf,Snapshot.trainYi, ...
            Snapshot.objMin,Snapshot.objSpan),50), ...
        'train_pair_obj_dist90_norm',percentileFinite( ...
            objectiveDistance(Snapshot.trainYf,Snapshot.trainYi, ...
            Snapshot.objMin,Snapshot.objSpan),90), ...
        'bmem_count',bmemCount(Snapshot.BMem), ...
        'maxCandidatePairsPerRef',Options.maxCandidatePairsPerRef, ...
        'snapshot_file',string(snapshotFile));
end

function writeStageRefProfile(outDir,Snapshot)
    refs = double(Snapshot.trainRef(:));
    if isempty(refs)
        T = table();
        writetable(T,fullfile(outDir,'stage_trainset_by_ref.csv'));
        return;
    end
    uniqueRefs = unique(refs(isfinite(refs)),'stable');
    Rows = repmat(emptyRefProfileRow(),numel(uniqueRefs),1);
    for i = 1 : numel(uniqueRefs)
        idx = refs == uniqueRefs(i);
        Y = Snapshot.trainObjs(idx,:);
        C = Snapshot.TrainC(idx,:);
        Tau = Snapshot.trainTau(idx,:);
        Rows(i).ref = uniqueRefs(i);
        Rows(i).count = sum(idx);
        Rows(i).condition_unique_count = uniqueRowCount(C,1e-12);
        Rows(i).condition_nn_dist50 = conditionNearestDistance(C,50);
        Rows(i).condition_nn_dist90 = conditionNearestDistance(C,90);
        Rows(i).tau_iqr = iqrFinite(Tau);
        Rows(i).tau_range = rangeFinite(Tau);
        Rows(i).y_span_f1_raw = columnSpan(Y,1);
        Rows(i).y_span_f2_raw = columnSpan(Y,2);
        Rows(i).y_nn_dist50_norm = nearestRowDistance( ...
            normalizeWithScale(Y,Snapshot.objMin,Snapshot.objSpan),50);
        Rows(i).y_nn_dist90_norm = nearestRowDistance( ...
            normalizeWithScale(Y,Snapshot.objMin,Snapshot.objSpan),90);
    end
    writetable(struct2table(Rows),fullfile(outDir, ...
        'stage_trainset_by_ref.csv'));
end

function Row = emptyRefProfileRow()
    Row = struct( ...
        'ref',NaN, ...
        'count',0, ...
        'condition_unique_count',0, ...
        'condition_nn_dist50',NaN, ...
        'condition_nn_dist90',NaN, ...
        'tau_iqr',NaN, ...
        'tau_range',NaN, ...
        'y_span_f1_raw',NaN, ...
        'y_span_f2_raw',NaN, ...
        'y_nn_dist50_norm',NaN, ...
        'y_nn_dist90_norm',NaN);
end

function writeTraceFigure(outDir,Trace,Options)
    if isempty(Trace) || height(Trace) == 0
        return;
    end
    ok = string(Trace.status) == "ok";
    Trace = Trace(ok,:);
    if isempty(Trace) || height(Trace) == 0
        return;
    end
    fig = figure('Visible','off','Color','w','Position',[100 100 1200 760]);
    layout = tiledlayout(fig,2,1,'TileSpacing','compact','Padding','compact');
    title(layout,sprintf('%s run=%d targetFE=%d G training trace', ...
        Options.problemName,Options.runId,Options.snapshotFE), ...
        'Interpreter','none');
    variants = unique(string(Trace.variant),'stable');
    ax1 = nexttile(layout);
    hold(ax1,'on');
    ax2 = nexttile(layout);
    hold(ax2,'on');
    for i = 1 : numel(variants)
        idx = string(Trace.variant) == variants(i);
        x = double(Trace.checkpoint_epoch(idx));
        plot(ax1,x,double(Trace.train_x_rec90(idx)),'-o', ...
            'DisplayName',variants(i));
        plot(ax2,x,double(Trace.train_y_rec90_norm(idx)),'-o', ...
            'DisplayName',variants(i));
    end
    set(ax1,'XScale','log');
    set(ax2,'XScale','log');
    ylabel(ax1,'train x rec90');
    ylabel(ax2,'train y rec90 norm');
    xlabel(ax2,'cumulative epoch');
    grid(ax1,'on');
    grid(ax2,'on');
    legend(ax1,'Interpreter','none','Location','best');
    legend(ax2,'Interpreter','none','Location','best');
    exportgraphics(fig,fullfile(outDir,'single_stage_overfit_trace.png'), ...
        'Resolution',180);
    close(fig);
end

function writeReconstructionFigure(outDir,Snapshot,Variants,Recon,Options)
    fig = figure('Visible','off','Color','w','Position',[100 100 1500 950]);
    n = numel(Variants);
    cols = 2;
    rows = ceil(n/cols);
    layout = tiledlayout(fig,rows,cols,'TileSpacing','compact', ...
        'Padding','compact');
    title(layout,sprintf('%s run=%d targetFE=%d single-stage overfit', ...
        Snapshot.problem,Snapshot.run,Snapshot.targetFE), ...
        'Interpreter','none');
    for i = 1 : n
        ax = nexttile(layout);
        scatter(ax,Snapshot.trainObjs(:,1),Snapshot.trainObjs(:,2),36, ...
            [1.0 0.62 0.12],'s','filled','MarkerEdgeColor',[0.25 0.25 0.25]);
        hold(ax,'on');
        if ~isempty(Recon{i}.obj)
            scatter(ax,Recon{i}.obj(:,1),Recon{i}.obj(:,2),28, ...
                [0.90 0.12 0.15],'o','filled');
        end
        title(ax,string(Variants(i).variant),'Interpreter','none');
        xlabel(ax,'f_1');
        ylabel(ax,'f_2');
        grid(ax,'on');
        axis(ax,'tight');
    end
    outFile = fullfile(outDir,sprintf( ...
        '%s_run%d_targetFE%06d_overfit_reconstruction.png', ...
        char(Options.problemName),Options.runId,Options.snapshotFE));
    exportgraphics(fig,outFile,'Resolution',180);
    close(fig);
end

function Row = emptyMetricRow()
    Row = struct( ...
        'problem',"", ...
        'run',NaN, ...
        'targetFE',NaN, ...
        'actualFE',NaN, ...
        'gen',NaN, ...
        'variant',"", ...
        'condition_mode',"", ...
        'boundaryTargetMode',"", ...
        'snapshot_file',"", ...
        'train_count',0, ...
        'query_count',0, ...
        'condition_dim',0, ...
        'condition_unique_count',0, ...
        'condition_duplicate_count',0, ...
        'condition_nn_dist50',NaN, ...
        'condition_nn_dist90',NaN, ...
        'train_tau_iqr',NaN, ...
        'train_tau_range',NaN, ...
        'train_tau_nonzero_rate',NaN, ...
        'train_pair_dec_dist50',NaN, ...
        'train_pair_dec_dist90',NaN, ...
        'advWeight',NaN, ...
        'reconstructionWeight',NaN, ...
        'huberDelta',NaN, ...
        'iter',NaN, ...
        'checkpoint_epoch',NaN, ...
        'chunk_epoch',NaN, ...
        'trainMode',"", ...
        'miniBatch',NaN, ...
        'g_updates',NaN, ...
        'd_updates',NaN, ...
        'runtime_sec',NaN, ...
        'train_x_rec50',NaN, ...
        'train_x_rec90',NaN, ...
        'train_x_recmax',NaN, ...
        'train_y_rec50_raw',NaN, ...
        'train_y_rec90_raw',NaN, ...
        'train_y_recmax_raw',NaN, ...
        'train_y_rec50_norm',NaN, ...
        'train_y_rec90_norm',NaN, ...
        'train_y_recmax_norm',NaN, ...
        'status',"pending", ...
        'error_message',"");
end
