function varargout = PairBoundaryWGAN_RC(action,varargin)
%PAIRBOUNDARYWGAN_RC Absolute endpoint pair CGAN with relation ED.
%   Conditions are [reference-vector, side]. Pair IDs are metadata only.
%   Training uses every active complete pair in shuffled, no-replacement
%   epochs. A triggered update is adopted immediately and used directly.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use PlatEMO for research.
%--------------------------------------------------------------------------

    switch lower(strtrim(string(action)))
        case "trainifneeded"
            [varargout{1:nargout}] = trainIfNeeded(varargin{:});
        case "sample"
            [varargout{1:nargout}] = sampleByCondition(varargin{:});
        otherwise
            error('CBSPairGuide:BadWGANAction', ...
                'Unsupported pair WGAN action: %s.',action);
    end
end

function [Model,Status] = trainIfNeeded(Model,Data,Gate,Problem,Options)
%TRAINIFNEEDED Gate, trigger, continue training, and use the result.

    Options = fillOptions(Options);
    Model = normalizeModelMetadata(Model);
    Status = emptyStatus(Model.ready,Options.trainingSigma);
    firstTraining = ~Model.ready;
    if firstTraining
        epochCount = Options.initialEpoch;
        Status.trainingKind = "initial";
    else
        epochCount = Options.retrainEpoch;
        Status.trainingKind = "retrain";
    end
    Status.requestedEpochs = epochCount;
    Status.nCritic = Options.nCritic;
    if ~Gate.eligible || Data.count < Options.pairMinPairs
        return;
    end
    if epochCount == 0
        Status.reason = "disabled";
        return;
    end

    [changed,newRegions,changedRows] = changesSince(Data,Model.lastPairIds, ...
        Model.lastPairFE,Model.lastRefs);
    Status.changedPairs = changed;
    Status.newRegions = newRegions;
    needsUpdate = ~Model.ready || ...
        changed >= Options.pairRetrainChanges || ...
        newRegions >= Options.pairNewRegionChanges;
    if ~needsUpdate
        Status.reason = "current";
        Status.useModel = Model.ready;
        return;
    end

    if Options.collectDiagnostics
        Status.preDiagnostics = pairModelDiagnostics( ...
            Model,Data,changedRows,Options);
    end
    Model = prepareState(Model,Problem.D,size(Data.cF,2),Options);
    Model = trainModel(Model,Data,Options,epochCount);
    Model.ready = true;
    Model.lastPairIds = reshape(double(Data.id),[],1);
    Model.lastPairFE = reshape(double(Data.lastFE),[],1);
    Model.lastRefs = unique(double(Data.ref),'stable');
    Status.trained = true;
    Status.useModel = true;
    Status.epochs = Model.eventEpochs;
    Status.updates = Model.eventUpdates;
    Status.pairVisits = Model.eventPairVisits;
    Status.trainingPairs = Data.count;
    Status.batchesPerEpoch = Model.eventBatchesPerEpoch;
    Status.criticUpdates = Options.nCritic*Model.eventUpdates;
    Status.criticPairVisits = Options.nCritic*Model.eventPairVisits;
    if Options.collectDiagnostics
        Status.postDiagnostics = pairModelDiagnostics( ...
            Model,Data,changedRows,Options);
    end
    Status.reason = "trained";
end

function [Dec,Info] = sampleByCondition(Model,QueryC,Problem,Options)
%SAMPLEBYCONDITION Generate absolute infeasible-side candidates from sigma Z.

    Options = fillOptions(Options);
    QueryC = double(QueryC);
    Dec = zeros(0,Problem.D);
    Info = struct('projectionRate',zeros(0,1), ...
        'normalized',zeros(0,Problem.D), ...
        'sampleSigma',Options.sampleSigma);
    if isempty(QueryC) || isempty(Model) || ~isstruct(Model) || ...
            ~isfield(Model,'ready') || ~Model.ready
        return;
    end
    if size(QueryC,2) ~= Model.C
        error('CBSPairGuide:BadQueryCondition', ...
            'Pair-guide query width does not match the model.');
    end
    Z = randn(size(QueryC,1),Model.zDim);
    if Options.sampleSigma ~= 1
        Z = Options.sampleSigma*Z;
    end
    scaled = generateScaled(Model,Z,QueryC);
    normalized = (scaled+1)/2;
    normalized = max(0,min(1,normalized));
    lower = double(Problem.lower);
    span = double(Problem.upper)-lower;
    span(span <= eps) = 1;
    Dec = lower+normalized.*span;
    Info.projectionRate = zeros(size(QueryC,1),1);
    Info.normalized = normalized;
end

function Model = trainModel(Model,Data,Options,epochCount)
%TRAINMODEL Traverse every complete training pair once per epoch.

    pairBatch = max(1,min(Data.count,floor(Options.miniBatch/2)));
    batchesPerEpoch = ceil(Data.count/pairBatch);
    Model.eventEpochs = 0;
    Model.eventUpdates = 0;
    Model.eventPairVisits = 0;
    Model.eventBatchesPerEpoch = batchesPerEpoch;
    for epoch = 1 : epochCount
        order = balancedPairEpochOrder(Data.ref);
        if numel(order) ~= Data.count || numel(unique(order)) ~= Data.count
            error('CBSPairGuide:IncompleteEpoch', ...
                'Every training pair must occur exactly once per epoch.');
        end
        for first = 1 : pairBatch : Data.count
            idx = order(first:min(first+pairBatch-1,Data.count));
            for critic = 1 : Options.nCritic
                Model = updateCritic(Model,Data,idx,Options);
            end
            Model = updateGenerator(Model,Data,idx,Options);
            Model.eventUpdates = Model.eventUpdates+1;
            Model.eventPairVisits = Model.eventPairVisits+numel(idx);
        end
        Model.eventEpochs = epoch;
    end
end

function Model = updateCritic(Model,Data,idx,Options)
%UPDATECRITIC One conditional WGAN-GP update on one complete-pair batch.

    Real = single(2*[Data.xF(idx,:);Data.xI(idx,:)]'-1);
    Cond = single([Data.cF(idx,:);Data.cI(idx,:)]');
    batchCount = size(Real,2);
    Z = single(gaussianNoise( ...
        Model.zDim,batchCount,Options.trainingSigma));
    dlReal = dlarray(Real,'CB');
    dlCond = dlarray(Cond,'CB');
    dlZ = dlarray(Z,'CB');
    fake = extractdata(generatorForward(Model.netG,dlZ,dlCond));
    epsilon = rand(1,batchCount,'single');
    hat = epsilon.*Real+(1-epsilon).*fake;
    gradients = dlfeval(@criticGradients,Model.netC,dlReal,dlCond, ...
        fake,dlarray(hat,'CB'),single(Options.gpLambda));
    Model.iterC = Model.iterC+1;
    [Model.netC,Model.avgC,Model.avgSqC] = adamupdate( ...
        Model.netC,gradients,Model.avgC,Model.avgSqC,Model.iterC, ...
        Options.lrD,0,0.9);
end

function gradients = criticGradients(netC,dlReal,dlCond,fake,dlHat,gpLambda)
%CRITICGRADIENTS Wasserstein loss plus input gradient penalty.

    dlFake = dlarray(fake,'CB');
    realScore = forward(netC,[dlReal;dlCond]);
    fakeScore = forward(netC,[dlFake;dlCond]);
    hatScore = forward(netC,[dlHat;dlCond]);
    hatGradient = dlgradient(sum(hatScore,'all'),dlHat, ...
        'EnableHigherDerivatives',true);
    gradientNorm = sqrt(sum(hatGradient.^2,1)+single(1e-12));
    penalty = mean((gradientNorm-1).^2,'all');
    loss = mean(fakeScore,'all')-mean(realScore,'all')+gpLambda*penalty;
    gradients = dlgradient(loss,netC.Learnables, ...
        'EnableHigherDerivatives',false);
end

function Model = updateGenerator(Model,Data,idx,Options)
%UPDATEGENERATOR Adversarial + distribution relation + mode-seeking loss.

    cF = dlarray(single(Data.cF(idx,:)'),'CB');
    cI = dlarray(single(Data.cI(idx,:)'),'CB');
    realRelation = dlarray(single([Data.w(idx,:),Data.delta(idx,:)]'),'CB');
    pairCount = numel(idx);
    zAdv = dlarray(single(gaussianNoise( ...
        Model.zDim,2*pairCount,Options.trainingSigma)),'CB');
    zPair = dlarray(single(gaussianNoise( ...
        Model.zDim,pairCount,Options.trainingSigma)),'CB');
    z1 = dlarray(single(gaussianNoise( ...
        Model.zDim,2*pairCount,Options.trainingSigma)),'CB');
    z2 = dlarray(single(gaussianNoise( ...
        Model.zDim,2*pairCount,Options.trainingSigma)),'CB');
    gradients = dlfeval(@generatorGradients,Model.netG,Model.netC, ...
        cF,cI,realRelation,zAdv,zPair,z1,z2, ...
        single(Options.pairRelationWeight), ...
        single(Options.pairLatentWeight), ...
        single(Options.pairLatentThreshold));
    Model.iterG = Model.iterG+1;
    [Model.netG,Model.avgG,Model.avgSqG] = adamupdate( ...
        Model.netG,gradients,Model.avgG,Model.avgSqG,Model.iterG, ...
        Options.lrG,0,0.9);
end

function gradients = generatorGradients(netG,netC,cF,cI,realRelation, ...
        zAdv,zPair,z1,z2,lambdaRel,lambdaMS,tauMS)
%GENERATORGRADIENTS Same-z only defines a generated joint relation sample.

    cBoth = [cF,cI];
    advOutput = generatorForward(netG,zAdv,cBoth);
    adversarial = -mean(forward(netC,[advOutput;cBoth]),'all');

    generatedF = generatorForward(netG,zPair,cF);
    generatedI = generatorForward(netG,zPair,cI);
    w = cF(1:end-1,:);
    generatedDelta = (generatedI-generatedF)/2;
    generatedRelation = [w;generatedDelta];
    relation = energyDistanceDL(realRelation,generatedRelation);

    latent1 = generatorForward(netG,z1,cBoth);
    latent2 = generatorForward(netG,z2,cBoth);
    outputDifference = mean(abs(latent1-latent2),1);
    latentDifference = mean(abs(z1-z2),1)+single(1e-12);
    sensitivity = outputDifference./latentDifference;
    modeSeeking = mean(max(single(0),tauMS-sensitivity).^2,'all');
    loss = adversarial+lambdaRel*relation+lambdaMS*modeSeeking;
    gradients = dlgradient(loss,netG.Learnables, ...
        'EnableHigherDerivatives',false);
end

function output = generatorForward(netG,dlZ,dlC)
%GENERATORFORWARD Network tanh is the absolute normalized decision map.

    output = forward(netG,[dlZ;dlC]);
end

function Z = gaussianNoise(rows,columns,sigma)
%GAUSSIANNOISE Draw latent noise without changing sigma=1 RNG behavior.

    Z = randn(rows,columns);
    if sigma ~= 1
        Z = sigma*Z;
    end
end

function value = energyDistanceDL(A,B)
%ENERGYDISTANCEDL Differentiable biased energy-distance estimate.

    A = stripdims(A);
    B = stripdims(B);
    ab = pairwiseDistanceDL(A,B);
    aa = pairwiseDistanceDL(A,A);
    bb = pairwiseDistanceDL(B,B);
    value = max(single(0),2*mean(ab,'all')- ...
        mean(aa,'all')-mean(bb,'all'));
end

function distance = pairwiseDistanceDL(A,B)
%PAIRWISEDISTANCEDL Column-sample Euclidean distances.

    distance2 = sum(A.^2,1)'+sum(B.^2,1)-2*(A'*B);
    distance = sqrt(max(single(0),distance2)+single(1e-12));
end

function output = generateScaled(Model,Z,C)
%GENERATESCALED Forward double rows through the tanh generator.

    dlZ = dlarray(single(double(Z)'),'CB');
    dlC = dlarray(single(double(C)'),'CB');
    output = double(extractdata(generatorForward(Model.netG,dlZ,dlC)))';
end

function order = balancedPairEpochOrder(refs)
%BALANCEDPAIREPOCHORDER Interleave refs in one shuffled no-replacement pass.

    refs = reshape(double(refs),[],1);
    values = unique(refs,'stable');
    groups = cell(numel(values),1);
    cursor = ones(numel(values),1);
    for group = 1 : numel(values)
        rows = find(refs == values(group));
        groups{group} = rows(randperm(numel(rows)));
    end
    order = zeros(numel(refs),1);
    next = 0;
    while next < numel(refs)
        groupOrder = randperm(numel(values));
        for group = groupOrder
            if cursor(group) <= numel(groups{group})
                next = next+1;
                order(next) = groups{group}(cursor(group));
                cursor(group) = cursor(group)+1;
            end
        end
    end
end

function [changed,newRegions,changedRows] = changesSince(Data,ids,lastFE,refs)
%CHANGESSINCE Count training-set membership/endpoint changes and new refs.

    changed = 0;
    changedRows = false(Data.count,1);
    ids = reshape(double(ids),[],1);
    lastFE = reshape(double(lastFE),[],1);
    for row = 1 : Data.count
        previous = find(ids == Data.id(row),1);
        if isempty(previous) || previous > numel(lastFE) || ...
                Data.lastFE(row) > lastFE(previous)+1e-12
            changed = changed+1;
            changedRows(row) = true;
        end
    end
    changed = changed+nnz(~ismember(ids,double(Data.id)));
    newRegions = numel(setdiff(unique(double(Data.ref)),double(refs)));
end

function Model = prepareState(Model,D,C,Options)
%PREPARESTATE Warm-start compatible weights or initialize networks.

    required = {'netG','netC','D','C','zDim', ...
        'generatorHidden','criticHidden'};
    compatible = all(isfield(Model,required)) && ...
        Model.D == D && Model.C == C && Model.zDim == Options.zDim && ...
        isequal(Model.generatorHidden,Options.generatorHidden) && ...
        isequal(Model.criticHidden,Options.criticHidden);
    if compatible
        return;
    end
    Model = normalizeModelMetadata([]);
    Model.D = D;
    Model.C = C;
    Model.zDim = Options.zDim;
    Model.generatorHidden = Options.generatorHidden;
    Model.criticHidden = Options.criticHidden;
    Model.netG = createGenerator(C+Options.zDim,D,Options.generatorHidden);
    Model.netC = createCritic(D+C,Options.criticHidden);
end

function Model = normalizeModelMetadata(Model)
%NORMALIZEMODELMETADATA Fill training and trigger state.

    if isempty(Model) || ~isstruct(Model)
        Model = struct();
    end
    defaults = struct('ready',false,'lastPairIds',zeros(0,1), ...
        'lastPairFE',zeros(0,1),'lastRefs',zeros(0,1), ...
        'avgG',[],'avgSqG',[],'avgC',[],'avgSqC',[], ...
        'iterG',0,'iterC',0,'eventEpochs',0,'eventUpdates',0, ...
        'eventPairVisits',0,'eventBatchesPerEpoch',0);
    names = fieldnames(defaults);
    for i = 1 : numel(names)
        if ~isfield(Model,names{i})
            Model.(names{i}) = defaults.(names{i});
        end
    end
end

function netG = createGenerator(inputDim,D,hidden)
%CREATEGENERATOR Two hidden layers and bounded absolute output.

    layers = featureInputLayer(inputDim, ...
        'Normalization','none','Name','pair_guide_g_in');
    for i = 1 : numel(hidden)
        layers = [layers;fullyConnectedLayer(hidden(i), ...
            'Name',sprintf('pair_guide_g_fc%d',i)); ...
            leakyReluLayer(0.2, ...
            'Name',sprintf('pair_guide_g_lrelu%d',i))]; %#ok<AGROW>
    end
    layers = [layers;fullyConnectedLayer(D,'Name','pair_guide_g_out'); ...
        tanhLayer('Name','pair_guide_g_tanh')];
    netG = dlnetwork(layerGraph(layers));
end

function netC = createCritic(inputDim,hidden)
%CREATECRITIC Conditional endpoint critic with linear output.

    layers = featureInputLayer(inputDim, ...
        'Normalization','none','Name','pair_guide_c_in');
    for i = 1 : numel(hidden)
        layers = [layers;fullyConnectedLayer(hidden(i), ...
            'Name',sprintf('pair_guide_c_fc%d',i)); ...
            leakyReluLayer(0.2, ...
            'Name',sprintf('pair_guide_c_lrelu%d',i))]; %#ok<AGROW>
    end
    layers = [layers;fullyConnectedLayer(1,'Name','pair_guide_c_out')];
    netC = dlnetwork(layerGraph(layers));
end

function Diagnostics = pairModelDiagnostics(Model,Data,changedRows,Options)
%PAIRMODELDIAGNOSTICS Fixed-probe fit and temporal-generalization signals.

    Diagnostics = emptyModelDiagnostics();
    if isempty(Model) || ~isstruct(Model) || ~isfield(Model,'ready') || ...
            ~Model.ready || ~all(isfield(Model,{'netG','netC'})) || ...
            Data.count < 1
        return;
    end
    savedRNG = rng;
    cleanup = onCleanup(@()rng(savedRNG));
    rng(24681357,'twister');
    count = Data.count;
    cF = single(Data.cF');
    cI = single(Data.cI');
    Z = single(gaussianNoise(Model.zDim,count,Options.trainingSigma));
    generatedF = extractdata(generatorForward( ...
        Model.netG,dlarray(Z,'CB'),dlarray(cF,'CB')));
    generatedI = extractdata(generatorForward( ...
        Model.netG,dlarray(Z,'CB'),dlarray(cI,'CB')));
    generatedDelta = double((generatedI-generatedF)'/2);
    squared = mean((generatedDelta-double(Data.delta)).^2,2);
    Diagnostics.allRelationRMSE = sqrt(mean(squared));
    changedRows = reshape(logical(changedRows),[],1);
    if numel(changedRows) == count && any(changedRows)
        Diagnostics.changedRelationRMSE = sqrt(mean(squared(changedRows)));
    end

    real = single(2*[Data.xF;Data.xI]'-1);
    fake = single([generatedF,generatedI]);
    conditions = single([Data.cF;Data.cI]');
    realScore = extractdata(forward(Model.netC, ...
        dlarray([real;conditions],'CB')));
    fakeScore = extractdata(forward(Model.netC, ...
        dlarray([fake;conditions],'CB')));
    Diagnostics.criticGap = double(mean(realScore,'all')- ...
        mean(fakeScore,'all'));

    generated = double((fake'+1)/2);
    training = double([Data.xF;Data.xI]);
    nearest = inf(size(generated,1),1);
    for i = 1 : size(training,1)
        nearest = min(nearest, ...
            sqrt(sum((generated-training(i,:)).^2,2)));
    end
    Diagnostics.generatedNearestMedian = median(nearest);
    Diagnostics.generatedSpread = mean(std(generated,0,1));
    clear cleanup;
end

function Diagnostics = emptyModelDiagnostics()
%EMPTYMODELDIAGNOSTICS Default unavailable fixed-probe measurements.

    Diagnostics = struct('allRelationRMSE',NaN, ...
        'changedRelationRMSE',NaN,'criticGap',NaN, ...
        'generatedNearestMedian',NaN,'generatedSpread',NaN);
end

function Status = emptyStatus(useModel,trainingSigma)
%EMPTYSTATUS Default blocked event state.

    Status = struct('trained',false,'useModel',logical(useModel), ...
        'trainingKind',"",'requestedEpochs',0,'nCritic',0,'epochs',0, ...
        'updates',0,'pairVisits',0,'trainingPairs',0, ...
        'criticUpdates',0,'criticPairVisits',0, ...
        'batchesPerEpoch',0,'trainingSigma',double(trainingSigma), ...
        'reason',"gate",'changedPairs',0,'newRegions',0, ...
        'preDiagnostics',emptyModelDiagnostics(), ...
        'postDiagnostics',emptyModelDiagnostics());
end

function Options = fillOptions(Options)
%FILLOPTIONS Locked PairGuide training values.

    Options = defaultOption(Options,'zDim',6);
    Options = defaultOption(Options,'epochs',100);
    Options = defaultOption(Options,'initialEpoch',Options.epochs);
    Options = defaultOption(Options,'retrainEpoch',Options.epochs);
    Options = defaultOption(Options,'miniBatch',64);
    Options = defaultOption(Options,'lrD',1e-4);
    Options = defaultOption(Options,'lrG',1e-4);
    Options = defaultOption(Options,'gpLambda',10);
    Options = defaultOption(Options,'nCritic',4);
    Options = defaultOption(Options,'collectDiagnostics',false);
    Options = defaultOption(Options,'trainingSigma',1);
    Options = defaultOption(Options,'sampleSigma',1);
    Options = defaultOption(Options,'generatorHidden',[32 32]);
    Options = defaultOption(Options,'criticHidden',[32 32]);
    Options = defaultOption(Options,'pairMinPairs',32);
    Options = defaultOption(Options,'pairRetrainChanges',8);
    Options = defaultOption(Options,'pairNewRegionChanges',2);
    Options = defaultOption(Options,'pairRelationWeight',0.2);
    Options = defaultOption(Options,'pairLatentWeight',0.05);
    Options = defaultOption(Options,'pairLatentThreshold',0.05);
    Options.zDim = max(1,round(double(Options.zDim)));
    Options.epochs = max(0,round(double(Options.epochs)));
    Options.initialEpoch = max(0,round(double(Options.initialEpoch)));
    Options.retrainEpoch = max(0,round(double(Options.retrainEpoch)));
    Options.miniBatch = max(2,round(double(Options.miniBatch)));
    Options.nCritic = max(1,round(double(Options.nCritic)));
    if ~isscalar(Options.collectDiagnostics) || ...
            ~(islogical(Options.collectDiagnostics) || ...
            isnumeric(Options.collectDiagnostics)) || ...
            ~isfinite(double(Options.collectDiagnostics))
        error('CBSPairGuide:BadTrainingDiagnosticsOption', ...
            'collectDiagnostics must be one finite logical scalar.');
    end
    Options.collectDiagnostics = logical(Options.collectDiagnostics);
    Options.trainingSigma = double(Options.trainingSigma);
    if ~isscalar(Options.trainingSigma) || ...
            ~isfinite(Options.trainingSigma) || Options.trainingSigma < 0
        error('CBSPairGuide:BadTrainingSigma', ...
            'trainingSigma must be one finite nonnegative scalar.');
    end
    Options.sampleSigma = double(Options.sampleSigma);
    if ~isscalar(Options.sampleSigma) || ~isfinite(Options.sampleSigma) || ...
            Options.sampleSigma < 0
        error('CBSPairGuide:BadSampleSigma', ...
            'sampleSigma must be one finite nonnegative scalar.');
    end
    Options.lrD = double(Options.lrD);
    Options.lrG = double(Options.lrG);
    Options.gpLambda = max(0,double(Options.gpLambda));
    Options.generatorHidden = hiddenVector(Options.generatorHidden);
    Options.criticHidden = hiddenVector(Options.criticHidden);
    numeric = {'pairMinPairs','pairRetrainChanges', ...
        'pairNewRegionChanges'};
    for i = 1 : numel(numeric)
        name = numeric{i};
        Options.(name) = max(1,round(double(Options.(name))));
    end
    Options.pairRelationWeight = max(0,double( ...
        Options.pairRelationWeight));
    Options.pairLatentWeight = max(0,double(Options.pairLatentWeight));
    Options.pairLatentThreshold = max(0,double( ...
        Options.pairLatentThreshold));
end

function S = defaultOption(S,name,value)
%DEFAULTOPTION Fill one missing structure field.

    if ~isfield(S,name) || isempty(S.(name))
        S.(name) = value;
    end
end

function hidden = hiddenVector(hidden)
%HIDDENVECTOR Normalize hidden-layer widths.

    hidden = double(hidden(:)');
    hidden = max(1,round(hidden(isfinite(hidden) & hidden > 0)));
    if isempty(hidden)
        hidden = [32 32];
    end
end
