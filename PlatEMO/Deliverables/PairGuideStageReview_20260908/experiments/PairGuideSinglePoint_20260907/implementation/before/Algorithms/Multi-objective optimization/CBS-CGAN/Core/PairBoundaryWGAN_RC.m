function varargout = PairBoundaryWGAN_RC(action,varargin)
%PAIRBOUNDARYWGAN_RC Absolute endpoint pair CGAN with native interpolation.
%   Conditions are [reference-vector, side]. Pair IDs are metadata only.
%   Training uses current complete pairs in shuffled batches. Generator
%   update budgets and content/generation triggers are independent of service.

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
%TRAINIFNEEDED Independent training and service gates; content-based updates.

    Options = fillOptions(Options);
    Model = normalizeModelMetadata(Model);
    Status = emptyStatus(Model.ready && Data.count > 0,Options.trainingSigma);
    Status.generation = Options.generation;
    Status.requestedUpdates = Options.initialEpoch;
    Status.trainingKind = "initial";
    if Model.ready
        Status.requestedUpdates = Options.retrainEpoch;
        Status.trainingKind = "retrain";
    end
    Status.requestedEpochs = Status.requestedUpdates; % Legacy field, update units.
    Status.nCritic = Options.nCritic;
    if Data.count < Options.pairMinPairs || ~Gate.eligible
        Status.reason = "insufficient_pairs";
        return;
    end
    [delta,changedRows,newRegions] = changesSince(Data,Model.lastData,Problem.D);
    Status.contentChange = delta;
    Status.changedPairs = nnz(changedRows);
    Status.newRegions = newRegions;
    due = Options.generation-Model.lastTrainGeneration >= Options.retrainGenerations;
    if Model.ready && delta < Options.retrainChange && ~due
        Status.reason = "current";
        return;
    end
    if Model.lastAttemptGeneration == Options.generation
        Status.reason = "already_attempted";
        return;
    end
    if Status.requestedUpdates == 0
        Status.reason = "disabled";
        return;
    end
    if ~Model.ready
        Status.trigger = "initial";
    elseif delta >= Options.retrainChange
        Status.trigger = "content";
    else
        Status.trigger = "generation";
    end
    Previous = Model;
    Model.lastAttemptGeneration = Options.generation;
    Candidate = prepareState(Model,Problem.D,size(Data.cF,2),Options);
    diagnosticTimer = tic;
    Status.preDiagnostics = pairModelDiagnostics(Candidate,Data,changedRows,Options);
    Status.diagnosticGeneratorRows = 18*Data.count;
    Status.diagnosticCriticRows = 4*Data.count;
    Status.diagnosticSeconds = toc(diagnosticTimer);
    trainingTimer = tic;
    try
        Candidate = trainModel(Candidate,Data,Options,Status.requestedUpdates);
        Status.generatorForwardRows = Candidate.eventGeneratorRows;
        Status.criticForwardRows = Candidate.eventCriticRows;
        finiteWeights = all(cellfun(@(x)all(isfinite(extractdata(x)),'all'), ...
            Candidate.netG.Learnables.Value)) && ...
            all(cellfun(@(x)all(isfinite(extractdata(x)),'all'), ...
            Candidate.netC.Learnables.Value));
        if ~finiteWeights || Candidate.eventFailed
            error('CBSPairGuide:NonfiniteTraining','Nonfinite model parameters.');
        end
    catch err
        Status.trainingSeconds = toc(trainingTimer);
        if strcmp(err.identifier,'CBSPairGuide:NonfiniteTraining')
            Model = Previous;
            Model.lastAttemptGeneration = Options.generation;
            Status.reason = "numerical_failure";
            return;
        end
        rethrow(err);
    end
    Status.trainingSeconds = toc(trainingTimer);
    Model = Candidate;
    Model.ready = true;
    Model.lastData = Data;
    Model.lastTrainGeneration = Options.generation;
    Model.lastAttemptGeneration = Options.generation;
    Model.version = Previous.version+1;
    Status.trained = true;
    Status.useModel = true;
    Status.epochs = Model.eventEpochs;
    Status.updates = Model.eventUpdates;
    Status.pairVisits = Model.eventPairVisits;
    Status.trainingPairs = Data.count;
    Status.batchesPerEpoch = Model.eventBatchesPerEpoch;
    Status.criticUpdates = Options.nCritic*Model.eventUpdates;
    Status.criticPairVisits = Options.nCritic*Model.eventPairVisits;
    Status.generatorForwardRows = 2*(Options.nCritic+1)*Model.eventPairVisits;
    Status.criticForwardRows = (6*Options.nCritic+2)*Model.eventPairVisits;
    Status.diagnosticGeneratorRows = 36*Data.count;
    Status.diagnosticCriticRows = 8*Data.count;
    diagnosticTimer = tic;
    Status.postDiagnostics = pairModelDiagnostics(Model,Data,changedRows,Options);
    Status.diagnosticSeconds = Status.diagnosticSeconds+toc(diagnosticTimer);
    Status.reason = "trained";
end

function [Dec,Info] = sampleByCondition(Model,QueryC,Problem,Options)
%SAMPLEBYCONDITION Interpolate two endpoint outputs sharing exactly one z.

    Options = fillOptions(Options);
    Dec = zeros(0,Problem.D);
    Info = struct('projectionRate',zeros(0,1),'normalized',Dec, ...
        'sampleSigma',Options.sampleSigma,'generatedF',Dec,'generatedI',Dec, ...
        'z',zeros(0,Options.zDim),'alpha',zeros(0,1), ...
        'endpointForwardRows',0,'forwardSeconds',0);
    if isempty(QueryC) || isempty(Model) || ~Model.ready
        return;
    end
    if size(QueryC,2) ~= Model.C
        error('CBSPairGuide:BadQueryCondition','Incorrect condition width.');
    end
    timer = tic;
    count = size(QueryC,1);
    Z = gaussianNoise(count,Model.zDim,Options.sampleSigma);
    cF = double(QueryC); cF(:,end) = 1;
    cI = double(QueryC); cI(:,end) = 0;
    outputs = (generateScaled(Model,[Z;Z],[cF;cI])+1)/2;
    generatedF = outputs(1:count,:);
    generatedI = outputs(count+1:end,:);
    alpha = 0.4+0.2*rand(count,1);
    normalized = (1-alpha).*generatedF+alpha.*generatedI;
    Dec = double(Problem.lower)+normalized.*(double(Problem.upper)-double(Problem.lower));
    Info.normalized = normalized;
    Info.generatedF = generatedF;
    Info.generatedI = generatedI;
    Info.z = Z;
    Info.alpha = alpha;
    Info.projectionRate = zeros(count,1);
    Info.endpointForwardRows = 2*count;
    Info.forwardSeconds = toc(timer);
end

function Model = trainModel(Model,Data,Options,updateBudget)
%TRAINMODEL Shuffled complete-pair batches, bounded by generator updates.

    pairBatch = max(1,min([Data.count,16,floor(Options.miniBatch/2)]));
    batchesPerEpoch = ceil(Data.count/pairBatch);
    Model.eventEpochs = 0;
    Model.eventUpdates = 0;
    Model.eventPairVisits = 0;
    Model.eventGeneratorRows = 0;
    Model.eventCriticRows = 0;
    Model.eventFailed = false;
    Model.eventBatchesPerEpoch = batchesPerEpoch;
    Training = prepareTrainingArrays(Data);
    while Model.eventUpdates < updateBudget
        order = balancedPairEpochOrder(Data.ref);
        for first = 1 : pairBatch : Data.count
            if Model.eventUpdates >= updateBudget
                break;
            end
            idx = order(first:min(first+pairBatch-1,Data.count));
            try
                for critic = 1 : Options.nCritic
                    Model.eventGeneratorRows = Model.eventGeneratorRows+2*numel(idx);
                    Model.eventCriticRows = Model.eventCriticRows+6*numel(idx);
                    Model = updateCritic(Model,Training,idx,Options);
                end
                Model.eventGeneratorRows = Model.eventGeneratorRows+2*numel(idx);
                Model.eventCriticRows = Model.eventCriticRows+2*numel(idx);
                Model = updateGenerator(Model,Training,idx,Options);
            catch err
                if strcmp(err.identifier,'CBSPairGuide:NonfiniteTraining')
                    Model.eventFailed = true;
                    return;
                end
                rethrow(err);
            end
            Model.eventUpdates = Model.eventUpdates+1;
            Model.eventPairVisits = Model.eventPairVisits+numel(idx);
        end
        Model.eventEpochs = Model.eventUpdates/batchesPerEpoch;
    end
end

function Training = prepareTrainingArrays(Data)
%PREPARETRAININGARRAYS Convert invariant batch data once per train event.

    count = Data.count;
    real = single(2*[Data.xF;Data.xI]'-1);
    conditions = single([Data.cF;Data.cI]');
    Training = struct( ...
        'realF',real(:,1:count), ...
        'realI',real(:,count+1:end), ...
        'cF',conditions(:,1:count), ...
        'cI',conditions(:,count+1:end), ...
        'targetF',single(Data.xF'), ...
        'targetI',single(Data.xI'));
end

function Model = updateCritic(Model,Training,idx,Options)
%UPDATECRITIC One conditional WGAN-GP update on one complete-pair batch.

    Real = [Training.realF(:,idx),Training.realI(:,idx)];
    Cond = [Training.cF(:,idx),Training.cI(:,idx)];
    batchCount = size(Real,2);
    Z = single(gaussianNoise( ...
        Model.zDim,numel(idx),Options.trainingSigma));
    Z = [Z,Z];
    dlReal = dlarray(Real,'CB');
    dlCond = dlarray(Cond,'CB');
    dlZ = dlarray(Z,'CB');
    fake = extractdata(generatorForward(Model.netG,dlZ,dlCond));
    epsilon = rand(1,batchCount,'single');
    hat = epsilon.*Real+(1-epsilon).*fake;
    gradients = dlfeval(@criticGradients,Model.netC,dlReal,dlCond, ...
        fake,dlarray(hat,'CB'),single(Options.gpLambda));
    if ~all(cellfun(@(x)all(isfinite(extractdata(x)),'all'),gradients.Value))
        error('CBSPairGuide:NonfiniteTraining','Nonfinite critic gradients.');
    end
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

function Model = updateGenerator(Model,Training,idx,Options)
%UPDATEGENERATOR Reuse the same paired output for adversarial and gap losses.

    cF = dlarray(Training.cF(:,idx),'CB');
    cI = dlarray(Training.cI(:,idx),'CB');
    targetF = dlarray(Training.targetF(:,idx),'CB');
    targetI = dlarray(Training.targetI(:,idx),'CB');
    zPair = dlarray(single(gaussianNoise( ...
        Model.zDim,numel(idx),Options.trainingSigma)),'CB');
    gradients = dlfeval(@generatorGradients,Model.netG,Model.netC, ...
        cF,cI,targetF,targetI,zPair);
    if ~all(cellfun(@(x)all(isfinite(extractdata(x)),'all'),gradients.Value))
        error('CBSPairGuide:NonfiniteTraining','Nonfinite generator gradients.');
    end
    Model.iterG = Model.iterG+1;
    [Model.netG,Model.avgG,Model.avgSqG] = adamupdate( ...
        Model.netG,gradients,Model.avgG,Model.avgSqG,Model.iterG, ...
        Options.lrG,0,0.9);
end

function gradients = generatorGradients(netG,netC,cF,cI,targetF,targetI,zPair)
%GENERATORGRADIENTS Ladv + 10 Lend + Ldelta, relative to each true pair gap.

    cBoth = [cF,cI];
    output = generatorForward(netG,[zPair,zPair],cBoth);
    adversarial = -mean(forward(netC,[output;cBoth]),'all');
    count = size(targetF,2);
    generatedF = (output(:,1:count)+1)/2;
    generatedI = (output(:,count+1:end)+1)/2;
    gapSquared = max(sum((targetI-targetF).^2,1),single(1e-6*size(targetF,1)));
    endpoint = mean((sum((generatedF-targetF).^2,1)+ ...
        sum((generatedI-targetI).^2,1))./(2*gapSquared),'all');
    direction = mean(sum(((generatedI-generatedF)-(targetI-targetF)).^2,1) ...
        ./gapSquared,'all');
    loss = adversarial+10*endpoint+direction;
    gradients = dlgradient(loss,netG.Learnables,'EnableHigherDerivatives',false);
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

function [delta,changedRows,newRegions] = changesSince(Data,Previous,D)
%CHANGESSINCE Content change over the union of reference directions.

    changedRows = true(Data.count,1);
    if isempty(Previous) || ~isstruct(Previous) || ~isfield(Previous,'ref')
        delta = 1;
        newRegions = Data.count;
        return;
    end
    refs = union(Data.ref,Previous.ref);
    changes = ones(numel(refs),1);
    for k = 1:numel(refs)
        now = find(Data.ref == refs(k),1);
        old = find(Previous.ref == refs(k),1);
        if isempty(now) || isempty(old)
            continue;
        end
        oldGap = norm(Previous.xI(old,:)-Previous.xF(old,:));
        movement = max(norm(Data.xF(now,:)-Previous.xF(old,:)), ...
            norm(Data.xI(now,:)-Previous.xI(old,:)));
        changes(k) = min(1,movement/max(oldGap,1e-3*sqrt(D)));
        if any(abs(Data.w(now,:)-Previous.w(old,:)) > 1e-12)
            changes(k) = 1; % The actual input condition also changed.
        end
        changedRows(now) = changes(k) > 0;
    end
    delta = sum(changes)/max(1,numel(refs));
    newRegions = numel(setdiff(Data.ref,Previous.ref));
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
    defaults = struct('ready',false,'lastData',[], ...
        'lastTrainGeneration',-Inf,'lastAttemptGeneration',-Inf,'version',0, ...
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
%PAIRMODELDIAGNOSTICS Endpoint fit, pair direction, and conditional width.

    Diagnostics = emptyModelDiagnostics();
    if isempty(Model) || ~isstruct(Model) || ~isfield(Model,'ready') || ...
            ~all(isfield(Model,{'netG','netC'})) || ...
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
    generatedFNorm = double((generatedF'+1)/2);
    generatedINorm = double((generatedI'+1)/2);
    squaredF = mean((generatedFNorm-double(Data.xF)).^2,2);
    squaredI = mean((generatedINorm-double(Data.xI)).^2,2);
    Diagnostics.feasibleEndpointRMSE = sqrt(mean(squaredF));
    Diagnostics.infeasibleEndpointRMSE = sqrt(mean(squaredI));
    Diagnostics.allEndpointRMSE = sqrt(mean([squaredF;squaredI]));
    changedRows = reshape(logical(changedRows),[],1);
    if numel(changedRows) == count && any(changedRows)
        Diagnostics.changedEndpointRMSE = sqrt(mean([ ...
            squaredF(changedRows);squaredI(changedRows)]));
    end
    generatedDelta = generatedINorm-generatedFNorm;
    Diagnostics.pairDifferenceRMSE = sqrt(mean( ...
        (generatedDelta-double(Data.delta)).^2,'all'));

    real = single(2*[Data.xF;Data.xI]'-1);
    fake = single([generatedF,generatedI]);
    conditions = single([Data.cF;Data.cI]');
    realScore = extractdata(forward(Model.netC, ...
        dlarray([real;conditions],'CB')));
    fakeScore = extractdata(forward(Model.netC, ...
        dlarray([fake;conditions],'CB')));
    Diagnostics.criticGap = double(mean(realScore,'all')- ...
        mean(fakeScore,'all'));

    repeatCount = 8;
    allConditions = double([Data.cF;Data.cI]);
    repeatedConditions = repelem(allConditions,repeatCount,1);
    repeatedNoise = gaussianNoise(Model.zDim, ...
        count*repeatCount,Options.trainingSigma)';
    repeatedNoise = [repeatedNoise;repeatedNoise];
    repeatedGenerated = (generateScaled(Model,repeatedNoise, ...
        repeatedConditions)+1)/2;
    withinSquared = zeros(size(allConditions,1),1);
    for i = 1 : size(allConditions,1)
        rows = (i-1)*repeatCount+(1:repeatCount);
        values = repeatedGenerated(rows,:);
        withinSquared(i) = mean((values-mean(values,1)).^2,'all');
    end
    Diagnostics.sameConditionThickness = sqrt(mean(withinSquared));
    gaps = max(sqrt(sum(Data.delta.^2,2)),1e-3*sqrt(size(Data.xF,2)));
    Diagnostics.relativeEndpointError = mean(sqrt(0.5*size(Data.xF,2)* ...
        (squaredF+squaredI))./gaps);
    Diagnostics.relativeThickness = mean(sqrt(size(Data.xF,2)* ...
        0.5*(withinSquared(1:count)+withinSquared(count+1:end)))./gaps);
    clear cleanup;
end

function Diagnostics = emptyModelDiagnostics()
%EMPTYMODELDIAGNOSTICS Default unavailable fixed-probe measurements.

    Diagnostics = struct('feasibleEndpointRMSE',NaN, ...
        'infeasibleEndpointRMSE',NaN,'allEndpointRMSE',NaN, ...
        'changedEndpointRMSE',NaN,'pairDifferenceRMSE',NaN, ...
        'sameConditionThickness',NaN,'criticGap',NaN, ...
        'relativeEndpointError',NaN,'relativeThickness',NaN);
end

function Status = emptyStatus(useModel,trainingSigma)
%EMPTYSTATUS Default blocked event state.

    Status = struct('trained',false,'useModel',logical(useModel), ...
        'trainingKind',"",'requestedEpochs',0,'nCritic',0,'epochs',0, ...
        'updates',0,'pairVisits',0,'trainingPairs',0, ...
        'criticUpdates',0,'criticPairVisits',0, ...
        'batchesPerEpoch',0,'trainingSeconds',0, ...
        'trainingSigma',double(trainingSigma), ...
        'reason',"gate",'trigger',"",'generation',0,'contentChange',0, ...
        'requestedUpdates',0,'diagnosticSeconds',0,'changedPairs',0,'newRegions',0, ...
        'generatorForwardRows',0,'criticForwardRows',0, ...
        'diagnosticGeneratorRows',0,'diagnosticCriticRows',0, ...
        'preDiagnostics',emptyModelDiagnostics(), ...
        'postDiagnostics',emptyModelDiagnostics());
end

function Options = fillOptions(Options)
%FILLOPTIONS Locked PairGuide training values.

    Options = defaultOption(Options,'zDim',6);
    Options = defaultOption(Options,'epochs',200);
    Options = defaultOption(Options,'initialEpoch',Options.epochs);
    Options = defaultOption(Options,'retrainEpoch',20);
    Options = defaultOption(Options,'miniBatch',32);
    Options = defaultOption(Options,'lrD',1e-4);
    Options = defaultOption(Options,'lrG',1e-4);
    Options = defaultOption(Options,'gpLambda',10);
    Options = defaultOption(Options,'nCritic',5);
    Options = defaultOption(Options,'collectDiagnostics',false);
    Options = defaultOption(Options,'trainingSigma',0.3);
    Options = defaultOption(Options,'sampleSigma',0.3);
    Options = defaultOption(Options,'generatorHidden',[32 32]);
    Options = defaultOption(Options,'criticHidden',[32 32]);
    Options = defaultOption(Options,'pairMinPairs',8);
    Options = defaultOption(Options,'retrainChange',0.2);
    Options = defaultOption(Options,'retrainGenerations',10);
    Options = defaultOption(Options,'generation',0);
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
    numeric = {'pairMinPairs','retrainGenerations'};
    for i = 1 : numel(numeric)
        name = numeric{i};
        Options.(name) = max(1,round(double(Options.(name))));
    end
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
