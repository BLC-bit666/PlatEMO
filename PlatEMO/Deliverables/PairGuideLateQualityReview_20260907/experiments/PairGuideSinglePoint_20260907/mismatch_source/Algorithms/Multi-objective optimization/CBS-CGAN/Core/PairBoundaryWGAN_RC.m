function varargout = PairBoundaryWGAN_RC(action,varargin)
%PAIRBOUNDARYWGAN_RC Single-point conditional WGAN on real boundary endpoints.
%   Conditions are [reference-vector, side]. Pair IDs are metadata only.
%   Deduplicated endpoints are balanced across labels and directions. Generator
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
    if Model.ready && ~isfield(Model,'singlePoint')
        Model = normalizeModelMetadata([]); % Old paired models have different semantics.
    end
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
    Status.endpointVisits = Model.eventEndpointVisits;
    Status.trainingSamples = Model.eventTrainingSamples;
    Status.generatorForwardRows = Model.eventGeneratorRows;
    Status.criticForwardRows = Model.eventCriticRows;
    Status.diagnosticGeneratorRows = 36*Data.count;
    Status.diagnosticCriticRows = 8*Data.count;
    diagnosticTimer = tic;
    Status.postDiagnostics = pairModelDiagnostics(Model,Data,changedRows,Options);
    Status.diagnosticSeconds = Status.diagnosticSeconds+toc(diagnosticTimer);
    Status.reason = "trained";
end

function [Dec,Info] = sampleByCondition(Model,QueryC,Problem,Options)
%SAMPLEBYCONDITION One native candidate per (reference, binary side) request.
    Options = fillOptions(Options); Dec = zeros(0,Problem.D);
    Info = struct('normalized',Dec,'sampleSigma',Options.sampleSigma, ...
        'z',zeros(0,Options.zDim),'conditions',QueryC,'sides',zeros(0,1), ...
        'endpointForwardRows',0,'forwardSeconds',0);
    if isempty(QueryC) || isempty(Model) || ~Model.ready; return; end
    if size(QueryC,2) ~= Model.C || any(~isfinite(QueryC),'all') || ...
            any(~ismember(QueryC(:,end),[0 1]))
        error('CBSPairGuide:BadQueryCondition','Expected finite [w, binary side] conditions.');
    end
    timer = tic; count = size(QueryC,1);
    Z = gaussianNoise(count,Model.zDim,Options.sampleSigma);
    normalized = (generateScaled(Model,Z,QueryC)+1)/2;
    Dec = double(Problem.lower)+normalized.*(double(Problem.upper)-double(Problem.lower));
    Info.normalized = normalized; Info.z = Z; Info.sides = QueryC(:,end);
    Info.endpointForwardRows = count; Info.forwardSeconds = toc(timer);
end

function Model = trainModel(Model,Data,Options,updateBudget)
%TRAINMODEL Independent endpoints; equal label mass and balanced directions.
    Training = prepareTrainingArrays(Data);
    batch = max(2,2*floor(min(Options.miniBatch,Training.count)/2));
    Model.eventEpochs = 0; Model.eventUpdates = 0; Model.eventPairVisits = 0;
    Model.eventEndpointVisits = 0; Model.eventTrainingSamples = Training.count;
    Model.eventGeneratorRows = 0; Model.eventCriticRows = 0; Model.eventFailed = false;
    Model.eventBatchesPerEpoch = ceil(Training.count/batch);
    for update = 1:updateBudget
        idx = balancedEndpointBatch(Training,batch);
        for critic = 1:Options.nCritic
            Model = updateCritic(Model,Training,idx,Options);
            Model.eventGeneratorRows = Model.eventGeneratorRows+batch;
            Model.eventCriticRows = Model.eventCriticRows+(3+double(Options.mismatchFraction>0))*batch;
        end
        Model = updateGenerator(Model,Training,idx,Options);
        Model.eventGeneratorRows = Model.eventGeneratorRows+batch;
        Model.eventCriticRows = Model.eventCriticRows+batch;
        Model.eventUpdates = update;
        Model.eventEndpointVisits = Model.eventEndpointVisits+batch;
    end
    Model.eventEpochs = Model.eventEndpointVisits/Training.count;
end

function T = prepareTrainingArrays(Data)
%PREPARETRAININGARRAYS Sharing an infeasible endpoint never duplicates its mass.
    X = [Data.xF;Data.xI]; C = [Data.cF;Data.cI];
    [~,keep] = unique([X,C(:,end)],'rows','stable'); X = X(keep,:); C = C(keep,:);
    [~,~,groups] = unique(C,'rows');
    T = struct('real',single(2*X'-1),'conditions',single(C'), ...
        'side',C(:,end),'groups',groups,'count',numel(keep));
end

function idx = balancedEndpointBatch(T,count)
%BALANCEDENDPOINTBATCH Equal sides, uniform direction groups within each side.
    idx = zeros(1,count); half = count/2;
    for side = 0:1
        groups = unique(T.groups(T.side == side));
        order = zeros(0,1);
        while numel(order) < half
            order = [order;groups(randperm(numel(groups)))]; %#ok<AGROW>
        end
        for k = 1:half
            rows = find(T.groups == order(k));
            idx(side*half+k) = rows(randi(numel(rows)));
        end
    end
end

function Model = updateCritic(Model,T,idx,O)
%UPDATECRITIC Conditional Wasserstein difference and decision gradient penalty.
    Real = T.real(:,idx); Cond = T.conditions(:,idx); n = numel(idx);
    Z = single(gaussianNoise(Model.zDim,n,O.trainingSigma));
    dlCond = dlarray(Cond,'CB');
    fake = extractdata(generatorForward(Model.netG,dlarray(Z,'CB'),dlCond));
    epsilon = rand(1,n,'single'); hat = epsilon.*Real+(1-epsilon).*fake;
    gradients = dlfeval(@criticGradients,Model.netC,dlarray(Real,'CB'),dlCond, ...
        fake,dlarray(hat,'CB'),single(O.gpLambda),O.mismatchFraction);
    if ~all(cellfun(@(x)all(isfinite(extractdata(x)),'all'),gradients.Value))
        error('CBSPairGuide:NonfiniteTraining','Nonfinite critic gradients.');
    end
    Model.iterC = Model.iterC+1;
    [Model.netC,Model.avgC,Model.avgSqC] = adamupdate(Model.netC,gradients, ...
        Model.avgC,Model.avgSqC,Model.iterC,O.lrD,0,0.9);
end

function gradients = criticGradients(netC,dlReal,dlCond,fake,dlHat,gpLambda,mismatchFraction)
%CRITICGRADIENTS Wasserstein loss plus input gradient penalty.

    dlFake = dlarray(fake,'CB');
    realScore = forward(netC,[dlReal;dlCond]);
    fakeScore = forward(netC,[dlFake;dlCond]);
    hatScore = forward(netC,[dlHat;dlCond]);
    hatGradient = dlgradient(sum(hatScore,'all'),dlHat, ...
        'EnableHigherDerivatives',true);
    gradientNorm = sqrt(sum(hatGradient.^2,1)+single(1e-12));
    penalty = mean((gradientNorm-1).^2,'all');
    negative = mean(fakeScore,'all');
    if mismatchFraction > 0
        n = size(dlCond,2); half = n/2;
        % Swap the two equal side batches: labels are certainly wrong, while
        % the joint (w,s) marginal is identical to the matched batch.
        wrongCond = dlCond(:,[half+1:n,1:half]);
        wrongScore = forward(netC,[dlReal;wrongCond]);
        negative = (1-mismatchFraction)*negative+mismatchFraction*mean(wrongScore,'all');
    end
    loss = negative-mean(realScore,'all')+gpLambda*penalty;
    gradients = dlgradient(loss,netC.Learnables, ...
        'EnableHigherDerivatives',false);
end

function Model = updateGenerator(Model,T,idx,O)
%UPDATEGENERATOR Adversarial distribution learning without endpoint regression.
    C = dlarray(T.conditions(:,idx),'CB');
    Z = dlarray(single(gaussianNoise(Model.zDim,numel(idx),O.trainingSigma)),'CB');
    gradients = dlfeval(@generatorGradients,Model.netG,Model.netC,C,Z);
    if ~all(cellfun(@(x)all(isfinite(extractdata(x)),'all'),gradients.Value))
        error('CBSPairGuide:NonfiniteTraining','Nonfinite generator gradients.');
    end
    Model.iterG = Model.iterG+1;
    [Model.netG,Model.avgG,Model.avgSqG] = adamupdate(Model.netG,gradients, ...
        Model.avgG,Model.avgSqG,Model.iterG,O.lrG,0,0.9);
end

function gradients = generatorGradients(netG,netC,C,Z)
    output = generatorForward(netG,Z,C);
    loss = -mean(forward(netC,[output;C]),'all');
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
        if any(abs(Data.cF(now,:)-Previous.cF(old,:)) > 1e-12) || ...
                any(abs(Data.cI(now,:)-Previous.cI(old,:)) > 1e-12)
            changes(k) = 1; % The actual input condition also changed.
        end
        changedRows(now) = changes(k) > 0;
    end
    delta = sum(changes)/max(1,numel(refs));
    newRegions = numel(setdiff(Data.ref,Previous.ref));
end

function Model = prepareState(Model,D,C,Options)
%PREPARESTATE Warm-start compatible weights or initialize networks.

    required = {'singlePoint','netG','netC','D','C','zDim', ...
        'generatorHidden','criticHidden'};
    compatible = all(isfield(Model,required)) && ...
        Model.D == D && Model.C == C && Model.zDim == Options.zDim && ...
        isequal(Model.generatorHidden,Options.generatorHidden) && ...
        isequal(Model.criticHidden,Options.criticHidden);
    if compatible
        return;
    end
    Model = normalizeModelMetadata([]);
    Model.singlePoint = true;
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
    % Pair differences are not a training target for this single-point model.

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
        'updates',0,'pairVisits',0,'endpointVisits',0,'trainingSamples',0,'trainingPairs',0, ...
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
    Options = defaultOption(Options,'mismatchFraction',0);
    assert(isscalar(Options.mismatchFraction) && isfinite(Options.mismatchFraction) && ...
        Options.mismatchFraction >= 0 && Options.mismatchFraction < 1, ...
        'CBSPairGuide:BadMismatchFraction','Mismatch fraction must be in [0,1).');
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
