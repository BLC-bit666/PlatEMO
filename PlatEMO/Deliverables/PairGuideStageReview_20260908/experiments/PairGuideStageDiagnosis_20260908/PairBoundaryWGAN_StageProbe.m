function varargout = PairBoundaryWGAN_StageProbe(action,varargin)
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
        case "gradientaudit"
            [varargout{1:nargout}]=gradientAudit(varargin{:});
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
    [Candidate,TrainingData] = modelTrainingData(Candidate,Data,Options);
    diagnosticTimer = tic;
    Status.preDiagnostics = pairModelDiagnostics(Candidate,TrainingData,changedRows,Options);
    Status.diagnosticGeneratorRows = 18*Data.count;
    Status.diagnosticCriticRows = 4*Data.count;
    Status.diagnosticSeconds = toc(diagnosticTimer);
    trainingTimer = tic;
    try
        Candidate = trainModel(Candidate,TrainingData,Options,Status.requestedUpdates);
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
    Model.lastTrainingData = TrainingData;
    Status.conditionScale = TrainingData.referenceScale;
    Status.actualCF = networkConditions(TrainingData.cF,Model);
    Status.actualCI = networkConditions(TrainingData.cI,Model);
    Status.useSideCondition = Model.useSideCondition;
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
    Status.postDiagnostics = pairModelDiagnostics(Model,TrainingData,changedRows,Options);
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
    Info.conditions = networkConditions(QueryC,Model);
    Info.useSideCondition = ~isfield(Model,'useSideCondition') || Model.useSideCondition;
    if isfield(Options,'referenceScale')
        Info.conditionScale = Options.referenceScale;
    else
        Info.conditionScale = Model.lastData.referenceScale;
    end
    if isfield(Model,'stableConditionSpan') && Model.stableConditionSpan
        Info.conditionScale.span = Model.conditionSpan;
    end
    Info.endpointForwardRows = count; Info.forwardSeconds = toc(timer);
end

function Model = trainModel(Model,Data,Options,updateBudget)
%TRAINMODEL Independent endpoints; equal label mass and balanced directions.
    Training = prepareTrainingArrays(Data);
    Training.conditions = single(networkConditions(double(Training.conditions'),Model)');
    batch = max(2,2*floor(min(Options.miniBatch,Training.count)/2));
    Model.eventEpochs = 0; Model.eventUpdates = 0; Model.eventPairVisits = 0;
    Model.eventEndpointVisits = 0; Model.eventTrainingSamples = Training.count;
    Model.eventGeneratorRows = 0; Model.eventCriticRows = 0; Model.eventFailed = false;
    Model.eventBatchesPerEpoch = ceil(Training.count/batch);
    for update = 1:updateBudget
        if isfield(Options,'diagnosticBatchIndices')
            idx = Options.diagnosticBatchIndices(update,:);
            assert(numel(idx)==batch && all(idx>=1 & idx<=Training.count & idx==fix(idx)), ...
                'CBSPairGuide:BadDiagnosticBatch','Invalid fixed diagnostic sample schedule.');
        else
            idx = balancedEndpointBatch(Training,batch);
        end
        for critic = 1:Options.nCritic
            Model = updateCritic(Model,Training,idx,Options);
            Model.eventGeneratorRows = Model.eventGeneratorRows+batch;
            Model.eventCriticRows = Model.eventCriticRows+3*batch;
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
        fake,dlarray(hat,'CB'),single(O.gpLambda));
    if ~all(cellfun(@(x)all(isfinite(extractdata(x)),'all'),gradients.Value))
        error('CBSPairGuide:NonfiniteTraining','Nonfinite critic gradients.');
    end
    Model.iterC = Model.iterC+1;
    [Model.netC,Model.avgC,Model.avgSqC] = adamupdate(Model.netC,gradients, ...
        Model.avgC,Model.avgSqC,Model.iterC,O.lrD,0,0.9);
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

function Model = updateGenerator(Model,T,idx,O)
% Diagnostic clone: identical native D/batches/Adam; only G gradient can change.
    C=dlarray(T.conditions(:,idx),'CB');
    Z=dlarray(single(gaussianNoise(Model.zDim,numel(idx),O.trainingSigma)),'CB');
    Real=dlarray(T.real(:,idx),'CB');
    [adv,distance]=dlfeval(@probeBatchGradients,Model.netG,Model.netC,C,Z,Real);
    a=gradientVector(adv); d=gradientVector(distance);
    gradients=adv; lossKind='adversarial';
    if isfield(O,'probeLoss'); lossKind=char(O.probeLoss); end
    if strcmp(lossKind,'distance')
        factor=norm(a)/max(norm(d),1e-12); gradients=distance;
        for k=1:height(gradients); gradients.Value{k}=gradients.Value{k}*single(factor); end
    else
        assert(strcmp(lossKind,'adversarial'));
    end
    offset=0; if isfield(O,'probeBase'); offset=O.probeBase; end
    step=Model.iterG-offset+1;
    observe=step<=5 || mod(step,20)==0;
    if observe
        weights=endpointWeights(T);
        [starBefore,full]=dlfeval(@probeFullDistance,Model.netG,T,weights,Model.zDim);
        f=gradientVector(full); thetaBefore=gradientVector(Model.netG.Learnables);
        starBeforeValue=probeDistanceValue(Model.netG,T,weights,Model.zDim);
    end
    assert(all(isfinite(gradientVector(gradients))));
    Model.iterG=Model.iterG+1;
    [Model.netG,Model.avgG,Model.avgSqG]=adamupdate(Model.netG,gradients, ...
        Model.avgG,Model.avgSqG,Model.iterG,O.lrG,0,0.9);
    if observe
        starAfter=probeDistanceValue(Model.netG,T,weights,Model.zDim);
        delta=gradientVector(Model.netG.Learnables)-thetaBefore;
        used=gradientVector(gradients);
        R=struct('addedUpdates',step,'distanceBefore',starBeforeValue, ...
            'distanceAfter',starAfter,'distanceChange',starAfter-starBeforeValue, ...
            'fullVsAdversarialCosine',dot(f,a)/max(norm(f)*norm(a),1e-30), ...
            'fullVsUsedCosine',dot(f,used)/max(norm(f)*norm(used),1e-30), ...
            'adamDirectionalDerivative',dot(f,delta),'adamStepNorm',norm(delta), ...
            'adversarialGradientNorm',norm(a),'distanceGradientNorm',norm(d), ...
            'usedGradientNorm',norm(used));
        if ~isfield(Model,'probeRecords') || isempty(Model.probeRecords); Model.probeRecords=R;
        else; Model.probeRecords(end+1)=R; end
    end
end

function [adv,distance] = probeBatchGradients(netG,netC,C,Z,Real)
    output=generatorForward(netG,Z,C);
    loss=-mean(forward(netC,[output;C]),'all');
    exact=mean(sqrt(sum((output-Real).^2,1)+single(1e-12)),'all');
    adv=dlgradient(loss,netG.Learnables,'RetainData',true);
    distance=dlgradient(exact,netG.Learnables);
end

function weights=endpointWeights(T)
    weights=zeros(1,T.count);
    for side=0:1
        groups=unique(T.groups(T.side==side));
        for g=reshape(groups,1,[])
            rows=T.groups==g; weights(rows)=.5/numel(groups)/nnz(rows);
        end
    end
    assert(abs(sum(weights)-1)<1e-12);
end

function [loss,gradients]=probeFullDistance(netG,T,weights,zDim)
    C=dlarray(T.conditions,'CB'); Real=dlarray(T.real,'CB');
    output=generatorForward(netG,dlarray(zeros(zDim,T.count,'single'),'CB'),C);
    loss=sum(sqrt(sum((output-Real).^2,1)+single(1e-12)).*single(weights),'all');
    gradients=dlgradient(loss,netG.Learnables);
end

function value=probeDistanceValue(netG,T,weights,zDim)
    output=extractdata(forward(netG,dlarray([zeros(zDim,T.count,'single');T.conditions],'CB')));
    value=sum(sqrt(sum((double(output)-double(T.real)).^2,1)+1e-12).*weights);
end

function v=gradientVector(table)
    values=cellfun(@(x)double(extractdata(x(:))),table.Value,'UniformOutput',false);
    v=vertcat(values{:});
end

function [R,Extra]=gradientAudit(Model,Data)
    T=prepareTrainingArrays(Data); T.conditions=single(networkConditions(double(T.conditions'),Model)');
    weights=endpointWeights(T);
    [star,gstar]=dlfeval(@probeFullDistance,Model.netG,T,weights,Model.zDim);
    [adv,gadv]=dlfeval(@probeFullAdversarial,Model.netG,Model.netC,T,weights,Model.zDim);
    a=gradientVector(gadv); d=gradientVector(gstar);
    R=struct('distance',double(extractdata(star)),'adversarialLoss',double(extractdata(adv)), ...
        'gradientCosine',dot(a,d)/max(norm(a)*norm(d),1e-30), ...
        'starAlongAdversarialDescent',-dot(a,d)/max(norm(a),1e-30), ...
        'starGradientNorm',norm(d),'adversarialGradientNorm',norm(a));
    direction=gadv;
    for k=1:height(direction); direction.Value{k}=-direction.Value{k}/single(max(norm(a),1e-30)); end
    h=.001; plus=Model.netG; minus=Model.netG;
    for k=1:height(direction)
        plus.Learnables.Value{k}=plus.Learnables.Value{k}+single(h)*direction.Value{k};
        minus.Learnables.Value{k}=minus.Learnables.Value{k}-single(h)*direction.Value{k};
    end
    R.finiteDifference=(probeDistanceValue(plus,T,weights,Model.zDim)-probeDistanceValue(minus,T,weights,Model.zDim))/(2*h);
    X=(double(T.real')+1)/2; C=double(T.conditions');
    Y=double(extractdata(forward(Model.netG,dlarray([zeros(Model.zDim,T.count,'single');T.conditions],'CB'))))'; Y=(Y+1)/2;
    [~,~,groups]=unique(C,'rows'); means=zeros(size(X)); variance=0;
    for g=reshape(unique(groups),1,[])
        ix=groups==g; w=weights(ix)'/sum(weights(ix)); mu=sum(X(ix,:).*w,1);
        means(ix,:)=repmat(mu,nnz(ix),1);
        variance=variance+sum(mean((X(ix,:)-mu).^2,2).*weights(ix)');
    end
    mse=sum(mean((Y-X).^2,2).*weights'); bias=sum(mean((Y-means).^2,2).*weights');
    assert(abs(mse-variance-bias)<1e-10);
    R.weightedDecisionMSE=mse; R.conditionalVariance=variance; R.varianceFraction=variance/max(mse,eps);
    R.conditionalBiasMSE=bias; R.trainingEndpoints=T.count;
    R.observedConditions=numel(unique(groups)); counts=accumarray(groups,1);
    R.singletonConditions=nnz(counts==1);
    Extra=struct('X',X,'generatedX',Y,'conditions',C,'weights',weights','group',groups,'counts',counts);
end

function [loss,gradients]=probeFullAdversarial(netG,netC,T,weights,zDim)
    C=dlarray(T.conditions,'CB');
    output=forward(netG,dlarray([zeros(zDim,T.count,'single');T.conditions],'CB'));
    loss=-sum(forward(netC,[output;C]).*single(weights),'all');
    gradients=dlgradient(loss,netG.Learnables);
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

    C = networkConditions(C,Model);
    dlZ = dlarray(single(double(Z)'),'CB');
    dlC = dlarray(single(double(C)'),'CB');
    output = double(extractdata(generatorForward(Model.netG,dlZ,dlC)))';
end

function [Model,Actual] = modelTrainingData(Model,Data,Options)
%MODELTRAININGDATA Keep the archive/gate frame separate from network labels.
    Actual = Data;
    Model.useSideCondition = logical(Options.useSideCondition);
    Model.stableConditionSpan = logical(Options.stableConditionSpan);
    if ~Model.stableConditionSpan; return; end
    assert(all(isfield(Data,{'yF','yI','W','referenceScale'})), ...
        'CBSPairGuide:MissingConditionObjectives','Stable span needs cached objectives and W.');
    if ~isfield(Model,'conditionSpan') || isempty(Model.conditionSpan)
        Model.conditionSpan = Data.referenceScale.span;
    end
    Actual.referenceScale.span = Model.conditionSpan;
    rF = AssignReferenceVectors_CBS(Data.yF,Data.W,Actual.referenceScale);
    rI = AssignReferenceVectors_CBS(Data.yI,Data.W,Actual.referenceScale);
    Actual.cF = [Data.W(rF,:),ones(Data.count,1)];
    Actual.cI = [Data.W(rI,:),zeros(Data.count,1)];
end

function C = networkConditions(C,Model)
%NETWORKCONDITIONS Constant side ablation preserves shape and sample weights.
    if isfield(Model,'useSideCondition') && ~Model.useSideCondition
        C(:,end) = 0;
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
    cF = single(networkConditions(Data.cF,Model)');
    cI = single(networkConditions(Data.cI,Model)');
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
    conditions = single(networkConditions([Data.cF;Data.cI],Model)');
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
    Options = defaultOption(Options,'epochs',1000);
    Options = defaultOption(Options,'initialEpoch',Options.epochs);
    Options = defaultOption(Options,'retrainEpoch',20);
    Options = defaultOption(Options,'miniBatch',32);
    Options = defaultOption(Options,'lrD',1e-3);
    Options = defaultOption(Options,'lrG',1e-3);
    Options = defaultOption(Options,'gpLambda',10);
    Options = defaultOption(Options,'nCritic',5);
    Options = defaultOption(Options,'collectDiagnostics',false);
    Options = defaultOption(Options,'trainingSigma',0);
    Options = defaultOption(Options,'sampleSigma',0);
    Options = defaultOption(Options,'generatorHidden',[32 32]);
    Options = defaultOption(Options,'criticHidden',[32 32]);
    Options = defaultOption(Options,'pairMinPairs',8);
    Options = defaultOption(Options,'retrainChange',0.2);
    Options = defaultOption(Options,'retrainGenerations',10);
    Options = defaultOption(Options,'generation',0);
    Options = defaultOption(Options,'stableConditionSpan',false);
    Options = defaultOption(Options,'useSideCondition',true);
    for name = ["stableConditionSpan","useSideCondition"]
        assert(isscalar(Options.(name)) && ismember(double(Options.(name)),[0 1]), ...
            'CBSPairGuide:BadConditionOption','Condition options must be logical scalars.');
    end
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
