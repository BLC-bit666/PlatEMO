classdef GANCMO_V2Model
% Conditional WGAN-GP trained in complete shuffled endpoint epochs.
    methods(Static)
        function Model = train(Model,Data,D,S)
            % The caller checks archive change; this method enforces training eligibility.
            if numel(unique(Data.ref))<S.minimumDirections; return; end
            if isempty(Data.xF) || isempty(Data.xI) || ~all(isfinite([Data.xF;Data.xI]),'all'); return; end
            Previous = Model; initial = isempty(Model);
            if isempty(Model)
                C = 1;
                Model = struct('lastData',[],'settings',S, ...
                    'avgG',[],'avgSqG',[],'avgC',[],'avgSqC',[],'iterG',0,'iterC',0);
                Model.netG = createGenerator(C,D,S);
                Model.netC = createCritic(D+C,S.criticHidden);
                epochs = S.initialEpochs;
            else
                epochs = S.updateEpochs;
            end
            % Advance the caller-owned training stream across events.
            % The outer stream scope restores the independent search stream.
            try
                Training = prepareTrainingArrays(Data);
                if ~initial
                    epochs = min(epochs,floor(S.maxContinuationGUpdates/Training.batchesPerEpoch));
                end
                assert(epochs>=1,'GANCMO_V2:EpochBudget','The budget must allow one complete epoch.');
                % Network topology stays fixed. Reuse parameter cells during this
                % event instead of traversing network layers at every Adam step.
                State = Model;
                State.netG = Model.netG.Learnables.Value;
                State.netC = Model.netC.Learnables.Value;
                for field = {'avgG','avgSqG','avgC','avgSqC'}
                    if ~isempty(State.(field{1}))
                        State.(field{1}) = State.(field{1}).Value;
                    end
                end
                % Both networks use the same fixed rate in every training event.
                rate = Model.settings.learningRate;
                for epoch = 1:epochs
                    batches = epochBatches(Training);
                    for batch = 1:Training.batchesPerEpoch
                        idx = batches{batch}; weights = Training.batchWeights{batch};
                        for critic = 1:S.criticSteps
                            State = updateCritic(State,Training,idx,weights,rate);
                        end
                        State = updateGenerator(State,Training,idx,weights,rate);
                    end
                end
                assertFinite(State.netG); assertFinite(State.netC);
                Model.netG.Learnables.Value = State.netG;
                Model.netC.Learnables.Value = State.netC;
                for field = {'avgG','avgSqG','avgC','avgSqC'}
                    if endsWith(field{1},'G'); values=Model.netG.Learnables;
                    else; values=Model.netC.Learnables; end
                    values.Value=State.(field{1}); Model.(field{1})=values;
                end
                Model.iterG=State.iterG; Model.iterC=State.iterC;
            catch err
                if strcmp(err.identifier,'GANCMO_V2:NonfiniteTraining')
                    Model = Previous; return;
                end
                rethrow(err);
            end
            % Keep the last successful training snapshot; failures roll back it too.
            Model.lastData = Data;
        end

        function Dec = sample(Model,Conditions,Problem,Noise)
            assert(size(Conditions,2)==1 && all(isfinite(Conditions),'all') && ...
                all(ismember(Conditions,[0 1])),'GANCMO_V2:BadQueryCondition', ...
                'Expected one binary feasibility condition; no reference input.');
            if nargin<4
                Noise = Model.settings.noiseSigma*randn(size(Conditions,1),Problem.D);
            else
                assert(isequal(size(Noise),[size(Conditions,1),Problem.D]) && all(isfinite(Noise),'all'), ...
                    'GANCMO_V2:BadQueryNoise','Expected one finite D-dimensional noise row per condition.');
            end
            Conditions = encodeConditions(double(Conditions));
            generated = generatorForward(Model.netG.Learnables.Value,dlarray(single(Noise'),'CB'), ...
                dlarray(single(double(Conditions)'),'CB'));
            normalized = (double(extractdata(generated))'+1)/2;
            normalized = min(1,max(0,normalized));
            Dec = double(Problem.lower)+normalized.*(double(Problem.upper)-double(Problem.lower));
        end

        function Scores = score(Model,Dec,Conditions,Problem)
        % Conditional critic ranking only: no oracle calls, training or RNG.
            assert(size(Dec,2)==Problem.D && size(Conditions,2)==1 && ...
                size(Dec,1)==size(Conditions,1) && all(isfinite(Dec),'all'), ...
                'GANCMO_V2:BadScoreInput','Expected finite decisions and matching conditions.');
            Conditions = encodeConditions(double(Conditions));
            if isempty(Dec); Scores = zeros(0,1); return; end
            span = double(Problem.upper)-double(Problem.lower); span(span<=eps) = 1;
            normalized = (double(Dec)-double(Problem.lower))./span;
            input = dlarray(single([2*normalized-1,Conditions]'),'CB');
            Scores = double(extractdata(criticForward(Model.netC.Learnables.Value,input)))';
        end
    end
end

function assertFinite(values)
    if ~all(cellfun(@(x)all(isfinite(extractdata(x)),'all'),values))
        error('GANCMO_V2:NonfiniteTraining','Nonfinite network gradients or parameters.');
    end
end

function Model = updateCritic(Model,T,idx,weights,rate)
    Real = T.real(:,idx); Cond = T.conditions(:,idx); n = numel(idx);
    Z = single(Model.settings.noiseSigma*randn(size(T.real,1),n)); dlCond = dlarray(Cond,'CB');
    fake = extractdata(generatorForward(Model.netG,dlarray(Z,'CB'),dlCond));
    epsilon = rand(1,n,'single'); hat = epsilon.*Real+(1-epsilon).*fake;
    % Differentiate only the critic parameters and interpolated decisions.
    evaluate = @(netC,dlHat) criticGradients(netC,dlarray(Real,'CB'),dlCond, ...
        fake,dlHat,weights,single(10));
    gradients = dlfeval(evaluate,Model.netC,dlarray(hat,'CB'));
    assertFinite(gradients); Model.iterC = Model.iterC+1;
    [Model.netC,Model.avgC,Model.avgSqC] = adamupdate(Model.netC,gradients, ...
        Model.avgC,Model.avgSqC,Model.iterC,rate,0,.9);
end

function Model = updateGenerator(Model,T,idx,weights,rate)
    C = dlarray(T.conditions(:,idx),'CB');
    Z = dlarray(single(Model.settings.noiseSigma*randn(size(T.real,1),numel(idx))),'CB');
    % Critic parameters, sampled inputs and training pools are constants
    % during a G update. Their values still enter the same forward/loss.
    evaluate = @(netG) generatorGradients(netG,Model.netC,C,Z, ...
        T,idx,weights,Model.settings.endpointWeight);
    gradients=dlfeval(evaluate,Model.netG);
    assertFinite(gradients); Model.iterG = Model.iterG+1;
    [Model.netG,Model.avgG,Model.avgSqG] = adamupdate(Model.netG,gradients, ...
        Model.avgG,Model.avgSqG,Model.iterG,rate,0,.9);
end

function T = prepareTrainingArrays(Data)
%PREPARETRAININGARRAYS Sharing an infeasible endpoint never duplicates its mass.
    X = [Data.xF;Data.xI]; Meta = [Data.cF;Data.cI];
    % Full archive metadata controls the unchanged endpoint schedule only.
    assert(all(isfinite(Meta),'all') && all(ismember(Meta(:,end),[0 1])), ...
        'GANCMO_V2:BadCondition','Expected valid archive direction/side metadata.');
    direction = Meta(:,1:end-1);
    Meta(:,1:end-1) = direction./max(vecnorm(direction,2,2),eps);
    [~,keep] = unique([X,Meta(:,end)],'rows','stable'); X = X(keep,:); Meta = Meta(keep,:);
    [~,~,scheduleGroups] = unique(Meta,'rows');
    % Both networks and endpoint matching receive only the retained condition.
    C = Meta(:,end);
    [~,~,groups] = unique(C,'rows');
    T = struct('real',single(2*X'-1),'conditions',single(C'),'groups',groups);
    % Membership is fixed throughout this training event. Cache it once;
    % retain the original sorted group/row order and every RNG call below.
    T.realDouble = double(T.real);
    T.sideGroups = {unique(scheduleGroups(Meta(:,end)==0)),unique(scheduleGroups(Meta(:,end)==1))};
    T.groupRows = cell(max(groups),1);
    for group = 1:numel(T.groupRows)
        T.groupRows{group} = find(groups==group);
    end
    % A singleton group has exactly one possible nearest target. Reuse the
    % sampled endpoint directly; retain the original nearest rule otherwise.
    T.multiGroups=find(cellfun(@numel,T.groupRows)>1);
    T.sideRows={find(Meta(:,end)==1)',find(Meta(:,end)==0)'};
    countF=numel(T.sideRows{1}); countI=numel(T.sideRows{2});
    nominal=2*max(cellfun(@numel,T.sideGroups));
    T.batchesPerEpoch=min(ceil((countF+countI)/nominal),min(countF,countI));
    K=T.batchesPerEpoch;
    T.cutsF=floor((0:K)*countF/K); T.cutsI=floor((0:K)*countI/K);
    T.batchWeights=cell(1,K);
    for k=1:K
        nF=T.cutsF(k+1)-T.cutsF(k); nI=T.cutsI(k+1)-T.cutsI(k);
        T.batchWeights{k}=single([repmat((nF+nI)/(2*nF),1,nF),repmat((nF+nI)/(2*nI),1,nI)]);
    end
end

function C = encodeConditions(C)
% Only the unchanged scalar 0/1 side enters either network.
    assert(all(isfinite(C),'all') && size(C,2)==1 && all(ismember(C,[0 1])), ...
        'GANCMO_V2:BadCondition','Invalid reduced condition.');
end

function batches = epochBatches(T)
% Every unique endpoint is scheduled once, including both tail slices.
    orderF=T.sideRows{1}(randperm(numel(T.sideRows{1})));
    orderI=T.sideRows{2}(randperm(numel(T.sideRows{2})));
    batches=cell(1,T.batchesPerEpoch);
    for k=1:T.batchesPerEpoch
        batches{k}=[orderF(T.cutsF(k)+1:T.cutsF(k+1)),orderI(T.cutsI(k)+1:T.cutsI(k+1))];
    end
end

function gradients = criticGradients(netC,dlReal,dlCond,fake,dlHat,weights,gpLambda)
%CRITICGRADIENTS Wasserstein loss plus input gradient penalty.

    dlFake = dlarray(fake,'CB');
    realScore = criticForward(netC,[dlReal;dlCond]);
    fakeScore = criticForward(netC,[dlFake;dlCond]);
    hatScore = criticForward(netC,[dlHat;dlCond]);
    hatGradient = dlgradient(sum(hatScore,'all'),dlHat, ...
        'EnableHigherDerivatives',true);
    gradientNorm = sqrt(sum(hatGradient.^2,1)+single(1e-12));
    penalty = mean((gradientNorm-1).^2.*weights,'all');
    loss = mean((fakeScore-realScore).*weights,'all')+gpLambda*penalty;
    gradients = dlgradient(loss,netC, ...
        'EnableHigherDerivatives',false);
end

function gradients = generatorGradients(netG,netC,C,Z,T,idx,weights,endpointWeight)
    output = generatorForward(netG,Z,C);
    loss = -mean(criticForward(netC,[output;C]).*weights,'all');
    % Singleton condition groups have no nearest-target search to perform.
    % Multiple endpoints retain the original double-precision distance rule.
    values=double(extractdata(output))';
    target=T.real(:,idx); batchGroups=T.groups(idx);
    for group=reshape(T.multiGroups,1,[])
        columns=find(batchGroups==group);
        if isempty(columns); continue; end
        rows=T.groupRows{group};
        [~,nearest]=min(pdist2(values(columns,:),T.realDouble(:,rows)'),[],2);
        target(:,columns)=T.real(:,rows(nearest));
    end
    distance=sqrt(mean((output-dlarray(target,'CB')).^2,1)+single(1e-8));
    loss=loss+single(endpointWeight)*mean(distance.*weights,'all');
    gradients=dlgradient(loss,netG,'EnableHigherDerivatives',false);
end

function output = generatorForward(V,dlZ,dlC)
% Cache only the forward primitives; gradient/loss evaluation is unchanged.
    persistent forward
    if isempty(forward); forward=dlaccelerate(@generatorEvaluation); end
    output=forward(V,dlZ,dlC);
end

function output = generatorEvaluation(V,dlZ,dlC)
% Noise and condition jointly enter the learned nonlinear hidden layer.
    output=fullyconnect(sin(fullyconnect([dlZ;dlC],V{1},V{2})),V{3},V{4});
end

function netG = createGenerator(C,D,S)
% One joint input [D-dimensional noise; true side].
    inputDim=D+C;
    layer=fullyConnectedLayer(S.generatorHidden,'Name','pair_guide_g_fc1');
    % Retain condition initialization scales; noise columns use Glorot bounds.
    limit=single(sqrt(6/(inputDim+S.generatorHidden)));
    weights=zeros(S.generatorHidden,inputDim,'single');
    weights(:,1:D)=limit*(2*rand(S.generatorHidden,D,'single')-1);
    weights(:,end)=single(2*pi)*rand(S.generatorHidden,1,'single');
    layer.Weights=weights;
    layers=[featureInputLayer(inputDim,'Normalization','none','Name','pair_guide_g_in'); ...
        layer;functionLayer(@sin,'Formattable',true,'Name','pair_guide_g_sine1'); ...
        fullyConnectedLayer(D,'Name','pair_guide_g_out')];
    netG=dlnetwork(layerGraph(layers));
end

function netC = createCritic(inputDim,hidden)
% Conditional endpoint critic; its scalar output is unrestricted.
    layers=[featureInputLayer(inputDim,'Normalization','none','Name','pair_guide_c_in'); ...
        fullyConnectedLayer(hidden,'Name','pair_guide_c_fc1'); ...
        leakyReluLayer(.2,'Name','pair_guide_c_lrelu1'); ...
        fullyConnectedLayer(1,'Name','pair_guide_c_out')];
    netC=dlnetwork(layerGraph(layers));
end

function output = criticForward(V,X)
    persistent forward
    if isempty(forward); forward=dlaccelerate(@criticEvaluation); end
    output=forward(V,X);
end

function output = criticEvaluation(V,X)
% Same fully connected / leaky ReLU primitives, without graph dispatch.
    output=fullyconnect(leakyrelu(fullyconnect(X,V{1},V{2}),.2),V{3},V{4});
end
