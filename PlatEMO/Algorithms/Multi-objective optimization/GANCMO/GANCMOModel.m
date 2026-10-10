classdef GANCMOModel
% Conditional WGAN-GP, with online training budgets expressed in G updates.
    methods(Static)
        function Model = train(Model,Data,D,generation,S)
            minimum = S.minimumPairs;
            if ~isempty(Model); minimum = S.minimumPairsContinue; end
            if Data.count<minimum; return; end
            if ~isempty(Model) && changesSince(Data,Model.lastData,D)<S.changeThreshold && ...
                    generation-Model.lastTrainGeneration<S.trainInterval
                return;
            end
            Previous = Model;
            if isempty(Model)
                C = size(Data.cF,2);
                Model = struct('ready',false,'lastData',[],'lastTrainGeneration',-Inf, ...
                    'settings',S, ...
                    'avgG',[],'avgSqG',[],'avgC',[],'avgSqC',[],'iterG',0,'iterC',0);
                Model.netG = createGenerator(C,D,S);
                Model.netC = createCritic(D+C,S.criticHidden);
                updates = S.initialG;
            else
                updates = S.updateG;
            end
            % Advance the caller-owned training stream across events.
            % The outer stream scope restores the independent search stream.
            try
                Training = prepareTrainingArrays(Data);
                % Cover direction groups on both sides, regardless of pair count.
                batch = 2*max(cellfun(@numel,Training.sideGroups));
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
                for update = 1:updates
                    idx = balancedEndpointBatch(Training,batch);
                    rate = trainingRate(Model,update,updates);
                    for critic = 1:S.criticSteps
                        State = updateCritic(State,Training,idx,rate);
                    end
                    State = updateGenerator(State,Training,idx,rate);
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
                if strcmp(err.identifier,'GANCMO:NonfiniteTraining')
                    Model = Previous; return;
                end
                rethrow(err);
            end
            % The full retained archive controls the next content-change trigger.
            Model.ready = true; Model.lastData = Data; Model.lastTrainGeneration = generation;
        end

        function Dec = sample(Model,Conditions,Problem)
            assert(size(Conditions,2)==Problem.M+1 && all(isfinite(Conditions),'all') && ...
                all(ismember(Conditions(:,end),[0 1])),'GANCMO:BadQueryCondition', ...
                'Expected finite [reference, binary side] conditions.');
            Z = Model.settings.noiseSigma*randn(size(Conditions,1),Problem.D);
            Conditions = encodeConditions(double(Conditions));
            generated = generatorForward(Model.netG.Learnables.Value,dlarray(single(Z'),'CB'), ...
                dlarray(single(double(Conditions)'),'CB'));
            normalized = (double(extractdata(generated))'+1)/2;
            normalized = min(1,max(0,normalized));
            Dec = double(Problem.lower)+normalized.*(double(Problem.upper)-double(Problem.lower));
        end

        function Scores = score(Model,Dec,Conditions,Problem)
        % Conditional critic ranking only: no oracle calls, training or RNG.
            assert(size(Dec,2)==Problem.D && size(Conditions,2)==Problem.M+1 && ...
                size(Dec,1)==size(Conditions,1) && all(isfinite(Dec),'all'), ...
                'GANCMO:BadScoreInput','Expected finite decisions and matching conditions.');
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
        error('GANCMO:NonfiniteTraining','Nonfinite network gradients or parameters.');
    end
end

function rate = trainingRate(Model,update,updates)
% Each training event decays from 1e-3 to 1e-6.
    rate = Model.settings.learningRate* ...
        Model.settings.learningRateDecay^((update-1)/max(1,updates-1));
end

function Model = updateCritic(Model,T,idx,rate)
    Real = T.real(:,idx); Cond = T.conditions(:,idx); n = numel(idx);
    Z = single(Model.settings.noiseSigma*randn(size(T.real,1),n)); dlCond = dlarray(Cond,'CB');
    fake = extractdata(generatorForward(Model.netG,dlarray(Z,'CB'),dlCond));
    epsilon = rand(1,n,'single'); hat = epsilon.*Real+(1-epsilon).*fake;
    % Differentiate only the critic parameters and interpolated decisions.
    evaluate = @(netC,dlHat) criticGradients(netC,dlarray(Real,'CB'),dlCond, ...
        fake,dlHat,single(10));
    gradients = dlfeval(evaluate,Model.netC,dlarray(hat,'CB'));
    assertFinite(gradients); Model.iterC = Model.iterC+1;
    [Model.netC,Model.avgC,Model.avgSqC] = adamupdate(Model.netC,gradients, ...
        Model.avgC,Model.avgSqC,Model.iterC,rate,0,.9);
end

function Model = updateGenerator(Model,T,idx,rate)
    C = dlarray(T.conditions(:,idx),'CB');
    Z = dlarray(single(Model.settings.noiseSigma*randn(size(T.real,1),numel(idx))),'CB');
    % Critic parameters, sampled inputs and training pools are constants
    % during a G update. Their values still enter the same forward/loss.
    evaluate = @(netG) generatorGradients(netG,Model.netC,C,Z, ...
        T,idx,Model.settings.endpointWeight);
    gradients=dlfeval(evaluate,Model.netG);
    assertFinite(gradients); Model.iterG = Model.iterG+1;
    [Model.netG,Model.avgG,Model.avgSqG] = adamupdate(Model.netG,gradients, ...
        Model.avgG,Model.avgSqG,Model.iterG,rate,0,.9);
end

function T = prepareTrainingArrays(Data)
%PREPARETRAININGARRAYS Sharing an infeasible endpoint never duplicates its mass.
    X = [Data.xF;Data.xI]; C = [Data.cF;Data.cI];
    C = encodeConditions(C);
    [~,keep] = unique([X,C(:,end)],'rows','stable'); X = X(keep,:); C = C(keep,:);
    [~,~,groups] = unique(C,'rows');
    T = struct('real',single(2*X'-1),'conditions',single(C'),'groups',groups);
    % Membership is fixed throughout this training event. Cache it once;
    % retain the original sorted group/row order and every RNG call below.
    T.realDouble = double(T.real);
    T.sideGroups = {unique(groups(C(:,end)==0)),unique(groups(C(:,end)==1))};
    T.groupRows = cell(max(groups),1);
    for group = 1:numel(T.groupRows)
        T.groupRows{group} = find(groups==group);
    end
    % A singleton group has exactly one possible nearest target. Reuse the
    % sampled endpoint directly; retain the original nearest rule otherwise.
    T.multiGroups=find(cellfun(@numel,T.groupRows)>1);
end

function C = encodeConditions(C)
% Exactly M unit-direction coordinates plus the unchanged true 0/1 side bit.
    assert(all(isfinite(C),'all') && all(ismember(C(:,end),[0 1])), ...
        'GANCMO:BadCondition','Expected finite directions and binary true-side labels.');
    direction = C(:,1:end-1);
    C(:,1:end-1) = direction./max(vecnorm(direction,2,2),eps);
end

function idx = balancedEndpointBatch(T,count)
%BALANCEDENDPOINTBATCH Equal sides, uniform direction groups within each side.
    idx = zeros(1,count); half = count/2;
    for side = 0:1
        groups = T.sideGroups{side+1}; groupCount = numel(groups);
        order = zeros(ceil(half/groupCount)*groupCount,1);
        for first = 1:groupCount:numel(order)
            order(first:first+groupCount-1) = groups(randperm(groupCount));
        end
        for k = 1:half
            rows = T.groupRows{order(k)};
            idx(side*half+k) = rows(randi(numel(rows)));
        end
    end
end

function gradients = criticGradients(netC,dlReal,dlCond,fake,dlHat,gpLambda)
%CRITICGRADIENTS Wasserstein loss plus input gradient penalty.

    dlFake = dlarray(fake,'CB');
    realScore = criticForward(netC,[dlReal;dlCond]);
    fakeScore = criticForward(netC,[dlFake;dlCond]);
    hatScore = criticForward(netC,[dlHat;dlCond]);
    hatGradient = dlgradient(sum(hatScore,'all'),dlHat, ...
        'EnableHigherDerivatives',true);
    gradientNorm = sqrt(sum(hatGradient.^2,1)+single(1e-12));
    penalty = mean((gradientNorm-1).^2,'all');
    loss = mean(fakeScore,'all')-mean(realScore,'all')+gpLambda*penalty;
    gradients = dlgradient(loss,netC, ...
        'EnableHigherDerivatives',false);
end

function gradients = generatorGradients(netG,netC,C,Z,T,idx,endpointWeight)
    output = generatorForward(netG,Z,C);
    loss = -mean(criticForward(netC,[output;C]),'all');
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
    loss=loss+single(endpointWeight)*mean(distance,'all');
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

function delta = changesSince(Data,Previous,D)
%CHANGESSINCE Content change over the union of reference directions.

    refs = union(Data.ref,Previous.ref);
    changes = ones(numel(refs),1);
    changedNow = relabelledEndpoints(Data,Previous);
    changedOld = relabelledEndpoints(Previous,Data);
    for k = 1:numel(refs)
        now = find(Data.ref == refs(k));
        old = find(Previous.ref == refs(k));
        if isempty(now) || isempty(old)
            continue;
        end
        oldGap = vecnorm(Previous.xI(old,:)-Previous.xF(old,:),2,2);
        movement = max(pdist2(Data.xF(now,:),Previous.xF(old,:)), ...
            pdist2(Data.xI(now,:),Previous.xI(old,:)));
        movement = movement./max(oldGap',1e-3*sqrt(D));
        changes(k) = min(1,.5*(mean(min(movement,[],2))+mean(min(movement,[],1))));
        if any(changedNow(now)) || any(changedOld(old)) || ...
                ~isequal(unique([Data.cF(now,:),Data.cI(now,:)],'rows'), ...
                unique([Previous.cF(old,:),Previous.cI(old,:)],'rows'))
            changes(k) = 1; % The actual input condition also changed.
        end
    end
    delta = sum(changes)/max(1,numel(refs));
end

function changed = relabelledEndpoints(Data,Other)
% A retained endpoint can change condition without changing the condition set.
% Compare each true side separately; do not relabel the last-training snapshot.
    changed = false(numel(Data.ref),1);
    for side = {'F','I'}
        X = Data.(['x',side{1}]); C = Data.(['c',side{1}]);
        otherX = Other.(['x',side{1}]); otherC = Other.(['c',side{1}]);
        [found,where] = ismember(X,otherX,'rows');
        changed(found) = changed(found) | any(C(found,:)~=otherC(where(found),:),2);
    end
end


function netG = createGenerator(C,D,S)
% One joint input [D-dimensional noise; M-dimensional direction; true side].
    inputDim=D+C;
    layer=fullyConnectedLayer(S.generatorHidden,'Name','pair_guide_g_fc1');
    % Retain condition initialization scales; noise columns use Glorot bounds.
    limit=single(sqrt(6/(inputDim+S.generatorHidden)));
    weights=zeros(S.generatorHidden,inputDim,'single');
    weights(:,1:D)=limit*(2*rand(S.generatorHidden,D,'single')-1);
    weights(:,D+1:end-1)=S.referenceWeightScale*randn(S.generatorHidden,C-1,'single');
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
