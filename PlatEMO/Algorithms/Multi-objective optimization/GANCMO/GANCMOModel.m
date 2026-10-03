classdef GANCMOModel
% Conditional WGAN-GP, with online training budgets expressed in G updates.
    methods(Static)
        function [Model,Info] = train(Model,Data,D,generation,progress,S)
            Info = struct('attempted',false,'updated',false,'reset',false, ...
                'failed',false,'updatesG',0,'updatesC',0,'batchSize',0);
            reset = ~isempty(Model) && generation-Model.lastResetGeneration>=S.resetInterval && ...
                Data.count>=S.minimumPairs;
            minimum = S.minimumPairs;
            if ~isempty(Model) && ~reset; minimum = S.minimumPairsContinue; end
            if Data.count<minimum; return; end
            if ~isempty(Model) && ~reset && changesSince(Data,Model.lastData,D)<S.changeThreshold && ...
                    generation-Model.lastTrainGeneration<S.trainInterval
                return;
            end
            Previous = Model; Info.attempted = true; Info.reset = reset;
            if isempty(Model) || reset
                C = size(Data.cF,2);
                Model = struct('ready',false,'lastData',[],'lastTrainGeneration',-Inf, ...
                    'lastResetGeneration',generation,'settings',S, ...
                    'avgG',[],'avgSqG',[],'avgC',[],'avgSqC',[],'iterG',0,'iterC',0);
                Model.netG = createGenerator(C+12,D,S.generatorHidden,S.generatorActivation,S.generatorFrequency);
                Model.netC = createCritic(D+C,S.criticHidden);
                updates = S.initialG;
            else
                updates = round(S.updateG(1)+diff(S.updateG)*progress);
            end
            % Advance the caller-owned training stream across events.
            % The outer stream scope restores the independent search stream.
            try
                Training = prepareTrainingArrays(Data);
                batch = max(2,2*floor(min(S.batchSize,Training.count)/2));
                if isfield(S,'fullDirectionBatch') && S.fullDirectionBatch
                    batch = max(batch,2*max(cellfun(@numel,Training.sideGroups)));
                end
                Info.batchSize = batch;
                for update = 1:updates
                    idx = balancedEndpointBatch(Training,batch);
                    rate = trainingRate(Model,update,updates);
                    for critic = 1:S.criticSteps
                        Model = updateCritic(Model,Training,idx,rate);
                        Info.updatesC = Info.updatesC+1;
                    end
                    Model = updateGenerator(Model,Training,idx,rate);
                    Info.updatesG = Info.updatesG+1;
                end
                assertFinite(Model.netG.Learnables.Value);
                assertFinite(Model.netC.Learnables.Value);
            catch err
                if strcmp(err.identifier,'GANCMO:NonfiniteTraining')
                    Info.failed = true; Model = Previous; return;
                end
                rethrow(err);
            end
            Model.ready = true; Model.lastData = Data; Model.lastTrainGeneration = generation;
            Info.updated = true;
        end

        function Dec = sample(Model,Conditions,Problem)
            assert(size(Conditions,2)==Problem.M+1 && all(isfinite(Conditions),'all') && ...
                all(ismember(Conditions(:,end),[0 1])),'GANCMO:BadQueryCondition', ...
                'Expected finite [reference, binary side] conditions.');
            Z = Model.settings.noiseSigma*randn(size(Conditions,1),12);
            Conditions = encodeConditions(double(Conditions));
            generated = generatorForward(Model.netG,dlarray(single(Z'),'CB'), ...
                dlarray(single(double(Conditions)'),'CB'));
            normalized = (double(extractdata(generated))'+1)/2;
            normalized = min(1,max(0,normalized));
            Dec = double(Problem.lower)+normalized.*(double(Problem.upper)-double(Problem.lower));
        end
    end
end

function assertFinite(values)
    if ~all(cellfun(@(x)all(isfinite(extractdata(x)),'all'),values))
        error('GANCMO:NonfiniteTraining','Nonfinite network gradients or parameters.');
    end
end

function rate = trainingRate(Model,update,updates)
% Optionally warm each online event before its final refinement quarter.
    rate = Model.settings.learningRate;
    if isfield(Model.settings,'learningRateSchedule') && strcmp(Model.settings.learningRateSchedule,'event_exp')
        rate = rate*Model.settings.learningRateDecay^((update-1)/max(1,updates-1));
        return;
    end
    position = Model.iterG; count = Model.settings.initialG;
    if isfield(Model.settings,'learningRateSchedule') && ...
            strcmp(Model.settings.learningRateSchedule,'event_tail')
        position = update-1; count = updates;
    end
    if position>=ceil(.75*count)
        rate = rate*Model.settings.learningRateDecay;
    end
end

function Model = updateCritic(Model,T,idx,rate)
    persistent acceleratedForward
    if isempty(acceleratedForward); acceleratedForward = dlaccelerate(@generatorForward); end
    Real = T.real(:,idx); Cond = T.conditions(:,idx); n = numel(idx);
    Z = single(Model.settings.noiseSigma*randn(12,n)); dlCond = dlarray(Cond,'CB');
    fake = extractdata(acceleratedForward(Model.netG,dlarray(Z,'CB'),dlCond));
    epsilon = rand(1,n,'single'); hat = epsilon.*Real+(1-epsilon).*fake;
    gradients = dlfeval(@criticGradients,Model.netC,dlarray(Real,'CB'),dlCond, ...
        fake,dlarray(hat,'CB'),single(10));
    assertFinite(gradients.Value); Model.iterC = Model.iterC+1;
    [Model.netC,Model.avgC,Model.avgSqC] = adamupdate(Model.netC,gradients, ...
        Model.avgC,Model.avgSqC,Model.iterC,rate,0,.9);
end

function Model = updateGenerator(Model,T,idx,rate)
    C = dlarray(T.conditions(:,idx),'CB');
    Z = dlarray(single(Model.settings.noiseSigma*randn(12,numel(idx))),'CB');
    gradients = dlfeval(@generatorGradients,Model.netG,Model.netC,C,Z, ...
        T,idx,Model.settings.endpointWeight);
    assertFinite(gradients.Value); Model.iterG = Model.iterG+1;
    [Model.netG,Model.avgG,Model.avgSqG] = adamupdate(Model.netG,gradients, ...
        Model.avgG,Model.avgSqG,Model.iterG,rate,0,.9);
end

function T = prepareTrainingArrays(Data)
%PREPARETRAININGARRAYS Sharing an infeasible endpoint never duplicates its mass.
    X = [Data.xF;Data.xI]; C = [Data.cF;Data.cI];
    C = encodeConditions(C);
    [~,keep] = unique([X,C(:,end)],'rows','stable'); X = X(keep,:); C = C(keep,:);
    [~,~,groups] = unique(C,'rows');
    T = struct('real',single(2*X'-1),'conditions',single(C'), ...
        'count',numel(keep),'groups',groups);
    % Membership is fixed throughout this training event. Cache it once;
    % retain the original sorted group/row order and every RNG call below.
    T.sideGroups = {unique(groups(C(:,end)==0)),unique(groups(C(:,end)==1))};
    T.groupRows = cell(max(groups),1);
    for group = 1:numel(T.groupRows)
        T.groupRows{group} = find(groups==group);
    end
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

function gradients = generatorGradients(netG,netC,C,Z,T,idx,endpointWeight)
    output = generatorForward(netG,Z,C);
    loss = -mean(forward(netC,[output;C]),'all');
    if endpointWeight>0
        % Match a real endpoint with the SAME direction and true side.
        % Nearest-target assignment preserves alternatives instead of pulling
        % a generated point toward the mean of distinct decision-space modes.
        % The assignment is piecewise constant; gradients flow through output.
        values = double(extractdata(output))';
        target = zeros(size(values'),'single'); batchGroups = T.groups(idx);
        for group = reshape(unique(batchGroups),1,[])
            columns = find(batchGroups==group); rows = T.groupRows{group};
            [~,nearest] = min(pdist2(values(columns,:),double(T.real(:,rows))'),[],2);
            target(:,columns) = T.real(:,rows(nearest));
        end
        distance = sqrt(mean((output-dlarray(target,'CB')).^2,1)+single(1e-8));
        loss = loss+single(endpointWeight)*mean(distance,'all');
    end
    gradients = dlgradient(loss,netG.Learnables,'EnableHigherDerivatives',false);
end

function output = generatorForward(netG,dlZ,dlC)
%GENERATORFORWARD Absolute normalized decision map; sampling enforces the box.

    output = forward(netG,[dlZ;dlC]);
end

function delta = changesSince(Data,Previous,D)
%CHANGESSINCE Content change over the union of reference directions.

    if isempty(Previous) || ~isstruct(Previous) || ~isfield(Previous,'ref')
        delta = 1;
        return;
    end
    refs = union(Data.ref,Previous.ref);
    changes = ones(numel(refs),1);
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
        if ~isequal(unique([Data.cF(now,:),Data.cI(now,:)],'rows'), ...
                unique([Previous.cF(old,:),Previous.cI(old,:)],'rows'))
            changes(k) = 1; % The actual input condition also changed.
        end
    end
    delta = sum(changes)/max(1,numel(refs));
end


function netG = createGenerator(inputDim,D,hidden,activation,frequency)
%CREATEGENERATOR Linear output avoids saturation when learning box endpoints.
% An empty hidden list learns one affine map from noise and conditions.

    layers = featureInputLayer(inputDim, ...
        'Normalization','none','Name','pair_guide_g_in');
    for i = 1 : numel(hidden)
        layer = fullyConnectedLayer(hidden(i),'Name',sprintf('pair_guide_g_fc%d',i));
        if i==1 && strcmp(activation,'sine') && frequency~=1
            % Random continuous features; no endpoint values or direction IDs.
            weights = .001*randn(hidden(i),inputDim,'single');
            weights(:,13:end-1) = frequency*randn(hidden(i),inputDim-13,'single');
            weights(:,end) = single(2*pi)*rand(hidden(i),1,'single');
            layer.Weights = weights;
        end
        layers = [layers;layer]; %#ok<AGROW>
        if strcmp(activation,'sine')
            layers = [layers;functionLayer(@sin,'Formattable',true, ...
                'Name',sprintf('pair_guide_g_sine%d',i))]; %#ok<AGROW>
        else
            layers = [layers;leakyReluLayer(0.2, ...
                'Name',sprintf('pair_guide_g_lrelu%d',i))]; %#ok<AGROW>
        end
    end
    layers = [layers;fullyConnectedLayer(D,'Name','pair_guide_g_out')];
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
