classdef GANCMOModel
% Conditional WGAN-GP with fixed, previously selected training settings.
    methods(Static)
        function Model = train(Model,Data,D,generation)
            if Data.count<8; return; end
            if ~isempty(Model) && changesSince(Data,Model.lastData,D)<.2 && ...
                    generation-Model.lastTrainGeneration<10
                return;
            end
            Previous = Model;
            if isempty(Model)
                C = size(Data.cF,2);
                Model = struct('ready',false,'lastData',[],'lastTrainGeneration',-Inf, ...
                    'avgG',[],'avgSqG',[],'avgC',[],'avgSqC',[],'iterG',0,'iterC',0);
                Model.netG = createGenerator(C+12,D,[8 8]);
                Model.netC = createCritic(D+C,[16 16]);
                updates = 1000;
            else
                updates = 20;
            end
            % The validated implementation trains on a copy of this stream.
            % Keep its event-to-event RNG semantics after removing diagnostics.
            trainingState = rng; rng(trainingState);
            try
                Training = prepareTrainingArrays(Data);
                batch = max(2,2*floor(min(64,Training.count)/2));
                for update = 1:updates
                    idx = balancedEndpointBatch(Training,batch);
                    for critic = 1:5
                        Model = updateCritic(Model,Training,idx);
                    end
                    Model = updateGenerator(Model,Training,idx);
                end
                assertFinite(Model.netG.Learnables.Value);
                assertFinite(Model.netC.Learnables.Value);
            catch err
                if strcmp(err.identifier,'GANCMO:NonfiniteTraining')
                    Model = Previous; return;
                end
                rethrow(err);
            end
            Model.ready = true; Model.lastData = Data; Model.lastTrainGeneration = generation;
        end

        function Dec = sample(Model,Conditions,Problem)
            assert(size(Conditions,2)==Problem.M+1 && all(isfinite(Conditions),'all') && ...
                all(ismember(Conditions(:,end),[0 1])),'GANCMO:BadQueryCondition', ...
                'Expected finite [reference, binary side] conditions.');
            Z = .1*randn(size(Conditions,1),12);
            generated = generatorForward(Model.netG,dlarray(single(Z'),'CB'), ...
                dlarray(single(double(Conditions)'),'CB'));
            normalized = (double(extractdata(generated))'+1)/2;
            Dec = double(Problem.lower)+normalized.*(double(Problem.upper)-double(Problem.lower));
        end
    end
end

function assertFinite(values)
    if ~all(cellfun(@(x)all(isfinite(extractdata(x)),'all'),values))
        error('GANCMO:NonfiniteTraining','Nonfinite network gradients or parameters.');
    end
end

function Model = updateCritic(Model,T,idx)
    persistent acceleratedForward
    if isempty(acceleratedForward); acceleratedForward = dlaccelerate(@generatorForward); end
    Real = T.real(:,idx); Cond = T.conditions(:,idx); n = numel(idx);
    Z = single(.1*randn(12,n)); dlCond = dlarray(Cond,'CB');
    fake = extractdata(acceleratedForward(Model.netG,dlarray(Z,'CB'),dlCond));
    epsilon = rand(1,n,'single'); hat = epsilon.*Real+(1-epsilon).*fake;
    gradients = dlfeval(@criticGradients,Model.netC,dlarray(Real,'CB'),dlCond, ...
        fake,dlarray(hat,'CB'),single(10));
    assertFinite(gradients.Value); Model.iterC = Model.iterC+1;
    [Model.netC,Model.avgC,Model.avgSqC] = adamupdate(Model.netC,gradients, ...
        Model.avgC,Model.avgSqC,Model.iterC,.001,0,.9);
end

function Model = updateGenerator(Model,T,idx)
    C = dlarray(T.conditions(:,idx),'CB'); Z = dlarray(single(.1*randn(12,numel(idx))),'CB');
    gradients = dlfeval(@generatorGradients,Model.netG,Model.netC,C,Z);
    assertFinite(gradients.Value); Model.iterG = Model.iterG+1;
    [Model.netG,Model.avgG,Model.avgSqG] = adamupdate(Model.netG,gradients, ...
        Model.avgG,Model.avgSqG,Model.iterG,.001,0,.9);
end

function T = prepareTrainingArrays(Data)
%PREPARETRAININGARRAYS Sharing an infeasible endpoint never duplicates its mass.
    X = [Data.xF;Data.xI]; C = [Data.cF;Data.cI];
    [~,keep] = unique([X,C(:,end)],'rows','stable'); X = X(keep,:); C = C(keep,:);
    [~,~,groups] = unique(C,'rows');
    T = struct('real',single(2*X'-1),'conditions',single(C'), ...
        'count',numel(keep));
    % Membership is fixed throughout this training event. Cache it once;
    % retain the original sorted group/row order and every RNG call below.
    T.sideGroups = {unique(groups(C(:,end)==0)),unique(groups(C(:,end)==1))};
    T.groupRows = cell(max(groups),1);
    for group = 1:numel(T.groupRows)
        T.groupRows{group} = find(groups==group);
    end
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

function gradients = generatorGradients(netG,netC,C,Z)
    output = generatorForward(netG,Z,C);
    loss = -mean(forward(netC,[output;C]),'all');
    gradients = dlgradient(loss,netG.Learnables,'EnableHigherDerivatives',false);
end

function output = generatorForward(netG,dlZ,dlC)
%GENERATORFORWARD Network tanh is the absolute normalized decision map.

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
    end
    delta = sum(changes)/max(1,numel(refs));
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
