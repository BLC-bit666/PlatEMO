function varargout = BoundaryWGAN_RC(action,varargin)
%BOUNDARYWGAN_RC Lean fixed-mainline conditional WGAN-GP.

    switch lower(strtrim(string(action)))
        case "train"
            Options = fillOptions(varargin{5});
            varargout{1} = trainBoundaryWGAN(varargin{1},varargin{2}, ...
                varargin{3},varargin{4},Options);
        case "samplebycondition"
            Options = fillOptions(varargin{3});
            varargout{1} = sampleByCondition(varargin{1},varargin{2},Options);
        otherwise
            error('CBSRegionWGAN:UnknownGANAction', ...
                'Unknown mainline WGAN action: %s.',action);
    end
end

function GAN = trainBoundaryWGAN(GAN,TrainX,TrainC,Problem,Options)
    if isempty(TrainX) || isempty(TrainC) || Options.iter <= 0
        return;
    end
    persistent hasDeepLearningToolbox
    if isempty(hasDeepLearningToolbox)
        hasDeepLearningToolbox = exist('dlnetwork','file') == 2;
    end
    if ~hasDeepLearningToolbox
        error('CBSRegionWGAN:DeepLearningToolboxRequired', ...
            'BoundaryWGAN_RC requires Deep Learning Toolbox.');
    end

    TrainX = double(TrainX);
    TrainC = double(TrainC);
    D = size(TrainX,2);
    M = size(TrainC,2);
    lower = double(Problem.lower);
    upper = double(Problem.upper);
    span = upper - lower;
    span(span <= eps) = 1;
    XScaled = 2*(TrainX - lower)./span - 1;
    XScaled = max(-1,min(1,XScaled));
    XScaledT = single(XScaled');
    TrainCT = single(TrainC');
    GAN = prepareWGANState(GAN,D,M,lower,upper,Options);

    [TrainIdx,HoldoutIdx] = splitTrainingRows(size(XScaled,1));
    trainCount = numel(TrainIdx);
    miniBatch = min(Options.miniBatch,trainCount);
    iterCount = Options.iter;
    nCritic = Options.nCritic;
    zDim = Options.zDim;
    gpLambda = single(Options.gpLambda);
    lrD = Options.lrD;
    lrG = Options.lrG;
    beta1 = 0.0;
    beta2 = 0.9;
    for iter = 1 : iterCount
        for critic = 1 : nCritic
            idx = TrainIdx(randi(trainCount,1,miniBatch));
            GAN = updateCriticBatch(GAN,XScaledT,TrainCT,idx, ...
                zDim,gpLambda,lrD,beta1,beta2);
        end
        idx = TrainIdx(randi(trainCount,1,miniBatch));
        GAN = updateGeneratorBatch(GAN,TrainCT,idx,zDim,lrG,beta1,beta2);
    end

    % The removed legacy diagnostics consumed these random values. Retaining
    % only the RNG advancement keeps the validated optimization trajectory
    % identical without their extra network forwards or metric storage.
    advanceLegacyDiagnosticRNG(TrainIdx,HoldoutIdx,zDim);
end

function GAN = updateCriticBatch(GAN,XScaledT,TrainCT,idx, ...
        zDim,gpLambda,lrD,beta1,beta2)
    batchCount = numel(idx);
    dlX = dlarray(XScaledT(:,idx),'CB');
    dlC = dlarray(TrainCT(:,idx),'CB');
    Z = randn(batchCount,zDim,'single');
    dlZ = dlarray(Z','CB');
    fakeData = extractdata(forward(GAN.netG,[dlZ;dlC]));
    epsilonData = rand(1,batchCount,'single');
    hatData = epsilonData.*extractdata(dlX) + ...
        (1-epsilonData).*fakeData;
    dlHat = dlarray(hatData,'CB');
    gradC = dlfeval(@criticGradients,GAN.netC, ...
        dlX,dlC,fakeData,dlHat,gpLambda);
    GAN.iterC = GAN.iterC + 1;
    [GAN.netC,GAN.avgC,GAN.avgSqC] = adamupdate( ...
        GAN.netC,gradC,GAN.avgC,GAN.avgSqC,GAN.iterC, ...
        lrD,beta1,beta2);
end

function gradC = criticGradients( ...
        netC,dlX,dlC,fakeData,dlHat,gpLambda)
    dlFake = dlarray(fakeData,'CB');
    scoreReal = forward(netC,[dlX;dlC]);
    scoreFake = forward(netC,[dlFake;dlC]);
    scoreHat = forward(netC,[dlHat;dlC]);
    gradHat = dlgradient(sum(scoreHat,'all'),dlHat, ...
        'EnableHigherDerivatives',true);
    gradNorm = sqrt(sum(gradHat.^2,1) + single(1e-12));
    penalty = mean((gradNorm-1).^2,'all');
    loss = mean(scoreFake,'all') - mean(scoreReal,'all') + ...
        gpLambda*penalty;
    gradC = dlgradient(loss,netC.Learnables, ...
        'EnableHigherDerivatives',false);
end

function GAN = updateGeneratorBatch(GAN,TrainCT,idx, ...
        zDim,lrG,beta1,beta2)
    batchCount = numel(idx);
    dlC = dlarray(TrainCT(:,idx),'CB');
    Z = randn(batchCount,zDim,'single');
    dlZ = dlarray(Z','CB');
    gradG = dlfeval(@generatorGradients, ...
        GAN.netG,GAN.netC,dlC,dlZ);
    GAN.iterG = GAN.iterG + 1;
    [GAN.netG,GAN.avgG,GAN.avgSqG] = adamupdate( ...
        GAN.netG,gradG,GAN.avgG,GAN.avgSqG,GAN.iterG, ...
        lrG,beta1,beta2);
end

function gradG = generatorGradients(netG,netC,dlC,dlZ)
    dlFake = forward(netG,[dlZ;dlC]);
    scoreFake = forward(netC,[dlFake;dlC]);
    loss = -mean(scoreFake,'all');
    gradG = dlgradient(loss,netG.Learnables, ...
        'EnableHigherDerivatives',false);
end

function [TrainIdx,HoldoutIdx] = splitTrainingRows(N)
    if N <= 1
        TrainIdx = 1:N;
        HoldoutIdx = zeros(1,0);
        return;
    end
    holdoutCount = min(N-1,max(1,round(0.2*N)));
    order = randperm(N);
    HoldoutIdx = order(1:holdoutCount);
    TrainIdx = order(holdoutCount+1:end);
end

function advanceLegacyDiagnosticRNG(TrainIdx,HoldoutIdx,zDim)
    diagnosticCount = min(numel(TrainIdx),128);
    ignored = randperm(numel(TrainIdx),diagnosticCount); %#ok<NASGU>
    ignored = randn(diagnosticCount,zDim); %#ok<NASGU>
    if ~isempty(HoldoutIdx)
        ignored = randn(numel(HoldoutIdx),zDim); %#ok<NASGU>
    end
end

function Dec = sampleByCondition(GAN,QueryC,Options)
    if isempty(GAN) || ~isstruct(GAN) || ...
            ~isfield(GAN,'netG') || isempty(QueryC)
        Dec = zeros(0,0);
        return;
    end
    C = double(QueryC);
    Z = Options.sampleSigma*randn(size(C,1),GAN.zDim);
    Z(~isfinite(Z)) = 0;
    dlC = dlarray(single(C'),'CB');
    dlZ = dlarray(single(Z'),'CB');
    XScaled = double(extractdata( ...
        forward(GAN.netG,[dlZ;dlC])))';
    XScaled = max(-1,min(1,XScaled));
    span = GAN.upper - GAN.lower;
    span(span <= eps) = 1;
    Dec = GAN.lower + (XScaled+1).*span/2;
    Dec = max(min(Dec,GAN.upper),GAN.lower);
end

function GAN = prepareWGANState(GAN,D,M,lower,upper,Options)
    incompatible = isempty(GAN) || ~isstruct(GAN) || ...
        ~isfield(GAN,'netG') || ~isfield(GAN,'netC') || ...
        GAN.D ~= D || GAN.M ~= M || GAN.zDim ~= Options.zDim || ...
        ~isequal(GAN.generatorHidden,Options.generatorHidden) || ...
        ~isequal(GAN.criticHidden,Options.criticHidden);
    if incompatible
        GAN = initializeWGAN(D,M,lower,upper,Options);
    end
end

function GAN = initializeWGAN(D,M,lower,upper,Options)
    GAN = struct();
    GAN.D = D;
    GAN.M = M;
    GAN.zDim = Options.zDim;
    GAN.generatorHidden = Options.generatorHidden;
    GAN.criticHidden = Options.criticHidden;
    GAN.lower = lower;
    GAN.upper = upper;
    GAN.netG = createGenerator(M+Options.zDim,D,Options.generatorHidden);
    GAN.netC = createCritic(D+M,Options.criticHidden);
    GAN.avgG = [];
    GAN.avgSqG = [];
    GAN.avgC = [];
    GAN.avgSqC = [];
    GAN.iterG = 0;
    GAN.iterC = 0;
end

function netG = createGenerator(inputDim,D,hidden)
    layers = featureInputLayer(inputDim, ...
        'Normalization','none','Name','g_in');
    for i = 1 : numel(hidden)
        layers = [layers; ...
            fullyConnectedLayer(hidden(i), ...
                'Name',sprintf('g_fc%d',i)); ...
            leakyReluLayer(0.2,'Name',sprintf('g_lrelu%d',i))]; %#ok<AGROW>
    end
    layers = [layers; ...
        fullyConnectedLayer(D,'Name','g_out'); ...
        tanhLayer('Name','g_tanh')];
    netG = dlnetwork(layerGraph(layers));
end

function netC = createCritic(inputDim,hidden)
    layers = featureInputLayer(inputDim, ...
        'Normalization','none','Name','c_in');
    for i = 1 : numel(hidden)
        layers = [layers; ...
            fullyConnectedLayer(hidden(i), ...
                'Name',sprintf('c_fc%d',i)); ...
            leakyReluLayer(0.2,'Name',sprintf('c_lrelu%d',i))]; %#ok<AGROW>
    end
    layers = [layers;fullyConnectedLayer(1,'Name','c_out')];
    netC = dlnetwork(layerGraph(layers));
end

function Options = fillOptions(Options)
    Options = defaultOption(Options,'zDim',6);
    Options = defaultOption(Options,'iter',100);
    Options = defaultOption(Options,'miniBatch',32);
    Options = defaultOption(Options,'lrD',1e-4);
    Options = defaultOption(Options,'lrG',1e-4);
    Options = defaultOption(Options,'sampleSigma',0.3);
    Options = defaultOption(Options,'gpLambda',10);
    Options = defaultOption(Options,'nCritic',4);
    Options = defaultOption(Options,'generatorHidden',[32 32]);
    Options = defaultOption(Options,'criticHidden',[32 32]);
    Options.zDim = max(1,round(double(Options.zDim)));
    Options.iter = max(0,round(double(Options.iter)));
    Options.miniBatch = max(1,round(double(Options.miniBatch)));
    Options.lrD = double(Options.lrD);
    Options.lrG = double(Options.lrG);
    Options.sampleSigma = finiteSigma(Options.sampleSigma,0.3);
    Options.gpLambda = max(0,double(Options.gpLambda));
    Options.nCritic = max(1,round(double(Options.nCritic)));
    Options.generatorHidden = hiddenVector(Options.generatorHidden);
    Options.criticHidden = hiddenVector(Options.criticHidden);
end

function S = defaultOption(S,name,value)
    if ~isfield(S,name) || isempty(S.(name))
        S.(name) = value;
    end
end

function value = finiteSigma(value,fallback)
    value = double(value);
    if isempty(value) || ~isfinite(value(1))
        value = fallback;
    else
        value = value(1);
    end
end

function hidden = hiddenVector(hidden)
    hidden = double(hidden(:)');
    hidden = max(1,round(hidden(isfinite(hidden) & hidden > 0)));
    if isempty(hidden)
        hidden = [32 32];
    end
end
