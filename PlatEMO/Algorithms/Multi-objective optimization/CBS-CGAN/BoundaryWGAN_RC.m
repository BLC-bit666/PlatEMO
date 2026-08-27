function varargout = BoundaryWGAN_RC(action,varargin)
%BOUNDARYWGAN_RC Train and sample the reference-conditioned boundary WGAN.
%   TRAIN updates a full warm-start generator and critic from conditioned
%   boundary decisions. The mainline supplies feasible rows at flag 1 and
%   their paired infeasible rows at flag 0.
%   SAMPLEBYCONDITION generates full decision vectors for given conditions.
%   SCOREBYCONDITION evaluates decisions with the matching conditions.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Dispatch the requested operation
    switch lower(strtrim(string(action)))
        case "train"
            Options = fillOptions(varargin{5});
            varargout{1} = trainBoundaryWGAN(varargin{1},varargin{2}, ...
                varargin{3},varargin{4},Options);
        case "samplebycondition"
            Options = fillOptions(varargin{3});
            varargout{1} = sampleByCondition(varargin{1},varargin{2},Options);
        case "scorebycondition"
            varargout{1} = scoreByCondition( ...
                varargin{1},varargin{2},varargin{3});
        otherwise
            error('CBSRegionWGAN:UnknownGANAction', ...
                'Unknown mainline WGAN action: %s.',action);
    end
end

function Score = scoreByCondition(GAN,Dec,Cond)
%SCOREBYCONDITION Score each decision under its own query condition.

    if isempty(Dec)
        Score = zeros(0,1);
        return;
    end
    if isempty(GAN) || ~isstruct(GAN) || ~isfield(GAN,'netC')
        error('CBSRegionWGAN:MissingCritic', ...
            'A trained critic is required for conditional scoring.');
    end
    Dec = double(Dec);
    Cond = double(Cond);
    if size(Dec,1) ~= size(Cond,1) || size(Dec,2) ~= GAN.D || ...
            size(Cond,2) ~= GAN.M
        error('CBSRegionWGAN:BadCriticInput', ...
            'Decision and condition rows must match the trained critic.');
    end
    span = GAN.upper-GAN.lower;
    span(span <= eps) = 1;
    XScaled = 2*(Dec-GAN.lower)./span-1;
    XScaled = max(-1,min(1,XScaled));
    dlX = dlarray(single(XScaled'),'CB');
    dlC = dlarray(single(Cond'),'CB');
    Score = double(extractdata(forward(GAN.netC,[dlX;dlC])))';
    Score = reshape(Score,[],1);
end

function GAN = trainBoundaryWGAN(GAN,TrainX,TrainC,Problem,Options)
%TRAINBOUNDARYWGAN Train on every available conditioned boundary row.

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

    %% Scale and format the fixed training data
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

    %% Train the critic and generator with replacement sampling
    trainCount = size(XScaled,1);
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
            idx = randi(trainCount,1,miniBatch);
            GAN = updateCriticBatch(GAN,XScaledT,TrainCT,idx, ...
                zDim,gpLambda,lrD,beta1,beta2);
        end
        idx = randi(trainCount,1,miniBatch);
        GAN = updateGeneratorBatch(GAN,TrainCT,idx,zDim,lrG,beta1,beta2);
    end
end

function GAN = updateCriticBatch(GAN,XScaledT,TrainCT,idx, ...
        zDim,gpLambda,lrD,beta1,beta2)
%UPDATECRITICBATCH Apply one WGAN-GP critic update.

    batchCount = numel(idx);
    realData = XScaledT(:,idx);
    dlX = dlarray(realData,'CB');
    dlC = dlarray(TrainCT(:,idx),'CB');
    Z = randn(batchCount,zDim,'single');
    dlZ = dlarray(Z','CB');
    fakeData = extractdata(forward(GAN.netG,[dlZ;dlC]));
    epsilonData = rand(1,batchCount,'single');
    hatData = epsilonData.*realData + ...
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
%CRITICGRADIENTS Differentiate Wasserstein loss and gradient penalty.

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
%UPDATEGENERATORBATCH Apply one generator update.

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
%GENERATORGRADIENTS Differentiate the Wasserstein generator objective.

    dlFake = forward(netG,[dlZ;dlC]);
    scoreFake = forward(netC,[dlFake;dlC]);
    loss = -mean(scoreFake,'all');
    gradG = dlgradient(loss,netG.Learnables, ...
        'EnableHigherDerivatives',false);
end

function Dec = sampleByCondition(GAN,QueryC,Options)
%SAMPLEBYCONDITION Generate bounded decisions for reference conditions.

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
%PREPAREWGANSTATE Reuse compatible networks or initialize a new state.

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
%INITIALIZEWGAN Create generator, critic, and Adam optimizer state.

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
%CREATEGENERATOR Build the fully connected conditional generator.

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
%CREATECRITIC Build the fully connected conditional critic.

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
%FILLOPTIONS Fill and normalize WGAN-GP options.

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
%DEFAULTOPTION Fill one missing structure field.

    if ~isfield(S,name) || isempty(S.(name))
        S.(name) = value;
    end
end

function value = finiteSigma(value,fallback)
%FINITESIGMA Normalize the sampling standard deviation.

    value = double(value);
    if isempty(value) || ~isfinite(value(1))
        value = fallback;
    else
        value = value(1);
    end
end

function hidden = hiddenVector(hidden)
%HIDDENVECTOR Normalize a hidden-layer width vector.

    hidden = double(hidden(:)');
    hidden = max(1,round(hidden(isfinite(hidden) & hidden > 0)));
    if isempty(hidden)
        hidden = [32 32];
    end
end
