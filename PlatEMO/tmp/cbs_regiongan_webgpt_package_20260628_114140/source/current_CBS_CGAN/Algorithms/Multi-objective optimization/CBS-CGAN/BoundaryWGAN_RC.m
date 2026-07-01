function varargout = BoundaryWGAN_RC(action,varargin)
%BOUNDARYWGAN_RC Conditional WGAN-GP for region boundary-cloud sampling.
%
%   Contract:
%       train:             G(z,c) -> x, C(x,c) -> critic score
%       samplebycondition: sample only from externally supplied QueryC

    switch lower(char(action))
        case 'train'
            GAN = varargin{1};
            TrainX = varargin{2};
            TrainC = varargin{3};
            Problem = varargin{4};
            Options = fillOptions(varargin{5});
            varargout{1} = trainBoundaryWGAN(GAN,TrainX,TrainC, ...
                Problem,Options);
        case 'samplebycondition'
            GAN = varargin{1};
            QueryC = varargin{2};
            queryPerCondition = varargin{3};
            Options = fillOptions(varargin{4});
            [varargout{1:nargout}] = sampleByCondition(GAN,QueryC, ...
                queryPerCondition,Options);
        otherwise
            error('CBSRegionWGAN:UnknownGANAction', ...
                'Unknown BoundaryWGAN_RC action: %s',action);
    end
end

function GAN = trainBoundaryWGAN(GAN,TrainX,TrainC,Problem,Options)
    if isempty(TrainX) || isempty(TrainC) || Options.iter <= 0
        return;
    end
    if exist('dlnetwork','file') ~= 2
        error('CBSRegionWGAN:DeepLearningToolboxRequired', ...
            'BoundaryWGAN_RC requires dlnetwork from Deep Learning Toolbox.');
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
    N = size(XScaled,1);
    if isempty(GAN) || ~isfield(GAN,'netG') || ~isfield(GAN,'netC') || ...
            GAN.D ~= D || GAN.M ~= M || GAN.zDim ~= Options.zDim || ...
            ~isfield(GAN,'generatorHidden') || ...
            ~isequal(GAN.generatorHidden,Options.generatorHidden) || ...
            ~isfield(GAN,'criticHidden') || ...
            ~isequal(GAN.criticHidden,Options.criticHidden)
        GAN = initializeWGAN(D,M,Options.zDim,lower,upper, ...
            Options.generatorHidden,Options.criticHidden);
    end

    miniBatch = min(Options.miniBatch,N);
    beta1 = 0.0;
    beta2 = 0.9;
    for iter = 1 : Options.iter
        for cStep = 1 : Options.nCritic
            idx = randi(N,1,miniBatch);
            [GAN,lossInfo] = updateCriticBatch(GAN,XScaled,TrainC, ...
                idx,Options,beta1,beta2);
            GAN.last_critic_loss = lossInfo.lossC;
            GAN.last_gradient_penalty = lossInfo.gp;
            GAN.last_score_real = lossInfo.scoreReal;
            GAN.last_score_fake = lossInfo.scoreFake;
        end
        idx = randi(N,1,miniBatch);
        [GAN,lossG] = updateGeneratorBatch(GAN,TrainC,idx, ...
            Options,beta1,beta2);
        GAN.last_generator_loss = lossG;
    end
end

function [GAN,Info] = updateCriticBatch(GAN,XScaled,TrainC,idx,Options, ...
        beta1,beta2)
    idx = round(double(idx(:)'));
    batchN = numel(idx);
    dlX = dlarray(single(XScaled(idx,:)'),'CB');
    dlC = dlarray(single(TrainC(idx,:)'),'CB');
    dlZ = dlarray(single(latentSamples(Options,Options.zDim,batchN, ...
        "train")'),'CB');
    [gradC,lossC,gp,scoreReal,scoreFake] = dlfeval( ...
        @criticGradients,GAN.netG,GAN.netC,dlX,dlC,dlZ,Options);
    GAN.iterC = GAN.iterC + 1;
    [GAN.netC,GAN.avgC,GAN.avgSqC] = adamupdate(GAN.netC,gradC, ...
        GAN.avgC,GAN.avgSqC,GAN.iterC,Options.lrD,beta1,beta2);
    Info = struct( ...
        'lossC',scalarExtract(lossC), ...
        'gp',scalarExtract(gp), ...
        'scoreReal',scalarExtract(scoreReal), ...
        'scoreFake',scalarExtract(scoreFake));
end

function [GAN,lossValue] = updateGeneratorBatch(GAN,TrainC,idx,Options, ...
        beta1,beta2)
    idx = round(double(idx(:)'));
    batchN = numel(idx);
    dlC = dlarray(single(TrainC(idx,:)'),'CB');
    dlZ = dlarray(single(latentSamples(Options,Options.zDim,batchN, ...
        "train")'),'CB');
    [gradG,lossG] = dlfeval(@generatorGradients, ...
        GAN.netG,GAN.netC,dlC,dlZ);
    GAN.iterG = GAN.iterG + 1;
    [GAN.netG,GAN.avgG,GAN.avgSqG] = adamupdate(GAN.netG,gradG, ...
        GAN.avgG,GAN.avgSqG,GAN.iterG,Options.lrG,beta1,beta2);
    lossValue = scalarExtract(lossG);
end

function [gradC,lossC,gp,scoreRealMean,scoreFakeMean] = criticGradients( ...
        netG,netC,dlX,dlC,dlZ,Options)
    dlFake = forward(netG,[dlZ;dlC]);
    dlFake = dlarray(extractdata(dlFake),'CB');
    scoreReal = forward(netC,[dlX;dlC]);
    scoreFake = forward(netC,[dlFake;dlC]);
    scoreRealMean = mean(scoreReal,'all');
    scoreFakeMean = mean(scoreFake,'all');

    batchN = size(dlX,2);
    epsHat = dlarray(single(rand(1,batchN)),'CB');
    dlHat = epsHat.*dlX + (1 - epsHat).*dlFake;
    scoreHat = forward(netC,[dlHat;dlC]);
    gradHat = dlgradient(sum(scoreHat,'all'),dlHat, ...
        'EnableHigherDerivatives',true);
    gradNorm = sqrt(sum(gradHat.^2,1) + single(1e-12));
    gp = mean((gradNorm - 1).^2,'all');

    lossC = scoreFakeMean - scoreRealMean + single(Options.gpLambda)*gp;
    gradC = dlgradient(lossC,netC.Learnables);
end

function [gradG,lossG] = generatorGradients(netG,netC,dlC,dlZ)
    dlFake = forward(netG,[dlZ;dlC]);
    scoreFake = forward(netC,[dlFake;dlC]);
    lossG = -mean(scoreFake,'all');
    gradG = dlgradient(lossG,netG.Learnables);
end

function [Dec,Info] = sampleByCondition(GAN,QueryC,queryPerCondition,Options)
    Info = emptySampleInfo(size(QueryC,2));
    if isempty(GAN) || ~isfield(GAN,'netG') || isempty(QueryC)
        Dec = zeros(0,0);
        return;
    end
    K = max(1,round(double(queryPerCondition)));
    C = repelem(double(QueryC),K,1);
    n = size(C,1);
    Info.query_index = repelem((1:size(QueryC,1))',K,1);
    Info.condition = C;
    dlC = dlarray(single(C'),'CB');
    Z = latentSamples(Options,GAN.zDim,n,"sample");
    Info.z = Z;
    dlZ = dlarray(single(Z'),'CB');
    dlX = forward(GAN.netG,[dlZ;dlC]);
    XScaled = double(gather(extractdata(dlX)))';
    XScaled = max(-1,min(1,XScaled));
    span = GAN.upper - GAN.lower;
    span(span <= eps) = 1;
    Dec = GAN.lower + (XScaled + 1).*span/2;
    Dec = max(min(Dec,GAN.upper),GAN.lower);
end

function Info = emptySampleInfo(M)
    Info = struct( ...
        'query_index',zeros(0,1), ...
        'condition',zeros(0,M), ...
        'z',zeros(0,0));
end

function Z = latentSamples(Options,zDim,n,purpose)
    if nargin < 4 || isempty(purpose)
        purpose = "sample";
    end
    purpose = lower(strtrim(string(purpose)));
    if isfield(Options,'sampleZ') && ~isempty(Options.sampleZ)
        Z = double(Options.sampleZ);
        if isequal(size(Z),[zDim n])
            Z = Z';
        end
        if ~isequal(size(Z),[n zDim])
            error('CBSRegionWGAN:BadSampleZ', ...
                'Options.sampleZ must be n-by-zDim or zDim-by-n.');
        end
        return;
    end
    mode = "random";
    if purpose == "train" && isfield(Options,'trainZMode') && ...
            ~isempty(Options.trainZMode)
        mode = lower(strtrim(string(Options.trainZMode)));
    elseif isfield(Options,'sampleZMode') && ~isempty(Options.sampleZMode)
        mode = lower(strtrim(string(Options.sampleZMode)));
    end
    switch mode
        case "zero"
            Z = zeros(n,zDim);
        case "random"
            Z = finiteSigma(Options.sigma)*randn(n,zDim);
        otherwise
            error('CBSRegionWGAN:BadSampleZMode', ...
                'Unsupported sampleZMode: %s.',mode);
    end
    Z(~isfinite(Z)) = 0;
end

function GAN = initializeWGAN(D,M,zDim,lower,upper,generatorHidden,criticHidden)
    GAN = struct();
    GAN.D = D;
    GAN.M = M;
    GAN.zDim = zDim;
    GAN.generatorHidden = generatorHidden;
    GAN.criticHidden = criticHidden;
    GAN.lower = lower;
    GAN.upper = upper;
    GAN.netG = createGenerator(M + zDim,D,generatorHidden);
    GAN.netC = createCritic(D + M,criticHidden);
    GAN.avgG = [];
    GAN.avgSqG = [];
    GAN.avgC = [];
    GAN.avgSqC = [];
    GAN.iterG = 0;
    GAN.iterC = 0;
    GAN.last_critic_loss = NaN;
    GAN.last_generator_loss = NaN;
    GAN.last_gradient_penalty = NaN;
    GAN.last_score_real = NaN;
    GAN.last_score_fake = NaN;
end

function netG = createGenerator(inputDim,D,hidden)
    layers = featureInputLayer(inputDim,'Normalization','none','Name','g_in');
    for i = 1 : numel(hidden)
        layers = [layers; ...
            fullyConnectedLayer(hidden(i),'Name',sprintf('g_fc%d',i)); ...
            leakyReluLayer(0.2,'Name',sprintf('g_lrelu%d',i))]; %#ok<AGROW>
    end
    layers = [layers; ...
        fullyConnectedLayer(D,'Name','g_out'); ...
        tanhLayer('Name','g_tanh')];
    netG = dlnetwork(layerGraph(layers));
end

function netC = createCritic(inputDim,hidden)
    layers = featureInputLayer(inputDim,'Normalization','none','Name','c_in');
    for i = 1 : numel(hidden)
        layers = [layers; ...
            fullyConnectedLayer(hidden(i),'Name',sprintf('c_fc%d',i)); ...
            leakyReluLayer(0.2,'Name',sprintf('c_lrelu%d',i))]; %#ok<AGROW>
    end
    layers = [layers; fullyConnectedLayer(1,'Name','c_out')];
    netC = dlnetwork(layerGraph(layers));
end

function Options = fillOptions(Options)
    Options = ensureField(Options,'zDim',6);
    Options = ensureField(Options,'iter',50);
    Options = ensureField(Options,'miniBatch',32);
    Options = ensureField(Options,'lrD',1e-4);
    Options = ensureField(Options,'lrG',1e-4);
    Options = ensureField(Options,'sigma',1.0);
    Options = ensureField(Options,'gpLambda',10);
    Options = ensureField(Options,'nCritic',5);
    Options = ensureField(Options,'sampleZMode',"random");
    Options = ensureField(Options,'generatorHidden',[32 32]);
    Options = ensureField(Options,'criticHidden',[32 32]);
    Options.zDim = max(1,round(double(Options.zDim)));
    Options.iter = max(0,round(double(Options.iter)));
    Options.miniBatch = max(1,round(double(Options.miniBatch)));
    Options.lrD = double(Options.lrD);
    Options.lrG = double(Options.lrG);
    Options.gpLambda = max(0,double(Options.gpLambda));
    Options.nCritic = max(1,round(double(Options.nCritic)));
    Options.generatorHidden = normalizeHiddenVector( ...
        Options.generatorHidden,[32 32]);
    Options.criticHidden = normalizeHiddenVector(Options.criticHidden,[32 32]);
end

function S = ensureField(S,name,value)
    if ~isfield(S,name) || isempty(S.(name))
        S.(name) = value;
    end
end

function hidden = normalizeHiddenVector(hidden,defaultValue)
    if isempty(hidden)
        hidden = defaultValue;
        return;
    end
    hidden = double(hidden(:)');
    hidden = hidden(isfinite(hidden) & hidden > 0);
    if isempty(hidden)
        hidden = defaultValue;
    else
        hidden = max(1,round(hidden));
    end
end

function sigma = finiteSigma(sigma)
    sigma = double(sigma);
    if isempty(sigma) || ~isfinite(sigma)
        sigma = 1.0;
    end
end

function value = scalarExtract(x)
    value = double(gather(extractdata(x)));
    if ~isscalar(value)
        value = mean(value(:));
    end
end
