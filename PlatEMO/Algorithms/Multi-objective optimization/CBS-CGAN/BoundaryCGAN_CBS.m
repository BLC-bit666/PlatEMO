function varargout = BoundaryCGAN_CBS(action,varargin)
%BOUNDARYCGAN_CBS Minimal conditional GAN for boundary point inverse mapping.
%
%   Contract:
%       train:             G(z,c) -> x, D(x,c) -> score
%       samplebycondition: sample only from externally supplied QueryC

    switch lower(char(action))
        case 'train'
            GAN = varargin{1};
            TrainX = varargin{2};
            TrainC = varargin{3};
            Problem = varargin{4};
            Options = fillOptions(varargin{5});
            varargout{1} = trainBoundaryCGAN(GAN,TrainX,TrainC,Problem,Options);
        case 'samplebycondition'
            GAN = varargin{1};
            QueryC = varargin{2};
            queryPerCondition = varargin{3};
            Options = fillOptions(varargin{4});
            [varargout{1:nargout}] = sampleByCondition(GAN,QueryC, ...
                queryPerCondition,Options);
        otherwise
            error('CBSCGAN:UnknownGANAction', ...
                'Unknown BoundaryCGAN_CBS action: %s',action);
    end
end

function GAN = trainBoundaryCGAN(GAN,TrainX,TrainC,Problem,Options)
    if isempty(TrainX) || isempty(TrainC) || Options.iter <= 0
        return;
    end
    if exist('dlnetwork','file') ~= 2
        error('CBSCGAN:DeepLearningToolboxRequired', ...
            'BoundaryCGAN_CBS requires dlnetwork from Deep Learning Toolbox.');
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
    if isempty(GAN) || ~isfield(GAN,'netG') || GAN.D ~= D || ...
            GAN.M ~= M || GAN.zDim ~= Options.zDim || ...
            ~isfield(GAN,'generatorHidden') || ...
            ~isequal(GAN.generatorHidden,Options.generatorHidden) || ...
            ~isfield(GAN,'discriminatorHidden') || ...
            ~isequal(GAN.discriminatorHidden,Options.discriminatorHidden)
        GAN = initializeGAN(D,M,Options.zDim,lower,upper, ...
            Options.generatorHidden,Options.discriminatorHidden);
    end

    miniBatch = min(Options.miniBatch,N);
    beta1 = 0.5;
    beta2 = 0.999;
    trainAdversarial = Options.advWeight > 0;

    if trainAdversarial
        for iter = 1 : Options.dPretrainIter
            idx = randi(N,1,miniBatch);
            dlX = dlarray(single(XScaled(idx,:)'),'CB');
            dlC = dlarray(single(TrainC(idx,:)'),'CB');
            dlZ = dlarray(single(trainingLatentSamples( ...
                Options,Options.zDim,miniBatch)'),'CB');
            [gradD,~] = dlfeval(@discriminatorGradients, ...
                GAN.netG,GAN.netD,dlX,dlC,dlZ,Options);
            GAN.iterD = GAN.iterD + 1;
            [GAN.netD,GAN.avgD,GAN.avgSqD] = adamupdate(GAN.netD,gradD, ...
                GAN.avgD,GAN.avgSqD,GAN.iterD,Options.lrD,beta1,beta2);
        end
    end

    switch Options.trainMode
        case "epoch"
            for epoch = 1 : Options.iter
                order = randperm(N);
                for startIdx = 1 : miniBatch : numel(order)
                    idx = order(startIdx:min(startIdx+miniBatch-1,numel(order)));
                    GAN = updateDiscriminatorSteps(GAN,XScaled,TrainC, ...
                        idx,Options,beta1,beta2,trainAdversarial);
                    GAN = updateGeneratorSteps(GAN,TrainC, ...
                        idx,Options,beta1,beta2);
                end
            end
        otherwise
            for iter = 1 : Options.iter
                if trainAdversarial
                    for dStep = 1 : Options.dSteps
                        idx = randi(N,1,miniBatch);
                        GAN = updateDiscriminatorBatch(GAN,XScaled, ...
                            TrainC,idx,Options,beta1,beta2);
                    end
                end
                for gStep = 1 : Options.gSteps
                    idx = randi(N,1,miniBatch);
                    GAN = updateGeneratorBatch(GAN,TrainC, ...
                        idx,Options,beta1,beta2);
                end
            end
    end
end

function GAN = updateDiscriminatorSteps(GAN,XScaled,TrainC,idx,Options, ...
        beta1,beta2,trainAdversarial)
    if ~trainAdversarial
        return;
    end
    for dStep = 1 : Options.dSteps
        GAN = updateDiscriminatorBatch(GAN,XScaled,TrainC,idx,Options, ...
            beta1,beta2);
    end
end

function GAN = updateGeneratorSteps(GAN,TrainC,idx,Options,beta1,beta2)
    for gStep = 1 : Options.gSteps
        GAN = updateGeneratorBatch(GAN,TrainC,idx,Options, ...
            beta1,beta2);
    end
end

function GAN = updateDiscriminatorBatch(GAN,XScaled,TrainC,idx,Options, ...
        beta1,beta2)
    idx = round(double(idx(:)'));
    batchN = numel(idx);
    dlX = dlarray(single(XScaled(idx,:)'),'CB');
    dlC = dlarray(single(TrainC(idx,:)'),'CB');
    dlZ = dlarray(single(trainingLatentSamples( ...
        Options,Options.zDim,batchN)'),'CB');
    [gradD,~] = dlfeval(@discriminatorGradients, ...
        GAN.netG,GAN.netD,dlX,dlC,dlZ,Options);
    GAN.iterD = GAN.iterD + 1;
    [GAN.netD,GAN.avgD,GAN.avgSqD] = adamupdate(GAN.netD,gradD, ...
        GAN.avgD,GAN.avgSqD,GAN.iterD,Options.lrD,beta1,beta2);
end

function GAN = updateGeneratorBatch(GAN,TrainC,idx,Options, ...
        beta1,beta2)
    idx = round(double(idx(:)'));
    batchN = numel(idx);
    dlC = dlarray(single(TrainC(idx,:)'),'CB');
    dlZ = dlarray(single(trainingLatentSamples( ...
        Options,Options.zDim,batchN)'),'CB');
    [gradG,~] = dlfeval(@generatorGradients, ...
        GAN.netG,GAN.netD,dlC,dlZ,Options);
    GAN.iterG = GAN.iterG + 1;
    [GAN.netG,GAN.avgG,GAN.avgSqG] = adamupdate(GAN.netG,gradG, ...
        GAN.avgG,GAN.avgSqG,GAN.iterG,Options.lrG,beta1,beta2);
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
    Z = latentSamplesForCondition(Options,GAN.zDim,n);
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

function Z = latentSamplesForCondition(Options,zDim,n)
    if isfield(Options,'sampleZ') && ~isempty(Options.sampleZ)
        Z = double(Options.sampleZ);
        if isequal(size(Z),[zDim n])
            Z = Z';
        end
        if ~isequal(size(Z),[n zDim])
            error('CBSCGAN:BadSampleZ', ...
                'Options.sampleZ must be n-by-zDim or zDim-by-n.');
        end
    else
        mode = lower(strtrim(string(Options.sampleZMode)));
        switch mode
            case "zero"
                Z = zeros(n,zDim);
            case "random"
                Z = finiteSigma(Options.sigma)*randn(n,zDim);
            otherwise
                error('CBSCGAN:BadSampleZMode', ...
                    'Unsupported sampleZMode: %s.',mode);
        end
    end
    Z(~isfinite(Z)) = 0;
end

function Z = trainingLatentSamples(Options,zDim,n)
    mode = lower(strtrim(string(Options.trainZMode)));
    switch mode
        case "zero"
            Z = zeros(n,zDim);
        case "random"
            Z = finiteSigma(Options.sigma)*randn(n,zDim);
        otherwise
            error('CBSCGAN:BadTrainZMode', ...
                'Unsupported trainZMode: %s.',mode);
    end
end

function sigma = finiteSigma(sigma)
    sigma = double(sigma);
    if isempty(sigma) || ~isfinite(sigma)
        sigma = 0.05;
    end
end

function GAN = initializeGAN(D,M,zDim,lower,upper,generatorHidden, ...
        discriminatorHidden)
    GAN = struct();
    GAN.D = D;
    GAN.M = M;
    GAN.zDim = zDim;
    GAN.generatorHidden = generatorHidden;
    GAN.discriminatorHidden = discriminatorHidden;
    GAN.lower = lower;
    GAN.upper = upper;
    GAN.netG = createGenerator(M + zDim,D,generatorHidden);
    GAN.netD = createDiscriminator(D + M,discriminatorHidden);
    GAN.avgG = [];
    GAN.avgSqG = [];
    GAN.avgD = [];
    GAN.avgSqD = [];
    GAN.iterG = 0;
    GAN.iterD = 0;
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

function netD = createDiscriminator(inputDim,hidden)
    layers = featureInputLayer(inputDim,'Normalization','none','Name','d_in');
    for i = 1 : numel(hidden)
        layers = [layers; ...
            fullyConnectedLayer(hidden(i),'Name',sprintf('d_fc%d',i)); ...
            leakyReluLayer(0.2,'Name',sprintf('d_lrelu%d',i))]; %#ok<AGROW>
    end
    layers = [layers; ...
        fullyConnectedLayer(1,'Name','d_out'); ...
        sigmoidLayer('Name','d_sigmoid')];
    netD = dlnetwork(layerGraph(layers));
end

function [gradD,lossD] = discriminatorGradients(netG,netD,dlX,dlC,dlZ,Options)
    dlFake = forward(netG,[dlZ;dlC]);
    scoreReal = forward(netD,[dlX;dlC]);
    scoreFake = forward(netD,[dlFake;dlC]);
    lossReal = binaryCrossEntropy(scoreReal,Options.realLabel);
    lossFake = binaryCrossEntropy(scoreFake,0);
    lossD = (lossReal + lossFake)/2;
    gradD = dlgradient(lossD,netD.Learnables);
end

function [gradG,lossG] = generatorGradients( ...
    netG,netD,dlC,dlZ,Options)
    dlFake = forward(netG,[dlZ;dlC]);
    lossG = mean(dlFake*single(0),'all');
    if Options.advWeight > 0
        scoreFake = forward(netD,[dlFake;dlC]);
        advLoss = binaryCrossEntropy(scoreFake,Options.realLabel);
        lossG = lossG + single(Options.advWeight)*advLoss;
    end
    gradG = dlgradient(lossG,netG.Learnables);
end

function loss = binaryCrossEntropy(score,target)
    epsVal = single(1e-7);
    T = score*0 + single(target);
    loss = -mean(T.*log(score + epsVal) + (1 - T).*log(1 - score + epsVal),'all');
end

function Options = fillOptions(Options)
    Options = ensureField(Options,'zDim',2);
    Options = ensureField(Options,'iter',30);
    Options = ensureField(Options,'miniBatch',32);
    Options = ensureField(Options,'lrD',1e-4);
    Options = ensureField(Options,'lrG',2e-4);
    Options = ensureField(Options,'dPretrainIter',0);
    Options = ensureField(Options,'dSteps',1);
    Options = ensureField(Options,'gSteps',1);
    Options = ensureField(Options,'sigma',0.05);
    Options = ensureField(Options,'advWeight',1.0);
    Options = ensureField(Options,'trainZMode',"zero");
    Options = ensureField(Options,'sampleZMode',"zero");
    Options = ensureField(Options,'trainMode',"iter");
    Options = ensureField(Options,'realLabel',0.9);
    Options = ensureField(Options,'generatorHidden',[64 64 64]);
    Options = ensureField(Options,'discriminatorHidden',[64 64 32]);
    Options.advWeight = max(0,double(Options.advWeight));
    Options.generatorHidden = normalizeHiddenVector( ...
        Options.generatorHidden,[64 64 64]);
    Options.discriminatorHidden = normalizeHiddenVector( ...
        Options.discriminatorHidden,[64 64 32]);
    Options.trainZMode = lower(strtrim(string(Options.trainZMode)));
    Options.sampleZMode = lower(strtrim(string(Options.sampleZMode)));
    Options.trainMode = lower(strtrim(string(Options.trainMode)));
    switch Options.trainZMode
        case {"random","zero"}
        otherwise
            error('CBSCGAN:BadTrainZMode', ...
                'Unsupported trainZMode: %s.',Options.trainZMode);
    end
    switch Options.sampleZMode
        case {"random","zero"}
        otherwise
            error('CBSCGAN:BadSampleZMode', ...
                'Unsupported sampleZMode: %s.',Options.sampleZMode);
    end
    switch Options.trainMode
        case {"iter","epoch"}
        otherwise
            error('CBSCGAN:BadTrainMode', ...
                'Unsupported trainMode: %s.',Options.trainMode);
    end
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
