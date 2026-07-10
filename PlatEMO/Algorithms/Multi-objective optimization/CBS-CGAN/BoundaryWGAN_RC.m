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

    [TrainIdx,HoldoutIdx] = splitWGANTrainHoldout(N);
    trainPoolN = numel(TrainIdx);
    miniBatch = min(Options.miniBatch,trainPoolN);
    beta1 = 0.0;
    beta2 = 0.9;
    for iter = 1 : Options.iter
        for cStep = 1 : Options.nCritic
            idx = TrainIdx(randi(trainPoolN,1,miniBatch));
            [GAN,lossInfo] = updateCriticBatch(GAN,XScaled,TrainC, ...
                idx,Options,beta1,beta2);
            GAN.last_critic_loss = lossInfo.lossC;
            GAN.last_gradient_penalty = lossInfo.gp;
            GAN.last_score_real = lossInfo.scoreReal;
            GAN.last_score_fake = lossInfo.scoreFake;
        end
        idx = TrainIdx(randi(trainPoolN,1,miniBatch));
        [GAN,lossG] = updateGeneratorBatch(GAN,TrainC,idx, ...
            Options,beta1,beta2);
        GAN.last_generator_loss = lossG;
    end
    GAN = appendWGANTrainingDiagnostics(GAN,XScaled,TrainC,Options, ...
        TrainIdx,HoldoutIdx);
end

function [TrainIdx,HoldoutIdx] = splitWGANTrainHoldout(N)
    N = max(0,round(double(N)));
    if N <= 1
        TrainIdx = 1:N;
        HoldoutIdx = zeros(1,0);
        return;
    end
    holdoutN = max(1,round(0.2*N));
    holdoutN = min(N - 1,holdoutN);
    perm = randperm(N);
    HoldoutIdx = perm(1:holdoutN);
    TrainIdx = perm(holdoutN+1:end);
end

function GAN = appendWGANTrainingDiagnostics(GAN,XScaled,TrainC,Options, ...
        TrainIdx,HoldoutIdx)
    N = size(XScaled,1);
    if N <= 0 || isempty(TrainIdx)
        return;
    end
    TrainIdx = round(double(TrainIdx(:)'));
    HoldoutIdx = round(double(HoldoutIdx(:)'));
    diagN = min(numel(TrainIdx),128);
    trainIdx = TrainIdx(randperm(numel(TrainIdx),diagN));
    [scoreReal,scoreFake] = scoreRealAndFake(GAN,XScaled,TrainC, ...
        trainIdx,Options);
    GAN.critic_train_real_score_mean = finiteSummary(scoreReal,@mean);
    GAN.critic_train_fake_score_mean = finiteSummary(scoreFake,@mean);
    GAN.critic_train_gap = GAN.critic_train_real_score_mean - ...
        GAN.critic_train_fake_score_mean;
    GAN.critic_train_diag_count = double(numel(trainIdx));

    if isempty(HoldoutIdx)
        GAN.critic_holdout_real_score_mean = NaN;
        GAN.critic_holdout_fake_score_mean = NaN;
        GAN.critic_holdout_gap = NaN;
        GAN.critic_holdout_count = 0;
    else
        [scoreReal,scoreFake] = scoreRealAndFake(GAN,XScaled,TrainC, ...
            HoldoutIdx,Options);
        GAN.critic_holdout_real_score_mean = finiteSummary(scoreReal,@mean);
        GAN.critic_holdout_fake_score_mean = finiteSummary(scoreFake,@mean);
        GAN.critic_holdout_gap = GAN.critic_holdout_real_score_mean - ...
            GAN.critic_holdout_fake_score_mean;
        GAN.critic_holdout_count = double(numel(HoldoutIdx));
    end
end

function [scoreReal,scoreFake] = scoreRealAndFake(GAN,XScaled,TrainC, ...
        idx,Options)
    idx = round(double(idx(:)'));
    if isempty(idx)
        scoreReal = zeros(0,1);
        scoreFake = zeros(0,1);
        return;
    end
    batchN = numel(idx);
    dlX = dlarray(single(XScaled(idx,:)'),'CB');
    dlC = dlarray(single(TrainC(idx,:)'),'CB');
    dlZ = dlarray(single(latentSamples(Options,GAN.zDim,batchN, ...
        "train")'),'CB');
    dlFake = forward(GAN.netG,[dlZ;dlC]);
    scoreReal = forward(GAN.netC,[dlX;dlC]);
    scoreFake = forward(GAN.netC,[dlFake;dlC]);
    scoreReal = double(gather(extractdata(scoreReal)))';
    scoreFake = double(gather(extractdata(scoreFake)))';
    scoreReal = scoreReal(:);
    scoreFake = scoreFake(:);
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
    screenK = max(1,round(double(Options.prescreenMultiplier)));
    Info.prescreen_multiplier = double(screenK);
    if screenK <= 1
        Z = latentSamples(Options,GAN.zDim,n,"sample");
        Dec = generateDecisions(GAN,C,Z);
        Scores = criticScores(GAN,Dec,C);
        Info.z = Z;
        Info.prescreen_candidate_count = double(n);
        Info.prescreen_selected_count = double(n);
        Info.generated_critic_score = Scores;
        Info.prescreen_selected_score = Scores;
        Info.prescreen_raw_score = Scores;
        Info.prescreen_raw_selected = true(size(Scores));
        Info.prescreen_raw_condition_index = (1:n)';
        Info.prescreen_score_min = finiteSummary(Scores,@min);
        Info.prescreen_score_max = finiteSummary(Scores,@max);
        Info.prescreen_score_mean = finiteSummary(Scores,@mean);
        return;
    end

    CandidateC = repelem(C,screenK,1);
    CandidateZ = latentSamples(Options,GAN.zDim,size(CandidateC,1), ...
        "sample");
    CandidateDec = generateDecisions(GAN,CandidateC,CandidateZ);
    Scores = criticScores(GAN,CandidateDec,CandidateC);
    selected = zeros(n,1);
    for i = 1 : n
        rows = (i - 1)*screenK + (1:screenK);
        [~,best] = max(Scores(rows));
        selected(i) = rows(best);
    end
    Dec = CandidateDec(selected,:);
    Info.z = CandidateZ(selected,:);
    Info.prescreen_candidate_count = double(size(CandidateDec,1));
    Info.prescreen_selected_count = double(size(Dec,1));
    Info.prescreen_selected_score = Scores(selected);
    Info.generated_critic_score = Scores(selected);
    Info.prescreen_raw_score = Scores;
    Info.prescreen_raw_selected = false(size(Scores));
    Info.prescreen_raw_selected(selected) = true;
    Info.prescreen_raw_condition_index = repelem((1:n)',screenK,1);
    Info.prescreen_score_min = finiteSummary(Scores,@min);
    Info.prescreen_score_max = finiteSummary(Scores,@max);
    Info.prescreen_score_mean = finiteSummary(Scores,@mean);
end

function Info = emptySampleInfo(M)
    Info = struct( ...
        'query_index',zeros(0,1), ...
        'condition',zeros(0,M), ...
        'z',zeros(0,0), ...
        'prescreen_multiplier',1, ...
        'prescreen_candidate_count',0, ...
        'prescreen_selected_count',0, ...
        'prescreen_selected_score',zeros(0,1), ...
        'generated_critic_score',zeros(0,1), ...
        'prescreen_raw_score',zeros(0,1), ...
        'prescreen_raw_selected',false(0,1), ...
        'prescreen_raw_condition_index',zeros(0,1), ...
        'prescreen_score_min',NaN, ...
        'prescreen_score_max',NaN, ...
        'prescreen_score_mean',NaN);
end

function Dec = generateDecisions(GAN,C,Z)
    dlC = dlarray(single(C'),'CB');
    dlZ = dlarray(single(Z'),'CB');
    dlX = forward(GAN.netG,[dlZ;dlC]);
    XScaled = double(gather(extractdata(dlX)))';
    XScaled = max(-1,min(1,XScaled));
    span = GAN.upper - GAN.lower;
    span(span <= eps) = 1;
    Dec = GAN.lower + (XScaled + 1).*span/2;
    Dec = max(min(Dec,GAN.upper),GAN.lower);
end

function Scores = criticScores(GAN,Dec,C)
    span = GAN.upper - GAN.lower;
    span(span <= eps) = 1;
    XScaled = 2*(double(Dec) - GAN.lower)./span - 1;
    XScaled = max(-1,min(1,XScaled));
    dlX = dlarray(single(XScaled'),'CB');
    dlC = dlarray(single(double(C)'),'CB');
    dlScore = forward(GAN.netC,[dlX;dlC]);
    Scores = double(gather(extractdata(dlScore)))';
    Scores = Scores(:);
end

function value = finiteSummary(values,fn)
    values = double(values(:));
    values = values(isfinite(values));
    if isempty(values)
        value = NaN;
    else
        value = fn(values);
    end
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
            Z = latentSigma(Options,purpose)*randn(n,zDim);
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
    Options = ensureField(Options,'trainSigma',[]);
    Options = ensureField(Options,'sampleSigma',[]);
    Options = ensureField(Options,'gpLambda',10);
    Options = ensureField(Options,'nCritic',5);
    Options = ensureField(Options,'trainZMode',"random");
    Options = ensureField(Options,'sampleZMode',"random");
    Options = ensureField(Options,'prescreenMultiplier',1);
    Options = ensureField(Options,'generatorHidden',[32 32]);
    Options = ensureField(Options,'criticHidden',[32 32]);
    Options.zDim = max(1,round(double(Options.zDim)));
    Options.iter = max(0,round(double(Options.iter)));
    Options.miniBatch = max(1,round(double(Options.miniBatch)));
    Options.lrD = double(Options.lrD);
    Options.lrG = double(Options.lrG);
    Options.gpLambda = max(0,double(Options.gpLambda));
    Options.nCritic = max(1,round(double(Options.nCritic)));
    Options.trainZMode = lower(strtrim(string(Options.trainZMode)));
    Options.sampleZMode = lower(strtrim(string(Options.sampleZMode)));
    Options.trainSigma = optionalFiniteSigma(Options.trainSigma);
    Options.sampleSigma = optionalFiniteSigma(Options.sampleSigma);
    Options.prescreenMultiplier = max(1,round(double( ...
        Options.prescreenMultiplier)));
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

function sigma = latentSigma(Options,purpose)
    sigma = [];
    if purpose == "train" && isfield(Options,'trainSigma') && ...
            ~isempty(Options.trainSigma)
        sigma = Options.trainSigma;
    elseif purpose == "sample" && isfield(Options,'sampleSigma') && ...
            ~isempty(Options.sampleSigma)
        sigma = Options.sampleSigma;
    end
    if isempty(sigma)
        sigma = Options.sigma;
    end
    sigma = finiteSigma(sigma);
end

function sigma = optionalFiniteSigma(sigma)
    if isempty(sigma)
        return;
    end
    sigma = double(sigma);
    if isempty(sigma) || ~isfinite(sigma(1))
        sigma = [];
    else
        sigma = sigma(1);
    end
end

function value = scalarExtract(x)
    value = double(gather(extractdata(x)));
    if ~isscalar(value)
        value = mean(value(:));
    end
end
