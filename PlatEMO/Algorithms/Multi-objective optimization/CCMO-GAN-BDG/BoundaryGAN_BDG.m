function varargout = BoundaryGAN_BDG(action,varargin)
% BoundaryGAN_BDG - Direct complete-decision boundary GAN dispatcher.
%
% Current experiment-2 contract:
%   target-conditioned boundary:   G(z,y_t,d_t) -> x, D(x,y_t,d_t)
%   target-unconditioned boundary: G(z) -> x, D(x)
%
% Only the target-conditioned discriminator sees three batch types:
%   matched real, mismatched real, and fake matched.

    switch lower(action)
        case 'traindirectboundary'
            varargout{1} = TrainDirectBoundaryGAN_BDG(varargin{:});
        case 'sample'
            [varargout{1:nargout}] = SampleBoundaryGAN_BDG(varargin{:});
        case 'probe'
            varargout{1} = ProbeBoundaryGAN_BDG(varargin{:});
        case 'diagnosesample'
            [varargout{1:nargout}] = DiagnosticSampleBoundaryGAN_BDG(varargin{:});
        case 'traindiagnose'
            varargout{1} = TrainDiagnostic_BDG(varargin{:});
        otherwise
            error('BoundaryGAN_BDG:UnknownAction','Unknown action "%s".',action);
    end
end

function [Offspring,Diag] = DiagnosticSampleBoundaryGAN_BDG( ...
        Problem,GAN,conditionIndex,ZRows)
    Diag = EmptySampleDiag_BDG(Problem.M,Problem.D);
    if nargin < 4
        ZRows = [];
    end
    if isempty(GAN)
        Offspring = [];
        return;
    end
    n = DiagnosticSampleCount_BDG(GAN,conditionIndex,ZRows);
    if n <= 0
        Offspring = [];
        return;
    end
    [X,Info] = GenerateRawDecisionsByConditionIndex_BDG( ...
        GAN,conditionIndex,ZRows,n);
    S = Problem.Evaluation(X);
    feasible = IsFeasibleSet_BDG(S);
    Diag = SetSolutionLayer_BDG(Diag,'rawgen',S,feasible);
    Diag = SetSolutionLayer_BDG(Diag,'injected',S,feasible);
    Diag.rawgen_condition_index = Info.conditionIndex(:);
    Diag.injected_condition_index = Info.conditionIndex(:);
    Diag.rawgen_var_clip_rate = Info.varClipRate;
    Diag.rawgen_var_clip_count = Info.varClipCount;
    Diag.rawgen_var_value_count = Info.varValueCount;
    Offspring = S;
end

function n = DiagnosticSampleCount_BDG(GAN,conditionIndex,ZRows)
    n = 0;
    if ~isempty(conditionIndex)
        n = numel(conditionIndex);
    end
    if ~isempty(ZRows)
        zRows = double(ZRows);
        if isvector(zRows) && numel(zRows) == GAN.zDim
            zCount = 1;
        else
            zCount = size(zRows,1);
        end
        if n == 0
            n = zCount;
        else
            assert(zCount == n, ...
                'BoundaryGAN_BDG:BadDiagnosticZRows', ...
                'Diagnostic Z rows must match requested condition count.');
        end
    end
end

function GAN = TrainDirectBoundaryGAN_BDG(XTrain,XAI,lower,upper,zDim,maxIter,GAN,Options)
    if nargin < 7
        GAN = [];
    end
    if nargin < 8 || isempty(Options)
        Options = struct();
    end

    D = numel(lower);
    zDim = max(1,round(double(zDim)));
    Options = NormalizeDirectGANOptions_BDG(Options);
    XTrain = PrepareTrainingDecs_BDG(XTrain,D);
    XAI = PrepareTrainingDecs_BDG(XAI,D);
    assert(~isempty(XTrain), ...
        'BoundaryGAN_BDG:EmptyTrainingSet', ...
        'Direct boundary GAN requires non-empty AF decisions.');

    XPositive = single(ScaleToTanh_BDG(XTrain,lower,upper));
    XAI = single(ScaleToTanh_BDG(XAI,lower,upper));
    CTrain = PrepareConditionData_BDG(Options.conditionData, ...
        size(XPositive,1),Options.generatorMode);
    condDim = size(CTrain,2);
    ValidateModeCombo_BDG(Options,condDim,size(XPositive,1));
    generatorCondDim = GeneratorConditionDim_BDG(Options.generatorMode,condDim);
    sampleWeights = NormalizeSampleWeights_BDG(Options.sampleWeights, ...
        size(XPositive,1));
    realLabels = NormalizeRealLabels_BDG(Options.realLabels, ...
        size(XPositive,1),Options.realLabel);
    Options.realLabels = realLabels;

    if isempty(GAN) || ReinitializeDirectBoundaryGAN_BDG( ...
            GAN,zDim,D,Options.archMode,Options.generatorMode, ...
            condDim,generatorCondDim,Options.criticMode, ...
            Options.generatorLossMode)
        netG = CreateGenerator_BDG(zDim,D,Options.archMode,generatorCondDim);
        netD = CreateDirectBoundaryDiscriminator_BDG(D,Options.archMode,condDim);
    else
        netG = GAN.netG;
        netD = GAN.netD;
    end

    miniBatch = min(Options.miniBatch,max(1,size(XPositive,1)));
    trailingAvgG   = [];
    trailingAvgSqG = [];
    trailingAvgD   = [];
    trailingAvgSqD = [];
    lrG = Options.lrG;
    lrD = Options.lrD;
    beta1 = 0.5;
    beta2 = 0.999;
    dUpdate = 0;
    gUpdate = 0;
    targetDiag = EmptyTargetConditionedTrainDiag_BDG();

    for iter = 1 : Options.dPretrainIter
        dUpdate = dUpdate + 1;
        [netD,trailingAvgD,trailingAvgSqD,targetDiag] = ...
            UpdateDiscriminator_BDG(netG,netD,XPositive,CTrain, ...
            sampleWeights,zDim,miniBatch,Options,targetDiag, ...
            trailingAvgD,trailingAvgSqD,dUpdate,lrD,beta1,beta2);
    end
    pretrainDiag = PrefixTrainDiagnostic_BDG( ...
        EvaluateTrainDiagnostic_BDG(netG,netD,XPositive,XAI, ...
        lower,upper,zDim,100,CTrain,generatorCondDim), ...
        'gan_pretrain_d_');

    trainCount = max(0,round(double(maxIter)));
    if Options.trainMode == "epoch" && isempty(sampleWeights)
        for epoch = 1 : trainCount
            order = randperm(size(XPositive,1));
            for startIdx = 1 : miniBatch : numel(order)
                batchIdx = order(startIdx:min(startIdx+miniBatch-1, ...
                    numel(order)));
                for dStep = 1 : Options.dSteps
                    dUpdate = dUpdate + 1;
                    [netD,trailingAvgD,trailingAvgSqD,targetDiag] = ...
                        UpdateDiscriminator_BDG(netG,netD,XPositive, ...
                        CTrain,sampleWeights,zDim,miniBatch,Options, ...
                        targetDiag,trailingAvgD,trailingAvgSqD,dUpdate, ...
                        lrD,beta1,beta2,batchIdx);
                end
                for gStep = 1 : Options.gSteps
                    gUpdate = gUpdate + 1;
                    [netG,trailingAvgG,trailingAvgSqG,targetDiag] = ...
                        UpdateGenerator_BDG(netG,netD,XPositive,CTrain, ...
                        sampleWeights,zDim,miniBatch,Options,targetDiag, ...
                        trailingAvgG,trailingAvgSqG,gUpdate,lrG,beta1, ...
                        beta2,generatorCondDim,batchIdx);
                end
            end
        end
    else
        for iter = 1 : trainCount
            for dStep = 1 : Options.dSteps
                dUpdate = dUpdate + 1;
                [netD,trailingAvgD,trailingAvgSqD,targetDiag] = ...
                    UpdateDiscriminator_BDG(netG,netD,XPositive,CTrain, ...
                    sampleWeights,zDim,miniBatch,Options,targetDiag, ...
                    trailingAvgD,trailingAvgSqD,dUpdate,lrD,beta1,beta2);
            end
            for gStep = 1 : Options.gSteps
                gUpdate = gUpdate + 1;
                [netG,trailingAvgG,trailingAvgSqG,targetDiag] = ...
                    UpdateGenerator_BDG(netG,netD,XPositive,CTrain, ...
                    sampleWeights,zDim,miniBatch,Options,targetDiag, ...
                    trailingAvgG,trailingAvgSqG,gUpdate,lrG,beta1,beta2, ...
                    generatorCondDim);
            end
        end
    end

    GAN = struct( ...
        'netG',netG, ...
        'netD',netD, ...
        'lower',double(lower), ...
        'upper',double(upper), ...
        'zDim',double(zDim), ...
        'D',double(D), ...
        'trainPosCount',double(size(XPositive,1)), ...
        'trainAFCount',double(size(XPositive,1)), ...
        'trainAICount',double(size(XAI,1)), ...
        'trainAIRealCount',0, ...
        'trainAIPositiveCount',0, ...
        'trainAINegativeCount',0, ...
        'trainNegCount',0, ...
        'trainWeightCount',double(numel(sampleWeights)), ...
        'trainRealLabelCount',double(numel(realLabels)), ...
        'trainRealLabelMean',MeanOrNaN_BDG(realLabels), ...
        'trainRealLabelMedian',MedianOrNaN_BDG(realLabels), ...
        'trainRealLabelP10',PercentileOrNaN_BDG(realLabels,10), ...
        'trainRealLabelP90',PercentileOrNaN_BDG(realLabels,90), ...
        'generatorMode',char(Options.generatorMode), ...
        'generatorLossMode',char(Options.generatorLossMode), ...
        'criticMode',char(Options.criticMode), ...
        'conditionDim',double(condDim), ...
        'generatorConditionDim',double(generatorCondDim), ...
        'conditionCount',double(size(CTrain,1)), ...
        'conditionData',single(CTrain), ...
        'conditionSamplingWeights',double(sampleWeights(:)), ...
        'archMode',char(Options.archMode), ...
        'miniBatch',double(miniBatch), ...
        'realLabel',double(Options.realLabel), ...
        'lrD',double(Options.lrD), ...
        'lrG',double(Options.lrG), ...
        'trainMode',char(Options.trainMode), ...
        'dPretrainIter',double(Options.dPretrainIter), ...
        'dSteps',double(Options.dSteps), ...
        'gSteps',double(Options.gSteps), ...
        'pretrainDiag',pretrainDiag, ...
        'targetConditionedDiag',FinalizeTargetConditionedDiag_BDG( ...
        targetDiag));
end

function [netD,trailingAvgD,trailingAvgSqD,targetDiag] = ...
        UpdateDiscriminator_BDG(netG,netD,XPositive,CTrain,sampleWeights, ...
        zDim,miniBatch,Options,targetDiag,trailingAvgD,trailingAvgSqD, ...
        dUpdate,lrD,beta1,beta2,batchIdx)
    if nargin < 16 || isempty(batchIdx)
        [XBatch,batchIdx] = DrawTrainBatch_BDG(XPositive,miniBatch, ...
            sampleWeights);
    else
        XBatch = DrawTrainBatchByIndex_BDG(XPositive,batchIdx);
        miniBatch = numel(batchIdx);
    end
    CBatch = DrawConditionBatch_BDG(CTrain,batchIdx);
    realLabelBatch = DrawRealLabelBatch_BDG(Options.realLabels,batchIdx, ...
        Options.realLabel);
    Z = dlarray(randn(zDim,miniBatch,'single'),"CB");
    if UsesTargetConditionedCritic_BDG(Options.criticMode)
        [CMismatch,MismatchDiag] = DrawMismatchedConditionBatch_BDG( ...
            CTrain,batchIdx);
        [~,gradD] = dlfeval(@TargetConditionedDiscriminatorGradients_BDG, ...
            netG,netD,XBatch,CBatch,CMismatch,Z, ...
            GeneratorConditionDim_BDG(Options.generatorMode,size(CTrain,2)), ...
            realLabelBatch);
        targetDiag = AccumulateTargetConditionedDiag_BDG( ...
            targetDiag,MismatchDiag);
    else
        CFake = DrawRandomConditionBatch_BDG(CTrain,miniBatch);
        [~,gradD] = dlfeval(@DirectBoundaryDiscriminatorGradients_BDG, ...
            netG,netD,XBatch,CBatch,Z,CFake, ...
            GeneratorConditionDim_BDG(Options.generatorMode,size(CTrain,2)), ...
            realLabelBatch);
    end
    [netD,trailingAvgD,trailingAvgSqD] = adamupdate( ...
        netD,gradD,trailingAvgD,trailingAvgSqD,dUpdate, ...
        lrD,beta1,beta2);
end

function [netG,trailingAvgG,trailingAvgSqG,targetDiag] = ...
        UpdateGenerator_BDG(netG,netD,XPositive,CTrain,sampleWeights, ...
        zDim,miniBatch,Options,targetDiag,trailingAvgG,trailingAvgSqG, ...
        gUpdate,lrG,beta1,beta2,generatorCondDim,batchIdx)
    if nargin < 17
        batchIdx = [];
    end
    if UsesTargetConditionedCritic_BDG(Options.criticMode)
        if isempty(batchIdx)
            [~,batchIdx] = DrawTrainBatch_BDG(XPositive, ...
                miniBatch,sampleWeights);
        else
            batchIdx = max(1,min(size(XPositive,1),round(double(batchIdx(:)'))));
            miniBatch = numel(batchIdx);
        end
        CFake = DrawConditionBatch_BDG(CTrain,batchIdx);
    else
        if ~isempty(batchIdx)
            miniBatch = numel(batchIdx);
        end
        CFake = DrawRandomConditionBatch_BDG(CTrain,miniBatch);
    end
    Z = dlarray(randn(zDim,miniBatch,'single'),"CB");
    [~,gradG] = dlfeval( ...
        @DirectBoundaryGeneratorGradients_BDG,netG,netD,Z, ...
        CFake,generatorCondDim);
    [netG,trailingAvgG,trailingAvgSqG] = adamupdate( ...
        netG,gradG,trailingAvgG,trailingAvgSqG,gUpdate, ...
        lrG,beta1,beta2);
end

function Options = NormalizeDirectGANOptions_BDG(Options)
    if ~isstruct(Options)
        error('BoundaryGAN_BDG:BadTrainingOptions', ...
            'Direct GAN training options must be a struct.');
    end
    Options = WithDefault_BDG(Options,'dPretrainIter',0);
    Options = WithDefault_BDG(Options,'dSteps',1);
    Options = WithDefault_BDG(Options,'gSteps',1);
    Options = WithDefault_BDG(Options,'sampleWeights',[]);
    Options = WithDefault_BDG(Options,'realLabels',[]);
    Options = WithDefault_BDG(Options,'generatorMode',"objective_target_unconditioned");
    Options = WithDefault_BDG(Options,'generatorLossMode',"adversarial");
    Options = WithDefault_BDG(Options,'conditionData',[]);
    Options = WithDefault_BDG(Options,'criticMode',"af_like");
    Options = WithDefault_BDG(Options,'miniBatch',16);
    Options = WithDefault_BDG(Options,'archMode',"small");
    Options = WithDefault_BDG(Options,'realLabel',1.0);
    Options = WithDefault_BDG(Options,'lrD',1e-4);
    Options = WithDefault_BDG(Options,'lrG',2e-4);
    Options = WithDefault_BDG(Options,'trainMode',"iter");
    Options.dPretrainIter = max(0,round(double(Options.dPretrainIter)));
    Options.dSteps = max(1,round(double(Options.dSteps)));
    Options.gSteps = max(1,round(double(Options.gSteps)));
    Options.miniBatch = max(1,round(double(Options.miniBatch)));
    Options.realLabel = double(Options.realLabel);
    assert(isscalar(Options.realLabel) && isfinite(Options.realLabel) && ...
            Options.realLabel > 0 && Options.realLabel <= 1, ...
        'BoundaryGAN_BDG:BadRealLabel', ...
        'realLabel must be a finite scalar in (0,1].');
    Options.lrD = NormalizeLearningRate_BDG(Options.lrD,'lrD');
    Options.lrG = NormalizeLearningRate_BDG(Options.lrG,'lrG');
    Options.trainMode = NormalizeTrainMode_BDG(Options.trainMode);
    Options.archMode = NormalizeArchMode_BDG(Options.archMode);
    Options.generatorMode = NormalizeGeneratorMode_BDG(Options.generatorMode);
    Options.generatorLossMode = NormalizeGeneratorLossMode_BDG( ...
        Options.generatorLossMode);
    Options.criticMode = NormalizeCriticMode_BDG(Options.criticMode);
end

function ValidateModeCombo_BDG(Options,condDim,nTrain)
    if Options.criticMode == "target_conditioned"
        assert(Options.generatorMode == "objective_target_conditioned", ...
            'BoundaryGAN_BDG:BadTargetConditionedCombo', ...
            'target_conditioned critic requires objective_target_conditioned generator.');
        assert(Options.generatorLossMode == "conditional_adversarial", ...
            'BoundaryGAN_BDG:BadTargetConditionedLoss', ...
            'target_conditioned critic requires conditional_adversarial loss.');
        assert(condDim > 0 && nTrain >= 2, ...
            'BoundaryGAN_BDG:InsufficientTargetConditionData', ...
            'Target-conditioned GAN requires at least two condition rows.');
    else
        assert(Options.generatorMode == "objective_target_unconditioned", ...
            'BoundaryGAN_BDG:BadBaselineCombo', ...
            'af_like critic supports objective_target_unconditioned generator mode.');
        assert(Options.generatorLossMode == "adversarial", ...
            'BoundaryGAN_BDG:BadBaselineLoss', ...
            'af_like critic only supports adversarial generator loss.');
    end
end

function mode = NormalizeArchMode_BDG(mode)
    mode = lower(strtrim(string(mode)));
    assert(ismember(mode,["small","large","g2sl"]), ...
        'BoundaryGAN_BDG:BadArchMode', ...
        'archMode must be small, large, or g2sl.');
end

function mode = NormalizeTrainMode_BDG(mode)
    mode = lower(strtrim(string(mode)));
    assert(ismember(mode,["iter","epoch"]), ...
        'BoundaryGAN_BDG:BadTrainMode', ...
        'trainMode must be iter or epoch.');
end

function lr = NormalizeLearningRate_BDG(lr,name)
    lr = double(lr);
    assert(isscalar(lr) && isfinite(lr) && lr > 0, ...
        'BoundaryGAN_BDG:BadLearningRate', ...
        '%s must be a positive finite scalar.',name);
end

function mode = NormalizeGeneratorMode_BDG(mode)
    mode = lower(strtrim(string(mode)));
    valid = ["objective_target_conditioned","objective_target_unconditioned"];
    assert(ismember(mode,valid), ...
        'BoundaryGAN_BDG:BadGeneratorMode', ...
        'generatorMode must be one of: %s.',strjoin(valid,", "));
end

function mode = NormalizeGeneratorLossMode_BDG(mode)
    mode = lower(strtrim(string(mode)));
    valid = ["adversarial","conditional_adversarial"];
    assert(ismember(mode,valid), ...
        'BoundaryGAN_BDG:BadGeneratorLossMode', ...
        'generatorLossMode must be one of: %s.',strjoin(valid,", "));
end

function mode = NormalizeCriticMode_BDG(mode)
    mode = lower(strtrim(string(mode)));
    valid = ["af_like","target_conditioned"];
    assert(ismember(mode,valid), ...
        'BoundaryGAN_BDG:BadCriticMode', ...
        'criticMode must be one of: %s.',strjoin(valid,", "));
end

function flag = UsesTargetConditionedCritic_BDG(mode)
    flag = NormalizeCriticMode_BDG(mode) == "target_conditioned";
end

function dim = GeneratorConditionDim_BDG(generatorMode,criticCondDim)
    generatorMode = NormalizeGeneratorMode_BDG(generatorMode);
    if generatorMode == "objective_target_unconditioned"
        dim = 0;
    else
        dim = criticCondDim;
    end
end

function S = WithDefault_BDG(S,fieldName,value)
    if ~isfield(S,fieldName) || isempty(S.(fieldName))
        S.(fieldName) = value;
    end
end

function X = PrepareTrainingDecs_BDG(X,D)
    if isempty(X)
        X = zeros(0,D);
        return;
    end
    X = double(X);
    assert(size(X,2) == D, ...
        'BoundaryGAN_BDG:BadTrainingDimension', ...
        'Training decisions must have %d columns.',D);
end

function C = PrepareConditionData_BDG(C,n,mode)
    mode = NormalizeGeneratorMode_BDG(mode);
    if mode == "objective_target_unconditioned"
        C = zeros(n,0,'single');
        return;
    end
    assert(~isempty(C), ...
        'BoundaryGAN_BDG:MissingConditionData', ...
        'Conditional generator modes require conditionData.');
    C = single(C);
    assert(size(C,1) == n, ...
        'BoundaryGAN_BDG:BadConditionRows', ...
        'conditionData must contain one row per AF training decision.');
    assert(all(isfinite(C),'all'), ...
        'BoundaryGAN_BDG:BadConditionData', ...
        'conditionData must be finite.');
end

function flag = ReinitializeDirectBoundaryGAN_BDG(GAN,zDim,D,archMode, ...
        generatorMode,condDim,generatorCondDim,criticMode,generatorLossMode)
    flag = ~isfield(GAN,'zDim') || double(GAN.zDim) ~= double(zDim) || ...
        ~isfield(GAN,'D') || double(GAN.D) ~= double(D) || ...
        ~isfield(GAN,'netG') || ~isfield(GAN,'netD') || ...
        ~isfield(GAN,'archMode') || string(GAN.archMode) ~= string(archMode) || ...
        ~isfield(GAN,'generatorMode') || ...
        string(GAN.generatorMode) ~= string(generatorMode) || ...
        ~isfield(GAN,'criticMode') || ...
        string(GAN.criticMode) ~= string(criticMode) || ...
        ~isfield(GAN,'generatorLossMode') || ...
        string(GAN.generatorLossMode) ~= string(generatorLossMode) || ...
        ~isfield(GAN,'conditionDim') || ...
        double(GAN.conditionDim) ~= double(condDim) || ...
        ~isfield(GAN,'generatorConditionDim') || ...
        double(GAN.generatorConditionDim) ~= double(generatorCondDim);
end

function weights = NormalizeSampleWeights_BDG(weights,n)
    if isempty(weights)
        weights = [];
        return;
    end
    weights = double(weights(:));
    assert(numel(weights) == n, ...
        'BoundaryGAN_BDG:BadSampleWeights', ...
        'sampleWeights must contain one value per training decision.');
    weights(~isfinite(weights) | weights < 0) = 0;
    if sum(weights) <= 0
        weights = [];
    else
        weights = weights ./ sum(weights);
    end
end

function labels = NormalizeRealLabels_BDG(labels,n,defaultLabel)
    if isempty(labels)
        labels = [];
        return;
    end
    labels = double(labels(:));
    assert(numel(labels) == n, ...
        'BoundaryGAN_BDG:BadRealLabels', ...
        'realLabels must contain one value per training decision.');
    labels(~isfinite(labels)) = double(defaultLabel);
    labels = min(max(labels,0),1);
end

function [XBatch,idx] = DrawTrainBatch_BDG(X,miniBatch,weights)
    if nargin < 3
        weights = [];
    end
    if isempty(weights)
        idx = randi(size(X,1),1,miniBatch);
    else
        idx = WeightedSampleIndices_BDG(weights,miniBatch);
    end
    XBatch = dlarray(X(idx,:)',"CB");
end

function XBatch = DrawTrainBatchByIndex_BDG(X,idx)
    idx = max(1,min(size(X,1),round(double(idx(:)'))));
    XBatch = dlarray(X(idx,:)',"CB");
end

function CBatch = DrawConditionBatch_BDG(C,idx)
    if isempty(C) || size(C,2) == 0
        CBatch = [];
        return;
    end
    CBatch = dlarray(single(C(idx,:)'),"CB");
end

function labels = DrawRealLabelBatch_BDG(realLabels,idx,defaultLabel)
    if isempty(realLabels)
        labels = single(defaultLabel);
        return;
    end
    idx = max(1,min(numel(realLabels),round(double(idx(:)'))));
    labels = dlarray(single(realLabels(idx)'),'CB');
end

function [CBatch,idx] = DrawRandomConditionBatch_BDG(C,miniBatch)
    if isempty(C) || size(C,2) == 0
        CBatch = [];
        idx = ones(1,miniBatch);
        return;
    end
    idx = randi(size(C,1),1,miniBatch);
    CBatch = DrawConditionBatch_BDG(C,idx);
end

function [CBatch,Diag] = DrawMismatchedConditionBatch_BDG(C,matchIdx)
    Diag = struct( ...
        'target_mismatch_count',0, ...
        'target_mismatch_identity_count',0);
    if isempty(C) || size(C,2) == 0
        CBatch = [];
        return;
    end
    rowN = size(C,1);
    matchIdx = max(1,min(rowN,round(double(matchIdx(:)'))));
    if rowN < 2
        CBatch = DrawConditionBatch_BDG(C,matchIdx);
        Diag.target_mismatch_identity_count = numel(matchIdx);
        return;
    end
    mismatchIdx = mod(matchIdx,rowN) + 1;
    Diag.target_mismatch_count = numel(mismatchIdx);
    Diag.target_mismatch_identity_count = sum(mismatchIdx == matchIdx);
    CBatch = DrawConditionBatch_BDG(C,mismatchIdx);
end

function idx = WeightedSampleIndices_BDG(weights,n)
    edges = cumsum(weights(:));
    edges(end) = 1;
    idx = zeros(1,n);
    u = rand(1,n);
    for i = 1 : n
        idx(i) = find(edges >= u(i),1,'first');
    end
end

function [Offspring,Diag] = SampleBoundaryGAN_BDG(Problem,GAN,nGen)
    Diag = EmptySampleDiag_BDG(Problem.M,Problem.D);
    if isempty(GAN) || nGen <= 0
        Offspring = [];
        return;
    end
    [X,Info] = GenerateRawDecisions_BDG(GAN,nGen);
    S = Problem.Evaluation(X);
    feasible = IsFeasibleSet_BDG(S);
    Diag = SetSolutionLayer_BDG(Diag,'rawgen',S,feasible);
    Diag = SetSolutionLayer_BDG(Diag,'injected',S,feasible);
    Diag.rawgen_condition_index = Info.conditionIndex(:);
    Diag.injected_condition_index = Info.conditionIndex(:);
    Diag.rawgen_var_clip_rate = Info.varClipRate;
    Diag.rawgen_var_clip_count = Info.varClipCount;
    Diag.rawgen_var_value_count = Info.varValueCount;
    Offspring = S;
end

function Diag = ProbeBoundaryGAN_BDG(Problem,GAN,nProbe)
    Diag = struct( ...
        'probe_raw_count',0, ...
        'probe_raw_decs',zeros(0,Problem.D), ...
        'probe_raw_objs',zeros(0,Problem.M), ...
        'probe_raw_cons',zeros(0,0), ...
        'probe_raw_feasible',false(0,1), ...
        'probe_raw_condition_index',zeros(0,1), ...
        'probe_raw_FE',0);
    if isempty(GAN) || nProbe <= 0
        return;
    end
    [X,Info] = GenerateRawDecisions_BDG(GAN,nProbe);
    S = Problem.Evaluation(X);
    Diag.probe_raw_count = numel(S);
    Diag.probe_raw_decs = S.decs;
    Diag.probe_raw_objs = S.objs;
    Diag.probe_raw_cons = S.cons;
    Diag.probe_raw_feasible = IsFeasibleSet_BDG(S);
    Diag.probe_raw_condition_index = Info.conditionIndex(:);
    Diag.probe_raw_var_clip_rate = Info.varClipRate;
    Diag.probe_raw_var_clip_count = Info.varClipCount;
    Diag.probe_raw_var_value_count = Info.varValueCount;
    Diag.probe_raw_FE = numel(S);
end

function Diag = TrainDiagnostic_BDG(GAN,XTrain,XAI,lower,upper,nDiag)
    Diag = EmptyTrainDiagnostic_BDG();
    if nargin < 6 || isempty(nDiag)
        nDiag = 128;
    end
    if isempty(GAN) || ~isfield(GAN,'netG') || ~isfield(GAN,'netD') || ...
            isempty(XTrain)
        return;
    end
    D = numel(lower);
    XTrain = single(ScaleToTanh_BDG(PrepareTrainingDecs_BDG(XTrain,D), ...
        lower,upper));
    XAI = single(ScaleToTanh_BDG(PrepareTrainingDecs_BDG(XAI,D), ...
        lower,upper));
    CTrain = StoredConditionData_BDG(GAN,size(XTrain,1));
    Diag = EvaluateTrainDiagnostic_BDG(GAN.netG,GAN.netD,XTrain,XAI, ...
        lower,upper,GAN.zDim,nDiag,CTrain, ...
        StoredGeneratorConditionDim_BDG(GAN));
end

function Diag = EvaluateTrainDiagnostic_BDG( ...
        netG,netD,XTrain,XAI,lower,upper,zDim,nDiag,CTrain, ...
        generatorCondDim)
    Diag = EmptyTrainDiagnostic_BDG();
    if isempty(XTrain)
        return;
    end
    if nargin < 9
        CTrain = zeros(size(XTrain,1),0,'single');
    end
    if nargin < 10 || isempty(generatorCondDim)
        generatorCondDim = size(CTrain,2);
    end
    nDiag = min(max(1,round(double(nDiag))),128);
    try
        [XReal,batchIdx] = DrawTrainBatch_BDG(XTrain,nDiag);
        CReal = DrawConditionBatch_BDG(CTrain,batchIdx);
        CFake = DrawRandomConditionBatch_BDG(CTrain,nDiag);
        XAIBatch = DrawOptionalTrainBatch_BDG(XAI,nDiag);
        Z = dlarray(randn(zDim,nDiag,'single'),"CB");
        XFake = predict(netG,AppendGeneratorConditionToFeatures_BDG( ...
            Z,CFake,generatorCondDim));
        XUniform = DrawUniformScaledBatch_BDG(lower,upper,nDiag);
        YReal = predict(netD,AppendConditionToFeatures_BDG(XReal,CReal));
        PMismatch = [];
        if ~isempty(CTrain) && size(CTrain,2) > 0 && size(CTrain,1) >= 2
            CMismatch = DrawMismatchedConditionBatch_BDG(CTrain,batchIdx);
            YMismatch = predict(netD,AppendConditionToFeatures_BDG( ...
                XReal,CMismatch));
            PMismatch = FirstProb_BDG(YMismatch);
        end
        if isempty(XAIBatch)
            YAI = [];
        else
            CAI = DrawRandomConditionBatch_BDG(CTrain,nDiag);
            YAI = predict(netD,AppendConditionToFeatures_BDG(XAIBatch,CAI));
        end
        YFake = predict(netD,AppendConditionToFeatures_BDG(XFake,CFake));
        CUniform = DrawRandomConditionBatch_BDG(CTrain,nDiag);
        YUniform = predict(netD,AppendConditionToFeatures_BDG(XUniform,CUniform));
        Diag = SummarizeTrainDiagnostic_BDG(FirstProb_BDG(YReal), ...
            FirstProb_BDG(YAI),FirstProb_BDG(YFake), ...
            FirstProb_BDG(YUniform),PMismatch);
    catch
        Diag = EmptyTrainDiagnostic_BDG();
        Diag.gan_d_diag_status = -1;
    end
    Diag.gan_d_diag_count = double(nDiag);
end

function XBatch = DrawOptionalTrainBatch_BDG(X,miniBatch)
    if isempty(X)
        XBatch = [];
    else
        XBatch = DrawTrainBatch_BDG(X,miniBatch);
    end
end

function Diag = EmptySampleDiag_BDG(M,D)
    Diag = struct( ...
        'rawgen_count',0, ...
        'rawgen_decs',zeros(0,D), ...
        'rawgen_objs',zeros(0,M), ...
        'rawgen_cons',zeros(0,0), ...
        'rawgen_feasible',false(0,1), ...
        'rawgen_condition_index',zeros(0,1), ...
        'rawgen_var_clip_rate',NaN, ...
        'rawgen_var_clip_count',0, ...
        'rawgen_var_value_count',0, ...
        'injected_count',0, ...
        'injected_decs',zeros(0,D), ...
        'injected_objs',zeros(0,M), ...
        'injected_cons',zeros(0,0), ...
        'injected_feasible',false(0,1), ...
        'injected_condition_index',zeros(0,1));
end

function Diag = SetSolutionLayer_BDG(Diag,prefix,S,feasible)
    Diag.([prefix,'_count']) = numel(S);
    Diag.([prefix,'_decs']) = S.decs;
    Diag.([prefix,'_objs']) = S.objs;
    Diag.([prefix,'_cons']) = S.cons;
    Diag.([prefix,'_feasible']) = feasible(:);
end

function [X,Info] = GenerateRawDecisions_BDG(GAN,n)
    Info = struct('varClipRate',NaN,'varClipCount',0, ...
        'varValueCount',0,'conditionIndex',zeros(0,1));
    n = max(0,round(double(n)));
    D = double(GAN.D);
    if n == 0
        X = zeros(0,D);
        return;
    end
    Z = dlarray(randn(GAN.zDim,n,'single'),"CB");
    [C,conditionIndex] = SampleStoredCondition_BDG(GAN,n);
    Info.conditionIndex = conditionIndex(:);
    X = predict(GAN.netG,AppendGeneratorConditionToFeatures_BDG( ...
        Z,C,StoredGeneratorConditionDim_BDG(GAN)));
    X = double(extractdata(X)');
    XUnclipped = (X + 1)/2 .* (GAN.upper - GAN.lower) + GAN.lower;
    clipMask = XUnclipped < GAN.lower | XUnclipped > GAN.upper;
    X = min(max(XUnclipped,GAN.lower),GAN.upper);
    Info.varClipCount = sum(clipMask(:));
    Info.varValueCount = numel(clipMask);
    if Info.varValueCount > 0
        Info.varClipRate = Info.varClipCount ./ Info.varValueCount;
    end
end

function [X,Info] = GenerateRawDecisionsByConditionIndex_BDG( ...
        GAN,conditionIndex,ZRows,n)
    Info = struct('varClipRate',NaN,'varClipCount',0, ...
        'varValueCount',0,'conditionIndex',zeros(0,1));
    n = max(0,round(double(n)));
    D = double(GAN.D);
    if n == 0
        X = zeros(0,D);
        return;
    end
    [CBatch,idx] = StoredConditionByIndex_BDG(GAN,conditionIndex,n);
    Z = DiagnosticZBatch_BDG(GAN,ZRows,n);
    Info.conditionIndex = idx(:);
    X = predict(GAN.netG,AppendGeneratorConditionToFeatures_BDG( ...
        Z,CBatch,StoredGeneratorConditionDim_BDG(GAN)));
    X = double(extractdata(X)');
    XUnclipped = (X + 1)/2 .* (GAN.upper - GAN.lower) + GAN.lower;
    clipMask = XUnclipped < GAN.lower | XUnclipped > GAN.upper;
    X = min(max(XUnclipped,GAN.lower),GAN.upper);
    Info.varClipCount = sum(clipMask(:));
    Info.varValueCount = numel(clipMask);
    if Info.varValueCount > 0
        Info.varClipRate = Info.varClipCount ./ Info.varValueCount;
    end
end

function Z = DiagnosticZBatch_BDG(GAN,ZRows,n)
    if isempty(ZRows)
        Z = dlarray(randn(GAN.zDim,n,'single'),"CB");
        return;
    end
    ZRows = double(ZRows);
    if isvector(ZRows) && numel(ZRows) == GAN.zDim
        ZRows = reshape(ZRows,1,[]);
    end
    assert(size(ZRows,1) == n && size(ZRows,2) == GAN.zDim, ...
        'BoundaryGAN_BDG:BadDiagnosticZRows', ...
        'Diagnostic Z rows must be n-by-zDim.');
    Z = dlarray(single(ZRows'),"CB");
end

function C = StoredConditionData_BDG(GAN,nTrain)
    condDim = max(0,round(double(GAN.conditionDim)));
    C = zeros(nTrain,condDim,'single');
    if condDim == 0 || ~isfield(GAN,'conditionData') || isempty(GAN.conditionData)
        return;
    end
    Data = single(GAN.conditionData);
    if size(Data,2) ~= condDim
        return;
    end
    n = min(nTrain,size(Data,1));
    C(1:n,:) = Data(1:n,:);
end

function [CBatch,idx] = StoredConditionByIndex_BDG(GAN,conditionIndex,n)
    condDim = max(0,round(double(GAN.conditionDim)));
    if condDim == 0
        CBatch = [];
        idx = zeros(n,1);
        return;
    end
    if ~isfield(GAN,'conditionData') || isempty(GAN.conditionData)
        CBatch = dlarray(zeros(condDim,n,'single'),"CB");
        idx = zeros(n,1);
        return;
    end
    Data = single(GAN.conditionData);
    if size(Data,2) ~= condDim
        CBatch = dlarray(zeros(condDim,n,'single'),"CB");
        idx = zeros(n,1);
        return;
    end
    if isempty(conditionIndex)
        weights = StoredConditionSamplingWeights_BDG(GAN,size(Data,1));
        if isempty(weights)
            idx = randi(size(Data,1),1,n);
        else
            idx = WeightedSampleIndices_BDG(weights,n);
        end
    else
        idx = max(1,min(size(Data,1),round(double(conditionIndex(:)'))));
        assert(numel(idx) == n, ...
            'BoundaryGAN_BDG:BadDiagnosticConditionIndex', ...
            'Diagnostic condition index count must match requested samples.');
    end
    CBatch = dlarray(Data(idx,:)',"CB");
end

function [CBatch,idx] = SampleStoredCondition_BDG(GAN,n)
    condDim = max(0,round(double(GAN.conditionDim)));
    if condDim == 0
        CBatch = [];
        idx = zeros(n,1);
        return;
    end
    if ~isfield(GAN,'conditionData') || isempty(GAN.conditionData)
        CBatch = dlarray(zeros(condDim,n,'single'),"CB");
        idx = zeros(n,1);
        return;
    end
    Data = single(GAN.conditionData);
    if size(Data,2) ~= condDim
        CBatch = dlarray(zeros(condDim,n,'single'),"CB");
        idx = zeros(n,1);
        return;
    end
    weights = StoredConditionSamplingWeights_BDG(GAN,size(Data,1));
    if isempty(weights)
        idx = randi(size(Data,1),1,n);
    else
        idx = WeightedSampleIndices_BDG(weights,n);
    end
    CBatch = dlarray(Data(idx,:)',"CB");
end

function weights = StoredConditionSamplingWeights_BDG(GAN,nData)
    weights = [];
    if ~isfield(GAN,'conditionSamplingWeights') || ...
            isempty(GAN.conditionSamplingWeights)
        return;
    end
    candidate = double(GAN.conditionSamplingWeights(:));
    if numel(candidate) ~= nData
        return;
    end
    candidate(~isfinite(candidate) | candidate < 0) = 0;
    total = sum(candidate);
    if total <= 0
        return;
    end
    weights = candidate ./ total;
end

function dim = StoredGeneratorConditionDim_BDG(GAN)
    if isfield(GAN,'generatorConditionDim')
        dim = max(0,round(double(GAN.generatorConditionDim)));
    elseif isfield(GAN,'conditionDim')
        dim = max(0,round(double(GAN.conditionDim)));
    else
        dim = 0;
    end
end

function netG = CreateGenerator_BDG(zDim,D,archMode,condDim)
    archMode = NormalizeArchMode_BDG(archMode);
    switch archMode
        case "large"
            widths = [64 128 64];
        case "g2sl"
            widths = [16 32 128];
        otherwise
            widths = [16 32];
    end
    layersG = [featureInputLayer(zDim + condDim,Normalization="none",Name="in")];
    for i = 1 : numel(widths)
        layersG = [layersG
            fullyConnectedLayer(widths(i),Name=sprintf("fc%d",i))]; %#ok<AGROW>
        if archMode == "g2sl"
            layersG = [layersG
                leakyReluLayer(0.2,Name=sprintf("lrelu%d",i))]; %#ok<AGROW>
        else
            layersG = [layersG
                reluLayer(Name=sprintf("relu%d",i))]; %#ok<AGROW>
        end
    end
    layersG = [layersG
        fullyConnectedLayer(D,Name=sprintf("fc%d",numel(widths)+1))
        tanhLayer(Name="tanh")];
    netG = dlnetwork(layerGraph(layersG));
end

function netD = CreateDirectBoundaryDiscriminator_BDG(D,archMode,condDim)
    switch NormalizeArchMode_BDG(archMode)
        case "large"
            widths = [64 128 64];
        case "g2sl"
            widths = [128 32 16];
        otherwise
            widths = [32 16];
    end
    layersD = [featureInputLayer(D + condDim,Normalization="none",Name="in")];
    for i = 1 : numel(widths)
        layersD = [layersD
            fullyConnectedLayer(widths(i),Name=sprintf("fc%d",i))
            leakyReluLayer(0.2,Name=sprintf("lrelu%d",i))]; %#ok<AGROW>
    end
    layersD = [layersD
        fullyConnectedLayer(1,Name=sprintf("fc%d",numel(widths)+1))
        sigmoidLayer(Name="sigmoid")];
    netD = dlnetwork(layerGraph(layersD));
end

function [lossD,gradD] = DirectBoundaryDiscriminatorGradients_BDG( ...
        netG,netD,XReal,CReal,Z,CFake,generatorCondDim,realLabel)
    if nargin < 7 || isempty(generatorCondDim)
        generatorCondDim = size(CFake,1);
    end
    if nargin < 8 || isempty(realLabel)
        realLabel = 1;
    end
    XFake = forward(netG,AppendGeneratorConditionToFeatures_BDG( ...
        Z,CFake,generatorCondDim));
    YReal = forward(netD,AppendConditionToFeatures_BDG(XReal,CReal));
    YFake = forward(netD,AppendConditionToFeatures_BDG(XFake,CFake));
    epsVal = single(1e-8);
    realLoss = mean(single(realLabel) .* ...
        log(ClipProb_BDG(YReal)+epsVal),'all');
    lossD = -(realLoss + ...
        mean(log(1-ClipProb_BDG(YFake)+epsVal),'all'));
    gradD = dlgradient(lossD,netD.Learnables);
end

function [lossD,gradD] = TargetConditionedDiscriminatorGradients_BDG( ...
        netG,netD,XReal,CReal,CMismatch,Z,generatorCondDim,realLabel)
    if nargin < 7 || isempty(generatorCondDim)
        generatorCondDim = size(CReal,1);
    end
    if nargin < 8 || isempty(realLabel)
        realLabel = 1;
    end
    XFake = forward(netG,AppendGeneratorConditionToFeatures_BDG( ...
        Z,CReal,generatorCondDim));
    YMatched = forward(netD,AppendConditionToFeatures_BDG(XReal,CReal));
    YMismatched = forward(netD,AppendConditionToFeatures_BDG(XReal,CMismatch));
    YFake = forward(netD,AppendConditionToFeatures_BDG(XFake,CReal));
    epsVal = single(1e-8);
    matchedLoss = mean(single(realLabel) .* ...
        log(ClipProb_BDG(YMatched)+epsVal),'all');
    mismatchLoss = mean(log(1-ClipProb_BDG(YMismatched)+epsVal),'all');
    fakeLoss = mean(log(1-ClipProb_BDG(YFake)+epsVal),'all');
    lossD = -(matchedLoss + mismatchLoss + fakeLoss);
    gradD = dlgradient(lossD,netD.Learnables);
end

function [lossG,gradG] = DirectBoundaryGeneratorGradients_BDG( ...
        netG,netD,Z,CFake,generatorCondDim)
    if nargin < 5 || isempty(generatorCondDim)
        generatorCondDim = size(CFake,1);
    end
    epsVal = single(1e-8);
    XFake = forward(netG,AppendGeneratorConditionToFeatures_BDG( ...
        Z,CFake,generatorCondDim));
    YFake = forward(netD,AppendConditionToFeatures_BDG(XFake,CFake));
    lossG = -mean(log(ClipProb_BDG(YFake)+epsVal),'all');
    gradG = dlgradient(lossG,netG.Learnables);
end

function XC = AppendConditionToFeatures_BDG(X,C)
    if isempty(C)
        XC = X;
    else
        XC = [X;C];
    end
end

function XC = AppendGeneratorConditionToFeatures_BDG(X,C,generatorCondDim)
    generatorCondDim = max(0,round(double(generatorCondDim)));
    if generatorCondDim == 0
        XC = X;
        return;
    end
    assert(~isempty(C) && size(C,1) >= generatorCondDim, ...
        'BoundaryGAN_BDG:MissingGeneratorCondition', ...
        'Generator condition data must have at least %d rows.', ...
        generatorCondDim);
    XC = [X;C(1:generatorCondDim,:)];
end

function XBatch = DrawUniformScaledBatch_BDG(lower,upper,nDiag)
    X = rand(nDiag,numel(lower)) .* (double(upper) - double(lower)) + ...
        double(lower);
    X = single(ScaleToTanh_BDG(X,lower,upper));
    XBatch = dlarray(X',"CB");
end

function P = FirstProb_BDG(Y)
    if isempty(Y)
        P = [];
        return;
    end
    P = min(max(double(extractdata(Y(:))),0),1);
end

function Diag = EmptyTrainDiagnostic_BDG()
    Diag = struct( ...
        'gan_d_diag_status',0, ...
        'gan_d_diag_count',0, ...
        'gan_d_real_count',0, ...
        'gan_d_ai_count',0, ...
        'gan_d_fake_count',0, ...
        'gan_d_mismatch_count',0, ...
        'gan_d_uniform_count',0, ...
        'gan_d_real_prob_mean',NaN, ...
        'gan_d_ai_prob_mean',NaN, ...
        'gan_d_fake_prob_mean',NaN, ...
        'gan_d_mismatch_prob_mean',NaN, ...
        'gan_d_uniform_prob_mean',NaN, ...
        'gan_d_prob_gap',NaN, ...
        'gan_d_confusion_abs_margin',NaN, ...
        'gan_d_real_acc',NaN, ...
        'gan_d_ai_acc',NaN, ...
        'gan_d_fake_acc',NaN, ...
        'gan_d_mismatch_acc',NaN, ...
        'gan_d_uniform_acc',NaN, ...
        'gan_d_bal_acc',NaN);
end

function Out = PrefixTrainDiagnostic_BDG(Diag,prefix)
    Out = struct();
    src = ["diag_status","diag_count","real_count","ai_count", ...
        "fake_count","mismatch_count","uniform_count", ...
        "real_prob_mean","fake_prob_mean","mismatch_prob_mean", ...
        "ai_prob_mean","uniform_prob_mean","prob_gap", ...
        "confusion_abs_margin","real_acc","ai_acc","fake_acc", ...
        "mismatch_acc","uniform_acc","bal_acc"];
    for i = 1 : numel(src)
        name = char("gan_d_" + src(i));
        if isfield(Diag,name)
            Out.([char(prefix),char(src(i))]) = Diag.(name);
        end
    end
end

function Diag = SummarizeTrainDiagnostic_BDG(PReal,PAI,PFake,PUniform, ...
        PMismatch)
    if nargin < 5
        PMismatch = [];
    end
    Diag = EmptyTrainDiagnostic_BDG();
    PReal = PReal(isfinite(PReal));
    PAI = PAI(isfinite(PAI));
    PFake = PFake(isfinite(PFake));
    PMismatch = PMismatch(isfinite(PMismatch));
    PUniform = PUniform(isfinite(PUniform));
    n = min([numel(PReal),numel(PFake),numel(PUniform)]);
    if n == 0
        return;
    end
    PReal = PReal(1:n);
    PFake = PFake(1:n);
    PUniform = PUniform(1:n);
    if ~isempty(PAI)
        PAI = PAI(1:min(n,numel(PAI)));
    end
    realMean = mean(PReal);
    fakeMean = mean(PFake);
    aiMean = MeanOrNaN_BDG(PAI);
    mismatchMean = MeanOrNaN_BDG(PMismatch);
    uniformMean = mean(PUniform);
    Diag.gan_d_diag_status = 1;
    Diag.gan_d_diag_count = double(n);
    Diag.gan_d_real_count = double(numel(PReal));
    Diag.gan_d_ai_count = double(numel(PAI));
    Diag.gan_d_fake_count = double(numel(PFake));
    Diag.gan_d_mismatch_count = double(numel(PMismatch));
    Diag.gan_d_uniform_count = double(numel(PUniform));
    Diag.gan_d_real_prob_mean = realMean;
    Diag.gan_d_ai_prob_mean = aiMean;
    Diag.gan_d_fake_prob_mean = fakeMean;
    Diag.gan_d_mismatch_prob_mean = mismatchMean;
    Diag.gan_d_uniform_prob_mean = uniformMean;
    negativeMeans = [fakeMean,aiMean,mismatchMean];
    negativeMeans = negativeMeans(isfinite(negativeMeans));
    Diag.gan_d_prob_gap = realMean - max(negativeMeans);
    Diag.gan_d_confusion_abs_margin = 0.5 .* ...
        (abs(realMean - 0.5) + abs(fakeMean - 0.5));
    Diag.gan_d_real_acc = mean(PReal >= 0.5);
    Diag.gan_d_ai_acc = MeanOrNaN_BDG(PAI < 0.5);
    Diag.gan_d_fake_acc = mean(PFake < 0.5);
    Diag.gan_d_mismatch_acc = MeanOrNaN_BDG(PMismatch < 0.5);
    Diag.gan_d_uniform_acc = mean(PUniform < 0.5);
    acc = [Diag.gan_d_real_acc,Diag.gan_d_fake_acc, ...
        Diag.gan_d_uniform_acc];
    if isfinite(Diag.gan_d_ai_acc)
        acc(end+1) = Diag.gan_d_ai_acc; %#ok<AGROW>
    end
    if isfinite(Diag.gan_d_mismatch_acc)
        acc(end+1) = Diag.gan_d_mismatch_acc; %#ok<AGROW>
    end
    Diag.gan_d_bal_acc = mean(acc);
end

function Diag = EmptyTargetConditionedTrainDiag_BDG()
    Diag = struct( ...
        'target_mismatch_count',0, ...
        'target_mismatch_identity_count',0, ...
        'target_mismatch_batch_count',0);
end

function Diag = AccumulateTargetConditionedDiag_BDG(Diag,Step)
    Diag.target_mismatch_count = Diag.target_mismatch_count + ...
        double(Step.target_mismatch_count);
    Diag.target_mismatch_identity_count = ...
        Diag.target_mismatch_identity_count + ...
        double(Step.target_mismatch_identity_count);
    Diag.target_mismatch_batch_count = Diag.target_mismatch_batch_count + 1;
end

function Diag = FinalizeTargetConditionedDiag_BDG(Diag)
    if Diag.target_mismatch_count > 0
        Diag.target_mismatch_identity_rate = ...
            Diag.target_mismatch_identity_count ./ ...
            Diag.target_mismatch_count;
    else
        Diag.target_mismatch_identity_rate = NaN;
    end
end

function P = ClipProb_BDG(P)
    P = min(max(P,single(1e-7)),single(1-1e-7));
end

function X = ScaleToTanh_BDG(X,lower,upper)
    X = 2*((double(X) - double(lower))./ ...
        (double(upper) - double(lower) + 1e-12)) - 1;
    X = min(max(X,-1),1);
end

function flag = IsFeasibleSet_BDG(P)
    if isempty(P)
        flag = false(0,1);
    elseif isempty(P.cons)
        flag = true(numel(P),1);
    else
        flag = all(P.cons <= 0,2);
    end
end

function value = MeanOrNaN_BDG(x)
    x = double(x(:));
    x = x(isfinite(x));
    if isempty(x)
        value = NaN;
    else
        value = mean(x);
    end
end

function value = MedianOrNaN_BDG(x)
    x = double(x(:));
    x = x(isfinite(x));
    if isempty(x)
        value = NaN;
    else
        value = median(x);
    end
end

function value = PercentileOrNaN_BDG(x,p)
    x = sort(double(x(isfinite(x))));
    if isempty(x)
        value = NaN;
        return;
    end
    p = min(max(double(p),0),100);
    value = x(max(1,ceil((p/100)*numel(x))));
end
