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
        case 'diagnoselatentscale'
            GAN = varargin{1};
            TrainX = varargin{2};
            TrainC = varargin{3};
            Problem = varargin{4};
            Options = fillOptions(varargin{5});
            [varargout{1:nargout}] = latentScaleCoverageDiagnostics( ...
                GAN,TrainX,TrainC,Problem,Options);
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
            ~isequal(GAN.criticHidden,Options.criticHidden) || ...
            ~isfield(GAN,'structuredZMode') || ...
            string(GAN.structuredZMode) ~= Options.structuredZMode || ...
            ~isfield(GAN,'structuredZMaxModes') || ...
            GAN.structuredZMaxModes ~= Options.structuredZMaxModes
        GAN = initializeWGAN(D,M,Options.zDim,lower,upper, ...
            Options.generatorHidden,Options.criticHidden,Options);
    end
    TrainActiveModeCount = conditionActiveModeCounts(TrainC,Options);

    if Options.captureTrainHistory
        GAN.train_history = repmat(emptyWGANHistoryRow(),0,1);
    else
        GAN.train_history = struct([]);
    end
    GAN = resetWGANMappingDiagnostics(GAN);
    NewIdx = zeros(0,1);
    if Options.mappingDiagnostics
        [NewIdx,seenCount] = prequentialRowPartition(GAN,TrainX,TrainC);
        GAN.prequential_new_count = numel(NewIdx);
        GAN.prequential_seen_count = seenCount;
        Pre = prequentialMappingProbe(GAN,XScaled,TrainC,NewIdx,Options);
        GAN = copyPrequentialProbe(GAN,Pre,"pre");
    end

    [TrainIdx,HoldoutIdx] = splitWGANTrainHoldout(N);
    trainPoolN = numel(TrainIdx);
    miniBatch = min(Options.miniBatch,trainPoolN);
    beta1 = 0.0;
    beta2 = 0.9;
    for iter = 1 : Options.iter
        criticTimer = tic;
        for cStep = 1 : Options.nCritic
            idx = TrainIdx(randi(trainPoolN,1,miniBatch));
            [GAN,lossInfo] = updateCriticBatch(GAN,XScaled,TrainC, ...
                TrainActiveModeCount,idx,Options,beta1,beta2);
            GAN.last_critic_loss = lossInfo.lossC;
            GAN.last_gradient_penalty = lossInfo.gp;
            GAN.last_score_real = lossInfo.scoreReal;
            GAN.last_score_fake = lossInfo.scoreFake;
        end
        elapsedCritic = toc(criticTimer);
        generatorTimer = tic;
        idx = TrainIdx(randi(trainPoolN,1,miniBatch));
        [GAN,generatorInfo] = updateGeneratorBatch(GAN, ...
            TrainC,TrainActiveModeCount,idx,Options,beta1,beta2);
        GAN.last_generator_loss = generatorInfo.lossG;
        GAN.last_generator_adversarial_loss = generatorInfo.lossAdv;
        GAN.last_mi_loss = generatorInfo.lossMI;
        GAN.last_mode_decoder_accuracy = generatorInfo.modeAccuracy;
        elapsedGenerator = toc(generatorTimer);
        if Options.captureTrainHistory
            GAN.train_history(end+1,1) = wganHistoryRow( ...
                iter,GAN,elapsedCritic,elapsedGenerator);
        end
    end
    if Options.mappingDiagnostics
        Post = prequentialMappingProbe(GAN,XScaled,TrainC,NewIdx,Options);
        GAN = copyPrequentialProbe(GAN,Post,"post");
        GAN = appendDeterministicConditionDiagnostics( ...
            GAN,TrainC,Options);
        GAN = rememberSeenTrainingRows(GAN,TrainX,TrainC);
    end
    GAN = appendWGANTrainingDiagnostics(GAN,XScaled,TrainC,Options, ...
        TrainIdx,HoldoutIdx);
end

function Row = emptyWGANHistoryRow()
    Row = struct( ...
        'step',NaN, ...
        'loss_d',NaN, ...
        'loss_g',NaN, ...
        'loss_g_adversarial',NaN, ...
        'loss_mi',NaN, ...
        'mode_decoder_accuracy',NaN, ...
        'gradient_penalty',NaN, ...
        'd_real_acc',NaN, ...
        'd_fake_acc',NaN, ...
        'd_bal_acc',NaN, ...
        'g_fool_rate',NaN, ...
        'score_real_mean',NaN, ...
        'score_fake_mean',NaN, ...
        'score_random_mean',NaN, ...
        'random_as_fake_rate',NaN, ...
        'elapsed_critic',NaN, ...
        'elapsed_generator',NaN);
end

function Row = wganHistoryRow(step,GAN,elapsedCritic,elapsedGenerator)
    Row = emptyWGANHistoryRow();
    Row.step = double(step);
    Row.loss_d = double(GAN.last_critic_loss);
    Row.loss_g = double(GAN.last_generator_loss);
    Row.loss_g_adversarial = double(GAN.last_generator_adversarial_loss);
    Row.loss_mi = double(GAN.last_mi_loss);
    Row.mode_decoder_accuracy = double(GAN.last_mode_decoder_accuracy);
    Row.gradient_penalty = double(GAN.last_gradient_penalty);
    Row.score_real_mean = double(GAN.last_score_real);
    Row.score_fake_mean = double(GAN.last_score_fake);
    Row.elapsed_critic = double(elapsedCritic);
    Row.elapsed_generator = double(elapsedGenerator);
end

function GAN = resetWGANMappingDiagnostics(GAN)
    if ~isfield(GAN,'seen_train_x')
        GAN.seen_train_x = zeros(0,GAN.D);
    end
    if ~isfield(GAN,'seen_train_c')
        GAN.seen_train_c = zeros(0,GAN.M);
    end
    GAN.prequential_new_count = 0;
    GAN.prequential_seen_count = 0;
    prefixes = ["pre","post"];
    for i = 1 : numel(prefixes)
        prefix = "prequential_" + prefixes(i) + "_";
        GAN.(char(prefix + "critic_gap")) = NaN;
        GAN.(char(prefix + "dec_dist50")) = NaN;
        GAN.(char(prefix + "dec_dist90")) = NaN;
    end
    GAN.condition_diag_condition_count = 0;
    GAN.condition_diag_z_count = 0;
    GAN.same_z_diff_c_dec_median = NaN;
    GAN.same_c_diff_z_dec_median = NaN;
    GAN.same_c_diff_z_collapse_rate = NaN;
    GAN.condition_effect_ratio_dec = NaN;
end

function [NewIdx,seenCount] = prequentialRowPartition(GAN,TrainX,TrainC)
    if ~isfield(GAN,'seen_train_x') || isempty(GAN.seen_train_x) || ...
            ~isfield(GAN,'seen_train_c') || isempty(GAN.seen_train_c)
        seen = false(size(TrainX,1),1);
    else
        seen = ismember([double(TrainX),double(TrainC)], ...
            [double(GAN.seen_train_x),double(GAN.seen_train_c)],'rows');
    end
    NewIdx = find(~seen);
    seenCount = sum(seen);
end

function Probe = prequentialMappingProbe(GAN,XScaled,TrainC,Idx,Options)
    Probe = struct('critic_gap',NaN,'dec_dist50',NaN,'dec_dist90',NaN);
    Idx = round(double(Idx(:)));
    if isempty(Idx)
        return;
    end
    C = double(TrainC(Idx,:));
    activeModeCount = conditionActiveModeCounts(C,Options);
    Z = deterministicLatent(Options,GAN.zDim,numel(Idx),activeModeCount);
    FakeScaled = generateScaledDecisions(GAN,C,Z);
    dlReal = dlarray(single(XScaled(Idx,:)'),'CB');
    dlFake = dlarray(single(FakeScaled'),'CB');
    dlC = dlarray(single(C'),'CB');
    scoreReal = double(gather(extractdata( ...
        forward(GAN.netC,[dlReal;dlC]))));
    scoreFake = double(gather(extractdata( ...
        forward(GAN.netC,[dlFake;dlC]))));
    Probe.critic_gap = mean(scoreReal(:),'omitnan') - ...
        mean(scoreFake(:),'omitnan');
    dist = sqrt(mean(((FakeScaled - XScaled(Idx,:))/2).^2,2));
    Probe.dec_dist50 = percentileFinite(dist,50);
    Probe.dec_dist90 = percentileFinite(dist,90);
end

function GAN = copyPrequentialProbe(GAN,Probe,stage)
    prefix = "prequential_" + string(stage) + "_";
    GAN.(char(prefix + "critic_gap")) = Probe.critic_gap;
    GAN.(char(prefix + "dec_dist50")) = Probe.dec_dist50;
    GAN.(char(prefix + "dec_dist90")) = Probe.dec_dist90;
end

function GAN = rememberSeenTrainingRows(GAN,TrainX,TrainC)
    All = [[double(GAN.seen_train_x),double(GAN.seen_train_c)]; ...
        [double(TrainX),double(TrainC)]];
    if isempty(All)
        GAN.seen_train_x = zeros(0,GAN.D);
        GAN.seen_train_c = zeros(0,GAN.M);
    else
        All = unique(All,'rows','stable');
        GAN.seen_train_x = All(:,1:GAN.D);
        GAN.seen_train_c = All(:,GAN.D+1:end);
    end
end

function GAN = appendDeterministicConditionDiagnostics(GAN,TrainC,Options)
    C = unique(double(TrainC),'rows','stable');
    if isempty(C)
        return;
    end
    maxConditions = min(size(C,1),Options.mappingDiagMaxConditions);
    if size(C,1) > maxConditions
        selected = unique(round(linspace(1,size(C,1),maxConditions)), ...
            'stable');
        C = C(selected,:);
    end
    zCount = Options.mappingDiagZSamples;
    Z = deterministicLatent(Options,GAN.zDim,zCount, ...
        repmat(Options.structuredZMaxModes,zCount,1));
    conditionCount = size(C,1);
    sameZ = NaN(zCount,1);
    for z = 1 : zCount
        X = generateScaledDecisions(GAN,C,repmat(Z(z,:),conditionCount,1));
        sameZ(z) = pairwiseRmsMedian((X + 1)/2);
    end
    sameC = NaN(conditionCount,1);
    for c = 1 : conditionCount
        X = generateScaledDecisions(GAN,repmat(C(c,:),zCount,1),Z);
        sameC(c) = pairwiseRmsMedian((X + 1)/2);
    end
    GAN.condition_diag_condition_count = conditionCount;
    GAN.condition_diag_z_count = zCount;
    GAN.same_z_diff_c_dec_median = median(sameZ,'omitnan');
    GAN.same_c_diff_z_dec_median = median(sameC,'omitnan');
    GAN.same_c_diff_z_collapse_rate = mean(sameC <= 1e-6,'omitnan');
    GAN.condition_effect_ratio_dec = GAN.same_z_diff_c_dec_median / ...
        max(GAN.same_c_diff_z_dec_median,eps);
end

function [Rows,Summary] = latentScaleCoverageDiagnostics( ...
        GAN,TrainX,TrainC,Problem,Options)
    condDim = size(TrainC,2);
    Rows = repmat(emptyLatentScaleConditionRow(condDim),0,1);
    Summary = emptyLatentScaleSummary();
    if isempty(GAN) || ~isstruct(GAN) || ~isfield(GAN,'netG') || ...
            isempty(TrainX) || isempty(TrainC)
        return;
    end
    TrainX = double(TrainX);
    TrainC = double(TrainC);
    if size(TrainX,1) ~= size(TrainC,1) || ...
            size(TrainX,2) ~= GAN.D || size(TrainC,2) ~= GAN.M
        error('CBSRegionWGAN:BadLatentScaleTrainingData', ...
            'TrainX/TrainC dimensions must match the trained WGAN.');
    end
    lower = double(Problem.lower);
    upper = double(Problem.upper);
    span = upper - lower;
    span(span <= eps) = 1;
    XScaled = 2*(TrainX - lower)./span - 1;
    XScaled = max(-1,min(1,XScaled));
    valid = all(isfinite(XScaled),2) & all(isfinite(TrainC),2);
    XScaled = XScaled(valid,:);
    TrainC = TrainC(valid,:);
    if isempty(TrainC)
        return;
    end

    [Conditions,~,group] = unique(TrainC,'rows','stable');
    zCount = Options.mappingDiagZSamples;
    trainSigma = latentSigma(Options,"train");
    sampleSigma = latentSigma(Options,"sample");
    AllC = repelem(Conditions,zCount,1);
    if structuredZEnabled(Options)
        conditionCounts = accumarray(group,1,[size(Conditions,1),1]);
        activeModeCount = min(Options.structuredZMaxModes,conditionCounts);
        AllActiveModeCount = repelem(activeModeCount,zCount,1);
        ordinal = repmat((1:zCount)',size(Conditions,1),1);
        AllZTrain = deterministicStructuredLatent(Options,GAN.zDim, ...
            AllActiveModeCount,ordinal,trainSigma);
        AllZSample = deterministicStructuredLatent(Options,GAN.zDim, ...
            AllActiveModeCount,ordinal,sampleSigma);
    else
        Z0 = deterministicStandardLatent(GAN.zDim,zCount);
        AllZ0 = repmat(Z0,size(Conditions,1),1);
        AllZTrain = trainSigma*AllZ0;
        AllZSample = sampleSigma*AllZ0;
    end
    AllGeneratedTrain01 = (generateScaledDecisions( ...
        GAN,AllC,AllZTrain) + 1)/2;
    AllGeneratedSample01 = (generateScaledDecisions( ...
        GAN,AllC,AllZSample) + 1)/2;
    Rows = repmat(emptyLatentScaleConditionRow(condDim), ...
        size(Conditions,1),1);
    for c = 1 : size(Conditions,1)
        trainRows = group == c;
        Train01 = (XScaled(trainRows,:) + 1)/2;
        generatedRows = (c - 1)*zCount + (1:zCount);
        GeneratedTrain01 = AllGeneratedTrain01(generatedRows,:);
        GeneratedSample01 = AllGeneratedSample01(generatedRows,:);
        trainPair = pairwiseRmsDistances(Train01);
        trainPair90 = percentileFinite(trainPair,90);
        TrainScale = latentScaleMetrics( ...
            GeneratedTrain01,Train01,trainPair90);
        SampleScale = latentScaleMetrics( ...
            GeneratedSample01,Train01,trainPair90);

        Rows(c).condition = Conditions(c,:);
        Rows(c).train_count = size(Train01,1);
        Rows(c).train_dec_pair_count = numel(trainPair);
        Rows(c).train_dec_pair90 = trainPair90;
        Rows(c).z_count = zCount;
        Rows(c).train_sigma = trainSigma;
        Rows(c).sample_sigma = sampleSigma;
        Rows(c) = copyLatentScaleMetrics(Rows(c),TrainScale,"train_scale");
        Rows(c) = copyLatentScaleMetrics(Rows(c),SampleScale,"sample_scale");
    end

    Summary.latent_scale_ref_count = numel(Rows);
    Summary.latent_scale_z_count = zCount;
    Summary.latent_scale_train_sigma = trainSigma;
    Summary.latent_scale_sample_sigma = sampleSigma;
    Summary.latent_scale_train_gen_to_train_dec50_median = ...
        medianLatentScaleField(Rows,'train_scale_gen_to_train_dec50');
    Summary.latent_scale_train_gen_to_train_dec90_median = ...
        medianLatentScaleField(Rows,'train_scale_gen_to_train_dec90');
    Summary.latent_scale_sample_gen_to_train_dec50_median = ...
        medianLatentScaleField(Rows,'sample_scale_gen_to_train_dec50');
    Summary.latent_scale_sample_gen_to_train_dec90_median = ...
        medianLatentScaleField(Rows,'sample_scale_gen_to_train_dec90');
    Summary.latent_scale_train_anchor_utilization_median = ...
        medianLatentScaleField(Rows, ...
        'train_scale_anchor_utilization_rate');
    Summary.latent_scale_sample_anchor_utilization_median = ...
        medianLatentScaleField(Rows, ...
        'sample_scale_anchor_utilization_rate');
    Summary.latent_scale_train_diversity_ratio90_median = ...
        medianLatentScaleField(Rows,'train_scale_diversity_ratio90');
    Summary.latent_scale_sample_diversity_ratio90_median = ...
        medianLatentScaleField(Rows,'sample_scale_diversity_ratio90');
end

function Row = emptyLatentScaleConditionRow(condDim)
    Row = struct( ...
        'condition',zeros(1,condDim), ...
        'train_count',0, ...
        'train_dec_pair_count',0, ...
        'train_dec_pair90',NaN, ...
        'z_count',0, ...
        'train_sigma',NaN, ...
        'sample_sigma',NaN, ...
        'train_scale_generated_dec_pair_count',0, ...
        'train_scale_generated_dec_pair90',NaN, ...
        'train_scale_gen_to_train_dec50',NaN, ...
        'train_scale_gen_to_train_dec90',NaN, ...
        'train_scale_anchor_utilized_count',0, ...
        'train_scale_anchor_utilization_rate',NaN, ...
        'train_scale_diversity_ratio90',NaN, ...
        'sample_scale_generated_dec_pair_count',0, ...
        'sample_scale_generated_dec_pair90',NaN, ...
        'sample_scale_gen_to_train_dec50',NaN, ...
        'sample_scale_gen_to_train_dec90',NaN, ...
        'sample_scale_anchor_utilized_count',0, ...
        'sample_scale_anchor_utilization_rate',NaN, ...
        'sample_scale_diversity_ratio90',NaN);
end

function Summary = emptyLatentScaleSummary()
    Summary = struct( ...
        'latent_scale_ref_count',0, ...
        'latent_scale_z_count',0, ...
        'latent_scale_train_sigma',NaN, ...
        'latent_scale_sample_sigma',NaN, ...
        'latent_scale_train_gen_to_train_dec50_median',NaN, ...
        'latent_scale_train_gen_to_train_dec90_median',NaN, ...
        'latent_scale_sample_gen_to_train_dec50_median',NaN, ...
        'latent_scale_sample_gen_to_train_dec90_median',NaN, ...
        'latent_scale_train_anchor_utilization_median',NaN, ...
        'latent_scale_sample_anchor_utilization_median',NaN, ...
        'latent_scale_train_diversity_ratio90_median',NaN, ...
        'latent_scale_sample_diversity_ratio90_median',NaN);
end

function Metrics = latentScaleMetrics(Generated,Train,trainPair90)
    generatedPair = pairwiseRmsDistances(Generated);
    distance = pointRmsDistances(Generated,Train);
    [nearestDistance,nearestAnchor] = min(distance,[],2);
    utilizedCount = numel(unique(nearestAnchor));
    Metrics = struct( ...
        'generated_dec_pair_count',numel(generatedPair), ...
        'generated_dec_pair90',percentileFinite(generatedPair,90), ...
        'gen_to_train_dec50',percentileFinite(nearestDistance,50), ...
        'gen_to_train_dec90',percentileFinite(nearestDistance,90), ...
        'anchor_utilized_count',utilizedCount, ...
        'anchor_utilization_rate',utilizedCount/max(1,size(Train,1)), ...
        'diversity_ratio90',safeFiniteRatio( ...
        percentileFinite(generatedPair,90),trainPair90));
end

function Row = copyLatentScaleMetrics(Row,Metrics,prefix)
    names = fieldnames(Metrics);
    prefix = char(string(prefix) + "_");
    for i = 1 : numel(names)
        Row.([prefix,names{i}]) = double(Metrics.(names{i}));
    end
end

function value = medianLatentScaleField(Rows,name)
    if isempty(Rows)
        value = NaN;
        return;
    end
    value = median(double([Rows.(name)]),'omitnan');
end

function value = safeFiniteRatio(numerator,denominator)
    if isfinite(numerator) && isfinite(denominator) && denominator > eps
        value = numerator/denominator;
    else
        value = NaN;
    end
end

function distance = pointRmsDistances(A,B)
    distance = zeros(size(A,1),size(B,1));
    for i = 1 : size(B,1)
        distance(:,i) = sqrt(mean((A - B(i,:)).^2,2));
    end
end

function values = pairwiseRmsDistances(X)
    n = size(X,1);
    if n < 2
        values = zeros(0,1);
        return;
    end
    values = zeros(n*(n-1)/2,1);
    row = 0;
    for i = 1 : n-1
        d = sqrt(mean((X(i+1:n,:) - X(i,:)).^2,2));
        values(row+1:row+numel(d)) = d;
        row = row + numel(d);
    end
end

function value = pairwiseRmsMedian(X)
    values = pairwiseRmsDistances(X);
    if isempty(values)
        value = NaN;
        return;
    end
    value = median(values,'omitnan');
end

function Z = deterministicLatent(Options,zDim,n,activeModeCount)
    if nargin < 4 || isempty(activeModeCount)
        activeModeCount = repmat(Options.structuredZMaxModes,n,1);
    end
    if structuredZEnabled(Options)
        Z = deterministicStructuredLatent(Options,zDim, ...
            activeModeCount,(1:n)',latentSigma(Options,"train"));
    else
        Z = latentSigma(Options,"train")* ...
            deterministicStandardLatent(zDim,n);
    end
end

function Z = deterministicStructuredLatent(Options,zDim, ...
        activeModeCount,ordinal,sigma)
    n = numel(activeModeCount);
    activeModeCount = normalizeActiveModeCounts(activeModeCount,n,Options);
    ordinal = round(double(ordinal(:)));
    if numel(ordinal) ~= n
        error('CBSRegionWGAN:BadStructuredZOrdinal', ...
            'Structured-z ordinal count must match active-mode count.');
    end
    labels = 1 + mod(ordinal - 1,activeModeCount);
    Z = zeros(n,zDim);
    linear = sub2ind([n,zDim],(1:n)',labels);
    Z(linear) = 1;
    residualDim = zDim - Options.structuredZMaxModes;
    if residualDim > 0
        Z(:,Options.structuredZMaxModes+1:end) = sigma* ...
            deterministicStandardLatent(residualDim,n);
    end
end

function Z = deterministicStandardLatent(zDim,n)
    total = max(1,n*zDim);
    probability = ((1:total) - 0.5)/total;
    values = sqrt(2)*erfinv(2*probability - 1);
    Z = reshape(values,zDim,n)';
    Z(~isfinite(Z)) = 0;
end

function value = percentileFinite(values,p)
    values = sort(double(values(:)));
    values = values(isfinite(values));
    if isempty(values)
        value = NaN;
        return;
    end
    if isscalar(values)
        value = values(1);
        return;
    end
    position = 1 + (numel(values) - 1)*double(p)/100;
    lo = floor(position);
    hi = ceil(position);
    value = values(lo) + (position - lo)*(values(hi) - values(lo));
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
    activeModeCount = conditionActiveModeCounts(TrainC,Options);
    dlZ = dlarray(single(latentSamples(Options,GAN.zDim,batchN, ...
        "train",activeModeCount(idx))'),'CB');
    dlFake = forward(GAN.netG,[dlZ;dlC]);
    scoreReal = forward(GAN.netC,[dlX;dlC]);
    scoreFake = forward(GAN.netC,[dlFake;dlC]);
    scoreReal = double(gather(extractdata(scoreReal)))';
    scoreFake = double(gather(extractdata(scoreFake)))';
    scoreReal = scoreReal(:);
    scoreFake = scoreFake(:);
end

function [GAN,Info] = updateCriticBatch(GAN,XScaled,TrainC, ...
        TrainActiveModeCount,idx,Options,beta1,beta2)
    idx = round(double(idx(:)'));
    batchN = numel(idx);
    dlX = dlarray(single(XScaled(idx,:)'),'CB');
    dlC = dlarray(single(TrainC(idx,:)'),'CB');
    dlZ = dlarray(single(latentSamples(Options,Options.zDim,batchN, ...
        "train",TrainActiveModeCount(idx))'),'CB');
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

function [GAN,Info] = updateGeneratorBatch(GAN,TrainC, ...
        TrainActiveModeCount,idx,Options,beta1,beta2)
    idx = round(double(idx(:)'));
    batchN = numel(idx);
    dlC = dlarray(single(TrainC(idx,:)'),'CB');
    [Z,labels] = latentSamples(Options,Options.zDim,batchN, ...
        "train",TrainActiveModeCount(idx));
    dlZ = dlarray(single(Z'),'CB');
    if structuredZEnabled(Options)
        dlTarget = dlarray(modeTargets(labels, ...
            Options.structuredZMaxModes),'CB');
        [gradG,gradQ,lossG,lossAdv,lossMI,modeLogits] = dlfeval( ...
            @structuredGeneratorGradients,GAN.netG,GAN.netC,GAN.netQ, ...
            dlC,dlZ,dlTarget,Options.structuredZLambda);
        GAN.iterQ = GAN.iterQ + 1;
        [GAN.netQ,GAN.avgQ,GAN.avgSqQ] = adamupdate( ...
            GAN.netQ,gradQ,GAN.avgQ,GAN.avgSqQ,GAN.iterQ, ...
            Options.lrG,beta1,beta2);
        [predicted,~] = modePredictionsFromLogits(modeLogits);
        modeAccuracy = mean(double(predicted == labels));
        lossMI = scalarExtract(lossMI);
        lossAdv = scalarExtract(lossAdv);
    else
        [gradG,lossG] = dlfeval(@generatorGradients, ...
            GAN.netG,GAN.netC,dlC,dlZ);
        lossAdv = scalarExtract(lossG);
        lossMI = NaN;
        modeAccuracy = NaN;
    end
    GAN.iterG = GAN.iterG + 1;
    [GAN.netG,GAN.avgG,GAN.avgSqG] = adamupdate(GAN.netG,gradG, ...
        GAN.avgG,GAN.avgSqG,GAN.iterG,Options.lrG,beta1,beta2);
    Info = struct( ...
        'lossG',scalarExtract(lossG), ...
        'lossAdv',lossAdv, ...
        'lossMI',lossMI, ...
        'modeAccuracy',modeAccuracy);
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

function [gradG,gradQ,lossG,lossAdv,lossMI,modeLogits] = ...
        structuredGeneratorGradients(netG,netC,netQ,dlC,dlZ, ...
        dlTarget,lambdaMI)
    dlFake = forward(netG,[dlZ;dlC]);
    scoreFake = forward(netC,[dlFake;dlC]);
    modeLogits = forward(netQ,[dlFake;dlC]);
    lossAdv = -mean(scoreFake,'all');
    shifted = modeLogits - max(modeLogits,[],1);
    logProb = shifted - log(sum(exp(shifted),1));
    lossMI = -mean(sum(dlTarget.*logProb,1),'all');
    lossG = lossAdv + single(lambdaMI)*lossMI;
    [gradG,gradQ] = dlgradient(lossG, ...
        netG.Learnables,netQ.Learnables);
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
    activeModeCount = queryActiveModeCounts(Options,size(QueryC,1));
    activeModeCount = repelem(activeModeCount,K,1);
    [Z,modeLabel,activeModeCount] = latentSamples( ...
        Options,GAN.zDim,n,"sample",activeModeCount);
    Dec = generateDecisions(GAN,C,Z);
    Scores = criticScores(GAN,Dec,C);
    Info.z = Z;
    Info.generated_critic_score = Scores;
    if structuredZEnabled(Options)
        Info.mode_label = modeLabel;
        Info.active_mode_count = activeModeCount;
        [prediction,confidence] = modeDecoderPredictions(GAN,Dec,C);
        Info.mode_prediction = prediction;
        Info.mode_confidence = confidence;
        Info.mode_correct = double(prediction == modeLabel);
    end
end

function Info = emptySampleInfo(M)
    Info = struct( ...
        'query_index',zeros(0,1), ...
        'condition',zeros(0,M), ...
        'z',zeros(0,0), ...
        'generated_critic_score',zeros(0,1), ...
        'mode_label',zeros(0,1), ...
        'active_mode_count',zeros(0,1), ...
        'mode_prediction',zeros(0,1), ...
        'mode_confidence',zeros(0,1), ...
        'mode_correct',zeros(0,1));
end

function Dec = generateDecisions(GAN,C,Z)
    XScaled = generateScaledDecisions(GAN,C,Z);
    span = GAN.upper - GAN.lower;
    span(span <= eps) = 1;
    Dec = GAN.lower + (XScaled + 1).*span/2;
    Dec = max(min(Dec,GAN.upper),GAN.lower);
end

function XScaled = generateScaledDecisions(GAN,C,Z)
    dlC = dlarray(single(C'),'CB');
    dlZ = dlarray(single(Z'),'CB');
    dlX = forward(GAN.netG,[dlZ;dlC]);
    XScaled = double(gather(extractdata(dlX)))';
    XScaled = max(-1,min(1,XScaled));
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

function [prediction,confidence] = modeDecoderPredictions(GAN,Dec,C)
    span = GAN.upper - GAN.lower;
    span(span <= eps) = 1;
    XScaled = 2*(double(Dec) - GAN.lower)./span - 1;
    XScaled = max(-1,min(1,XScaled));
    dlX = dlarray(single(XScaled'),'CB');
    dlC = dlarray(single(double(C)'),'CB');
    logits = forward(GAN.netQ,[dlX;dlC]);
    [prediction,confidence] = modePredictionsFromLogits(logits);
end

function [prediction,confidence] = modePredictionsFromLogits(logits)
    values = double(gather(extractdata(logits)));
    values = values - max(values,[],1);
    probabilities = exp(values);
    probabilities = probabilities./max(sum(probabilities,1),eps);
    [confidence,prediction] = max(probabilities,[],1);
    prediction = prediction(:);
    confidence = confidence(:);
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

function [Z,labels,activeModeCount] = latentSamples( ...
        Options,zDim,n,purpose,activeModeCount)
    if nargin < 4 || isempty(purpose)
        purpose = "sample";
    end
    if nargin < 5
        activeModeCount = [];
    end
    purpose = lower(strtrim(string(purpose)));
    labels = zeros(n,1);
    activeModeCount = normalizeActiveModeCounts( ...
        activeModeCount,n,Options);
    if isfield(Options,'sampleZ') && ~isempty(Options.sampleZ)
        Z = double(Options.sampleZ);
        if isequal(size(Z),[zDim n])
            Z = Z';
        end
        if ~isequal(size(Z),[n zDim])
            error('CBSRegionWGAN:BadSampleZ', ...
                'Options.sampleZ must be n-by-zDim or zDim-by-n.');
        end
        if structuredZEnabled(Options)
            [~,labels] = max(Z(:,1:Options.structuredZMaxModes),[],2);
        end
        return;
    end
    if structuredZEnabled(Options)
        labels = 1 + floor(rand(n,1).*activeModeCount);
        labels = min(labels,activeModeCount);
        Z = zeros(n,zDim);
        linear = sub2ind([n,zDim],(1:n)',labels);
        Z(linear) = 1;
        residualDim = zDim - Options.structuredZMaxModes;
        if residualDim > 0
            Z(:,Options.structuredZMaxModes+1:end) = ...
                latentSigma(Options,purpose)*randn(n,residualDim);
        end
    else
        Z = latentSigma(Options,purpose)*randn(n,zDim);
    end
    Z(~isfinite(Z)) = 0;
end

function GAN = initializeWGAN(D,M,zDim,lower,upper,generatorHidden, ...
        criticHidden,Options)
    GAN = struct();
    GAN.D = D;
    GAN.M = M;
    GAN.zDim = zDim;
    GAN.generatorHidden = generatorHidden;
    GAN.criticHidden = criticHidden;
    GAN.structuredZMode = Options.structuredZMode;
    GAN.structuredZMaxModes = Options.structuredZMaxModes;
    GAN.lower = lower;
    GAN.upper = upper;
    GAN.netG = createGenerator(M + zDim,D,generatorHidden);
    GAN.netC = createCritic(D + M,criticHidden);
    GAN.netQ = [];
    if structuredZEnabled(Options)
        GAN.netQ = createModeDecoder(D + M,Options.structuredZMaxModes);
    end
    GAN.avgG = [];
    GAN.avgSqG = [];
    GAN.avgC = [];
    GAN.avgSqC = [];
    GAN.avgQ = [];
    GAN.avgSqQ = [];
    GAN.iterG = 0;
    GAN.iterC = 0;
    GAN.iterQ = 0;
    GAN.last_critic_loss = NaN;
    GAN.last_generator_loss = NaN;
    GAN.last_generator_adversarial_loss = NaN;
    GAN.last_mi_loss = NaN;
    GAN.last_mode_decoder_accuracy = NaN;
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

function netQ = createModeDecoder(inputDim,modeCount)
    layers = [ ...
        featureInputLayer(inputDim,'Normalization','none','Name','q_in'); ...
        fullyConnectedLayer(modeCount,'Name','q_out')];
    netQ = dlnetwork(layerGraph(layers));
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
    Options = ensureField(Options,'captureTrainHistory',false);
    Options = ensureField(Options,'mappingDiagnostics',false);
    Options = ensureField(Options,'mappingDiagMaxConditions',8);
    Options = ensureField(Options,'mappingDiagZSamples',8);
    Options = ensureField(Options,'generatorHidden',[32 32]);
    Options = ensureField(Options,'criticHidden',[32 32]);
    Options = ensureField(Options,'structuredZMode',"off");
    Options = ensureField(Options,'structuredZMaxModes',5);
    Options = ensureField(Options,'structuredZLambda',1.0);
    Options = ensureField(Options,'queryActiveModeCount',[]);
    Options.zDim = max(1,round(double(Options.zDim)));
    Options.iter = max(0,round(double(Options.iter)));
    Options.miniBatch = max(1,round(double(Options.miniBatch)));
    Options.lrD = double(Options.lrD);
    Options.lrG = double(Options.lrG);
    Options.gpLambda = max(0,double(Options.gpLambda));
    Options.nCritic = max(1,round(double(Options.nCritic)));
    Options.trainSigma = optionalFiniteSigma(Options.trainSigma);
    Options.sampleSigma = optionalFiniteSigma(Options.sampleSigma);
    Options.captureTrainHistory = logical(Options.captureTrainHistory);
    Options.mappingDiagnostics = logical(Options.mappingDiagnostics);
    Options.mappingDiagMaxConditions = max(1,round(double( ...
        Options.mappingDiagMaxConditions)));
    Options.mappingDiagZSamples = max(2,round(double( ...
        Options.mappingDiagZSamples)));
    Options.generatorHidden = normalizeHiddenVector( ...
        Options.generatorHidden,[32 32]);
    Options.criticHidden = normalizeHiddenVector(Options.criticHidden,[32 32]);
    Options.structuredZMode = lower(strip(string( ...
        Options.structuredZMode)));
    if ~isscalar(Options.structuredZMode) || ...
            ~ismember(Options.structuredZMode,["off","categorical_mi"])
        error('CBSRegionWGAN:BadStructuredZMode', ...
            'structuredZMode must be "off" or "categorical_mi".');
    end
    Options.structuredZMaxModes = max(1,round(double( ...
        Options.structuredZMaxModes)));
    Options.structuredZLambda = double(Options.structuredZLambda);
    if ~isscalar(Options.structuredZLambda) || ...
            ~isfinite(Options.structuredZLambda) || ...
            Options.structuredZLambda < 0
        error('CBSRegionWGAN:BadStructuredZLambda', ...
            'structuredZLambda must be a finite nonnegative scalar.');
    end
    if structuredZEnabled(Options) && ...
            Options.zDim < Options.structuredZMaxModes + 1
        error('CBSRegionWGAN:StructuredZTooSmall', ...
            ['Structured-z requires zDim >= structuredZMaxModes + 1 ', ...
             'to retain a continuous residual dimension.']);
    end
end

function enabled = structuredZEnabled(Options)
    enabled = isstruct(Options) && isfield(Options,'structuredZMode') && ...
        string(Options.structuredZMode) == "categorical_mi";
end

function activeModeCount = conditionActiveModeCounts(Conditions,Options)
    n = size(Conditions,1);
    if ~structuredZEnabled(Options) || n == 0
        activeModeCount = ones(n,1);
        return;
    end
    [~,~,group] = unique(double(Conditions),'rows');
    countByCondition = accumarray(group,1);
    activeModeCount = min(Options.structuredZMaxModes, ...
        countByCondition(group));
    activeModeCount = max(1,round(double(activeModeCount(:))));
end

function activeModeCount = queryActiveModeCounts(Options,queryCount)
    if ~structuredZEnabled(Options)
        activeModeCount = ones(queryCount,1);
        return;
    end
    activeModeCount = [];
    if isfield(Options,'queryActiveModeCount') && ...
            ~isempty(Options.queryActiveModeCount)
        activeModeCount = Options.queryActiveModeCount;
    end
    activeModeCount = normalizeActiveModeCounts( ...
        activeModeCount,queryCount,Options);
end

function activeModeCount = normalizeActiveModeCounts( ...
        activeModeCount,n,Options)
    if isempty(activeModeCount)
        activeModeCount = ones(n,1);
    else
        activeModeCount = double(activeModeCount(:));
        if isscalar(activeModeCount) && n ~= 1
            activeModeCount = repmat(activeModeCount,n,1);
        end
        if numel(activeModeCount) ~= n || any(~isfinite(activeModeCount))
            error('CBSRegionWGAN:BadActiveModeCount', ...
                'Active-mode counts must be finite and match sample rows.');
        end
    end
    activeModeCount = max(1,min(Options.structuredZMaxModes, ...
        round(activeModeCount)));
end

function Target = modeTargets(labels,modeCount)
    labels = round(double(labels(:)));
    n = numel(labels);
    Target = zeros(modeCount,n,'single');
    Target(sub2ind([modeCount,n],labels',(1:n))) = 1;
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
    if isa(x,'dlarray')
        value = double(gather(extractdata(x)));
    else
        value = double(x);
    end
    if ~isscalar(value)
        value = mean(value(:));
    end
end
