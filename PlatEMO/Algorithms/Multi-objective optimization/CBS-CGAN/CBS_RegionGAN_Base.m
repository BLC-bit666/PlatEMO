classdef (Abstract) CBS_RegionGAN_Base < ALGORITHM
%CBS_REGIONGAN_BASE Shared protected runner for region GAN variants.

    methods(Access = protected)
        function Algorithm = CBS_RegionGAN_Base(varargin)
            Algorithm@ALGORITHM(varargin{:});
        end

        function runRegionGAN(Algorithm,Problem,ganKind,Config)
            ganKind = lower(strtrim(string(ganKind)));
            if ganKind == "wgan" || ganKind == "wgangp"
                ganKind = "wgan-gp";
            end
            if ganKind ~= "cgan" && ganKind ~= "wgan-gp"
                error('CBSRegionGAN:BadGANKind', ...
                    'Unsupported region GAN kind: %s.',ganKind);
            end

            trainGap   = max(1,round(double(Config.trainGap)));
            archiveGap = max(1,round(double(Config.archiveGap)));
            nGen       = max(0,round(double(Config.nGen)));
            zDim       = max(1,round(double(Config.zDim)));
            ganIter    = max(0,round(double(Config.ganIter)));
            ganMiniBatch       = max(1,round(double(Config.ganMiniBatch)));
            frontDepth         = max(1,round(double(Config.frontDepth)));
            pairNeighborRefRadius = max(0,round(double(Config.pairNeighborRefRadius)));
            refDivisor         = max(1,round(double(Config.refDivisor)));
            minBoundaryLength  = max(1,round(double(Config.minBoundaryLength)));
            queryPerCondition  = max(1,round(double(Config.queryPerCondition)));
            maxAnchorsPerRef = max(1,round(double( ...
                optionalConfig(Config,'maxAnchorsPerRef',5))));
            minGANTrainCount = max(1,round(double( ...
                optionalConfig(Config,'minGANTrainCount',32))));
            conditionDiagGap = max(0,round(double( ...
                optionalConfig(Config,'conditionDiagGap',0))));
            conditionDiagMaxConditions = max(1,round(double( ...
                optionalConfig(Config,'conditionDiagMaxConditions',8))));
            conditionDiagZSamples = max(1,round(double( ...
                optionalConfig(Config,'conditionDiagZSamples',8))));
            conditionDiagAllWZPerRef = max(1,round(double( ...
                optionalConfig(Config,'conditionDiagAllWZPerRef',2))));
            ganLrD = double(Config.ganLrD);
            ganLrG = double(Config.ganLrG);
            gpLambda = double(Config.gpLambda);
            nCritic = double(Config.nCritic);
            if ganKind == "wgan-gp"
                gpLambda = max(0,double(gpLambda));
                nCritic = max(1,round(double(nCritic)));
            end

            nRef = max(2,round(Problem.N/refDivisor));
            [W,~] = UniformPoint(nRef,Problem.M);
            MemOptions = struct( ...
                'frontDepth',frontDepth, ...
                'pairNeighborRefRadius',pairNeighborRefRadius, ...
                'minBoundaryLength',minBoundaryLength, ...
                'maxAnchorsPerRef',maxAnchorsPerRef, ...
                'bmemBandMode',regionBMemBandModeFromControl( ...
                    optionalConfig(Config,'bmemBandMode',"current")), ...
                'bandMaxAnchorsPerRef',regionBandMaxAnchorsPerRefFromControl( ...
                    optionalConfig(Config,'bandMaxAnchorsPerRef',[])), ...
                'minGANTrainCount',minGANTrainCount, ...
                'prevBMemMode',regionPrevBMemModeFromControl( ...
                    optionalConfig(Config,'prevBMemMode',"current_only")));
            DatasetOptions = struct();
            GANOptions = regionGANOptions(ganKind,zDim,ganIter,ganMiniBatch, ...
                ganLrD,ganLrG,gpLambda,nCritic);
            GANOptions = applyRegionGANConfigOptions(GANOptions,Config);
            GANOptions = applyRegionGANExperimentOptions(GANOptions);
            GANOptions.minTrainCount = minGANTrainCount;
            ConditionDiagOptions = struct( ...
                'maxConditions',conditionDiagMaxConditions, ...
                'zSamples',conditionDiagZSamples, ...
                'allWZPerRef',conditionDiagAllWZPerRef, ...
                'neighborRadius',pairNeighborRefRadius);
            QueryMode = regionQueryModeFromControl( ...
                optionalConfig(Config,'queryMode',"boundary_populated"));

            Population1 = Problem.Initialization();
            Population2 = Problem.Initialization();
            DatasetOptions.conditionScale = ...
                initRegionConditionScale([Population1,Population2],Problem);
            Fitness1 = CalFitness_CBS(Population1.objs,Population1.cons);
            Fitness2 = CalFitness_CBS(Population2.objs);

            BMem = [];
            GAN  = [];
            gen  = 0;
            EmptyDiag = emptyRegionDiag(ganKind,zDim,gpLambda,nCritic);
            [lastMetric,historyMetric,cloudMetric] = ...
                RunRegionGAN_RC('metricnames',ganKind);
            Algorithm.metric.(lastMetric) = EmptyDiag;
            Algorithm.metric.(historyMetric) = EmptyDiag([]);
            Algorithm.metric.(cloudMetric) = [];
            Algorithm.metric.region_gan_last = EmptyDiag;
            Algorithm.metric.region_gan_history = EmptyDiag([]);
            StageObserver = initRegionStageObserver(Algorithm,Problem);
            Algorithm.metric.region_gan_stage_snapshots = ...
                StageObserver.snapshots;
            if ganKind == "cgan"
                Algorithm.metric.region_cgan_stage_snapshots = ...
                    StageObserver.snapshots;
            elseif ganKind == "wgan-gp"
                Algorithm.metric.region_wgan_gp_stage_snapshots = ...
                    StageObserver.snapshots;
            end

            while Algorithm.NotTerminated(Population1)
                gen = gen + 1;
                MatingPool1 = TournamentSelection(2,2*Problem.N,Fitness1);
                MatingPool2 = TournamentSelection(2,2*Problem.N,Fitness2);
                Offspring1  = OperatorDE(Problem,Population1, ...
                    Population1(MatingPool1(1:end/2)), ...
                    Population1(MatingPool1(end/2+1:end)));
                Offspring2  = OperatorDE(Problem,Population2, ...
                    Population2(MatingPool2(1:end/2)), ...
                    Population2(MatingPool2(end/2+1:end)));

                Diag = emptyRegionDiag(ganKind,zDim,gpLambda,nCritic);
                Diag.generation = gen;
                Diag.query_mode = QueryMode;
                EventGANOptions = RunRegionGAN_RC('resolveganoptions', ...
                    GANOptions,Problem.FE,regionProblemMaxFE(Problem));
                Diag = copyGANOptionInfo(Diag,EventGANOptions);
                DatasetInfo = emptyRegionDatasetInfo(Problem.M);
                if mod(gen,archiveGap) == 0
                    [BMem,MemDiag] = UpdateBoundaryMemory_RC(BMem, ...
                        Population1,Offspring1,Population2,Offspring2, ...
                        W,MemOptions);
                    Diag = copyMemDiag(Diag,MemDiag);
                    Algorithm.metric.(cloudMetric) = updateDensestCloud( ...
                        Algorithm.metric.(cloudMetric),BMem,Problem);
                end

                OffspringG = Offspring1([]);
                if mod(gen,trainGap) == 0 && ~isempty(BMem) && nGen > 0
                        [TrainX,TrainC,QueryC,BMem,DatasetInfo] = ...
                            BuildBoundaryDataset_RC( ...
                        BMem,[Population1,Offspring1,Population2,Offspring2], ...
                        W,Problem,DatasetOptions);
                    Diag.train_count = size(TrainX,1);
                    Diag.query_count = regionQueryPoolCount( ...
                        QueryMode,QueryC,W);
                    if size(TrainX,1) >= max(minBoundaryLength, ...
                            minGANTrainCount) && ~isempty(QueryC)
                        [SampleC,QueryAllocation,SampleRefs,DiagQueryC, ...
                            DiagQueryRefs] = makeRegionQuerySamples( ...
                            QueryMode,QueryC,DatasetInfo,W, ...
                            queryPerCondition,nGen);
                        Diag.query_sample_count = size(SampleC,1);
                        Diag.query_per_region_min = minNonempty(QueryAllocation);
                        Diag.query_per_region_max = maxNonempty(QueryAllocation);
                        Diag.query_unique_ref_count = numel(unique( ...
                            SampleRefs(isfinite(SampleRefs) & ...
                            SampleRefs > 0)));
                        [GAN,RawDec] = RunRegionGAN_RC('trainandsample', ...
                            ganKind,GAN, ...
                            TrainX,TrainC,SampleC,1, ...
                            Problem,EventGANOptions);
                        Diag = copyGANTrainInfo(Diag,GAN);
                        if size(RawDec,1) > nGen
                            RawDec = RawDec(1:nGen,:);
                        end
                        if numel(SampleRefs) > size(RawDec,1)
                            SampleRefs = SampleRefs(1:size(RawDec,1));
                        end
                        Diag.raw_generated_count = size(RawDec,1);
                        if ~isempty(RawDec)
                            OffspringG = Problem.Evaluation(RawDec);
                            CVG = sum(max(0,OffspringG.cons),2);
                            Diag.feasible_generated_count = sum(CVG <= 0);
                            Diag.feasible_rate = mean(double(CVG <= 0));
                            BoundaryDiag = regionGeneratedBoundaryDiagnostics( ...
                                OffspringG.objs,OffspringG.cons,RawDec, ...
                                SampleRefs,BMem,W,Problem, ...
                                struct('seed',786433 + gen));
                            Diag = copyGeneratedBoundaryDiag(Diag, ...
                                BoundaryDiag);
                        end
                        if conditionDiagGap > 0 && ...
                                mod(gen,conditionDiagGap) == 0
                            RunDiagOptions = ConditionDiagOptions;
                            RunDiagOptions.seed = 104729 + gen;
                            CondDiag = ConditionControlDiagnostics_RC( ...
                                ganKind,GAN,DiagQueryC,DiagQueryRefs, ...
                                SampleC,SampleRefs,RawDec,BMem,W,Problem, ...
                                EventGANOptions,RunDiagOptions);
                            Diag = copyConditionControlDiag(Diag,CondDiag);
                        end
                    end
                end

                Algorithm.metric.(lastMetric) = Diag;
                Algorithm.metric.(historyMetric)(end+1) = Diag;
                Algorithm.metric.region_gan_last = Diag;
                Algorithm.metric.region_gan_history(end+1) = Diag;
                CurrentSamples = [Population1,Offspring1,Population2, ...
                    Offspring2];
                [StageObserver,captured] = maybeCaptureRegionStages( ...
                    StageObserver,Algorithm,Problem,gen,CurrentSamples, ...
                    DatasetInfo,OffspringG,Diag);
                if captured
                    Algorithm.metric.region_gan_stage_snapshots = ...
                        StageObserver.snapshots;
                    if ganKind == "cgan"
                        Algorithm.metric.region_cgan_stage_snapshots = ...
                            StageObserver.snapshots;
                    elseif ganKind == "wgan-gp"
                        Algorithm.metric.region_wgan_gp_stage_snapshots = ...
                            StageObserver.snapshots;
                    end
                end

                Union = [Population1,Population2,Offspring1,Offspring2, ...
                    OffspringG];
                [Population1,Fitness1] = EnvironmentalSelection_CBS( ...
                    Union,Problem.N,true);
                [Population2,Fitness2] = EnvironmentalSelection_CBS( ...
                    Union,Problem.N,false);
            end
        end
    end
end

function Options = regionGANOptions(ganKind,zDim,iter,miniBatch,lrD,lrG, ...
        gpLambda,nCritic)
    Options = struct( ...
        'zDim',zDim, ...
        'iter',iter, ...
        'miniBatch',miniBatch, ...
        'lrD',double(lrD), ...
        'lrG',double(lrG), ...
        'trainZMode',"random", ...
        'sampleZMode',"random", ...
        'sigma',1.0, ...
        'trainSigma',[], ...
        'sampleSigma',[], ...
        'generatorHidden',[32 32]);
    switch ganKind
        case "cgan"
            Options.dSteps = 1;
            Options.gSteps = 1;
            Options.advWeight = 1.0;
            Options.trainMode = "iter";
            Options.realLabel = 0.9;
            Options.discriminatorHidden = [32 32];
        case "wgan-gp"
            Options.gpLambda = gpLambda;
            Options.nCritic = nCritic;
            Options.criticHidden = [32 32];
    end
end

function Options = applyRegionGANExperimentOptions(Options)
    Control = regionGANExperimentControl();
    if isempty(Control) || ~isstruct(Control)
        return;
    end
    if isfield(Control,'sampleZMode') && ~isempty(Control.sampleZMode)
        Options.sampleZMode = lower(strtrim(string(Control.sampleZMode)));
    end
    if isfield(Control,'trainZMode') && ~isempty(Control.trainZMode)
        Options.trainZMode = lower(strtrim(string(Control.trainZMode)));
    end
    if isfield(Control,'sigma') && ~isempty(Control.sigma)
        Options.sigma = double(Control.sigma);
    end
    if isfield(Control,'trainSigma') && ~isempty(Control.trainSigma)
        Options.trainSigma = double(Control.trainSigma);
    end
    if isfield(Control,'sampleSigma') && ~isempty(Control.sampleSigma)
        Options.sampleSigma = double(Control.sampleSigma);
    end
    if isfield(Control,'prescreenMultiplier') && ...
            ~isempty(Control.prescreenMultiplier)
        Options.prescreenMultiplier = max(1,round(double( ...
            Control.prescreenMultiplier)));
    end
    if isfield(Control,'ganIterSchedule') && ...
            ~isempty(Control.ganIterSchedule)
        Options.ganIterSchedule = lower(strtrim(string( ...
            Control.ganIterSchedule)));
    end
    if isfield(Control,'ganIterStart') && ~isempty(Control.ganIterStart)
        Options.ganIterStart = double(Control.ganIterStart);
    end
    if isfield(Control,'ganIterEnd') && ~isempty(Control.ganIterEnd)
        Options.ganIterEnd = double(Control.ganIterEnd);
    end
end

function Options = applyRegionGANConfigOptions(Options,Config)
    if ~isstruct(Config)
        return;
    end
    if isfield(Config,'sampleZMode') && ~isempty(Config.sampleZMode)
        Options.sampleZMode = lower(strtrim(string(Config.sampleZMode)));
    end
    if isfield(Config,'trainZMode') && ~isempty(Config.trainZMode)
        Options.trainZMode = lower(strtrim(string(Config.trainZMode)));
    end
    if isfield(Config,'sigma') && ~isempty(Config.sigma)
        Options.sigma = double(Config.sigma);
    end
    if isfield(Config,'trainSigma') && ~isempty(Config.trainSigma)
        Options.trainSigma = double(Config.trainSigma);
    end
    if isfield(Config,'sampleSigma') && ~isempty(Config.sampleSigma)
        Options.sampleSigma = double(Config.sampleSigma);
    end
    if isfield(Config,'prescreenMultiplier') && ...
            ~isempty(Config.prescreenMultiplier)
        Options.prescreenMultiplier = max(1,round(double( ...
            Config.prescreenMultiplier)));
    end
end

function value = optionalConfig(Config,name,defaultValue)
    if isstruct(Config) && isfield(Config,name) && ~isempty(Config.(name))
        value = Config.(name);
    else
        value = defaultValue;
    end
end

function Diag = copyGANTrainInfo(Diag,GAN)
    fields = {'last_critic_loss','last_generator_loss', ...
        'last_discriminator_loss','last_gradient_penalty', ...
        'last_score_real','last_score_fake','last_d_real_acc', ...
        'last_d_fake_acc','last_d_bal_acc','last_g_fool_rate', ...
        'last_score_random','last_random_as_fake_rate'};
    for i = 1 : numel(fields)
        if isstruct(GAN) && isfield(GAN,fields{i})
            Diag.(fields{i}) = double(GAN.(fields{i}));
        end
    end
    if isstruct(GAN) && isfield(GAN,'train_history')
        Diag.gan_train_history = GAN.train_history;
    end
    if isstruct(GAN) && isfield(GAN,'last_sample_info') && ...
            isstruct(GAN.last_sample_info)
        sampleFields = {'prescreen_multiplier', ...
            'prescreen_candidate_count','prescreen_selected_count', ...
            'prescreen_score_min','prescreen_score_max', ...
            'prescreen_score_mean'};
        for i = 1 : numel(sampleFields)
            name = sampleFields{i};
            if isfield(GAN.last_sample_info,name)
                Diag.(name) = double(GAN.last_sample_info.(name));
            end
        end
    end
end

function Diag = copyGANOptionInfo(Diag,Options)
    if ~isstruct(Options)
        return;
    end
    Diag.zDim = getOptionNumeric(Options,'zDim',Diag.zDim);
    Diag.gan_iter_used = getOptionNumeric(Options,'iter', ...
        Diag.gan_iter_used);
    Diag.gan_iter_schedule = getOptionString(Options, ...
        'ganIterSchedule',Diag.gan_iter_schedule);
    Diag.gan_iter_start = getOptionNumeric(Options,'ganIterStart', ...
        Diag.gan_iter_start);
    Diag.gan_iter_end = getOptionNumeric(Options,'ganIterEnd', ...
        Diag.gan_iter_end);
    Diag.sample_z_mode = getOptionString(Options,'sampleZMode', ...
        Diag.sample_z_mode);
    Diag.train_z_mode = getOptionString(Options,'trainZMode', ...
        Diag.train_z_mode);
    Diag.train_z_sigma = getOptionNumeric(Options,'trainSigma', ...
        getOptionNumeric(Options,'sigma',Diag.train_z_sigma));
    Diag.sample_z_sigma = getOptionNumeric(Options,'sampleSigma', ...
        getOptionNumeric(Options,'sigma',Diag.sample_z_sigma));
    Diag.prescreen_multiplier = getOptionNumeric(Options, ...
        'prescreenMultiplier',Diag.prescreen_multiplier);
end

function value = getOptionNumeric(S,name,defaultValue)
    value = defaultValue;
    if isstruct(S) && isfield(S,name) && ~isempty(S.(name))
        value = double(S.(name));
    end
    if isempty(value) || ~isfinite(value(1))
        value = defaultValue;
    else
        value = value(1);
    end
end

function value = getOptionString(S,name,defaultValue)
    value = string(defaultValue);
    if isstruct(S) && isfield(S,name) && ~isempty(S.(name))
        value = string(S.(name));
    end
    value = lower(strtrim(value));
end

function Refs = expandQueryRefs(DatasetInfo,QueryAllocation)
    Refs = zeros(0,1);
    if ~isstruct(DatasetInfo) || ~isfield(DatasetInfo,'queryRegions') || ...
            isempty(DatasetInfo.queryRegions) || isempty(QueryAllocation)
        return;
    end
    queryRefs = round(double(DatasetInfo.queryRegions(:)));
    QueryAllocation = round(double(QueryAllocation(:)));
    n = min(numel(queryRefs),numel(QueryAllocation));
    Refs = zeros(sum(max(0,QueryAllocation(1:n))),1);
    row = 0;
    for i = 1 : n
        c = max(0,QueryAllocation(i));
        if c <= 0
            continue;
        end
        Refs(row+1:row+c) = queryRefs(i);
        row = row + c;
    end
    Refs = Refs(1:row);
end

function [SampleC,Counts,SampleRefs,DiagQueryC,DiagQueryRefs] = ...
        makeRegionQuerySamples(QueryMode,QueryC,DatasetInfo,W, ...
        queryPerCondition,nGen)
    QueryMode = normalizeRegionQueryMode(QueryMode);
    switch QueryMode
        case "random_all_w"
            nRef = size(W,1);
            totalBudget = max(0,round(double(nGen)));
            if isempty(W) || nRef == 0 || totalBudget <= 0
                SampleC = zeros(0,size(W,2));
                Counts = zeros(nRef,1);
                SampleRefs = zeros(0,1);
                DiagQueryC = zeros(0,size(W,2));
                DiagQueryRefs = zeros(0,1);
                return;
            end
            SampleRefs = randi(nRef,totalBudget,1);
            SampleC = double(W(SampleRefs,:));
            Counts = accumarray(SampleRefs,1,[nRef 1],@sum,0);
            DiagQueryC = double(W);
            DiagQueryRefs = (1:nRef)';
        otherwise
            [SampleC,Counts] = RunRegionGAN_RC( ...
                'allocatequery',QueryC,queryPerCondition,nGen);
            SampleRefs = expandQueryRefs(DatasetInfo,Counts);
            DiagQueryC = QueryC;
            if isstruct(DatasetInfo) && isfield(DatasetInfo,'queryRegions')
                DiagQueryRefs = DatasetInfo.queryRegions(:);
            else
                DiagQueryRefs = zeros(size(QueryC,1),1);
            end
    end
end

function count = regionQueryPoolCount(QueryMode,QueryC,W)
    QueryMode = normalizeRegionQueryMode(QueryMode);
    if QueryMode == "random_all_w"
        count = size(W,1);
    else
        count = size(QueryC,1);
    end
end

function mode = regionQueryModeFromControl(defaultMode)
    mode = string(defaultMode);
    Control = regionGANExperimentControl();
    if ~isempty(Control) && isfield(Control,'queryMode') && ...
            ~isempty(Control.queryMode)
        mode = string(Control.queryMode);
    end
    mode = normalizeRegionQueryMode(mode);
end

function mode = regionPrevBMemModeFromControl(defaultMode)
    mode = string(defaultMode);
    Control = regionGANExperimentControl();
    if ~isempty(Control) && isfield(Control,'prevBMemMode') && ...
            ~isempty(Control.prevBMemMode)
        mode = string(Control.prevBMemMode);
    end
    mode = lower(strtrim(mode));
    switch mode
        case {"current","current_only","none","off"}
            mode = "current_only";
        case {"prev1_fair_union","fair_union","previous_fair_union"}
            mode = "prev1_fair_union";
        otherwise
            error('CBSRegionGAN:BadPrevBMemMode', ...
                'Unsupported previous BMem mode: %s.',mode);
    end
end

function mode = regionBMemBandModeFromControl(defaultMode)
    mode = string(defaultMode);
    Control = regionGANExperimentControl();
    if ~isempty(Control) && isfield(Control,'bmemBandMode') && ...
            ~isempty(Control.bmemBandMode)
        mode = string(Control.bmemBandMode);
    end
    mode = lower(strtrim(mode));
    switch mode
        case {"current","current_cloud","mixed_current","none","off"}
            mode = "current";
        case {"pfseed_band","pf_seed_band","local_pfseed","local_pf_seed"}
            mode = "pfseed_band";
        case {"pfseed_band_mincover","pf_seed_band_mincover", ...
                "local_pfseed_mincover","local_pf_seed_mincover"}
            mode = "pfseed_band_mincover";
        case {"pfseed_multiband_safety","pf_seed_multiband_safety", ...
                "multiband_safety","multi_band_safety"}
            mode = "pfseed_multiband_safety";
        otherwise
            error('CBSRegionGAN:BadBMemBandMode', ...
                'Unsupported BMem band mode: %s.',mode);
    end
end

function value = regionBandMaxAnchorsPerRefFromControl(defaultValue)
    value = defaultValue;
    Control = regionGANExperimentControl();
    if ~isempty(Control) && isfield(Control,'bandMaxAnchorsPerRef') && ...
            ~isempty(Control.bandMaxAnchorsPerRef)
        value = Control.bandMaxAnchorsPerRef;
    end
    if isempty(value)
        return;
    end
    value = max(1,round(double(value)));
end

function mode = normalizeRegionQueryMode(mode)
    mode = lower(strtrim(string(mode)));
    switch mode
        case {"boundary","boundary_populated","populated"}
            mode = "boundary_populated";
        case {"random","random_all_w","all_w_random"}
            mode = "random_all_w";
        otherwise
            error('CBSRegionGAN:BadQueryMode', ...
                'Unsupported region QueryC mode: %s.',mode);
    end
end

function Control = regionGANExperimentControl()
    if isappdata(0,'CBS_RegionGAN_ExperimentControl')
        Control = getappdata(0,'CBS_RegionGAN_ExperimentControl');
        if ~isstruct(Control)
            Control = [];
        end
    else
        Control = [];
    end
end

function Diag = copyConditionControlDiag(Diag,CondDiag)
    fields = fieldnames(CondDiag);
    for i = 1 : numel(fields)
        value = CondDiag.(fields{i});
        if isnumeric(value) || islogical(value)
            Diag.(fields{i}) = double(value);
        end
    end
end

function Diag = copyGeneratedBoundaryDiag(Diag,BoundaryDiag)
    fields = fieldnames(BoundaryDiag);
    for i = 1 : numel(fields)
        value = BoundaryDiag.(fields{i});
        if isnumeric(value) || islogical(value)
            Diag.(fields{i}) = double(value);
        end
    end
end

function Cloud = updateDensestCloud(Cloud,BMem,Problem)
    if isempty(BMem) || ~isfield(BMem,'x_b') || isempty(BMem.x_b)
        return;
    end
    if isempty(Cloud) || size(BMem.x_b,1) > size(Cloud.x_b,1)
        Cloud = struct('x_b',BMem.x_b,'ref',BMem.ref, ...
            'lower',Problem.lower,'upper',Problem.upper);
    end
end

function Scale = initRegionConditionScale(Population,Problem)
    if isempty(Population)
        Obj = zeros(0,Problem.M);
    else
        Obj = double(Population.objs);
    end
    Obj = Obj(all(isfinite(Obj),2),:);
    if isempty(Obj)
        objMin = zeros(1,Problem.M);
        objSpan = ones(1,Problem.M);
    else
        objMin = min(Obj,[],1);
        objSpan = max(Obj,[],1) - objMin;
        objSpan(objSpan <= eps) = 1;
    end
    Scale = struct('objMin',objMin,'objSpan',objSpan);
end

function Info = emptyRegionDatasetInfo(M)
    Info = struct( ...
        'objMin',zeros(1,M), ...
        'objSpan',ones(1,M), ...
        'trainObjs',zeros(0,M), ...
        'trainConditions',zeros(0,0), ...
        'trainRef',zeros(0,1), ...
        'queryConditions',zeros(0,0), ...
        'queryRegions',zeros(0,1), ...
        'queryObjs',zeros(0,M));
end

function Metrics = regionGeneratedBoundaryDiagnostics( ...
        GeneratedObj,GeneratedCon,~,TargetRefs,BMem,~, ...
        Problem,Options)
    Metrics = emptyRegionGeneratedBoundaryMetrics();
    if isempty(GeneratedObj) || isempty(BMem) || ...
            ~isfield(BMem,'y_b') || isempty(BMem.y_b)
        return;
    end
    if nargin < 8 || isempty(Options)
        Options = struct();
    end
    n = size(GeneratedObj,1);
    Feasible = regionFeasibleMask(GeneratedCon,n);

    rngState = rng;
    cleanupRng = onCleanup(@()rng(rngState));
    if isstruct(Options) && isfield(Options,'seed') && ...
            ~isempty(Options.seed) && isfinite(double(Options.seed))
        rng(round(double(Options.seed)),'twister');
    end
    RandomDec = randomRegionDecisions(Problem,n);
    [RandomObj,RandomCon] = EvaluateDecisions_CBS(Problem,RandomDec);

    Scale = regionObjectiveScale(regionObjectiveBase( ...
        GeneratedObj,RandomObj,BMem),Problem.M);
    [DistB,DistTarget,SegDist,GapRatio] = regionBoundaryDistances( ...
        GeneratedObj,TargetRefs,BMem,Scale);
    [~,~,~,RandomGapRatio] = regionBoundaryDistances( ...
        RandomObj,zeros(size(RandomObj,1),1),BMem,Scale);

    Metrics.dist_to_bmem50 = percentileFiniteLocal(DistB,50);
    Metrics.dist_to_bmem90 = percentileFiniteLocal(DistB,90);
    Metrics.dist_to_target_pm2_bmem50 = percentileFiniteLocal( ...
        DistTarget,50);
    Metrics.dist_to_target_pm2_bmem90 = percentileFiniteLocal( ...
        DistTarget,90);
    Metrics.segment_dist50 = percentileFiniteLocal(SegDist,50);
    Metrics.segment_dist90 = percentileFiniteLocal(SegDist,90);
    Metrics.gap_ratio50 = percentileFiniteLocal(GapRatio,50);
    Metrics.gap_ratio90 = percentileFiniteLocal(GapRatio,90);
    Metrics.near_boundary_rate_gap1 = finiteMaskRate(GapRatio <= 1, ...
        GapRatio);
    Metrics.near_boundary_feasible_rate_gap1 = finiteMaskRate( ...
        GapRatio <= 1 & Feasible,GapRatio);
    Metrics.random_gap_ratio50 = percentileFiniteLocal(RandomGapRatio,50);
    Metrics.random_gap_ratio90 = percentileFiniteLocal(RandomGapRatio,90);
    Metrics.random_near_boundary_rate_gap1 = finiteMaskRate( ...
        RandomGapRatio <= 1,RandomGapRatio);
    Metrics.better_than_random_gap_rate = pairedImprovementRate( ...
        GapRatio,RandomGapRatio);
    if isempty(RandomCon)
        Metrics.random_feasible_rate = 1;
    else
        Metrics.random_feasible_rate = mean(double( ...
            sum(max(0,RandomCon),2) <= 0),'omitnan');
    end
end

function Metrics = emptyRegionGeneratedBoundaryMetrics()
    Metrics = struct( ...
        'dist_to_bmem50',NaN, ...
        'dist_to_bmem90',NaN, ...
        'dist_to_target_pm2_bmem50',NaN, ...
        'dist_to_target_pm2_bmem90',NaN, ...
        'segment_dist50',NaN, ...
        'segment_dist90',NaN, ...
        'gap_ratio50',NaN, ...
        'gap_ratio90',NaN, ...
        'near_boundary_rate_gap1',NaN, ...
        'near_boundary_feasible_rate_gap1',NaN, ...
        'random_gap_ratio50',NaN, ...
        'random_gap_ratio90',NaN, ...
        'random_near_boundary_rate_gap1',NaN, ...
        'better_than_random_gap_rate',NaN, ...
        'random_feasible_rate',NaN);
end

function Feasible = regionFeasibleMask(Con,n)
    if isempty(Con)
        Feasible = true(n,1);
    else
        Feasible = sum(max(0,Con),2) <= 0;
        Feasible = Feasible(1:min(n,numel(Feasible)));
        if numel(Feasible) < n
            Feasible(end+1:n,1) = false;
        end
    end
end

function Dec = randomRegionDecisions(Problem,n)
    n = max(0,round(double(n)));
    if n == 0
        Dec = zeros(0,Problem.D);
        return;
    end
    lower = double(Problem.lower(:)');
    upper = double(Problem.upper(:)');
    Dec = repmat(lower,n,1) + rand(n,Problem.D).* ...
        repmat(upper - lower,n,1);
end

function Base = regionObjectiveBase(GeneratedObj,RandomObj,BMem)
    Base = [GeneratedObj;RandomObj];
    names = {'y_b','y_f','y_i'};
    for i = 1 : numel(names)
        if isfield(BMem,names{i}) && ~isempty(BMem.(names{i}))
            Base = [Base;BMem.(names{i})]; %#ok<AGROW>
        end
    end
    Base = Base(all(isfinite(Base),2),:);
end

function Scale = regionObjectiveScale(Base,M)
    if isempty(Base)
        Scale = struct('min',zeros(1,M),'span',ones(1,M));
        return;
    end
    minV = min(Base,[],1);
    spanV = max(Base,[],1) - minV;
    spanV(spanV <= eps) = 1;
    Scale = struct('min',minV,'span',spanV);
end

function [DistB,DistTarget,SegDist,GapRatio] = regionBoundaryDistances( ...
        Obj,TargetRefs,BMem,Scale)
    n = size(Obj,1);
    DistB = NaN(n,1);
    DistTarget = NaN(n,1);
    SegDist = NaN(n,1);
    GapRatio = NaN(n,1);
    if n == 0 || isempty(BMem) || ~isfield(BMem,'y_b') || ...
            isempty(BMem.y_b)
        return;
    end

    Y = normalizeRegionObj(Obj,Scale);
    Yb = normalizeRegionObj(BMem.y_b,Scale);
    D = pointDistanceLocal(Y,Yb);
    DistB = min(D,[],2);

    if ~isempty(TargetRefs) && isfield(BMem,'ref') && ~isempty(BMem.ref)
        TargetRefs = round(double(TargetRefs(:)));
        m = min(n,numel(TargetRefs));
        refs = round(double(BMem.ref(:)));
        for i = 1 : m
            if ~isfinite(TargetRefs(i)) || TargetRefs(i) <= 0
                continue;
            end
            keep = abs(refs - TargetRefs(i)) <= 2;
            if any(keep)
                DistTarget(i) = min(pointDistanceLocal(Y(i,:),Yb(keep,:)), ...
                    [],2);
            end
        end
    end

    if ~isfield(BMem,'y_f') || ~isfield(BMem,'y_i') || ...
            isempty(BMem.y_f) || isempty(BMem.y_i)
        return;
    end
    Yf = normalizeRegionObj(BMem.y_f,Scale);
    Yi = normalizeRegionObj(BMem.y_i,Scale);
    segmentLength = sqrt(sum((Yi - Yf).^2,2));
    for i = 1 : n
        d = pointToSegmentsDistanceLocal(Y(i,:),Yf,Yi);
        [SegDist(i),idx] = min(d,[],1);
        if ~isempty(idx) && idx >= 1 && idx <= numel(segmentLength) && ...
                segmentLength(idx) > eps
            GapRatio(i) = SegDist(i)/segmentLength(idx);
        end
    end
end

function Xn = normalizeRegionObj(X,Scale)
    Xn = (X - Scale.min)./Scale.span;
    Xn(~isfinite(Xn)) = 0;
end

function D = pointDistanceLocal(A,B)
    if isempty(A) || isempty(B)
        D = zeros(size(A,1),size(B,1));
        return;
    end
    D2 = max(sum(A.^2,2) + sum(B.^2,2)' - 2*(A*B'),0);
    D = sqrt(D2);
end

function D = pointToSegmentsDistanceLocal(P,A,B)
    if isempty(A) || isempty(B)
        D = zeros(0,1);
        return;
    end
    AB = B - A;
    denom = sum(AB.^2,2);
    T = sum((P - A).*AB,2)./max(denom,eps);
    T = max(0,min(1,T));
    Projection = A + T.*AB;
    D = sqrt(sum((P - Projection).^2,2));
    degenerate = denom <= eps;
    if any(degenerate)
        D(degenerate) = sqrt(sum((P - A(degenerate,:)).^2,2));
    end
end

function q = percentileFiniteLocal(X,p)
    X = X(isfinite(X));
    if isempty(X)
        q = NaN;
    else
        q = prctile(X,p);
    end
end

function rate = finiteMaskRate(mask,values)
    values = values(:);
    mask = logical(mask(:));
    n = min(numel(mask),numel(values));
    if n == 0
        rate = NaN;
        return;
    end
    mask = mask(1:n);
    values = values(1:n);
    valid = isfinite(values);
    if ~any(valid)
        rate = NaN;
    else
        rate = mean(double(mask(valid)),'omitnan');
    end
end

function rate = pairedImprovementRate(GapRatio,RandomGapRatio)
    n = min(numel(GapRatio),numel(RandomGapRatio));
    if n == 0
        rate = NaN;
        return;
    end
    GapRatio = GapRatio(1:n);
    RandomGapRatio = RandomGapRatio(1:n);
    valid = isfinite(GapRatio) & isfinite(RandomGapRatio);
    if ~any(valid)
        rate = NaN;
    else
        rate = mean(double(GapRatio(valid) <= RandomGapRatio(valid)), ...
            'omitnan');
    end
end

function Diag = emptyRegionDiag(ganKind,zDim,gpLambda,nCritic)
    Diag = struct( ...
        'generation',0, ...
        'gan_type',string(ganKind), ...
        'zDim',double(zDim), ...
        'gan_iter_used',NaN, ...
        'gan_iter_schedule',"fixed", ...
        'gan_iter_start',NaN, ...
        'gan_iter_end',NaN, ...
        'gpLambda',double(gpLambda), ...
        'nCritic',double(nCritic), ...
        'sample_z_mode',"random", ...
        'train_z_mode',"random", ...
        'train_z_sigma',1.0, ...
        'sample_z_sigma',1.0, ...
        'prescreen_multiplier',1, ...
        'prescreen_candidate_count',0, ...
        'prescreen_selected_count',0, ...
        'prescreen_score_min',NaN, ...
        'prescreen_score_max',NaN, ...
        'prescreen_score_mean',NaN, ...
        'bmem_count',0, ...
        'region_count',0, ...
        'points_per_region_median',0, ...
        'points_per_region_max',0, ...
        'prev_bmem_candidate_count',0, ...
        'prev_bmem_survivor_count',0, ...
        'median_gap',NaN, ...
        'max_gap',NaN, ...
        'train_count',0, ...
        'query_mode',"boundary_populated", ...
        'query_count',0, ...
        'query_sample_count',0, ...
        'query_unique_ref_count',0, ...
        'query_per_region_min',0, ...
        'query_per_region_max',0, ...
        'raw_generated_count',0, ...
        'feasible_generated_count',0, ...
        'feasible_rate',NaN, ...
        'dist_to_bmem50',NaN, ...
        'dist_to_bmem90',NaN, ...
        'dist_to_target_pm2_bmem50',NaN, ...
        'dist_to_target_pm2_bmem90',NaN, ...
        'segment_dist50',NaN, ...
        'segment_dist90',NaN, ...
        'gap_ratio50',NaN, ...
        'gap_ratio90',NaN, ...
        'near_boundary_rate_gap1',NaN, ...
        'near_boundary_feasible_rate_gap1',NaN, ...
        'random_gap_ratio50',NaN, ...
        'random_gap_ratio90',NaN, ...
        'random_near_boundary_rate_gap1',NaN, ...
        'random_feasible_rate',NaN, ...
        'better_than_random_gap_rate',NaN, ...
        'last_critic_loss',NaN, ...
        'last_discriminator_loss',NaN, ...
        'last_generator_loss',NaN, ...
        'last_gradient_penalty',NaN, ...
        'last_score_real',NaN, ...
        'last_score_fake',NaN, ...
        'last_score_random',NaN, ...
        'last_d_real_acc',NaN, ...
        'last_d_fake_acc',NaN, ...
        'last_d_bal_acc',NaN, ...
        'last_g_fool_rate',NaN, ...
        'last_random_as_fake_rate',NaN, ...
        'condition_diag_condition_count',0, ...
        'condition_diag_z_count',0, ...
        'same_z_diff_c_dec_median',NaN, ...
        'same_z_diff_c_obj_median',NaN, ...
        'same_z_diff_c_ref_unique_rate',NaN, ...
        'same_c_diff_z_dec_median',NaN, ...
        'same_c_diff_z_obj_median',NaN, ...
        'same_c_diff_z_ref_leak_rate',NaN, ...
        'same_c_diff_z_collapse_rate',NaN, ...
        'condition_effect_ratio_dec',NaN, ...
        'condition_effect_ratio_obj',NaN, ...
        'query_generated_count',0, ...
        'query_exact_ref_match_rate',NaN, ...
        'query_neighbor_ref_match_rate',NaN, ...
        'query_target_ref_rank_median',NaN, ...
        'query_target_ref_rank_mean',NaN, ...
        'query_feasible_rate_probe',NaN, ...
        'query_shuffled_exact_match_rate',NaN, ...
        'all_w_condition_count',0, ...
        'all_w_z_per_ref',0, ...
        'all_w_query_generated_count',0, ...
        'all_w_exact_ref_match_rate',NaN, ...
        'all_w_pm2_ref_match_rate',NaN, ...
        'all_w_shuffled_pm2_ref_match_rate',NaN, ...
        'all_w_seen_count',0, ...
        'all_w_unseen_count',0, ...
        'all_w_seen_pm2_ref_match_rate',NaN, ...
        'all_w_unseen_pm2_ref_match_rate',NaN, ...
        'all_w_target_ref_rank_median',NaN, ...
        'all_w_target_ref_rank_mean',NaN, ...
        'all_w_feasible_rate_probe',NaN, ...
        'gan_train_history',struct([]));
end

function value = minNonempty(X)
    if isempty(X)
        value = 0;
    else
        value = min(X);
    end
end

function value = maxNonempty(X)
    if isempty(X)
        value = 0;
    else
        value = max(X);
    end
end

function Diag = copyMemDiag(Diag,MemDiag)
    names = {'bmem_count','region_count','points_per_region_median', ...
        'points_per_region_max','median_gap','max_gap', ...
        'prev_bmem_candidate_count','prev_bmem_survivor_count'};
    for i = 1 : numel(names)
        if isfield(MemDiag,names{i})
            Diag.(names{i}) = MemDiag.(names{i});
        end
    end
end

function Observer = initRegionStageObserver(Algorithm,Problem)
    EmptySnapshot = emptyRegionStageSnapshot(Problem.M);
    Observer = struct( ...
        'run',doubleOrNaN(Algorithm.run), ...
        'targets',zeros(1,0), ...
        'next',1, ...
        'snapshots',EmptySnapshot([]));
    Control = regionGANExperimentControl();
    if isempty(Control) || ~isfield(Control,'stageTargets') || ...
            isempty(Control.stageTargets)
        return;
    end
    targets = double(Control.stageTargets(:)');
    targets = unique(targets(isfinite(targets) & targets > 0),'stable');
    if isprop(Problem,'maxFE') && ~isempty(Problem.maxFE)
        targets = targets(targets <= double(Problem.maxFE));
    end
    Observer.targets = targets;
end

function [Observer,captured] = maybeCaptureRegionStages(Observer, ...
        Algorithm,Problem,gen,Samples,DatasetInfo,OffspringG,Diag)
    captured = false;
    while Observer.next <= numel(Observer.targets) && ...
            Problem.FE >= Observer.targets(Observer.next)
        Snapshot = makeRegionStageSnapshot(Observer,Algorithm,Problem, ...
            gen,Samples,DatasetInfo,OffspringG,Diag, ...
            Observer.targets(Observer.next));
        Observer.snapshots(end+1) = Snapshot;
        Observer.next = Observer.next + 1;
        captured = true;
    end
end

function Snapshot = makeRegionStageSnapshot(Observer,Algorithm,Problem, ...
        gen,Samples,DatasetInfo,OffspringG,Diag,targetFE)
    Snapshot = emptyRegionStageSnapshot(Problem.M);
    Snapshot.problem = string(class(Problem));
    Snapshot.run = doubleOrNaN(Algorithm.run);
    if ~isfinite(Snapshot.run)
        Snapshot.run = Observer.run;
    end
    Snapshot.target_FE = double(targetFE);
    Snapshot.actual_FE = double(Problem.FE);
    Snapshot.generation = double(gen);
    Snapshot.query_mode = string(Diag.query_mode);
    Snapshot.gan_iter_used = double(Diag.gan_iter_used);
    Snapshot.gan_iter_schedule = string(Diag.gan_iter_schedule);
    Snapshot.sample_z_mode = string(Diag.sample_z_mode);
    Snapshot.train_z_mode = string(Diag.train_z_mode);
    Snapshot.train_z_sigma = double(Diag.train_z_sigma);
    Snapshot.sample_z_sigma = double(Diag.sample_z_sigma);
    Snapshot.prescreen_multiplier = double(Diag.prescreen_multiplier);
    Snapshot.prescreen_candidate_count = ...
        double(Diag.prescreen_candidate_count);
    Snapshot.prescreen_selected_count = ...
        double(Diag.prescreen_selected_count);
    Snapshot.train_count = double(Diag.train_count);
    Snapshot.prev_bmem_candidate_count = ...
        double(Diag.prev_bmem_candidate_count);
    Snapshot.prev_bmem_survivor_count = ...
        double(Diag.prev_bmem_survivor_count);
    Snapshot.raw_generated_count = double(Diag.raw_generated_count);
    Snapshot.feasible_rate = double(Diag.feasible_rate);
    Snapshot.gap_ratio50 = double(Diag.gap_ratio50);
    Snapshot.gap_ratio90 = double(Diag.gap_ratio90);
    Snapshot.near_boundary_rate_gap1 = ...
        double(Diag.near_boundary_rate_gap1);
    Snapshot.better_than_random_gap_rate = ...
        double(Diag.better_than_random_gap_rate);

    [Snapshot.sample_objs,Snapshot.sample_cons, ...
        Snapshot.sample_feasible] = regionSolutionData(Samples,Problem.M);
    if isstruct(DatasetInfo) && isfield(DatasetInfo,'trainObjs')
        Snapshot.train_objs = DatasetInfo.trainObjs;
    end
    [Snapshot.generated_objs,Snapshot.generated_cons, ...
        Snapshot.generated_feasible] = regionSolutionData( ...
        OffspringG,Problem.M);
end

function Snapshot = emptyRegionStageSnapshot(M)
    Snapshot = struct( ...
        'problem',"", ...
        'run',NaN, ...
        'target_FE',NaN, ...
        'actual_FE',NaN, ...
        'generation',NaN, ...
        'query_mode',"", ...
        'gan_iter_used',NaN, ...
        'gan_iter_schedule',"fixed", ...
        'sample_z_mode',"random", ...
        'train_z_mode',"random", ...
        'train_z_sigma',1.0, ...
        'sample_z_sigma',1.0, ...
        'prescreen_multiplier',1, ...
        'prescreen_candidate_count',0, ...
        'prescreen_selected_count',0, ...
        'train_count',0, ...
        'prev_bmem_candidate_count',0, ...
        'prev_bmem_survivor_count',0, ...
        'raw_generated_count',0, ...
        'feasible_rate',NaN, ...
        'gap_ratio50',NaN, ...
        'gap_ratio90',NaN, ...
        'near_boundary_rate_gap1',NaN, ...
        'better_than_random_gap_rate',NaN, ...
        'sample_objs',zeros(0,M), ...
        'sample_cons',zeros(0,0), ...
        'sample_feasible',false(0,1), ...
        'train_objs',zeros(0,M), ...
        'generated_objs',zeros(0,M), ...
        'generated_cons',zeros(0,0), ...
        'generated_feasible',false(0,1));
end

function [Obj,Con,Feasible] = regionSolutionData(Population,M)
    Obj = zeros(0,M);
    Con = zeros(0,0);
    Feasible = false(0,1);
    if isempty(Population)
        return;
    end
    Obj = double(Population.objs);
    Con = double(Population.cons);
    if isempty(Con)
        Feasible = true(size(Obj,1),1);
    else
        Feasible = sum(max(0,Con),2) <= 0;
    end
end

function value = doubleOrNaN(value)
    try
        value = double(value);
    catch
        value = NaN;
    end
    if isempty(value) || ~isfinite(value)
        value = NaN;
    else
        value = value(1);
    end
end

function value = regionProblemMaxFE(Problem)
    value = NaN;
    if isprop(Problem,'maxFE') && ~isempty(Problem.maxFE)
        value = double(Problem.maxFE);
    end
    if isempty(value) || ~isfinite(value(1)) || value(1) <= 0
        value = NaN;
    else
        value = value(1);
    end
end
