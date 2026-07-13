classdef (Abstract) CBS_RegionGAN_Base < ALGORITHM
%CBS_REGIONGAN_BASE Evolutionary loop for the region-conditioned WGAN.

    methods(Access = protected)
        function Algorithm = CBS_RegionGAN_Base(varargin)
            Algorithm@ALGORITHM(varargin{:});
        end

        function runRegionGAN(Algorithm,Problem,Config)
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
            maxAnchorsPerRef = max(1,round(double( ...
                optionalConfig(Config,'maxAnchorsPerRef',5))));
            minGANTrainCount = max(1,round(double( ...
                optionalConfig(Config,'minGANTrainCount',32))));
            ganLrD = double(Config.ganLrD);
            ganLrG = double(Config.ganLrG);
            gpLambda = double(Config.gpLambda);
            nCritic = double(Config.nCritic);
            gpLambda = max(0,double(gpLambda));
            nCritic = max(1,round(double(nCritic)));
            bmemMode = regionBMemModeFromControl();

            nRef = max(2,round(Problem.N/refDivisor));
            [W,~] = UniformPoint(nRef,Problem.M);
            MemOptions = struct( ...
                'frontDepth',frontDepth, ...
                'pairNeighborRefRadius',pairNeighborRefRadius, ...
                'minBoundaryLength',minBoundaryLength, ...
                'maxAnchorsPerRef',maxAnchorsPerRef, ...
                'bmemMode',bmemMode);
            DatasetOptions = struct( ...
                'trainDedupMode',regionTrainDedupModeFromControl());
            GANOptions = regionGANOptions(zDim,ganIter,ganMiniBatch, ...
                ganLrD,ganLrG,gpLambda,nCritic);
            GANOptions = applyRegionGANConfigOptions(GANOptions,Config);
            GANOptions = applyRegionGANExperimentOptions(GANOptions);
            GANOptions.minTrainCount = minGANTrainCount;
            BMemLearnabilityDiagnostics = ...
                regionBMemLearnabilityDiagnosticsFromControl();

            Population1 = Problem.Initialization();
            Population2 = Problem.Initialization();
            DatasetOptions.conditionScale = ...
                initRegionConditionScale([Population1,Population2],Problem);
            Fitness1 = CalFitness_CBS(Population1.objs,Population1.cons);
            Fitness2 = CalFitness_CBS(Population2.objs);

            BMem = [];
            GAN  = [];
            gen  = 0;
            QueryTracker = RegionQueryTracker_RC('init',Problem.D);
            QuerySampleLog = cell(0,1);
            EmptyDiag = emptyRegionDiag(zDim,gpLambda,nCritic);
            [lastMetric,historyMetric,cloudMetric] = ...
                RunRegionGAN_RC('metricnames');
            [trackerMetric,finalMetric,querySampleMetric] = ...
                regionAttributionMetricNames();
            Algorithm.metric.(lastMetric) = EmptyDiag;
            Algorithm.metric.(historyMetric) = EmptyDiag([]);
            Algorithm.metric.(cloudMetric) = [];
            Algorithm.metric.region_gan_last = EmptyDiag;
            Algorithm.metric.region_gan_history = EmptyDiag([]);
            Algorithm.metric.region_gan_bmem_history = cell(0,1);
            Algorithm.metric.region_gan_bmem_ref_history = cell(0,1);
            Algorithm.metric.region_gan_latent_scale_ref_history = cell(0,1);
            Algorithm.metric.region_gan_query_tracker = ...
                RegionQueryTracker_RC('export',QueryTracker);
            Algorithm.metric.(trackerMetric) = ...
                Algorithm.metric.region_gan_query_tracker;
            Algorithm.metric.region_gan_query_samples = ...
                exportRegionQuerySamples(QuerySampleLog);
            Algorithm.metric.(querySampleMetric) = ...
                Algorithm.metric.region_gan_query_samples;
            StageObserver = initRegionStageObserver(Algorithm,Problem);
            Algorithm.metric.region_gan_stage_snapshots = ...
                StageObserver.snapshots;
            Algorithm.metric.region_wgan_gp_stage_snapshots = ...
                StageObserver.snapshots;

            try
            while Algorithm.NotTerminated(Population1)
                gen = gen + 1;
                remainingFE = max(0,Problem.maxFE - Problem.FE);
                deBudget = min(2*Problem.N,remainingFE);
                deCount1 = min(Problem.N,ceil(deBudget/2));
                deCount2 = min(Problem.N,floor(deBudget/2));
                Offspring1 = generateRegionDEOffspring( ...
                    Problem,Population1,Fitness1,deCount1);
                Offspring2 = generateRegionDEOffspring( ...
                    Problem,Population2,Fitness2,deCount2);

                Diag = emptyRegionDiag(zDim,gpLambda,nCritic);
                Diag.generation = gen;
                Diag.fe = double(Problem.FE);
                EventGANOptions = GANOptions;
                Diag = copyGANOptionInfo(Diag,EventGANOptions);
                DatasetInfo = emptyRegionDatasetInfo(Problem.M);
                SampleRefs = zeros(0,1);
                SampleGroups = zeros(0,1);
                LatentScaleRows = struct([]);
                if mod(gen,archiveGap) == 0
                    phaseTimer = tic;
                    [BMem,MemDiag] = UpdateBoundaryMemory_RC(BMem, ...
                        Population1,Offspring1,Population2,Offspring2, ...
                        W,MemOptions);
                    Diag = copyMemDiag(Diag,MemDiag);
                    [QueryTracker,TrackerUpdate] = RegionQueryTracker_RC( ...
                        'updatebmem',QueryTracker,gen,Problem.FE,BMem);
                    Diag = copyTrackerUpdateDiag(Diag,TrackerUpdate);
                    Diag = copyBMemTrueBoundaryDiag(Diag,BMem,Problem);
                    if BMemLearnabilityDiagnostics
                        Algorithm.metric.region_gan_bmem_history{end+1,1} = ...
                            makeRegionBMemSnapshot(BMem,gen,Problem.FE);
                    end
                    Algorithm.metric.(cloudMetric) = updateDensestCloud( ...
                        Algorithm.metric.(cloudMetric),BMem,Problem);
                    Diag.time_bmem_dataset_query = ...
                        Diag.time_bmem_dataset_query + toc(phaseTimer);
                end

                OffspringG = Offspring1([]);
                periodicTrain = mod(gen,trainGap) == 0;
                remainingFE = max(0,Problem.maxFE - Problem.FE);
                eventNGen = min(nGen,remainingFE);
                inspectTrain = ~isempty(BMem) && eventNGen > 0 && ...
                    periodicTrain;
                if inspectTrain
                    phaseTimer = tic;
                    [TrainX,TrainC,QueryC,BMem,DatasetInfo] = ...
                        BuildBoundaryDataset_RC( ...
                        BMem,[Population1,Offspring1,Population2,Offspring2], ...
                        W,Problem,DatasetOptions);
                    Diag.train_count = size(TrainX,1);
                    Diag.train_count_raw = ...
                        double(DatasetInfo.raw_valid_train_count);
                    Diag.train_exact_duplicate_count = ...
                        double(DatasetInfo.exact_duplicate_train_count);
                    Diag.train_removed_duplicate_count = ...
                        double(DatasetInfo.removed_duplicate_train_count);
                    Diag.train_dedup_enabled = ...
                        double(DatasetInfo.train_dedup_enabled);
                    Diag.train_duplicate_fraction = safeRatioLocal( ...
                        Diag.train_exact_duplicate_count, ...
                        Diag.train_count_raw);
                    Diag.query_count = regionQueryPoolCount( ...
                        QueryC,DatasetInfo,W);
                    Diag.time_bmem_dataset_query = ...
                        Diag.time_bmem_dataset_query + toc(phaseTimer);
                    if size(TrainX,1) >= max(minBoundaryLength, ...
                            minGANTrainCount) && ~isempty(QueryC)
                        phaseTimer = tic;
                        [SampleC,QueryAllocation,SampleRefs,~,~, ...
                            SampleGroups] = makeRegionQuerySamples( ...
                            QueryC,DatasetInfo,W,eventNGen);
                        Diag.time_bmem_dataset_query = ...
                            Diag.time_bmem_dataset_query + toc(phaseTimer);
                        Diag.query_sample_count = size(SampleC,1);
                        Diag.query_per_region_min = minNonempty(QueryAllocation);
                        Diag.query_per_region_max = maxNonempty(QueryAllocation);
                        Diag.query_unique_ref_count = numel(unique( ...
                            SampleRefs(isfinite(SampleRefs) & ...
                            SampleRefs > 0)));
                        EventGANOptions.queryActiveModeCount = ...
                            regionQueryActiveModeCounts( ...
                            SampleRefs,SampleGroups,DatasetInfo, ...
                            getOptionNumeric(EventGANOptions, ...
                            'structuredZMaxModes',5));
                        phaseTimer = tic;
                        [GAN,RawDec] = RunRegionGAN_RC('trainandsample', ...
                            GAN,TrainX,TrainC,SampleC,1, ...
                            Problem,EventGANOptions);
                        Diag.time_gan_train_sample = toc(phaseTimer);
                        Diag = copyGANTrainInfo(Diag,GAN);
                        if getOptionNumeric(EventGANOptions, ...
                                'mappingDiagnostics',0) > 0
                            [LatentScaleRows,LatentScaleDiag] = ...
                                BoundaryWGAN_RC('diagnoselatentscale', ...
                                GAN,TrainX,TrainC,Problem,EventGANOptions);
                            LatentScaleRows = attachRegionLatentScaleRefs( ...
                                LatentScaleRows,DatasetInfo);
                            if isempty(LatentScaleRows)
                                LatentScaleDiag.latent_scale_ref_count = 0;
                            else
                                LatentScaleDiag.latent_scale_ref_count = ...
                                    numel(unique(double([LatentScaleRows.ref])));
                            end
                            Diag = copyStructFields(Diag,LatentScaleDiag);
                        end
                        remainingFE = max(0,Problem.maxFE - Problem.FE);
                        if size(RawDec,1) > remainingFE
                            RawDec = RawDec(1:remainingFE,:);
                        end
                        if numel(SampleRefs) > size(RawDec,1)
                            SampleRefs = SampleRefs(1:size(RawDec,1));
                        end
                        if numel(SampleGroups) > size(RawDec,1)
                            SampleGroups = SampleGroups(1:size(RawDec,1));
                        end
                        Diag.raw_generated_count = size(RawDec,1);
                        if ~isempty(RawDec)
                            phaseTimer = tic;
                            OffspringG = Problem.Evaluation(RawDec);
                            Diag.time_gan_evaluation = toc(phaseTimer);
                            CVG = sum(max(0,OffspringG.cons),2);
                            Diag.feasible_generated_count = sum(CVG <= 0);
                            Diag.feasible_rate = mean(double(CVG <= 0));
                            BoundaryDiag = regionGeneratedBoundaryDiagnostics( ...
                                OffspringG.objs,OffspringG.cons,RawDec, ...
                                SampleRefs,BMem,W,Problem, ...
                                struct('seed',786433 + gen));
                            Diag = copyGeneratedBoundaryDiag(Diag, ...
                                BoundaryDiag);
                            WidthDiag = regionWidthDiagnostics( ...
                                DatasetInfo,OffspringG,SampleRefs,BMem, ...
                                Problem);
                            Diag = copyGeneratedBoundaryDiag(Diag, ...
                                WidthDiag);
                            if getOptionNumeric(EventGANOptions, ...
                                    'mappingDiagnostics',0) > 0
                                MappingDiag = ...
                                    regionQueryGroupMappingDiagnostics( ...
                                    TrainX,DatasetInfo,OffspringG, ...
                                    SampleGroups,BMem,Problem);
                                Diag = copyStructFields(Diag,MappingDiag);
                            end
                            if BMemLearnabilityDiagnostics
                                [LearnabilityDiag,RefLearnability] = ...
                                    regionBMemLearnabilityDiagnostics( ...
                                    TrainX,DatasetInfo,OffspringG, ...
                                    SampleRefs,SampleGroups,BMem,Problem);
                                Diag = copyStructFields( ...
                                    Diag,LearnabilityDiag);
                                Algorithm.metric.region_gan_bmem_ref_history{ ...
                                    end+1,1} = ...
                                    makeRegionBMemRefSnapshot( ...
                                    RefLearnability,gen,Problem.FE);
                            end
                        end
                    end
                end

                CurrentSamples = [Population1,Offspring1,Population2, ...
                    Offspring2];
                Union = [Population1,Population2,Offspring1,Offspring2, ...
                    OffspringG];
                phaseTimer = tic;
                [NextPopulation1,NextFitness1] = EnvironmentalSelection_CBS( ...
                    Union,Problem.N,true);
                Diag.time_selection_P1 = toc(phaseTimer);
                phaseTimer = tic;
                [NextPopulation2,NextFitness2] = EnvironmentalSelection_CBS( ...
                    Union,Problem.N,false);
                Diag.time_selection_P2 = toc(phaseTimer);

                diagnosticTimer = tic;
                [GDec,GObj,GCon] = regionPopulationArrays( ...
                    OffspringG,Problem.D,Problem.M);
                [InP1,InP2,InUnion] = RunRegionGAN_RC( ...
                    'offspringsurvivalhandlemasks',OffspringG, ...
                    NextPopulation1,NextPopulation2);
                QuerySampleLog{end+1,1} = makeRegionQuerySampleRows( ...
                    gen,Problem.FE,SampleRefs,SampleGroups, ...
                    InP1,InP2,InUnion); %#ok<AGROW>
                GroupDiag = RunRegionGAN_RC('querygroupdiagnostics', ...
                    GObj,GCon,SampleGroups,InP1,InP2,InUnion, ...
                    regionProblemPFPreserveRNG(Problem),struct());
                Diag = copyStructFields(Diag,GroupDiag);
                Diag = copyQueryGroupUniqueRefCounts( ...
                    Diag,SampleRefs,SampleGroups);
                Diag = copyP1P2SurvivalDiag( ...
                    Diag,InP1,InP2,InUnion);
                if any(SampleGroups == 2)
                    DEPopulation = [Offspring1,Offspring2];
                    [~,DEObj,DECon] = regionPopulationArrays( ...
                        DEPopulation,Problem.D,Problem.M);
                    [DEInP1,DEInP2,DEInUnion] = RunRegionGAN_RC( ...
                        'offspringsurvivalhandlemasks',DEPopulation, ...
                        NextPopulation1,NextPopulation2);
                    DERefs = RunRegionGAN_RC('assignobjectivequeryrefs', ...
                        DEObj,W,DatasetInfo.objMin,DatasetInfo.objSpan);
                    GANMatchedData = struct( ...
                        'obj',GObj,'con',GCon,'ref',SampleRefs, ...
                        'group',SampleGroups,'survive_P1',InP1, ...
                        'survive_P2',InP2,'survive_union',InUnion);
                    DEMatchedData = struct( ...
                        'obj',DEObj,'con',DECon,'ref',DERefs, ...
                        'survive_P1',DEInP1,'survive_P2',DEInP2, ...
                        'survive_union',DEInUnion);
                    MatchedDiag = RunRegionGAN_RC( ...
                        'matchedfrontierdiagnostics', ...
                        GANMatchedData,DEMatchedData, ...
                        regionProblemPFPreserveRNG(Problem),struct());
                    Diag = copyStructFields(Diag,MatchedDiag);
                end
                Diag = copyGeneratedBoundaryDiag(Diag, ...
                    regionOffspringGSurvivalDiagnostics(OffspringG, ...
                    [NextPopulation1,NextPopulation2]));
                [QueryTracker,~] = RegionQueryTracker_RC('register', ...
                    QueryTracker,gen,Problem.FE,SampleRefs,SampleGroups, ...
                    GDec,InP1,InP2);
                Diag = copyTrackerSummaryDiag(Diag, ...
                    RegionQueryTracker_RC('summary',QueryTracker));
                Diag.fe = double(Problem.FE);
                if ~isempty(LatentScaleRows)
                    latentEventIndex = numel( ...
                        Algorithm.metric.region_gan_history) + 1;
                    Algorithm.metric.region_gan_latent_scale_ref_history{ ...
                        end+1,1} = makeRegionLatentScaleRefSnapshot( ...
                        LatentScaleRows,latentEventIndex,gen,Diag.fe);
                end
                Diag.time_diagnostics_serialization = toc(diagnosticTimer);
                Algorithm.metric.(lastMetric) = Diag;
                Algorithm.metric.(historyMetric)(end+1) = Diag;
                Algorithm.metric.region_gan_last = Diag;
                Algorithm.metric.region_gan_history(end+1) = Diag;
                [StageObserver,captured] = maybeCaptureRegionStages( ...
                    StageObserver,Algorithm,Problem,gen,CurrentSamples, ...
                    DatasetInfo,OffspringG,Diag);
                if captured
                    Algorithm.metric.region_gan_stage_snapshots = ...
                        StageObserver.snapshots;
                    Algorithm.metric.region_wgan_gp_stage_snapshots = ...
                        StageObserver.snapshots;
                end

                Population1 = NextPopulation1;
                Fitness1 = NextFitness1;
                Population2 = NextPopulation2;
                Fitness2 = NextFitness2;
            end

            catch Error
                if strcmp(Error.identifier,'PlatEMO:Termination')
                    storeFinalRegionMetrics(Algorithm,QueryTracker, ...
                        QuerySampleLog,Population1,Population2,Problem, ...
                        trackerMetric,finalMetric,querySampleMetric);
                end
                rethrow(Error);
            end
            storeFinalRegionMetrics(Algorithm,QueryTracker, ...
                QuerySampleLog,Population1,Population2,Problem, ...
                trackerMetric,finalMetric,querySampleMetric);
        end
    end
end

function Offspring = generateRegionDEOffspring( ...
        Problem,Population,Fitness,count)
    count = max(0,min(numel(Population),round(double(count))));
    if count == 0
        Offspring = Population([]);
        return;
    end
    matingPool = TournamentSelection(2,2*count,Fitness);
    if count == numel(Population)
        base = Population;
    else
        base = Population(randperm(numel(Population),count));
    end
    Offspring = OperatorDE(Problem,base, ...
        Population(matingPool(1:count)), ...
        Population(matingPool(count+1:end)));
end

function Options = regionGANOptions(zDim,iter,miniBatch,lrD,lrG, ...
        gpLambda,nCritic)
    Options = struct( ...
        'zDim',zDim, ...
        'iter',iter, ...
        'miniBatch',miniBatch, ...
        'lrD',double(lrD), ...
        'lrG',double(lrG), ...
        'sigma',1.0, ...
        'trainSigma',[], ...
        'sampleSigma',[], ...
        'structuredZMode',"off", ...
        'structuredZMaxModes',5, ...
        'structuredZLambda',1.0, ...
        'generatorHidden',[32 32], ...
        'gpLambda',gpLambda, ...
        'nCritic',nCritic, ...
        'criticHidden',[32 32]);
end

function Options = applyRegionGANExperimentOptions(Options)
    Control = regionGANExperimentControl();
    if isempty(Control) || ~isstruct(Control)
        return;
    end
    if isfield(Control,'captureWGANTrainHistory') && ...
            ~isempty(Control.captureWGANTrainHistory)
        Options.captureTrainHistory = logical( ...
            Control.captureWGANTrainHistory);
    end
    if isfield(Control,'wganMappingDiagnostics') && ...
            ~isempty(Control.wganMappingDiagnostics)
        Options.mappingDiagnostics = logical( ...
            Control.wganMappingDiagnostics);
    end
    if isfield(Control,'mappingDiagMaxConditions') && ...
            ~isempty(Control.mappingDiagMaxConditions)
        Options.mappingDiagMaxConditions = max(1,round(double( ...
            Control.mappingDiagMaxConditions)));
    end
    if isfield(Control,'mappingDiagZSamples') && ...
            ~isempty(Control.mappingDiagZSamples)
        Options.mappingDiagZSamples = max(2,round(double( ...
            Control.mappingDiagZSamples)));
    end
    if isfield(Control,'structuredZMode') && ...
            ~isempty(Control.structuredZMode)
        Options.structuredZMode = lower(strip(string( ...
            Control.structuredZMode)));
    end
    if isfield(Control,'structuredZMaxModes') && ...
            ~isempty(Control.structuredZMaxModes)
        Options.structuredZMaxModes = max(1,round(double( ...
            Control.structuredZMaxModes)));
    end
    if isfield(Control,'structuredZLambda') && ...
            ~isempty(Control.structuredZLambda)
        Options.structuredZLambda = double(Control.structuredZLambda);
    end
end

function Options = applyRegionGANConfigOptions(Options,Config)
    if ~isstruct(Config)
        return;
    end
    if isfield(Config,'sampleSigma') && ~isempty(Config.sampleSigma)
        Options.sampleSigma = double(Config.sampleSigma);
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
        'last_generator_adversarial_loss','last_mi_loss', ...
        'last_mode_decoder_accuracy','last_gradient_penalty', ...
        'last_score_real','last_score_fake'};
    for i = 1 : numel(fields)
        if isstruct(GAN) && isfield(GAN,fields{i})
            Diag.(fields{i}) = double(GAN.(fields{i}));
        end
    end
    if isstruct(GAN) && isfield(GAN,'train_history')
        Diag.gan_train_history = GAN.train_history;
    end
    if isstruct(GAN) && isfield(GAN,'iterQ')
        Diag.mode_decoder_update_count = double(GAN.iterQ);
    end
    diagnosticFields = {'critic_train_gap','critic_holdout_gap', ...
        'critic_train_real_score_mean','critic_train_fake_score_mean', ...
        'critic_holdout_real_score_mean', ...
        'critic_holdout_fake_score_mean', ...
        'critic_train_diag_count','critic_holdout_count', ...
        'prequential_new_count','prequential_seen_count', ...
        'prequential_pre_critic_gap','prequential_post_critic_gap', ...
        'prequential_pre_dec_dist50','prequential_pre_dec_dist90', ...
        'prequential_post_dec_dist50','prequential_post_dec_dist90', ...
        'condition_diag_condition_count','condition_diag_z_count', ...
        'same_z_diff_c_dec_median','same_c_diff_z_dec_median', ...
        'same_c_diff_z_collapse_rate','condition_effect_ratio_dec'};
    for i = 1 : numel(diagnosticFields)
        name = diagnosticFields{i};
        if isstruct(GAN) && isfield(GAN,name)
            Diag.(name) = double(GAN.(name));
        end
    end
    if isstruct(GAN) && isfield(GAN,'last_sample_info') && ...
            isstruct(GAN.last_sample_info)
        if isfield(GAN.last_sample_info,'generated_critic_score')
            scores = double(GAN.last_sample_info.generated_critic_score(:));
            Diag.generated_critic_score = scores;
            Diag.generated_critic_score_count = numel(scores);
            Diag.generated_critic_score_min = percentileFiniteLocal( ...
                scores,0);
            Diag.generated_critic_score_max = percentileFiniteLocal( ...
                scores,100);
            Diag.generated_critic_score_mean = meanFiniteLocal(scores);
        end
        Diag = copyStructuredZSampleInfo(Diag,GAN.last_sample_info);
    end
end

function Diag = copyStructuredZSampleInfo(Diag,Info)
    required = {'mode_label','mode_prediction','mode_confidence', ...
        'mode_correct','active_mode_count'};
    if ~all(isfield(Info,required)) || isempty(Info.mode_label)
        return;
    end
    label = round(double(Info.mode_label(:)));
    prediction = round(double(Info.mode_prediction(:)));
    confidence = double(Info.mode_confidence(:));
    correct = double(Info.mode_correct(:));
    active = round(double(Info.active_mode_count(:)));
    n = min([numel(label),numel(prediction),numel(confidence), ...
        numel(correct),numel(active)]);
    if n == 0
        return;
    end
    label = label(1:n);
    prediction = prediction(1:n);
    confidence = confidence(1:n);
    correct = correct(1:n);
    active = active(1:n);
    Diag.mode_sample_count = n;
    Diag.mode_multi_active_sample_count = sum(active > 1);
    Diag.mode_sample_decoder_accuracy = mean(correct,'omitnan');
    Diag.mode_sample_confidence_mean = mean(confidence,'omitnan');
    Diag.mode_active_count_mean = mean(active,'omitnan');
    Diag.mode_active_count_max = max(active,[],'omitnan');
    maxModes = max([1;active;label;prediction]);
    Diag.mode_sample_label_entropy = normalizedCategoricalEntropy( ...
        label,maxModes);
    Diag.mode_sample_prediction_entropy = normalizedCategoricalEntropy( ...
        prediction,maxModes);
end

function value = normalizedCategoricalEntropy(labels,K)
    labels = round(double(labels(:)));
    labels = labels(isfinite(labels) & labels >= 1 & labels <= K);
    if isempty(labels) || K <= 1
        value = 0;
        return;
    end
    counts = accumarray(labels,1,[K,1]);
    probability = counts/sum(counts);
    probability = probability(probability > 0);
    value = -sum(probability.*log(probability))/log(K);
end

function Diag = copyGANOptionInfo(Diag,Options)
    if ~isstruct(Options)
        return;
    end
    Diag.zDim = getOptionNumeric(Options,'zDim',Diag.zDim);
    Diag.gan_iter_used = getOptionNumeric(Options,'iter', ...
        Diag.gan_iter_used);
    Diag.train_z_sigma = getOptionNumeric(Options,'trainSigma', ...
        getOptionNumeric(Options,'sigma',Diag.train_z_sigma));
    Diag.sample_z_sigma = getOptionNumeric(Options,'sampleSigma', ...
        getOptionNumeric(Options,'sigma',Diag.sample_z_sigma));
    Diag.structured_z_enabled = double(isfield(Options, ...
        'structuredZMode') && string(Options.structuredZMode) == ...
        "categorical_mi");
    Diag.structured_z_max_modes = getOptionNumeric(Options, ...
        'structuredZMaxModes',Diag.structured_z_max_modes);
    Diag.structured_z_lambda = getOptionNumeric(Options, ...
        'structuredZLambda',Diag.structured_z_lambda);
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

function [SampleC,Counts,SampleRefs,DiagQueryC,DiagQueryRefs, ...
        SampleGroups] = ...
        makeRegionQuerySamples(QueryC,DatasetInfo,W,nGen)
    [SampleC,Counts,SampleRefs,DiagQueryC,DiagQueryRefs,SampleGroups] = ...
        RunRegionGAN_RC('regionquerysamples',QueryC,DatasetInfo,W,nGen);
end

function activeModeCount = regionQueryActiveModeCounts( ...
        SampleRefs,SampleGroups,DatasetInfo,maxModes)
    SampleRefs = round(double(SampleRefs(:)));
    SampleGroups = round(double(SampleGroups(:)));
    n = min(numel(SampleRefs),numel(SampleGroups));
    activeModeCount = ones(n,1);
    if n == 0 || ~isstruct(DatasetInfo) || ...
            ~isfield(DatasetInfo,'trainRef') || isempty(DatasetInfo.trainRef)
        return;
    end
    trainRef = round(double(DatasetInfo.trainRef(:)));
    populated = SampleGroups(1:n) == 1;
    populatedRows = find(populated);
    for i = 1 : numel(populatedRows)
        row = populatedRows(i);
        activeModeCount(row) = min(maxModes, ...
            max(1,sum(trainRef == SampleRefs(row))));
    end
end

function count = regionQueryPoolCount(QueryC,DatasetInfo,W)
    count = RunRegionGAN_RC('regionquerypoolcount',QueryC,DatasetInfo,W);
end

function enabled = regionBMemLearnabilityDiagnosticsFromControl()
    enabled = false;
    Control = regionGANExperimentControl();
    if ~isempty(Control) && ...
            isfield(Control,'bmemLearnabilityDiagnostics') && ...
            ~isempty(Control.bmemLearnabilityDiagnostics)
        enabled = logical(Control.bmemLearnabilityDiagnostics);
    end
end

function mode = regionBMemModeFromControl()
    mode = "legacy";
    Control = regionGANExperimentControl();
    if ~isempty(Control) && isfield(Control,'bmemMode') && ...
            ~isempty(Control.bmemMode)
        mode = lower(strip(string(Control.bmemMode)));
    end
    if ~isscalar(mode) || ~ismember(mode,["legacy","coherent"])
        error('CBSRegionGAN:BadBMemMode', ...
            'Experiment control bmemMode must be "legacy" or "coherent".');
    end
end

function mode = regionTrainDedupModeFromControl()
    mode = "off";
    Control = regionGANExperimentControl();
    if ~isempty(Control) && isfield(Control,'trainDedupMode') && ...
            ~isempty(Control.trainDedupMode)
        mode = lower(strip(string(Control.trainDedupMode)));
    end
    if ~isscalar(mode) || ~ismember(mode,["off","exact_ref_x"])
        error('CBSRegionGAN:BadTrainDedupMode', ...
            'Experiment control trainDedupMode must be "off" or "exact_ref_x".');
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

function Diag = copyGeneratedBoundaryDiag(Diag,BoundaryDiag)
    fields = fieldnames(BoundaryDiag);
    for i = 1 : numel(fields)
        value = BoundaryDiag.(fields{i});
        if isnumeric(value) || islogical(value)
            Diag.(fields{i}) = double(value);
        end
    end
end

function Metrics = regionWidthDiagnostics(DatasetInfo,OffspringG, ...
        SampleRefs,BMem,Problem)
    Metrics = emptyRegionWidthMetrics();
    if ~isstruct(DatasetInfo) || ~isfield(DatasetInfo,'trainObjs') || ...
            isempty(DatasetInfo.trainObjs)
        return;
    end
    TrainObj = double(DatasetInfo.trainObjs);
    TrainRef = zeros(size(TrainObj,1),1);
    if isfield(DatasetInfo,'trainRef') && ~isempty(DatasetInfo.trainRef)
        TrainRef = round(double(DatasetInfo.trainRef(:)));
        TrainRef = TrainRef(1:min(numel(TrainRef),size(TrainObj,1)));
        if numel(TrainRef) < size(TrainObj,1)
            TrainRef(end+1:size(TrainObj,1),1) = 0;
        end
    end
    GenObj = zeros(0,Problem.M);
    GenDec = zeros(0,Problem.D);
    if ~isempty(OffspringG)
        GenObj = double(OffspringG.objs);
        GenDec = double(OffspringG.decs);
    end
    Base = TrainObj;
    if ~isempty(GenObj)
        Base = [Base;GenObj];
    end
    if isstruct(BMem) && isfield(BMem,'y_i') && ~isempty(BMem.y_i)
        Base = [Base;double(BMem.y_i)];
    end
    Scale = regionObjectiveScale(Base,Problem.M);
    TrainN = normalizeRegionObj(TrainObj,Scale);
    TrainWidth = groupedCentroidDistances(TrainN,TrainRef);
    Metrics.train_width50 = percentileFiniteLocal(TrainWidth,50);
    Metrics.train_width90 = percentileFiniteLocal(TrainWidth,90);
    Metrics.train_width_count = numel(TrainWidth);

    if isempty(GenObj)
        return;
    end
    GenN = normalizeRegionObj(GenObj,Scale);
    GenRef = zeros(size(GenObj,1),1);
    if ~isempty(SampleRefs)
        GenRef = round(double(SampleRefs(:)));
        GenRef = GenRef(1:min(numel(GenRef),size(GenObj,1)));
        if numel(GenRef) < size(GenObj,1)
            GenRef(end+1:size(GenObj,1),1) = 0;
        end
    end
    GenWidth = groupedCentroidDistances(GenN,GenRef);
    Metrics.gen_width50 = percentileFiniteLocal(GenWidth,50);
    Metrics.gen_width90 = percentileFiniteLocal(GenWidth,90);
    Metrics.gen_width_count = numel(GenWidth);

    DObj = pointDistanceLocal(GenN,TrainN);
    if ~isempty(DObj)
        minObjDist = min(DObj,[],2);
        Metrics.gen_to_train_dist50 = percentileFiniteLocal(minObjDist,50);
        Metrics.gen_to_train_dist90 = percentileFiniteLocal(minObjDist,90);
        Metrics.gen_to_train_dist_count = numel(minObjDist);
    end
    if isstruct(BMem) && isfield(BMem,'x_b') && ~isempty(BMem.x_b) && ...
            ~isempty(GenDec)
        TrainDecN = normalizeRegionDec(double(BMem.x_b),Problem);
        GenDecN = normalizeRegionDec(GenDec,Problem);
        DDec = pointDistanceLocal(GenDecN,TrainDecN);
        if ~isempty(DDec)
            minDecDist = min(DDec,[],2);
            Metrics.gen_to_train_dec_dist50 = ...
                percentileFiniteLocal(minDecDist,50);
            Metrics.gen_to_train_dec_dist90 = ...
                percentileFiniteLocal(minDecDist,90);
        end
    end
end

function Metrics = emptyRegionWidthMetrics()
    Metrics = struct( ...
        'train_width50',NaN, ...
        'train_width90',NaN, ...
        'train_width_count',0, ...
        'gen_width50',NaN, ...
        'gen_width90',NaN, ...
        'gen_width_count',0, ...
        'gen_to_train_dist50',NaN, ...
        'gen_to_train_dist90',NaN, ...
        'gen_to_train_dist_count',0, ...
        'gen_to_train_dec_dist50',NaN, ...
        'gen_to_train_dec_dist90',NaN);
end

function Metrics = regionQueryGroupMappingDiagnostics(TrainX,DatasetInfo, ...
        OffspringG,Groups,BMem,Problem)
    Metrics = struct();
    groupNames = ["populated","frontier","remote"];
    suffixes = ["gen_to_train_dec_dist50", ...
        "gen_to_train_dec_dist90","gen_to_train_obj_dist50", ...
        "gen_to_train_obj_dist90","pair_normal_abs50", ...
        "pair_normal_abs90","pair_tangent_dist50", ...
        "pair_tangent_dist90"];
    for group = 1 : numel(groupNames)
        for s = 1 : numel(suffixes)
            Metrics.(char("query_" + groupNames(group) + "_" + ...
                suffixes(s))) = NaN;
        end
    end
    if isempty(OffspringG) || isempty(TrainX) || ...
            ~isstruct(DatasetInfo) || ...
            ~isfield(DatasetInfo,'trainObjs') || ...
            isempty(DatasetInfo.trainObjs)
        return;
    end
    GenDec = double(OffspringG.decs);
    GenObj = double(OffspringG.objs);
    Groups = double(Groups(:));
    n = min([size(GenDec,1),size(GenObj,1),numel(Groups)]);
    if n <= 0
        return;
    end
    GenDec = GenDec(1:n,:);
    GenObj = GenObj(1:n,:);
    Groups = Groups(1:n);

    TrainDec = double(TrainX);
    TrainObj = double(DatasetInfo.trainObjs);
    trainN = min(size(TrainDec,1),size(TrainObj,1));
    TrainDec = TrainDec(1:trainN,:);
    TrainObj = TrainObj(1:trainN,:);
    if trainN <= 0
        return;
    end
    Scale = struct('min',double(DatasetInfo.objMin(:)'), ...
        'span',double(DatasetInfo.objSpan(:)'));
    if numel(Scale.min) ~= Problem.M || ...
            numel(Scale.span) ~= Problem.M
        Scale = regionObjectiveScale([TrainObj;GenObj],Problem.M);
    end
    Scale.span(~isfinite(Scale.span) | Scale.span <= eps) = 1;
    TrainObjN = normalizeRegionObj(TrainObj,Scale);
    GenObjN = normalizeRegionObj(GenObj,Scale);
    TrainDecN = normalizeRegionDec(TrainDec,Problem);
    GenDecN = normalizeRegionDec(GenDec,Problem);
    DObj = pointDistanceLocal(GenObjN,TrainObjN);
    DDec = pointDistanceLocal(GenDecN,TrainDecN);
    minObj = min(DObj,[],2);
    minDec = min(DDec,[],2);

    normalAbs = NaN(n,1);
    tangentDist = NaN(n,1);
    if isstruct(BMem) && isfield(BMem,'y_f') && ...
            isfield(BMem,'y_i') && ~isempty(BMem.y_f) && ...
            size(BMem.y_f,1) == size(BMem.y_i,1)
        FeasibleN = normalizeRegionObj(double(BMem.y_f),Scale);
        InfeasibleN = normalizeRegionObj(double(BMem.y_i),Scale);
        DBMem = pointDistanceLocal(GenObjN,FeasibleN);
        [~,nearest] = min(DBMem,[],2);
        pairVector = InfeasibleN(nearest,:) - FeasibleN(nearest,:);
        pairNorm = sqrt(sum(pairVector.^2,2));
        validPair = isfinite(pairNorm) & pairNorm > eps;
        unitNormal = zeros(size(pairVector));
        unitNormal(validPair,:) = pairVector(validPair,:)./ ...
            pairNorm(validPair);
        displacement = GenObjN - FeasibleN(nearest,:);
        projection = sum(displacement.*unitNormal,2);
        normalAbs(validPair) = abs(projection(validPair));
        tangentSquared = sum(displacement.^2,2) - projection.^2;
        tangentDist(validPair) = sqrt(max(0,tangentSquared(validPair)));
    end

    for group = 1 : numel(groupNames)
        rows = Groups == group;
        prefix = "query_" + groupNames(group) + "_";
        Metrics.(char(prefix + "gen_to_train_dec_dist50")) = ...
            percentileFiniteLocal(minDec(rows),50);
        Metrics.(char(prefix + "gen_to_train_dec_dist90")) = ...
            percentileFiniteLocal(minDec(rows),90);
        Metrics.(char(prefix + "gen_to_train_obj_dist50")) = ...
            percentileFiniteLocal(minObj(rows),50);
        Metrics.(char(prefix + "gen_to_train_obj_dist90")) = ...
            percentileFiniteLocal(minObj(rows),90);
        Metrics.(char(prefix + "pair_normal_abs50")) = ...
            percentileFiniteLocal(normalAbs(rows),50);
        Metrics.(char(prefix + "pair_normal_abs90")) = ...
            percentileFiniteLocal(normalAbs(rows),90);
        Metrics.(char(prefix + "pair_tangent_dist50")) = ...
            percentileFiniteLocal(tangentDist(rows),50);
        Metrics.(char(prefix + "pair_tangent_dist90")) = ...
            percentileFiniteLocal(tangentDist(rows),90);
    end
end

function Snapshot = makeRegionBMemSnapshot(BMem,generation,fe)
    Snapshot = struct( ...
        'generation',double(generation), ...
        'fe',double(fe), ...
        'bmem',BMem);
end

function Snapshot = makeRegionBMemRefSnapshot(Rows,generation,fe)
    Snapshot = struct( ...
        'generation',double(generation), ...
        'fe',double(fe), ...
        'rows',Rows);
end

function Snapshot = makeRegionLatentScaleRefSnapshot( ...
        Rows,eventIndex,generation,fe)
    Snapshot = struct( ...
        'event_index',double(eventIndex), ...
        'generation',double(generation), ...
        'fe',double(fe), ...
        'rows',Rows);
end

function Rows = attachRegionLatentScaleRefs(Rows,DatasetInfo)
    if isempty(Rows)
        return;
    end
    if ~isstruct(DatasetInfo) || ...
            ~isfield(DatasetInfo,'trainConditions') || ...
            ~isfield(DatasetInfo,'trainRef')
        error('CBSRegionGAN:MissingLatentScaleRefMetadata', ...
            'Latent-scale diagnostics require trainConditions and trainRef.');
    end
    TrainC = double(DatasetInfo.trainConditions);
    TrainRef = round(double(DatasetInfo.trainRef(:)));
    if size(TrainC,1) ~= numel(TrainRef)
        error('CBSRegionGAN:BadLatentScaleRefMetadata', ...
            'trainConditions and trainRef must have the same row count.');
    end
    for i = 1 : numel(Rows)
        matches = ismember(TrainC,double(Rows(i).condition),'rows');
        refs = unique(TrainRef(matches & isfinite(TrainRef) & ...
            TrainRef > 0),'stable');
        if numel(refs) ~= 1 || sum(matches) ~= Rows(i).train_count
            error('CBSRegionGAN:LatentScaleConditionRefMismatch', ...
                ['Each latent-scale condition must map to exactly one ', ...
                'populated reference and all of its training rows.']);
        end
        Rows(i).ref = double(refs(1));
    end
    Rows = rmfield(Rows,'condition');
end

function [Metrics,Rows] = regionBMemLearnabilityDiagnostics( ...
        TrainX,DatasetInfo,OffspringG,SampleRefs,SampleGroups,~,Problem)
    Metrics = emptyRegionBMemLearnabilityMetrics();
    Rows = repmat(emptyRegionBMemRefLearnabilityRow(),0,1);
    if isempty(TrainX) || ~isstruct(DatasetInfo) || ...
            ~isfield(DatasetInfo,'trainObjs') || ...
            ~isfield(DatasetInfo,'trainRef') || ...
            isempty(DatasetInfo.trainObjs) || isempty(DatasetInfo.trainRef)
        return;
    end

    TrainDec = double(TrainX);
    TrainObj = double(DatasetInfo.trainObjs);
    TrainRef = round(double(DatasetInfo.trainRef(:)));
    trainCount = min([size(TrainDec,1),size(TrainObj,1),numel(TrainRef)]);
    if trainCount <= 0
        return;
    end
    TrainDec = TrainDec(1:trainCount,:);
    TrainObj = TrainObj(1:trainCount,:);
    TrainRef = TrainRef(1:trainCount);
    TrainSource = datasetNumericVector( ...
        DatasetInfo,'trainSource',trainCount,0);
    TrainAge = datasetNumericVector(DatasetInfo,'trainAge',trainCount,0);
    TrainFrontRank = datasetNumericVector( ...
        DatasetInfo,'trainFrontRank',trainCount,NaN);

    GenDec = zeros(0,Problem.D);
    if ~isempty(OffspringG)
        GenDec = double(OffspringG.decs);
    end
    GenRef = round(double(SampleRefs(:)));
    GenGroup = round(double(SampleGroups(:)));
    genCount = min([size(GenDec,1),numel(GenRef),numel(GenGroup)]);
    GenDec = GenDec(1:genCount,:);
    GenRef = GenRef(1:genCount);
    GenGroup = GenGroup(1:genCount);

    TrainDecN = normalizeRegionDec(TrainDec,Problem);
    GenDecN = normalizeRegionDec(GenDec,Problem);
    Scale = struct( ...
        'min',double(DatasetInfo.objMin(:)'), ...
        'span',double(DatasetInfo.objSpan(:)'));
    if numel(Scale.min) ~= size(TrainObj,2) || ...
            numel(Scale.span) ~= size(TrainObj,2)
        Scale = regionObjectiveScale(TrainObj,size(TrainObj,2));
    end
    Scale.span(~isfinite(Scale.span) | Scale.span <= eps) = 1;
    TrainObjN = normalizeRegionObj(TrainObj,Scale);

    refs = unique([TrainRef;GenRef(isfinite(GenRef) & GenRef > 0)], ...
        'stable');
    refs = refs(isfinite(refs) & refs > 0);
    Rows = repmat(emptyRegionBMemRefLearnabilityRow(),numel(refs),1);
    allTrainDecPair = zeros(0,1);
    allTrainObjPair = zeros(0,1);
    allTrainDecObjRatio = zeros(0,1);
    allGenDecPair = zeros(0,1);
    allGenSameRefDist = zeros(0,1);
    utilizedAnchorCount = 0;
    queriedAnchorCount = 0;

    for k = 1 : numel(refs)
        ref = refs(k);
        trainRows = find(TrainRef == ref);
        genRows = find(GenRef == ref);
        Row = emptyRegionBMemRefLearnabilityRow();
        Row.ref = double(ref);
        Row.train_count = numel(trainRows);
        Row.current_count = sum(TrainSource(trainRows) == 0);
        Row.previous_count = sum(TrainSource(trainRows) ~= 0);
        Row.previous_ratio = safeRatioLocal( ...
            Row.previous_count,Row.train_count);
        Row.age50 = percentileFiniteLocal(TrainAge(trainRows),50);
        Row.age90 = percentileFiniteLocal(TrainAge(trainRows),90);
        Row.age_max = maxFiniteLocal(TrainAge(trainRows));
        Row.front1_count = sum(TrainFrontRank(trainRows) == 1);
        Row.front2_count = sum(TrainFrontRank(trainRows) == 2);
        Row.other_front_count = sum(isfinite(TrainFrontRank(trainRows)) & ...
            TrainFrontRank(trainRows) > 2);

        trainDecPair = withinSetPairDistances(TrainDecN(trainRows,:));
        trainObjPair = withinSetPairDistances(TrainObjN(trainRows,:));
        validPair = isfinite(trainDecPair) & isfinite(trainObjPair);
        trainDecObjRatio = trainDecPair(validPair)./ ...
            max(trainObjPair(validPair),eps);
        Row.train_pair_count = numel(trainDecPair);
        Row.train_dec_pair50 = percentileFiniteLocal(trainDecPair,50);
        Row.train_dec_pair90 = percentileFiniteLocal(trainDecPair,90);
        Row.train_obj_pair50 = percentileFiniteLocal(trainObjPair,50);
        Row.train_obj_pair90 = percentileFiniteLocal(trainObjPair,90);
        Row.train_dec_obj_ratio50 = ...
            percentileFiniteLocal(trainDecObjRatio,50);
        Row.train_dec_obj_ratio90 = ...
            percentileFiniteLocal(trainDecObjRatio,90);
        allTrainDecPair = [allTrainDecPair;trainDecPair]; %#ok<AGROW>
        allTrainObjPair = [allTrainObjPair;trainObjPair]; %#ok<AGROW>
        allTrainDecObjRatio = [allTrainDecObjRatio; ...
            trainDecObjRatio]; %#ok<AGROW>

        Row.generated_count = numel(genRows);
        Row.populated_generated_count = sum(GenGroup(genRows) == 1);
        Row.frontier_generated_count = sum(GenGroup(genRows) == 2);
        Row.remote_generated_count = sum(GenGroup(genRows) == 3);
        genDecPair = withinSetPairDistances(GenDecN(genRows,:));
        Row.generated_pair_count = numel(genDecPair);
        Row.generated_dec_pair50 = percentileFiniteLocal(genDecPair,50);
        Row.generated_dec_pair90 = percentileFiniteLocal(genDecPair,90);
        allGenDecPair = [allGenDecPair;genDecPair]; %#ok<AGROW>

        if ~isempty(trainRows) && ~isempty(genRows)
            D = pointDistanceLocal( ...
                GenDecN(genRows,:),TrainDecN(trainRows,:));
            [minDist,nearest] = min(D,[],2);
            Row.generated_to_same_ref_train_count = numel(minDist);
            Row.generated_to_same_ref_train_dec50 = ...
                percentileFiniteLocal(minDist,50);
            Row.generated_to_same_ref_train_dec90 = ...
                percentileFiniteLocal(minDist,90);
            Row.anchor_utilized_count = numel(unique(nearest));
            Row.anchor_utilization_rate = safeRatioLocal( ...
                Row.anchor_utilized_count,Row.train_count);
            allGenSameRefDist = [allGenSameRefDist;minDist]; %#ok<AGROW>
            utilizedAnchorCount = utilizedAnchorCount + ...
                Row.anchor_utilized_count;
            queriedAnchorCount = queriedAnchorCount + Row.train_count;
        end
        Row.generated_train_diversity_ratio50 = safeRatioLocal( ...
            Row.generated_dec_pair50,Row.train_dec_pair50);
        Row.generated_train_diversity_ratio90 = safeRatioLocal( ...
            Row.generated_dec_pair90,Row.train_dec_pair90);
        Rows(k,1) = Row;
    end

    Metrics.bmem_within_ref_dec_pair_count = numel(allTrainDecPair);
    Metrics.bmem_within_ref_dec_pair50 = ...
        percentileFiniteLocal(allTrainDecPair,50);
    Metrics.bmem_within_ref_dec_pair90 = ...
        percentileFiniteLocal(allTrainDecPair,90);
    Metrics.bmem_within_ref_obj_pair50 = ...
        percentileFiniteLocal(allTrainObjPair,50);
    Metrics.bmem_within_ref_obj_pair90 = ...
        percentileFiniteLocal(allTrainObjPair,90);
    Metrics.bmem_within_ref_dec_obj_ratio50 = ...
        percentileFiniteLocal(allTrainDecObjRatio,50);
    Metrics.bmem_within_ref_dec_obj_ratio90 = ...
        percentileFiniteLocal(allTrainDecObjRatio,90);
    Metrics.bmem_previous_ratio = safeRatioLocal( ...
        sum(TrainSource ~= 0),trainCount);
    Metrics.bmem_age50 = percentileFiniteLocal(TrainAge,50);
    Metrics.bmem_age90 = percentileFiniteLocal(TrainAge,90);
    Metrics.bmem_age_max = maxFiniteLocal(TrainAge);
    Metrics.bmem_front1_ratio = safeRatioLocal( ...
        sum(TrainFrontRank == 1),trainCount);
    Metrics.bmem_front2_ratio = safeRatioLocal( ...
        sum(TrainFrontRank == 2),trainCount);
    Metrics.generated_within_ref_dec_pair_count = numel(allGenDecPair);
    Metrics.generated_within_ref_dec_pair50 = ...
        percentileFiniteLocal(allGenDecPair,50);
    Metrics.generated_within_ref_dec_pair90 = ...
        percentileFiniteLocal(allGenDecPair,90);
    Metrics.generated_same_ref_to_train_dec50 = ...
        percentileFiniteLocal(allGenSameRefDist,50);
    Metrics.generated_same_ref_to_train_dec90 = ...
        percentileFiniteLocal(allGenSameRefDist,90);
    Metrics.generated_queried_anchor_utilized_count = utilizedAnchorCount;
    Metrics.generated_queried_anchor_count = queriedAnchorCount;
    Metrics.generated_queried_anchor_utilization_rate = safeRatioLocal( ...
        utilizedAnchorCount,queriedAnchorCount);
    Metrics.generated_train_dec_diversity_ratio50 = safeRatioLocal( ...
        Metrics.generated_within_ref_dec_pair50, ...
        Metrics.bmem_within_ref_dec_pair50);
    Metrics.generated_train_dec_diversity_ratio90 = safeRatioLocal( ...
        Metrics.generated_within_ref_dec_pair90, ...
        Metrics.bmem_within_ref_dec_pair90);
end

function value = datasetNumericVector(S,name,n,defaultValue)
    value = repmat(double(defaultValue),n,1);
    if isstruct(S) && isfield(S,name) && numel(S.(name)) >= n
        candidate = double(S.(name)(:));
        value = candidate(1:n);
    end
end

function Dist = withinSetPairDistances(X)
    n = size(X,1);
    if n < 2
        Dist = zeros(0,1);
        return;
    end
    D = pointDistanceLocal(X,X);
    Dist = D(triu(true(n),1));
    Dist = Dist(isfinite(Dist));
end

function value = safeRatioLocal(numerator,denominator)
    if isempty(numerator) || isempty(denominator) || ...
            ~isfinite(numerator) || ~isfinite(denominator) || ...
            denominator <= eps
        value = NaN;
    else
        value = double(numerator)./double(denominator);
    end
end

function value = maxFiniteLocal(X)
    X = X(isfinite(X));
    if isempty(X)
        value = NaN;
    else
        value = max(X);
    end
end

function Metrics = emptyRegionBMemLearnabilityMetrics()
    Metrics = struct( ...
        'bmem_within_ref_dec_pair_count',0, ...
        'bmem_within_ref_dec_pair50',NaN, ...
        'bmem_within_ref_dec_pair90',NaN, ...
        'bmem_within_ref_obj_pair50',NaN, ...
        'bmem_within_ref_obj_pair90',NaN, ...
        'bmem_within_ref_dec_obj_ratio50',NaN, ...
        'bmem_within_ref_dec_obj_ratio90',NaN, ...
        'bmem_previous_ratio',NaN, ...
        'bmem_age50',NaN, ...
        'bmem_age90',NaN, ...
        'bmem_age_max',NaN, ...
        'bmem_front1_ratio',NaN, ...
        'bmem_front2_ratio',NaN, ...
        'generated_within_ref_dec_pair_count',0, ...
        'generated_within_ref_dec_pair50',NaN, ...
        'generated_within_ref_dec_pair90',NaN, ...
        'generated_same_ref_to_train_dec50',NaN, ...
        'generated_same_ref_to_train_dec90',NaN, ...
        'generated_queried_anchor_utilized_count',0, ...
        'generated_queried_anchor_count',0, ...
        'generated_queried_anchor_utilization_rate',NaN, ...
        'generated_train_dec_diversity_ratio50',NaN, ...
        'generated_train_dec_diversity_ratio90',NaN);
end

function Row = emptyRegionBMemRefLearnabilityRow()
    Row = struct( ...
        'ref',NaN, ...
        'train_count',0, ...
        'current_count',0, ...
        'previous_count',0, ...
        'previous_ratio',NaN, ...
        'age50',NaN, ...
        'age90',NaN, ...
        'age_max',NaN, ...
        'front1_count',0, ...
        'front2_count',0, ...
        'other_front_count',0, ...
        'train_pair_count',0, ...
        'train_dec_pair50',NaN, ...
        'train_dec_pair90',NaN, ...
        'train_obj_pair50',NaN, ...
        'train_obj_pair90',NaN, ...
        'train_dec_obj_ratio50',NaN, ...
        'train_dec_obj_ratio90',NaN, ...
        'generated_count',0, ...
        'populated_generated_count',0, ...
        'frontier_generated_count',0, ...
        'remote_generated_count',0, ...
        'generated_pair_count',0, ...
        'generated_dec_pair50',NaN, ...
        'generated_dec_pair90',NaN, ...
        'generated_to_same_ref_train_count',0, ...
        'generated_to_same_ref_train_dec50',NaN, ...
        'generated_to_same_ref_train_dec90',NaN, ...
        'anchor_utilized_count',0, ...
        'anchor_utilization_rate',NaN, ...
        'generated_train_diversity_ratio50',NaN, ...
        'generated_train_diversity_ratio90',NaN);
end

function Dist = groupedCentroidDistances(X,Group)
    Dist = zeros(0,1);
    if isempty(X)
        return;
    end
    Group = round(double(Group(:)));
    if numel(Group) < size(X,1)
        Group(end+1:size(X,1),1) = 0;
    end
    Group = Group(1:size(X,1));
    refs = unique(Group(isfinite(Group) & Group > 0),'stable');
    for i = 1 : numel(refs)
        idx = find(Group == refs(i));
        if isempty(idx)
            continue;
        end
        center = mean(X(idx,:),1,'omitnan');
        Dist = [Dist;sqrt(sum((X(idx,:) - center).^2,2))]; %#ok<AGROW>
    end
end

function Xn = normalizeRegionDec(X,Problem)
    lower = double(Problem.lower(:)');
    upper = double(Problem.upper(:)');
    span = upper - lower;
    span(span <= eps) = 1;
    Xn = (double(X) - lower)./span;
    Xn(~isfinite(Xn)) = 0;
end

function Metrics = regionOffspringGSurvivalDiagnostics(OffspringG,Selected)
    if isempty(OffspringG)
        Metrics = RunRegionGAN_RC('offspringgsurvivaldiagnostics', ...
            zeros(0,0),[],zeros(0,0));
        return;
    end
    if isempty(Selected)
        SelectedDec = zeros(0,size(OffspringG.decs,2));
    else
        SelectedDec = double(Selected.decs);
    end
    Metrics = RunRegionGAN_RC('offspringgsurvivaldiagnostics', ...
        double(OffspringG.decs),double(OffspringG.cons),SelectedDec);
end

function [Dec,Obj,Con] = regionPopulationArrays(Population,D,M)
    if isempty(Population)
        Dec = zeros(0,D);
        Obj = zeros(0,M);
        Con = [];
        return;
    end
    Dec = double(Population.decs);
    Obj = double(Population.objs);
    Con = double(Population.cons);
end

function Diag = copyP1P2SurvivalDiag(Diag,InP1,InP2,InUnion)
    n = numel(InUnion);
    Diag.offspringG_survive_P1_count = sum(InP1);
    Diag.offspringG_survive_P2_count = sum(InP2);
    Diag.offspringG_survive_union_count = sum(InUnion);
    if n > 0
        Diag.offspringG_survive_P1_rate = mean(double(InP1));
        Diag.offspringG_survive_P2_rate = mean(double(InP2));
        Diag.offspringG_survive_union_rate = mean(double(InUnion));
    end
end

function Diag = copyQueryGroupUniqueRefCounts(Diag,Refs,Groups)
    Refs = double(Refs(:));
    Groups = double(Groups(:));
    if numel(Refs) ~= numel(Groups)
        error('CBSRegionGAN:BadQueryRefRows', ...
            'Sample refs and query groups must have equal rows.');
    end
    names = ["populated","frontier","remote"];
    for code = 1 : 3
        values = Refs(Groups == code);
        values = values(isfinite(values) & values > 0);
        Diag.(char("query_" + names(code) + ...
            "_unique_ref_count")) = numel(unique(values));
    end
end

function Diag = copyBMemTrueBoundaryDiag(Diag,BMem,Problem)
    Obj = zeros(0,Problem.M);
    if isstruct(BMem) && isfield(BMem,'y_b') && ~isempty(BMem.y_b)
        Obj = double(BMem.y_b);
    elseif isstruct(BMem) && isfield(BMem,'y_f') && ~isempty(BMem.y_f)
        Obj = double(BMem.y_f);
    end
    Metrics = RunRegionGAN_RC('trueboundarydiagnostics',Obj,[], ...
        regionProblemPFPreserveRNG(Problem),struct());
    Diag.bmem_bdist50_true = Metrics.bdist50_true;
    Diag.bmem_bwidth90_10_true = Metrics.bwidth90_10_true;
    Diag.bmem_bcover_eps_true = Metrics.bcover_eps_true;
end

function PF = regionProblemPFPreserveRNG(Problem)
    state = rng;
    cleanup = onCleanup(@()rng(state));
    PF = regionProblemPF(Problem);
end

function Diag = copyTrackerUpdateDiag(Diag,Update)
    Diag.ref_conversion_new_count = Update.ref_conversion_count;
    Diag.direct_bmem_entry_new_count = Update.direct_bmem_entry_count;
    Diag.ref_conversion_new_lag_gen50 = ...
        medianFinite(Update.ref_conversion_lag_gen);
    Diag.ref_conversion_new_lag_fe50 = ...
        medianFinite(Update.ref_conversion_lag_fe);
    Diag.direct_bmem_entry_new_lag_gen50 = ...
        medianFinite(Update.direct_bmem_entry_lag_gen);
    Diag.direct_bmem_entry_new_lag_fe50 = ...
        medianFinite(Update.direct_bmem_entry_lag_fe);
end

function Diag = copyTrackerSummaryDiag(Diag,Summary)
    names = fieldnames(Summary);
    for i = 1 : numel(names)
        Diag.(char("tracker_" + string(names{i}))) = Summary.(names{i});
    end
end

function value = medianFinite(values)
    values = double(values(:));
    values = values(isfinite(values));
    if isempty(values)
        value = NaN;
    else
        value = median(values);
    end
end

function [trackerMetric,finalMetric,querySampleMetric] = ...
        regionAttributionMetricNames()
    trackerMetric = 'region_wgan_gp_query_tracker';
    finalMetric = 'region_wgan_gp_final';
    querySampleMetric = 'region_wgan_gp_query_samples';
end

function storeFinalRegionMetrics(Algorithm,QueryTracker,QuerySampleLog, ...
        Population1,Population2,Problem,trackerMetric,finalMetric, ...
        querySampleMetric)
    Algorithm.metric.region_gan_query_tracker = ...
        RegionQueryTracker_RC('export',QueryTracker);
    Algorithm.metric.(trackerMetric) = ...
        Algorithm.metric.region_gan_query_tracker;
    Algorithm.metric.region_gan_query_samples = ...
        exportRegionQuerySamples(QuerySampleLog);
    Algorithm.metric.(querySampleMetric) = ...
        Algorithm.metric.region_gan_query_samples;
    Algorithm.metric.region_gan_final = regionFinalPopulationStruct( ...
        Population1,Population2,Problem.M,Problem.D);
    Algorithm.metric.(finalMetric) = Algorithm.metric.region_gan_final;
end

function Rows = makeRegionQuerySampleRows(gen,FE,Refs,Groups, ...
        InP1,InP2,InUnion)
    Refs = double(Refs(:));
    Groups = double(Groups(:));
    InP1 = double(InP1(:));
    InP2 = double(InP2(:));
    InUnion = double(InUnion(:));
    n = numel(Groups);
    if numel(Refs) ~= n || numel(InP1) ~= n || ...
            numel(InP2) ~= n || numel(InUnion) ~= n
        error('CBSRegionGAN:BadQuerySampleLogRows', ...
            'Query sample log inputs must have equal rows.');
    end
    Rows = [repmat(double(gen),n,1),repmat(double(FE),n,1), ...
        (1:n)',Refs,Groups,InP1,InP2,InUnion];
end

function Export = exportRegionQuerySamples(QuerySampleLog)
    if isempty(QuerySampleLog)
        Rows = zeros(0,8);
    else
        Rows = vertcat(QuerySampleLog{:});
    end
    Export = struct( ...
        'columns',["generation","fe","event_sample_index", ...
            "query_ref","query_group","survive_P1", ...
            "survive_P2","survive_union"], ...
        'rows',Rows);
end

function Final = regionFinalPopulationStruct(Population1,Population2,M,D)
    [P1Dec,P1Obj,P1Con] = finalPopulationArrays(Population1,M,D);
    [P2Dec,P2Obj,P2Con] = finalPopulationArrays(Population2,M,D);
    if isempty(P1Con)
        feasible = true(size(P1Obj,1),1);
    else
        feasible = sum(max(0,P1Con),2) <= 0;
    end
    feasible = feasible & all(isfinite(P1Obj),2);
    feasibleRows = find(feasible);
    ndRows = zeros(0,1);
    if ~isempty(feasibleRows)
        FrontNo = NDSort(P1Obj(feasibleRows,:),1);
        ndRows = feasibleRows(FrontNo == 1);
    end
    Final = struct( ...
        'p1_dec',P1Dec, ...
        'p1_obj',P1Obj, ...
        'p1_con',P1Con, ...
        'p2_dec',P2Dec, ...
        'p2_obj',P2Obj, ...
        'p2_con',P2Con, ...
        'p1_feasible_nd_dec',P1Dec(ndRows,:), ...
        'p1_feasible_nd_obj',P1Obj(ndRows,:), ...
        'p1_feasible_nd_con',P1Con(ndRows,:));
end

function [Dec,Obj,Con] = finalPopulationArrays(Population,M,D)
    if isempty(Population)
        Dec = zeros(0,D);
        Obj = zeros(0,M);
        Con = zeros(0,0);
        return;
    end
    Dec = double(Population.decs);
    Obj = double(Population.objs);
    try
        Con = double(Population.cons);
    catch
        Con = zeros(size(Dec,1),0);
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
        'trainSource',zeros(0,1), ...
        'trainAge',zeros(0,1), ...
        'trainFrontRank',zeros(0,1), ...
        'queryConditions',zeros(0,0), ...
        'queryRegions',zeros(0,1), ...
        'allQueryConditions',zeros(0,0), ...
        'allQueryRegions',zeros(0,1));
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
    TrueDiag = RunRegionGAN_RC('trueboundarydiagnostics', ...
        GeneratedObj,GeneratedCon,regionProblemPF(Problem),Options);
    Metrics = copyStructFields(Metrics,TrueDiag);
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
        'random_feasible_rate',NaN, ...
        'bdist50_true',NaN, ...
        'bwidth90_10_true',NaN, ...
        'bcover_eps_true',NaN);
end

function PF = regionProblemPF(Problem)
    try
        PF = Problem.PF;
    catch
        try
            PF = Problem.GetPF();
        catch
            PF = [];
        end
    end
end

function S = copyStructFields(S,Extra)
    if ~isstruct(Extra)
        return;
    end
    names = fieldnames(Extra);
    for i = 1 : numel(names)
        S.(names{i}) = Extra.(names{i});
    end
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

function value = meanFiniteLocal(X)
    X = X(isfinite(X));
    if isempty(X)
        value = NaN;
    else
        value = mean(X);
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

function Diag = emptyRegionDiag(zDim,gpLambda,nCritic)
    Diag = struct( ...
        'generation',0, ...
        'fe',0, ...
        'zDim',double(zDim), ...
        'gan_iter_used',NaN, ...
        'gpLambda',double(gpLambda), ...
        'nCritic',double(nCritic), ...
        'train_z_sigma',1.0, ...
        'sample_z_sigma',1.0, ...
        'structured_z_enabled',0, ...
        'structured_z_max_modes',5, ...
        'structured_z_lambda',NaN, ...
        'mode_decoder_update_count',0, ...
        'mode_sample_count',0, ...
        'mode_multi_active_sample_count',0, ...
        'mode_sample_decoder_accuracy',NaN, ...
        'mode_sample_confidence_mean',NaN, ...
        'mode_sample_label_entropy',NaN, ...
        'mode_sample_prediction_entropy',NaN, ...
        'mode_active_count_mean',NaN, ...
        'mode_active_count_max',NaN, ...
        'generated_critic_score_count',0, ...
        'generated_critic_score_min',NaN, ...
        'generated_critic_score_max',NaN, ...
        'generated_critic_score_mean',NaN, ...
        'generated_critic_score',zeros(0,1), ...
        'bmem_count',0, ...
        'region_count',0, ...
        'points_per_region_median',0, ...
        'points_per_region_max',0, ...
        'prev_bmem_candidate_count',0, ...
        'prev_bmem_survivor_count',0, ...
        'median_gap',NaN, ...
        'max_gap',NaN, ...
        'train_count',0, ...
        'train_count_raw',0, ...
        'train_exact_duplicate_count',0, ...
        'train_removed_duplicate_count',0, ...
        'train_duplicate_fraction',NaN, ...
        'train_dedup_enabled',0, ...
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
        'bdist50_true',NaN, ...
        'bwidth90_10_true',NaN, ...
        'bcover_eps_true',NaN, ...
        'train_width50',NaN, ...
        'train_width90',NaN, ...
        'train_width_count',0, ...
        'gen_width50',NaN, ...
        'gen_width90',NaN, ...
        'gen_width_count',0, ...
        'gen_to_train_dist50',NaN, ...
        'gen_to_train_dist90',NaN, ...
        'gen_to_train_dist_count',0, ...
        'gen_to_train_dec_dist50',NaN, ...
        'gen_to_train_dec_dist90',NaN, ...
        'offspringG_count',0, ...
        'offspringG_survive_count',0, ...
        'offspringG_survival_rate',NaN, ...
        'offspringG_feasible_count',0, ...
        'offspringG_feasible_survive_count',0, ...
        'offspringG_feasible_survival_rate',NaN, ...
        'critic_train_gap',NaN, ...
        'critic_holdout_gap',NaN, ...
        'critic_train_real_score_mean',NaN, ...
        'critic_train_fake_score_mean',NaN, ...
        'critic_holdout_real_score_mean',NaN, ...
        'critic_holdout_fake_score_mean',NaN, ...
        'critic_train_diag_count',0, ...
        'critic_holdout_count',0, ...
        'prequential_new_count',0, ...
        'prequential_seen_count',0, ...
        'prequential_pre_critic_gap',NaN, ...
        'prequential_post_critic_gap',NaN, ...
        'prequential_pre_dec_dist50',NaN, ...
        'prequential_pre_dec_dist90',NaN, ...
        'prequential_post_dec_dist50',NaN, ...
        'prequential_post_dec_dist90',NaN, ...
        'last_critic_loss',NaN, ...
        'last_generator_loss',NaN, ...
        'last_generator_adversarial_loss',NaN, ...
        'last_mi_loss',NaN, ...
        'last_mode_decoder_accuracy',NaN, ...
        'last_gradient_penalty',NaN, ...
        'last_score_real',NaN, ...
        'last_score_fake',NaN, ...
        'condition_diag_condition_count',0, ...
        'condition_diag_z_count',0, ...
        'same_z_diff_c_dec_median',NaN, ...
        'same_c_diff_z_dec_median',NaN, ...
        'same_c_diff_z_collapse_rate',NaN, ...
        'condition_effect_ratio_dec',NaN, ...
        'gan_train_history',struct([]));
    Diag = copyStructFields(Diag,emptyRegionBMemLearnabilityMetrics());
    Diag = copyStructFields(Diag,emptyRegionLatentScaleMetrics());
    Diag = addAttributionDiagDefaults(Diag);
end

function Metrics = emptyRegionLatentScaleMetrics()
    Metrics = struct( ...
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

function Diag = addAttributionDiagDefaults(Diag)
    groupNames = ["populated","frontier","remote"];
    countNames = ["sample_count","unique_ref_count","feasible_count", ...
        "survive_P1_count","survive_P2_count","survive_union_count"];
    rateNames = ["feasible_rate","survive_P1_rate", ...
        "survive_P2_rate","survive_union_rate"];
    geometryNames = ["bdist50_true","bwidth90_10_true", ...
        "bcover_eps_true","gen_to_train_dec_dist50", ...
        "gen_to_train_dec_dist90","gen_to_train_obj_dist50", ...
        "gen_to_train_obj_dist90","pair_normal_abs50", ...
        "pair_normal_abs90","pair_tangent_dist50", ...
        "pair_tangent_dist90"];
    for i = 1 : numel(groupNames)
        prefix = "query_" + groupNames(i) + "_";
        for j = 1 : numel(countNames)
            Diag.(char(prefix + countNames(j))) = 0;
        end
        for j = 1 : numel(rateNames)
            Diag.(char(prefix + rateNames(j))) = NaN;
        end
        for j = 1 : numel(geometryNames)
            Diag.(char(prefix + geometryNames(j))) = NaN;
        end
    end
    Diag.offspringG_survive_P1_count = 0;
    Diag.offspringG_survive_P1_rate = NaN;
    Diag.offspringG_survive_P2_count = 0;
    Diag.offspringG_survive_P2_rate = NaN;
    Diag.offspringG_survive_union_count = 0;
    Diag.offspringG_survive_union_rate = NaN;
    Diag.bmem_bdist50_true = NaN;
    Diag.bmem_bwidth90_10_true = NaN;
    Diag.bmem_bcover_eps_true = NaN;
    Diag.ref_conversion_new_count = 0;
    Diag.ref_conversion_new_lag_gen50 = NaN;
    Diag.ref_conversion_new_lag_fe50 = NaN;
    Diag.direct_bmem_entry_new_count = 0;
    Diag.direct_bmem_entry_new_lag_gen50 = NaN;
    Diag.direct_bmem_entry_new_lag_fe50 = NaN;
    Diag.tracker_ref_exposure_count = 0;
    Diag.tracker_ref_frontier_exposure_count = 0;
    Diag.tracker_ref_remote_exposure_count = 0;
    Diag.tracker_ref_exposure_event_count = 0;
    Diag.tracker_ref_generated_count = 0;
    Diag.tracker_ref_converted_count = 0;
    Diag.tracker_ref_conversion_rate = NaN;
    Diag.tracker_ref_conversion_lag_gen50 = NaN;
    Diag.tracker_ref_conversion_lag_fe50 = NaN;
    Diag.tracker_direct_eligible_sample_count = 0;
    Diag.tracker_direct_bmem_entry_count = 0;
    Diag.tracker_direct_bmem_entry_rate = NaN;
    Diag.tracker_direct_bmem_entry_lag_gen50 = NaN;
    Diag.tracker_direct_bmem_entry_lag_fe50 = NaN;
    Diag.frontier_query_ref_count = 0;
    Diag.frontier_matched_de_available = 0;
    Diag.frontier_matched_de_available_ref_count = 0;
    Diag.frontier_matched_de_unavailable_ref_count = 0;
    matchedPrefixes = ["frontier_matched_gan_","frontier_matched_de_"];
    matchedCounts = ["sample_count","feasible_count", ...
        "survive_P1_count","survive_P2_count","survive_union_count"];
    matchedRates = ["feasible_rate","survive_P1_rate", ...
        "survive_P2_rate","survive_union_rate"];
    matchedGeometry = ["bdist50_true","bwidth90_10_true", ...
        "bcover_eps_true"];
    matchedRefEqual = ["refeq_feasible_rate","refeq_survive_P1_rate", ...
        "refeq_survive_P2_rate","refeq_survive_union_rate", ...
        "refeq_bdist50_true","refeq_bwidth90_10_true", ...
        "refeq_bcover_eps_true"];
    for i = 1 : numel(matchedPrefixes)
        for j = 1 : numel(matchedCounts)
            Diag.(char(matchedPrefixes(i) + matchedCounts(j))) = 0;
        end
        for j = 1 : numel(matchedRates)
            Diag.(char(matchedPrefixes(i) + matchedRates(j))) = NaN;
        end
        for j = 1 : numel(matchedGeometry)
            Diag.(char(matchedPrefixes(i) + matchedGeometry(j))) = NaN;
        end
        for j = 1 : numel(matchedRefEqual)
            Diag.(char(matchedPrefixes(i) + matchedRefEqual(j))) = NaN;
        end
    end
    Diag.time_bmem_dataset_query = 0;
    Diag.time_gan_train_sample = 0;
    Diag.time_gan_evaluation = 0;
    Diag.time_selection_P1 = 0;
    Diag.time_selection_P2 = 0;
    Diag.time_diagnostics_serialization = 0;
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
    Snapshot.gan_iter_used = double(Diag.gan_iter_used);
    Snapshot.train_z_sigma = double(Diag.train_z_sigma);
    Snapshot.sample_z_sigma = double(Diag.sample_z_sigma);
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
    Snapshot.bdist50_true = double(Diag.bdist50_true);
    Snapshot.bwidth90_10_true = double(Diag.bwidth90_10_true);
    Snapshot.bcover_eps_true = double(Diag.bcover_eps_true);
    Snapshot.train_width50 = double(Diag.train_width50);
    Snapshot.train_width90 = double(Diag.train_width90);
    Snapshot.gen_width50 = double(Diag.gen_width50);
    Snapshot.gen_width90 = double(Diag.gen_width90);
    Snapshot.gen_to_train_dist50 = double(Diag.gen_to_train_dist50);
    Snapshot.gen_to_train_dist90 = double(Diag.gen_to_train_dist90);
    Snapshot.offspringG_survival_rate = ...
        double(Diag.offspringG_survival_rate);
    Snapshot.critic_train_gap = double(Diag.critic_train_gap);
    Snapshot.critic_holdout_gap = double(Diag.critic_holdout_gap);
    Snapshot.generated_critic_score_mean = ...
        double(Diag.generated_critic_score_mean);

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
        'gan_iter_used',NaN, ...
        'train_z_sigma',1.0, ...
        'sample_z_sigma',1.0, ...
        'train_count',0, ...
        'prev_bmem_candidate_count',0, ...
        'prev_bmem_survivor_count',0, ...
        'raw_generated_count',0, ...
        'feasible_rate',NaN, ...
        'gap_ratio50',NaN, ...
        'gap_ratio90',NaN, ...
        'near_boundary_rate_gap1',NaN, ...
        'better_than_random_gap_rate',NaN, ...
        'bdist50_true',NaN, ...
        'bwidth90_10_true',NaN, ...
        'bcover_eps_true',NaN, ...
        'train_width50',NaN, ...
        'train_width90',NaN, ...
        'gen_width50',NaN, ...
        'gen_width90',NaN, ...
        'gen_to_train_dist50',NaN, ...
        'gen_to_train_dist90',NaN, ...
        'offspringG_survival_rate',NaN, ...
        'critic_train_gap',NaN, ...
        'critic_holdout_gap',NaN, ...
        'generated_critic_score_mean',NaN, ...
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
