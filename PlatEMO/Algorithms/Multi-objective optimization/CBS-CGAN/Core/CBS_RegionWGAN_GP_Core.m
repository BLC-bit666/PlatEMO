classdef CBS_RegionWGAN_GP_Core < ALGORITHM
%CBS_REGIONWGAN_GP_CORE Shared implementation for CBS-CGAN GUI entries.
% Reference vector-conditioned boundary WGAN-GP
% Before the CGAN cutoff, the production path uses 25% GA, 55% ordinary
% DE, and 20% all-reference critic-filtered CGAN candidates mapped through
% the local A/h/T target path. Random20, DE20, and GA20 replace only the
% last share. Afterwards
% every path uses certified feasible boundary targets for the remaining
% 20%. Missing targets fall back to ordinary DE.
% The unconstrained population uses 25% GA and 75% ordinary DE throughout.
% rawGuideCount    --- 500 --- Raw all-reference CGAN queries per event
% zDim             ---   6 --- Dimension of the generator noise vector
% ganIter          --- 100 --- Generator updates per training event
% ganMiniBatch     ---  32 --- Mini-batch size of WGAN-GP training
% nCritic          ---   4 --- Critic updates per generator update
% minGANTrainCount ---  32 --- Minimum conditioned rows required for training
% sampleSigma      --- 0.3 --- Standard deviation of generator sampling noise

%------------------------------- Reference --------------------------------
% [1] Y. Tian, T. Zhang, J. Xiao, X. Zhang, and Y. Jin. A coevolutionary
% framework for constrained multi-objective optimization problems. IEEE
% Transactions on Evolutionary Computation, 2021, 25(1): 102-116.
% [2] I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. Courville.
% Improved training of Wasserstein GANs. Advances in Neural Information
% Processing Systems, 2017, 30.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    properties(Access = private)
        BoundaryTargetXf = [];
        BoundaryTargetYf = [];
        GuideStats = struct();
        DiagnosticOptions = struct();
    end

    methods
        function Algorithm = CBS_RegionWGAN_GP_Core(varargin)
            Algorithm@ALGORITHM(varargin{:});
        end

        function main(Algorithm,Problem)
            configurePlatEMOUtilityPath();
            Config = Algorithm.algorithmConfiguration();
            Config = mergeCutoffDiagnosticOptions( ...
                Config,Algorithm.DiagnosticOptions);
            Algorithm.runMainline(Problem,Config);
        end

        function configureCutoffDiagnostics(Algorithm,Options)
        %CONFIGURECUTOFFDIAGNOSTICS Enable behavior-neutral observations.
            Algorithm.DiagnosticOptions = ...
                validateCutoffDiagnosticOptions(Options);
        end

        function Snapshot = guideExperimentSnapshot(Algorithm)
        %GUIDEEXPERIMENTSNAPSHOT Return aggregate CGAN mechanism evidence.
            Snapshot = finalizeGuideStats(Algorithm.GuideStats);
        end
    end

    methods(Access = protected)
        function Config = algorithmConfiguration(Algorithm)
        %ALGORITHMCONFIGURATION Parse the production configuration.
            Defaults = CBS_RegionWGAN_GP_Core.mainlineDefaults();
            [rawGuideCount,zDim,ganIter,ganMiniBatch,nCritic, ...
                minGANTrainCount,sampleSigma] = ...
                Algorithm.ParameterSet( ...
                Defaults.rawGuideCount,Defaults.zDim,Defaults.ganIter, ...
                Defaults.ganMiniBatch,Defaults.nCritic, ...
                Defaults.minGANTrainCount,Defaults.sampleSigma);
            Config = Defaults;
            Config.rawGuideCount = rawGuideCount;
            Config.zDim = zDim;
            Config.ganIter = ganIter;
            Config.ganMiniBatch = ganMiniBatch;
            Config.nCritic = nCritic;
            Config.minGANTrainCount = minGANTrainCount;
            Config.sampleSigma = sampleSigma;
            % -1 distinguishes production from frozen historical arm A0.
            Config.experimentArm = -1;
            Config.guideGenerationMode = "global_critic";
            Config.guideUseMode = "local_target";
        end
    end

    methods(Static)
        function Defaults = mainlineDefaults()
        %MAINLINEDEFAULTS Return public parameters and fixed mainline values.
            Defaults = struct( ...
                'rawGuideCount',500, ...
                'criticKeepCount',200, ...
                'legacyGuideSlots',20, ...
                'legacyCandidatesPerSlot',5, ...
                'zDim',6, ...
                'ganIter',100, ...
                'ganMiniBatch',32, ...
                'ganLrD',1e-4, ...
                'ganLrG',1e-4, ...
                'frontDepth',2, ...
                'pairNeighborRefRadius',2, ...
                'refDivisor',2, ...
                'minBoundaryLength',2, ...
                'gpLambda',10, ...
                'nCritic',4, ...
                'maxAnchorsPerRef',5, ...
                'minGANTrainCount',32, ...
                'sampleSigma',0.3, ...
                'ganStopFraction',0.5, ...
                'calibrationBudget',20, ...
                'generatorHidden',[32 32], ...
                'criticHidden',[32 32], ...
                'keepUnpairedAnchors',false, ...
                'pairNeighborRefCount',5, ...
                'trainGateMode',"total", ...
                'minPositiveTrainCount',16, ...
                'minNegativeTrainCount',8, ...
                'minTrainRefCount',4, ...
                'batchSamplingMode',"uniform", ...
                'parentSourceMode',"population", ...
                'mechanismAudit',false, ...
                'cutoffDiagnosticsEnabled',false, ...
                'stopAtCGANEnd',false, ...
                'disableOracleAudit',false);
        end
    end

    methods(Access = private)
        function recordBoundaryBrackets(Algorithm,Brackets)
        %RECORDBOUNDARYBRACKETS Store certified feasible boundary targets.
            if ~isstruct(Brackets) || ~isfield(Brackets,'xf') || ...
                    ~isfield(Brackets,'yf') || isempty(Brackets.xf)
                return;
            end
            valid = all(isfinite(Brackets.xf),2) & ...
                all(isfinite(Brackets.yf),2);
            Algorithm.BoundaryTargetXf = [Algorithm.BoundaryTargetXf; ...
                Brackets.xf(valid,:)];
            Algorithm.BoundaryTargetYf = [Algorithm.BoundaryTargetYf; ...
                Brackets.yf(valid,:)];
            cap = 1500;
            excess = size(Algorithm.BoundaryTargetXf,1)-cap;
            if excess > 0
                Algorithm.BoundaryTargetXf(1:excess,:) = [];
                Algorithm.BoundaryTargetYf(1:excess,:) = [];
            end
        end

        function [GuideDecs,GuideRefs,GuideSlots,GuideObjs,strictRefs] = ...
                guidePoolAtFE( ...
                Algorithm,GuideDecs,GuideRefs,GuideSlots,currentFE, ...
                ganFELimit,Problem)
        %GUIDEPOOLATFE Switch from generated to certified boundary targets.
            GuideObjs = zeros(0,Problem.M);
            strictRefs = false;
            if double(currentFE) < double(ganFELimit)
                return;
            end

            strictRefs = true;
            if isempty(Algorithm.BoundaryTargetXf)
                GuideDecs = zeros(0,Problem.D);
                GuideRefs = zeros(0,1);
                GuideSlots = zeros(0,1);
                return;
            end
            newestFirst = size(Algorithm.BoundaryTargetXf,1):-1:1;
            GuideDecs = Algorithm.BoundaryTargetXf(newestFirst,:);
            GuideObjs = Algorithm.BoundaryTargetYf(newestFirst,:);
            GuideRefs = zeros(numel(newestFirst),1);
            GuideSlots = (1:numel(newestFirst))';
        end

        function recordMemoryTrace(Algorithm,Trace,currentFE)
        %RECORDMEMORYTRACE Aggregate one boundary-memory construction event.
            if isempty(Trace) || ~isstruct(Trace)
                return;
            end
            S = Algorithm.GuideStats;
            S.memoryEvents = S.memoryEvents+1;
            names = ["trueFeasible","afterFront","frontDropped", ...
                "frontOpportunityRefs","afterCap","capDropped", ...
                "retained","pairedBeforeMAD","unpairedBeforeMAD", ...
                "paired","unpaired","madDropped","legalWithin5", ...
                "legalWithin10","legalAny","dominanceRejected", ...
                "pairRank1To5","pairRank6To10","pairRankOver10", ...
                "previousUnpaired","previousUnpairedPaired"];
            for name = names
                S.(name+"Sum") = S.(name+"Sum")+double(Trace.(name));
            end
            finiteNames = ["pairGapMedian","pairGapP90", ...
                "pairAngleMedian","pairAngleP90"];
            for name = finiteNames
                if isfinite(Trace.(name))
                    S.(name+"Sum") = S.(name+"Sum")+Trace.(name);
                    S.(name+"Count") = S.(name+"Count")+1;
                end
            end
            S.frontOpportunityEvents = S.frontOpportunityEvents+ ...
                (Trace.frontOpportunityRefs > 0);
            S.capActiveEvents = S.capActiveEvents+(Trace.capDropped > 0);
            if isinf(S.firstEligibleAnchorFE) && Trace.afterCap > 0
                S.firstEligibleAnchorFE = double(currentFE);
            end
            if isinf(S.firstLegalPairFE) && Trace.paired > 0
                S.firstLegalPairFE = double(currentFE);
            end
            Algorithm.GuideStats = S;
        end

        function recordTrainingTrace(Algorithm,TrainC,QueryRefs,eligible,currentFE)
        %RECORDTRAININGTRACE Aggregate one possible WGAN training event.
            S = Algorithm.GuideStats;
            S.dataEvents = S.dataEvents+1;
            if isempty(TrainC)
                positive = 0;
                negative = 0;
            else
                positive = nnz(TrainC(:,end) >= 0.5);
                negative = size(TrainC,1)-positive;
            end
            refs = numel(unique(QueryRefs));
            S.positiveRowsSum = S.positiveRowsSum+positive;
            S.negativeRowsSum = S.negativeRowsSum+negative;
            S.trainingRefsSum = S.trainingRefsSum+refs;
            unsafe = size(TrainC,1) >= 32 && ...
                (positive < 16 || negative < 8 || refs < 4);
            S.unsafeTrainingEvents = S.unsafeTrainingEvents+unsafe;
            imbalanced = positive > 0 && negative > 0 && ...
                (positive/negative > 2 || positive/negative < 0.5);
            S.imbalancedDataEvents = S.imbalancedDataEvents+imbalanced;
            if eligible
                S.trainingEvents = S.trainingEvents+1;
                S.imbalancedTrainingEvents = ...
                    S.imbalancedTrainingEvents+imbalanced;
                if isinf(S.firstTrainingFE)
                    S.firstTrainingFE = double(currentFE);
                end
            else
                S.trainingBlockedEvents = S.trainingBlockedEvents+1;
            end
            Algorithm.GuideStats = S;
        end

        function recordGenerationTrace(Algorithm,Trace,currentFE)
        %RECORDGENERATIONTRACE Aggregate one eligible CGAN query event.
            if isempty(Trace) || ~isstruct(Trace) || ~Trace.active
                return;
            end
            S = Algorithm.GuideStats;
            S.generationEvents = S.generationEvents+1;
            S.rawCandidates = S.rawCandidates+Trace.rawCount;
            S.criticKept = S.criticKept+Trace.keptCount;
            S.rawConditionSum = S.rawConditionSum+Trace.rawConditions;
            S.keptConditionSum = S.keptConditionSum+Trace.keptConditions;
            kept = Trace.percentile(Trace.keepIdx);
            rejected = Trace.percentile;
            rejected(Trace.keepIdx) = [];
            kept = kept(isfinite(kept));
            rejected = rejected(isfinite(rejected));
            S.keptPercentileSum = S.keptPercentileSum+sum(kept);
            S.keptPercentileCount = S.keptPercentileCount+numel(kept);
            S.rejectedPercentileSum = ...
                S.rejectedPercentileSum+sum(rejected);
            S.rejectedPercentileCount = ...
                S.rejectedPercentileCount+numel(rejected);
            S = accumulateBoundarySummary(S,"raw", ...
                Trace.rawBoundaryDistance,Trace.rawBoundaryBand, ...
                Trace.rawBoundarySupported);
            S = accumulateBoundarySummary(S,"kept", ...
                Trace.keptBoundaryDistance,Trace.keptBoundaryBand, ...
                Trace.keptBoundarySupported);
            S = accumulateBoundarySummary(S,"rejected", ...
                Trace.rejectedBoundaryDistance, ...
                Trace.rejectedBoundaryBand, ...
                Trace.rejectedBoundarySupported);
            S = accumulateFiniteScalar(S,"criticBoundarySpearman", ...
                Trace.criticBoundarySpearman);
            S.criticBoundaryPairCount = S.criticBoundaryPairCount+ ...
                Trace.criticBoundaryPairCount;
            S = accumulateFiniteScalar(S,"rawDirectionCoverage", ...
                Trace.rawDirectionCoverage);
            S = accumulateFiniteScalar(S,"rawDirectionEntropy", ...
                Trace.rawDirectionEntropy);
            S = accumulateFiniteScalar(S,"rawNearDuplicateRate", ...
                Trace.rawNearDuplicateRate);
            S = accumulateFiniteScalar(S,"keptDirectionCoverage", ...
                Trace.keptDirectionCoverage);
            S = accumulateFiniteScalar(S,"keptDirectionEntropy", ...
                Trace.keptDirectionEntropy);
            S = accumulateFiniteScalar(S,"keptNearDuplicateRate", ...
                Trace.keptNearDuplicateRate);
            S.rawOracleCount = S.rawOracleCount+Trace.rawOracleCount;
            S.rawOracleFeasible = ...
                S.rawOracleFeasible+Trace.rawOracleFeasible;
            S.trainConditionSum = ...
                S.trainConditionSum+Trace.trainConditions;
            S.rawSupportedCount = ...
                S.rawSupportedCount+Trace.rawSupportedCount;
            S.rawSupportedFeasible = ...
                S.rawSupportedFeasible+Trace.rawSupportedFeasible;
            S.rawUnsupportedCount = ...
                S.rawUnsupportedCount+Trace.rawUnsupportedCount;
            S.rawUnsupportedFeasible = ...
                S.rawUnsupportedFeasible+Trace.rawUnsupportedFeasible;
            S.rawReferenceMatch = ...
                S.rawReferenceMatch+Trace.rawReferenceMatch;
            S.rawSupportedReferenceMatch = ...
                S.rawSupportedReferenceMatch+ ...
                Trace.rawSupportedReferenceMatch;
            S.rawUnsupportedReferenceMatch = ...
                S.rawUnsupportedReferenceMatch+ ...
                Trace.rawUnsupportedReferenceMatch;
            if isinf(S.firstGuidePoolFE) && Trace.keptCount > 0
                S.firstGuidePoolFE = double(currentFE);
            end
            Algorithm.GuideStats = S;
        end

        function recordUseTrace(Algorithm,Trace,SelectedPopulation,currentFE)
        %RECORDUSETRACE Aggregate evaluated CGAN child mechanisms.
            if isempty(Trace) || ~isstruct(Trace) || ~Trace.active
                return;
            end
            S = Algorithm.GuideStats;
            S.useEvents = S.useEvents+1;
            S.guidedRequested = S.guidedRequested+Trace.requested;
            S.guidedSelected = S.guidedSelected+Trace.selected;
            S.guidedFallback = S.guidedFallback+Trace.fallback;
            S.fallbackNoPool = S.fallbackNoPool+Trace.fallbackNoPool;
            S.fallbackNoCurrentFeasible = S.fallbackNoCurrentFeasible+ ...
                Trace.fallbackNoCurrentFeasible;
            S.fallbackInvalidScale = ...
                S.fallbackInvalidScale+Trace.fallbackInvalidScale;
            S.fallbackMapping = S.fallbackMapping+Trace.fallbackMapping;
            S.memoryParentsAdded = ...
                S.memoryParentsAdded+Trace.memoryParentsAdded;
            S.memoryParentsUsed = ...
                S.memoryParentsUsed+Trace.memoryParentsUsed;
            S.selectedConditionSum = ...
                S.selectedConditionSum+Trace.selectedConditions;
            S.mappedValid = S.mappedValid+Trace.mappedValid;
            S.mapDropped = S.mapDropped+Trace.mapDropped;
            S.alphaSum = S.alphaSum+sum(Trace.alpha);
            S.alphaCount = S.alphaCount+numel(Trace.alpha);
            S.hSum = S.hSum+sum(Trace.h);
            S.dSum = S.dSum+sum(Trace.d);
            S.centerStepSum = S.centerStepSum+sum(Trace.centerStep);
            S.actualStepSum = S.actualStepSum+sum(Trace.actualStep);
            cosine = Trace.directionCosine(isfinite(Trace.directionCosine));
            S.directionCosineSum = S.directionCosineSum+sum(cosine);
            S.directionCosineCount = S.directionCosineCount+numel(cosine);
            S.guidedChildren = S.guidedChildren+Trace.selected;
            S.guidedFeasible = S.guidedFeasible+Trace.feasibleChildren;
            S.guidedDominating = S.guidedDominating+Trace.dominatingChildren;
            S.selectedTargetCount = ...
                S.selectedTargetCount+Trace.selectedTargetCount;
            S.selectedTargetFeasible = ...
                S.selectedTargetFeasible+Trace.selectedTargetFeasible;
            S.selectedTargetUseful = ...
                S.selectedTargetUseful+Trace.selectedTargetUseful;
            S.centerFeasible = S.centerFeasible+Trace.centerFeasible;
            S.centerUseful = S.centerUseful+Trace.centerUseful;
            S.childUseful = S.childUseful+Trace.childUseful;
            S.targetFeasibleLostAtCenter = ...
                S.targetFeasibleLostAtCenter+ ...
                Trace.targetFeasibleLostAtCenter;
            S.targetInfeasibleRecoveredAtCenter = ...
                S.targetInfeasibleRecoveredAtCenter+ ...
                Trace.targetInfeasibleRecoveredAtCenter;
            S.centerFeasibleLostAtMutation = ...
                S.centerFeasibleLostAtMutation+ ...
                Trace.centerFeasibleLostAtMutation;
            S.centerInfeasibleRecoveredAtMutation = ...
                S.centerInfeasibleRecoveredAtMutation+ ...
                Trace.centerInfeasibleRecoveredAtMutation;
            S = accumulateBoundarySummary(S,"selected", ...
                Trace.selectedBoundaryDistance, ...
                Trace.selectedBoundaryBand, ...
                Trace.selectedBoundarySupported);
            S = accumulateBoundarySummary(S,"center", ...
                Trace.centerBoundaryDistance,Trace.centerBoundaryBand, ...
                Trace.centerBoundarySupported);
            S = accumulateBoundarySummary(S,"child", ...
                Trace.childBoundaryDistance,Trace.childBoundaryBand, ...
                Trace.childBoundarySupported);
            S = accumulateFiniteScalar(S,"selectedDirectionCoverage", ...
                Trace.selectedDirectionCoverage);
            S = accumulateFiniteScalar(S,"selectedDirectionEntropy", ...
                Trace.selectedDirectionEntropy);
            S = accumulateFiniteScalar(S,"selectedNearDuplicateRate", ...
                Trace.selectedNearDuplicateRate);
            S = accumulateDistanceChange(S,"selectedToCenter", ...
                Trace.selectedBoundaryDistance,Trace.centerBoundaryDistance);
            S = accumulateDistanceChange(S,"centerToChild", ...
                Trace.centerBoundaryDistance,Trace.childBoundaryDistance);
            if ~isempty(Trace.childDecs)
                survived = ismember(Trace.childDecs, ...
                    double(SelectedPopulation.decs),'rows');
                S.guidedSurvived = S.guidedSurvived+sum(survived);
            end
            if isinf(S.firstGuidedUseFE) && Trace.selected > 0
                S.firstGuidedUseFE = double(currentFE);
            end
            if isinf(S.firstUsefulGuidedFE) && Trace.childUseful > 0
                S.firstUsefulGuidedFE = double(currentFE);
            end
            Algorithm.GuideStats = S;
        end

        function nofinish = auditNotTerminated(Algorithm,Population,Problem,W)
        %AUDITNOTTERMINATED Save behavior-neutral population checkpoints.
            S = Algorithm.GuideStats;
            feasible = sum(max(0,Population.cons),2) <= 0;
            if isinf(S.firstPopulationFeasibleFE) && any(feasible)
                S.firstPopulationFeasibleFE = double(Problem.FE);
            end
            rows = find(isnan(S.checkpointFE) & ...
                S.checkpointTargets <= double(Problem.FE));
            diagnose = cutoffDiagnosticsEnabled(S);
            firstObservation = diagnose && isnan(S.initialFE);
            cutoffObservation = diagnose && isnan(S.cganEndFE) && ...
                double(Problem.FE) >= double(S.cganEndTarget);
            if ~isempty(rows) || firstObservation || cutoffObservation
                Metrics = populationDiagnosticMetrics( ...
                    Population,Problem,W,diagnose);
            else
                Metrics = struct();
            end
            if ~isempty(rows)
                S.checkpointFE(rows) = double(Problem.FE);
                S.checkpointFeasibleCount(rows) = Metrics.feasibleCount;
                S.checkpointNondominatedFeasibleCount(rows) = ...
                    Metrics.nondominatedFeasibleCount;
                S.checkpointRefCoverage(rows) = Metrics.refCoverage;
                S.checkpointRefEntropy(rows) = Metrics.refEntropy;
                S.checkpointRefOccupancyCV(rows) = Metrics.refOccupancyCV;
                S.checkpointTrainingEvents(rows) = S.trainingEvents;
                S.checkpointGuidedSelected(rows) = S.guidedSelected;
                S.checkpointGuidedFeasible(rows) = S.guidedFeasible;
                S.checkpointGuidedSurvived(rows) = S.guidedSurvived;
                if diagnose
                    S.checkpointIGD(rows) = Metrics.igd;
                    S.checkpointHV(rows) = Metrics.hv;
                end
            end
            if firstObservation
                S.initialFE = double(Problem.FE);
                S.initialIGD = Metrics.igd;
                S.initialHV = Metrics.hv;
                S.initialFeasibleCount = Metrics.feasibleCount;
                S.initialNondominatedFeasibleCount = ...
                    Metrics.nondominatedFeasibleCount;
                S.initialRefCoverage = Metrics.refCoverage;
                S.initialRefEntropy = Metrics.refEntropy;
                S.initialRefOccupancyCV = Metrics.refOccupancyCV;
            end
            if cutoffObservation
                S.cganEndFE = double(Problem.FE);
                S.cganEndIGD = Metrics.igd;
                S.cganEndHV = Metrics.hv;
                S.cganEndFeasibleCount = Metrics.feasibleCount;
                S.cganEndNondominatedFeasibleCount = ...
                    Metrics.nondominatedFeasibleCount;
                S.cganEndRefCoverage = Metrics.refCoverage;
                S.cganEndRefEntropy = Metrics.refEntropy;
                S.cganEndRefOccupancyCV = Metrics.refOccupancyCV;
                S.cganEndDecs = double(Population.decs);
                S.cganEndObjs = double(Population.objs);
                S.cganEndCons = double(Population.cons);
            end
            Algorithm.GuideStats = S;
            Algorithm.metric.CBSAudit = finalizeGuideStats(S, ...
                cutoffObservation || double(Problem.FE) >= ...
                double(Problem.maxFE));
            nofinish = Algorithm.NotTerminated(Population);
            if cutoffObservation && S.stopAtCGANEnd
                error('PlatEMO:Termination','');
            end
        end

        function runMainline(Algorithm,Problem,Config)
        %RUNMAINLINE Execute the unique pairflag CGAN-guided search.
            Algorithm.BoundaryTargetXf = [];
            Algorithm.BoundaryTargetYf = [];
            Algorithm.GuideStats = emptyGuideStats(Config,Problem.maxFE);
            rawGuideCount = max(0,round(double(Config.rawGuideCount)));
            refDivisor = max(1,round(double(Config.refDivisor)));
            minBoundaryLength = max(1,round(double( ...
                Config.minBoundaryLength)));
            minGANTrainCount = max(1,round(double( ...
                Config.minGANTrainCount)));
            ganFELimit = double(Config.ganStopFraction)*Problem.maxFE;
            generationMode = string(Config.guideGenerationMode);
            validGenerationModes = ["legacy","global_critic","random", ...
                "traditional_de","traditional_ga"];
            if ~isscalar(generationMode) || ...
                    ~ismember(generationMode,validGenerationModes)
                error('CBSRegionGAN:BadGuideGenerationMode', ...
                    'Unsupported guide generation mode.');
            end
            usesCGAN = ismember(generationMode,["legacy","global_critic"]);

            %% Reference vectors, models, and two coevolving populations
            [W,~] = UniformPoint( ...
                max(2,round(Problem.N/refDivisor)),Problem.M);
            MemOptions = struct( ...
                'frontDepth',max(1,round(double(Config.frontDepth))), ...
                'pairNeighborRefRadius',max(0,round(double( ...
                    Config.pairNeighborRefRadius))), ...
                'pairNeighborRefCount',double(Config.pairNeighborRefCount), ...
                'keepUnpairedAnchors',logical(Config.keepUnpairedAnchors), ...
                'minBoundaryLength',minBoundaryLength, ...
                'maxAnchorsPerRef',max(1,round(double( ...
                    Config.maxAnchorsPerRef))));
            GANOptions = regionGANOptions(Config,minGANTrainCount);

            Population1 = Problem.Initialization();
            Population2 = Problem.Initialization();
            Fitness1 = CalFitness_CBS(Population1.objs,Population1.cons);
            Fitness2 = CalFitness_CBS(Population2.objs);
            BMem = [];
            GAN = [];
            GuideDecs = zeros(0,Problem.D);
            GuideRefs = zeros(0,1);
            GuideSlots = zeros(0,1);
            % Reference conditions and next-generation parents share this
            % objective normalization frame.
            GuideRefScale = [];

            %% Optimization
            while Algorithm.auditNotTerminated(Population1,Problem,W)
                generationFE = double(Problem.FE);
                plainDE = 0.55;
                if generationFE >= ganFELimit && ...
                        isempty(Algorithm.BoundaryTargetXf)
                    plainDE = 0.75;
                end
                remainingFE = max(0,Problem.maxFE-Problem.FE);
                offspringBudget = min(2*Problem.N,remainingFE);
                count1 = min(Problem.N,ceil(offspringBudget/2));
                count2 = min(Problem.N,floor(offspringBudget/2));

                [GuideDecs,GuideRefs,GuideSlots,GuideObjs,strictGuideRefs] = ...
                    Algorithm.guidePoolAtFE(GuideDecs,GuideRefs, ...
                    GuideSlots,generationFE,ganFELimit,Problem);
                [Offspring1,GuideTrace] = generateGuidedOffspring(Problem, ...
                    Population1,Fitness1,count1,GuideDecs,GuideRefs, ...
                    GuideSlots,GuideObjs,strictGuideRefs,GuideRefScale,W, ...
                    plainDE,Config,BMem);
                % A generated pool guides exactly one generation.
                GuideDecs = zeros(0,Problem.D);
                GuideRefs = zeros(0,1);
                GuideSlots = zeros(0,1);
                GuideRefScale = [];
                Offspring2 = generateBackboneOffspring( ...
                    Problem,Population2,Fitness2,count2);

                %% Active boundary calibration (real evaluations)
                Calibration = Offspring1([]);
                remainingFE = max(0,Problem.maxFE-Problem.FE);
                budget = min(double(Config.calibrationBudget),remainingFE);
                if budget > 0
                    [Calibration,CalBrackets] = ...
                        RefineBoundaryObservations_RC( ...
                        Problem,Population1,Population2,budget);
                    Algorithm.recordBoundaryBrackets(CalBrackets);
                end

                %% Pairflag training and feasible-side generation
                if Problem.FE < ganFELimit && ...
                        (usesCGAN || cutoffDiagnosticsEnabled(Config))
                    Harvest1 = Offspring1;
                    if ~isempty(Calibration)
                        Harvest1 = [Offspring1,Calibration];
                    end
                    [BMem,RefScale,MemoryTrace] = UpdateBoundaryMemory_RC(BMem, ...
                        Population1,Harvest1,Population2,Offspring2,W, ...
                        MemOptions);
                    Algorithm.recordMemoryTrace( ...
                        MemoryTrace,Problem.FE);
                    if usesCGAN && rawGuideCount > 0
                        [TrainX,TrainC,QueryRefs] = ...
                            BuildBoundaryDataset_RC(BMem,W,Problem);
                        eligible = trainingEligible(TrainC,QueryRefs, ...
                            minBoundaryLength,minGANTrainCount,Config);
                        Algorithm.recordTrainingTrace( ...
                            TrainC,QueryRefs,eligible,Problem.FE);
                        if eligible
                            [GAN,GuideDecs,GuideRefs,GuideSlots,PoolTrace] = ...
                                generateCGANGuidePool(GAN,TrainX,TrainC, ...
                                QueryRefs,W,RefScale,Problem,GANOptions, ...
                                Config,BMem);
                            Algorithm.recordGenerationTrace( ...
                                PoolTrace,Problem.FE);
                            if ~isempty(GuideDecs)
                                GuideRefScale = RefScale;
                            end
                        end
                    end
                end

                %% Ordinary environmental selection; raw CGAN rows are absent
                Union = [Population1,Population2,Offspring1, ...
                    Offspring2,Calibration];
                [Population1,Fitness1] = EnvironmentalSelection_CBS( ...
                    Union,Problem.N,true);
                [Population2,Fitness2] = EnvironmentalSelection_CBS( ...
                    Union,Problem.N,false);
                Algorithm.recordUseTrace( ...
                    GuideTrace,Population1,Problem.FE);
            end
        end
    end
end

function [GAN,GuideDecs,GuideRefs,GuideSlots,Trace] = ...
        generateCGANGuidePool(GAN,TrainX,TrainC,QueryRefs,W,RefScale,Problem, ...
        GANOptions,Config,BMem)
%GENERATECGANGUIDEPOOL Build either the legacy or balanced critic pool.

    Trace = emptyPoolTrace();
    if string(Config.guideGenerationMode) == "legacy"
        [SampleC,SampleRefs] = RunRegionGAN_RC( ...
            'regionquerysamples',QueryRefs,W,Config.legacyGuideSlots);
        SampleSlots = (1:size(SampleC,1))';
        repeats = max(1,round(double(Config.legacyCandidatesPerSlot)));
        SampleC = repelem(SampleC,repeats,1);
        SampleRefs = repelem(SampleRefs,repeats,1);
        SampleSlots = repelem(SampleSlots,repeats,1);
        QueryC = [SampleC,ones(size(SampleC,1),1)];
        [GAN,GuideDecs] = RunRegionGAN_RC('trainandsample',GAN, ...
            TrainX,TrainC,QueryC,Problem,GANOptions);
        rowCount = size(GuideDecs,1);
        GuideRefs = reshape(SampleRefs(1:rowCount),[],1);
        GuideSlots = reshape(SampleSlots(1:rowCount),[],1);
        Trace.active = rowCount > 0;
        Trace.rawCount = rowCount;
        Trace.keptCount = rowCount;
        Trace.rawConditions = numel(unique(GuideRefs));
        Trace.keptConditions = Trace.rawConditions;
        Trace.keepIdx = (1:rowCount)';
        Trace.percentile = nan(rowCount,1);
        Trace.trainConditions = numel(unique(QueryRefs));
        Trace = auditBoundaryGuidePool(Trace,Problem,GuideDecs, ...
            GuideRefs,Trace.keepIdx,BMem,W,Config);
        Trace = auditRawGuidePool( ...
            Trace,Problem,GuideDecs,GuideRefs,QueryRefs,W,RefScale,Config);
        return;
    end

    [SampleC,SampleRefs] = RunRegionGAN_RC( ...
        'balancedquerysamples',W,Config.rawGuideCount);
    QueryC = [SampleC,ones(size(SampleC,1),1)];
    [GAN,RawDec,RawScore] = RunRegionGAN_RC('trainandsample',GAN, ...
        TrainX,TrainC,QueryC,Problem,GANOptions);
    rowCount = size(RawDec,1);
    RawRefs = reshape(SampleRefs(1:rowCount),[],1);
    [GuideDecs,GuideRefs,KeepIdx,Percentile] = RunRegionGAN_RC( ...
        'conditioncriticfilter',RawDec,RawRefs,RawScore, ...
        Config.criticKeepCount,W);
    GuideSlots = (1:size(GuideDecs,1))';
    Trace.active = rowCount > 0;
    Trace.rawCount = rowCount;
    Trace.keptCount = size(GuideDecs,1);
    Trace.rawConditions = numel(unique(RawRefs));
    Trace.keptConditions = numel(unique(GuideRefs));
    Trace.keepIdx = KeepIdx;
    Trace.percentile = Percentile;
    Trace.trainConditions = numel(unique(QueryRefs));
    Trace = auditBoundaryGuidePool(Trace,Problem,RawDec,RawRefs, ...
        KeepIdx,BMem,W,Config);
    Trace = auditRawGuidePool( ...
        Trace,Problem,RawDec,RawRefs,QueryRefs,W,RefScale,Config);
end

function [Offspring,Trace] = generateGuidedOffspring( ...
        Problem,Population,Fitness,count,GuideDecs,GuideRefs,GuideSlots, ...
        GuideObjs,strictGuideRefs,GuideRefScale,W,plainDEShare,Config,BMem)
%GENERATEGUIDEDOFFSPRING GA + plain DE + target-guided DE offspring.
%   Before the cutoff, production and historical A0/A1/A2 select their
%   configured CGAN path, while Random20/DE20/GA20 replace the same quota.
%   Certified late targets retain the shared legacy exact-reference path.

    Trace = emptyUseTrace();
    count = max(0,min(numel(Population),round(double(count))));
    if count == 0
        Offspring = Population([]);
        return;
    end
    gaCount = min(count,round(0.25*count));
    plainCount = min(count-gaCount,round(plainDEShare*count));
    guidedCount = count-gaCount-plainCount;

    if gaCount > 0
        Offspring = gaOffspring(Problem,Population,Fitness,gaCount);
    else
        Offspring = Population([]);
    end
    if plainCount > 0
        PlainOffspring = deOffspring(Problem,Population,Fitness,plainCount);
        Offspring = [Offspring,PlainOffspring];
    end

    feasible = sum(max(0,Population.cons),2) <= 0;
    if guidedCount == 0
        return;
    end
    generationMode = string(Config.guideGenerationMode);
    ablationModes = ["random","traditional_de","traditional_ga"];
    if ~strictGuideRefs && ismember(generationMode,ablationModes)
        if generationMode == "random"
            QuotaOffspring = Problem.Initialization(guidedCount);
        elseif generationMode == "traditional_de"
            QuotaOffspring = deOffspring( ...
                Problem,Population,Fitness,guidedCount);
        else
            QuotaOffspring = gaOffspring( ...
                Problem,Population,Fitness,guidedCount);
        end
        Offspring = [Offspring,QuotaOffspring];
        Trace.active = true;
        Trace.requested = guidedCount;
        Trace.selected = guidedCount;
        Trace.feasibleChildren = sum( ...
            sum(max(0,QuotaOffspring.cons),2) <= 0);
        Trace.childDecs = double(QuotaOffspring.decs);
        return;
    end
    Trace.active = ~strictGuideRefs;
    Trace.requested = guidedCount;
    if isempty(GuideDecs) || isempty(GuideSlots)
        Offspring = [Offspring,deOffspring( ...
            Problem,Population,Fitness,guidedCount)];
        if Trace.active
            Trace.fallback = guidedCount;
            Trace.fallbackNoPool = guidedCount;
        end
        return;
    end

    currentFeasibleCount = sum(feasible);
    FeasiblePop = Population(feasible);
    FeasDecs = double(FeasiblePop.decs);
    FeasObjs = double(FeasiblePop.objs);
    feasFitness = reshape(Fitness(feasible),[],1);
    useMemory = ~strictGuideRefs && ...
        string(Config.guideUseMode) == "local_target" && ...
        string(Config.parentSourceMode) == "memory_fallback" && ...
        ~hasUsableLocalScale(Problem,FeasDecs);
    if useMemory
        [FeasDecs,FeasObjs,Trace.memoryParentsAdded] = ...
            supplementMemoryParents(FeasDecs,FeasObjs,BMem);
        if Trace.memoryParentsAdded > 0
            feasFitness = CalFitness_CBS(FeasObjs);
        end
    end
    if isempty(FeasDecs)
        Offspring = [Offspring,deOffspring( ...
            Problem,Population,Fitness,guidedCount)];
        if Trace.active
            Trace.fallback = guidedCount;
            Trace.fallbackNoCurrentFeasible = guidedCount;
        end
        return;
    end

    if strictGuideRefs
        rowCount = min([size(GuideDecs,1),numel(GuideRefs), ...
            numel(GuideSlots),size(GuideObjs,1)]);
        GuideDecs = GuideDecs(1:rowCount,:);
        GuideSlots = reshape(GuideSlots(1:rowCount),[],1);
        GuideObjs = GuideObjs(1:rowCount,:);
        if rowCount == 0
            Offspring = [Offspring,deOffspring( ...
                Problem,Population,Fitness,guidedCount)];
            return;
        end
        jointRefs = AssignReferenceVectors_CBS( ...
            [FeasObjs;GuideObjs],W);
        feasRefs = jointRefs(1:numel(FeasiblePop));
        GuideRefs = jointRefs(numel(FeasiblePop)+1:end);
    else
        feasRefs = AssignReferenceVectors_CBS( ...
            FeasObjs,W,GuideRefScale);
    end
    if ~strictGuideRefs && string(Config.guideUseMode) == "local_target"
        [ChildDecs,parentRow,selectedRefs,Map] = ...
            localTargetGuideDecisions(Problem,FeasDecs,feasRefs, ...
            feasFitness,GuideDecs,GuideRefs,W,guidedCount);
        Trace.mappedValid = Map.validCount;
        Trace.mapDropped = Map.droppedCount;
        Trace.alpha = Map.alpha;
        Trace.h = Map.h;
        Trace.d = Map.d;
        Trace.centerStep = Map.centerStep;
        Trace.actualStep = Map.actualStep;
        Trace.directionCosine = Map.directionCosine;
        if Map.noValidScale
            Trace.fallbackInvalidScale = guidedCount;
        end
        AuditTargets = Map.auditTargets;
        AuditCenters = Map.auditCenters;
    else
        if ~strictGuideRefs && ...
                string(Config.guideGenerationMode) == "global_critic"
            elite = eliteFeasibleRows(feasFitness);
            globalRows = vacancyWeightedMaximin(Problem,GuideDecs, ...
                GuideRefs,FeasDecs(elite,:),feasRefs(elite),W, ...
                guidedCount,1e-12);
            GuideDecs = GuideDecs(globalRows,:);
            GuideRefs = GuideRefs(globalRows);
            GuideSlots = (1:numel(globalRows))';
        end
        [ChildDecs,parentRow,selectedRefs,AuditTargets,AuditCenters] = ...
            legacyGuideDecisions( ...
            Problem,FeasDecs,feasRefs,feasFitness,GuideDecs,GuideRefs, ...
            GuideSlots,W,guidedCount,strictGuideRefs);
    end
    selectedCount = size(ChildDecs,1);
    if Trace.active && cutoffDiagnosticsEnabled(Config) && selectedCount > 0
        Trace = auditBoundaryUseTrace(Trace,Problem,AuditTargets, ...
            AuditCenters,ChildDecs,selectedRefs,BMem,W);
    end

    fallbackCount = guidedCount-selectedCount;
    if Trace.active && fallbackCount > 0 && ...
            Trace.fallbackInvalidScale == 0
        Trace.fallbackMapping = fallbackCount;
    end
    if fallbackCount > 0
        Offspring = [Offspring,deOffspring( ...
            Problem,Population,Fitness,fallbackCount)];
    end
    if selectedCount > 0
        Guided = Problem.Evaluation(ChildDecs);
        Offspring = [Offspring,Guided];
        if Trace.active
            feasibleChild = sum(max(0,Guided.cons),2) <= 0;
            childObjs = double(Guided.objs);
            parentObjs = FeasObjs(parentRow,:);
            dominates = all(childObjs <= parentObjs+1e-12,2) & ...
                any(childObjs < parentObjs-1e-12,2);
            Trace.feasibleChildren = sum(feasibleChild);
            Trace.dominatingChildren = sum(dominates);
            Trace.childDecs = ChildDecs;
            if mechanismAuditEnabled(Config) && ...
                    size(AuditTargets,1) == selectedCount && ...
                    size(AuditCenters,1) == selectedCount
                TargetPopulation = auditEvaluation(Problem,AuditTargets);
                CenterPopulation = auditEvaluation(Problem,AuditCenters);
                targetFeasible = sum(max(0,TargetPopulation.cons),2) <= 0;
                centerFeasible = sum(max(0,CenterPopulation.cons),2) <= 0;
                targetObjs = double(TargetPopulation.objs);
                centerObjs = double(CenterPopulation.objs);
                targetDominates = all(targetObjs <= parentObjs+1e-12,2) & ...
                    any(targetObjs < parentObjs-1e-12,2);
                centerDominates = all(centerObjs <= parentObjs+1e-12,2) & ...
                    any(centerObjs < parentObjs-1e-12,2);
                Trace.selectedTargetCount = selectedCount;
                Trace.selectedTargetFeasible = sum(targetFeasible);
                Trace.selectedTargetUseful = ...
                    sum(targetFeasible & targetDominates);
                Trace.centerFeasible = sum(centerFeasible);
                Trace.centerUseful = sum(centerFeasible & centerDominates);
                Trace.childUseful = sum(feasibleChild & dominates);
                Trace.targetFeasibleLostAtCenter = ...
                    sum(targetFeasible & ~centerFeasible);
                Trace.targetInfeasibleRecoveredAtCenter = ...
                    sum(~targetFeasible & centerFeasible);
                Trace.centerFeasibleLostAtMutation = ...
                    sum(centerFeasible & ~feasibleChild);
                Trace.centerInfeasibleRecoveredAtMutation = ...
                    sum(~centerFeasible & feasibleChild);
            end
        end
    end
    if Trace.active
        Trace.selected = selectedCount;
        Trace.fallback = fallbackCount;
        Trace.selectedConditions = numel(unique(selectedRefs));
        Trace.memoryParentsUsed = sum(parentRow > currentFeasibleCount);
    end
end

function yes = hasUsableLocalScale(Problem,FeasDecs)
%HASUSABLELOCALSCALE Check whether local A/h/T mapping has a valid scale.

    if isempty(FeasDecs)
        yes = false;
        return;
    end
    lower = double(Problem.lower);
    span = double(Problem.upper)-lower;
    span(span <= eps) = 1;
    h = robustLocalScale((double(FeasDecs)-lower)./span,1e-12);
    yes = any(isfinite(h) & h > 1e-12);
end

function [Decs,Objs,added] = supplementMemoryParents(Decs,Objs,BMem)
%SUPPLEMENTMEMORYPARENTS Append unique evaluated true-feasible anchors.

    added = 0;
    if isempty(BMem) || ~isstruct(BMem) || ...
            ~isfield(BMem,'x_b') || ~isfield(BMem,'y_b')
        return;
    end
    MemoryDecs = double(BMem.x_b);
    MemoryObjs = double(BMem.y_b);
    valid = all(isfinite(MemoryDecs),2) & all(isfinite(MemoryObjs),2);
    MemoryDecs = MemoryDecs(valid,:);
    MemoryObjs = MemoryObjs(valid,:);
    if isempty(MemoryDecs)
        return;
    end
    [~,rows] = unique(MemoryDecs,'rows','stable');
    rows = sort(rows);
    MemoryDecs = MemoryDecs(rows,:);
    MemoryObjs = MemoryObjs(rows,:);
    if ~isempty(Decs)
        novel = ~ismember(MemoryDecs,double(Decs),'rows');
        MemoryDecs = MemoryDecs(novel,:);
        MemoryObjs = MemoryObjs(novel,:);
    end
    added = size(MemoryDecs,1);
    Decs = [double(Decs);MemoryDecs];
    Objs = [double(Objs);MemoryObjs];
end

function [ChildDecs,parentRows,selectedRefs,TargetDecs,CenterDecs] = ...
        legacyGuideDecisions( ...
        Problem,FeasDecs,feasRefs,feasFitness,GuideDecs,GuideRefs, ...
        GuideSlots,W,guidedCount,strictRefs)
%LEGACYGUIDEDECISIONS Preserve the fixed-F boundary-target implementation.

    [plannedF,callGroup] = mainlineGuideScales(guidedCount);
    [childGuide,parentRows,childSlot] = selectGuideCandidates( ...
        Problem,FeasDecs,feasRefs,feasFitness,GuideDecs,GuideRefs, ...
        GuideSlots,W,plannedF,guidedCount,strictRefs);
    selectedCount = numel(childGuide);
    childF = plannedF(1:selectedCount);
    childGroup = callGroup(1:selectedCount);
    if numel(unique(childGuide)) ~= selectedCount || ...
            numel(unique(childSlot)) ~= selectedCount
        error('CBSRegionGAN:GuidePoolSelectionReuse', ...
            'Each guide slot and selected candidate must be unique.');
    end
    ChildDecs = zeros(selectedCount,Problem.D);
    TargetDecs = GuideDecs(childGuide,:);
    CenterDecs = zeros(selectedCount,Problem.D);
    lower = double(Problem.lower);
    upper = double(Problem.upper);
    for group = 1 : 3
        rows = find(childGroup == group);
        if isempty(rows)
            continue;
        end
        f = childF(rows(1));
        if any(childF(rows) ~= f)
            error('CBSRegionGAN:InconsistentGuideFGroup', ...
                'Each fixed OperatorDE call group must use one F value.');
        end
        A = FeasDecs(parentRows(rows),:);
        G = GuideDecs(childGuide(rows),:);
        CenterDecs(rows,:) = min(max(A+f*(G-A),lower),upper);
        ChildDecs(rows,:) = OperatorDE(Problem,A,G,A,{1,f,1,20});
    end
    selectedRefs = reshape(GuideRefs(childGuide),[],1);
end

function [ChildDecs,parentRows,selectedRefs,Map] = ...
        localTargetGuideDecisions(Problem,FeasDecs,feasRefs, ...
        feasFitness,GuideDecs,GuideRefs,W,guidedCount)
%LOCALTARGETGUIDEDECISIONS Map G to local T, then select globally in T-space.

    tau = 1e-12;
    lower = double(Problem.lower);
    upper = double(Problem.upper);
    span = upper-lower;
    span(span <= eps) = 1;
    FeasNorm = (FeasDecs-lower)./span;
    hAll = robustLocalScale(FeasNorm,tau);
    validH = isfinite(hAll) & hAll > tau;
    Map = struct('validCount',0,'droppedCount',size(GuideDecs,1), ...
        'noValidScale',false, ...
        'alpha',zeros(0,1),'h',zeros(0,1),'d',zeros(0,1), ...
        'centerStep',zeros(0,1),'actualStep',zeros(0,1), ...
        'directionCosine',zeros(0,1), ...
        'auditTargets',zeros(0,Problem.D), ...
        'auditCenters',zeros(0,Problem.D));
    if ~any(validH)
        Map.noValidScale = true;
        ChildDecs = zeros(0,Problem.D);
        parentRows = zeros(0,1);
        selectedRefs = zeros(0,1);
        return;
    end
    fallbackH = median(hAll(validH));
    elite = eliteFeasibleRows(feasFitness);
    EliteNorm = FeasNorm(elite,:);
    rowCount = min(size(GuideDecs,1),numel(GuideRefs));
    Targets = zeros(rowCount,Problem.D);
    mappedParent = zeros(rowCount,1);
    mappedSource = zeros(rowCount,1);
    mappedAlpha = zeros(rowCount,1);
    mappedH = zeros(rowCount,1);
    mappedD = zeros(rowCount,1);
    mappedCount = 0;
    for row = 1 : rowCount
        G = double(GuideDecs(row,:));
        ref = double(GuideRefs(row));
        if any(~isfinite(G)) || ~isfinite(ref) || ref ~= fix(ref) || ...
                ref < 1 || ref > size(W,1)
            continue;
        end
        GNorm = (G-lower)./span;
        distance = sqrt(sum((EliteNorm-GNorm).^2,2));
        nearest = distance <= min(distance)+tau;
        candidateParents = elite(nearest);
        refDistance = sqrt(sum((W(feasRefs(candidateParents),:)- ...
            W(ref,:)).^2,2));
        tieData = [refDistance,feasFitness(candidateParents), ...
            candidateParents];
        [~,order] = sortrows(tieData,[1 2 3]);
        parent = candidateParents(order(1));
        h = hAll(parent);
        if ~isfinite(h) || h <= tau
            h = fallbackH;
        end
        A = FeasDecs(parent,:);
        d = sqrt(sum(((G-A)./span).^2));
        if ~isfinite(d) || d <= tau || ~isfinite(h) || h <= tau
            continue;
        end
        alpha = min(1,h/d);
        T = A+alpha*(G-A);
        T = min(max(T,lower),upper);
        mappedCount = mappedCount+1;
        Targets(mappedCount,:) = T;
        mappedParent(mappedCount) = parent;
        mappedSource(mappedCount) = row;
        mappedAlpha(mappedCount) = alpha;
        mappedH(mappedCount) = h;
        mappedD(mappedCount) = d;
    end
    Targets = Targets(1:mappedCount,:);
    mappedParent = mappedParent(1:mappedCount);
    mappedSource = mappedSource(1:mappedCount);
    mappedAlpha = mappedAlpha(1:mappedCount);
    mappedH = mappedH(1:mappedCount);
    mappedD = mappedD(1:mappedCount);
    Map.validCount = mappedCount;
    Map.droppedCount = rowCount-mappedCount;
    if mappedCount == 0
        ChildDecs = zeros(0,Problem.D);
        parentRows = zeros(0,1);
        selectedRefs = zeros(0,1);
        return;
    end
    mappedRefs = reshape(GuideRefs(mappedSource),[],1);
    selected = vacancyWeightedMaximin(Problem,Targets,mappedRefs, ...
        FeasDecs(elite,:),feasRefs(elite),W,guidedCount,tau);
    parentRows = mappedParent(selected);
    selectedRefs = mappedRefs(selected);
    A = FeasDecs(parentRows,:);
    T = Targets(selected,:);
    ChildDecs = OperatorDE(Problem,A,T,A,{1,1,1,20});
    Map.alpha = mappedAlpha(selected);
    Map.h = mappedH(selected);
    Map.d = mappedD(selected);
    Map.auditTargets = GuideDecs(mappedSource(selected),:);
    Map.auditCenters = T;
    centerVector = (T-A)./span;
    childVector = (ChildDecs-A)./span;
    Map.centerStep = sqrt(sum(centerVector.^2,2));
    Map.actualStep = sqrt(sum(childVector.^2,2));
    Map.directionCosine = sum(centerVector.*childVector,2)./ ...
        max(Map.centerStep.*Map.actualStep,eps);
end

function rows = eliteFeasibleRows(feasFitness)
%ELITEFEASIBLEROWS Use Fitness<1, falling back to all feasible rows.

    rows = find(feasFitness < 1);
    if isempty(rows)
        rows = (1:numel(feasFitness))';
    end
end

function h = robustLocalScale(FeasNorm,tau)
%ROBUSTLOCALSCALE Median of up to three nearest positive neighbor distances.

    count = size(FeasNorm,1);
    h = nan(count,1);
    if count < 2
        return;
    end
    norm2 = sum(FeasNorm.^2,2);
    distance = sqrt(max(norm2+norm2'-2*(FeasNorm*FeasNorm'),0));
    for row = 1 : count
        positive = sort(distance(row,distance(row,:) > tau),'ascend');
        if numel(positive) >= 2
            h(row) = median(positive(1:min(3,numel(positive))));
        end
    end
end

function selected = vacancyWeightedMaximin(Problem,Candidates, ...
        CandidateRefs,BaseDecs,BaseRefs,W,requestedCount,tau)
%VACANCYWEIGHTEDMAXIMIN Select diverse targets with reference occupancy cost.

    Candidates = double(Candidates);
    CandidateRefs = reshape(double(CandidateRefs),[],1);
    requestedCount = max(0,round(double(requestedCount)));
    rowCount = min(size(Candidates,1),numel(CandidateRefs));
    Candidates = Candidates(1:rowCount,:);
    CandidateRefs = CandidateRefs(1:rowCount);
    lower = double(Problem.lower);
    span = double(Problem.upper)-lower;
    span(span <= eps) = 1;
    CandidateNorm = (Candidates-lower)./span;
    BaseNorm = (double(BaseDecs)-lower)./span;
    valid = all(isfinite(CandidateNorm),2) & isfinite(CandidateRefs) & ...
        CandidateRefs == fix(CandidateRefs) & CandidateRefs >= 1 & ...
        CandidateRefs <= size(W,1);
    remaining = valid;
    occupancy = accumarray(reshape(double(BaseRefs),[],1),1, ...
        [size(W,1),1],@sum,0);
    selectedOccupancy = zeros(size(W,1),1);
    delta = minimumPointDistance(CandidateNorm,BaseNorm);
    selected = zeros(min(requestedCount,sum(valid)),1);
    selectedCount = 0;
    while selectedCount < requestedCount && any(remaining)
        denominator = 1+occupancy(CandidateRefs(valid))+ ...
            selectedOccupancy(CandidateRefs(valid));
        validRows = find(valid);
        merit = -inf(rowCount,1);
        merit(validRows) = delta(validRows)./denominator;
        merit(~remaining) = -inf;
        [best,pick] = max(merit);
        if ~isfinite(best) || delta(pick) <= tau
            break;
        end
        selectedCount = selectedCount+1;
        selected(selectedCount) = pick;
        remaining(pick) = false;
        ref = CandidateRefs(pick);
        selectedOccupancy(ref) = selectedOccupancy(ref)+1;
        addedDistance = sqrt(sum((CandidateNorm-CandidateNorm(pick,:)).^2,2));
        delta = min(delta,addedDistance);
    end
    selected = selected(1:selectedCount);
end

function distance = minimumPointDistance(A,B)
%MINIMUMPOINTDISTANCE Minimum Euclidean distance from each A row to B.

    if isempty(A)
        distance = zeros(0,1);
    elseif isempty(B)
        distance = inf(size(A,1),1);
    else
        a2 = sum(A.^2,2);
        b2 = sum(B.^2,2)';
        distance = sqrt(min(max(a2+b2-2*(A*B'),0),[],2));
    end
end

function [plannedF,callGroup] = mainlineGuideScales(guidedCount)
%MAINLINEGUIDESCALES Fixed three-scale cycle of the unique mainline.

    ladder = [0.4,0.65,0.85];
    callGroup = reshape(mod(0:guidedCount-1,numel(ladder))+1,[],1);
    plannedF = reshape(ladder(callGroup),[],1);
end

function [selectedRows,parentRows,selectedSlots] = ...
        selectGuideCandidates(Problem,FeasDecs,feasRefs,feasFitness, ...
        GuideDecs,GuideRefs,GuideSlots,W,plannedF,requestedCount,strictRefs)
%SELECTGUIDECANDIDATES Select one candidate from each five-row query slot.

    rowCount = min([size(GuideDecs,1),numel(GuideRefs),numel(GuideSlots)]);
    GuideDecs = double(GuideDecs(1:rowCount,:));
    GuideRefs = reshape(double(GuideRefs(1:rowCount)),[],1);
    GuideSlots = reshape(double(GuideSlots(1:rowCount)),[],1);
    validSlots = isfinite(GuideSlots) & GuideSlots == fix(GuideSlots) & ...
        GuideSlots > 0;
    slotOrder = unique(GuideSlots(validSlots),'stable');

    lower = double(Problem.lower);
    upper = double(Problem.upper);
    span = upper-lower;
    span(span <= eps) = 1;
    FeasNorm = (double(FeasDecs)-lower)./span;
    feasibleCount = size(FeasNorm,1);
    if feasibleCount > 1
        norm2 = sum(FeasNorm.^2,2);
        distance2 = max(norm2 + norm2' - ...
            2*(FeasNorm*FeasNorm'),0);
        distance2(1:feasibleCount+1:end) = inf;
        localScale = sqrt(min(distance2,[],2));
        localScale(~isfinite(localScale)) = 0;
    else
        localScale = zeros(feasibleCount,1);
    end

    selectedRows = zeros(requestedCount,1);
    parentRows = zeros(requestedCount,1);
    selectedSlots = zeros(requestedCount,1);
    selectedCount = 0;
    for slot = 1 : numel(slotOrder)
        if selectedCount >= requestedCount
            break;
        end
        rows = find(GuideSlots == slotOrder(slot));
        valid = all(isfinite(GuideDecs(rows,:)),2) & ...
            isfinite(GuideRefs(rows)) & GuideRefs(rows) == ...
            fix(GuideRefs(rows)) & GuideRefs(rows) >= 1 & ...
            GuideRefs(rows) <= size(W,1);
        rows = rows(valid);
        if isempty(rows)
            continue;
        end
        target = GuideRefs(rows(1));
        rows = rows(GuideRefs(rows) == target);
        if isempty(rows)
            continue;
        end
        if strictRefs
            ties = find(feasRefs == target);
            if isempty(ties)
                continue;
            end
        else
            refDistance = sqrt(sum((W(feasRefs,:)-W(target,:)).^2,2));
            ties = find(refDistance <= min(refDistance)+1e-12);
        end
        [~,best] = min(feasFitness(ties));
        parent = ties(best);

        next = selectedCount+1;
        f = plannedF(next);
        A = FeasDecs(parent,:);
        Preview = A + f*(GuideDecs(rows,:)-A);
        Preview = min(max(Preview,lower),upper);
        move = sqrt(sum(((Preview-A)./span).^2,2));
        [~,pick] = min(abs(move-localScale(parent)));

        selectedRows(next,1) = rows(pick);
        parentRows(next,1) = parent;
        selectedSlots(next,1) = slotOrder(slot);
        selectedCount = next;
    end
    selectedRows = selectedRows(1:selectedCount);
    parentRows = parentRows(1:selectedCount);
    selectedSlots = selectedSlots(1:selectedCount);
end

function Offspring = generateBackboneOffspring( ...
        Problem,Population,Fitness,count)
%GENERATEBACKBONEOFFSPRING Fixed 25%% SBX+PM and 75%% ordinary-DE mixture.

    count = max(0,min(numel(Population),round(double(count))));
    if count == 0
        Offspring = Population([]);
        return;
    end
    gaCount = min(count,round(0.25*count));
    if gaCount > 0
        Offspring = gaOffspring(Problem,Population,Fitness,gaCount);
    else
        Offspring = Population([]);
    end
    if count > gaCount
        Offspring = [Offspring,deOffspring( ...
            Problem,Population,Fitness,count-gaCount)];
    end
end

function Offspring = deOffspring(Problem,Population,Fitness,count)
%DEOFFSPRING Ordinary DE with mutually distinct parent rows.

    Offspring = OperatorDEDistinct_CBS(Problem,Population,Fitness,count);
end

function Offspring = gaOffspring(Problem,Population,Fitness,count)
%GAOFFSPRING Platform SBX and polynomial-mutation offspring.

    matingPool = platformTournamentSelection(2,2*count,Fitness);
    Offspring = OperatorGAhalf(Problem,Population(matingPool));
end

function index = platformTournamentSelection(K,N,Fitness)
%PLATFORMTOURNAMENTSELECTION Preserve stable fitness-tie behavior.

    [~,order] = sortrows(reshape(Fitness,[],1));
    rank = zeros(numel(order),1);
    rank(order) = 1:numel(order);
    index = TournamentSelection(K,N,rank);
end

function configurePlatEMOUtilityPath()
%CONFIGUREPLATEMOUTILITYPATH Keep live code ahead of Data snapshots.

    algorithmsRoot = fileparts(which('ALGORITHM'));
    repoRoot = fileparts(algorithmsRoot);
    supportRoot = fullfile(repoRoot,'Algorithms', ...
        'Multi-objective optimization','CBS-CGAN','Support');
    if isfile(fullfile(supportRoot,'addCBSPaths.m'))
        addpath(supportRoot,'-begin');
        addCBSPaths(repoRoot);
    end
    utilityRoot = fullfile(algorithmsRoot,'Utility functions');
    currentSelection = string(which('TournamentSelection'));
    if ~startsWith(currentSelection,string(utilityRoot)+filesep)
        addpath(utilityRoot,'-begin');
    end
end

function Options = regionGANOptions(Config,minTrainCount)
%REGIONGANOPTIONS Convert the fixed configuration to WGAN options.

    Options = struct( ...
        'zDim',max(1,round(double(Config.zDim))), ...
        'iter',max(0,round(double(Config.ganIter))), ...
        'miniBatch',max(1,round(double(Config.ganMiniBatch))), ...
        'lrD',double(Config.ganLrD), ...
        'lrG',double(Config.ganLrG), ...
        'sampleSigma',double(Config.sampleSigma), ...
        'gpLambda',max(0,double(Config.gpLambda)), ...
        'nCritic',max(1,round(double(Config.nCritic))), ...
        'generatorHidden',double(Config.generatorHidden(:)'), ...
        'criticHidden',double(Config.criticHidden(:)'), ...
        'batchSamplingMode',string(Config.batchSamplingMode), ...
        'minTrainCount',double(minTrainCount));
end

function eligible = trainingEligible(TrainC,QueryRefs, ...
        minBoundaryLength,minGANTrainCount,Config)
%TRAININGELIGIBLE Apply the total-row or split-class training gate.

    total = size(TrainC,1);
    eligible = total >= max(minBoundaryLength,minGANTrainCount) && ...
        ~isempty(QueryRefs);
    mode = string(Config.trainGateMode);
    if mode == "total"
        return;
    elseif mode ~= "split"
        error('CBSRegionGAN:BadTrainGateMode', ...
            'Unsupported CGAN training gate mode.');
    end
    if isempty(TrainC)
        positive = 0;
        negative = 0;
    else
        positive = nnz(TrainC(:,end) >= 0.5);
        negative = total-positive;
    end
    eligible = eligible && ...
        positive >= double(Config.minPositiveTrainCount) && ...
        negative >= double(Config.minNegativeTrainCount) && ...
        numel(unique(QueryRefs)) >= double(Config.minTrainRefCount);
end

function Trace = emptyPoolTrace()
%EMPTYPOOLTRACE Empty generation-side mechanism record.

    Trace = struct('active',false,'rawCount',0,'keptCount',0, ...
        'rawConditions',0,'keptConditions',0,'keepIdx',zeros(0,1), ...
        'percentile',zeros(0,1),'rawOracleCount',0, ...
        'rawOracleFeasible',0,'trainConditions',0, ...
        'rawSupportedCount',0,'rawSupportedFeasible',0, ...
        'rawUnsupportedCount',0,'rawUnsupportedFeasible',0, ...
        'rawReferenceMatch',0,'rawSupportedReferenceMatch',0, ...
        'rawUnsupportedReferenceMatch',0, ...
        'rawBoundaryDistance',zeros(0,1), ...
        'rawBoundaryBand',zeros(0,1), ...
        'rawBoundarySupported',false(0,1), ...
        'keptBoundaryDistance',zeros(0,1), ...
        'keptBoundaryBand',zeros(0,1), ...
        'keptBoundarySupported',false(0,1), ...
        'rejectedBoundaryDistance',zeros(0,1), ...
        'rejectedBoundaryBand',zeros(0,1), ...
        'rejectedBoundarySupported',false(0,1), ...
        'criticBoundarySpearman',NaN, ...
        'criticBoundaryPairCount',0, ...
        'rawDirectionCoverage',NaN,'rawDirectionEntropy',NaN, ...
        'rawNearDuplicateRate',NaN, ...
        'keptDirectionCoverage',NaN,'keptDirectionEntropy',NaN, ...
        'keptNearDuplicateRate',NaN);
end

function Trace = emptyUseTrace()
%EMPTYUSETRACE Empty utilization-side mechanism record.

    Trace = struct('active',false,'requested',0,'selected',0, ...
        'fallback',0,'fallbackNoPool',0, ...
        'fallbackNoCurrentFeasible',0,'fallbackInvalidScale',0, ...
        'fallbackMapping',0,'memoryParentsAdded',0, ...
        'memoryParentsUsed',0,'selectedConditions',0,'mappedValid',0, ...
        'mapDropped',0,'alpha',zeros(0,1),'h',zeros(0,1), ...
        'd',zeros(0,1),'centerStep',zeros(0,1), ...
        'actualStep',zeros(0,1),'directionCosine',zeros(0,1), ...
        'feasibleChildren',0,'dominatingChildren',0, ...
        'selectedTargetCount',0,'selectedTargetFeasible',0, ...
        'selectedTargetUseful',0,'centerFeasible',0, ...
        'centerUseful',0,'childUseful',0, ...
        'targetFeasibleLostAtCenter',0, ...
        'targetInfeasibleRecoveredAtCenter',0, ...
        'centerFeasibleLostAtMutation',0, ...
        'centerInfeasibleRecoveredAtMutation',0, ...
        'childDecs',zeros(0,0), ...
        'selectedBoundaryDistance',zeros(0,1), ...
        'selectedBoundaryBand',zeros(0,1), ...
        'selectedBoundarySupported',false(0,1), ...
        'centerBoundaryDistance',zeros(0,1), ...
        'centerBoundaryBand',zeros(0,1), ...
        'centerBoundarySupported',false(0,1), ...
        'childBoundaryDistance',zeros(0,1), ...
        'childBoundaryBand',zeros(0,1), ...
        'childBoundarySupported',false(0,1), ...
        'selectedDirectionCoverage',NaN, ...
        'selectedDirectionEntropy',NaN, ...
        'selectedNearDuplicateRate',NaN);
end

function S = emptyGuideStats(Config,maxFE)
%EMPTYGUIDESTATS Initialize aggregate experiment counters.

    checkpointTargets = double(maxFE)*[0.05 0.10 0.20 0.30 ...
        0.40 0.50 0.75 1.00];
    S = struct( ...
        'arm',double(Config.experimentArm), ...
        'generationMode',string(Config.guideGenerationMode), ...
        'useMode',string(Config.guideUseMode), ...
        'keepUnpairedAnchors',logical(Config.keepUnpairedAnchors), ...
        'pairNeighborRefCount',double(Config.pairNeighborRefCount), ...
        'frontDepth',double(Config.frontDepth), ...
        'maxAnchorsPerRef',double(Config.maxAnchorsPerRef), ...
        'trainGateMode',string(Config.trainGateMode), ...
        'batchSamplingMode',string(Config.batchSamplingMode), ...
        'parentSourceMode',string(Config.parentSourceMode), ...
        'memoryEvents',0,'trueFeasibleSum',0,'afterFrontSum',0, ...
        'frontDroppedSum',0,'frontOpportunityRefsSum',0, ...
        'frontOpportunityEvents',0,'capActiveEvents',0, ...
        'afterCapSum',0,'capDroppedSum',0,'retainedSum',0, ...
        'pairedBeforeMADSum',0,'unpairedBeforeMADSum',0, ...
        'pairedSum',0,'unpairedSum',0,'madDroppedSum',0, ...
        'legalWithin5Sum',0,'legalWithin10Sum',0,'legalAnySum',0, ...
        'dominanceRejectedSum',0,'pairRank1To5Sum',0, ...
        'pairRank6To10Sum',0,'pairRankOver10Sum',0, ...
        'previousUnpairedSum',0,'previousUnpairedPairedSum',0, ...
        'pairGapMedianSum',0,'pairGapMedianCount',0, ...
        'pairGapP90Sum',0,'pairGapP90Count',0, ...
        'pairAngleMedianSum',0,'pairAngleMedianCount',0, ...
        'pairAngleP90Sum',0,'pairAngleP90Count',0, ...
        'dataEvents',0,'positiveRowsSum',0,'negativeRowsSum',0, ...
        'trainingRefsSum',0,'unsafeTrainingEvents',0, ...
        'imbalancedDataEvents',0,'trainingEvents',0, ...
        'imbalancedTrainingEvents',0,'trainingBlockedEvents',0, ...
        'generationEvents',0,'rawCandidates',0,'criticKept',0, ...
        'rawConditionSum',0,'keptConditionSum',0, ...
        'keptPercentileSum',0,'keptPercentileCount',0, ...
        'rejectedPercentileSum',0,'rejectedPercentileCount',0, ...
        'rawOracleCount',0,'rawOracleFeasible',0, ...
        'trainConditionSum',0, ...
        'rawSupportedCount',0,'rawSupportedFeasible',0, ...
        'rawUnsupportedCount',0,'rawUnsupportedFeasible',0, ...
        'rawReferenceMatch',0,'rawSupportedReferenceMatch',0, ...
        'rawUnsupportedReferenceMatch',0, ...
        'useEvents',0,'guidedRequested',0,'guidedSelected',0, ...
        'guidedFallback',0,'fallbackNoPool',0, ...
        'fallbackNoCurrentFeasible',0,'fallbackInvalidScale',0, ...
        'fallbackMapping',0,'memoryParentsAdded',0, ...
        'memoryParentsUsed',0,'selectedConditionSum',0, ...
        'mappedValid',0,'mapDropped',0,'alphaSum',0,'alphaCount',0, ...
        'hSum',0,'dSum',0,'centerStepSum',0,'actualStepSum',0, ...
        'directionCosineSum',0,'directionCosineCount',0, ...
        'guidedChildren',0,'guidedFeasible',0,'guidedDominating',0, ...
        'selectedTargetCount',0,'selectedTargetFeasible',0, ...
        'selectedTargetUseful',0,'centerFeasible',0, ...
        'centerUseful',0,'childUseful',0, ...
        'targetFeasibleLostAtCenter',0, ...
        'targetInfeasibleRecoveredAtCenter',0, ...
        'centerFeasibleLostAtMutation',0, ...
        'centerInfeasibleRecoveredAtMutation',0, ...
        'guidedSurvived',0, ...
        'firstPopulationFeasibleFE',Inf,'firstEligibleAnchorFE',Inf, ...
        'firstLegalPairFE',Inf,'firstTrainingFE',Inf, ...
        'firstGuidePoolFE',Inf,'firstGuidedUseFE',Inf, ...
        'firstUsefulGuidedFE',Inf, ...
        'checkpointTargets',checkpointTargets, ...
        'checkpointFE',nan(size(checkpointTargets)), ...
        'checkpointFeasibleCount',nan(size(checkpointTargets)), ...
        'checkpointNondominatedFeasibleCount',nan(size(checkpointTargets)), ...
        'checkpointRefCoverage',nan(size(checkpointTargets)), ...
        'checkpointRefEntropy',nan(size(checkpointTargets)), ...
        'checkpointRefOccupancyCV',nan(size(checkpointTargets)), ...
        'checkpointIGD',nan(size(checkpointTargets)), ...
        'checkpointHV',nan(size(checkpointTargets)), ...
        'checkpointTrainingEvents',nan(size(checkpointTargets)), ...
        'checkpointGuidedSelected',nan(size(checkpointTargets)), ...
        'checkpointGuidedFeasible',nan(size(checkpointTargets)), ...
        'checkpointGuidedSurvived',nan(size(checkpointTargets)));
    S.diagnosticsEnabled = cutoffDiagnosticsEnabled(Config);
    S.diagnosticSchemaVersion = "CBS-CGAN-cutoff-v1";
    S.boundaryProxyDefinition = ...
        "same-ref nearest normalized midpoint of finite BMem x_b/x_i pair";
    S.nearDuplicateTolerance = 1e-6;
    S.stopAtCGANEnd = logical(Config.stopAtCGANEnd);
    S.maxFE = double(maxFE);
    S.cganEndTarget = double(Config.ganStopFraction)*double(maxFE);
    S.initialFE = NaN;
    S.initialIGD = NaN;
    S.initialHV = NaN;
    S.initialFeasibleCount = NaN;
    S.initialNondominatedFeasibleCount = NaN;
    S.initialRefCoverage = NaN;
    S.initialRefEntropy = NaN;
    S.initialRefOccupancyCV = NaN;
    S.cganEndFE = NaN;
    S.cganEndIGD = NaN;
    S.cganEndHV = NaN;
    S.cganEndFeasibleCount = NaN;
    S.cganEndNondominatedFeasibleCount = NaN;
    S.cganEndRefCoverage = NaN;
    S.cganEndRefEntropy = NaN;
    S.cganEndRefOccupancyCV = NaN;
    S.cganEndDecs = zeros(0,0);
    S.cganEndObjs = zeros(0,0);
    S.cganEndCons = zeros(0,0);
    S.criticBoundaryPairCount = 0;
    distancePrefixes = ["raw","kept","rejected", ...
        "selected","center","child"];
    for prefix = distancePrefixes
        S.(char(prefix+"BoundaryDistanceMedianSum")) = 0;
        S.(char(prefix+"BoundaryDistanceMedianCount")) = 0;
        S.(char(prefix+"BoundaryDistanceP90Sum")) = 0;
        S.(char(prefix+"BoundaryDistanceP90Count")) = 0;
        S.(char(prefix+"BoundaryBandHits")) = 0;
        S.(char(prefix+"BoundaryBandCount")) = 0;
        S.(char(prefix+"BoundarySupported")) = 0;
        S.(char(prefix+"BoundaryTotal")) = 0;
        S.(char(prefix+"BoundaryDistanceValues")) = zeros(0,1);
    end
    scalarPrefixes = ["criticBoundarySpearman", ...
        "rawDirectionCoverage","rawDirectionEntropy", ...
        "rawNearDuplicateRate","keptDirectionCoverage", ...
        "keptDirectionEntropy","keptNearDuplicateRate", ...
        "selectedDirectionCoverage","selectedDirectionEntropy", ...
        "selectedNearDuplicateRate"];
    for prefix = scalarPrefixes
        S.(char(prefix+"Sum")) = 0;
        S.(char(prefix+"Count")) = 0;
    end
    changePrefixes = ["selectedToCenter","centerToChild"];
    for prefix = changePrefixes
        S.(char(prefix+"BoundaryChangeSum")) = 0;
        S.(char(prefix+"BoundaryChangeCount")) = 0;
    end
end

function Snapshot = finalizeGuideStats(S,ExactBoundaryQuantiles)
%FINALIZEGUIDESTATS Add interpretable rates and means to raw counters.

    if nargin < 2
        ExactBoundaryQuantiles = true;
    end
    if isempty(S) || isempty(fieldnames(S))
        Snapshot = struct();
        return;
    end
    Snapshot = S;
    Snapshot.meanTrueFeasible = safeRatio( ...
        S.trueFeasibleSum,S.memoryEvents);
    Snapshot.meanRetainedAnchors = safeRatio(S.retainedSum,S.memoryEvents);
    Snapshot.pairRate = safeRatio(S.pairedSum,S.afterCapSum);
    Snapshot.unpairedRate = safeRatio(S.unpairedSum,S.retainedSum);
    Snapshot.frontDropRate = safeRatio( ...
        S.frontDroppedSum,S.trueFeasibleSum);
    Snapshot.frontOpportunityEventRate = safeRatio( ...
        S.frontOpportunityEvents,S.memoryEvents);
    Snapshot.capDropRate = safeRatio(S.capDroppedSum,S.afterFrontSum);
    Snapshot.capActiveEventRate = safeRatio( ...
        S.capActiveEvents,S.memoryEvents);
    Snapshot.legalWithin5Rate = safeRatio( ...
        S.legalWithin5Sum,S.afterCapSum);
    Snapshot.legalWithin10Rate = safeRatio( ...
        S.legalWithin10Sum,S.afterCapSum);
    Snapshot.legalAnyRate = safeRatio(S.legalAnySum,S.afterCapSum);
    Snapshot.pairRank1To5Rate = safeRatio( ...
        S.pairRank1To5Sum,S.pairedBeforeMADSum);
    Snapshot.pairRank6To10Rate = safeRatio( ...
        S.pairRank6To10Sum,S.pairedBeforeMADSum);
    Snapshot.pairRankOver10Rate = safeRatio( ...
        S.pairRankOver10Sum,S.pairedBeforeMADSum);
    Snapshot.meanPairGapMedian = safeRatio( ...
        S.pairGapMedianSum,S.pairGapMedianCount);
    Snapshot.meanPairGapP90 = safeRatio( ...
        S.pairGapP90Sum,S.pairGapP90Count);
    Snapshot.meanPairAngleMedian = safeRatio( ...
        S.pairAngleMedianSum,S.pairAngleMedianCount);
    Snapshot.meanPairAngleP90 = safeRatio( ...
        S.pairAngleP90Sum,S.pairAngleP90Count);
    Snapshot.previousUnpairedConversionRate = safeRatio( ...
        S.previousUnpairedPairedSum,S.previousUnpairedSum);
    Snapshot.meanPositiveRows = safeRatio(S.positiveRowsSum,S.dataEvents);
    Snapshot.meanNegativeRows = safeRatio(S.negativeRowsSum,S.dataEvents);
    Snapshot.meanTrainingRefs = safeRatio(S.trainingRefsSum,S.dataEvents);
    Snapshot.unsafeTrainingEventRate = safeRatio( ...
        S.unsafeTrainingEvents,S.dataEvents);
    Snapshot.imbalancedDataEventRate = safeRatio( ...
        S.imbalancedDataEvents,S.dataEvents);
    Snapshot.imbalancedTrainingEventRate = safeRatio( ...
        S.imbalancedTrainingEvents,S.trainingEvents);
    Snapshot.trainingActivationRate = safeRatio( ...
        S.trainingEvents,S.dataEvents);
    Snapshot.criticRetentionRate = safeRatio(S.criticKept,S.rawCandidates);
    Snapshot.meanRawConditions = ...
        safeRatio(S.rawConditionSum,S.generationEvents);
    Snapshot.meanKeptConditions = ...
        safeRatio(S.keptConditionSum,S.generationEvents);
    Snapshot.meanKeptPercentile = ...
        safeRatio(S.keptPercentileSum,S.keptPercentileCount);
    Snapshot.meanRejectedPercentile = ...
        safeRatio(S.rejectedPercentileSum,S.rejectedPercentileCount);
    Snapshot.rawOracleFeasibleRate = ...
        safeRatio(S.rawOracleFeasible,S.rawOracleCount);
    Snapshot.meanTrainConditions = ...
        safeRatio(S.trainConditionSum,S.generationEvents);
    Snapshot.rawSupportedFeasibleRate = ...
        safeRatio(S.rawSupportedFeasible,S.rawSupportedCount);
    Snapshot.rawUnsupportedFeasibleRate = ...
        safeRatio(S.rawUnsupportedFeasible,S.rawUnsupportedCount);
    Snapshot.rawReferenceMatchRate = ...
        safeRatio(S.rawReferenceMatch,S.rawOracleCount);
    Snapshot.rawSupportedReferenceMatchRate = ...
        safeRatio(S.rawSupportedReferenceMatch,S.rawSupportedCount);
    Snapshot.rawUnsupportedReferenceMatchRate = ...
        safeRatio(S.rawUnsupportedReferenceMatch,S.rawUnsupportedCount);
    Snapshot.selectionRate = ...
        safeRatio(S.guidedSelected,S.guidedRequested);
    Snapshot.fallbackRate = ...
        safeRatio(S.guidedFallback,S.guidedRequested);
    Snapshot.parentFallbackRate = safeRatio( ...
        S.fallbackNoCurrentFeasible+S.fallbackInvalidScale, ...
        S.guidedRequested);
    Snapshot.noPoolFallbackRate = safeRatio( ...
        S.fallbackNoPool,S.guidedRequested);
    Snapshot.mappingFallbackRate = safeRatio( ...
        S.fallbackMapping,S.guidedRequested);
    Snapshot.memoryParentUseRate = safeRatio( ...
        S.memoryParentsUsed,S.guidedSelected);
    Snapshot.meanSelectedConditions = ...
        safeRatio(S.selectedConditionSum,S.useEvents);
    Snapshot.meanAlpha = safeRatio(S.alphaSum,S.alphaCount);
    Snapshot.meanH = safeRatio(S.hSum,S.alphaCount);
    Snapshot.meanRawDistance = safeRatio(S.dSum,S.alphaCount);
    Snapshot.meanCenterStep = safeRatio(S.centerStepSum,S.alphaCount);
    Snapshot.meanActualStep = safeRatio(S.actualStepSum,S.alphaCount);
    Snapshot.meanDirectionCosine = ...
        safeRatio(S.directionCosineSum,S.directionCosineCount);
    Snapshot.guidedFeasibleRate = ...
        safeRatio(S.guidedFeasible,S.guidedChildren);
    if ismember(S.generationMode, ...
            ["random","traditional_de","traditional_ga"])
        Snapshot.guidedDominanceRate = NaN;
    else
        Snapshot.guidedDominanceRate = ...
            safeRatio(S.guidedDominating,S.guidedChildren);
    end
    Snapshot.guidedSurvivalRate = ...
        safeRatio(S.guidedSurvived,S.guidedChildren);
    Snapshot.selectedTargetFeasibleRate = ...
        safeRatio(S.selectedTargetFeasible,S.selectedTargetCount);
    Snapshot.selectedTargetUsefulRate = ...
        safeRatio(S.selectedTargetUseful,S.selectedTargetCount);
    Snapshot.centerFeasibleRate = ...
        safeRatio(S.centerFeasible,S.selectedTargetCount);
    Snapshot.centerUsefulRate = ...
        safeRatio(S.centerUseful,S.selectedTargetCount);
    Snapshot.childUsefulRate = ...
        safeRatio(S.childUseful,S.selectedTargetCount);
    Snapshot.targetFeasibleLossRate = ...
        safeRatio(S.targetFeasibleLostAtCenter,S.selectedTargetFeasible);
    Snapshot.targetInfeasibleRecoveryRate = safeRatio( ...
        S.targetInfeasibleRecoveredAtCenter, ...
        S.selectedTargetCount-S.selectedTargetFeasible);
    Snapshot.centerFeasibleLossRate = ...
        safeRatio(S.centerFeasibleLostAtMutation,S.centerFeasible);
    Snapshot.centerInfeasibleRecoveryRate = safeRatio( ...
        S.centerInfeasibleRecoveredAtMutation, ...
        S.selectedTargetCount-S.centerFeasible);
    distancePrefixes = ["raw","kept","rejected", ...
        "selected","center","child"];
    for prefix = distancePrefixes
        values = S.(char(prefix+"BoundaryDistanceValues"));
        if ExactBoundaryQuantiles
            Snapshot.(char(prefix+"BoundaryDistanceMedian")) = ...
                diagnosticPercentile(values,0.5);
            Snapshot.(char(prefix+"BoundaryDistanceP90")) = ...
                diagnosticPercentile(values,0.9);
        else
            Snapshot.(char(prefix+"BoundaryDistanceMedian")) = ...
                safeRatio(S.(char(prefix+"BoundaryDistanceMedianSum")), ...
                S.(char(prefix+"BoundaryDistanceMedianCount")));
            Snapshot.(char(prefix+"BoundaryDistanceP90")) = ...
                safeRatio(S.(char(prefix+"BoundaryDistanceP90Sum")), ...
                S.(char(prefix+"BoundaryDistanceP90Count")));
        end
        Snapshot.(char(prefix+"BoundaryBandRate")) = safeRatio( ...
            S.(char(prefix+"BoundaryBandHits")), ...
            S.(char(prefix+"BoundaryBandCount")));
        Snapshot.(char(prefix+"BoundarySupportRate")) = safeRatio( ...
            S.(char(prefix+"BoundarySupported")), ...
            S.(char(prefix+"BoundaryTotal")));
    end
    scalarPrefixes = ["criticBoundarySpearman", ...
        "rawDirectionCoverage","rawDirectionEntropy", ...
        "rawNearDuplicateRate","keptDirectionCoverage", ...
        "keptDirectionEntropy","keptNearDuplicateRate", ...
        "selectedDirectionCoverage","selectedDirectionEntropy", ...
        "selectedNearDuplicateRate"];
    for prefix = scalarPrefixes
        Snapshot.(char(prefix)) = safeRatio( ...
            S.(char(prefix+"Sum")),S.(char(prefix+"Count")));
    end
    Snapshot.selectedToCenterBoundaryChangeMean = safeRatio( ...
        S.selectedToCenterBoundaryChangeSum, ...
        S.selectedToCenterBoundaryChangeCount);
    Snapshot.centerToChildBoundaryChangeMean = safeRatio( ...
        S.centerToChildBoundaryChangeSum, ...
        S.centerToChildBoundaryChangeCount);
    [Snapshot.frontHalfIGDAUC,Snapshot.frontHalfIGDAUCCoverage] = ...
        diagnosticTrajectoryAUC(S,"IGD");
    [Snapshot.frontHalfHVAUC,Snapshot.frontHalfHVAUCCoverage] = ...
        diagnosticTrajectoryAUC(S,"HV");
    valueFields = cellstr(distancePrefixes+"BoundaryDistanceValues");
    Snapshot = rmfield(Snapshot,valueFields);
end

function Config = mergeCutoffDiagnosticOptions(Config,Options)
%MERGECUTOFFDIAGNOSTICOPTIONS Overlay non-search diagnostic controls.

    if isempty(Options) || ~isstruct(Options) || isempty(fieldnames(Options))
        return;
    end
    Config.cutoffDiagnosticsEnabled = logical(Options.enabled);
    Config.stopAtCGANEnd = logical(Options.stopAtCGANEnd);
    Config.disableOracleAudit = logical(Options.disableOracleAudit);
end

function Options = validateCutoffDiagnosticOptions(Options)
%VALIDATECUTOFFDIAGNOSTICOPTIONS Validate the public diagnostic contract.

    Defaults = struct('enabled',false,'stopAtCGANEnd',false, ...
        'disableOracleAudit',false);
    if nargin < 1 || isempty(Options)
        Options = Defaults;
        return;
    end
    if ~isstruct(Options) || ~isscalar(Options)
        error('CBSRegionGAN:BadDiagnosticOptions', ...
            'Cutoff diagnostic options must be a scalar struct.');
    end
    names = fieldnames(Options);
    allowed = fieldnames(Defaults);
    if any(~ismember(names,allowed))
        error('CBSRegionGAN:BadDiagnosticOptions', ...
            'Unknown cutoff diagnostic option.');
    end
    for i = 1 : numel(allowed)
        name = allowed{i};
        if isfield(Options,name)
            value = Options.(name);
            if ~isscalar(value) || ~(islogical(value) || isnumeric(value)) || ...
                    ~isfinite(double(value)) || ...
                    (~islogical(value) && ~ismember(double(value),[0 1]))
                error('CBSRegionGAN:BadDiagnosticOptions', ...
                    'Diagnostic option %s must be a finite logical scalar.', ...
                    name);
            end
            Defaults.(name) = logical(value);
        end
    end
    Options = Defaults;
end

function enabled = cutoffDiagnosticsEnabled(Data)
%CUTOFFDIAGNOSTICSENABLED Read either configuration or stored state.

    enabled = false;
    if ~isstruct(Data) || ~isscalar(Data)
        return;
    end
    if isfield(Data,'cutoffDiagnosticsEnabled')
        value = Data.cutoffDiagnosticsEnabled;
    elseif isfield(Data,'diagnosticsEnabled')
        value = Data.diagnosticsEnabled;
    else
        return;
    end
    enabled = isscalar(value) && logical(value);
end

function Metrics = populationDiagnosticMetrics(Population,Problem,W,diagnose)
%POPULATIONDIAGNOSTICMETRICS Measure a stored population without evaluation.

    feasible = sum(max(0,Population.cons),2) <= 0;
    Metrics = struct('feasibleCount',sum(feasible), ...
        'nondominatedFeasibleCount',0,'refCoverage',0, ...
        'refEntropy',0,'refOccupancyCV',NaN,'igd',NaN,'hv',NaN);
    if any(feasible)
        FeasiblePopulation = Population(feasible);
        Metrics.nondominatedFeasibleCount = numel(FeasiblePopulation.best);
        refs = AssignReferenceVectors_CBS( ...
            double(FeasiblePopulation.objs),W);
        [coverage,entropyValue,occupancyCV] = ...
            referenceDistribution(refs,size(W,1));
        Metrics.refCoverage = coverage*size(W,1);
        Metrics.refEntropy = entropyValue;
        Metrics.refOccupancyCV = occupancyCV;
    end
    if diagnose
        Metrics.igd = safePopulationMetric(Problem,'IGD',Population);
        Metrics.hv = safePopulationMetric(Problem,'HV',Population);
    end
end

function value = safePopulationMetric(Problem,name,Population)
%SAFEPOPULATIONMETRIC Preserve FE and RNG around metric evaluation.

    savedFE = Problem.FE;
    savedRNG = rng;
    value = NaN;
    try
        value = double(Problem.CalMetric(name,Population));
        if ~isscalar(value) || ~isfinite(value)
            value = NaN;
        end
    catch
        value = NaN;
    end
    Problem.FE = savedFE;
    rng(savedRNG);
end

function Trace = auditBoundaryGuidePool(Trace,Problem,RawDec,RawRefs, ...
        KeepIdx,BMem,W,Config)
%AUDITBOUNDARYGUIDEPOOL Relate G rows to the current paired-memory proxy.

    if ~cutoffDiagnosticsEnabled(Config)
        return;
    end
    rowCount = min(size(RawDec,1),numel(RawRefs));
    RawDec = double(RawDec(1:rowCount,:));
    RawRefs = reshape(double(RawRefs(1:rowCount)),[],1);
    [distance,band,supported] = boundaryBracketDistance( ...
        Problem,RawDec,RawRefs,BMem);
    keep = false(rowCount,1);
    KeepIdx = reshape(double(KeepIdx),[],1);
    KeepIdx = KeepIdx(isfinite(KeepIdx) & KeepIdx == fix(KeepIdx) & ...
        KeepIdx >= 1 & KeepIdx <= rowCount);
    keep(KeepIdx) = true;
    Trace.rawBoundaryDistance = distance;
    Trace.rawBoundaryBand = band;
    Trace.rawBoundarySupported = supported;
    Trace.keptBoundaryDistance = distance(keep);
    Trace.keptBoundaryBand = band(keep);
    Trace.keptBoundarySupported = supported(keep);
    Trace.rejectedBoundaryDistance = distance(~keep);
    Trace.rejectedBoundaryBand = band(~keep);
    Trace.rejectedBoundarySupported = supported(~keep);
    [Trace.rawDirectionCoverage,Trace.rawDirectionEntropy] = ...
        referenceDistribution(RawRefs,size(W,1));
    [Trace.keptDirectionCoverage,Trace.keptDirectionEntropy] = ...
        referenceDistribution(RawRefs(keep),size(W,1));
    Trace.rawNearDuplicateRate = decisionNearDuplicateRate(Problem,RawDec);
    Trace.keptNearDuplicateRate = ...
        decisionNearDuplicateRate(Problem,RawDec(keep,:));
    valid = isfinite(Trace.percentile) & isfinite(distance);
    Trace.criticBoundaryPairCount = sum(valid);
    Trace.criticBoundarySpearman = spearmanCorrelation( ...
        Trace.percentile(valid),-distance(valid));
end

function Trace = auditBoundaryUseTrace(Trace,Problem,Targets,Centers, ...
        Children,Refs,BMem,W)
%AUDITBOUNDARYUSETRACE Trace G-to-T-to-child boundary geometry.

    rowCount = min([size(Targets,1),size(Centers,1), ...
        size(Children,1),numel(Refs)]);
    Targets = double(Targets(1:rowCount,:));
    Centers = double(Centers(1:rowCount,:));
    Children = double(Children(1:rowCount,:));
    Refs = reshape(double(Refs(1:rowCount)),[],1);
    [Trace.selectedBoundaryDistance,Trace.selectedBoundaryBand, ...
        Trace.selectedBoundarySupported] = boundaryBracketDistance( ...
        Problem,Targets,Refs,BMem);
    [Trace.centerBoundaryDistance,Trace.centerBoundaryBand, ...
        Trace.centerBoundarySupported] = boundaryBracketDistance( ...
        Problem,Centers,Refs,BMem);
    [Trace.childBoundaryDistance,Trace.childBoundaryBand, ...
        Trace.childBoundarySupported] = boundaryBracketDistance( ...
        Problem,Children,Refs,BMem);
    [Trace.selectedDirectionCoverage,Trace.selectedDirectionEntropy] = ...
        referenceDistribution(Refs,size(W,1));
    Trace.selectedNearDuplicateRate = ...
        decisionNearDuplicateRate(Problem,Targets);
end

function [distance,band,supported] = boundaryBracketDistance( ...
        Problem,Decs,Refs,BMem)
%BOUNDARYBRACKETDISTANCE Same-condition midpoint distance and band hit.
% The midpoint and half-width are geometric proxies for a finite feasible /
% infeasible bracket; no generated decision is evaluated or classified.

    rowCount = min(size(Decs,1),numel(Refs));
    distance = nan(rowCount,1);
    band = nan(rowCount,1);
    supported = false(rowCount,1);
    required = {'x_b','x_i','ref'};
    if rowCount == 0 || isempty(BMem) || ~isstruct(BMem) || ...
            ~all(isfield(BMem,required))
        return;
    end
    Xb = double(BMem.x_b);
    Xi = double(BMem.x_i);
    memoryRefs = reshape(double(BMem.ref),[],1);
    memoryCount = min([size(Xb,1),size(Xi,1),numel(memoryRefs)]);
    Xb = Xb(1:memoryCount,:);
    Xi = Xi(1:memoryCount,:);
    memoryRefs = memoryRefs(1:memoryCount);
    valid = all(isfinite(Xb),2) & all(isfinite(Xi),2) & ...
        isfinite(memoryRefs) & memoryRefs == fix(memoryRefs);
    if ~any(valid)
        return;
    end
    lower = double(Problem.lower);
    span = double(Problem.upper)-lower;
    span(span <= eps) = 1;
    Mid = ((Xb(valid,:)+Xi(valid,:))/2-lower)./span;
    HalfWidth = 0.5*sqrt(sum(((Xb(valid,:)-Xi(valid,:))./span).^2,2));
    validRefs = memoryRefs(valid);
    Decs = (double(Decs(1:rowCount,:))-lower)./span;
    Refs = reshape(double(Refs(1:rowCount)),[],1);
    for i = 1 : rowCount
        rows = find(validRefs == Refs(i));
        if isempty(rows) || any(~isfinite(Decs(i,:)))
            continue;
        end
        delta = sqrt(sum((Mid(rows,:)-Decs(i,:)).^2,2));
        [distance(i),which] = min(delta);
        supported(i) = true;
        band(i) = double(distance(i) <= HalfWidth(rows(which))+1e-12);
    end
end

function [coverage,entropyValue,occupancyCV] = ...
        referenceDistribution(Refs,refCount)
%REFERENCEDISTRIBUTION Normalized requested-reference coverage and entropy.

    refCount = max(0,round(double(refCount)));
    Refs = reshape(double(Refs),[],1);
    valid = isfinite(Refs) & Refs == fix(Refs) & Refs >= 1 & ...
        Refs <= refCount;
    if refCount == 0 || ~any(valid)
        coverage = 0;
        entropyValue = 0;
        occupancyCV = NaN;
        return;
    end
    counts = accumarray(Refs(valid),1,[refCount,1],@sum,0);
    coverage = nnz(counts)/refCount;
    probability = counts(counts > 0)/sum(counts);
    if refCount <= 1
        entropyValue = 1;
    else
        entropyValue = -sum(probability.*log(probability))/log(refCount);
    end
    occupancyCV = std(counts,0)/mean(counts);
end

function rate = decisionNearDuplicateRate(Problem,Decs)
%DECISIONNEARDUPLICATERATE Quantized normalized duplicate fraction.

    Decs = double(Decs);
    if size(Decs,1) < 2
        rate = 0;
        return;
    end
    lower = double(Problem.lower);
    span = double(Problem.upper)-lower;
    span(span <= eps) = 1;
    normalized = (Decs-lower)./span;
    valid = all(isfinite(normalized),2);
    normalized = normalized(valid,:);
    if size(normalized,1) < 2
        rate = 0;
        return;
    end
    bins = round(normalized/1e-6);
    uniqueCount = size(unique(bins,'rows'),1);
    rate = 1-uniqueCount/size(normalized,1);
end

function value = spearmanCorrelation(X,Y)
%SPEARMANCORRELATION Safe rank correlation for supported candidate pairs.

    X = reshape(double(X),[],1);
    Y = reshape(double(Y),[],1);
    valid = isfinite(X) & isfinite(Y);
    X = X(valid);
    Y = Y(valid);
    value = NaN;
    if numel(X) < 3 || numel(unique(X)) < 2 || numel(unique(Y)) < 2
        return;
    end
    try
        value = corr(X,Y,'Type','Spearman','Rows','complete');
    catch
        value = NaN;
    end
end

function S = accumulateBoundarySummary(S,prefix,distance,band,supported)
%ACCUMULATEBOUNDARYSUMMARY Retain pooled distances and compact rates.

    prefix = string(prefix);
    distance = reshape(double(distance),[],1);
    band = reshape(double(band),[],1);
    supported = reshape(logical(supported),[],1);
    values = distance(isfinite(distance));
    if ~isempty(values)
        medianField = char(prefix+"BoundaryDistanceMedianSum");
        medianCountField = char(prefix+"BoundaryDistanceMedianCount");
        p90Field = char(prefix+"BoundaryDistanceP90Sum");
        p90CountField = char(prefix+"BoundaryDistanceP90Count");
        valuesField = char(prefix+"BoundaryDistanceValues");
        S.(medianField) = S.(medianField)+diagnosticPercentile(values,0.5);
        S.(medianCountField) = S.(medianCountField)+1;
        S.(p90Field) = S.(p90Field)+diagnosticPercentile(values,0.9);
        S.(p90CountField) = S.(p90CountField)+1;
        S.(valuesField) = [S.(valuesField);values];
    end
    band = band(isfinite(band));
    S.(char(prefix+"BoundaryBandHits")) = ...
        S.(char(prefix+"BoundaryBandHits"))+sum(band > 0.5);
    S.(char(prefix+"BoundaryBandCount")) = ...
        S.(char(prefix+"BoundaryBandCount"))+numel(band);
    S.(char(prefix+"BoundarySupported")) = ...
        S.(char(prefix+"BoundarySupported"))+sum(supported);
    S.(char(prefix+"BoundaryTotal")) = ...
        S.(char(prefix+"BoundaryTotal"))+numel(supported);
end

function S = accumulateFiniteScalar(S,prefix,value)
%ACCUMULATEFINITESCALAR Add one finite event-level diagnostic.

    prefix = string(prefix);
    if isscalar(value) && isfinite(value)
        S.(char(prefix+"Sum")) = S.(char(prefix+"Sum"))+double(value);
        S.(char(prefix+"Count")) = S.(char(prefix+"Count"))+1;
    end
end

function S = accumulateDistanceChange(S,prefix,before,after)
%ACCUMULATEDISTANCECHANGE Aggregate signed bracket-distance changes.

    prefix = string(prefix);
    before = reshape(double(before),[],1);
    after = reshape(double(after),[],1);
    count = min(numel(before),numel(after));
    change = after(1:count)-before(1:count);
    change = change(isfinite(change));
    S.(char(prefix+"BoundaryChangeSum")) = ...
        S.(char(prefix+"BoundaryChangeSum"))+sum(change);
    S.(char(prefix+"BoundaryChangeCount")) = ...
        S.(char(prefix+"BoundaryChangeCount"))+numel(change);
end

function [area,coverage] = diagnosticTrajectoryAUC(S,metricName)
%DIAGNOSTICTRAJECTORYAUC Integrate only adjacent finite observations.

    metricName = upper(string(metricName));
    if metricName == "IGD"
        initial = S.initialIGD;
        checkpoints = S.checkpointIGD;
        endpoint = S.cganEndIGD;
    else
        initial = S.initialHV;
        checkpoints = S.checkpointHV;
        endpoint = S.cganEndHV;
    end
    within = isfinite(S.checkpointFE) & ...
        S.checkpointFE <= S.cganEndFE;
    x = [double(S.initialFE),double(S.checkpointFE(within)), ...
        double(S.cganEndFE)]/double(S.maxFE);
    y = [initial,double(checkpoints(within)),endpoint];
    [x,order] = sort(x);
    y = y(order);
    [x,uniqueRows] = unique(x,'stable');
    y = y(uniqueRows);
    if numel(x) < 2
        area = NaN;
        coverage = 0;
        return;
    end
    adjacent = isfinite(x(1:end-1)) & isfinite(x(2:end)) & ...
        isfinite(y(1:end-1)) & isfinite(y(2:end));
    widths = diff(x);
    rows = find(adjacent);
    area = sum(widths(rows).*(y(rows)+y(rows+1))/2);
    coverage = sum(widths(adjacent));
    if coverage <= 0
        area = NaN;
    end
end

function value = diagnosticPercentile(values,q)
%DIAGNOSTICPERCENTILE Linear-interpolated finite percentile.

    values = sort(reshape(double(values),[],1));
    values = values(isfinite(values));
    if isempty(values)
        value = NaN;
    elseif numel(values) == 1
        value = values(1);
    else
        position = 1+min(max(double(q),0),1)*(numel(values)-1);
        low = floor(position);
        high = ceil(position);
        weight = position-low;
        value = values(low)*(1-weight)+values(high)*weight;
    end
end

function Trace = auditRawGuidePool( ...
        Trace,Problem,GuideDecs,GuideRefs,TrainRefs,W,RefScale,Config)
%AUDITRAWGUIDEPOOL Oracle-check generated decisions without consuming FE.

    if ~mechanismAuditEnabled(Config) || isempty(GuideDecs)
        return;
    end
    Population = auditEvaluation(Problem,GuideDecs);
    feasible = sum(max(0,Population.cons),2) <= 0;
    rowCount = min(numel(Population),numel(GuideRefs));
    requestedRefs = reshape(GuideRefs(1:rowCount),[],1);
    supported = ismember(requestedRefs,TrainRefs);
    feasible = feasible(1:rowCount);
    actualRefs = AssignReferenceVectors_CBS( ...
        double(Population(1:rowCount).objs),W,RefScale);
    matches = actualRefs == requestedRefs;
    Trace.rawOracleCount = rowCount;
    Trace.rawOracleFeasible = sum(feasible);
    Trace.rawSupportedCount = sum(supported);
    Trace.rawSupportedFeasible = sum(feasible & supported);
    Trace.rawUnsupportedCount = sum(~supported);
    Trace.rawUnsupportedFeasible = sum(feasible & ~supported);
    Trace.rawReferenceMatch = sum(matches);
    Trace.rawSupportedReferenceMatch = sum(matches & supported);
    Trace.rawUnsupportedReferenceMatch = sum(matches & ~supported);
end

function enabled = mechanismAuditEnabled(Config)
%MECHANISMAUDITENABLED Restrict oracle probes to an explicit test branch.

    enabled = isfield(Config,'mechanismAudit') && ...
        isscalar(Config.mechanismAudit) && logical(Config.mechanismAudit) && ...
        (~isfield(Config,'disableOracleAudit') || ...
        ~logical(Config.disableOracleAudit));
end

function Population = auditEvaluation(Problem,Decs)
%AUDITEVALUATION Evaluate deterministic probes while preserving FE and RNG.

    savedFE = Problem.FE;
    savedRNG = rng;
    try
        Population = Problem.Evaluation(Decs);
    catch err
        Problem.FE = savedFE;
        rng(savedRNG);
        rethrow(err);
    end
    Problem.FE = savedFE;
    rng(savedRNG);
end

function value = safeRatio(numerator,denominator)
%SAFERATIO Return NaN when an aggregate has no observations.

    if denominator > 0
        value = double(numerator)/double(denominator);
    else
        value = NaN;
    end
end
