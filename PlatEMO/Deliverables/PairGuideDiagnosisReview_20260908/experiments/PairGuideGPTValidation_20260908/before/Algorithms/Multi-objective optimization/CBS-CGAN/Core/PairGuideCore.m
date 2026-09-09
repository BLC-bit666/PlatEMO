classdef PairGuideCore < ALGORITHM
%PAIRGUIDECORE PairGuide implementation.
% The algorithm learns independent real endpoints and generates single points.
% Pending candidates receive direct complete evaluation next generation.
% The unconstrained population uses 25% GA and 75% ordinary DE throughout.
% rawGuideCount    --- 500 --- Native single-point proposals per query event
% zDim             ---   6 --- Generator noise dimension
% ganUpdates       --- 1000 --- Generator updates for initial training
% ganMiniBatch     ---  32 --- Independent endpoints per mini-batch
% nCritic          ---   5 --- Critic updates per generator update
% minGANTrainCount ---   8 --- Minimum active pairs required for training
% sampleSigma      ---   0 --- Production inference noise standard deviation

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
        GuideStats = struct();
        PairEvidence = struct();
        ComparisonMode = "cgan";
        DiagnosticOptions = struct();
        ObjectiveSnapshotOptions = struct();
        FirstPairGuideUseState = struct();
        PairGuideTrainingExperimentOptions = struct();
        BoundaryExperimentOptions = struct();
    end

    methods
        function Algorithm = PairGuideCore(varargin)
            Algorithm@ALGORITHM(varargin{:});
        end

        function main(Algorithm,Problem)
            configurePlatEMOUtilityPath();
            Config = Algorithm.algorithmConfiguration();
            Config = mergeCutoffDiagnosticOptions( ...
                Config,Algorithm.DiagnosticOptions);
            Config = mergePairGuideTrainingExperimentOptions( ...
                Config,Algorithm.PairGuideTrainingExperimentOptions);
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
            Snapshot.evidence = Algorithm.PairEvidence;
        end

        function configureObjectiveSpaceSnapshots(Algorithm,Options)
        %CONFIGUREOBJECTIVESPACESNAPSHOTS Enable behavior-neutral point capture.
            Algorithm.ObjectiveSnapshotOptions = ...
                validateObjectiveSnapshotOptions(Options);
        end

        function Snapshots = objectiveSpaceSnapshots(Algorithm)
        %OBJECTIVESPACESNAPSHOTS Cached observations; never call real oracles.
            Snapshots = pairEvidenceSnapshots(Algorithm.PairEvidence, ...
                Algorithm.ObjectiveSnapshotOptions);
        end

        function configureComparison(Algorithm,mode)
        %CONFIGURECOMPARISON Same identity and budget, two causal controls.
            mode = string(mode);
            if ~isscalar(mode) || ~ismember(mode,["cgan","fallback_only","pair_only"])
                error('CBSPairGuide:ComparisonMode','Unknown comparison mode.');
            end
            Algorithm.ComparisonMode = mode;
        end

        function configureFirstPairGuideUseCapture(~,~)
        %CONFIGUREFIRSTPAIRGUIDEUSECAPTURE Enable the isolated epoch probe.
            error('CBSPairGuide:RetiredProbe', ...
                'The old epoch probe is retired. Read guideExperimentSnapshot evidence.');
        end

        function Capture = firstPairGuideUseCapture(Algorithm)
        %FIRSTPAIRGUIDEUSECAPTURE Return configured training/use events.
            Capture = Algorithm.FirstPairGuideUseState;
        end

        function configurePairGuideTrainingExperiment(Algorithm,Options)
        %CONFIGUREPAIRGUIDETRAININGEXPERIMENT Isolate schedule experiments.
            Algorithm.PairGuideTrainingExperimentOptions = ...
                validatePairGuideTrainingExperimentOptions(Options);
        end

        function configureBoundaryExperiment(Algorithm,Options)
        %CONFIGUREBOUNDARYEXPERIMENT Isolate pool and archive-front comparisons.
            if ~isstruct(Options) || ~isscalar(Options) || ...
                    ~isequal(sort(fieldnames(Options)),sort({'selectionPool';'archiveFrontDepth'}))
                error('CBSPairGuide:BoundaryExperiment','Specify selectionPool and archiveFrontDepth.');
            end
            pool = string(Options.selectionPool);
            depth = Options.archiveFrontDepth;
            if ~isscalar(pool) || ~ismember(pool,["shared","ccmo"]) || ...
                    ~isnumeric(depth) || ~isscalar(depth) || ~ismember(depth,[0 1 2])
                error('CBSPairGuide:BoundaryExperiment','Pool must be shared/ccmo and front depth 0/1/2.');
            end
            Algorithm.BoundaryExperimentOptions = struct( ...
                'selectionPool',pool,'archiveFrontDepth',double(depth));
        end
    end

    methods(Access = protected)
        function Config = algorithmConfiguration(Algorithm)
        %ALGORITHMCONFIGURATION Parse and lock PairGuide parameters.
            Defaults = PairGuideCore.mainlineDefaults();
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
            Config.experimentArm = 7;
            Config.guideGenerationMode = "pair_guide";
            Config.guideUseMode = "pair_guide";
            Config.guideOffspringShare = 0.20;
            Config.ganStopFraction = inf;
            Config.pairMinPairs = max(1,round(double(minGANTrainCount)));
            Config.pairGanEpoch = max(0,round(double(ganIter)));
            Config.pairInitialEpoch = Config.pairGanEpoch;
            Config.disableOracleAudit = true;
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
                'ganIter',1000, ...
                'ganMiniBatch',32, ...
                'ganLrD',1e-3, ...
                'ganLrG',1e-3, ...
                'frontDepth',1, ...
                'selectionPool',"ccmo", ...
                'archiveFrontDepth',1, ...
                'pairNeighborRefRadius',2, ...
                'refDivisor',1, ...
                'minBoundaryLength',2, ...
                'gpLambda',10, ...
                'nCritic',5, ...
                'maxAnchorsPerRef',5, ...
                'minGANTrainCount',8, ...
                'trainingSigma',0, ...
                'sampleSigma',0, ...
                'ganStopFraction',inf, ...
                'guideOffspringShare',0.20, ...
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
                'disableOracleAudit',true);
            Defaults.pairArchivePerRef = 1;
            Defaults.pairArchiveCapacity = 500;
            Defaults.pairGanEpoch = 1000;
            Defaults.pairInitialEpoch = 1000;
            Defaults.pairRetrainEpoch = 20;
            Defaults.pairMinPairs = 8;
            Defaults.retrainChange = 0.2;
            Defaults.retrainGenerations = 10;
            Defaults.pairDuplicateTolerance = 1e-6;
            Defaults.pairImprovementTolerance = 1e-12;
        end
    end

    methods(Access = private)
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
            if isfield(Trace,'added')
                S.pairArchiveAdded = S.pairArchiveAdded+double(Trace.added);
            end
            if isfield(Trace,'tightenedFeasible')
                S.pairArchiveTightenedFeasible = ...
                    S.pairArchiveTightenedFeasible+ ...
                    double(Trace.tightenedFeasible);
            end
            if isfield(Trace,'tightenedInfeasible')
                S.pairArchiveTightenedInfeasible = ...
                    S.pairArchiveTightenedInfeasible+ ...
                    double(Trace.tightenedInfeasible);
            end
            if isfield(Trace,'removed')
                S.pairArchiveRemoved = S.pairArchiveRemoved+ ...
                    double(Trace.removed);
            end
            sourceFields = { ...
                'guidedTightenedFeasible','guidedTightenedInfeasible', ...
                'ordinaryTightenedFeasible','ordinaryTightenedInfeasible'};
            targetFields = { ...
                'pairGuidedTightenedFeasible', ...
                'pairGuidedTightenedInfeasible', ...
                'pairOrdinaryTightenedFeasible', ...
                'pairOrdinaryTightenedInfeasible'};
            for i = 1 : numel(sourceFields)
                if isfield(Trace,sourceFields{i})
                    S.(targetFields{i}) = S.(targetFields{i})+ ...
                        double(Trace.(sourceFields{i}));
                end
            end
            if isfield(Trace,'strong')
                S.pairArchiveStrongLast = double(Trace.strong);
                S.pairArchiveStrongMax = max( ...
                    S.pairArchiveStrongMax,double(Trace.strong));
            end
            if isfield(Trace,'weak')
                S.pairArchiveWeakLast = double(Trace.weak);
                S.pairArchiveWeakMax = max( ...
                    S.pairArchiveWeakMax,double(Trace.weak));
            end
            if isfield(Trace,'generatedWeak')
                S.pairGeneratedWeakLast = double(Trace.generatedWeak);
                S.pairGeneratedWeakMax = max( ...
                    S.pairGeneratedWeakMax,double(Trace.generatedWeak));
            end
            if isfield(Trace,'resumeEligible')
                S.pairArchiveResumeEligibleLast = ...
                    double(Trace.resumeEligible);
            end
            if isfield(Trace,'pairGapMedian')
                S.pairGapMedianLast = double(Trace.pairGapMedian);
            end
            if isfield(Trace,'pairGapP90')
                S.pairGapP90Last = double(Trace.pairGapP90);
            end
            Event = emptyPairArchiveEvent();
            Event.fe = double(currentFE);
            eventFields = {'retained','active','inactive','resumeEligible', ...
                'added','tightenedFeasible','tightenedInfeasible', ...
                'guidedTightenedFeasible','guidedTightenedInfeasible', ...
                'ordinaryTightenedFeasible','ordinaryTightenedInfeasible', ...
                'removed','pairGapMedian','pairGapP90'};
            for i = 1 : numel(eventFields)
                if isfield(Trace,eventFields{i})
                    Event.(eventFields{i}) = double(Trace.(eventFields{i}));
                end
            end
            S.pairArchiveLog(end+1,1) = Event;
            if isinf(S.firstEligibleAnchorFE) && Trace.afterCap > 0
                S.firstEligibleAnchorFE = double(currentFE);
            end
            if isinf(S.firstLegalPairFE) && Trace.paired > 0
                S.firstLegalPairFE = double(currentFE);
            end
            Algorithm.GuideStats = S;
        end

        function recordPairTrainingTrace(Algorithm,Gate,Status,currentFE)
        %RECORDPAIRTRAININGTRACE Expose pair gate and training events.
            if isempty(Gate) || ~isstruct(Gate)
                return;
            end
            S = Algorithm.GuideStats;
            S.pairGateEvents = S.pairGateEvents+1;
            names = ["effective","active","regions","pairs"];
            for name = names
                if isfield(Gate,name) && isfinite(double(Gate.(name)))
                    value = double(Gate.(name));
                    S.("pair"+upper(extractBefore(name,2))+ ...
                        extractAfter(name,1)+"Last") = value;
                    S.("pair"+upper(extractBefore(name,2))+ ...
                        extractAfter(name,1)+"Max") = max( ...
                        S.("pair"+upper(extractBefore(name,2))+ ...
                        extractAfter(name,1)+"Max"),value);
                end
            end
            if isfield(Gate,'eligible') && logical(Gate.eligible)
                S.pairEligibleEvents = S.pairEligibleEvents+1;
            end
            if isempty(Status) || ~isstruct(Status)
                Algorithm.GuideStats = S;
                return;
            end
            if isfield(Status,'trained') && logical(Status.trained)
                S.pairTrainingEvents = S.pairTrainingEvents+1;
                if isfield(Status,'epochs')
                    S.pairTrainingEpochs = S.pairTrainingEpochs+ ...
                        double(Status.epochs);
                end
                if isfield(Status,'updates')
                    S.pairTrainingUpdates = S.pairTrainingUpdates+ ...
                        double(Status.updates);
                end
                if isfield(Status,'pairVisits')
                    S.pairTrainingPairVisits = ...
                        S.pairTrainingPairVisits+double(Status.pairVisits);
                end
                if isinf(S.firstPairTrainingFE)
                    S.firstPairTrainingFE = double(currentFE);
                end
                Event = emptyPairTrainingEvent();
                Event.fe = double(currentFE);
                statusFields = {'trainingKind','nCritic','epochs', ...
                    'trainingPairs','changedPairs','newRegions','updates', ...
                    'pairVisits','batchesPerEpoch','trainingSeconds'};
                eventFields = {'kind','nCritic','epochs','trainingPairs', ...
                    'changedPairs','newRegions','updates','pairVisits', ...
                    'batchesPerEpoch','trainingSeconds'};
                for i = 1 : numel(statusFields)
                    if isfield(Status,statusFields{i})
                        Event.(eventFields{i}) = Status.(statusFields{i});
                    end
                end
                Event = copyPairModelDiagnostics( ...
                    Event,Status,'preDiagnostics','pre');
                Event = copyPairModelDiagnostics( ...
                    Event,Status,'postDiagnostics','post');
                S.pairTrainingLog(end+1,1) = Event;
            end
            if isfield(Status,'useModel') && logical(Status.useModel)
                S.pairModelReadyEvents = S.pairModelReadyEvents+1;
            end
            if isfield(Status,'reason') && string(Status.reason) == "current"
                S.pairCurrentReuseEvents = S.pairCurrentReuseEvents+1;
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
            if S.generationMode == "pair_guide"
                % PairGuide has no critic screening; geometric validity is
                % recorded separately from critic retention.
                S.criticKept = S.criticKept+Trace.rawCount;
            else
                S.criticKept = S.criticKept+Trace.keptCount;
            end
            if isfield(Trace,'matchFailures')
                S.pairMatchFailures = S.pairMatchFailures+ ...
                    double(Trace.matchFailures);
            end
            if isfield(Trace,'invalidCount')
                S.pairInvalidCandidates = S.pairInvalidCandidates+ ...
                    double(Trace.invalidCount);
            end
            if isfield(Trace,'validCount')
                S.pairPoolValid = S.pairPoolValid+ ...
                    double(Trace.validCount);
            end
            if isfield(Trace,'objectiveFE')
                S.pairObjectiveFE = S.pairObjectiveFE+ ...
                    double(Trace.objectiveFE);
            end
            if isfield(Trace,'objectiveCandidateCount')
                S.pairObjectiveCandidates = S.pairObjectiveCandidates+ ...
                    double(Trace.objectiveCandidateCount);
            end
            if isfield(Trace,'corridorPass')
                S.pairCorridorPass = S.pairCorridorPass+ ...
                    double(Trace.corridorPass);
            end
            if isfield(Trace,'sphereRejectCount')
                S.pairSphereReject = S.pairSphereReject+ ...
                    double(Trace.sphereRejectCount);
            end
            if isfield(Trace,'spherePassCount')
                S.pairSpherePass = S.pairSpherePass+ ...
                    double(Trace.spherePassCount);
            end
            if isfield(Trace,'localDominancePass')
                S.pairLocalDominancePass = ...
                    S.pairLocalDominancePass+double(Trace.localDominancePass);
            end
            if isfield(Trace,'jointPassCount')
                S.pairJointPass = ...
                    S.pairJointPass+double(Trace.jointPassCount);
            end
            if isfield(Trace,'rawProjectionRate')
                values = double(Trace.rawProjectionRate(:));
                values = values(isfinite(values));
                S.pairProjectionRateSum = ...
                    S.pairProjectionRateSum+sum(values);
                S.pairProjectionRateCount = ...
                    S.pairProjectionRateCount+numel(values);
            end
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
            Event = emptyPairGenerationEvent();
            Event.fe = double(currentFE);
            Event.modelVersion = double(S.pairTrainingEvents);
            eventFields = {'rawCount','invalidCount','matchFailures', ...
                'sphereRejectCount','spherePassCount', ...
                'objectiveCandidateCount', ...
                'localDominancePass','corridorPass','jointPassCount', ...
                'keptCount','objectiveFE','rawConditions','keptConditions'};
            for i = 1 : numel(eventFields)
                if isfield(Trace,eventFields{i})
                    Event.(eventFields{i}) = double(Trace.(eventFields{i}));
                end
            end
            S.pairGenerationLog(end+1,1) = Event;
            S.pendingPairPoolFE = Event.fe;
            S.pendingPairModelVersion = Event.modelVersion;
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
            if S.generationMode == "pair_guide"
                S.pairPoolSelected = S.pairPoolSelected+Trace.selected;
                S.pairGuideFullFE = S.pairGuideFullFE+ ...
                    double(Trace.fullGuideFE);
                S.pairGuideConstraintFE = S.pairGuideConstraintFE+ ...
                    double(Trace.constraintGuideFE);
            end
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
            survivedCount = 0;
            if ~isempty(Trace.childDecs)
                survived = ismember(Trace.childDecs, ...
                    double(SelectedPopulation.decs),'rows');
                survivedCount = sum(survived);
                S.guidedSurvived = S.guidedSurvived+survivedCount;
            end
            if isinf(S.firstGuidedUseFE) && Trace.selected > 0
                S.firstGuidedUseFE = double(currentFE);
            end
            if isinf(S.firstUsefulGuidedFE) && Trace.childUseful > 0
                S.firstUsefulGuidedFE = double(currentFE);
            end
            Event = emptyPairUseEvent();
            Event.fe = double(currentFE);
            Event.useFE = double(currentFE);
            Event.modelVersion = double(S.pendingPairModelVersion);
            Event.poolFE = double(S.pendingPairPoolFE);
            Event.requested = double(Trace.requested);
            Event.selected = double(Trace.selected);
            Event.fallback = double(Trace.fallback);
            Event.feasible = double(Trace.feasibleChildren);
            Event.dominating = double(Trace.dominatingChildren);
            Event.feasibleDominating = double(Trace.childUseful);
            Event.survivedP1 = double(survivedCount);
            S.pairUseLog(end+1,1) = Event;
            S.pendingPairPoolFE = NaN;
            S.pendingPairModelVersion = NaN;
            Algorithm.GuideStats = S;
        end

        function nofinish = auditNotTerminated(Algorithm,Population,Problem,W)
        %AUDITNOTTERMINATED Save behavior-neutral population checkpoints.
            S = Algorithm.GuideStats;
            feasible = sum(max(0,Population.cons),2) <= 0;
            if isinf(S.firstPopulationFeasibleFE) && any(feasible)
                S.firstPopulationFeasibleFE = double(Problem.FE);
            end
            % This completed state remains the latest available state until
            % the next generation finishes. Never fill a target with future FE.
            nextFE = double(Problem.FE)+min(2*Problem.N, ...
                max(0,double(Problem.maxFE)-double(Problem.FE)));
            rows = find(isnan(S.checkpointFE) & ...
                S.checkpointTargets >= double(Problem.FE) & ...
                (S.checkpointTargets < nextFE | ...
                S.checkpointTargets == double(Problem.FE)));
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
                S.checkpointPairActive(rows) = S.pairArchiveStrongLast;
                S.checkpointPairGapMedian(rows) = S.pairGapMedianLast;
                S.checkpointPairGapP90(rows) = S.pairGapP90Last;
                S.checkpointPairArchiveAdded(rows) = S.pairArchiveAdded;
                S.checkpointPairTightenedFeasible(rows) = ...
                    S.pairArchiveTightenedFeasible;
                S.checkpointPairTightenedInfeasible(rows) = ...
                    S.pairArchiveTightenedInfeasible;
                S.checkpointPairGuidedTightened(rows) = ...
                    S.pairGuidedTightenedFeasible+ ...
                    S.pairGuidedTightenedInfeasible;
                S.checkpointPairTrainingEvents(rows) = ...
                    S.pairTrainingEvents;
                S.checkpointPairGenerationEvents(rows) = ...
                    numel(S.pairGenerationLog);
                S.checkpointPairUseEvents(rows) = numel(S.pairUseLog);
                S.checkpointPairDonors(rows) = S.guidedSelected;
                S.checkpointPairFallback(rows) = S.guidedFallback;
                S.checkpointPairObjectiveFE(rows) = S.pairObjectiveFE;
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
        %RUNMAINLINE CCMO selection, complete archive Union, delayed guidance.
            PairGuideCost_RC('start',Problem);
            costCleanup = onCleanup(@()PairGuideCost_RC('stop'));
            [W,~] = UniformPoint(max(2,round(Problem.N/Config.refDivisor)),Problem.M);
            Options = pairGANOptions(Config);
            BoundaryOptions = Algorithm.BoundaryExperimentOptions;
            if isempty(fieldnames(BoundaryOptions))
                BoundaryOptions = struct('selectionPool',Config.selectionPool, ...
                    'archiveFrontDepth',Config.archiveFrontDepth);
            end
            Config.frontDepth = BoundaryOptions.archiveFrontDepth;
            Algorithm.GuideStats = emptyGuideStats(Config,Problem.maxFE);
            Options.archiveFrontDepth = BoundaryOptions.archiveFrontDepth;
            Options.W = W;
            Options.pairOnly = Algorithm.ComparisonMode == "pair_only";
            Options.guideQuota = round(0.20*Problem.N);
            seed = pairRunSeed();
            trainStream = RandStream('mt19937ar','Seed',mod(seed+104729,2^32-1));
            queryStream = RandStream('mt19937ar','Seed',mod(seed+130363,2^32-1));
            Population1 = Problem.Initialization(min(Problem.N,Problem.maxFE-Problem.FE));
            Population2 = Population1([]);
            if Problem.FE < Problem.maxFE
                Population2 = Problem.Initialization(min(Problem.N,Problem.maxFE-Problem.FE));
            end
            Fitness1 = CalFitness_CBS(Population1.objs,Population1.cons);
            Fitness2 = [];
            if ~isempty(Population2); Fitness2 = CalFitness_CBS(Population2.objs); end
            Pending = struct();
            GAN = [];
            generation = 0;
            Evidence = struct('schema',"PairGuide-single-v3",'mode',Algorithm.ComparisonMode, ...
                'seed',seed,'W',W,'generations',{{}},'queries',{{}},'training',{{}}, ...
                'evaluations',{{}},'population',{{}},'fullFE',double(Problem.FE), ...
                'objectiveOnlyFE',0,'constraintOnlyFE',0,'networkEndpointRows',0, ...
                'trainingSeconds',0,'forwardSeconds',0,'archiveSeconds',0, ...
                'diagnosticSeconds',0,'evaluationSeconds',0,'totalSeconds',0, ...
                'trainingGeneratorRows',0,'trainingCriticRows',0, ...
                'diagnosticGeneratorRows',0,'diagnosticCriticRows',0);
            Evidence.boundaryExperiment = BoundaryOptions;
            Initial = [Population1,Population2];
            Evidence.evaluations{1} = struct('firstEvalID',1,'lastEvalID',double(Problem.FE), ...
                'origin',"initialization",'decisions',double(Initial.decs), ...
                'objectives',double(Initial.objs), ...
                'constraints',double(Initial.cons));
            EvaluationIndex = containers.Map('KeyType','char','ValueType','any');
            indexEvaluations(EvaluationIndex,Evidence.evaluations{1});
            timer = tic;
            [Archive,Scale,~] = PairBoundaryArchive_RC('update',[],Population1, ...
                [Population1,Population2],W,Problem,Options,struct(),Problem.FE);
            Evidence.archiveSeconds = Evidence.archiveSeconds+toc(timer);
            Evidence.population{1} = cachedPopulation(Problem.FE,Population1,Population2,Archive);
            Evidence.population{1}.referenceScale = Scale;
            Evidence.oracleCalls = PairGuideCost_RC('snapshot');
            Algorithm.PairEvidence = Evidence;
            runTimer = tic;
            while Algorithm.auditNotTerminated(Population1,Problem,W)
                generation = generation+1;
                Options.generation = generation;
                startFE = Problem.FE;
                budget = min(2*Problem.N,max(0,Problem.maxFE-Problem.FE));
                count1 = min(Problem.N,ceil(budget/2));
                count2 = min(Problem.N,floor(budget/2));
                timer = tic;
                [Offspring1,GuideTrace,EvaluationLog] = pairOffspring(Problem, ...
                    Population1,Fitness1,count1,Pending,Config,seed,generation);
                Evidence.evaluationSeconds = Evidence.evaluationSeconds+toc(timer);
                Evidence.evaluations{end+1} = EvaluationLog;
                indexEvaluations(EvaluationIndex,EvaluationLog);
                timer = tic;
                beforeP2 = Problem.FE;
                Offspring2 = scopedBackbone(Problem,Population2,Fitness2,count2,seed,generation);
                Evidence.evaluationSeconds = Evidence.evaluationSeconds+toc(timer);
                Evidence.evaluations{end+1} = evaluationRecord(Offspring2,beforeP2,"P2_GA_DE");
                ga2 = round(0.25*count2);
                Evidence.evaluations{end}.origin = [repmat("P2_GA",ga2,1); ...
                    repmat("P2_DE",count2-ga2,1)];
                indexEvaluations(EvaluationIndex,Evidence.evaluations{end});
                Pending = struct();
                Union = [Population1,Population2,Offspring1,Offspring2];
                % Archive reconstruction always receives the complete Union.
                if BoundaryOptions.selectionPool == "ccmo"
                    [Population1,Fitness1] = EnvironmentalSelection_CBS( ...
                        [Population1,Offspring1,Offspring2],Problem.N,true);
                    [Population2,Fitness2] = EnvironmentalSelection_CBS( ...
                        [Population2,Offspring1,Offspring2],Problem.N,false);
                else
                    [Population1,Fitness1] = EnvironmentalSelection_CBS(Union,Problem.N,true);
                    [Population2,Fitness2] = EnvironmentalSelection_CBS(Union,Problem.N,false);
                end
                timer = tic;
                [Archive,Scale,MemoryTrace] = PairBoundaryArchive_RC('update',Archive, ...
                    Population1,Union,W,Problem,Options,GuideTrace,Problem.FE);
                Evidence.archiveSeconds = Evidence.archiveSeconds+toc(timer);
                for eventRow = 1:numel(MemoryTrace.events)
                    event = MemoryTrace.events(eventRow);
                    key = evaluationKey(event.decision,event.objectives,event.feasible);
                    if isKey(EvaluationIndex,key)
                        entry = EvaluationIndex(key);
                        MemoryTrace.events(eventRow).evalID = entry.id;
                        MemoryTrace.events(eventRow).origin = entry.origin;
                        MemoryTrace.events(eventRow).evalFE = entry.id;
                    end
                end
                Algorithm.recordUseTrace(GuideTrace,Population1,Problem.FE);
                Algorithm.recordMemoryTrace(MemoryTrace,Problem.FE);
                Options.referenceScale = Scale;
                [Data,Gate,TrainC,QueryRefs] = PairBoundaryArchive_RC( ...
                    'trainingdata',Archive,W,Problem,Options);
                Algorithm.recordTrainingTrace(TrainC,QueryRefs,Gate.eligible,Problem.FE);
                GuideTrace.survivedP1 = ismember(GuideTrace.childDecs,double(Population1.decs),'rows');
                GuideTrace.survivedP2 = ismember(GuideTrace.childDecs,double(Population2.decs),'rows');
                Evidence.generations{end+1} = struct('generation',generation, ...
                    'startFE',startFE,'consumptionFE',double(Problem.FE), ...
                    'use',GuideTrace,'archive',MemoryTrace);
                Evidence.population{end+1} = cachedPopulation(Problem.FE,Population1,Population2,Archive);
                Evidence.population{end}.referenceScale = Scale;
                Evidence.fullFE = double(Problem.FE);
                % Reserve only slots that the next P1 batch can actually use.
                remaining = min(2*Problem.N,max(0,Problem.maxFE-Problem.FE));
                nextCount1 = min(Problem.N,ceil(remaining/2));
                Options.guideQuota = round(0.20*nextCount1);
                if Options.guideQuota > 0 && Algorithm.ComparisonMode ~= "fallback_only"
                    if Algorithm.ComparisonMode == "cgan"
                        scope = pairStreamScope(trainStream); %#ok<NASGU>
                        [GAN,Status] = PairBoundaryWGAN_RC('trainifneeded',GAN,Data,Gate,Problem,Options);
                        clear scope;
                        Algorithm.recordPairTrainingTrace(Gate,Status,Problem.FE);
                        if Status.trained
                            Evidence.lastModel = GAN;
                            if GAN.version == 1; Evidence.firstModel = GAN; end
                        end
                        Evidence.training{end+1} = Status;
                        Evidence.trainingSeconds = Evidence.trainingSeconds+Status.trainingSeconds;
                        Evidence.diagnosticSeconds = Evidence.diagnosticSeconds+Status.diagnosticSeconds;
                        Evidence.trainingGeneratorRows = Evidence.trainingGeneratorRows+Status.generatorForwardRows;
                        Evidence.trainingCriticRows = Evidence.trainingCriticRows+Status.criticForwardRows;
                        Evidence.diagnosticGeneratorRows = Evidence.diagnosticGeneratorRows+Status.diagnosticGeneratorRows;
                        Evidence.diagnosticCriticRows = Evidence.diagnosticCriticRows+Status.diagnosticCriticRows;
                        ready = Status.useModel && Data.count > 0;
                    else
                        ready = Data.count > 0;
                    end
                    if ready && Config.rawGuideCount > 0
                        scope = pairStreamScope(queryStream); %#ok<NASGU>
                        [QueryC,Info] = PairBoundaryArchive_RC('querycontexts', ...
                            Archive,W,Options,Config.rawGuideCount);
                        if Algorithm.ComparisonMode == "cgan"
                            [Raw,Sample] = PairBoundaryWGAN_RC('sample',GAN,QueryC,Problem,Options);
                            version = GAN.version;
                        else
                            [Raw,Sample] = pairOnlySample(Archive,Info,Problem);
                            version = 0;
                        end
                        if Options.pairOnly
                            Info.generatedF = Sample.generatedF; Info.generatedI = Sample.generatedI;
                            Info.sides = nan(size(Raw,1),1);
                        end
                        Current = [Population1,Population2];
                        Options.currentDecs = double(Current.decs);
                        feasible = all(Population1.cons <= 0,2);
                        Options.p1Objs = double(Population1(feasible).objs);
                        Options.p1Refs = AssignReferenceVectors_CBS(Options.p1Objs,W,Scale);
                        [Decs,Refs,Ids,Pool] = PairBoundaryArchive_RC('selectcandidates', ...
                            Raw,Info,Archive,Scale,Problem,Options);
                        clear scope;
                        Algorithm.recordGenerationTrace(Pool,Problem.FE);
                        k = Pool.keepIdx;
                        Pending = struct('decs',Decs,'refs',Refs,'ids',Ids, ...
                            'xf',Pool.xf(k,:),'xi',Pool.xi(k,:), ...
                            'yf',Pool.yf(k,:),'yi',Pool.yi(k,:), ...
                            'sides',Info.sides(k),'productionFE',double(Problem.FE),'productionGeneration',generation, ...
                            'modelVersion',version);
                        Evidence.networkEndpointRows = Evidence.networkEndpointRows+Sample.endpointForwardRows;
                        Evidence.forwardSeconds = Evidence.forwardSeconds+Sample.forwardSeconds;
                        Evidence.queries{end+1} = struct('productionFE',double(Problem.FE), ...
                            'generation',generation,'modelVersion',version, ...
                            'rawDecs',Raw,'conditions',QueryC,'refs',Info.refs,'sides',Info.sides, ...
                            'sample',Sample,'pool',Pool,'pending',Pending);
                    end
                end
                Evidence.totalSeconds = toc(runTimer);
                Evidence.oracleCalls = PairGuideCost_RC('snapshot');
                Algorithm.PairEvidence = Evidence;
            end
            Evidence.totalSeconds = toc(runTimer);
            Evidence.oracleCalls = PairGuideCost_RC('snapshot');
            Algorithm.PairEvidence = Evidence;
        end

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



function Options = pairGANOptions(Config)
%PAIRGANOPTIONS Isolate PairGuide archive and WGAN settings from mainline.

    names = ["zDim","pairGanEpoch","pairInitialEpoch", ...
        "pairRetrainEpoch","ganMiniBatch","ganLrD","ganLrG", ...
        "gpLambda","nCritic","trainingSigma","sampleSigma", ...
        "generatorHidden","criticHidden", ...
        "pairArchivePerRef","pairArchiveCapacity", ...
        "pairNeighborRefCount","pairMinPairs", ...
        "retrainChange","retrainGenerations", ...
        "pairDuplicateTolerance","pairImprovementTolerance"];
    Options = struct();
    for name = names
        field = char(name);
        if isfield(Config,field)
            Options.(field) = Config.(field);
        end
    end
    Options.epochs = Config.pairGanEpoch;
    if isfield(Config,'pairInitialEpoch')
        Options.initialEpoch = Config.pairInitialEpoch;
    else
        Options.initialEpoch = Config.pairGanEpoch;
    end
    if isfield(Config,'pairRetrainEpoch')
        Options.retrainEpoch = Config.pairRetrainEpoch;
    else
        Options.retrainEpoch = Config.pairGanEpoch;
    end
    Options.miniBatch = Config.ganMiniBatch;
    Options.lrD = Config.ganLrD;
    Options.lrG = Config.ganLrG;
    Options.collectDiagnostics = true;
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
        'childDecs',zeros(0,0),'childObjs',zeros(0,0), ...
        'childCons',zeros(0,0),'parentObjs',zeros(0,0), ...
        'selectedTargetDecs',zeros(0,0), ...
        'selectedCenterDecs',zeros(0,0), ...
        'matchedPairIds',zeros(0,1), ...
        'fullGuideFE',0,'constraintGuideFE',0, ...
        'pairDecisionRngBefore',struct(), ...
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

    checkpointTargets = double(maxFE)*(0.10:0.10:1.00);
    pairInitialEpoch = double(Config.pairGanEpoch);
    pairRetrainEpoch = double(Config.pairGanEpoch);
    if isfield(Config,'pairInitialEpoch')
        pairInitialEpoch = double(Config.pairInitialEpoch);
    end
    if isfield(Config,'pairRetrainEpoch')
        pairRetrainEpoch = double(Config.pairRetrainEpoch);
    end
    S = struct( ...
        'arm',double(Config.experimentArm), ...
        'generationMode',string(Config.guideGenerationMode), ...
        'useMode',string(Config.guideUseMode), ...
        'nCritic',double(Config.nCritic), ...
        'trainingSigma',double(Config.trainingSigma), ...
        'sampleSigma',double(Config.sampleSigma), ...
        'pairGanEpoch',double(Config.pairGanEpoch), ...
        'pairInitialEpoch',pairInitialEpoch, ...
        'pairRetrainEpoch',pairRetrainEpoch, ...
        'guideOffspringShare',double(Config.guideOffspringShare), ...
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
        'checkpointGuidedSurvived',nan(size(checkpointTargets)), ...
        'pairTrainingLog',repmat(emptyPairTrainingEvent(),0,1));
    S.diagnosticsEnabled = cutoffDiagnosticsEnabled(Config);
    S.diagnosticSchemaVersion = "CBS-CGAN-cutoff-v1";
    S.pairGuideSchema = "PairGuide";
    S.oracleAuditDisabled = isfield(Config,'disableOracleAudit') && ...
        isscalar(Config.disableOracleAudit) && ...
        logical(Config.disableOracleAudit);
    S.pairArchiveAdded = 0;
    S.pairArchiveTightenedFeasible = 0;
    S.pairArchiveTightenedInfeasible = 0;
    S.pairGuidedTightenedFeasible = 0;
    S.pairGuidedTightenedInfeasible = 0;
    S.pairOrdinaryTightenedFeasible = 0;
    S.pairOrdinaryTightenedInfeasible = 0;
    S.pairArchiveRemoved = 0;
    S.pairArchiveResumeEligibleLast = 0;
    S.pairGapMedianLast = NaN;
    S.pairGapP90Last = NaN;
    S.pairArchiveLog = repmat(emptyPairArchiveEvent(),0,1);
    S.pairGenerationLog = repmat(emptyPairGenerationEvent(),0,1);
    S.pairUseLog = repmat(emptyPairUseEvent(),0,1);
    S.pendingPairPoolFE = NaN;
    S.pendingPairModelVersion = NaN;
    S.pairArchiveStrongLast = 0;
    S.pairArchiveStrongMax = 0;
    S.pairArchiveWeakLast = 0;
    S.pairArchiveWeakMax = 0;
    S.pairGeneratedWeakLast = 0;
    S.pairGeneratedWeakMax = 0;
    S.pairGateEvents = 0;
    S.pairEligibleEvents = 0;
    gateNames = ["Effective","Active","Regions","Pairs"];
    for name = gateNames
        S.("pair"+name+"Last") = 0;
        S.("pair"+name+"Max") = 0;
    end
    S.pairTrainingEvents = 0;
    S.pairModelReadyEvents = 0;
    S.pairCurrentReuseEvents = 0;
    S.pairTrainingEpochs = 0;
    S.pairTrainingUpdates = 0;
    S.pairTrainingPairVisits = 0;
    S.pairPoolValid = 0;
    S.pairPoolSelected = 0;
    S.pairObjectiveCandidates = 0;
    S.pairObjectiveFE = 0;
    S.pairCorridorPass = 0;
    S.pairInvalidCandidates = 0;
    S.pairSphereReject = 0;
    S.pairSpherePass = 0;
    S.pairLocalDominancePass = 0;
    S.pairJointPass = 0;
    S.pairGuideFullFE = 0;
    S.pairGuideConstraintFE = 0;
    S.pairMatchFailures = 0;
    S.pairProjectionRateSum = 0;
    S.pairProjectionRateCount = 0;
    S.firstPairTrainingFE = Inf;
    checkpointShape = size(checkpointTargets);
    S.checkpointPairActive = nan(checkpointShape);
    S.checkpointPairGapMedian = nan(checkpointShape);
    S.checkpointPairGapP90 = nan(checkpointShape);
    S.checkpointPairArchiveAdded = nan(checkpointShape);
    S.checkpointPairTightenedFeasible = nan(checkpointShape);
    S.checkpointPairTightenedInfeasible = nan(checkpointShape);
    S.checkpointPairGuidedTightened = nan(checkpointShape);
    S.checkpointPairTrainingEvents = nan(checkpointShape);
    S.checkpointPairGenerationEvents = nan(checkpointShape);
    S.checkpointPairUseEvents = nan(checkpointShape);
    S.checkpointPairDonors = nan(checkpointShape);
    S.checkpointPairFallback = nan(checkpointShape);
    S.checkpointPairObjectiveFE = nan(checkpointShape);
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

function Event = emptyPairTrainingEvent()
%EMPTYPAIRTRAININGEVENT One behavior-neutral incremental-training record.

    Event = struct('fe',NaN,'kind',"",'nCritic',NaN,'epochs',NaN, ...
        'trainingPairs',NaN,'changedPairs',NaN,'newRegions',NaN, ...
        'updates',NaN,'pairVisits',NaN,'batchesPerEpoch',NaN, ...
        'trainingSeconds',NaN, ...
        'preFeasibleEndpointRMSE',NaN, ...
        'preInfeasibleEndpointRMSE',NaN, ...
        'preAllEndpointRMSE',NaN,'preChangedEndpointRMSE',NaN, ...
        'prePairDifferenceRMSE',NaN,'preSameConditionThickness',NaN, ...
        'preCriticGap',NaN,'postFeasibleEndpointRMSE',NaN, ...
        'postInfeasibleEndpointRMSE',NaN,'postAllEndpointRMSE',NaN, ...
        'postChangedEndpointRMSE',NaN,'postPairDifferenceRMSE',NaN, ...
        'postSameConditionThickness',NaN,'postCriticGap',NaN);
end

function Event = emptyPairArchiveEvent()
%EMPTYPAIRARCHIVEEVENT One compact archive state/update record.

    Event = struct('fe',NaN,'retained',NaN,'active',NaN, ...
        'inactive',NaN,'resumeEligible',NaN,'added',NaN, ...
        'tightenedFeasible',NaN,'tightenedInfeasible',NaN, ...
        'guidedTightenedFeasible',NaN, ...
        'guidedTightenedInfeasible',NaN, ...
        'ordinaryTightenedFeasible',NaN, ...
        'ordinaryTightenedInfeasible',NaN, ...
        'removed',NaN,'pairGapMedian',NaN,'pairGapP90',NaN);
end

function Event = emptyPairGenerationEvent()
%EMPTYPAIRGENERATIONEVENT One raw-to-donor funnel record.

    Event = struct('fe',NaN,'modelVersion',NaN,'rawCount',NaN, ...
        'invalidCount',NaN,'matchFailures',NaN, ...
        'sphereRejectCount',NaN,'spherePassCount',NaN, ...
        'objectiveCandidateCount',NaN, ...
        'localDominancePass',NaN,'corridorPass',NaN, ...
        'jointPassCount',NaN,'keptCount',NaN,'objectiveFE',NaN, ...
        'rawConditions',NaN,'keptConditions',NaN);
end

function Event = emptyPairUseEvent()
%EMPTYPAIRUSEEVENT One donor-to-evaluated-child record.

    Event = struct('poolFE',NaN,'fe',NaN,'useFE',NaN, ...
        'modelVersion',NaN, ...
        'requested',NaN,'selected',NaN,'fallback',NaN, ...
        'feasible',NaN,'dominating',NaN,'feasibleDominating',NaN, ...
        'survivedP1',NaN);
end

function Event = copyPairModelDiagnostics(Event,Status,statusField,prefix)
%COPYPAIRMODELDIAGNOSTICS Copy fixed-probe diagnostics into one log row.

    if ~isfield(Status,statusField) || ~isstruct(Status.(statusField))
        return;
    end
    Source = Status.(statusField);
    names = {'feasibleEndpointRMSE','infeasibleEndpointRMSE', ...
        'allEndpointRMSE','changedEndpointRMSE', ...
        'pairDifferenceRMSE','sameConditionThickness','criticGap'};
    for i = 1 : numel(names)
        if isfield(Source,names{i})
            target = char(string(prefix)+upper(extractBefore(names{i},2))+ ...
                extractAfter(names{i},1));
            Event.(target) = double(Source.(names{i}));
        end
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
    Snapshot.pairGateActivationRate = safeRatio( ...
        S.pairEligibleEvents,S.pairGateEvents);
    Snapshot.pairModelReadyRate = safeRatio( ...
        S.pairModelReadyEvents,S.pairEligibleEvents);
    Snapshot.meanPairTrainingUpdates = safeRatio( ...
        S.pairTrainingUpdates,S.pairTrainingEvents);
    Snapshot.meanPairTrainingEpochs = safeRatio( ...
        S.pairTrainingEpochs,S.pairTrainingEvents);
    Snapshot.meanPairTrainingPairVisits = safeRatio( ...
        S.pairTrainingPairVisits,S.pairTrainingEvents);
    Snapshot.pairCandidateValidityRate = safeRatio( ...
        S.pairPoolValid,S.rawCandidates);
    Snapshot.pairCandidateSelectionRate = safeRatio( ...
        S.pairPoolSelected,S.rawCandidates);
    Snapshot.pairSpherePassRate = safeRatio( ...
        S.pairSpherePass,S.rawCandidates);
    Snapshot.pairSphereRejectRate = safeRatio( ...
        S.pairSphereReject,S.rawCandidates);
    Snapshot.pairJointPassRate = safeRatio( ...
        S.pairJointPass,S.pairObjectiveCandidates);
    Snapshot.ObjFE = S.pairObjectiveFE;
    Snapshot.PairGuideFullFE = S.pairGuideFullFE;
    Snapshot.PairGuideConFE = S.pairGuideConstraintFE;
    Snapshot.pairCorridorPassRate = safeRatio( ...
        S.pairCorridorPass,S.pairObjectiveCandidates);
    Snapshot.pairMatchFailureRate = safeRatio( ...
        S.pairMatchFailures,S.rawCandidates);
    Snapshot.pairInvalidCandidateRate = safeRatio( ...
        S.pairInvalidCandidates,S.rawCandidates);
    Snapshot.meanPairProjectionRate = safeRatio( ...
        S.pairProjectionRateSum,S.pairProjectionRateCount);
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
    Snapshot.meanPairGuideFg = Snapshot.meanAlpha;
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

function Config = mergePairGuideTrainingExperimentOptions(Config,Options)
%MERGEPAIRGUIDETRAININGEXPERIMENTOPTIONS Overlay one explicit experiment.

    if isempty(Options) || ~isstruct(Options) || isempty(fieldnames(Options))
        return;
    end
    if ~isfield(Config,'guideGenerationMode') || ...
            string(Config.guideGenerationMode) ~= "pair_guide"
        error('CBSPairGuide:TrainingExperimentRequiresPairGuide', ...
            'Pair-guide training experiments require the PairGuide algorithm.');
    end
    Config.pairInitialEpoch = double(Options.initialEpoch);
    Config.pairRetrainEpoch = double(Options.retrainEpoch);
    Config.nCritic = double(Options.nCritic);
    mapping = {'lrG','ganLrG';'lrD','ganLrD';'miniBatch','ganMiniBatch'; ...
        'generatorHidden','generatorHidden';'criticHidden','criticHidden'; ...
        'trainingSigma','trainingSigma';'sampleSigma','sampleSigma';'gpLambda','gpLambda'};
    for k = 1:size(mapping,1)
        if isfield(Options,mapping{k,1}); Config.(mapping{k,2}) = Options.(mapping{k,1}); end
    end
    Config.pairTrainingExperiment = true;
end

function Options = validatePairGuideTrainingExperimentOptions(Options)
%VALIDATEPAIRGUIDETRAININGEXPERIMENTOPTIONS Validate isolated factors.

    required = {'initialEpoch','retrainEpoch','nCritic'};
    optional = {'lrG','lrD','miniBatch','generatorHidden','criticHidden', ...
        'trainingSigma','sampleSigma','gpLambda'};
    if ~isstruct(Options) || ~isscalar(Options) || ...
            ~all(isfield(Options,required)) || ...
            any(~ismember(fieldnames(Options),[required,optional]))
        error('CBSPairGuide:BadTrainingExperimentOptions', ...
            'Specify update budgets, nCritic, and supported network/training factors.');
    end
    for i = 1 : numel(required)
        name = required{i};
        value = double(Options.(name));
        minimum = double(strcmp(name,'nCritic'));
        if ~isscalar(value) || ~isfinite(value) || ...
                value < minimum || value ~= round(value)
            error('CBSPairGuide:BadTrainingExperimentOptions', ...
                '%s must be an integer not smaller than %d.',name,minimum);
        end
        Options.(name) = value;
    end
    for name = reshape(string(optional),1,[])
        if ~isfield(Options,name); continue; end
        v = double(Options.(name));
        widths = ismember(name,["generatorHidden","criticHidden"]);
        assert(isnumeric(Options.(name)) && ~isempty(v) && all(isfinite(v),'all') && ...
            (widths && isvector(v) || ~widths && isscalar(v)), ...
            'CBSPairGuide:BadTrainingExperimentOptions','Invalid training factor.');
        if widths || name == "miniBatch"
            assert(all(v >= 1 & v == round(v),'all'));
            if name == "miniBatch"; assert(v >= 2); end
        elseif ismember(name,["lrG","lrD"])
            assert(v > 0);
        else
            assert(v >= 0);
        end
        Options.(name) = v;
    end
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
    elseif isscalar(values)
        value = values(1);
    else
        position = 1+min(max(double(q),0),1)*(numel(values)-1);
        low = floor(position);
        high = ceil(position);
        weight = position-low;
        value = values(low)*(1-weight)+values(high)*weight;
    end
end







function Options = validateObjectiveSnapshotOptions(Options)
%VALIDATEOBJECTIVESNAPSHOTOPTIONS Validate behavior-neutral point capture.

    if nargin < 1 || isempty(Options)
        Options = struct();
    end
    if ~isstruct(Options) || ~isscalar(Options)
        error('CBSRegionGAN:BadObjectiveSnapshotOptions', ...
            'Objective-space snapshot options must be a scalar struct.');
    end
    defaults = struct('enabled',true,'targetFE',zeros(1,0), ...
        'expectedRawCount',500,'expectedGuidedCount',20);
    names = fieldnames(defaults);
    for i = 1 : numel(names)
        if ~isfield(Options,names{i})
            Options.(names{i}) = defaults.(names{i});
        end
    end
    validFields = string(names);
    supplied = string(fieldnames(Options));
    if any(~ismember(supplied,validFields)) || ...
            ~isscalar(Options.enabled)
        error('CBSRegionGAN:BadObjectiveSnapshotOptions', ...
            'Unexpected objective-space snapshot option.');
    end
    targetFE = unique(double(Options.targetFE(:)'),'sorted');
    counts = double([Options.expectedRawCount, ...
        Options.expectedGuidedCount]);
    if logical(Options.enabled) && ...
            (isempty(targetFE) || any(~isfinite(targetFE) | targetFE <= 0))
        error('CBSRegionGAN:BadObjectiveSnapshotTargets', ...
            'Snapshot target FE values must be finite and positive.');
    end
    if any(~isfinite(counts) | counts < 1 | counts ~= round(counts))
        error('CBSRegionGAN:BadObjectiveSnapshotCounts', ...
            'Expected raw and guided counts must be positive integers.');
    end
    Options.enabled = logical(Options.enabled);
    Options.targetFE = targetFE;
    Options.expectedRawCount = counts(1);
    Options.expectedGuidedCount = counts(2);
end





























function value = safeRatio(numerator,denominator)
%SAFERATIO Return NaN when an aggregate has no observations.

    if denominator > 0
        value = double(numerator)/double(denominator);
    else
        value = NaN;
    end
end

function seed = pairRunSeed()
%PAIRRUNSEED Derive independent streams without advancing ordinary evolution.
    state = rng;
    data = state.State;
    if iscell(data); data = data{1}; end
    seed = mod(sum(double(data(1:min(16,numel(data))))),2^32-1);
end

function key = evaluationKey(x,y,feasible)
%EVALUATIONKEY Audit-only identity, never used to rank or train a pair.
    key = sprintf('%.17g,',[x,y,double(feasible)]);
end

function indexEvaluations(Index,Record)
%INDEXEVALUATIONS Associate reused endpoints with a real evaluation record.
    for k = 1:size(Record.decisions,1)
        feasible = all(Record.constraints(k,:) <= 0);
        key = evaluationKey(Record.decisions(k,:),Record.objectives(k,:),feasible);
        if isKey(Index,key); continue; end
        origin = Record.origin(min(k,numel(Record.origin)));
        Index(key) = struct('id',Record.firstEvalID+k-1,'origin',origin);
    end
end

function cleanup = pairStreamScope(stream)
%PAIRSTREAMSCOPE Restore the caller's stream after training/query.
    previous = RandStream.getGlobalStream;
    RandStream.setGlobalStream(stream);
    cleanup = onCleanup(@()RandStream.setGlobalStream(previous));
end

function Population = scopedOperator(kind,Problem,Parents,Fitness,count,seed)
%SCOPEDOPERATOR Ordinary operators have independent generation/slot streams.
    if count <= 0
        Population = Parents([]);
        return;
    end
    stream = RandStream('mt19937ar','Seed',mod(seed,2^32-1));
    cleanup = pairStreamScope(stream); %#ok<NASGU>
    if kind == "ga"
        Population = gaOffspring(Problem,Parents,Fitness,count);
    else
        Population = deOffspring(Problem,Parents,Fitness,count);
    end
    clear cleanup;
end

function Offspring = scopedBackbone(Problem,Population,Fitness,count,seed,generation)
%SCOPEDBACKBONE P2 remains 25% GA and 75% ordinary DE.
    ga = round(0.25*count);
    A = scopedOperator("ga",Problem,Population,Fitness,ga,seed+1009*generation+3);
    B = scopedOperator("de",Problem,Population,Fitness,count-ga,seed+1009*generation+4);
    Offspring = [A,B];
end

function [Offspring,Trace,Log] = pairOffspring(Problem,Population,Fitness, ...
        count,Pending,Config,seed,generation)
%PAIROFFSPRING Consume exactly the saved q; missing slots use the same DE.
    startFE = Problem.FE;
    Trace = emptyUseTrace();
    Trace.childDecs = zeros(0,Problem.D);
    Trace.childObjs = zeros(0,Problem.M);
    Trace.childCons = zeros(0,1);
    ga = round(0.25*count); quota = round(0.20*count);
    de = count-ga-quota;
    A = scopedOperator("ga",Problem,Population,Fitness,ga,seed+1009*generation+1);
    B = scopedOperator("de",Problem,Population,Fitness,de,seed+1009*generation+2);
    Offspring = [A,B];
    Trace.active = quota > 0; Trace.requested = quota;
    selected = zeros(0,1);
    if isfield(Pending,'decs')
        span = double(Problem.upper)-double(Problem.lower); span(span <= eps) = 1;
        base = [double(Population.decs);double(Offspring.decs)];
        for k = 1:size(Pending.decs,1)
            q = Pending.decs(k,:);
            if any(~isfinite(q)) || any(q < Problem.lower-1e-12) || ...
                    any(q > Problem.upper+1e-12) || ...
                    (Pending.ids(k) > 0 && nnz(Pending.ids(selected) == Pending.ids(k)) >= 2)
                continue;
            end
            if ~isempty(base) && any(sqrt(sum(((base-q)./span).^2,2)) ...
                    <= Config.pairDuplicateTolerance)
                continue;
            end
            selected(end+1,1) = k; %#ok<AGROW>
            base(end+1,:) = q; %#ok<AGROW>
            if numel(selected) >= quota; break; end
        end
    end
    if quota == 0; selected = zeros(0,1); end
    fallback = quota-numel(selected);
    C = scopedOperator("de",Problem,Population,Fitness,fallback,seed+1009*generation+5);
    Offspring = [Offspring,C];
    Trace.fallback = fallback;
    Trace.fallbackNoPool = fallback*double(~isfield(Pending,'decs'));
    Trace.fallbackMapping = fallback-Trace.fallbackNoPool;
    Trace.selected = numel(selected); Trace.mappedValid = numel(selected);
    Trace.productionFE = NaN;
    Trace.productionGeneration = NaN;
    Trace.consumptionGeneration = generation;
    Trace.modelVersion = NaN;
    Trace.generatedXf = zeros(0,Problem.D); Trace.generatedXi = zeros(0,Problem.D);
    Trace.evalIDs = zeros(0,1);
    if ~isempty(selected)
        before = Problem.FE;
        Guided = Problem.Evaluation(Pending.decs(selected,:));
        if ~isequal(double(Guided.decs),Pending.decs(selected,:))
            error('CBSPairGuide:ChangedPending','Evaluation changed a pending native candidate.');
        end
        Offspring = [Offspring,Guided];
        Trace.childDecs = double(Guided.decs);
        Trace.childObjs = double(Guided.objs);
        Trace.childCons = double(Guided.cons);
        Trace.matchedPairIds = Pending.ids(selected);
        Trace.requestedRefs = Pending.refs(selected); Trace.requestedSides = Pending.sides(selected);
        Trace.generatedXf = Pending.xf(selected,:);
        Trace.generatedXi = Pending.xi(selected,:);
        Trace.parentObjs = Pending.yf(selected,:);
        Trace.productionFE = Pending.productionFE;
        Trace.productionGeneration = Pending.productionGeneration;
        Trace.modelVersion = Pending.modelVersion;
        Trace.selectedTargetDecs = Trace.childDecs;
        Trace.selectedCenterDecs = Trace.childDecs;
        Trace.evalIDs = before+(1:numel(selected))';
        Trace.fullGuideFE = numel(selected);
        Trace.constraintGuideFE = numel(selected);
        feasible = all(Guided.cons <= 0,2);
        dominates = all(Trace.childObjs <= Trace.parentObjs+1e-12,2) & ...
            any(Trace.childObjs < Trace.parentObjs-1e-12,2);
        Trace.feasibleChildren = nnz(feasible);
        Trace.dominatingChildren = nnz(dominates);
        Trace.childUseful = nnz(feasible & dominates);
        if all(Pending.ids(selected)==0)
            Trace.dominatingChildren=NaN; Trace.childUseful=NaN;
        end
        Trace.selectedConditions = numel(unique(Pending.refs(selected)));
    end
    Log = evaluationRecord(Offspring,startFE,"P1");
    Log.origin = [repmat("GA",ga,1);repmat("ordinary_DE",de,1); ...
        repmat("fallback_DE",fallback,1);repmat("PairGuide",numel(selected),1)];
end

function Record = evaluationRecord(Population,startFE,origin)
%EVALUATIONRECORD Unique real evaluation IDs live only in the audit.
    Record = struct('firstEvalID',double(startFE)+1, ...
        'lastEvalID',double(startFE)+numel(Population),'origin',origin, ...
        'decisions',double(Population.decs),'objectives',double(Population.objs), ...
        'constraints',double(Population.cons));
end

function Record = cachedPopulation(fe,P1,P2,Archive)
%CACHEDPOPULATION Cache real results at every completed generation.
    Record = struct('observationFE',double(fe),'p1Decs',double(P1.decs), ...
        'p1Objs',double(P1.objs),'p1Cons',double(P1.cons), ...
        'p2Decs',double(P2.decs),'p2Objs',double(P2.objs), ...
        'p2Cons',double(P2.cons),'archive',Archive);
end

function [Raw,Sample] = pairOnlySample(Archive,Info,Problem)
%PAIRONLYSAMPLE Causal control with the identical true interval geometry.
    [~,rows] = ismember(Info.pairIds,Archive.id);
    lower = double(Problem.lower);
    span = double(Problem.upper)-lower; span(span <= eps) = 1;
    f = (Archive.xf(rows,:)-lower)./span;
    i = (Archive.xi(rows,:)-lower)./span;
    alpha = 0.4+0.2*rand(numel(rows),1);
    Raw = lower+((1-alpha).*f+alpha.*i).*span;
    Sample = struct('generatedF',f,'generatedI',i,'alpha',alpha, ...
        'z',zeros(numel(rows),0),'endpointForwardRows',0,'forwardSeconds',0);
end

function Snapshots = pairEvidenceSnapshots(Evidence,Options)
%PAIREVIDENCESNAPSHOTS At-or-before target states, explicit missing/stale clouds.
    Snapshots = struct([]);
    if isempty(fieldnames(Evidence)) || isempty(fieldnames(Options)) || ~Options.enabled
        return;
    end
    times = cellfun(@(p)p.observationFE,Evidence.population);
    for target = reshape(Options.targetFE,1,[])
        p = find(times <= target,1,'last');
        S = struct('targetFE',target,'observationFE',NaN,'actualFE',NaN, ...
            'productionFE',NaN,'consumptionFE',NaN,'poolFE',NaN, ...
            'stale',true,'missing',true,'rawDecs',[],'rawObjs',[],'rawCons',[], ...
            'guidedDecs',[],'guidedObjs',[],'guidedCons',[], ...
            'population1Decs',[],'population1Objs',[],'population1Cons',[], ...
            'population2Objs',[],'population2Cons',[], ...
            'archiveFeasibleDecs',[],'archiveInfeasibleDecs',[], ...
            'archiveFeasibleObjs',[],'archiveInfeasibleObjs',[], ...
            'trainingPairCount',0,'rawCount',0,'guidedCount',0, ...
            'requestedCount',0,'fallbackCount',0);
        if ~isempty(p)
            P = Evidence.population{p};
            S.missing = false; S.observationFE = P.observationFE;
            S.actualFE = P.observationFE;
            S.population1Decs = P.p1Decs; S.population1Objs = P.p1Objs;
            S.population1Cons = P.p1Cons;
            S.population2Objs = P.p2Objs; S.population2Cons = P.p2Cons;
            active = P.archive.active;
            S.archiveFeasibleDecs = P.archive.xf(active,:);
            S.archiveInfeasibleDecs = P.archive.xi(active,:);
            S.archiveFeasibleObjs = P.archive.yf(active,:);
            S.archiveInfeasibleObjs = P.archive.yi(active,:);
            S.trainingPairCount = nnz(active);
            if p > 1
                U = Evidence.generations{p-1}.use;
                S.guidedDecs = U.childDecs; S.guidedObjs = U.childObjs;
                S.guidedCons = U.childCons; S.guidedCount = U.selected;
                S.requestedCount = U.requested; S.fallbackCount = U.fallback;
                S.productionFE = U.productionFE; S.poolFE = U.productionFE;
                S.consumptionFE = P.observationFE;
                q = find(cellfun(@(q)q.productionFE == U.productionFE,Evidence.queries),1,'last');
                if ~isempty(q)
                    S.rawDecs = Evidence.queries{q}.rawDecs;
                    S.rawCount = size(S.rawDecs,1);
                end
            end
            S.stale = P.observationFE < target;
        end
        if isempty(Snapshots)
            Snapshots = S;
        else
            Snapshots(end+1) = orderfields(S,Snapshots); %#ok<AGROW>
        end
    end
end
