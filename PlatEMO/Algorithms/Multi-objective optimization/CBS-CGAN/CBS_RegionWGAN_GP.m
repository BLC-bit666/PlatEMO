classdef CBS_RegionWGAN_GP < ALGORITHM
% <2026> <multi> <real> <constrained>
% Reference vector-conditioned boundary WGAN-GP
% Before the CGAN cutoff, the constrained population uses 25% GA, 55%
% ordinary DE, and 20% CGAN-guided DE. Afterwards it keeps 25% GA and 55%
% ordinary DE, while the remaining 20% uses certified feasible boundary
% targets. Missing targets fall back to ordinary DE. The unconstrained
% population uses 25% GA and 75% ordinary DE throughout the run.
% rawGuideCount    --- 500 --- Raw all-reference CGAN queries per training event
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
    end

    methods
        function Algorithm = CBS_RegionWGAN_GP(varargin)
            Algorithm@ALGORITHM(varargin{:});
        end

        function main(Algorithm,Problem)
            configurePlatEMOUtilityPath();
            Config = Algorithm.algorithmConfiguration();
            Algorithm.runMainline(Problem,Config);
        end

        function Snapshot = guideExperimentSnapshot(Algorithm)
        %GUIDEEXPERIMENTSNAPSHOT Return aggregate CGAN mechanism evidence.
            Snapshot = finalizeGuideStats(Algorithm.GuideStats);
        end
    end

    methods(Access = protected)
        function Config = algorithmConfiguration(Algorithm)
        %ALGORITHMCONFIGURATION Parse the production A2 configuration.
            Defaults = CBS_RegionWGAN_GP.mainlineDefaults();
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
            Config.experimentArm = 2;
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
                'criticHidden',[32 32]);
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

        function recordGenerationTrace(Algorithm,Trace)
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
            Algorithm.GuideStats = S;
        end

        function recordUseTrace(Algorithm,Trace,SelectedPopulation)
        %RECORDUSETRACE Aggregate evaluated CGAN child mechanisms.
            if isempty(Trace) || ~isstruct(Trace) || ~Trace.active
                return;
            end
            S = Algorithm.GuideStats;
            S.useEvents = S.useEvents+1;
            S.guidedRequested = S.guidedRequested+Trace.requested;
            S.guidedSelected = S.guidedSelected+Trace.selected;
            S.guidedFallback = S.guidedFallback+Trace.fallback;
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
            if ~isempty(Trace.childDecs)
                survived = ismember(Trace.childDecs, ...
                    double(SelectedPopulation.decs),'rows');
                S.guidedSurvived = S.guidedSurvived+sum(survived);
            end
            Algorithm.GuideStats = S;
        end

        function runMainline(Algorithm,Problem,Config)
        %RUNMAINLINE Execute the unique pairflag CGAN-guided search.
            Algorithm.BoundaryTargetXf = [];
            Algorithm.BoundaryTargetYf = [];
            Algorithm.GuideStats = emptyGuideStats(Config);
            rawGuideCount = max(0,round(double(Config.rawGuideCount)));
            refDivisor = max(1,round(double(Config.refDivisor)));
            minBoundaryLength = max(1,round(double( ...
                Config.minBoundaryLength)));
            minGANTrainCount = max(1,round(double( ...
                Config.minGANTrainCount)));
            ganFELimit = double(Config.ganStopFraction)*Problem.maxFE;

            %% Reference vectors, models, and two coevolving populations
            [W,~] = UniformPoint( ...
                max(2,round(Problem.N/refDivisor)),Problem.M);
            MemOptions = struct( ...
                'frontDepth',max(1,round(double(Config.frontDepth))), ...
                'pairNeighborRefRadius',max(0,round(double( ...
                    Config.pairNeighborRefRadius))), ...
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
            while Algorithm.NotTerminated(Population1)
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
                    plainDE,Config);
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
                if Problem.FE < ganFELimit
                    Harvest1 = Offspring1;
                    if ~isempty(Calibration)
                        Harvest1 = [Offspring1,Calibration];
                    end
                    [BMem,RefScale] = UpdateBoundaryMemory_RC(BMem, ...
                        Population1,Harvest1,Population2,Offspring2,W, ...
                        MemOptions);
                    if ~isempty(BMem) && rawGuideCount > 0
                        [TrainX,TrainC,QueryRefs] = ...
                            BuildBoundaryDataset_RC(BMem,W,Problem);
                        eligible = size(TrainX,1) >= ...
                            max(minBoundaryLength,minGANTrainCount) && ...
                            ~isempty(QueryRefs);
                        if eligible
                            [GAN,GuideDecs,GuideRefs,GuideSlots,PoolTrace] = ...
                                generateCGANGuidePool(GAN,TrainX,TrainC, ...
                                QueryRefs,W,Problem,GANOptions,Config);
                            Algorithm.recordGenerationTrace(PoolTrace);
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
                Algorithm.recordUseTrace(GuideTrace,Population1);
            end
        end
    end
end

function [GAN,GuideDecs,GuideRefs,GuideSlots,Trace] = ...
        generateCGANGuidePool(GAN,TrainX,TrainC,QueryRefs,W,Problem, ...
        GANOptions,Config)
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
end

function [Offspring,Trace] = generateGuidedOffspring( ...
        Problem,Population,Fitness,count,GuideDecs,GuideRefs,GuideSlots, ...
        GuideObjs,strictGuideRefs,GuideRefScale,W,plainDEShare,Config)
%GENERATEGUIDEDOFFSPRING GA + plain DE + target-guided DE offspring.
%   Before the cutoff, A2 maps critic-filtered candidates to local targets;
%   A0/A1 are available only through the experiment subclass. Certified
%   late targets retain the legacy exact-reference path.

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
    Trace.active = ~strictGuideRefs && ~isempty(GuideDecs);
    Trace.requested = guidedCount;
    if isempty(GuideDecs) || isempty(GuideSlots) || ~any(feasible)
        Offspring = [Offspring,deOffspring( ...
            Problem,Population,Fitness,guidedCount)];
        if Trace.active
            Trace.fallback = guidedCount;
        end
        return;
    end

    FeasiblePop = Population(feasible);
    feasFitness = reshape(Fitness(feasible),[],1);
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
            [FeasiblePop.objs;GuideObjs],W);
        feasRefs = jointRefs(1:numel(FeasiblePop));
        GuideRefs = jointRefs(numel(FeasiblePop)+1:end);
    else
        feasRefs = AssignReferenceVectors_CBS( ...
            FeasiblePop.objs,W,GuideRefScale);
    end
    FeasDecs = double(FeasiblePop.decs);
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
        [ChildDecs,parentRow,selectedRefs] = legacyGuideDecisions( ...
            Problem,FeasDecs,feasRefs,feasFitness,GuideDecs,GuideRefs, ...
            GuideSlots,W,guidedCount,strictGuideRefs);
    end
    selectedCount = size(ChildDecs,1);

    fallbackCount = guidedCount-selectedCount;
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
            parentObjs = double(FeasiblePop(parentRow).objs);
            dominates = all(childObjs <= parentObjs+1e-12,2) & ...
                any(childObjs < parentObjs-1e-12,2);
            Trace.feasibleChildren = sum(feasibleChild);
            Trace.dominatingChildren = sum(dominates);
            Trace.childDecs = ChildDecs;
        end
    end
    if Trace.active
        Trace.selected = selectedCount;
        Trace.fallback = fallbackCount;
        Trace.selectedConditions = numel(unique(selectedRefs));
    end
end

function [ChildDecs,parentRows,selectedRefs] = legacyGuideDecisions( ...
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
        'alpha',zeros(0,1),'h',zeros(0,1),'d',zeros(0,1), ...
        'centerStep',zeros(0,1),'actualStep',zeros(0,1), ...
        'directionCosine',zeros(0,1));
    if ~any(validH)
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
%CONFIGUREPLATEMOUTILITYPATH Give official utility functions precedence.

    algorithmRoot = fileparts(mfilename('fullpath'));
    algorithmsRoot = fileparts(fileparts(algorithmRoot));
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
        'minTrainCount',double(minTrainCount));
end

function Trace = emptyPoolTrace()
%EMPTYPOOLTRACE Empty generation-side mechanism record.

    Trace = struct('active',false,'rawCount',0,'keptCount',0, ...
        'rawConditions',0,'keptConditions',0,'keepIdx',zeros(0,1), ...
        'percentile',zeros(0,1));
end

function Trace = emptyUseTrace()
%EMPTYUSETRACE Empty utilization-side mechanism record.

    Trace = struct('active',false,'requested',0,'selected',0, ...
        'fallback',0,'selectedConditions',0,'mappedValid',0, ...
        'mapDropped',0,'alpha',zeros(0,1),'h',zeros(0,1), ...
        'd',zeros(0,1),'centerStep',zeros(0,1), ...
        'actualStep',zeros(0,1),'directionCosine',zeros(0,1), ...
        'feasibleChildren',0,'dominatingChildren',0, ...
        'childDecs',zeros(0,0));
end

function S = emptyGuideStats(Config)
%EMPTYGUIDESTATS Initialize aggregate experiment counters.

    S = struct( ...
        'arm',double(Config.experimentArm), ...
        'generationMode',string(Config.guideGenerationMode), ...
        'useMode',string(Config.guideUseMode), ...
        'generationEvents',0,'rawCandidates',0,'criticKept',0, ...
        'rawConditionSum',0,'keptConditionSum',0, ...
        'keptPercentileSum',0,'keptPercentileCount',0, ...
        'rejectedPercentileSum',0,'rejectedPercentileCount',0, ...
        'useEvents',0,'guidedRequested',0,'guidedSelected',0, ...
        'guidedFallback',0,'selectedConditionSum',0, ...
        'mappedValid',0,'mapDropped',0,'alphaSum',0,'alphaCount',0, ...
        'hSum',0,'dSum',0,'centerStepSum',0,'actualStepSum',0, ...
        'directionCosineSum',0,'directionCosineCount',0, ...
        'guidedChildren',0,'guidedFeasible',0,'guidedDominating',0, ...
        'guidedSurvived',0);
end

function Snapshot = finalizeGuideStats(S)
%FINALIZEGUIDESTATS Add interpretable rates and means to raw counters.

    if isempty(S) || isempty(fieldnames(S))
        Snapshot = struct();
        return;
    end
    Snapshot = S;
    Snapshot.criticRetentionRate = safeRatio(S.criticKept,S.rawCandidates);
    Snapshot.meanRawConditions = ...
        safeRatio(S.rawConditionSum,S.generationEvents);
    Snapshot.meanKeptConditions = ...
        safeRatio(S.keptConditionSum,S.generationEvents);
    Snapshot.meanKeptPercentile = ...
        safeRatio(S.keptPercentileSum,S.keptPercentileCount);
    Snapshot.meanRejectedPercentile = ...
        safeRatio(S.rejectedPercentileSum,S.rejectedPercentileCount);
    Snapshot.selectionRate = ...
        safeRatio(S.guidedSelected,S.guidedRequested);
    Snapshot.fallbackRate = ...
        safeRatio(S.guidedFallback,S.guidedRequested);
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
    Snapshot.guidedDominanceRate = ...
        safeRatio(S.guidedDominating,S.guidedChildren);
    Snapshot.guidedSurvivalRate = ...
        safeRatio(S.guidedSurvived,S.guidedChildren);
end

function value = safeRatio(numerator,denominator)
%SAFERATIO Return NaN when an aggregate has no observations.

    if denominator > 0
        value = double(numerator)/double(denominator);
    else
        value = NaN;
    end
end
