classdef CBS_RegionWGAN_GP < ALGORITHM
% <2026> <multi> <real> <constrained>
% Reference vector-conditioned boundary WGAN-GP
% Before the CGAN cutoff, the constrained population uses 25% GA, 55%
% ordinary DE, and 20% CGAN-guided DE. Afterwards it keeps 25% GA and 55%
% ordinary DE, while the remaining 20% uses certified feasible boundary
% targets. Missing targets fall back to ordinary DE. The unconstrained
% population uses 25% GA and 75% ordinary DE throughout the run.
% nGen             ---  20 --- Number of region-query slots per training event
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
    end

    methods
        function main(Algorithm,Problem)
            configurePlatEMOUtilityPath();

            Defaults = CBS_RegionWGAN_GP.mainlineDefaults();
            [nGen,zDim,ganIter,ganMiniBatch,nCritic, ...
                minGANTrainCount,sampleSigma] = ...
                Algorithm.ParameterSet( ...
                Defaults.nGen,Defaults.zDim,Defaults.ganIter, ...
                Defaults.ganMiniBatch,Defaults.nCritic, ...
                Defaults.minGANTrainCount,Defaults.sampleSigma);
            Config = Defaults;
            Config.nGen = nGen;
            Config.zDim = zDim;
            Config.ganIter = ganIter;
            Config.ganMiniBatch = ganMiniBatch;
            Config.nCritic = nCritic;
            Config.minGANTrainCount = minGANTrainCount;
            Config.sampleSigma = sampleSigma;
            Algorithm.runMainline(Problem,Config);
        end
    end

    methods(Static)
        function Defaults = mainlineDefaults()
        %MAINLINEDEFAULTS Return public parameters and fixed mainline values.
            Defaults = struct( ...
                'nGen',20, ...
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
                'guideCandidatesPerSlot',5, ...
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

        function runMainline(Algorithm,Problem,Config)
        %RUNMAINLINE Execute the unique pairflag CGAN-guided search.
            Algorithm.BoundaryTargetXf = [];
            Algorithm.BoundaryTargetYf = [];
            nGen = max(0,round(double(Config.nGen)));
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
                Offspring1 = generateGuidedOffspring(Problem, ...
                    Population1,Fitness1,count1,GuideDecs,GuideRefs, ...
                    GuideSlots,GuideObjs,strictGuideRefs,GuideRefScale,W, ...
                    plainDE);
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
                    if ~isempty(BMem) && nGen > 0
                        [TrainX,TrainC,QueryRefs] = ...
                            BuildBoundaryDataset_RC(BMem,W,Problem);
                        eligible = size(TrainX,1) >= ...
                            max(minBoundaryLength,minGANTrainCount) && ...
                            ~isempty(QueryRefs);
                        if eligible
                            [SampleC,SampleRefs] = RunRegionGAN_RC( ...
                                'regionquerysamples',QueryRefs,W,nGen);
                            SampleSlots = (1:size(SampleC,1))';
                            repeats = Config.guideCandidatesPerSlot;
                            SampleC = repelem(SampleC,repeats,1);
                            SampleRefs = repelem(SampleRefs,repeats,1);
                            SampleSlots = repelem(SampleSlots,repeats,1);
                            % The generator is queried only on the feasible
                            % side; paired infeasible rows remain training data.
                            QueryC = [SampleC,ones(size(SampleC,1),1)];
                            [GAN,RawDec] = RunRegionGAN_RC( ...
                                'trainandsample',GAN,TrainX,TrainC, ...
                                QueryC,Problem,GANOptions);
                            if ~isempty(RawDec)
                                GuideDecs = RawDec;
                                GuideRefs = reshape(SampleRefs( ...
                                    1:size(RawDec,1)),[],1);
                                GuideSlots = reshape(SampleSlots( ...
                                    1:size(RawDec,1)),[],1);
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
            end
        end
    end
end

function Offspring = generateGuidedOffspring( ...
        Problem,Population,Fitness,count,GuideDecs,GuideRefs,GuideSlots, ...
        GuideObjs,strictGuideRefs,GuideRefScale,W,plainDEShare)
%GENERATEGUIDEDOFFSPRING GA + plain DE + target-guided DE offspring.
%   The 20%% guided share uses CGAN targets before the cutoff and certified
%   feasible boundary targets afterwards. CGAN query slots own five
%   unevaluated candidates; each certified target owns one slot. Candidate
%   movement is matched to the feasible parent's local spacing. Missing
%   targets always fall back to ordinary DE.

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
    if isempty(GuideDecs) || isempty(GuideSlots) || ~any(feasible)
        Offspring = [Offspring,deOffspring( ...
            Problem,Population,Fitness,guidedCount)];
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
    FeasDecs = FeasiblePop.decs;
    [plannedF,callGroup] = mainlineGuideScales(guidedCount);
    [childGuide,parentRow,childSlot] = selectGuideCandidates( ...
        Problem,FeasDecs,feasRefs,feasFitness,GuideDecs,GuideRefs, ...
        GuideSlots,W,plannedF,guidedCount,strictGuideRefs);
    selectedCount = numel(childGuide);
    childF = plannedF(1:selectedCount);
    childGroup = callGroup(1:selectedCount);
    if numel(unique(childGuide)) ~= selectedCount || ...
            numel(unique(childSlot)) ~= selectedCount
        error('CBSRegionGAN:GuidePoolSelectionReuse', ...
            'Each guide slot and selected candidate must be unique.');
    end

    fallbackCount = guidedCount-selectedCount;
    if fallbackCount > 0
        Offspring = [Offspring,deOffspring( ...
            Problem,Population,Fitness,fallbackCount)];
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
        A = FeasDecs(parentRow(rows),:);
        G = GuideDecs(childGuide(rows),:);
        ChildDecs(rows,:) = OperatorDE(Problem,A,G,A,{1,f,1,20});
    end
    if selectedCount > 0
        Offspring = [Offspring,Problem.Evaluation(ChildDecs)];
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
