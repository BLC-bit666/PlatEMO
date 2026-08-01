classdef CBS_RegionWGAN_GP < ALGORITHM
% <2026> <multi> <real> <constrained>
% Reference vector-conditioned boundary WGAN-GP
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
                'guideShare',0.2, ...
                'guideCandidatesPerSlot',5, ...
                'calibrationBudget',20, ...
                'generatorHidden',[32 32], ...
                'criticHidden',[32 32]);
        end
    end

    methods(Access = protected)
        function enabled = cganModuleEnabled(~)
        %CGANMODULEENABLED Keep the complete CGAN module on in mainline.
            enabled = true;
        end

        function enabled = randomGuideSlotsEnabled( ...
                ~,currentFE,ganFELimit) %#ok<INUSD>
        %RANDOMGUIDESLOTSENABLED Mainline never replaces guides by random rows.
            enabled = false;
        end

        function share = plainGAShare(~,guidedShare,currentFE,ganFELimit)
        %PLAINGASHARE 40% GA during the CGAN phase, 25% GA afterwards.
        %   The refinement phase after the CGAN cutoff generates 25% GA and
        %   75% plain DE offspring in the first population (adopted from
        %   the 2026-08 B1 configuration screening).
            if currentFE >= ganFELimit
                share = 0.25;
            else
                share = (1-double(guidedShare))/2;
            end
        end

        function params = deParameterCycle(~,currentFE,ganFELimit) %#ok<INUSD>
        %DEPARAMETERCYCLE Mainline always uses the platform default DE setting.
            params = {};
        end

        function fraction = ganStopFractionValue(~,configFraction)
        %GANSTOPFRACTIONVALUE Mainline keeps the configured stop fraction.
            fraction = double(configFraction);
        end

        function budget = calibrationBudgetNow( ...
                ~,configBudget,currentFE,ganFELimit) %#ok<INUSD>
        %CALIBRATIONBUDGETNOW Mainline keeps calibration active all run.
            budget = double(configBudget);
        end
    end

    methods(Access = private)
        function runMainline(Algorithm,Problem,Config)
        %RUNMAINLINE Execute the unique pairflag CGAN-guided search.
            nGen = max(0,round(double(Config.nGen)));
            refDivisor = max(1,round(double(Config.refDivisor)));
            minBoundaryLength = max(1,round(double( ...
                Config.minBoundaryLength)));
            minGANTrainCount = max(1,round(double( ...
                Config.minGANTrainCount)));
            ganFELimit = Algorithm.ganStopFractionValue( ...
                Config.ganStopFraction)*Problem.maxFE;
            cganEnabled = Algorithm.cganModuleEnabled();

            %% Reference vectors, models, and two coevolving populations
            if cganEnabled
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
            else
                W = zeros(0,Problem.M);
                MemOptions = struct();
                GANOptions = struct();
            end

            Population1 = Problem.Initialization();
            Population2 = Problem.Initialization();
            Fitness1 = CalFitness_CBS(Population1.objs,Population1.cons);
            Fitness2 = CalFitness_CBS(Population2.objs);
            BMem = [];
            GAN = [];
            GuideDecs = zeros(0,Problem.D);
            GuideRefs = zeros(0,1);
            GuideSlots = zeros(0,1);

            %% Optimization
            while Algorithm.NotTerminated(Population1)
                randomGuideSlots = Algorithm.randomGuideSlotsEnabled( ...
                    double(Problem.FE),ganFELimit);
                plainGA = Algorithm.plainGAShare(Config.guideShare, ...
                    double(Problem.FE),ganFELimit);
                deParams = Algorithm.deParameterCycle( ...
                    double(Problem.FE),ganFELimit);
                remainingFE = max(0,Problem.maxFE-Problem.FE);
                offspringBudget = min(2*Problem.N,remainingFE);
                count1 = min(Problem.N,ceil(offspringBudget/2));
                count2 = min(Problem.N,floor(offspringBudget/2));

                if Problem.FE >= ganFELimit
                    GuideDecs = zeros(0,Problem.D);
                    GuideRefs = zeros(0,1);
                    GuideSlots = zeros(0,1);
                end
                Offspring1 = generateGuidedOffspring(Problem, ...
                    Population1,Fitness1,count1,GuideDecs,GuideRefs, ...
                    GuideSlots,W,Config.guideShare,randomGuideSlots, ...
                    plainGA,deParams);
                % A generated pool guides exactly one generation.
                GuideDecs = zeros(0,Problem.D);
                GuideRefs = zeros(0,1);
                GuideSlots = zeros(0,1);
                Offspring2 = generateBackboneOffspring(Problem, ...
                    Population2,Fitness2,count2,deParams);

                %% Active boundary calibration (real evaluations)
                Calibration = Offspring1([]);
                remainingFE = max(0,Problem.maxFE-Problem.FE);
                calBudget = Algorithm.calibrationBudgetNow( ...
                    Config.calibrationBudget,double(Problem.FE),ganFELimit);
                budget = min(calBudget,remainingFE);
                if budget > 0
                    Calibration = RefineBoundaryObservations_RC( ...
                        Problem,Population1,Population2,budget);
                end

                %% Pairflag training and feasible-side generation
                if cganEnabled && Problem.FE < ganFELimit
                    Harvest1 = Offspring1;
                    if ~isempty(Calibration)
                        Harvest1 = [Offspring1,Calibration];
                    end
                    BMem = UpdateBoundaryMemory_RC(BMem,Population1, ...
                        Harvest1,Population2,Offspring2,W,MemOptions);
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

function Offspring = generateGuidedOffspring(Problem,Population,Fitness, ...
        count,GuideDecs,GuideRefs,GuideSlots,W,guidedShare, ...
        randomGuideSlots,plainGAShare,deParams)
%GENERATEGUIDEDOFFSPRING GA + plain DE + CGAN-guided DE offspring.
%   The mainline shares are 40%% GA + 40%% DE + 20%% CGAN-guided DE; the
%   GA share is supplied by the plainGAShare hook so experimental arms can
%   rebalance it per phase. Each query slot owns five unevaluated
%   candidates. One candidate is selected by matching its previewed
%   movement to the feasible parent's local spacing. The explicit no-CGAN
%   ablation uses uniformly initialized solutions in these slots before
%   the CGAN cutoff; otherwise missing guides fall back to ordinary DE.

    count = max(0,min(numel(Population),round(double(count))));
    if count == 0
        Offspring = Population([]);
        return;
    end
    plainShare = (1-double(guidedShare))/2;
    gaCount = min(count,round(double(plainGAShare)*count));
    plainCount = min(count-gaCount,round(plainShare*count));
    guidedCount = count-gaCount-plainCount;

    if gaCount > 0
        Offspring = gaOffspring(Problem,Population,Fitness,gaCount);
    else
        Offspring = Population([]);
    end
    if plainCount > 0
        PlainOffspring = deOffspring( ...
            Problem,Population,Fitness,plainCount,deParams);
        Offspring = [Offspring,PlainOffspring];
    end

    feasible = sum(max(0,Population.cons),2) <= 0;
    if guidedCount == 0
        return;
    end
    if randomGuideSlots
        Offspring = [Offspring,Problem.Initialization(guidedCount)];
        return;
    end
    if isempty(GuideDecs) || isempty(GuideSlots) || ~any(feasible)
        Offspring = [Offspring,deOffspring( ...
            Problem,Population,Fitness,guidedCount,deParams)];
        return;
    end

    FeasiblePop = Population(feasible);
    feasFitness = reshape(Fitness(feasible),[],1);
    feasRefs = assignReferences(FeasiblePop.objs,W);
    FeasDecs = FeasiblePop.decs;
    [plannedF,callGroup] = mainlineGuideScales(guidedCount);
    [childGuide,parentRow,childSlot] = selectGuideCandidates( ...
        Problem,FeasDecs,feasRefs,feasFitness,GuideDecs,GuideRefs, ...
        GuideSlots,W,plannedF,guidedCount);
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
            Problem,Population,Fitness,fallbackCount,deParams)];
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
        ChildDecs(rows,:) = OperatorDE( ...
            Problem,A,G,A,{1,f,1,20});
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
        GuideDecs,GuideRefs,GuideSlots,W,plannedF,requestedCount)
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

    selectedRows = zeros(0,1);
    parentRows = zeros(0,1);
    selectedSlots = zeros(0,1);
    for slot = 1 : numel(slotOrder)
        if numel(selectedRows) >= requestedCount
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
        refDistance = sqrt(sum((W(feasRefs,:)-W(target,:)).^2,2));
        ties = find(refDistance <= min(refDistance)+1e-12);
        [~,best] = min(feasFitness(ties));
        parent = ties(best);

        next = numel(selectedRows)+1;
        f = plannedF(next);
        A = FeasDecs(parent,:);
        Preview = A + f*(GuideDecs(rows,:)-A);
        Preview = min(max(Preview,lower),upper);
        move = sqrt(sum(((Preview-A)./span).^2,2));
        [~,pick] = min(abs(move-localScale(parent)));

        selectedRows(next,1) = rows(pick);
        parentRows(next,1) = parent;
        selectedSlots(next,1) = slotOrder(slot);
    end
end

function Ref = assignReferences(Y,W)
%ASSIGNREFERENCES Assign objective rows to normalized reference vectors.

    n = size(Y,1);
    minimum = min(Y,[],1);
    span = max(Y,[],1)-minimum;
    span(span <= eps) = 1;
    Yn = (Y-minimum)./span;
    Yn(~isfinite(Yn)) = 0;
    Wn = W./max(sqrt(sum(W.^2,2)),eps);
    NormY = sqrt(sum(Yn.^2,2));
    Yu = Yn./max(NormY,eps);
    [~,Ref] = max(Yu*Wn',[],2);
    zeroRows = NormY <= eps;
    if any(zeroRows)
        distance2 = max(0,sum(Yn(zeroRows,:).^2,2) + ...
            sum(W.^2,2)' - 2*(Yn(zeroRows,:)*W'));
        [~,Ref(zeroRows)] = min(distance2,[],2);
    end
    Ref = reshape(Ref,n,1);
end

function Offspring = generateBackboneOffspring( ...
        Problem,Population,Fitness,count,deParams)
%GENERATEBACKBONEOFFSPRING Half SBX+PM and half ordinary DE.

    count = max(0,min(numel(Population),round(double(count))));
    if count == 0
        Offspring = Population([]);
        return;
    end
    gaCount = ceil(count/2);
    Offspring = gaOffspring(Problem,Population,Fitness,gaCount);
    if count > gaCount
        Offspring = [Offspring,deOffspring( ...
            Problem,Population,Fitness,count-gaCount,deParams)];
    end
end

function Offspring = deOffspring(Problem,Population,Fitness,count,deParams)
%DEOFFSPRING Validated ordinary DE path.
%   deParams is empty for the platform default parameter set; experimental
%   arms may pass an explicit {CR,F,proM,disM} cell instead.

    matingPool = platformTournamentSelection(2,2*count,Fitness);
    if count == numel(Population)
        base = Population;
    else
        base = Population(randperm(numel(Population),count));
    end
    if isempty(deParams)
        Offspring = OperatorDE(Problem,base, ...
            Population(matingPool(1:count)), ...
            Population(matingPool(count+1:end)));
    else
        Offspring = OperatorDE(Problem,base, ...
            Population(matingPool(1:count)), ...
            Population(matingPool(count+1:end)),deParams);
    end
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
