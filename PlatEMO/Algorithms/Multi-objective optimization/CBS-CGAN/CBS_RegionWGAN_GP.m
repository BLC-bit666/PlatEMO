classdef CBS_RegionWGAN_GP < ALGORITHM
% <2026> <multi> <real> <constrained>
% Reference vector-conditioned boundary WGAN-GP
% nGen            ---  30 --- Number of generated solutions per eligible event
% zDim            ---   6 --- Dimension of the generator noise vector
% ganIter         --- 100 --- Generator updates per eligible training event
% ganMiniBatch    ---  32 --- Mini-batch size of WGAN-GP training
% nCritic         ---   4 --- Critic updates per generator update
% minGANTrainCount ---  32 --- Minimum feasible-anchor rows required for training
% sampleSigma     --- 0.3 --- Standard deviation of generator sampling noise

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

    properties(Access = private,Transient)
        operatorMode = "ga_de_half";  % Mainline offspring operator (S2)
        boundarySearch = "on";        % Mainline post-stop line search (BLS)
    end

    methods
        function Algorithm = CBS_RegionWGAN_GP(varargin)
        %CBS_REGIONWGAN_GP Construct the algorithm and optional switches.
            Algorithm@ALGORITHM(varargin{:});
            Algorithm.operatorMode = readOperatorMode(varargin);
            Algorithm.boundarySearch = readBoundarySearch(varargin);
        end

        function mode = effectiveOperatorMode(Algorithm)
        %EFFECTIVEOPERATORMODE Return the live offspring-operator mode.
            mode = Algorithm.operatorMode;
        end

        function state = effectiveBoundarySearch(Algorithm)
        %EFFECTIVEBOUNDARYSEARCH Return the live boundary-search switch.
            state = Algorithm.boundarySearch;
        end

        function main(Algorithm,Problem)
            configurePlatEMOUtilityPath();

            %% Parameter setting
            Defaults = CBS_RegionWGAN_GP.mainlineDefaults();
            [nGen,zDim,ganIter,ganMiniBatch,nCritic, ...
                minGANTrainCount,sampleSigma] = Algorithm.ParameterSet( ...
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

            %% Optimization
            Algorithm.runRegionGAN(Problem,Config);
        end
    end

    methods(Static)
        function Defaults = mainlineDefaults()
        %MAINLINEDEFAULTS Return public defaults and fixed mainline constants.
            Defaults = struct( ...
                'nGen',30, ...
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
                'generatorHidden',[32 32], ...
                'criticHidden',[32 32]);
        end
    end

    methods(Access = private)
        function runRegionGAN(Algorithm,Problem,Config)
        %RUNREGIONGAN Execute the two-population anchor-guided search.
            nGen = max(0,round(double(Config.nGen)));
            refDivisor = max(1,round(double(Config.refDivisor)));
            minBoundaryLength = max(1,round(double( ...
                Config.minBoundaryLength)));
            minGANTrainCount = max(1,round(double( ...
                Config.minGANTrainCount)));
            ganFELimit = Config.ganStopFraction*Problem.maxFE;

            %% Generate reference vectors and initialize state
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
            Fitness1 = CalFitness_CBS( ...
                Population1.objs,Population1.cons);
            Fitness2 = CalFitness_CBS(Population2.objs);
            BMem = [];
            GAN = [];

            %% Optimization
            while Algorithm.NotTerminated(Population1)
                remainingFE = max(0,Problem.maxFE-Problem.FE);
                deBudget = min(2*Problem.N,remainingFE);
                count1 = min(Problem.N,ceil(deBudget/2));
                count2 = min(Problem.N,floor(deBudget/2));
                Offspring1 = generateRegionOffspring(Problem, ...
                    Population1,Fitness1,count1,Algorithm.operatorMode);
                Offspring2 = generateRegionOffspring(Problem, ...
                    Population2,Fitness2,count2,Algorithm.operatorMode);

                OffspringG = Offspring1([]);
                if Problem.FE < ganFELimit
                    % Update the boundary memory before training so every
                    % evaluated offspring can contribute to the current
                    % event. After the configured stop point the memory,
                    % training, and sampling are all skipped and the saved
                    % budget flows back into the DE loop.
                    BMem = UpdateBoundaryMemory_RC(BMem,Population1, ...
                        Offspring1,Population2,Offspring2,W,MemOptions);
                    remainingFE = max(0,Problem.maxFE-Problem.FE);
                    eventBudget = min(nGen,remainingFE);
                    if ~isempty(BMem) && eventBudget > 0
                        [TrainX,TrainC,QueryRefs] = ...
                            BuildBoundaryDataset_RC(BMem,W,Problem);
                        eligible = size(TrainX,1) >= ...
                            max(minBoundaryLength,minGANTrainCount) && ...
                            ~isempty(QueryRefs);
                        if eligible
                            SampleC = RunRegionGAN_RC( ...
                                'regionquerysamples',QueryRefs,W, ...
                                eventBudget);
                            [GAN,RawDec] = RunRegionGAN_RC( ...
                                'trainandsample', ...
                                GAN,TrainX,TrainC,SampleC,Problem, ...
                                GANOptions);
                            remainingFE = max(0,Problem.maxFE-Problem.FE);
                            if size(RawDec,1) > remainingFE
                                RawDec = RawDec(1:remainingFE,:);
                            end
                            if ~isempty(RawDec)
                                OffspringG = Problem.Evaluation(RawDec);
                            end
                        end
                    end
                end

                OffspringL = Offspring1([]);
                if Algorithm.boundarySearch == "on" && ...
                        Problem.FE >= ganFELimit
                    remainingFE = max(0,Problem.maxFE-Problem.FE);
                    blsBudget = min(20,remainingFE);
                    if blsBudget > 0
                        OffspringL = boundaryLineSearch( ...
                            Problem,Population1,Population2,blsBudget);
                    end
                end

                Union1 = [Population1,Population2,Offspring1, ...
                    Offspring2,OffspringG,OffspringL];
                Union2 = Union1;
                [Population1,Fitness1] = EnvironmentalSelection_CBS( ...
                    Union1,Problem.N,true);
                [Population2,Fitness2] = EnvironmentalSelection_CBS( ...
                    Union2,Problem.N,false);
            end
        end
    end
end

function state = readBoundarySearch(argumentList)
%READBOUNDARYSEARCH Read the optional post-stop line-search switch.

    state = "on";
    names = argumentList(1:2:max(0,numel(argumentList)-1));
    index = find(cellfun(@(x)ischar(x) && ...
        strcmp(x,'boundarySearch'),names),1,'last');
    if isempty(index)
        return;
    end
    state = lower(strtrim(string(argumentList{2*index})));
    if ~(isscalar(state) && ismember(state,["off","on"]))
        error('CBSRegionGAN:BadBoundarySearch', ...
            'boundarySearch must be off or on.');
    end
end

function Offspring = boundaryLineSearch(Problem,Population1,Population2, ...
        budget)
%BOUNDARYLINESEARCH Membership-oracle line search after the CGAN stops.
%   Primitive 1 bisects feasible-infeasible decision segments so that the
%   feasible-side iterates land next to the constraint boundary; primitive
%   2 evaluates the decision midpoint of the sparsest adjacent feasible
%   nondominated pair, with one bisection repair step when the midpoint is
%   infeasible. Both follow the classic boundary-operator idea
%   (Michalewicz et al.; repair by binary interpolation, GECCO 2007) and
%   only require the single-bit feasibility oracle. Engineering utility,
%   not an algorithmic contribution.

    Offspring = Population1([]);
    All = [Population1,Population2];
    cv = sum(max(0,All.cons),2);
    Feasible = All(cv <= 0);
    if isempty(Feasible) || budget <= 0
        return;
    end
    front = NDSort(Feasible.objs,1) == 1;
    Front = Feasible(front);
    FrontDecs = Front.decs;
    Infeasible = All(cv > 0);
    used = 0;

    %% Primitive 1: pin anchors onto the boundary by bisection
    if ~isempty(Infeasible)
        InfDecs = Infeasible.decs;
        for anchor = 1 : min(3,size(FrontDecs,1))
            if used >= budget
                break;
            end
            pick = randi(size(FrontDecs,1));
            feasiblePoint = FrontDecs(pick,:);
            distance2 = sum((InfDecs-feasiblePoint).^2,2);
            [~,nearest] = min(distance2);
            infeasiblePoint = InfDecs(nearest,:);
            steps = min(4,budget-used);
            for k = 1 : steps
                middle = (feasiblePoint+infeasiblePoint)/2;
                candidate = Problem.Evaluation(middle);
                used = used + 1;
                Offspring = [Offspring,candidate]; %#ok<AGROW>
                if sum(max(0,candidate.cons),2) <= 0
                    feasiblePoint = middle;
                else
                    infeasiblePoint = middle;
                end
            end
        end
    end

    %% Primitive 2: fill the sparsest objective-space gaps
    if used < budget && numel(Front) >= 2
        FrontObjs = Front.objs;
        distance = sqrt(max(0,sum(FrontObjs.^2,2) + ...
            sum(FrontObjs.^2,2)' - 2*(FrontObjs*FrontObjs')));
        distance(1:numel(Front)+1:end) = inf;
        [nearestDist,nearestIdx] = min(distance,[],2);
        [~,order] = sort(nearestDist,'descend');
        for t = 1 : numel(order)
            if used >= budget
                break;
            end
            i = order(t);
            j = nearestIdx(i);
            middle = (FrontDecs(i,:)+FrontDecs(j,:))/2;
            candidate = Problem.Evaluation(middle);
            used = used + 1;
            Offspring = [Offspring,candidate]; %#ok<AGROW>
            if used < budget && sum(max(0,candidate.cons),2) > 0
                repaired = (middle+FrontDecs(i,:))/2;
                candidate = Problem.Evaluation(repaired);
                used = used + 1;
                Offspring = [Offspring,candidate]; %#ok<AGROW>
            end
        end
    end
end

function mode = readOperatorMode(argumentList)
%READOPERATORMODE Read the optional engineering offspring-operator switch.

    mode = "ga_de_half";
    names = argumentList(1:2:max(0,numel(argumentList)-1));
    index = find(cellfun(@(x)ischar(x) && ...
        strcmp(x,'operatorMode'),names),1,'last');
    if isempty(index)
        return;
    end
    mode = lower(strtrim(string(argumentList{2*index})));
    if ~(isscalar(mode) && ismember(mode,["de","imtcmo_de","ga_de_half"]))
        error('CBSRegionGAN:BadOperatorMode', ...
            'operatorMode must be de, imtcmo_de, or ga_de_half.');
    end
end

function Offspring = generateRegionOffspring( ...
        Problem,Population,Fitness,count,mode)
%GENERATEREGIONOFFSPRING Budget-limited offspring under the selected mode.
%   The default "de" branch is the validated mainline path, byte-identical
%   to the previous generateRegionDEOffspring. The two engineering
%   alternatives keep exactly the same evaluation budget: "imtcmo_de" uses
%   half DE/rand/1 plus half DE/pbest/1 with per-solution random F and CR,
%   and "ga_de_half" replaces half of the offspring with platform SBX+PM.

    count = max(0,min(numel(Population),round(double(count))));
    if count == 0
        Offspring = Population([]);
        return;
    end
    if mode == "imtcmo_de" && numel(Population) >= 5
        Offspring = randPBestDEOffspring(Problem,Population,Fitness,count);
    elseif mode == "ga_de_half"
        gaCount = ceil(count/2);
        Offspring = gaHalfOffspring(Problem,Population,Fitness,gaCount);
        if count > gaCount
            Offspring = [Offspring,classicDEOffspring( ...
                Problem,Population,Fitness,count-gaCount)];
        end
    else
        Offspring = classicDEOffspring(Problem,Population,Fitness,count);
    end
end

function Offspring = classicDEOffspring(Problem,Population,Fitness,count)
%CLASSICDEOFFSPRING The validated mainline DE path (unchanged wiring).

    matingPool = platformTournamentSelection(2,2*count,Fitness);
    if count == numel(Population)
        base = Population;
    else
        base = Population(randperm(numel(Population),count));
    end
    Offspring = OperatorDE(Problem,base, ...
        Population(matingPool(1:count)), ...
        Population(matingPool(count+1:end)));
end

function Offspring = gaHalfOffspring(Problem,Population,Fitness,count)
%GAHALFOFFSPRING SBX+PM offspring through the platform half operator.

    matingPool = platformTournamentSelection(2,2*count,Fitness);
    Offspring = OperatorGAhalf(Problem,Population(matingPool));
end

function Offspring = randPBestDEOffspring(Problem,Population,Fitness,count)
%RANDPBESTDEOFFSPRING Canonical DE/rand/1 plus DE/pbest/1 offspring.
%   The operator recipe (random F in {0.6,0.8,1.0} and CR in {0.1,0.2,1.0}
%   per solution, rand/1 plus pbest/1 halves with p = 0.1) is ported from
%   the bundled IMTCMO implementation by Kangjia Qiao. The rand/1 mates are
%   plain mutually distinct random triples; the IMTCMO-specific neighbor
%   pairing strategy is intentionally not adopted.

    n = numel(Population);
    randCount = ceil(count/2);
    bestCount = count - randCount;
    Decs = Population.decs;

    order = randperm(n);
    [r1,r2,~] = distinctRandomTriples(n,order);
    Rand1 = deVariation(Problem, ...
        Decs(order(1:randCount),:), ...
        Decs(r1(1:randCount),:) - Decs(r2(1:randCount),:));

    if bestCount > 0
        perm = randperm(n);
        [q1,q2,~] = distinctRandomTriples(n,perm);
        base = Decs(perm(1:bestCount),:);
        pNP = max(round(0.1*n),2);
        [~,indBest] = sort(reshape(Fitness,[],1),'ascend');
        randindex = max(1,ceil(rand(1,bestCount)*pNP));
        pbest = Decs(indBest(randindex),:);
        PBest1 = deVariation(Problem,base, ...
            pbest - base + Decs(q1(1:bestCount),:) - Decs(q2(1:bestCount),:));
    else
        PBest1 = zeros(0,size(Decs,2));
    end
    Offspring = Problem.Evaluation([Rand1;PBest1]);
end

function Offspring = deVariation(Problem,Base,Difference)
%DEVARIATION Random-parameter DE variation with polynomial mutation.
%   Per-row F and CR are drawn from {0.6,0.8,1.0} and {0.1,0.2,1.0} as in
%   the bundled IMTCMO operators; polynomial mutation follows the platform
%   convention with proM = 1 and disM = 20.

    [N,D] = size(Base);
    [proM,disM] = deal(1,20);
    Fm = [0.6,0.8,1.0];
    CRm = [0.1,0.2,1.0];
    F = Fm(randi(3,N,1))';
    F = F(:,ones(1,D));
    CR = CRm(randi(3,N,1))';
    Site = rand(N,D) < CR;
    Offspring = Base;
    Offspring(Site) = Offspring(Site) + F(Site).*Difference(Site);

    Lower = repmat(Problem.lower,N,1);
    Upper = repmat(Problem.upper,N,1);
    Site = rand(N,D) < proM/D;
    mu = rand(N,D);
    temp = Site & mu <= 0.5;
    Offspring = min(max(Offspring,Lower),Upper);
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).* ...
        ((2.*mu(temp)+(1-2.*mu(temp)).* ...
        (1-(Offspring(temp)-Lower(temp))./(Upper(temp)-Lower(temp))) ...
        .^(disM+1)).^(1/(disM+1))-1);
    temp = Site & mu > 0.5;
    Offspring(temp) = Offspring(temp)+(Upper(temp)-Lower(temp)).* ...
        (1-(2.*(1-mu(temp))+2.*(mu(temp)-0.5).* ...
        (1-(Upper(temp)-Offspring(temp))./(Upper(temp)-Lower(temp))) ...
        .^(disM+1)).^(1/(disM+1)));
end

function [r1,r2,r3] = distinctRandomTriples(n,r0)
%DISTINCTRANDOMTRIPLES Random indices distinct from r0 and one another.
%   Ported from the bundled IMTCMO gnR1R2R3 helper.

    count = numel(r0);
    r1 = floor(rand(1,count)*n)+1;
    for i = 1 : 999
        pos = r1 == r0;
        if ~any(pos); break; end
        r1(pos) = floor(rand(1,sum(pos))*n)+1;
    end
    r2 = floor(rand(1,count)*n)+1;
    for i = 1 : 999
        pos = (r2 == r1) | (r2 == r0);
        if ~any(pos); break; end
        r2(pos) = floor(rand(1,sum(pos))*n)+1;
    end
    r3 = floor(rand(1,count)*n)+1;
    for i = 1 : 999
        pos = (r3 == r1) | (r3 == r2) | (r3 == r0);
        if ~any(pos); break; end
        r3(pos) = floor(rand(1,sum(pos))*n)+1;
    end
end

function index = platformTournamentSelection(K,N,Fitness)
%PLATFORMTOURNAMENTSELECTION Use PlatEMO selection with a stable tie break.
%   The rank adapter preserves the established row-order tie behavior while
%   the stochastic tournament itself is delegated to TournamentSelection.

    [~,rank] = sortrows(reshape(Fitness,[],1));
    [~,rank] = sort(rank);
    index = TournamentSelection(K,N,rank);
end

function configurePlatEMOUtilityPath()
%CONFIGUREPLATEMOUTILITYPATH Give official utility functions precedence.
%   Another bundled algorithm contains a same-named local selection helper;
%   placing the platform utility folder first avoids accidental shadowing.

    algorithmRoot = fileparts(mfilename('fullpath'));
    algorithmsRoot = fileparts(fileparts(algorithmRoot));
    utilityRoot = fullfile(algorithmsRoot,'Utility functions');
    currentSelection = string(which('TournamentSelection'));
    if ~startsWith(currentSelection,string(utilityRoot) + filesep)
        addpath(utilityRoot,'-begin');
    end
end

function Options = regionGANOptions(Config,minTrainCount)
%REGIONGANOPTIONS Convert the validated mainline configuration to WGAN options.

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
