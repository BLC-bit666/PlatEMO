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

    methods
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
                Offspring1 = generateRegionDEOffspring( ...
                    Problem,Population1,Fitness1,count1);
                Offspring2 = generateRegionDEOffspring( ...
                    Problem,Population2,Fitness2,count2);

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

                Union1 = [Population1,Population2,Offspring1, ...
                    Offspring2,OffspringG];
                Union2 = Union1;
                [Population1,Fitness1] = EnvironmentalSelection_CBS( ...
                    Union1,Problem.N,true);
                [Population2,Fitness2] = EnvironmentalSelection_CBS( ...
                    Union2,Problem.N,false);
            end
        end
    end
end

function Offspring = generateRegionDEOffspring( ...
        Problem,Population,Fitness,count)
%GENERATEREGIONDEOFFSPRING Call PlatEMO's DE operator within the FE budget.
%   Parent selection is performed by TournamentSelection, and all variation
%   and evaluation are delegated to the platform OperatorDE implementation.

    count = max(0,min(numel(Population),round(double(count))));
    if count == 0
        Offspring = Population([]);
        return;
    end
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
