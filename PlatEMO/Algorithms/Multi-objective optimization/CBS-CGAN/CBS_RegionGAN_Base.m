classdef (Abstract) CBS_RegionGAN_Base < ALGORITHM
%CBS_REGIONGAN_BASE Lean evolutionary loop for the fixed WGAN-GP mainline.

    methods(Access = protected)
        function Algorithm = CBS_RegionGAN_Base(varargin)
            Algorithm@ALGORITHM(varargin{:});
        end

        function runRegionGAN(Algorithm,Problem,Config)
            nGen = max(0,round(double(Config.nGen)));
            refDivisor = max(1,round(double(Config.refDivisor)));
            minBoundaryLength = max(1,round(double( ...
                Config.minBoundaryLength)));
            minGANTrainCount = max(1,round(double( ...
                Config.minGANTrainCount)));

            [W,~] = UniformPoint(max(2,round(Problem.N/refDivisor)),Problem.M);
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

            while Algorithm.NotTerminated(Population1)
                remainingFE = max(0,Problem.maxFE-Problem.FE);
                deBudget = min(2*Problem.N,remainingFE);
                count1 = min(Problem.N,ceil(deBudget/2));
                count2 = min(Problem.N,floor(deBudget/2));
                Offspring1 = generateRegionDEOffspring( ...
                    Problem,Population1,Fitness1,count1);
                Offspring2 = generateRegionDEOffspring( ...
                    Problem,Population2,Fitness2,count2);

                BMem = UpdateBoundaryMemory_RC(BMem,Population1, ...
                    Offspring1,Population2,Offspring2,W,MemOptions);
                OffspringG = Offspring1([]);
                remainingFE = max(0,Problem.maxFE-Problem.FE);
                eventBudget = min(nGen,remainingFE);
                if ~isempty(BMem) && eventBudget > 0
                    [TrainX,TrainC,QueryRefs] = ...
                        BuildBoundaryDataset_RC(BMem,W,Problem);
                    eligible = size(TrainX,1) >= ...
                        max(minBoundaryLength,minGANTrainCount) && ...
                        ~isempty(QueryRefs);
                    if eligible
                        SampleC = RunRegionGAN_RC('regionquerysamples', ...
                            QueryRefs,W,eventBudget);
                        [GAN,RawDec] = RunRegionGAN_RC('trainandsample', ...
                            GAN,TrainX,TrainC,SampleC,Problem,GANOptions);
                        remainingFE = max(0,Problem.maxFE-Problem.FE);
                        if size(RawDec,1) > remainingFE
                            RawDec = RawDec(1:remainingFE,:);
                        end
                        if ~isempty(RawDec)
                            OffspringG = Problem.Evaluation(RawDec);
                        end
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

function Offspring = generateRegionDEOffspring(Problem,Population,Fitness,count)
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

function Options = regionGANOptions(Config,minTrainCount)
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
