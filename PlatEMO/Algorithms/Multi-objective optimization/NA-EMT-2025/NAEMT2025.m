classdef NAEMT2025 < ALGORITHM
% <2025> <multi> <real> <constrained>
% Network-Assisted Evolutionary Multitask Framework
% alpha       --- 0.9  --- Accuracy threshold for MLP retraining
% epsilon     --- 0.5  --- Threshold parameter in the CDPPV rule
% N1          --- 1000 --- Number of training samples

%------------------------------- Reference --------------------------------
% J. Ma, Y. Zhang, R. Zheng, C. He, A. W. Mohamed, M. Zuo, H. Li, and
% X. Yao. A Network-Assisted Evolutionary Multitask Framework for
% Multi-objective Optimization Problems with Unknown Constraints.
% Communications in Computer and Information Science, 2025, 15858: 127-138.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2025 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [alpha,epsilon,N1] = Algorithm.ParameterSet(0.9,0.5,1000);
            CR            = 1;
            F             = 0.5;
            proM          = 1;
            disM          = 20;
            archiveLength = 10;
            sampleQuota   = round(0.1*N1);

            %% Train the initial MLP model
            [DataX,DataY] = GenerateInitialData(Problem,N1);
            Model         = TrainISVPSModel(DataX,DataY);

            %% Generate the main and auxiliary populations
            Population1 = Problem.Initialization();
            Population2 = Problem.Initialization();

            %% Archive individuals from the latest 10 generations
            ArchiveWindow = cell(1,archiveLength);
            ArchiveCount  = 0;

            %% Optimization
            while Algorithm.NotTerminated(Population1)
                Offspring1 = OperatorDECurrentToRand1(Problem,Population1,{CR,F,proM,disM});
                Offspring2 = OperatorDECurrentToRand1(Problem,Population2,{CR,F,proM,disM});

                ArchiveCount = ArchiveCount + 1;
                ArchiveWindow{mod(ArchiveCount-1,archiveLength)+1} = [Offspring1,Offspring2];

                Population1 = MainTaskEnvironmentalSelection([Population1,Offspring1,Offspring2],Problem.N,Model,epsilon);
                Population2 = AuxiliaryTaskEnvironmentalSelection([Population2,Offspring2,Offspring1],Problem.N);

                Accuracy = CalculateAccuracyMetric(Model,Offspring1,Offspring2);
                if Accuracy < alpha && ~AllIndividualsFeasible(Population1,Population2)
                    ArchiveSet     = CollectArchiveSet(ArchiveWindow);
                    if CanUpdateTrainingData(ArchiveSet,sampleQuota)
                        [DataX,DataY] = UpdateTrainingData(DataX,DataY,ArchiveSet,N1,sampleQuota);
                        Model         = TrainISVPSModel(DataX,DataY);
                    end
                end
            end
        end
    end
end

function [DataX,DataY] = GenerateInitialData(Problem,N1)
    BatchSize = max(1,round(0.1*N1));
    PoolDec   = zeros(0,Problem.D);
    PoolY     = zeros(0,1);
    while size(PoolDec,1) < N1 || numel(unique(PoolY)) < 2
        Need     = max(N1-size(PoolDec,1),BatchSize);
        Samples  = RandomlySampleSolutions(Problem,Need);
        PoolDec  = [PoolDec; Samples.decs];
        PoolY    = [PoolY; FeasibleLabelFromCons(Samples.cons)];
    end

    FeasibleIndex   = find(PoolY==1);
    InfeasibleIndex = find(PoolY==0);
    Keep            = [FeasibleIndex(randperm(numel(FeasibleIndex),1)); ...
                       InfeasibleIndex(randperm(numel(InfeasibleIndex),1))];
    Remaining       = setdiff((1:size(PoolDec,1))',Keep,'stable');
    if N1 > numel(Keep)
        Remaining = Remaining(randperm(numel(Remaining),N1-numel(Keep)));
        Keep      = [Keep; Remaining];
    end
    Keep  = Keep(randperm(numel(Keep)));
    DataX = PoolDec(Keep,:);
    DataY = PoolY(Keep);
end

function Accuracy = CalculateAccuracyMetric(Model,Offspring1,Offspring2)
    Offspring = [Offspring1,Offspring2];
    Feasible  = IsFeasibleByCons(Offspring.cons);
    Predicted = PredictISVPS(Model,Offspring(Feasible).decs);
    Accuracy  = 1 - mean(abs(1 - Predicted));
end

function Flag = AllIndividualsFeasible(Population1,Population2)
    Population = [Population1,Population2];
    Flag       = all(IsFeasibleByCons(Population.cons));
end

function ArchiveSet = CollectArchiveSet(ArchiveWindow)
    NonEmpty = ArchiveWindow(~cellfun(@isempty,ArchiveWindow));
    if isempty(NonEmpty)
        ArchiveSet = [];
    else
        ArchiveSet = [NonEmpty{:}];
    end
end

function Flag = CanUpdateTrainingData(ArchiveSet,SampleQuota)
    if isempty(ArchiveSet)
        Flag = false;
        return;
    end
    FeasibleCount   = sum(IsFeasibleByCons(ArchiveSet.cons));
    InfeasibleCount = numel(ArchiveSet) - FeasibleCount;
    Flag            = FeasibleCount >= SampleQuota && InfeasibleCount >= SampleQuota;
end

function Feasible = IsFeasibleByCons(Cons)
    Feasible = ~any(Cons>0,2);
end

function Label = FeasibleLabelFromCons(Cons)
    Label = double(IsFeasibleByCons(Cons));
end

function Samples = RandomlySampleSolutions(Problem,N)
    Lower  = repmat(Problem.lower,N,1);
    Upper  = repmat(Problem.upper,N,1);
    Dec    = Lower + rand(N,Problem.D).*(Upper-Lower);
    Samples = Problem.Evaluation(Dec);
end
