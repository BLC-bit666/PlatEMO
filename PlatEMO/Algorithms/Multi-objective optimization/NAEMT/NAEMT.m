classdef NAEMT < ALGORITHM
% <2025> <multi> <real> <constrained>
% Network-assisted evolutionary multitask framework
% alpha   --- 0.9  --- Accuracy threshold for retraining the MLP
% epsilon --- 0.5  --- Probability threshold in CDPPV
% N1      --- 1000 --- Size of the MLP training set

%------------------------------- Reference --------------------------------
% J. Ma, Y. Zhang, R. Zheng, C. He, A. W. Mohamed, M. Zuo, H. Li, and
% X. Yao. A network-assisted evolutionary multitask framework for
% multi-objective optimization problems with unknown constraints.
% Proceedings of the International Conference on Intelligent Computing,
% Lecture Notes in Computer Science, 2025, 15858: 127-138.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference
% "Ye Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB
% platform for evolutionary multi-objective optimization [educational
% forum], IEEE Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            %% Parameter setting
            [alpha,epsilon,N1] = Algorithm.ParameterSet(0.9,0.5,1000);
            validateattributes(alpha,{'numeric'},{'scalar','real','>=',0,'<=',1});
            validateattributes(epsilon,{'numeric'},{'scalar','real','>',0,'<',1});
            validateattributes(N1,{'numeric'},{'scalar','real','finite','integer','>=',2});
            CR            = 1;
            F             = 0.5;
            proM          = 1;
            disM          = 20;
            archiveLength = 10;
            sampleQuota   = max(1,min(round(0.1*N1),floor(N1/2)));
            if Problem.maxFE < N1 + 2*Problem.N
                error('NAEMT:InsufficientBudget', ...
                    'maxFE must cover N1 training samples and two initial populations.');
            end

            %% Train the initial infeasible-solution-value predictor
            Samples = Problem.Initialization(N1);
            DataX   = Samples.decs;
            DataY   = double(all(Samples.cons<=0,2));
            Model   = NAEMTTrainModel(DataX,DataY);

            %% Generate the main and auxiliary populations
            Population1 = Problem.Initialization();
            Population2 = Problem.Initialization();

            %% Store offspring from the latest ten generations
            ArchiveWindow = cell(1,archiveLength);
            generation    = 0;

            %% Optimization
            while Algorithm.NotTerminated(Population1)
                Offspring1 = NAEMTOperatorDE(Problem,Population1,{CR,F,proM,disM});
                Offspring2 = NAEMTOperatorDE(Problem,Population2,{CR,F,proM,disM});

                generation = generation + 1;
                ArchiveWindow{mod(generation-1,archiveLength)+1} = [Offspring1,Offspring2];

                Population1 = NAEMTMainSelection([Population1,Offspring1,Offspring2],Problem.N,Model,epsilon);
                Population2 = NAEMTAuxiliarySelection([Population2,Offspring2,Offspring1],Problem.N);

                accuracy = AccuracyOnFeasibleOffspring(Model,[Offspring1,Offspring2]);
                if ~isnan(accuracy) && accuracy < alpha && ~all(all(Population1.cons<=0,2))
                    Archive = CollectArchive(ArchiveWindow);
                    archiveFeasible = all(Archive.cons<=0,2);
                    if any(archiveFeasible) && any(~archiveFeasible)
                        [DataX,DataY] = NAEMTUpdateData(DataX,DataY,Archive,N1,sampleQuota);
                        Model         = NAEMTTrainModel(DataX,DataY);
                    end
                end
            end
        end
    end
end

function accuracy = AccuracyOnFeasibleOffspring(Model,Offspring)
% Calculate the paper's difference-based accuracy on feasible offspring.

    feasible = all(Offspring.cons<=0,2);
    if any(feasible)
        predicted = NAEMTPredict(Model,Offspring(feasible).decs);
        if any(~isfinite(predicted))
            error('NAEMT:NonfinitePrediction', ...
                'The MLP returned a nonfinite feasibility prediction.');
        end
        accuracy  = 1 - mean(abs(1-predicted));
    else
        % The paper does not define accuracy when no feasible offspring
        % exists. NaN explicitly denotes the absence of an observable label.
        accuracy = NaN;
    end
end

function Archive = CollectArchive(ArchiveWindow)
% Concatenate the nonempty slots of the rolling archive.

    nonempty = ArchiveWindow(~cellfun(@isempty,ArchiveWindow));
    if isempty(nonempty)
        Archive = [];
    else
        Archive = [nonempty{:}];
    end
end
