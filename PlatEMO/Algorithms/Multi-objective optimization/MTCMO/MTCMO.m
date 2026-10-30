classdef MTCMO < ALGORITHM
% <2023> <multi> <real/integer/label/binary/permutation> <constrained>
% type --- 1 --- Type of operator (1. GA 2. DE)
% Multitasking constrained multi-objective optimization

%------------------------------- Reference --------------------------------
% K. Qiao, K. Yu, B. Qu, J. Liang, H. Song, C. Yue, H. Lin, and K. C. Tan.
% Dynamic auxiliary task-based evolutionary multitasking for constrained
% multi-objective optimization. IEEE Transactions on Evolutionary
% Computation, 2023, 27(3): 642-656.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% This function is written by Kangjia Qiao

    methods
        function main(Algorithm,Problem)
            type = Algorithm.ParameterSet(1);
            assert(isnumeric(type) && isscalar(type) && isreal(type) && ismember(type,[1 2]), ...
                'MTCMO:BadOperatorType','type must be 1 (GA) or 2 (DE).');
            utility = fullfile(fileparts(which('ALGORITHM')),'Utility functions');
            addpath(utility,'-begin');
            for name = {'TournamentSelection','OperatorGA','OperatorGAhalf','OperatorDE'}
                assert(strcmp(which(name{1}),fullfile(utility,[name{1},'.m'])), ...
                    'MTCMO:ShadowedOperator','Expected the native Utility functions implementation of %s.',name{1});
            end
            %% Generate random population
            Population1 = Problem.Initialization();
            Fitness1    = CalFitness(Population1.objs,Population1.cons);
            
            Population2 = Problem.Initialization();
            Fitness2    = CalFitness(Population2.objs,Population2.cons);
            
            cons = [Population1.cons;Population2.cons];
            cons(cons<0) = 0;
            VAR0 = max(sum(cons,2));
            if VAR0 == 0
                VAR0 = 1;
            end
            X=0;
            
            %% Optimization
            while Algorithm.NotTerminated(Population1)
                % Udate the epsilon value
                cp  = (-log(VAR0)-6)/log(1-0.5);
                VAR = VAR0*(1-X)^cp;
                % Offspring generation
                MatingPool1 = TournamentSelection(2,Problem.N,Fitness1);
                if type==1
                    Offspring1 = OperatorGAhalf(Problem,Population1(MatingPool1));
                else
                    n = floor(numel(MatingPool1)/2);
                    Offspring1 = OperatorDE(Problem,Population1(1:n), ...
                        Population1(MatingPool1(1:n)),Population1(MatingPool1(n+1:2*n)));
                end
                MatingPool2 = TournamentSelection(2,Problem.N,Fitness2);
                if type==1
                    Offspring2 = OperatorGAhalf(Problem,Population2(MatingPool2));
                else
                    n = floor(numel(MatingPool2)/2);
                    Offspring2 = OperatorDE(Problem,Population2(1:n), ...
                        Population2(MatingPool2(1:n)),Population2(MatingPool2(n+1:2*n)));
                end
                % Environmental selection
                [Population1,Fitness1] = Main_task_EnvironmentalSelection([Population1,Offspring1,Offspring2],Problem.N,true);
                [Population2,Fitness2] = Auxiliray_task_EnvironmentalSelection([Population2,Offspring2,Offspring1],Problem.N,VAR);
                X = X + 1/(Problem.maxFE/Problem.N);
            end
        end
    end
end