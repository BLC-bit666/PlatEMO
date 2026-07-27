classdef EADMMNSGAII < ALGORITHM
% <2024> <multi/many> <real> <constrained>
% EADMM with NSGA-II as the backbone algorithm

%------------------------------- Reference --------------------------------
% S. Li, K. Li, W. Li, and M. Yang. Evolutionary alternating direction
% method of multipliers for constrained multi-objective optimization with
% unknown constraints. IEEE Transactions on Evolutionary Computation,
% 2024, DOI: 10.1109/TEVC.2024.3425629.
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
            %% Check dependencies and the two-population initialization
            EADMMValidateSetup(Problem);

            %% Initialize the constrained and unconstrained populations
            Population1 = Problem.Initialization();
            Population2 = Problem.Initialization();
            [Population1,FrontNo1,CrowdDis1] = EADMMConstrainedSelection(Population1,Problem.N);
            [Population2,FrontNo2,CrowdDis2] = EADMMUnconstrainedSelection(Population2,Problem.N);

            %% Optimization
            while Algorithm.NotTerminated(Population1)
                nOffspring = Problem.N;
                % Modules 1 and 2: evolve the two complementary tasks.
                MatingPool1 = TournamentSelection(2,nOffspring,FrontNo1,-CrowdDis1);
                MatingPool2 = TournamentSelection(2,nOffspring,FrontNo2,-CrowdDis2);
                Offspring1  = OperatorGA(Problem,Population1(MatingPool1),{1,20,1,20});
                Offspring2  = OperatorGA(Problem,Population2(MatingPool2),{1,20,1,20});
                Population1 = EADMMConstrainedSelection([Population1,Offspring1],Problem.N);
                Population2 = EADMMUnconstrainedSelection([Population2,Offspring2],Problem.N);

                % Module 3, Steps 1 and 2: exchange offspring populations.
                [Population1,FrontNo1,CrowdDis1] = EADMMConstrainedSelection([Population1,Offspring2],Problem.N);
                [Population2,FrontNo2,CrowdDis2] = EADMMUnconstrainedSelection([Population2,Offspring1],Problem.N);

                % Module 3, Steps 3 and 4: archive promising main-task
                % offspring and solve the discrepancy subproblem around each.
                Archive = EADMMTemporaryArchive(Offspring1,Population1,Population2);
                if ~isempty(Archive)
                    Checked = EADMMLocalSearch(Problem,Archive,Population1);

                    % Module 3, Step 5: feed local-search results to both tasks.
                    if ~isempty(Checked)
                        [Population1,FrontNo1,CrowdDis1] = EADMMConstrainedSelection([Population1,Checked],Problem.N);
                        [Population2,FrontNo2,CrowdDis2] = EADMMUnconstrainedSelection([Population2,Checked],Problem.N);
                    end
                end
            end
        end
    end
end
