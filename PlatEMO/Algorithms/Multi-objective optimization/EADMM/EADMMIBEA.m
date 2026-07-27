classdef EADMMIBEA < ALGORITHM
% <2024> <multi/many> <real> <constrained>
% EADMM with IBEA as the backbone algorithm
% kappa --- 0.05 --- Fitness scaling factor of IBEA

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
            %% Parameter setting
            kappa = Algorithm.ParameterSet(0.05);
            validateattributes(kappa,{'numeric'},{'scalar','real','finite','positive'});
            EADMMValidateSetup(Problem);

            %% Initialize the constrained and unconstrained populations
            Population1 = Problem.Initialization();
            Population2 = Problem.Initialization();

            %% Optimization
            while Algorithm.NotTerminated(Population1)
                nOffspring = Problem.N;

                % Modules 1 and 2: retain the original IBEA mating scheme.
                Fitness1   = EADMMIBEACalFitness(Population1.objs,kappa);
                Fitness2   = EADMMIBEACalFitness(Population2.objs,kappa);
                MatingPool1 = TournamentSelection(2,nOffspring,-Fitness1);
                MatingPool2 = TournamentSelection(2,nOffspring,-Fitness2);
                Offspring1  = OperatorGA(Problem,Population1(MatingPool1),{1,20,1,20});
                Offspring2  = OperatorGA(Problem,Population2(MatingPool2),{1,20,1,20});
                Population1 = EADMMIBEAConstrainedUpdate(Population1,Offspring1,Problem.N,kappa);
                Population2 = EADMMIBEASelection([Population2,Offspring2],Problem.N,kappa);

                % Module 3, Steps 1 and 2: exchange offspring populations.
                Population1 = EADMMIBEAConstrainedUpdate(Population1,Offspring2,Problem.N,kappa);
                Population2 = EADMMIBEASelection([Population2,Offspring1],Problem.N,kappa);

                % Module 3, Steps 3 to 5.
                Archive = EADMMTemporaryArchive(Offspring1,Population1,Population2);
                if ~isempty(Archive)
                    Checked = EADMMLocalSearch(Problem,Archive,Population1);
                    if ~isempty(Checked)
                        Population1 = EADMMIBEAConstrainedUpdate(Population1,Checked,Problem.N,kappa);
                        Population2 = EADMMIBEASelection([Population2,Checked],Problem.N,kappa);
                    end
                end
            end
        end
    end
end
