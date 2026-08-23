classdef EADMMMOEAD < ALGORITHM
% <2024> <multi/many> <real> <constrained>
% EADMM with MOEA/D as the backbone algorithm

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
            %% Generate shared weight vectors and neighbourhoods
            requestedN = Problem.N;
            [W,actualN] = UniformPoint(requestedN,Problem.M,'MUD');
            if actualN ~= requestedN
                error('EADMMMOEAD:PopulationSizeMismatch', ...
                    'The weight generator must preserve the requested population size.');
            end
            W         = max(W,1e-6);
            W         = W./sum(W,2);
            Problem.N = actualN;
            EADMMValidateSetup(Problem);
            T = max(2,ceil(Problem.N/10));
            [~,B] = sort(pdist2(W,W),2);
            B = B(:,1:min(T,Problem.N));
            if size(B,2) < 2
                error('EADMMMOEAD:PopulationTooSmall', ...
                    'EADMM/MOEA-D requires at least two weight vectors.');
            end

            %% Initialize the constrained and unconstrained populations
            Population1 = Problem.Initialization();
            Population2 = Problem.Initialization();
            Z2          = min(Population2.objs,[],1);

            %% Optimization
            while Algorithm.NotTerminated(Population1)
                nSubproblem = Problem.N;
                sourceIndex = 1 : Problem.N;
                Offspring1 = cell(1,nSubproblem);
                Offspring2 = cell(1,nSubproblem);

                % Module 1: constrained MOEA/D using the four paper rules.
                for a = 1 : nSubproblem
                    k       = sourceIndex(a);
                    parents = B(k,randperm(size(B,2)));
                    child   = OperatorGAhalf(Problem,Population1(parents(1:2)),{1,20,1,20});
                    Offspring1{a} = child;
                    Z1          = min([Population1.objs;child.obj],[],1);
                    Population1 = EADMMMOEADConstrainedUpdate(Population1,child,B(k,:),W,Z1);
                end
                Offspring1 = [Offspring1{:}];

                % Module 2: vanilla unconstrained MOEA/D.
                for a = 1 : nSubproblem
                    k       = sourceIndex(a);
                    parents = B(k,randperm(size(B,2)));
                    child   = OperatorGAhalf(Problem,Population2(parents(1:2)),{1,20,1,20});
                    Offspring2{a} = child;
                    Z2          = min(Z2,child.obj);
                    Population2 = EADMMMOEADUnconstrainedUpdate(Population2,child,B(k,:),W,Z2);
                end
                Offspring2 = [Offspring2{:}];

                % Module 3, Steps 1 and 2: cross-population updates retain
                % the source subproblem and its neighbourhood.
                for a = 1 : nSubproblem
                    k           = sourceIndex(a);
                    Z1          = min([Population1.objs;Offspring2(a).obj],[],1);
                    Population1 = EADMMMOEADConstrainedUpdate(Population1,Offspring2(a),B(k,:),W,Z1);
                    Z2          = min(Z2,Offspring1(a).obj);
                    Population2 = EADMMMOEADUnconstrainedUpdate(Population2,Offspring1(a),B(k,:),W,Z2);
                end

                % Module 3, Steps 3 and 4: preserve source subproblem IDs.
                [Archive,archivePosition] = EADMMTemporaryArchive(Offspring1,Population1,Population2);
                if ~isempty(Archive)
                    archiveSource = sourceIndex(archivePosition);
                    Checked       = EADMMLocalSearch(Problem,Archive,Population1);

                    % Module 3, Step 5: a local solution inherits the
                    % subproblem of the main-task offspring that spawned it.
                    for a = 1 : numel(Checked)
                        k           = archiveSource(a);
                        Z1          = min([Population1.objs;Checked(a).obj],[],1);
                        Population1 = EADMMMOEADConstrainedUpdate(Population1,Checked(a),B(k,:),W,Z1);
                        Z2          = min(Z2,Checked(a).obj);
                        Population2 = EADMMMOEADUnconstrainedUpdate(Population2,Checked(a),B(k,:),W,Z2);
                    end
                end
            end
        end
    end
end
