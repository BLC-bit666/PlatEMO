classdef GANCMO_V3 < ALGORITHM
% <2026> <multi> <real> <constrained>
% Dual-population DE baseline; no CGAN module is initialized or called.
% References: Tian et al., IEEE TEVC 25(1):102-116, 2021;
% Gulrajani et al., Improved Training of Wasserstein GANs, NeurIPS 2017.
% Copyright (c) 2026 BIMK Group. PlatEMO research-use and citation terms apply.
    methods
        function Algorithm = GANCMO_V3(varargin)
            Algorithm@ALGORITHM(varargin{:});
        end

        function main(Algorithm,Problem)
            utility = fullfile(fileparts(which('ALGORITHM')),'Utility functions');
            if ~startsWith(string(which('TournamentSelection')),string(utility)+filesep)
                addpath(utility,'-begin');
            end
            state = rng; data = state.State;
            if iscell(data); data = data{1}; end
            seed = mod(sum(double(data(1:min(16,numel(data))))),2^32-1);
            P1 = Problem.Initialization(min(Problem.N,Problem.maxFE-Problem.FE));
            P2 = P1([]);
            if Problem.FE<Problem.maxFE
                P2 = Problem.Initialization(min(Problem.N,Problem.maxFE-Problem.FE));
            end
            F1 = GANCMO_V3Fitness(P1.objs,P1.cons); F2 = [];
            if ~isempty(P2); F2 = GANCMO_V3Fitness(P2.objs); end
            generation = 0; batch = Problem.N;
            while Algorithm.NotTerminated(P1)
                generation = generation+1; beforeFE = Problem.FE;
                budget = min(batch,max(0,Problem.maxFE-beforeFE));
                n1 = min(Problem.N,ceil(budget/2)); n2 = min(Problem.N,floor(budget/2));
                [O1,O2] = GANCMO_V3Offspring(Problem,P1,P2,F1,F2,n1,n2,seed,generation);
                assert(Problem.FE-beforeFE==budget && Problem.FE<=Problem.maxFE, ...
                    'GANCMO_V3:EvaluationBudget','Incorrect true evaluation count.');
                [P1,F1] = GANCMO_V3Selection([P1,O1,O2],Problem.N,true);
                [P2,F2] = GANCMO_V3Selection([P2,O1,O2],Problem.N,false);
            end
        end
    end
end
