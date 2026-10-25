classdef GANCMO_V4 < ALGORITHM
% <2026> <multi> <real> <constrained>
% Feasible-real-sample-only CGAN; archive, detector and batch budget retained.
% References: Tian et al., IEEE TEVC 25(1):102-116, 2021;
% Gulrajani et al., Improved Training of Wasserstein GANs, NeurIPS 2017.
% Copyright (c) 2026 BIMK Group. PlatEMO research-use and citation terms apply.
    properties(SetAccess=private)
        Settings = GANCMO_V4Settings();
    end
    methods
        function Algorithm = GANCMO_V4(varargin)
            Algorithm@ALGORITHM(varargin{:});
        end

        function main(Algorithm,Problem)
            utility = fullfile(fileparts(which('ALGORITHM')),'Utility functions');
            if ~startsWith(string(which('TournamentSelection')),string(utility)+filesep)
                addpath(utility,'-begin');
            end
            S = Algorithm.Settings;
            [W,~] = UniformPoint(max(2,round(Problem.N)),Problem.M);
            state = rng; data = state.State;
            if iscell(data); data = data{1}; end
            seed = mod(sum(double(data(1:min(16,numel(data))))),2^32-1);
            trainStream = RandStream('mt19937ar','Seed',mod(seed+104729,2^32-1));
            queryStream = RandStream('mt19937ar','Seed',mod(seed+130363,2^32-1));
            P1 = Problem.Initialization(min(Problem.N,Problem.maxFE-Problem.FE));
            P2 = P1([]);
            if Problem.FE<Problem.maxFE
                P2 = Problem.Initialization(min(Problem.N,Problem.maxFE-Problem.FE));
            end
            initialFE = Problem.FE;
            F1 = GANCMO_V4Fitness(P1.objs,P1.cons); F2 = [];
            if ~isempty(P2); F2 = GANCMO_V4Fitness(P2.objs); end
            Scale = struct('minimum',min([P1.objs;P2.objs],[],1), ...
                'span',ones(1,Problem.M));
            [Archive,Scale] = GANCMO_V4Archive.update([], ...
                [P1,P2],W,Problem,Scale);
            Model = []; generation = 0; batch = Problem.N;
            Detection = []; useCGAN = false;
            while Algorithm.NotTerminated(P1)
                generation = generation+1; beforeFE = Problem.FE;
                budget = min(batch,max(0,Problem.maxFE-beforeFE));
                n1 = min(Problem.N,ceil(budget/2)); n2 = min(Problem.N,floor(budget/2));
                [~,quota] = GANCMO_V4Control(beforeFE,initialFE,Problem.maxFE,n1);
                Data = GANCMO_V4Archive.training(Archive,W,Scale,Problem);
                q = zeros(0,Problem.D);
                due = isempty(Model);
                if ~due
                    [Detection,due] = GANCMO_V4Detection(Detection,Data,Model.lastData,size(W,1));
                end
                if due
                    useCGAN = false; beforeG = 0;
                    if ~isempty(Model); beforeG = Model.iterG; end
                    if quota>0
                        scope = streamScope(trainStream); %#ok<NASGU>
                        Model = GANCMO_V4Model.train(Model,Data,Problem.D,S);
                        clear scope;
                        useCGAN = ~isempty(Model) && Model.iterG>beforeG;
                    end
                end
                if quota>0 && ~isempty(Model)
                    scope = streamScope(queryStream); %#ok<NASGU>
                    q = GANCMO_V4Candidates(Model,W,Problem,quota,useCGAN);
                    clear scope;
                end
                [O1,O2,Generated] = GANCMO_V4Offspring(Problem,P1,P2,F1,F2,n1,n2,q,seed,generation);
                assert(Problem.FE-beforeFE==budget && Problem.FE<=Problem.maxFE, ...
                    'GANCMO_V4:EvaluationBudget','Incorrect true evaluation count.');
                % Batch minimum equals updating the historical ideal point
                % after every evaluated child, including infeasible/rejected ones.
                Scale.minimum = min([Scale.minimum;O1.objs;O2.objs],[],1);
                Union = [P1,P2,O1,O2];
                % O1 contains DE first and direct CGAN offspring last.
                DE1 = O1(1:numel(O1)-numel(Generated));
                [P1,F1] = GANCMO_V4Selection([P1,O1,O2],Problem.N,true);
                [P2,F2] = GANCMO_V4Selection([P2,DE1,O2],Problem.N,false);
                [Archive,Scale] = GANCMO_V4Archive.update(Archive,Union,W,Problem,Scale);
            end
        end
    end
end

function cleanup = streamScope(stream)
    previous = RandStream.getGlobalStream;
    RandStream.setGlobalStream(stream);
    cleanup = onCleanup(@()RandStream.setGlobalStream(previous));
end
