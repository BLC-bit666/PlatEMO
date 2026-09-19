classdef PairGuide < ALGORITHM
% <2026> <multi> <real> <constrained>
% Pair-guided cooperative evolution with coverage-weighted CGAN resources.
% The sole mainline uses boundary-local guidance and coverage alpha = 0.04.
% References: Tian et al., A Coevolutionary Framework for Constrained
% Multiobjective Optimization Problems, IEEE TEVC 25(1):102-116, 2021;
% Gulrajani et al., Improved Training of Wasserstein GANs, NeurIPS 2017.
% Copyright (c) 2026 BIMK Group. PlatEMO research-use and citation terms apply.
    methods
        function Algorithm = PairGuide(varargin)
            Algorithm@ALGORITHM(varargin{:});
        end

        function main(Algorithm,Problem)
            % Platform operators must precede identically named local helpers.
            utility = fullfile(fileparts(which('ALGORITHM')),'Utility functions');
            if ~startsWith(string(which('TournamentSelection')),string(utility)+filesep)
                addpath(utility,'-begin');
            end
            [W,~] = UniformPoint(max(2,round(Problem.N)),Problem.M);
            state = rng; data = state.State;
            if iscell(data); data = data{1}; end
            seed = mod(sum(double(data(1:min(16,numel(data))))),2^32-1);
            trainStream = RandStream('mt19937ar','Seed',mod(seed+104729,2^32-1));
            queryStream = RandStream('mt19937ar','Seed',mod(seed+130363,2^32-1));
            P1 = Problem.Initialization(min(Problem.N,Problem.maxFE-Problem.FE));
            P2 = P1([]);
            if Problem.FE < Problem.maxFE
                P2 = Problem.Initialization(min(Problem.N,Problem.maxFE-Problem.FE));
            end
            F1 = PairGuideFitness(P1.objs,P1.cons); F2 = [];
            if ~isempty(P2); F2 = PairGuideFitness(P2.objs); end
            [Archive,~] = PairGuideArchive.update([],[P1,P2],P1([]),W,Problem);
            Pending = zeros(0,Problem.D); Model = []; z = 20; stopped = false;
            generation = 0; batch = 2*Problem.N;
            while Algorithm.NotTerminated(P1)
                generation = generation+1;
                budget = min(batch,max(0,Problem.maxFE-Problem.FE));
                n1 = min(Problem.N,ceil(budget/2)); n2 = min(Problem.N,floor(budget/2));
                [O1,O2,Guided] = PairGuideOffspring(Problem,P1,P2,F1,F2,n1,n2, ...
                    Pending,Archive,seed,generation);
                Pending = zeros(0,Problem.D);
                Union = [P1,P2,O1,O2];
                [P1,F1] = PairGuideSelection([P1,O1,O2],Problem.N,true);
                [P2,F2] = PairGuideSelection([P2,O1,O2],Problem.N,false);
                % Once stopped, neither archive nor model can affect evolution.
                if stopped; continue; end
                [Archive,Scale] = PairGuideArchive.update(Archive,Union,Guided,W,Problem);
                [z,quota,cutoff] = PairGuideControl(P1,P2,W,Scale,z,Problem.maxFE,batch);
                stopped = Problem.FE+batch > cutoff;
                if stopped; continue; end
                remaining = min(batch,max(0,Problem.maxFE-Problem.FE));
                nextCount = min(Problem.N,ceil(remaining/2));
                quota = min(quota,nextCount-round(.25*nextCount));
                if quota == 0 || Problem.FE+remaining > cutoff; continue; end
                Data = PairGuideArchive.training(Archive,W,Scale,Problem);
                scope = streamScope(trainStream); %#ok<NASGU>
                Model = PairGuideModel.train(Model,Data,Problem.D,generation);
                clear scope;
                if ~isempty(Model) && Model.ready && Data.count > 0
                    scope = streamScope(queryStream); %#ok<NASGU>
                    Pending = PairGuideCandidates(Model,Archive,W,Scale,P1,P2,Problem,quota);
                    clear scope;
                end
            end
        end
    end
end

function cleanup = streamScope(stream)
    previous = RandStream.getGlobalStream;
    RandStream.setGlobalStream(stream);
    cleanup = onCleanup(@()RandStream.setGlobalStream(previous));
end
