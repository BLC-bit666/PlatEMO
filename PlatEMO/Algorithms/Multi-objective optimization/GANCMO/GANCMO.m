classdef GANCMO < ALGORITHM
% <2026> <multi> <real> <constrained>
% Direct feasible-condition CGAN offspring with progress-based DE mixing.
% References: Tian et al., IEEE TEVC 25(1):102-116, 2021;
% Gulrajani et al., Improved Training of Wasserstein GANs, NeurIPS 2017.
% Copyright (c) 2026 BIMK Group. PlatEMO research-use and citation terms apply.
    properties
        Settings = struct();
        RecordTrace = false;
        Trace = struct();
        FinalModel = [];
        FinalArchive = [];
    end
    methods
        function Algorithm = GANCMO(varargin)
            Algorithm@ALGORITHM(varargin{:});
        end

        function main(Algorithm,Problem)
            utility = fullfile(fileparts(which('ALGORITHM')),'Utility functions');
            if ~startsWith(string(which('TournamentSelection')),string(utility)+filesep)
                addpath(utility,'-begin');
            end
            S = GANCMOSettings(Algorithm.Settings); Algorithm.Settings = S;
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
            F1 = GANCMOFitness(P1.objs,P1.cons); F2 = [];
            if ~isempty(P2); F2 = GANCMOFitness(P2.objs); end
            FixedScale = [];
            if strcmp(S.referenceScaling,'objective')
                FixedScale = struct('minimum',zeros(1,Problem.M),'span',ones(1,Problem.M));
            end
            [Archive,Scale] = GANCMOArchive.update([], ...
                [P1,P2],P1([]),W,Problem,S.pairsPerDirection,S.activePairPriority,FixedScale);
            Model = []; generation = 0; batch = 2*Problem.N;
            if Algorithm.RecordTrace
                Algorithm.Trace = struct('columns',{{'generation','FE','progress', ...
                    'plannedG','actualG','feasibleG','survivedP1','survivedP2', ...
                    'trainingPairs','updatesG','updatesC','trainingSeconds', ...
                    'candidateSeconds','archiveSeconds','reset','trainingFailure', ...
                    'uniqueF','uniqueI'}},'rows',zeros(0,18),'generated',{{}}, ...
                    'initialFE',initialFE,'W',W,'coverage',{{}}, ...
                    'coverageFixedScale',struct('minimum',zeros(1,Problem.M),'span',ones(1,Problem.M)));
                coverageSeen = [];
            end
            while Algorithm.NotTerminated(P1)
                generation = generation+1; beforeFE = Problem.FE;
                budget = min(batch,max(0,Problem.maxFE-beforeFE));
                n1 = min(Problem.N,ceil(budget/2)); n2 = min(Problem.N,floor(budget/2));
                [progress,quota] = GANCMOControl(beforeFE,initialFE,Problem.maxFE,n1);
                if Algorithm.RecordTrace
                    beforeP1Y = double(P1.objs); beforeP2Y = double(P2.objs);
                    beforeArchiveY = [Archive.yf;Archive.yi]; beforeScale = Scale;
                    beforeArchiveFeasible = [true(size(Archive.yf,1),1);false(size(Archive.yi,1),1)];
                end
                Data = GANCMOArchive.training(Archive,W,Scale,Problem,S.maxTrainingObjectiveGap);
                q = zeros(0,Problem.D);
                Query = struct('ref',zeros(0,1),'targetRef',zeros(0,1),'distanceF',zeros(0,1), ...
                    'available',0,'diverse',0,'queried',0);
                Train = struct('updatesG',0,'updatesC',0,'reset',false,'failed',false);
                trainTime = 0; candidateTime = 0;
                if quota>0
                    timer = tic;
                    scope = streamScope(trainStream); %#ok<NASGU>
                    [Model,Train] = GANCMOModel.train(Model,Data,Problem.D,generation,progress,S);
                    clear scope;
                    trainTime = toc(timer);
                    if ~isempty(Model) && Model.ready && Data.count>0
                        timer = tic;
                        scope = streamScope(queryStream); %#ok<NASGU>
                        [q,Query] = GANCMOCandidates(Model,Archive,W,Scale,P1,P2,Problem,quota,S);
                        clear scope;
                        candidateTime = toc(timer);
                    end
                end
                [O1,O2,Generated] = GANCMOOffspring(Problem,P1,P2,F1,F2,n1,n2,q,seed,generation);
                assert(Problem.FE-beforeFE==budget && Problem.FE<=Problem.maxFE, ...
                    'GANCMO:EvaluationBudget','Incorrect true evaluation count.');
                Union = [P1,P2,O1,O2];
                % O1 contains DE first and direct CGAN offspring last.
                DE1 = O1(1:numel(O1)-numel(Generated));
                [P1,F1] = GANCMOSelection([P1,O1,O2],Problem.N,true);
                [P2,F2] = GANCMOSelection([P2,DE1,O2],Problem.N,false);
                timer = tic;
                [Archive,Scale] = GANCMOArchive.update(Archive,Union,Generated,W,Problem, ...
                    S.pairsPerDirection,S.activePairPriority,FixedScale);
                archiveTime = toc(timer);
                if Algorithm.RecordTrace
                    Algorithm.Trace.finalScale = Scale;
                    Algorithm.Trace.finalEligibilityPool = struct('Y',[double(Union.objs);beforeArchiveY], ...
                        'feasible',[all(double(Union.cons)<=0,2);beforeArchiveFeasible]);
                    GX = zeros(0,Problem.D); GY = zeros(0,Problem.M); GC = zeros(0,1);
                    survived1 = 0; survived2 = 0;
                    kept1 = false(0,1); kept2 = false(0,1);
                    if ~isempty(Generated)
                        GX = double(Generated.decs); GY = double(Generated.objs); GC = double(Generated.cons);
                        % Track evaluated individuals, not equal decision vectors.
                        kept1 = arrayfun(@(g)any(P1==g),Generated(:));
                        kept2 = arrayfun(@(g)any(P2==g),Generated(:));
                        survived1 = nnz(kept1); survived2 = nnz(kept2);
                    end
                    [coverage,coverageSeen] = GANCMOCoverage(W,beforeScale,beforeP1Y,beforeP2Y, ...
                        beforeArchiveY,GY,[double(O1.objs);double(O2.objs)],kept1,kept2,coverageSeen);
                    Algorithm.Trace.coverage{generation,1} = coverage;
                    Algorithm.Trace.rows(generation,:) = [generation,Problem.FE,progress, ...
                        quota,numel(Generated),nnz(all(GC<=0,2)),survived1,survived2, ...
                        Data.count,Train.updatesG,Train.updatesC,trainTime,candidateTime, ...
                        archiveTime,Train.reset,Train.failed,size(unique(Data.xF,'rows'),1), ...
                        size(unique(Data.xI,'rows'),1)];
                    Algorithm.Trace.generated{generation,1} = struct('X',single(GX), ...
                        'Y',GY,'C',GC,'requestedRef',Query.ref,'targetRef',Query.targetRef,'distanceF',Query.distanceF, ...
                        'queriedCandidates',Query.queried,'availableCandidates',Query.available, ...
                        'diverseCandidates',Query.diverse);
                    Algorithm.FinalModel = Model;
                    Algorithm.FinalArchive = Archive;
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
