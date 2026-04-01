classdef PRBCCMO1 < ALGORITHM
% <2026> <multi> <real> <constrained>
% PRBCCMO1
% Dual-population CCMO with an explicit sectorized boundary archive
%
% type      --- 1    --- 1.GA, 2.DE
% hidden    --- 20   --- Hidden neurons of the boundary MLP
% epoch     --- 20   --- Training epochs per update
% lr        --- 0.01 --- Learning rate
% minClass  --- 20   --- Minimum samples of each class before first training
% earlyGap  --- 5    --- Model update gap in early stage
% lateGap   --- 15   --- Model update gap in late stage
% switchRho --- 0.5  --- Early/late switch FE ratio
%
% Core fixes relative to the previous PRBCCMO1:
%   1) B is redefined as a sector-local boundary corridor between Pop_C and Pop_U;
%   2) exact-between points are preferred, nearest-outside points are used only
%      when exact-between points are insufficient, so that B can be filled;
%   3) B x opposite uses geometric side (lambda), not feasible/infeasible label;
%   4) sector coverage is also used in Pop_C / Pop_U truncation to improve IGD.

    methods
        function main(Algorithm,Problem)
            [W,Problem.N] = UniformPoint(Problem.N,Problem.M);
            [type,hidden,epoch,lr,minClass,earlyGap,lateGap,switchRho] = ...
                Algorithm.ParameterSet(1,20,20,0.01,20,5,15,0.5);

            MaxB         = 2*Problem.N;
            MaxTrain     = 6*Problem.N;
            Generation   = 0;

            PopulationC  = Problem.Initialization();
            PopulationU  = Problem.Initialization();
            B            = PopulationC([]);
            Model        = [];
            TrainArchive = InitTrainArchive(Problem.D);

            % Initial boundary seeding
            SeedB        = GenerateCrossBoundaryOffspring(Problem,PopulationC,PopulationU,W,type,Problem.N);
            B            = UpdateBoundaryArchive([B,SeedB],PopulationC,PopulationU,W,Model,MaxB);
            TrainArchive = UpdateTrainArchive(TrainArchive,B,PopulationC,PopulationU,W,MaxTrain);
            if CanTrainBoundaryModel(TrainArchive,minClass)
                Model = TrainBoundaryMLP(TrainArchive.Dec,TrainArchive.Label,hidden,epoch,lr,[]);
            end

            while Algorithm.NotTerminated(PopulationC)
                Generation = Generation + 1;

                % 1) Inner evolution of two main populations
                OffspringC = GenerateWithinOffspring(Problem,PopulationC,type,true);
                OffspringU = GenerateWithinOffspring(Problem,PopulationU,type,false);

                PoolC = KeepUniquePopulation([PopulationC,OffspringC]);
                PoolU = KeepUniquePopulation([PopulationU,OffspringU]);

                % 2) Boundary generation
                CrossB     = GenerateCrossBoundaryOffspring(Problem,PoolC,PoolU,W,type,Problem.N);
                SearchB    = GenerateArchiveBoundaryOffspring(Problem,B,PoolC,PoolU,W,Model,type,Problem.N);
                BoundaryOff = KeepUniquePopulation([CrossB,SearchB]);

                % 3) Boundary -> Pop_C migration (feasible only, sector improvement only)
                Migrants = ExtractBoundaryMigrants(BoundaryOff,PoolC,W);

                % 4) Update explicit boundary archive
                B = UpdateBoundaryArchive([B,BoundaryOff],PoolC,PoolU,W,Model,MaxB);

                % 5) Environmental selection of two main populations
                PopulationC = EnvironmentalSelectionC([PopulationC,OffspringC,Migrants],Problem.N,W);
                PopulationU = EnvironmentalSelectionU([PopulationU,OffspringU],Problem.N,W);

                % 6) Warm-start MLP update
                TrainArchive = UpdateTrainArchive(TrainArchive,B,PopulationC,PopulationU,W,MaxTrain);
                if CanTrainBoundaryModel(TrainArchive,minClass)
                    if Problem.FE/max(Problem.maxFE,1) < switchRho
                        Gap = earlyGap;
                    else
                        Gap = lateGap;
                    end
                    if isempty(Model) || mod(Generation,Gap) == 0
                        Model = TrainBoundaryMLP(TrainArchive.Dec,TrainArchive.Label,hidden,epoch,lr,Model);
                    end
                end
            end
        end
    end
end

%% ========== Offspring generation ==========

function Offspring = GenerateWithinOffspring(Problem,Population,type,isConstraintSide)
    if isempty(Population)
        Offspring = [];
        return;
    end
    N = numel(Population);
    if isConstraintSide
        [Flag,FrontNo,CrowdDis] = ConstraintSideIndicator(Population);
        MatingPool = TournamentSelection(2,2*N,Flag,FrontNo,-CrowdDis);
    else
        [FrontNo,CrowdDis] = ObjectiveSideIndicator(Population);
        MatingPool = TournamentSelection(2,2*N,FrontNo,-CrowdDis);
    end

    if type == 1
        Offspring = OperatorGAhalf(Problem,Population(MatingPool));
    else
        Offspring = OperatorDE(Problem,Population,...
            Population(MatingPool(1:N)),Population(MatingPool(N+1:end)));
    end
end

function Offspring = GenerateCrossBoundaryOffspring(Problem,PopulationC,PopulationU,W,type,NOff)
    if isempty(PopulationC) || isempty(PopulationU) || NOff <= 0
        Offspring = PopulationC([]);
        return;
    end
    RefObj = BuildReferenceObj(PopulationC,PopulationU,[]);
    RepC   = BuildSectorRepresentatives(PopulationC,W,RefObj,true);
    RepU   = BuildSectorRepresentatives(PopulationU,W,RefObj,false);
    Shared = find(RepC.has & RepU.has);

    if isempty(Shared)
        Offspring = PopulationC([]);
        return;
    end

    Shared = Shared(randperm(numel(Shared)));
    Use    = Shared(mod(0:NOff-1,numel(Shared))+1);

    ParentC  = PopulationC(RepC.idx(Use));
    ParentU  = PopulationU(RepU.idx(Use));
    Offspring = CrossTwoSets(Problem,ParentC,ParentU,type);
end

function Offspring = GenerateArchiveBoundaryOffspring(Problem,B,PopulationC,PopulationU,W,Model,type,NOff)
    if isempty(B) || isempty(PopulationC) || isempty(PopulationU) || NOff <= 0
        Offspring = B([]);
        return;
    end

    [Meta,RepC,RepU] = BuildBoundaryMeta(B,PopulationC,PopulationU,W,Model);
    Ranked = BuildBoundaryRankLists(Meta,size(W,1));
    Pick   = SectorRoundRobinPick(Ranked,NOff);
    if isempty(Pick)
        Offspring = B([]);
        return;
    end

    ParentB  = B(Pick);
    Opposite = PopulationC([]);
    for i = 1 : numel(Pick)
        idx = Pick(i);
        s   = Meta.sector(idx);
        % Geometric opposite:
        % lambda = 0 at U-side, lambda = 1 at C-side
        if Meta.lambda(idx) <= 0.5
            Opposite(end+1) = PopulationC(RepC.idx(s)); %#ok<AGROW>
        else
            Opposite(end+1) = PopulationU(RepU.idx(s)); %#ok<AGROW>
        end
    end

    Offspring = CrossTwoSets(Problem,ParentB,Opposite,type);
end

function Offspring = CrossTwoSets(Problem,Parent1,Parent2,type)
    if isempty(Parent1) || isempty(Parent2)
        Offspring = Parent1([]);
        return;
    end
    K = min(numel(Parent1),numel(Parent2));
    Parent1 = Parent1(1:K);
    Parent2 = Parent2(1:K);

    if type == 1
        Offspring = OperatorGAhalf(Problem,[Parent1,Parent2]);
    else
        % midpoint pull in decision space, but still only as a candidate generator
        Offspring = OperatorDE(Problem,Parent1,Parent2,Parent1);
    end
end

%% ========== Boundary archive ==========

function B = UpdateBoundaryArchive(CandidateB,PopulationC,PopulationU,W,Model,MaxB)
    if isempty(CandidateB)
        B = CandidateB;
        return;
    end

    CandidateB = KeepUniquePopulation(CandidateB);
    [Meta,~,~] = BuildBoundaryMeta(CandidateB,PopulationC,PopulationU,W,Model);

    Ranked = BuildBoundaryRankLists(Meta,size(W,1));
    Pick   = SectorRoundRobinPick(Ranked,MaxB);

    if isempty(Pick)
        B = CandidateB([]);
    else
        B = CandidateB(unique(Pick,'stable'));
    end
end

function Ranked = BuildBoundaryRankLists(Meta,K)
    Ranked = cell(K,1);
    if isempty(Meta.sector)
        return;
    end
    for s = 1 : K
        idx = find(Meta.hasPair & Meta.sector == s);
        if isempty(idx)
            continue;
        end
        % Priority:
        %   1) exact-between first
        %   2) nearest-to-corridor next
        %   3) |p-0.5| small
        %   4) close to middle of the corridor
        Key = [double(~Meta.inside(idx)), Meta.bandDist(idx), Meta.absp05(idx), Meta.midDist(idx)];
        [~,ord] = sortrows(Key,[1 2 3 4]);
        Ranked{s} = idx(ord);
    end
end

function Pick = SectorRoundRobinPick(Ranked,MaxPick)
    Pick = zeros(0,1);
    if isempty(Ranked) || MaxPick <= 0
        return;
    end
    Ptr = ones(numel(Ranked),1);
    while numel(Pick) < MaxPick
        Changed = false;
        for s = 1 : numel(Ranked)
            if Ptr(s) <= numel(Ranked{s})
                Pick(end+1,1) = Ranked{s}(Ptr(s)); %#ok<AGROW>
                Ptr(s) = Ptr(s) + 1;
                Changed = true;
                if numel(Pick) >= MaxPick
                    break;
                end
            end
        end
        if ~Changed
            break;
        end
    end
end

function [Meta,RepC,RepU] = BuildBoundaryMeta(Candidates,PopulationC,PopulationU,W,Model)
    Meta = struct();
    Meta.sector   = zeros(0,1);
    Meta.scalar   = zeros(0,1);
    Meta.prob     = zeros(0,1);
    Meta.absp05   = zeros(0,1);
    Meta.hasPair  = false(0,1);
    Meta.lambda   = zeros(0,1);
    Meta.inside   = false(0,1);
    Meta.bandDist = zeros(0,1);
    Meta.midDist  = zeros(0,1);

    if isempty(Candidates)
        RepC = BuildSectorRepresentatives(PopulationC,W,[],true);
        RepU = BuildSectorRepresentatives(PopulationU,W,[],false);
        return;
    end

    % IMPORTANT FIX:
    % RefObj only uses current main populations, not CandidateB itself.
    RefObj = BuildReferenceObj(PopulationC,PopulationU,[]);
    if isempty(RefObj)
        RefObj = Candidates.objs;
    end

    RepC = BuildSectorRepresentatives(PopulationC,W,RefObj,true);
    RepU = BuildSectorRepresentatives(PopulationU,W,RefObj,false);

    Sector = AssociateSectorsLocal(Candidates.objs,W,RefObj);
    Scalar = ComputeSectorScalar(Candidates.objs,W,RefObj,Sector);
    [Prob,~] = PredictBoundaryMLP(Model,Candidates.decs);

    N = numel(Candidates);
    HasPair  = false(N,1);
    Lambda   = nan(N,1);
    Inside   = false(N,1);
    BandDist = inf(N,1);
    MidDist  = inf(N,1);

    for i = 1 : N
        s = Sector(i);
        if s <= 0 || s > numel(RepC.has) || ~RepC.has(s) || ~RepU.has(s)
            continue;
        end
        HasPair(i) = true;
        gC = RepC.scalar(s);
        gU = RepU.scalar(s);
        if abs(gC-gU) < 1e-12
            Lambda(i) = 0.5;
        else
            % lambda = 0 at U, lambda = 1 at C
            Lambda(i) = (Scalar(i)-gU)/(gC-gU);
        end
        Inside(i)   = Lambda(i) >= 0 && Lambda(i) <= 1;
        BandDist(i) = max([0,-Lambda(i),Lambda(i)-1]);
        MidDist(i)  = abs(Lambda(i)-0.5);
    end

    Meta.sector   = Sector(:);
    Meta.scalar   = Scalar(:);
    Meta.prob     = Prob(:);
    Meta.absp05   = abs(Prob(:)-0.5);
    Meta.hasPair  = HasPair(:);
    Meta.lambda   = Lambda(:);
    Meta.inside   = Inside(:);
    Meta.bandDist = BandDist(:);
    Meta.midDist  = MidDist(:);
end

function Migrants = ExtractBoundaryMigrants(BoundaryOff,PopulationC,W)
    if isempty(BoundaryOff)
        Migrants = BoundaryOff;
        return;
    end

    BoundaryOff = FilterFeasiblePopulation(BoundaryOff);
    if isempty(BoundaryOff)
        Migrants = BoundaryOff;
        return;
    end

    RefObj    = BuildReferenceObj(PopulationC,[],BoundaryOff);
    SectorOff = AssociateSectorsLocal(BoundaryOff.objs,W,RefObj);
    ScalarOff = ComputeSectorScalar(BoundaryOff.objs,W,RefObj,SectorOff);
    RepC      = BuildSectorRepresentatives(PopulationC,W,RefObj,true);

    Pick = zeros(0,1);
    for s = unique(SectorOff(:))'
        idx = find(SectorOff == s);
        [bestScalar,loc] = min(ScalarOff(idx));
        bestIdx = idx(loc);
        if ~RepC.has(s) || bestScalar < RepC.scalar(s)
            Pick(end+1,1) = bestIdx; %#ok<AGROW>
        end
    end

    if isempty(Pick)
        Migrants = BoundaryOff([]);
    else
        Migrants = BoundaryOff(unique(Pick,'stable'));
    end
end

%% ========== Environmental selection ==========

function Population = EnvironmentalSelectionC(Population,N,W)
    Population = KeepUniquePopulation(Population);
    if isempty(Population)
        return;
    end

    Feasible = FilterFeasiblePopulation(Population);
    if numel(Feasible) >= N
        Population = SectorSelectByObjective(Feasible,N,W);
    else
        Next   = Feasible;
        Need   = N - numel(Next);
        Remain = RemovePopulationByDecision(Population,Next);
        if Need > 0 && ~isempty(Remain)
            Next = [Next,SectorSelectByObjective(Remain,min(Need,numel(Remain)),W)];
        end
        Population = PadPopulation(Next,N);
    end
end

function Population = EnvironmentalSelectionU(Population,N,W)
    Population = KeepUniquePopulation(Population);
    Population = SectorSelectByObjective(Population,min(N,numel(Population)),W);
    Population = PadPopulation(Population,N);
end

function Population = SectorSelectByObjective(Population,N,W)
    if isempty(Population)
        return;
    end
    if N <= 0
        Population = Population([]);
        return;
    end

    N = min(N,numel(Population));
    RefObj   = Population.objs;
    Sector   = AssociateSectorsLocal(Population.objs,W,RefObj);
    Scalar   = ComputeSectorScalar(Population.objs,W,RefObj,Sector);
    [FrontNo,~] = NDSort(Population.objs,numel(Population));
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);

    Ranked = cell(size(W,1),1);
    for s = 1 : size(W,1)
        idx = find(Sector == s);
        if isempty(idx)
            continue;
        end
        Key = [FrontNo(idx)', Scalar(idx), -CrowdDis(idx)'];
        [~,ord] = sortrows(Key,[1 2 3]);
        Ranked{s} = idx(ord);
    end

    Pick = SectorRoundRobinPick(Ranked,N);
    Population = Population(Pick);
end

function Population = PadPopulation(Population,N)
    if isempty(Population)
        return;
    end
    if numel(Population) < N
        Population = [Population,Population(mod(0:N-numel(Population)-1,numel(Population))+1)];
    else
        Population = Population(1:N);
    end
end

%% ========== Parent-selection indicators ==========

function [Flag,FrontNo,CrowdDis] = ConstraintSideIndicator(Population)
    [FrontNo,CrowdDis] = ObjectiveSideIndicator(Population);
    Flag = double(~all(Population.cons<=0,2)); % feasible:0, infeasible:1
end

function [FrontNo,CrowdDis] = ObjectiveSideIndicator(Population)
    [FrontNo,~] = NDSort(Population.objs,numel(Population));
    CrowdDis    = CrowdingDistance(Population.objs,FrontNo);
end

%% ========== Training archive ==========

function TrainArchive = InitTrainArchive(D)
    TrainArchive = struct( ...
        'Dec',zeros(0,D), ...
        'Label',zeros(0,1), ...
        'Source',zeros(0,1));   % 1: B, 2: Pop_C reps, 3: Pop_U reps
end

function TrainArchive = UpdateTrainArchive(TrainArchive,B,PopulationC,PopulationU,W,MaxTrain)
    RefObj = BuildReferenceObj(PopulationC,PopulationU,B);
    if isempty(RefObj)
        if ~isempty(B)
            RefObj = B.objs;
        else
            RefObj = [];
        end
    end

    RepC = BuildSectorRepresentatives(PopulationC,W,RefObj,true);
    RepU = BuildSectorRepresentatives(PopulationU,W,RefObj,false);

    TrainArchive = AppendTrainArchive(TrainArchive,B,1);
    if any(RepC.has)
        TrainArchive = AppendTrainArchive(TrainArchive,PopulationC(RepC.idx(RepC.has)),2);
    end
    if any(RepU.has)
        TrainArchive = AppendTrainArchive(TrainArchive,PopulationU(RepU.idx(RepU.has)),3);
    end
    TrainArchive = TrimTrainArchive(TrainArchive,MaxTrain);
end

function TrainArchive = AppendTrainArchive(TrainArchive,Population,Source)
    if isempty(Population)
        return;
    end
    TrainArchive.Dec    = [TrainArchive.Dec;Population.decs];
    TrainArchive.Label  = [TrainArchive.Label;double(all(Population.cons<=0,2))];
    TrainArchive.Source = [TrainArchive.Source;Source*ones(numel(Population),1)];

    Keep = KeepLatestDecisionRowsLocal(TrainArchive.Dec);
    TrainArchive.Dec    = TrainArchive.Dec(Keep,:);
    TrainArchive.Label  = TrainArchive.Label(Keep);
    TrainArchive.Source = TrainArchive.Source(Keep);
end

function TrainArchive = TrimTrainArchive(TrainArchive,MaxTrain)
    Count = size(TrainArchive.Dec,1);
    if Count <= MaxTrain
        return;
    end

    BQuota   = min(round(0.7*MaxTrain),sum(TrainArchive.Source==1));
    KeepB    = SelectLatestBalancedIndices(find(TrainArchive.Source==1),TrainArchive.Label,BQuota);
    RemQuota = MaxTrain - numel(KeepB);
    KeepO    = SelectLatestBalancedIndices(find(TrainArchive.Source~=1),TrainArchive.Label,RemQuota);

    Keep = unique([KeepB;KeepO],'stable');
    if numel(Keep) < MaxTrain
        Rest = setdiff((1:Count)',Keep,'stable');
        Need = min(MaxTrain-numel(Keep),numel(Rest));
        if Need > 0
            Keep = [Keep;Rest(end-Need+1:end)];
        end
    elseif numel(Keep) > MaxTrain
        Keep = Keep(end-MaxTrain+1:end);
    end

    Keep = sort(Keep);
    TrainArchive.Dec    = TrainArchive.Dec(Keep,:);
    TrainArchive.Label  = TrainArchive.Label(Keep);
    TrainArchive.Source = TrainArchive.Source(Keep);
end

function Idx = SelectLatestBalancedIndices(CandidateIdx,Label,Count)
    Idx = zeros(0,1);
    if isempty(CandidateIdx) || Count <= 0
        return;
    end

    CandidateIdx = CandidateIdx(:);
    Pos = CandidateIdx(Label(CandidateIdx)==1);
    Neg = CandidateIdx(Label(CandidateIdx)==0);

    Quota = floor(Count/2);
    if Quota > 0
        KeepPos = Pos(max(1,numel(Pos)-Quota+1):end);
        KeepNeg = Neg(max(1,numel(Neg)-Quota+1):end);
        Idx = unique([KeepPos;KeepNeg],'stable');
    end

    if numel(Idx) < Count
        Rest = setdiff(CandidateIdx,Idx,'stable');
        Need = min(Count-numel(Idx),numel(Rest));
        if Need > 0
            Idx = [Idx;Rest(end-Need+1:end)];
        end
    elseif numel(Idx) > Count
        Idx = Idx(end-Count+1:end);
    end

    Idx = sort(Idx);
end

function Flag = CanTrainBoundaryModel(TrainArchive,minClass)
    Flag = sum(TrainArchive.Label==1) >= minClass && ...
           sum(TrainArchive.Label==0) >= minClass;
end

%% ========== Boundary MLP ==========

function Model = TrainBoundaryMLP(X,Y,Hidden,Epoch,LR,PrevModel)
    if nargin < 6
        PrevModel = [];
    end
    Model = PrevModel;

    if isempty(X) || size(X,1) < 4
        return;
    end

    X = double(X);
    Y = double(Y(:) > 0);
    if numel(unique(Y)) < 2
        return;
    end

    Hidden    = max(2,round(Hidden));
    Epoch     = max(1,round(Epoch));
    LR        = max(double(LR),1e-4);
    [N,D]     = size(X);
    LambdaReg = 1e-4;

    Mu    = mean(X,1);
    Sigma = std(X,0,1);
    Sigma(Sigma<1e-12) = 1;

    if ~isempty(PrevModel) && IsWarmStartCompatible(PrevModel,D,Hidden)
        W1 = PrevModel.W1;  b1 = PrevModel.b1;
        W2 = PrevModel.W2;  b2 = PrevModel.b2;
    else
        W1 = 0.1*randn(D,Hidden);
        b1 = zeros(1,Hidden);
        W2 = 0.1*randn(Hidden,1);
        b2 = 0;
    end

    Xn = (X-Mu)./Sigma;
    [Weight,NormWeight] = BuildClassWeights(Y);

    for e = 1 : Epoch
        H = tanh(Xn*W1 + repmat(b1,N,1));
        Z = H*W2 + b2;
        P = 1./(1+exp(-Z));

        Delta2 = Weight.*(P-Y)./NormWeight;
        dW2 = H'*Delta2 + LambdaReg*W2;
        db2 = sum(Delta2);

        D1  = (Delta2*W2').*(1-H.^2);
        dW1 = Xn'*D1 + LambdaReg*W1;
        db1 = sum(D1,1);

        Step = LR/sqrt(e);
        W1 = W1 - Step*dW1;
        b1 = b1 - Step*db1;
        W2 = W2 - Step*dW2;
        b2 = b2 - Step*db2;
    end

    Model = struct();
    Model.Mu    = Mu;
    Model.Sigma = Sigma;
    Model.W1    = W1;
    Model.b1    = b1;
    Model.W2    = W2;
    Model.b2    = b2;
end

function [Prob,Stats] = PredictBoundaryMLP(Model,X)
    if nargin < 2 || isempty(X)
        Prob  = zeros(0,1);
        Stats = struct('logit',zeros(0,1));
        return;
    end
    if isempty(Model) || ~isfield(Model,'Mu')
        Prob  = 0.5*ones(size(X,1),1);
        Stats = struct('logit',zeros(size(X,1),1));
        return;
    end

    Xn = (double(X)-Model.Mu)./Model.Sigma;
    H  = tanh(Xn*Model.W1 + repmat(Model.b1,size(Xn,1),1));
    Z  = H*Model.W2 + Model.b2;
    Prob = 1./(1+exp(-Z));
    Prob = min(max(Prob,1e-6),1-1e-6);
    Stats = struct('logit',Z(:));
end

function Flag = IsWarmStartCompatible(Model,D,Hidden)
    Flag = ~isempty(Model) && isfield(Model,'W1') && isfield(Model,'W2') && ...
           size(Model.W1,1) == D && size(Model.W1,2) == Hidden && ...
           size(Model.W2,1) == Hidden;
end

function [Weight,NormWeight] = BuildClassWeights(Y)
    N    = numel(Y);
    Pos  = sum(Y==1);
    Neg  = N - Pos;
    WPos = N/(2*max(1,Pos));
    WNeg = N/(2*max(1,Neg));
    Weight     = WNeg + (WPos-WNeg).*Y;
    NormWeight = max(sum(Weight),1);
end

%% ========== Sector tools ==========

function Rep = BuildSectorRepresentatives(Population,W,RefObj,PreferFeasible)
    K   = size(W,1);
    Rep = struct('has',false(K,1), ...
                 'idx',zeros(K,1), ...
                 'scalar',inf(K,1));
    if isempty(Population)
        return;
    end

    Sector = AssociateSectorsLocal(Population.objs,W,RefObj);
    Scalar = ComputeSectorScalar(Population.objs,W,RefObj,Sector);
    Fea    = all(Population.cons<=0,2);

    for s = 1 : K
        idx = find(Sector == s);
        if isempty(idx)
            continue;
        end
        if PreferFeasible
            idxF = idx(Fea(idx));
            if ~isempty(idxF)
                idx = idxF;
            end
        end
        [Rep.scalar(s),loc] = min(Scalar(idx));
        Rep.idx(s) = idx(loc);
        Rep.has(s) = true;
    end
end

function RefObj = BuildReferenceObj(PopulationC,PopulationU,ExtraPop)
    RefObj = [];
    if nargin >= 1 && ~isempty(PopulationC)
        RefObj = [RefObj;PopulationC.objs];
    end
    if nargin >= 2 && ~isempty(PopulationU)
        RefObj = [RefObj;PopulationU.objs];
    end
    if nargin >= 3 && ~isempty(ExtraPop)
        RefObj = [RefObj;ExtraPop.objs];
    end
end

function [Sector,Count] = AssociateSectorsLocal(PopObj,W,RefObj)
    if nargin < 3 || isempty(RefObj)
        RefObj = PopObj;
    end
    if isempty(PopObj)
        Sector = zeros(0,1);
        Count  = zeros(size(W,1),1);
        return;
    end
    if isempty(W)
        Sector = ones(size(PopObj,1),1);
        Count  = size(PopObj,1);
        return;
    end

    MinObj = min(RefObj,[],1);
    MaxObj = max(RefObj,[],1);
    Range  = MaxObj - MinObj;
    Range(Range<1e-12) = 1;

    Obj = (PopObj - MinObj)./Range;
    ObjNorm = sqrt(sum(Obj.^2,2));
    ZeroMask = ObjNorm < 1e-12;
    Obj(ZeroMask,:) = 1;
    ObjNorm(ZeroMask) = sqrt(size(Obj,2));
    Obj = Obj./ObjNorm(:,ones(1,size(Obj,2)));

    WNorm = sqrt(sum(W.^2,2));
    WNorm(WNorm<1e-12) = 1;
    Wn = W./WNorm(:,ones(1,size(W,2)));

    Cosine = Obj*Wn';
    [~,Sector] = max(Cosine,[],2);
    Count = accumarray(Sector,1,[size(W,1),1]);
end

function Value = ComputeSectorScalar(Obj,W,RefObj,Sector)
    if isempty(Obj)
        Value = zeros(0,1);
        return;
    end
    if nargin < 2 || isempty(W)
        W = ones(1,size(Obj,2));
    end
    if nargin < 3 || isempty(RefObj)
        RefObj = Obj;
    end
    if nargin < 4 || isempty(Sector)
        Weight = repmat(W(1,:),size(Obj,1),1);
    else
        Weight = W(Sector,:);
    end

    MinObj = min(RefObj,[],1);
    MaxObj = max(RefObj,[],1);
    Range  = MaxObj - MinObj;
    Range(Range<1e-12) = 1;
    NormObj = (Obj-MinObj)./Range;

    Value = sum(NormObj.*Weight,2);
end

%% ========== Population utilities ==========

function Population = FilterFeasiblePopulation(Population)
    if isempty(Population)
        return;
    end
    Population = Population(all(Population.cons<=0,2));
end

function Population = KeepUniquePopulation(Population)
    if isempty(Population)
        return;
    end
    Keep = KeepLatestDecisionRowsLocal(Population.decs);
    Population = Population(Keep);
end

function Population = RemovePopulationByDecision(Population,Remove)
    if isempty(Population) || isempty(Remove)
        return;
    end
    Keep = ~ismember(Population.decs,Remove.decs,'rows');
    Population = Population(Keep);
end

function Keep = KeepLatestDecisionRowsLocal(Dec)
    if isempty(Dec)
        Keep = zeros(0,1);
        return;
    end
    [~,Keep] = unique(double(Dec),'rows','last');
    Keep = sort(Keep);
end