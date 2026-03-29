function [Population,Fitness,SelectedIdx] = EnvironmentalSelectionC(Population,N,MigrationPool,W)
% Reserve slots for improved feasible migrants, then apply feasible-first NSGA-II.

    if nargin < 3 || isempty(MigrationPool)
        MigrationPool = [];
    end
    if nargin < 4
        W = [];
    end
    if isempty(Population) && isempty(MigrationPool)
        Fitness = [];
        SelectedIdx = [];
        return;
    end

    BasePool = KeepUniquePopulation(Population);
    MigrationPool = KeepUniquePopulation(MigrationPool);
    Reserved = SelectReservedMigrants(BasePool,MigrationPool,W);
    if numel(Reserved) > N
        Reserved = Reserved(1:N);
    end

    CandidatePool = KeepUniquePopulation([BasePool,MigrationPool]);
    CandidatePool = RemovePopulationByDecision(CandidatePool,Reserved);

    Selected = Reserved;
    Need = max(0,N-numel(Selected));
    if Need > 0
        FeasiblePool = CandidatePool(all(CandidatePool.cons<=0,2));
        [Chosen,~] = SelectObjectivePopulation(FeasiblePool,min(Need,numel(FeasiblePool)));
        Selected = [Selected,Chosen];
    end

    Need = max(0,N-numel(Selected));
    if Need > 0
        Remaining = RemovePopulationByDecision(CandidatePool,Selected);
        [Chosen,~] = SelectObjectivePopulation(Remaining,min(Need,numel(Remaining)));
        Selected = [Selected,Chosen];
    end

    if isempty(Selected)
        Fitness = [];
        SelectedIdx = [];
        Population = [];
        return;
    end
    if numel(Selected) < N
        Repeat = Selected(mod(0:N-numel(Selected)-1,numel(Selected))+1);
        Selected = [Selected,Repeat];
    end

    Population = Selected(:)';
    Fitness    = CalFitness(Population.objs,Population.cons);
    SelectedIdx = 1:numel(Population);
end

function Reserved = SelectReservedMigrants(BasePool,MigrationPool,W)
    Reserved = [];
    if isempty(MigrationPool)
        return;
    end

    FeasibleBase = BasePool(all(BasePool.cons<=0,2));
    FeasibleMig  = MigrationPool(all(MigrationPool.cons<=0,2));
    if isempty(FeasibleMig)
        return;
    end

    if isempty(W)
        W = ones(1,size(FeasibleMig.objs,2));
    end
    RefObj = FeasibleMig.objs;
    if ~isempty(FeasibleBase)
        RefObj = [FeasibleBase.objs;FeasibleMig.objs];
    end

    SectorMig = AssociateSectors(FeasibleMig.objs,W,RefObj);
    if isempty(FeasibleBase)
        SectorBase = zeros(0,1);
    else
        SectorBase = AssociateSectors(FeasibleBase.objs,W,RefObj);
    end

    Keep = false(1,numel(FeasibleMig));
    Improve = -inf(1,numel(FeasibleMig));
    for s = unique(SectorMig(:))'
        MigIdx = find(SectorMig==s);
        MigValue = ComputeSectorScalar(FeasibleMig(MigIdx).objs,W,RefObj,repmat(s,numel(MigIdx),1));
        [BestMigValue,BestLocal] = min(MigValue);
        BestMigIdx = MigIdx(BestLocal);

        BaseIdx = find(SectorBase==s);
        if isempty(BaseIdx)
            Keep(BestMigIdx) = true;
            Improve(BestMigIdx) = inf;
            continue;
        end

        BaseValue = ComputeSectorScalar(FeasibleBase(BaseIdx).objs,W,RefObj,repmat(s,numel(BaseIdx),1));
        ChampionValue = min(BaseValue);
        if BestMigValue < ChampionValue
            Keep(BestMigIdx) = true;
            Improve(BestMigIdx) = ChampionValue - BestMigValue;
        end
    end

    Reserved = FeasibleMig(Keep);
    if isempty(Reserved)
        return;
    end
    Improve = Improve(Keep);
    [~,Order] = sort(Improve,'descend');
    Reserved = Reserved(Order);
end

function Population = KeepUniquePopulation(Population)
    if isempty(Population)
        return;
    end
    Keep = KeepLatestDecisionRows(Population.decs);
    Population = Population(Keep);
end

function Population = RemovePopulationByDecision(Population,Remove)
    if isempty(Population) || isempty(Remove)
        return;
    end
    Keep = ~ismember(Population.decs,Remove.decs,'rows');
    Population = Population(Keep);
end

function [Population,Idx] = SelectObjectivePopulation(Population,N)
    Idx = zeros(1,0);
    if isempty(Population) || N <= 0
        Population = [];
        return;
    end
    LocalIdx = SelectObjectiveIdx(Population.objs,min(N,numel(Population)));
    Population = Population(LocalIdx);
    Idx = LocalIdx(:)';
end

function Idx = SelectObjectiveIdx(PopObj,N)
    if isempty(PopObj)
        Idx = zeros(1,0);
        return;
    end
    [FrontNo,MaxFNo] = NDSort(PopObj,N);
    Next             = FrontNo < MaxFNo;
    CrowdDis         = CrowdingDistance(PopObj,FrontNo);
    Last             = find(FrontNo==MaxFNo);
    Need             = N - sum(Next);
    if Need > 0
        [~,Rank] = sort(CrowdDis(Last),'descend');
        Next(Last(Rank(1:Need))) = true;
    end
    Idx = find(Next);
    [~,Order] = sortrows([FrontNo(Idx)',-CrowdDis(Idx)'],[1 2]);
    Idx = Idx(Order)';
end
