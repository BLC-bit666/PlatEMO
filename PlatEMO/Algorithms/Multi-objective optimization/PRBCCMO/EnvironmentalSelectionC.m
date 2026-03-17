function [Population,Fitness,SelectedIdx] = EnvironmentalSelectionC(Population,N,Model,W)
% Environmental selection for the constrained population.

    if isempty(Population)
        Fitness = [];
        SelectedIdx = [];
        return;
    end

    FeasibleMask = all(Population.cons<=0,2);
    FeasibleIdx  = find(FeasibleMask);
    InfeasibleIdx = find(~FeasibleMask);
    SelectedFeasibleIdx = [];
    SelectedIdx         = [];

    if ~isempty(FeasibleIdx)
        LocalIdx   = SelectObjectiveIdx(Population(FeasibleIdx).objs,min(N,length(FeasibleIdx)));
        SelectedFeasibleIdx = FeasibleIdx(LocalIdx)';
        SelectedIdx         = SelectedFeasibleIdx;
    end

    Rest = N - length(SelectedIdx);
    if Rest > 0 && ~isempty(InfeasibleIdx)
        AddIdx = SelectInfeasibleIdx(Population,SelectedFeasibleIdx,InfeasibleIdx,Rest,Model,W,true);
        SelectedIdx = [SelectedIdx,AddIdx];
    end

    if length(SelectedIdx) < N && ~isempty(InfeasibleIdx)
        ReserveIdx = setdiff(InfeasibleIdx,SelectedIdx,'stable');
        AddIdx = SelectInfeasibleIdx(Population,SelectedFeasibleIdx,ReserveIdx,N-length(SelectedIdx),Model,W,false);
        SelectedIdx = [SelectedIdx,AddIdx];
    end

    SelectedIdx = SelectedIdx(:)';
    Population = Population(SelectedIdx);
    Fitness    = CalFitness(Population.objs,Population.cons);
end

function Idx = SelectObjectiveIdx(PopObj,N)
    if isempty(PopObj)
        Idx = [];
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

function AddIdx = SelectInfeasibleIdx(Population,SelectedFeasibleIdx,CandidateIdx,N,Model,W,UseSectorFilter)
    AddIdx = [];
    if N <= 0 || isempty(CandidateIdx)
        return;
    end

    Keep = true(1,length(CandidateIdx));
    if UseSectorFilter && ~isempty(SelectedFeasibleIdx)
        FeasibleObj = Population(SelectedFeasibleIdx).objs;
        CandObj     = Population(CandidateIdx).objs;
        RefObj      = [FeasibleObj;CandObj];
        [SectorF,CountF] = AssociateSectors(FeasibleObj,W,RefObj);
        SectorI          = AssociateSectors(CandObj,W,RefObj);
        for i = 1 : length(CandidateIdx)
            if CountF(SectorI(i)) <= 1
                continue;
            end
            SameSectorObj = FeasibleObj(SectorF==SectorI(i),:);
            if AnyDominates(SameSectorObj,CandObj(i,:))
                Keep(i) = false;
            end
        end
        if any(~Keep)
            CandidateIdx = CandidateIdx(Keep);
        end
    end

    if isempty(CandidateIdx)
        return;
    end

    CandObj = Population(CandidateIdx).objs;
    Prob    = PredictBoundaryMLP(Model,Population(CandidateIdx).decs);
    Score   = 1 - 2*abs(Prob-0.5);
    FrontNo = NDSort(CandObj,size(CandObj,1));
    Crowd   = CrowdingDistance(CandObj,FrontNo);

    if isempty(SelectedFeasibleIdx)
        SectorI = AssociateSectors(CandObj,W,CandObj);
        CountF  = accumarray(SectorI,1,[size(W,1),1]);
    else
        RefObj = [Population(SelectedFeasibleIdx).objs;CandObj];
        [~,CountF] = AssociateSectors(Population(SelectedFeasibleIdx).objs,W,RefObj);
        SectorI = AssociateSectors(CandObj,W,RefObj);
    end

    Sparsity = CountF(SectorI);
    RankMat  = [-Score(:),Sparsity(:),-Crowd(:),FrontNo(:)];
    [~,Rank] = sortrows(RankMat,[1 2 3 4]);
    AddIdx   = CandidateIdx(Rank(1:min(N,length(Rank))))';
end
