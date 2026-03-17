function [Population,Fitness,SelectedIdx] = EnvironmentalSelectionC(Population,N)
% Feasible-first environmental selection for the constrained population.

    if isempty(Population)
        Fitness = [];
        SelectedIdx = [];
        return;
    end

    FeasibleIdx = find(all(Population.cons<=0,2));
    if ~isempty(FeasibleIdx)
        KeepLocal = SelectObjectiveIdx(Population(FeasibleIdx).objs,min(N,numel(FeasibleIdx)));
        SelectedIdx = FeasibleIdx(KeepLocal)';
    else
        % Cold-start guard: if no feasible solutions exist yet, keep the best
        % objective-driven individuals so PopulationC remains evolvable.
        SelectedIdx = SelectObjectiveIdx(Population.objs,min(N,numel(Population)));
    end

    if numel(SelectedIdx) < N
        Repeat = SelectedIdx(mod(0:N-numel(SelectedIdx)-1,numel(SelectedIdx))+1);
        SelectedIdx = [SelectedIdx,Repeat];
    end

    SelectedIdx = SelectedIdx(:)';
    Population = Population(SelectedIdx);
    Fitness    = CalFitness(Population.objs,Population.cons);
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
