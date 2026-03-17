function [Offspring,Info] = SelectBoundaryCandidates(Problem,Pool,PopulationC,Model,W,Budget)
% Select and evaluate a fixed-budget set of boundary candidates.

    Offspring = [];
    Info.source    = zeros(0,1);
    Info.score     = zeros(0,1);
    Info.prob      = zeros(0,1);
    Info.proxyObjs = zeros(0,Problem.M);

    if Budget <= 0 || isempty(Pool.decs)
        return;
    end

    Prob  = PredictBoundaryMLP(Model,Pool.decs);
    Score = 1 - 2*abs(Prob-0.5);
    [~,Order] = sort(Score,'descend');

    Accept = GreedyAccept(Order,Pool.proxyObjs,PopulationC,W,Budget);
    if isempty(Accept)
        return;
    end

    DecsSel      = Pool.decs(Accept,:);
    ProxySel     = Pool.proxyObjs(Accept,:);
    SourceSel    = Pool.source(Accept);
    ScoreSel     = Score(Accept);
    ProbSel      = Prob(Accept);

    Offspring = Problem.Evaluation(DecsSel);
    Info.source    = SourceSel(:);
    Info.score     = ScoreSel(:);
    Info.prob      = ProbSel(:);
    Info.proxyObjs = ProxySel;
end

function Accept = GreedyAccept(Order,ProxyObjs,PopulationC,W,Budget)
    Accept = zeros(1,min(Budget,length(Order)));
    Count  = 0;
    if Budget <= 0 || isempty(Order)
        Accept = Accept([]);
        return;
    end

    FeasibleC = PopulationC(all(PopulationC.cons<=0,2));
    if isempty(FeasibleC)
        Accept = Order(1:min(Budget,length(Order)));
        return;
    end

    OrderedProxy = ProxyObjs(Order,:);
    RefObj       = [FeasibleC.objs;OrderedProxy];
    [SectorF,CountF] = AssociateSectors(FeasibleC.objs,W,RefObj);
    SectorC          = AssociateSectors(OrderedProxy,W,RefObj);

    for i = 1 : length(Order)
        if CountF(SectorC(i)) > 1
            SameSectorObj = FeasibleC(SectorF==SectorC(i)).objs;
            if AnyDominates(SameSectorObj,OrderedProxy(i,:))
                continue;
            end
        end
        Count = Count + 1;
        Accept(Count) = Order(i);
        if Count >= Budget
            break;
        end
    end
    Accept = Accept(1:Count);
end

function flag = AnyDominates(PopObj,obj)
    flag = false;
    if isempty(PopObj)
        return;
    end
    LessEqual = all(PopObj<=repmat(obj,size(PopObj,1),1),2);
    Less      = any(PopObj<repmat(obj,size(PopObj,1),1),2);
    flag      = any(LessEqual & Less);
end
