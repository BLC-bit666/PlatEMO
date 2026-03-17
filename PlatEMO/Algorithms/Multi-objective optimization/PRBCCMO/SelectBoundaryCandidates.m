function [Offspring,Info] = SelectBoundaryCandidates(Problem,Pool,PopulationC,Model,W,Budget)
% Select and evaluate boundary candidates with uncertainty pre-screening and reranking.

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
    PreCount = min(length(Order),3*Budget);
    Shortlist = Order(1:PreCount);

    FeasibleC = PopulationC(all(PopulationC.cons<=0,2));
    if isempty(FeasibleC)
        FeasibleObj = zeros(0,Problem.M);
    else
        FeasibleObj = FeasibleC.objs;
    end
    AcceptLocal = RerankBoundaryCandidates( ...
        Pool.proxyObjs(Shortlist,:),Score(Shortlist),Pool.source(Shortlist), ...
        FeasibleObj,W,Budget);
    Accept = Shortlist(AcceptLocal);
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
