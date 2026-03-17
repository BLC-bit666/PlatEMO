function [Offspring,Info] = SelectBoundaryCandidates(Problem,Pool,PopulationC,Model,W,HardNegativeArchive,Budget)
% Select and evaluate boundary candidates with boundary utility scoring.

    Offspring = [];
    Info.source    = zeros(0,1);
    Info.score     = zeros(0,1);
    Info.prob      = zeros(0,1);
    Info.entropy   = zeros(0,1);
    Info.hvGain    = zeros(0,1);
    Info.novelty   = zeros(0,1);
    Info.penalty   = zeros(0,1);
    Info.utility   = zeros(0,1);
    Info.sector    = zeros(0,1);
    Info.proxyObjs = zeros(0,Problem.M);

    if Budget <= 0 || isempty(Pool.decs)
        return;
    end

    FeasibleC = PopulationC(all(PopulationC.cons<=0,2));
    if isempty(FeasibleC)
        FeasibleObj = zeros(0,Problem.M);
    else
        FeasibleObj = FeasibleC.objs;
    end

    Detail = ScoreBoundaryCandidates( ...
        Problem,Pool.decs,Pool.proxyObjs,FeasibleObj,Model,W,HardNegativeArchive);
    [~,Order] = sort(Detail.utility,'descend');
    Accept = Order(1:min(Budget,length(Order)));
    if isempty(Accept)
        return;
    end

    DecsSel      = Pool.decs(Accept,:);
    ProxySel     = Pool.proxyObjs(Accept,:);
    SourceSel    = Pool.source(Accept);
    ProbSel      = Detail.prob(Accept);

    Offspring = Problem.Evaluation(DecsSel);
    Info.source    = SourceSel(:);
    Info.score     = Detail.utility(Accept);
    Info.prob      = ProbSel(:);
    Info.entropy   = Detail.entropy(Accept);
    Info.hvGain    = Detail.hvGain(Accept);
    Info.novelty   = Detail.sectorNovelty(Accept);
    Info.penalty   = Detail.penaltyFactor(Accept);
    Info.utility   = Detail.utility(Accept);
    Info.sector    = Detail.sector(Accept);
    Info.proxyObjs = ProxySel;
end
