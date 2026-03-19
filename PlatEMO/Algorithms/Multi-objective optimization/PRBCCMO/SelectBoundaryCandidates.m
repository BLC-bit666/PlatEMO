function [Offspring,Info] = SelectBoundaryCandidates(Problem,Pool,PopulationC,Model,W,HardNegativeArchive,Budget,RuntimeOptions)
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

    if nargin < 8 || ~isstruct(RuntimeOptions)
        RuntimeOptions = struct();
    end
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
    SelectionMode = ResolveSelectionMode(RuntimeOptions);
    RankScore = ResolveSelectionScore(Detail,SelectionMode);
    if SelectionMode == 3
        Accept = randperm(size(Pool.decs,1),min(Budget,size(Pool.decs,1)));
    else
        [~,Order] = sort(RankScore(:),'descend');
        Accept = Order(1:min(Budget,length(Order)));
    end
    if isempty(Accept)
        return;
    end

    DecsSel      = Pool.decs(Accept,:);
    ProxySel     = Pool.proxyObjs(Accept,:);
    SourceSel    = Pool.source(Accept);
    ProbSel      = Detail.prob(Accept);

    Offspring = Problem.Evaluation(DecsSel);
    Info.source    = SourceSel(:);
    Info.score     = RankScore(Accept);
    Info.prob      = ProbSel(:);
    Info.entropy   = Detail.entropy(Accept);
    Info.hvGain    = Detail.hvGain(Accept);
    Info.novelty   = Detail.sectorNovelty(Accept);
    Info.penalty   = Detail.penaltyFactor(Accept);
    Info.utility   = RankScore(Accept);
    Info.sector    = Detail.sector(Accept);
    Info.proxyObjs = ProxySel;
end

function SelectionMode = ResolveSelectionMode(RuntimeOptions)
    SelectionMode = 1;
    if isstruct(RuntimeOptions) && isfield(RuntimeOptions,'SelectionMode') && ~isempty(RuntimeOptions.SelectionMode)
        SelectionMode = max(1,min(3,round(RuntimeOptions.SelectionMode)));
    end
end

function Score = ResolveSelectionScore(Detail,SelectionMode)
    switch SelectionMode
        case 2
            Score = Detail.uncertaintyUtility(:);
        otherwise
            Score = Detail.utility(:);
    end
end
