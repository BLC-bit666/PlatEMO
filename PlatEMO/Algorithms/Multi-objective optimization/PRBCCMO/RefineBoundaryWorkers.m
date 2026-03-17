function [Offspring,Info,MigrationPool,BracketPairs,HardNegativeBatch] = RefineBoundaryWorkers( ...
    Problem,BoundarySeeds,BoundaryInfo,PopulationC,PopulationU,ArchiveF,ArchiveI,Model,W,HardNegativeArchive,Budget)
% Execute feasible expansion, infeasible bracketing, and hard-negative confirmation.

    Offspring = [];
    MigrationPool = [];
    BracketPairs  = EmptyBracketPairs();
    HardNegativeBatch.Dec    = zeros(0,Problem.D);
    HardNegativeBatch.Radius = zeros(0,1);
    Info = EmptyBoundaryInfo(Problem.M);
    if Budget <= 0 || isempty(BoundarySeeds)
        return;
    end

    Remaining = Budget;
    OffspringCell = cell(1,numel(BoundarySeeds));
    MigrationCell = cell(1,numel(BoundarySeeds));
    InfoCell      = cell(1,numel(BoundarySeeds));
    Filled        = 0;
    FeasibleRef = PopulationC(all(PopulationC.cons<=0,2));
    if isempty(FeasibleRef)
        FeasibleObj = zeros(0,Problem.M);
    else
        FeasibleObj = FeasibleRef.objs;
    end
    InfeasibleRef = [PopulationU(~all(PopulationU.cons<=0,2)),ArchiveI];
    FeasibleAnchorPool = [FeasibleRef,ArchiveF];

    for i = 1 : numel(BoundarySeeds)
        if Remaining <= 0
            break;
        end
        Seed = BoundarySeeds(i);
        Source = BoundaryInfo.source(i);
        if all(Seed.con<=0)
            [Desc,MoveAdd] = HandleFeasibleSeed( ...
                Problem,Seed,InfeasibleRef,Remaining,W);
            NewBracket = EmptyBracketPairs();
            NewHardNeg.Dec    = zeros(0,Problem.D);
            NewHardNeg.Radius = zeros(0,1);
        else
            [Desc,MoveAdd,NewBracket,NewHardNeg] = HandleInfeasibleSeed( ...
                Problem,Seed,FeasibleAnchorPool,Remaining,W);
        end

        Remaining = Remaining - numel(Desc);
        if isempty(Desc)
            BracketPairs = MergeBracketPairs(BracketPairs,NewBracket);
            HardNegativeBatch = MergeHardNegBatch(HardNegativeBatch,NewHardNeg);
            continue;
        end

        BracketPairs  = MergeBracketPairs(BracketPairs,NewBracket);
        HardNegativeBatch = MergeHardNegBatch(HardNegativeBatch,NewHardNeg);
        Filled = Filled + 1;
        OffspringCell{Filled} = Desc;
        MigrationCell{Filled} = MoveAdd;
        InfoCell{Filled} = AppendWorkerInfo( ...
            Problem,EmptyBoundaryInfo(Problem.M),Desc,Source,FeasibleObj,Model,W,HardNegativeArchive);
    end

    Offspring     = MergePopulationCells(OffspringCell(1:Filled));
    MigrationPool = MergePopulationCells(MigrationCell(1:Filled));
    Info          = MergeBoundaryInfoCells(InfoCell(1:Filled),Problem.M);
end

function [Desc,MigrationPool] = HandleFeasibleSeed(Problem,Seed,InfeasiblePool,Remaining,W)
    Desc = [];
    MigrationPool = [];
    if Remaining <= 0
        return;
    end

    Anchor = SelectAnchorBySector(Seed,InfeasiblePool,W);
    QueryCount = min(2,Remaining);
    Decs = zeros(QueryCount,Problem.D);
    for j = 1 : QueryCount
        StepScale = 0.10 + 0.10*(j-1);
        Decs(j,:) = GenerateExpansionCandidate(Problem,Seed,Anchor,StepScale);
    end
    Desc = Problem.Evaluation(Decs);

    FeasibleDesc = Desc(all(Desc.cons<=0,2));
    if ~isempty(FeasibleDesc)
        MigrationPool = FeasibleDesc;
    end
end

function [Desc,MigrationPool,BracketPairs,HardNegativeBatch] = HandleInfeasibleSeed(Problem,Seed,FeasiblePool,Remaining,W)
    Desc = [];
    MigrationPool = [];
    BracketPairs  = EmptyBracketPairs();
    HardNegativeBatch.Dec    = zeros(0,Problem.D);
    HardNegativeBatch.Radius = zeros(0,1);
    if Remaining <= 0
        return;
    end

    Anchor = SelectAnchorBySector(Seed,FeasiblePool,W);
    if isempty(Anchor)
        [Desc,MigrationPool,HardNegativeBatch] = ConfirmHardNegative(Problem,Seed,Anchor,Remaining);
        return;
    end

    FeasibleEnd = Anchor;
    InfeasibleEnd = Seed;
    FoundFeasible = false;
    MaxBracketStep = min(4,Remaining);
    MidCell = cell(1,MaxBracketStep);
    MidCount = 0;
    for k = 1 : MaxBracketStep
        MidDec = GenerateBracketMidpoint(Problem,FeasibleEnd.dec,InfeasibleEnd.dec,0.5);
        MidSol = Problem.Evaluation(MidDec);
        MidCount = MidCount + 1;
        MidCell{MidCount} = MidSol;
        if all(MidSol.con<=0)
            FeasibleEnd  = MidSol;
            FoundFeasible = true;
        else
            InfeasibleEnd = MidSol;
        end
    end
    Desc = MergePopulationCells(MidCell(1:MidCount));

    Remaining = Remaining - numel(Desc);
    if FoundFeasible
        Gap = ComputeDecisionGap(Problem,FeasibleEnd.dec,InfeasibleEnd.dec);
        BracketPairs.Feasible   = FeasibleEnd;
        BracketPairs.Infeasible = InfeasibleEnd;
        BracketPairs.Gap        = Gap;
        MigrationPool = FeasibleEnd;
        if Remaining > 0
            ExpDec = GenerateExpansionCandidate(Problem,FeasibleEnd,InfeasibleEnd,0.12);
            ExpSol = Problem.Evaluation(ExpDec);
            Desc   = [Desc,ExpSol];
            if all(ExpSol.con<=0)
                MigrationPool = [MigrationPool,ExpSol];
            end
        end
        return;
    end

    [HardDesc,MigrationPool,HardNegativeBatch] = ConfirmHardNegative(Problem,Seed,Anchor,Remaining);
    Desc = [Desc,HardDesc];
end

function [Desc,MigrationPool,HardNegativeBatch] = ConfirmHardNegative(Problem,Seed,Anchor,Remaining)
    Desc = [];
    MigrationPool = [];
    HardNegativeBatch.Dec    = zeros(0,Problem.D);
    HardNegativeBatch.Radius = zeros(0,1);
    if Remaining <= 0
        return;
    end

    if isempty(Anchor)
        BaseRadius = 0.10;
    else
        BaseRadius = max(0.03,0.5*ComputeDecisionGap(Problem,Seed.dec,Anchor.dec));
    end
    QueryCount = min(2,Remaining);
    Decs = zeros(QueryCount,Problem.D);
    Radius = zeros(QueryCount,1);
    for j = 1 : QueryCount
        Radius(j) = BaseRadius*(0.5^(j-1));
        Decs(j,:) = GenerateHardNegativeSample(Problem,Seed,Anchor,Radius(j));
    end
    Desc = Problem.Evaluation(Decs);
    FeasibleDesc = Desc(all(Desc.cons<=0,2));
    if ~isempty(FeasibleDesc)
        MigrationPool = FeasibleDesc;
        return;
    end

    HardNegativeBatch.Dec    = Seed.dec;
    HardNegativeBatch.Radius = BaseRadius;
end

function Info = AppendWorkerInfo(Problem,Info,Desc,Source,FeasibleObj,Model,W,HardNegativeArchive)
    Detail = ScoreBoundaryCandidates(Problem,Desc.decs,Desc.objs,FeasibleObj,Model,W,HardNegativeArchive);
    Count = numel(Desc);
    Info.source    = [Info.source;repmat(Source,Count,1)];
    Info.score     = [Info.score;Detail.utility(:)];
    Info.prob      = [Info.prob;Detail.prob(:)];
    Info.entropy   = [Info.entropy;Detail.entropy(:)];
    Info.hvGain    = [Info.hvGain;Detail.hvGain(:)];
    Info.novelty   = [Info.novelty;Detail.sectorNovelty(:)];
    Info.penalty   = [Info.penalty;Detail.penaltyFactor(:)];
    Info.utility   = [Info.utility;Detail.utility(:)];
    Info.sector    = [Info.sector;Detail.sector(:)];
    Info.proxyObjs = [Info.proxyObjs;Desc.objs];
end

function Anchor = SelectAnchorBySector(Seed,Pool,W)
    Anchor = [];
    if isempty(Pool)
        return;
    end
    Pick = MatchPartnersBySector(Seed.obj,Pool.objs,W);
    if ~isempty(Pick)
        Anchor = Pool(Pick(1));
    end
end

function Dec = GenerateExpansionCandidate(Problem,FeasibleSeed,InfeasibleAnchor,StepScale)
    if nargin < 4
        StepScale = 0.12;
    end
    Parent = [FeasibleSeed.dec;FeasibleSeed.dec];
    if ~isempty(InfeasibleAnchor)
        Parent = [FeasibleSeed.dec;InfeasibleAnchor.dec];
    end
    Dec = OperatorGAhalf(Problem,Parent,{1,20,1,20});
    Dec = Dec(1,:);

    RealIdx = find(Problem.encoding<=2);
    if ~isempty(RealIdx)
        Range = Problem.upper(RealIdx) - Problem.lower(RealIdx);
        Range(Range<1e-12) = 1;
        if isempty(InfeasibleAnchor)
            Direction = randn(1,numel(RealIdx));
        else
            Direction = (FeasibleSeed.dec(RealIdx)-InfeasibleAnchor.dec(RealIdx))./Range;
        end
        if norm(Direction) < 1e-12
            Direction = randn(1,numel(RealIdx));
        end
        Direction = Direction./max(norm(Direction),1e-12);
        Ortho = randn(1,numel(RealIdx));
        Ortho = Ortho - (Ortho*Direction')*Direction;
        Ortho = Ortho./max(norm(Ortho),1e-12);
        Step = StepScale*Direction + 0.05*Ortho;
        Dec(RealIdx) = FeasibleSeed.dec(RealIdx) + Step.*Range;
    end
    Dec = Problem.CalDec(Dec);
end

function Dec = GenerateBracketMidpoint(Problem,FeasibleDec,InfeasibleDec,Lambda)
    Dec = FeasibleDec;
    RealIdx = find(Problem.encoding<=2);
    if ~isempty(RealIdx)
        Dec(RealIdx) = FeasibleDec(RealIdx) + Lambda*(InfeasibleDec(RealIdx)-FeasibleDec(RealIdx));
    end
    OtherIdx = setdiff(1:Problem.D,RealIdx);
    if ~isempty(OtherIdx)
        Mask = rand(1,numel(OtherIdx)) < Lambda;
        Dec(OtherIdx(Mask)) = InfeasibleDec(OtherIdx(Mask));
    end
    Dec = Problem.CalDec(Dec);
end

function Dec = GenerateHardNegativeSample(Problem,Seed,Anchor,Radius)
    Dec = Seed.dec;
    RealIdx = find(Problem.encoding<=2);
    if ~isempty(RealIdx)
        Range = Problem.upper(RealIdx) - Problem.lower(RealIdx);
        Range(Range<1e-12) = 1;
        Noise = randn(1,numel(RealIdx));
        if ~isempty(Anchor)
            Drift = (Seed.dec(RealIdx)-Anchor.dec(RealIdx))./Range;
            Noise = Noise + 0.5*Drift;
        end
        Dec(RealIdx) = Seed.dec(RealIdx) + Radius*Noise.*Range;
    end
    if isempty(Anchor)
        Parent = [Seed.dec;Seed.dec];
    else
        Parent = [Seed.dec;Anchor.dec];
    end
    Mutant = OperatorGAhalf(Problem,Parent,{0,20,1,20});
    Mutant = Mutant(1,:);
    OtherIdx = setdiff(1:Problem.D,RealIdx);
    if ~isempty(OtherIdx)
        Dec(OtherIdx) = Mutant(OtherIdx);
    end
    Dec = Problem.CalDec(Dec);
end

function Gap = ComputeDecisionGap(Problem,Dec1,Dec2)
    Range = Problem.upper - Problem.lower;
    Range(Range<1e-12) = 1;
    Gap = norm((Dec1-Dec2)./Range)/sqrt(numel(Range));
end

function Pairs = EmptyBracketPairs()
    Pairs.Feasible   = [];
    Pairs.Infeasible = [];
    Pairs.Gap        = zeros(0,1);
end

function Pairs = MergeBracketPairs(Pairs,NewPairs)
    Pairs.Feasible   = [Pairs.Feasible,NewPairs.Feasible];
    Pairs.Infeasible = [Pairs.Infeasible,NewPairs.Infeasible];
    Pairs.Gap        = [Pairs.Gap(:);NewPairs.Gap(:)];
end

function Batch = MergeHardNegBatch(Batch,NewBatch)
    Batch.Dec    = [Batch.Dec;NewBatch.Dec];
    Batch.Radius = [Batch.Radius(:);NewBatch.Radius(:)];
end

function Population = MergePopulationCells(PopCell)
    Population = [];
    if isempty(PopCell)
        return;
    end
    Valid = ~cellfun(@isempty,PopCell);
    if any(Valid)
        Population = [PopCell{Valid}];
    end
end

function Info = MergeBoundaryInfoCells(InfoCell,M)
    Info = EmptyBoundaryInfo(M);
    if isempty(InfoCell)
        return;
    end
    Valid = cellfun(@(S) isstruct(S) && isfield(S,'source') && ~isempty(S.source),InfoCell);
    if ~any(Valid)
        return;
    end
    InfoCell = InfoCell(Valid);
    Fields = {'source','score','prob','entropy','hvGain','novelty','penalty','utility','sector','proxyObjs'};
    for i = 1 : numel(Fields)
        FieldCell = cellfun(@(S) S.(Fields{i}),InfoCell,'UniformOutput',false);
        Info.(Fields{i}) = vertcat(FieldCell{:});
    end
end

function Info = EmptyBoundaryInfo(M)
    Info.source    = zeros(0,1);
    Info.score     = zeros(0,1);
    Info.prob      = zeros(0,1);
    Info.entropy   = zeros(0,1);
    Info.hvGain    = zeros(0,1);
    Info.novelty   = zeros(0,1);
    Info.penalty   = zeros(0,1);
    Info.utility   = zeros(0,1);
    Info.sector    = zeros(0,1);
    Info.proxyObjs = zeros(0,M);
end
