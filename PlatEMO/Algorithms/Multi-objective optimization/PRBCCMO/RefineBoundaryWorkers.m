function [Offspring,Info,FeasiblePool,BracketPairs,HardNegativeBatch,WorkerAudit] = RefineBoundaryWorkers( ...
    Problem,BoundarySeeds,BoundaryInfo,FeasibleObj,Model,W,HardNegativeArchive,Budget,RuntimeOptions)
% Execute label-aware local refinement around real-evaluated bridge seeds.

    if nargin < 9 || ~isstruct(RuntimeOptions)
        RuntimeOptions = struct();
    end

    Offspring = [];
    FeasiblePool = [];
    BracketPairs  = EmptyBracketPairs();
    HardNegativeBatch.Dec    = zeros(0,Problem.D);
    HardNegativeBatch.Radius = zeros(0,1);
    Info = EmptyBoundaryInfo(Problem.M);
    WorkerAudit = repmat(InitWorkerAudit(Problem.D),0,1);
    if Budget <= 0 || isempty(BoundarySeeds)
        return;
    end

    Remaining = Budget;
    OffspringCell = cell(1,numel(BoundarySeeds));
    FeasibleCell  = cell(1,numel(BoundarySeeds));
    InfoCell      = cell(1,numel(BoundarySeeds));
    if nargin < 4 || isempty(FeasibleObj)
        FeasibleObj = zeros(0,Problem.M);
    end
    WorkerAudit = repmat(InitWorkerAudit(Problem.D),numel(BoundarySeeds),1);

    for i = 1 : numel(BoundarySeeds)
        WorkerAudit(i) = InitWorkerAudit(Problem.D);
        WorkerAudit(i).seedIndex = i;
        WorkerAudit(i).seedFeasible = all(BoundarySeeds(i).cons<=0,2);
        if Remaining <= 0
            WorkerAudit(i).lineageFeasibleDec = GetSeedLineageFeasibleDec(BoundarySeeds(i),[]);
            continue;
        end

        Seed   = BoundarySeeds(i);
        Source = SafeBoundaryInfo(BoundaryInfo,'source',i,0);
        Anchor = ResolveEndpoint(BoundaryInfo,'anchor',i);
        Helper = ResolveEndpoint(BoundaryInfo,'helper',i);

        if UseIsotropicLocal(RuntimeOptions)
            if all(Seed.cons<=0,2)
                [Desc,FeasibleAdd] = HandlePlainFeasibleSeed(Problem,Seed,Remaining);
            else
                [Desc,FeasibleAdd] = HandlePlainInfeasibleSeed(Problem,Seed,Remaining);
            end
            NewBracket = EmptyBracketPairs();
            NewHardNeg.Dec    = zeros(0,Problem.D);
            NewHardNeg.Radius = zeros(0,1);
        else
            if all(Seed.cons<=0,2)
                [Desc,FeasibleAdd] = HandleFeasibleSeed(Problem,Seed,Anchor,Helper,Remaining);
                NewBracket = EmptyBracketPairs();
                NewHardNeg.Dec    = zeros(0,Problem.D);
                NewHardNeg.Radius = zeros(0,1);
            else
                [Desc,FeasibleAdd,NewBracket,NewHardNeg] = HandleInfeasibleSeed( ...
                    Problem,Seed,Anchor,Helper,Remaining);
            end
        end

        Remaining = Remaining - numel(Desc);
        WorkerAudit(i).localEvalCount = numel(Desc);
        WorkerAudit(i).bracketGap = NaN;
        if ~isempty(NewBracket.Gap)
            WorkerAudit(i).bracketGap = NewBracket.Gap(1);
        end
        WorkerAudit(i).hardNegativeConfirmed = ~isempty(NewHardNeg.Dec);
        WorkerAudit(i).frrSuccess = ~all(Seed.cons<=0,2) && any(all(Desc.cons<=0,2));
        WorkerAudit(i).lineageFeasibleDec = GetSeedLineageFeasibleDec(Seed,Desc);
        WorkerAudit(i).lineageDescDec = SolutionDecMatrix(Desc,Problem.D);
        WorkerAudit(i).lineageDescLabel = double(all(Desc.cons<=0,2));

        BracketPairs = MergeBracketPairs(BracketPairs,NewBracket);
        HardNegativeBatch = MergeHardNegBatch(HardNegativeBatch,NewHardNeg);
        if isempty(Desc)
            continue;
        end

        OffspringCell{i} = Desc;
        FeasibleCell{i}  = FeasibleAdd;
        InfoCell{i} = AppendWorkerInfo( ...
            Problem,EmptyBoundaryInfo(Problem.M),Desc,Source,FeasibleObj,Model,W,HardNegativeArchive);
    end

    Offspring    = MergePopulationCells(OffspringCell);
    FeasiblePool = MergePopulationCells(FeasibleCell);
    Info         = MergeBoundaryInfoCells(InfoCell,Problem.M);
end

function [Desc,FeasiblePool] = HandleFeasibleSeed(Problem,Seed,~,Helper,Remaining)
    Desc = [];
    FeasiblePool = [];
    if Remaining <= 0
        return;
    end

    TrialDec = GenerateForwardTrial(Problem,Seed.dec,ResolveEndpointDec(Helper,Seed.dec),0.06,0.02);
    TrialSol = Problem.Evaluation(TrialDec);
    Desc = TrialSol;
    FeasiblePool = TrialSol(all(TrialSol.cons<=0,2));
    if any(TrialSol.cons>0,2) && Remaining > 1
        BackDec = GenerateBracketMidpoint(Problem,Seed.dec,TrialSol.dec,0.5);
        BackSol = Problem.Evaluation(BackDec);
        Desc = [Desc,BackSol];
        FeasiblePool = [FeasiblePool,BackSol(all(BackSol.cons<=0,2))];
    end
end

function [Desc,FeasiblePool,BracketPairs,HardNegativeBatch] = HandleInfeasibleSeed(Problem,Seed,Anchor,Helper,Remaining)
    Desc = [];
    FeasiblePool = [];
    BracketPairs  = EmptyBracketPairs();
    HardNegativeBatch.Dec    = zeros(0,Problem.D);
    HardNegativeBatch.Radius = zeros(0,1);
    if Remaining <= 0
        return;
    end

    if isempty(Anchor) || isempty(Anchor.dec)
        [Desc,FeasiblePool,HardNegativeBatch] = ConfirmHardNegative( ...
            Problem,Seed,Anchor,Remaining);
        return;
    end

    FeasibleEnd = Anchor;
    InfeasibleEnd = Seed;
    FoundFeasible = false;
    MaxBracketStep = min(3,Remaining);
    MidCell = cell(1,MaxBracketStep);
    MidCount = 0;
    for k = 1 : MaxBracketStep
        MidDec = GenerateBracketMidpoint(Problem,FeasibleEnd.dec,InfeasibleEnd.dec,0.5);
        MidSol = Problem.Evaluation(MidDec);
        MidCount = MidCount + 1;
        MidCell{MidCount} = MidSol;
        if all(MidSol.cons<=0,2)
            FeasibleEnd = MidSol;
            FoundFeasible = true;
        else
            InfeasibleEnd = MidSol;
        end
    end
    Desc = MergePopulationCells(MidCell(1:MidCount));
    FeasiblePool = Desc(all(Desc.cons<=0,2));

    Remaining = Remaining - numel(Desc);
    if FoundFeasible
        BracketPairs.Feasible   = FeasibleEnd;
        BracketPairs.Infeasible = InfeasibleEnd;
        BracketPairs.Gap        = ComputeDecisionGap(Problem,FeasibleEnd.dec,InfeasibleEnd.dec);
        if Remaining > 0
            TrialDec = GenerateForwardTrial( ...
                Problem,FeasibleEnd.dec,ResolveEndpointDec(Helper,Seed.dec),0.06,0.02);
            TrialSol = Problem.Evaluation(TrialDec);
            Desc = [Desc,TrialSol];
            FeasiblePool = [FeasiblePool,TrialSol(all(TrialSol.cons<=0,2))];
        end
        return;
    end

    [HardDesc,HardFeasible,HardNegativeBatch] = ConfirmHardNegative( ...
        Problem,Seed,Anchor,Remaining);
    Desc = [Desc,HardDesc];
    FeasiblePool = [FeasiblePool,HardFeasible];
end

function [Desc,FeasiblePool] = HandlePlainFeasibleSeed(Problem,Seed,Remaining)
    Desc = [];
    FeasiblePool = [];
    if Remaining <= 0
        return;
    end

    QueryCount = min(2,Remaining);
    Decs = zeros(QueryCount,Problem.D);
    for j = 1 : QueryCount
        Decs(j,:) = GenerateIsotropicLocalSample(Problem,Seed,0.10 + 0.06*(j-1));
    end
    Desc = Problem.Evaluation(Decs);
    FeasiblePool = Desc(all(Desc.cons<=0,2));
end

function [Desc,FeasiblePool] = HandlePlainInfeasibleSeed(Problem,Seed,Remaining)
    Desc = [];
    FeasiblePool = [];
    if Remaining <= 0
        return;
    end

    QueryCount = min(2,Remaining);
    Decs = zeros(QueryCount,Problem.D);
    for j = 1 : QueryCount
        Decs(j,:) = GenerateIsotropicLocalSample(Problem,Seed,0.06 + 0.04*(j-1));
    end
    Desc = Problem.Evaluation(Decs);
    FeasiblePool = Desc(all(Desc.cons<=0,2));
end

function [Desc,FeasiblePool,HardNegativeBatch] = ConfirmHardNegative(Problem,Seed,Anchor,Remaining)
    Desc = [];
    FeasiblePool = [];
    HardNegativeBatch.Dec    = zeros(0,Problem.D);
    HardNegativeBatch.Radius = zeros(0,1);
    if Remaining <= 0
        return;
    end

    Delta = 0.10;
    if ~isempty(Anchor) && isfield(Anchor,'dec') && ~isempty(Anchor.dec)
        Delta = max(ComputeDecisionGap(Problem,Seed.dec,Anchor.dec),0.02);
    end
    RadiusList = Delta*[0.5,0.25];
    QueryCount = min(numel(RadiusList),Remaining);
    Decs = zeros(QueryCount,Problem.D);
    for j = 1 : QueryCount
        Decs(j,:) = GenerateHardNegativeSample(Problem,Seed,Anchor,RadiusList(j));
    end
    Desc = Problem.Evaluation(Decs);
    FeasiblePool = Desc(all(Desc.cons<=0,2));
    if isempty(FeasiblePool)
        HardNegativeBatch.Dec    = Seed.dec;
        HardNegativeBatch.Radius = Delta;
    end
end

function Info = AppendWorkerInfo(Problem,Info,Desc,Source,FeasibleObj,Model,W,HardNegativeArchive)
    Detail = ScoreBoundaryCandidates(Problem,Desc.decs,Desc.objs,FeasibleObj,Model,W,HardNegativeArchive);
    Count = numel(Desc);
    Info.source        = [Info.source;repmat(Source,Count,1)];
    Info.score         = [Info.score;Detail.utility(:)];
    Info.prob          = [Info.prob;Detail.prob(:)];
    Info.queryScore    = [Info.queryScore;Detail.queryScore(:)];
    Info.disagreement  = [Info.disagreement;Detail.disagreement(:)];
    Info.paretoValue   = [Info.paretoValue;Detail.paretoValue(:)];
    Info.reliability   = [Info.reliability;Detail.reliability(:)];
    Info.boundaryTrust = [Info.boundaryTrust;Detail.boundaryTrust(:)];
    Info.utility       = [Info.utility;Detail.utility(:)];
    Info.sector        = [Info.sector;Detail.sector(:)];
    Info.eligible      = [Info.eligible;Detail.eligible(:)];
    Info.proxyObjs     = [Info.proxyObjs;Desc.objs];
end

function Endpoint = ResolveEndpoint(Info,Prefix,Index)
    Endpoint = [];
    DecField = [Prefix,'Dec'];
    ObjField = [Prefix,'Obj'];
    if ~isstruct(Info) || ~isfield(Info,DecField) || size(Info.(DecField),1) < Index
        return;
    end
    Endpoint = struct();
    Endpoint.dec = Info.(DecField)(Index,:);
    if isfield(Info,ObjField) && size(Info.(ObjField),1) >= Index
        Endpoint.obj = Info.(ObjField)(Index,:);
    else
        Endpoint.obj = [];
    end
end

function Value = SafeBoundaryInfo(Info,Field,Index,Default)
    Value = Default;
    if isstruct(Info) && isfield(Info,Field) && numel(Info.(Field)) >= Index
        Value = Info.(Field)(Index);
    end
end

function Dec = ResolveEndpointDec(Endpoint,DefaultDec)
    Dec = DefaultDec;
    if ~isempty(Endpoint) && isfield(Endpoint,'dec') && ~isempty(Endpoint.dec)
        Dec = Endpoint.dec;
    end
end

function Dec = GenerateForwardTrial(Problem,BaseDec,TargetDec,StepScale,OrthoScale)
    if nargin < 4
        StepScale = 0.06;
    end
    if nargin < 5
        OrthoScale = 0.02;
    end

    Dec = BaseDec;
    RealIdx = find(Problem.encoding<=2);
    if ~isempty(RealIdx)
        Range = Problem.upper(RealIdx) - Problem.lower(RealIdx);
        Range(Range<1e-12) = 1;
        Direction = (TargetDec(RealIdx)-BaseDec(RealIdx))./Range;
        if norm(Direction) < 1e-12
            Direction = randn(1,numel(RealIdx));
        end
        Direction = Direction./max(norm(Direction),1e-12);
        Noise = randn(1,numel(RealIdx));
        Noise = Noise - (Noise*Direction')*Direction;
        Noise = Noise./max(norm(Noise),1e-12);
        Step = StepScale*Direction + OrthoScale*Noise;
        Dec(RealIdx) = BaseDec(RealIdx) + Step.*Range;
    end

    Parent = [BaseDec;TargetDec];
    Mutant = OperatorGAhalf(Problem,Parent,{0,20,1,20});
    Mutant = Mutant(1,:);
    OtherIdx = setdiff(1:Problem.D,RealIdx);
    if ~isempty(OtherIdx)
        Dec(OtherIdx) = Mutant(OtherIdx);
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
        if ~isempty(Anchor) && isfield(Anchor,'dec') && ~isempty(Anchor.dec)
            Drift = (Seed.dec(RealIdx)-Anchor.dec(RealIdx))./Range;
            Noise = Noise + Drift;
        end
        Noise = Noise./max(norm(Noise),1e-12);
        Dec(RealIdx) = Seed.dec(RealIdx) + Radius*Noise.*Range;
    end
    Parent = [Seed.dec;ResolveEndpointDec(Anchor,Seed.dec)];
    Mutant = OperatorGAhalf(Problem,Parent,{0,20,1,20});
    Mutant = Mutant(1,:);
    OtherIdx = setdiff(1:Problem.D,RealIdx);
    if ~isempty(OtherIdx)
        Dec(OtherIdx) = Mutant(OtherIdx);
    end
    Dec = Problem.CalDec(Dec);
end

function Dec = GenerateIsotropicLocalSample(Problem,Seed,StepScale)
    Parent = [Seed.dec;Seed.dec];
    Dec = OperatorGAhalf(Problem,Parent,{0,20,1,20});
    Dec = Dec(1,:);
    RealIdx = find(Problem.encoding<=2);
    if ~isempty(RealIdx)
        Range = Problem.upper(RealIdx) - Problem.lower(RealIdx);
        Range(Range<1e-12) = 1;
        Noise = randn(1,numel(RealIdx));
        Noise = Noise./max(norm(Noise),1e-12);
        Dec(RealIdx) = Seed.dec(RealIdx) + StepScale*Noise.*Range;
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
    Fields = {'source','score','prob','queryScore','disagreement','paretoValue', ...
        'reliability','boundaryTrust','utility','sector','eligible','proxyObjs'};
    for i = 1 : numel(Fields)
        FieldCell = cellfun(@(S) S.(Fields{i}),InfoCell,'UniformOutput',false);
        Info.(Fields{i}) = vertcat(FieldCell{:});
    end
end

function Info = EmptyBoundaryInfo(M)
    Info = struct();
    Info.source        = zeros(0,1);
    Info.score         = zeros(0,1);
    Info.prob          = zeros(0,1);
    Info.queryScore    = zeros(0,1);
    Info.disagreement  = zeros(0,1);
    Info.paretoValue   = zeros(0,1);
    Info.reliability   = zeros(0,1);
    Info.boundaryTrust = zeros(0,1);
    Info.utility       = zeros(0,1);
    Info.sector        = zeros(0,1);
    Info.eligible      = false(0,1);
    Info.proxyObjs     = zeros(0,M);
end

function Audit = InitWorkerAudit(D)
    Audit = struct( ...
        'seedIndex',0, ...
        'seedFeasible',false, ...
        'localEvalCount',0, ...
        'frrSuccess',false, ...
        'bracketGap',NaN, ...
        'hardNegativeConfirmed',false, ...
        'lineageFeasibleDec',zeros(0,D), ...
        'lineageDescDec',zeros(0,D), ...
        'lineageDescLabel',zeros(0,1));
end

function Dec = SolutionDecMatrix(Solutions,D)
    if isempty(Solutions)
        Dec = zeros(0,D);
        return;
    end
    Dec = Solutions.decs;
end

function FeasibleDec = GetSeedLineageFeasibleDec(Seed,Desc)
    D = numel(Seed.dec);
    FeasibleDec = zeros(0,D);
    if all(Seed.cons<=0,2)
        FeasibleDec = Seed.dec;
    end
    if ~isempty(Desc)
        DescFeasible = Desc(all(Desc.cons<=0,2));
        if ~isempty(DescFeasible)
            FeasibleDec = [FeasibleDec;DescFeasible.decs];
        end
    end
    if isempty(FeasibleDec)
        return;
    end
    Keep = KeepLatestDecisionRows(FeasibleDec);
    FeasibleDec = FeasibleDec(Keep,:);
end

function Flag = UseIsotropicLocal(RuntimeOptions)
    Flag = false;
    if isstruct(RuntimeOptions) && isfield(RuntimeOptions,'LocalMode') && ~isempty(RuntimeOptions.LocalMode)
        Flag = round(RuntimeOptions.LocalMode) == 2;
    end
end
