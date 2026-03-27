function [Offspring,Info,FeasiblePool,BracketPairs,HardNegativeBatch,WorkerAudit] = RefineBoundaryWorkers( ...
    Problem,BoundarySeeds,BoundaryInfo,FeasibleObj,Model,W,HardNegativeArchive,Budget,RuntimeOptions)
% Execute one-step feasible exploit / infeasible bracket shrink around bridge seeds.

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
    if isempty(BoundarySeeds)
        return;
    end
    Remaining = max(0,Budget);
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

        Seed   = BoundarySeeds(i);
        Source = SafeBoundaryInfo(BoundaryInfo,'source',i,0);
        Anchor = ResolveEndpoint(BoundaryInfo,'anchor',i);
        Helper = ResolveEndpoint(BoundaryInfo,'helper',i);

        if all(Seed.cons<=0,2)
            [Desc,FeasibleAdd,DescBoundaryLocal,DescTrainKeep,ForwardSuccess] = HandleFeasibleSeed( ...
                Problem,Seed,Helper,Remaining,RuntimeOptions);
            NewBracket = EmptyBracketPairs();
            NewHardNeg.Dec    = zeros(0,Problem.D);
            NewHardNeg.Radius = zeros(0,1);
            InitialGap = NaN;
            FinalGap = NaN;
            ShrinkSuccess = false;
            TightSuccess = false;
        else
            [Desc,FeasibleAdd,NewBracket,NewHardNeg,DescBoundaryLocal,DescTrainKeep, ...
                InitialGap,FinalGap,ShrinkSuccess,TightSuccess] = HandleInfeasibleSeed( ...
                Problem,Seed,Anchor,Helper,Remaining,RuntimeOptions);
            ForwardSuccess = false;
        end

        Remaining = max(0,Remaining - numel(Desc));
        WorkerAudit(i).localEvalCount = numel(Desc);
        WorkerAudit(i).initialBracketGap = InitialGap;
        WorkerAudit(i).bracketGap = FinalGap;
        WorkerAudit(i).shrinkSuccess = ShrinkSuccess;
        WorkerAudit(i).tightSuccess = TightSuccess;
        WorkerAudit(i).feasibleForwardSuccess = ForwardSuccess;
        WorkerAudit(i).hardNegativeConfirmed = ~isempty(NewHardNeg.Dec);
        DescFeasibleMask = SolutionFeasibleMask(Desc);
        WorkerAudit(i).frrSuccess = ~all(Seed.cons<=0,2) && any(DescFeasibleMask);
        WorkerAudit(i).lineageFeasibleDec = GetSeedLineageFeasibleDec(Seed,Desc);
        WorkerAudit(i).lineageDescDec = SolutionDecMatrix(Desc,Problem.D);
        WorkerAudit(i).lineageDescLabel = double(DescFeasibleMask);

        BracketPairs = MergeBracketPairs(BracketPairs,NewBracket);
        HardNegativeBatch = MergeHardNegBatch(HardNegativeBatch,NewHardNeg);
        FeasibleCell{i}  = FeasibleAdd;
        if isempty(Desc)
            continue;
        end

        OffspringCell{i} = Desc;
        InfoCell{i} = AppendWorkerInfo( ...
            Problem,EmptyBoundaryInfo(Problem.M),Desc,Source,FeasibleObj, ...
            Model,W,HardNegativeArchive,RuntimeOptions,DescBoundaryLocal,DescTrainKeep);
    end

    Offspring    = MergePopulationCells(OffspringCell);
    FeasiblePool = MergePopulationCells(FeasibleCell);
    Info         = MergeBoundaryInfoCells(InfoCell,Problem.M);
end

function [Desc,FeasiblePool,BoundaryLocal,TrainKeep,ForwardSuccess] = HandleFeasibleSeed( ...
    Problem,Seed,Helper,Remaining,RuntimeOptions)
    Desc = [];
    FeasiblePool = Seed;
    BoundaryLocal = false(0,1);
    TrainKeep = false(0,1);
    ForwardSuccess = false;
    if ResolveDisableFeasibleForward(RuntimeOptions) || Remaining <= 0 ...
            || isempty(Helper) || ~isfield(Helper,'dec') || isempty(Helper.dec)
        return;
    end

    TrialDec = GenerateForwardExploit( ...
        Problem,Seed.dec,ResolveEndpointDec(Helper,Seed.dec),ResolveForwardAlpha(RuntimeOptions));
    TrialSol = Problem.Evaluation(TrialDec);
    Desc = TrialSol;
    BoundaryLocal = false(numel(Desc),1);
    TrainKeep = false(numel(Desc),1);
    ForwardSuccess = IsForwardImproved(Seed,TrialSol);
    if ForwardSuccess
        % idea.md requires an exclusive keep rule: retain z only when it is
        % still feasible and locally better; otherwise keep x*.
        FeasiblePool = TrialSol;
    end
end

function [Desc,FeasiblePool,BracketPairs,HardNegativeBatch,BoundaryLocal,TrainKeep, ...
    InitialGap,FinalGap,ShrinkSuccess,TightSuccess] = HandleInfeasibleSeed( ...
    Problem,Seed,Anchor,~,Remaining,RuntimeOptions)
    Desc = [];
    FeasiblePool = [];
    BracketPairs  = EmptyBracketPairs();
    HardNegativeBatch.Dec    = zeros(0,Problem.D);
    HardNegativeBatch.Radius = zeros(0,1);
    BoundaryLocal = false(0,1);
    TrainKeep = false(0,1);
    InitialGap = NaN;
    FinalGap = NaN;
    ShrinkSuccess = false;
    TightSuccess = false;

    if isempty(Anchor) || isempty(Anchor.dec)
        HardNegativeBatch = AppendHardNegative( ...
            HardNegativeBatch,Seed.dec,ResolveNegativeRadius(Problem,Seed.dec,Seed.dec));
        return;
    end

    BaseGap = ResolveNegativeRadius(Problem,Seed.dec,Anchor.dec);
    InitialGap = BaseGap;
    FinalGap = BaseGap;
    HardNegativeBatch = AppendHardNegative(HardNegativeBatch,Seed.dec,BaseGap);
    if ResolveDisableInfeasibleShrink(RuntimeOptions) || Remaining <= 0
        BracketPairs.FeasibleDec   = Anchor.dec;
        BracketPairs.InfeasibleDec = Seed.dec;
        BracketPairs.Gap        = BaseGap;
        TightSuccess = isfinite(FinalGap) && FinalGap <= ResolveBracketTightGap(RuntimeOptions);
        return;
    end

    TightGap = ResolveBracketTightGap(RuntimeOptions);
    MidDec = GenerateBracketMidpoint(Problem,Anchor.dec,Seed.dec,0.5);
    MidSol = Problem.Evaluation(MidDec);
    Desc = MidSol;
    if all(MidSol.cons<=0,2)
        NewGap = ComputeDecisionGap(Problem,MidSol.dec,Seed.dec);
        BracketPairs.FeasibleDec   = MidSol.dec;
        BracketPairs.InfeasibleDec = Seed.dec;
        BracketPairs.Gap        = NewGap;
    else
        HardNegativeBatch = AppendHardNegative( ...
            HardNegativeBatch,MidSol.dec,ResolveNegativeRadius(Problem,MidSol.dec,Anchor.dec));
        NewGap = ComputeDecisionGap(Problem,Anchor.dec,MidSol.dec);
        BracketPairs.FeasibleDec   = Anchor.dec;
        BracketPairs.InfeasibleDec = MidSol.dec;
        BracketPairs.Gap        = NewGap;
    end
    FinalGap = NewGap;
    ShrinkSuccess = isfinite(InitialGap) && isfinite(FinalGap) ...
        && FinalGap < InitialGap - ResolveGapImproveTolerance();
    TightSuccess = isfinite(FinalGap) && FinalGap <= TightGap;
    BoundaryLocal = repmat(isfinite(NewGap) && NewGap <= TightGap,numel(Desc),1);
    TrainKeep = BoundaryLocal;
end

function Info = AppendWorkerInfo( ...
    Problem,Info,Desc,Source,FeasibleObj,Model,W,HardNegativeArchive,RuntimeOptions,BoundaryLocal,TrainKeep)
    Detail = ScoreBoundaryCandidates( ...
        Problem,Desc.decs,Desc.objs,FeasibleObj,Model,W,HardNegativeArchive,RuntimeOptions);
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
    if nargin < 10 || isempty(BoundaryLocal)
        BoundaryLocal = false(Count,1);
    end
    if nargin < 11 || isempty(TrainKeep)
        TrainKeep = false(Count,1);
    end
    Info.boundaryLocal = [Info.boundaryLocal;logical(BoundaryLocal(:))];
    Info.trainKeep     = [Info.trainKeep;logical(TrainKeep(:))];
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

function Dec = GenerateForwardExploit(Problem,BaseDec,InfeasibleDec,Alpha)
    if nargin < 4
        Alpha = 0.10;
    end
    Dec = BaseDec;
    RealIdx = find(Problem.encoding<=2);
    if ~isempty(RealIdx)
        Dec(RealIdx) = BaseDec(RealIdx) + Alpha*(BaseDec(RealIdx)-InfeasibleDec(RealIdx));
    end
    Dec = Problem.CalDec(Dec);
end

function Dec = GenerateBracketMidpoint(Problem,FeasibleDec,InfeasibleDec,Lambda)
    Dec = InterpolateBoundaryPoint(Problem,FeasibleDec,InfeasibleDec,Lambda);
end

function Flag = IsForwardImproved(Seed,TrialSol)
    Flag = false;
    if isempty(TrialSol) || ~all(TrialSol.cons<=0,2)
        return;
    end
    if DominatesObj(TrialSol.obj,Seed.obj)
        Flag = true;
        return;
    end
    if DominatesObj(Seed.obj,TrialSol.obj)
        return;
    end
    Flag = sum(TrialSol.obj) < sum(Seed.obj);
end

function Flag = DominatesObj(ObjA,ObjB)
    Flag = all(ObjA <= ObjB) && any(ObjA < ObjB);
end

function Alpha = ResolveForwardAlpha(RuntimeOptions)
    Alpha = 0.10;
    if isstruct(RuntimeOptions) && isfield(RuntimeOptions,'ForwardAlpha') ...
            && ~isempty(RuntimeOptions.ForwardAlpha)
        Alpha = RuntimeOptions.ForwardAlpha;
    end
    Alpha = max(Alpha,0);
end

function Flag = ResolveDisableFeasibleForward(RuntimeOptions)
    Flag = isstruct(RuntimeOptions) && isfield(RuntimeOptions,'DisableFeasibleForward') ...
        && logical(RuntimeOptions.DisableFeasibleForward);
end

function Flag = ResolveDisableInfeasibleShrink(RuntimeOptions)
    Flag = isstruct(RuntimeOptions) && isfield(RuntimeOptions,'DisableInfeasibleShrink') ...
        && logical(RuntimeOptions.DisableInfeasibleShrink);
end

function Tol = ResolveGapImproveTolerance()
    Tol = 1e-12;
end

function Radius = ResolveNegativeRadius(Problem,Dec1,Dec2)
    Radius = 0.10;
    if nargin >= 3 && ~isempty(Dec1) && ~isempty(Dec2)
        Radius = max(ComputeDecisionGap(Problem,Dec1,Dec2),0.02);
    end
end

function Gap = ResolveBracketTightGap(RuntimeOptions)
    Gap = 0.03;
    if isstruct(RuntimeOptions) && isfield(RuntimeOptions,'BracketTightGap') ...
            && ~isempty(RuntimeOptions.BracketTightGap)
        Gap = max(RuntimeOptions.BracketTightGap,0);
    end
end

function Batch = AppendHardNegative(Batch,Dec,Radius)
    if isempty(Dec)
        return;
    end
    Batch.Dec = [Batch.Dec;Dec];
    Batch.Radius = [Batch.Radius(:);repmat(max(Radius,1e-6),size(Dec,1),1)];
end

function Gap = ComputeDecisionGap(Problem,Dec1,Dec2)
    Range = Problem.upper - Problem.lower;
    Range(Range<1e-12) = 1;
    Gap = norm((Dec1-Dec2)./Range)/sqrt(numel(Range));
end

function Pairs = EmptyBracketPairs()
    Pairs.FeasibleDec   = zeros(0,0);
    Pairs.InfeasibleDec = zeros(0,0);
    Pairs.Gap           = zeros(0,1);
end

function Pairs = MergeBracketPairs(Pairs,NewPairs)
    if isempty(Pairs.FeasibleDec)
        Pairs.FeasibleDec = NewPairs.FeasibleDec;
    elseif ~isempty(NewPairs.FeasibleDec)
        Pairs.FeasibleDec = [Pairs.FeasibleDec;NewPairs.FeasibleDec];
    end
    if isempty(Pairs.InfeasibleDec)
        Pairs.InfeasibleDec = NewPairs.InfeasibleDec;
    elseif ~isempty(NewPairs.InfeasibleDec)
        Pairs.InfeasibleDec = [Pairs.InfeasibleDec;NewPairs.InfeasibleDec];
    end
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
        'reliability','boundaryTrust','utility','sector','eligible','boundaryLocal','trainKeep','proxyObjs'};
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
    Info.boundaryLocal = false(0,1);
    Info.trainKeep     = false(0,1);
    Info.proxyObjs     = zeros(0,M);
end

function Audit = InitWorkerAudit(D)
    Audit = struct( ...
        'seedIndex',0, ...
        'seedFeasible',false, ...
        'localEvalCount',0, ...
        'feasibleForwardSuccess',false, ...
        'frrSuccess',false, ...
        'initialBracketGap',NaN, ...
        'bracketGap',NaN, ...
        'shrinkSuccess',false, ...
        'tightSuccess',false, ...
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

function Mask = SolutionFeasibleMask(Solutions)
    Mask = false(0,1);
    if isempty(Solutions)
        return;
    end
    Mask = all(Solutions.cons<=0,2);
end

function FeasibleDec = GetSeedLineageFeasibleDec(Seed,Desc)
    D = numel(Seed.dec);
    FeasibleDec = zeros(0,D);
    if all(Seed.cons<=0,2)
        FeasibleDec = Seed.dec;
    end
    if ~isempty(Desc)
        DescFeasible = Desc(SolutionFeasibleMask(Desc));
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
