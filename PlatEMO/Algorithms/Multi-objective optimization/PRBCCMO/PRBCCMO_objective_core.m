function varargout = PRBCCMO_objective_core(Action,varargin)
% Shared objective-boundary implementation for PRBCCMO and PRBCCMO_t.

    switch lower(char(Action))
        case 'run'
            runObjectiveBoundaryCore(varargin{:});
            varargout = {};
        case 'initobserver'
            varargout{1} = InitObserver(varargin{:});
        otherwise
            error('PRBCCMO_objective_core:UnknownAction','Unknown action: %s',char(Action));
    end
end

function runObjectiveBoundaryCore(~,Problem,TraceMode,Params,NotTerminated,Observer)
    [W,Problem.N] = UniformPoint(Problem.N,Problem.M);
    Params = normalizeParams(Params);
    hidden   = Params(1);
    epoch    = Params(2);
    lr       = Params(3);
    betaB    = Params(4);
    etaB     = Params(5);
    Tretrain = Params(6);
    Gstart   = Params(7);
    rhoRef   = 0.10;
    Kbis     = 3;

    NBPair = max(1,ceil(betaB*Problem.N));
    MaxTrain = max(2*NBPair,4*Problem.N);

    PopulationC = Problem.Initialization();
    PopulationU = Problem.Initialization();
    B           = PopulationC([]);
    BInfo       = InitBoundaryArchiveInfo(NBPair);
    TrainBuffer = InitTrainingBuffer(Problem.D);
    Model       = [];
    Generation  = 0;
    ResultPopulationC = BuildExternalResultPopulation(PopulationC,PopulationC([]));

    while NotTerminated(ResultPopulationC)
        Generation = Generation + 1;

        OffspringC = GenerateDEOffspring(Problem,PopulationC,true);
        OffspringUDec = GenerateDEOffspringDecision(Problem,PopulationU,false);
        [OffspringU,B,BInfo,ContractionDiag] = EvaluateHelperOffspringWithBoundaryRefinement( ...
            Problem,OffspringUDec,B,BInfo,rhoRef,Kbis,PopulationC,W,NBPair);
        [B,BInfo,~,BlendDiag] = UpdateBoundaryArchiveObjective( ...
            B,BInfo,PopulationC,PopulationU,OffspringC,OffspringU,W,NBPair,etaB,Problem);
        BInfo = ResetContractedPairAges(B,BInfo,ContractionDiag.accepted_key);
        BlendDiag.contracted = ContractionDiag.accepted;

        BoundaryBand = SelectBoundaryBandTrainingPoints( ...
            OffspringU,B,BInfo,PopulationC,W,max(1,ceil(0.10*Problem.N)));
        TrainBuffer = UpdateTrainingBufferT( ...
            TrainBuffer,ContractionDiag.refined,BoundaryBand,Generation,max(0,MaxTrain-2*NBPair));
        [Model,MLPDiag] = UpdateBoundaryMLPPeriodically( ...
            Model,B,BInfo,TrainBuffer,hidden,epoch,lr,Generation,Gstart,Tretrain,Problem);

        QC = KeepUniquePopulation([PopulationC,OffspringC,OffspringU]);
        QU = KeepUniquePopulation([PopulationU,OffspringC,OffspringU]);
        [PopulationC,SelectionDiag] = EnvironmentalSelectionC_ObjectBoundary( ...
            QC,Problem.N,W,Model,B,BInfo,PopulationC);
        PopulationU = EnvironmentalSelectionU(QU,Problem.N,W);
        ResultPopulationC = BuildExternalResultPopulation(PopulationC,ResultPopulationC);

        if TraceMode
            Observer = LogGenerationDiagnostics(Observer,Problem,Generation, ...
                B,BInfo,PopulationC,PopulationU,BlendDiag,MLPDiag,SelectionDiag);
            Observer = WriteBoundarySnapshotsIfDue(Observer,Problem,Generation,B,BInfo);
        end
    end
end

function Params = normalizeParams(Params)
    Defaults = [40,80,0.05,4,0.1,20,150,0];
    if isempty(Params)
        Params = Defaults;
    end
    Params = double(Params(:)');
    if numel(Params) < numel(Defaults)
        Params(end+1:numel(Defaults)) = Defaults(numel(Params)+1:end);
    end
    Params = Params(1:numel(Defaults));
    Params(1) = max(2,round(Params(1)));
    Params(2) = max(1,round(Params(2)));
    Params(3) = max(1e-4,Params(3));
    Params(4) = max(1,Params(4));
    Params(5) = max(0,min(1,Params(5)));
    Params(6) = max(1,round(Params(6)));
    Params(7) = max(0,round(Params(7)));
    Params(8) = double(Params(8) > 0);
end

%% Main-population offspring

function Offspring = GenerateDEOffspring(Problem,Population,useConstraintIndicator)
    if isempty(Population)
        Offspring = Population;
        return;
    end
    Offspring = Problem.Evaluation( ...
        GenerateDEOffspringDecision(Problem,Population,useConstraintIndicator));
end

function OffspringDec = GenerateDEOffspringDecision(Problem,Population,useConstraintIndicator)
    if isempty(Population)
        OffspringDec = zeros(0,Problem.D);
        return;
    end

    N = numel(Population);
    if useConstraintIndicator
        [Flag,FrontNo,CrowdDis] = ConstraintSideIndicator(Population);
        MatingPool = TournamentSelection(2,2*N,Flag,FrontNo,-CrowdDis);
    else
        [FrontNo,CrowdDis] = ObjectiveSideIndicator(Population);
        MatingPool = TournamentSelection(2,2*N,FrontNo,-CrowdDis);
    end

    Base = Population(randi(N,N,1));
    OffspringDec = OperatorDE(Problem,Base.decs, ...
        Population(MatingPool(1:N)).decs,Population(MatingPool(N+1:end)).decs);
end

function [Offspring,B,BInfo,Diag] = EvaluateHelperOffspringWithBoundaryRefinement( ...
    Problem,OffspringDec,B,BInfo,rhoRef,Kbis,PopulationC,W,NBPair)

    Diag = InitContractionDiag(B);
    if isempty(OffspringDec)
        Offspring = B([]);
        return;
    end
    TotalRefBudget = min(size(OffspringDec,1),floor(rhoRef*size(OffspringDec,1)));
    PairPlan = SelectBoundaryRefinementPairs( ...
        Problem,B,BInfo,max(0,floor(TotalRefBudget/max(Kbis,1))));
    if isempty(PairPlan) || Kbis <= 0
        Offspring = Problem.Evaluation(OffspringDec);
        return;
    end

    nRef = numel(PairPlan);
    UsedRows = nRef*Kbis;
    RefinedAll = B([]);
    for step = 1 : Kbis %#ok<NASGU>
        BatchDec = zeros(nRef,Problem.D);
        [BatchDec,RefineInfo] = InjectBoundaryRefinementIntoHelperOffspring( ...
            Problem,BatchDec,B,BInfo,PairPlan);
        Refined = Problem.Evaluation(BatchDec);
        [B,BInfo,StepDiag] = ContractBoundaryPairsByRefinedSamples( ...
            B,BInfo,Refined,RefineInfo,PopulationC,W,NBPair,Problem);
        Diag = MergeContractionDiag(Diag,StepDiag);
        RefinedAll = [RefinedAll,Refined]; %#ok<AGROW>
    end

    RestDec = OffspringDec(UsedRows+1:end,:);
    if isempty(RestDec)
        Rest = B([]);
    else
        Rest = Problem.Evaluation(RestDec);
    end
    Offspring = [RefinedAll,Rest];
end

function [OffspringDec,RefineInfo] = InjectBoundaryRefinementIntoHelperOffspring(Problem,OffspringDec,B,BInfo,PairPlan)
    RefineInfo = struct('rows',zeros(0,1),'pairs',zeros(0,1));
    if isempty(OffspringDec) || isempty(B) || BInfo.PairCount <= 0 || isempty(PairPlan)
        return;
    end

    PairPlan = double(PairPlan(:));
    PairCount = min([BInfo.PairCount,floor(numel(B)/2),numel(BInfo.Active)]);
    Valid = PairPlan >= 1 & PairPlan <= PairCount & BInfo.Active(PairPlan);
    Pick = PairPlan(Valid);
    nRef = min(size(OffspringDec,1),numel(Pick));
    if nRef <= 0
        return;
    end

    Pick = Pick(1:nRef);
    XRef = 0.5*(double(B(2*Pick-1).decs) + double(B(2*Pick).decs));
    XRef = ClipDecisionsToProblemBounds(XRef,Problem.lower,Problem.upper);
    OffspringDec(1:nRef,:) = XRef;
    RefineInfo.rows = (1:nRef)';
    RefineInfo.pairs = Pick(:);
end

function Pick = SelectBoundaryRefinementPairs(Problem,B,BInfo,MaxPairs)
    Pick = zeros(0,1);
    if isempty(B) || BInfo.PairCount <= 0 || MaxPairs <= 0
        return;
    end

    PairCount = min([BInfo.PairCount,floor(numel(B)/2),numel(BInfo.Active)]);
    Active = find(BInfo.Active(1:PairCount));
    if isempty(Active)
        return;
    end

    FeasibleDec = double(B(2*Active-1).decs);
    InfeasibleDec = double(B(2*Active).decs);
    DecGap = DecisionDistanceNormalized(FeasibleDec,InfeasibleDec,Problem.lower,Problem.upper);
    FrontGap = BInfo.FrontGap(Active);
    PairAge = BInfo.Age(Active);
    Valid = isfinite(FrontGap(:)) & isfinite(PairAge(:)) & isfinite(DecGap(:));
    Active = Active(Valid);
    DecGap = DecGap(Valid);
    FrontGap = FrontGap(Valid);
    PairAge = PairAge(Valid);
    if isempty(Active)
        return;
    end

    [~,ord] = sortrows([-DecGap(:),-PairAge(:),FrontGap(:),Active(:)],[1 2 3 4]);
    Pick = Active(ord(1:min(MaxPairs,numel(ord))));
end

%% Objective-space boundary archive

function BInfo = InitBoundaryArchiveInfo(NBPair)
    BInfo = struct( ...
        'PairCount',0, ...
        'Sector',zeros(NBPair,1), ...
        'FrontGap',inf(NBPair,1), ...
        'PairGap',inf(NBPair,1), ...
        'FeasibleSource',ones(NBPair,1), ...
        'InfeasibleSource',ones(NBPair,1), ...
        'Age',inf(NBPair,1), ...
        'Active',false(NBPair,1));
end

function [B,BInfo,Target,BlendDiag] = UpdateBoundaryArchiveObjective( ...
    B,BInfo,PopulationC,PopulationU,OffspringC,OffspringU,W,NBPair,etaB,Problem)

    [A,Source] = BuildBoundaryCandidatePool(B,BInfo,PopulationC,PopulationU,OffspringC,OffspringU);
    FeRatio = min(Problem.FE/max(Problem.maxFE,1),1);
    Target = BuildTargetBoundaryArchive(A,Source,PopulationC,W,NBPair,FeRatio);
    [B,BInfo,BlendDiag] = BlendBoundaryArchive(B,BInfo,Target,etaB,NBPair);
end

function [A,Source] = BuildBoundaryCandidatePool(B,BInfo,PopulationC,PopulationU,OffspringC,OffspringU)
    A = [B,PopulationC,PopulationU,OffspringC,OffspringU];
    Source = [BoundaryEndpointSources(B,BInfo); ...
        BoundarySourceCode('population_c')*ones(numel(PopulationC),1); ...
        BoundarySourceCode('population_u')*ones(numel(PopulationU),1); ...
        BoundarySourceCode('offspring_c')*ones(numel(OffspringC),1); ...
        BoundarySourceCode('offspring_u')*ones(numel(OffspringU),1)];
    [A,Source] = KeepUniquePopulationWithSource(A,Source);
end

function Source = BoundaryEndpointSources(B,BInfo)
    Source = BoundarySourceCode('b')*ones(numel(B),1);
    PairCount = min([floor(numel(B)/2),StructFieldOr(BInfo,'PairCount',0)]);
    if PairCount <= 0
        return;
    end
    [FeasibleSource,InfeasibleSource] = BoundaryPairSources(BInfo,PairCount);
    for p = 1 : PairCount
        Source(2*p-1) = FeasibleSource(p);
        Source(2*p) = InfeasibleSource(p);
    end
end

function [FeasibleSource,InfeasibleSource] = BoundaryPairSources(BInfo,PairCount)
    FeasibleSource = BoundarySourceCode('b')*ones(PairCount,1);
    InfeasibleSource = BoundarySourceCode('b')*ones(PairCount,1);
    if isstruct(BInfo) && isfield(BInfo,'FeasibleSource')
        FeasibleSource = MatchLength(BInfo.FeasibleSource,PairCount,BoundarySourceCode('b'));
    end
    if isstruct(BInfo) && isfield(BInfo,'InfeasibleSource')
        InfeasibleSource = MatchLength(BInfo.InfeasibleSource,PairCount,BoundarySourceCode('b'));
    end
end

function Code = BoundarySourceCode(Name)
    switch lower(char(Name))
        case 'b'
            Code = 1;
        case 'population_c'
            Code = 2;
        case 'population_u'
            Code = 3;
        case 'offspring_c'
            Code = 4;
        case 'offspring_u'
            Code = 5;
        case 'refinement'
            Code = 6;
        otherwise
            Code = 0;
    end
end

function [B,BInfo,Diag] = ContractBoundaryPairsByRefinedSamples( ...
    B,BInfo,OffspringU,RefineInfo,PopulationC,W,NBPair,Problem)

    Diag = InitContractionDiag(B);
    if isempty(B) || isempty(OffspringU) || ~isstruct(RefineInfo) || ...
            ~isfield(RefineInfo,'rows') || ~isfield(RefineInfo,'pairs')
        return;
    end

    PairCount = min([BInfo.PairCount,floor(numel(B)/2),numel(BInfo.Active),NBPair]);
    if PairCount <= 0
        return;
    end

    Rows = double(RefineInfo.rows(:));
    Pairs = double(RefineInfo.pairs(:));
    Count = min(numel(Rows),numel(Pairs));
    for k = 1 : Count
        row = Rows(k);
        p = Pairs(k);
        if row < 1 || row > numel(OffspringU) || p < 1 || p > PairCount || ...
                ~BInfo.Active(p) || 2*p > numel(B)
            continue;
        end

        Refined = OffspringU(row);
        OldF = B(2*p-1);
        OldI = B(2*p);
        if IsFeasibleSolution(Refined)
            CandF = Refined;
            CandI = OldI;
        else
            CandF = OldF;
            CandI = Refined;
        end

        [Accept,CandKey,CandSector] = IsBracketTighter( ...
            OldF,OldI,CandF,CandI,Problem,PopulationC,W);
        Diag.attempted = Diag.attempted + 1;
        Diag.refined = [Diag.refined,Refined]; %#ok<AGROW>
        if Accept
            B(2*p-1) = CandF;
            B(2*p) = CandI;
            BInfo.Sector(p) = CandSector;
            BInfo.FrontGap(p) = CandKey(1);
            BInfo.PairGap(p) = CandKey(2);
            BInfo.Age(p) = 0;
            [FeasibleSource,InfeasibleSource] = BoundaryPairSources(BInfo,PairCount);
            BInfo.FeasibleSource(p) = FeasibleSource(p);
            BInfo.InfeasibleSource(p) = InfeasibleSource(p);
            if IsFeasibleSolution(Refined)
                BInfo.FeasibleSource(p) = BoundarySourceCode('refinement');
            else
                BInfo.InfeasibleSource(p) = BoundarySourceCode('refinement');
            end
            Diag.accepted = Diag.accepted + 1;
            Key = PairKeyMatrix([CandF,CandI]);
            if isempty(Diag.accepted_key) || size(Diag.accepted_key,2) == size(Key,2)
                Diag.accepted_key = [Diag.accepted_key;Key];
            end
        else
            Diag.rejected = Diag.rejected + 1;
        end
    end
end

function Diag = InitContractionDiag(Prototype)
    D = 0;
    if ~isempty(Prototype)
        D = size(Prototype.decs,2);
    end
    Diag = struct( ...
        'attempted',0, ...
        'accepted',0, ...
        'rejected',0, ...
        'accepted_key',zeros(0,2*D), ...
        'refined',Prototype([]));
end

function Diag = MergeContractionDiag(Diag,StepDiag)
    Diag.attempted = Diag.attempted + StepDiag.attempted;
    Diag.accepted = Diag.accepted + StepDiag.accepted;
    Diag.rejected = Diag.rejected + StepDiag.rejected;
    if isempty(Diag.accepted_key)
        Diag.accepted_key = StepDiag.accepted_key;
    elseif ~isempty(StepDiag.accepted_key) && size(Diag.accepted_key,2) == size(StepDiag.accepted_key,2)
        Diag.accepted_key = [Diag.accepted_key;StepDiag.accepted_key];
    end
    Diag.refined = [Diag.refined,StepDiag.refined];
end

function BInfo = ResetContractedPairAges(B,BInfo,AcceptedKey)
    if isempty(B) || isempty(AcceptedKey)
        return;
    end
    PairCount = min([BInfo.PairCount,floor(numel(B)/2),numel(BInfo.Active)]);
    if PairCount <= 0
        return;
    end
    Active = find(BInfo.Active(1:PairCount));
    if isempty(Active)
        return;
    end
    Rows = reshape([2*Active(:)-1,2*Active(:)]',1,[]);
    CurrentKey = PairKeyMatrix(B(Rows));
    if size(CurrentKey,2) ~= size(AcceptedKey,2)
        return;
    end
    Retained = ismember(CurrentKey,AcceptedKey,'rows');
    BInfo.Age(Active(Retained)) = 0;
end

function Flag = IsFeasibleSolution(Solution)
    Cons = double(Solution.cons);
    Flag = isempty(Cons) || all(Cons <= 0);
end

function ResultPopulation = BuildExternalResultPopulation(Population,PreviousResult)
    if nargin < 2
        PreviousResult = Population([]);
    end
    ResultPopulation = Population.best;
    if isempty(ResultPopulation)
        ResultPopulation = PreviousResult;
    end
end

function [Flag,CandKey,CandSector] = IsBracketTighter( ...
    OldF,OldI,CandF,CandI,Problem,PopulationC,W)

    [OldKey,CandKey,CandSector] = CompareBoundaryPairObjectiveKeys( ...
        OldF,OldI,CandF,CandI,PopulationC,W);
    Flag = false;
    if ~IsFeasibleSolution(CandF) || IsFeasibleSolution(CandI) || any(~isfinite(CandKey))
        return;
    end

    OldDecGap = DecisionDistanceNormalized( ...
        double(OldF.decs),double(OldI.decs),Problem.lower,Problem.upper);
    NewDecGap = DecisionDistanceNormalized( ...
        double(CandF.decs),double(CandI.decs),Problem.lower,Problem.upper);
    if ~isfinite(NewDecGap)
        return;
    end
    if ~isfinite(OldDecGap)
        Flag = true;
        return;
    end

    Tol = 1e-12;
    Flag = NewDecGap < OldDecGap - Tol || ...
        (NewDecGap <= OldDecGap + Tol && IsObjPairGapNoWorse(CandKey,OldKey));
end

function [OldKey,CandKey,CandSector] = CompareBoundaryPairObjectiveKeys( ...
    OldF,OldI,CandF,CandI,PopulationC,W)

    PairObj = [double(OldF.objs);double(OldI.objs);double(CandF.objs);double(CandI.objs)];
    PopCObj = zeros(0,size(PairObj,2));
    PopCLabel = zeros(0,1);
    if ~isempty(PopulationC)
        PopCObj = double(PopulationC.objs);
        PopCLabel = double(all(PopulationC.cons<=0,2));
    end
    AllObj = [PairObj;PopCObj];
    [AllObjN,~,~] = NormalizeObjectives(AllObj);
    PairObjN = AllObjN(1:4,:);
    PopCObjN = AllObjN(5:end,:);

    if isempty(W)
        Neighbors = cell(1,1);
    else
        Neighbors = BuildSectorNeighbors(W,min(3,max(size(W,1)-1,0)));
    end
    FallbackObjN = PairObjN([1 3],:);
    FallbackLabel = [double(IsFeasibleSolution(OldF));double(IsFeasibleSolution(CandF))];
    BestFeasibleScalar = ComputeBestFeasibleScalar( ...
        PopCObjN,PopCLabel,W,Neighbors,FallbackObjN,FallbackLabel);

    [OldKey,~] = EvaluateBoundaryPairKeyFromNorm( ...
        PairObjN(1,:),PairObjN(2,:),W,BestFeasibleScalar);
    [CandKey,CandSector] = EvaluateBoundaryPairKeyFromNorm( ...
        PairObjN(3,:),PairObjN(4,:),W,BestFeasibleScalar);
end

function [Key,Sector] = EvaluateBoundaryPairKeyFromNorm(FObjN,IObjN,W,BestFeasibleScalar)
    MidObj = 0.5*(FObjN + IObjN);
    if isempty(W)
        Sector = 1;
    else
        Sector = AssociateSectorsLocal(MidObj,W,zeros(1,size(MidObj,2)));
    end
    PairGap = sqrt(sum((FObjN - IObjN).^2,2));
    Scalar = ComputeSectorScalar(MidObj,W,zeros(1,size(MidObj,2)),Sector);
    s = min(max(Sector,1),numel(BestFeasibleScalar));
    FrontGap = max(0,Scalar - BestFeasibleScalar(s));
    Key = [FrontGap,PairGap];
end

function Flag = IsObjPairGapNoWorse(CandKey,OldKey)
    if any(~isfinite(CandKey))
        Flag = false;
    elseif any(~isfinite(OldKey))
        Flag = true;
    else
        Flag = CandKey(2) <= OldKey(2) + 1e-12;
    end
end

function Target = InitTargetArchive(Prototype,NBPair)
    Target = struct( ...
        'Population',Prototype([]), ...
        'Sector',zeros(0,1), ...
        'FrontGap',zeros(0,1), ...
        'PairGap',zeros(0,1), ...
        'FeasibleSource',zeros(0,1), ...
        'InfeasibleSource',zeros(0,1), ...
        'AvailableSectorCount',0, ...
        'Quota',zeros(NBPair,1), ...
        'Priority',zeros(NBPair,1));
end

function Target = BuildTargetBoundaryArchive(A,Source,PopulationC,W,NBPair,FeRatio)
    Target = InitTargetArchive(A,NBPair);
    if isempty(A) || NBPair <= 0
        return;
    end
    Source = MatchLength(Source,numel(A),BoundarySourceCode('b'));

    Obj = double(A.objs);
    Label = double(all(A.cons<=0,2));
    if ~any(Label == 1) || ~any(Label == 0)
        return;
    end

    [ObjN,zmin,zmax] = NormalizeObjectives(Obj);
    K = max(1,size(W,1));
    if isempty(W)
        SectorAll = ones(size(ObjN,1),1);
        Neighbors = cell(1,1);
    else
        SectorAll = AssociateSectorsLocal(ObjN,W,zeros(1,size(ObjN,2)));
        Neighbors = BuildSectorNeighbors(W,min(3,max(K-1,0)));
    end

    PopCObjN = zeros(0,size(ObjN,2));
    PopCLabel = zeros(0,1);
    if ~isempty(PopulationC)
        PopCObjN = NormalizeObjectivesWithBounds(double(PopulationC.objs),zmin,zmax);
        PopCLabel = double(all(PopulationC.cons<=0,2));
    end
    BestFeasibleScalar = ComputeBestFeasibleScalar(PopCObjN,PopCLabel,W,Neighbors,ObjN,Label);

    LocalSectorMask = false(numel(Label),K);
    for s = 1 : K
        LocalSectors = ResolveLocalSectorSet(s,Neighbors,K);
        LocalSectorMask(:,s) = ismember(SectorAll,LocalSectors);
    end

    FeasibleMask = Label == 1;
    InfeasibleMask = Label == 0;
    PairLists = cell(K,1);
    for s = 1 : K
        LocalFeasible = find(FeasibleMask & LocalSectorMask(:,s));
        LocalInfeasible = find(InfeasibleMask & LocalSectorMask(:,s));
        LocalFeasible = SelectBoundaryShellByScalar(ObjN,LocalFeasible,s,W);
        LocalInfeasible = SelectBoundaryShellByScalar(ObjN,LocalInfeasible,s,W);
        PairLists{s} = BuildSectorCandidatePairsObjective( ...
            ObjN,Label,LocalFeasible,LocalInfeasible,s,W,BestFeasibleScalar);
    end

    Priority = ComputeSectorPriority(PairLists,K);
    Quota = AllocateSectorQuota(PairLists,Priority,NBPair,FeRatio);
    Target = SelectTargetPairsFromQuota(A,Source,PairLists,Quota,Priority,NBPair);
    Target.AvailableSectorCount = sum(cellfun(@(P)~isempty(P.FeasibleIndex),PairLists));
    Target.Quota = MatchLength(Quota,NBPair,0);
    Target.Priority = MatchLength(Priority,NBPair,0);
end

function Index = SelectBoundaryShellByScalar(ObjN,Index,Sector,W)
    Index = Index(:);
    if numel(Index) <= 2
        return;
    end
    ShellSize = min(numel(Index),max(2,ceil(sqrt(numel(Index)))));
    Scalar = ComputeSectorScalar(ObjN(Index,:),W,zeros(1,size(ObjN,2)),Sector*ones(numel(Index),1));
    [~,ord] = sortrows([Scalar(:),Index(:)],[1 2]);
    Index = Index(ord(1:ShellSize));
end

function PairList = InitPairList()
    PairList = struct( ...
        'FeasibleIndex',zeros(0,1), ...
        'InfeasibleIndex',zeros(0,1), ...
        'Sector',zeros(0,1), ...
        'PairGap',zeros(0,1), ...
        'FrontGap',zeros(0,1));
end

function PairList = BuildSectorCandidatePairsObjective( ...
    ObjN,~,LocalFeasible,LocalInfeasible,Sector,W,BestFeasibleScalar)
    PairList = InitPairList();
    if isempty(LocalFeasible) || isempty(LocalInfeasible)
        return;
    end

    [I,J] = ndgrid(LocalFeasible(:),LocalInfeasible(:));
    I = I(:);
    J = J(:);
    Delta = ObjN(I,:) - ObjN(J,:);
    PairGap = sqrt(sum(Delta.^2,2));
    Valid = isfinite(PairGap);
    I = I(Valid);
    J = J(Valid);
    PairGap = PairGap(Valid);
    if isempty(PairGap)
        return;
    end

    [~,ord] = sort(PairGap,'ascend');
    Used = false(size(ObjN,1),1);
    KeepI = zeros(numel(ord),1);
    KeepJ = zeros(numel(ord),1);
    KeepGap = zeros(numel(ord),1);
    Count = 0;
    for k = 1 : numel(ord)
        p = ord(k);
        if Used(I(p)) || Used(J(p))
            continue;
        end
        Count = Count + 1;
        KeepI(Count) = I(p);
        KeepJ(Count) = J(p);
        KeepGap(Count) = PairGap(p);
        Used(I(p)) = true;
        Used(J(p)) = true;
    end
    if Count <= 0
        return;
    end
    KeepI = KeepI(1:Count);
    KeepJ = KeepJ(1:Count);
    KeepGap = KeepGap(1:Count);

    MidObj = 0.5*(ObjN(KeepI,:) + ObjN(KeepJ,:));
    MidScalar = ComputeSectorScalar(MidObj,W,zeros(1,size(ObjN,2)),Sector*ones(Count,1));
    FrontGap = max(0,MidScalar - BestFeasibleScalar(min(Sector,numel(BestFeasibleScalar))));

    Key = [KeepGap(:),FrontGap(:),(1:Count)'];
    [~,ord] = sortrows(Key,[1 2 3]);
    PairList.FeasibleIndex = KeepI(ord);
    PairList.InfeasibleIndex = KeepJ(ord);
    PairList.Sector = Sector*ones(Count,1);
    PairList.PairGap = KeepGap(ord);
    PairList.FrontGap = FrontGap(ord);
end

function Priority = ComputeSectorPriority(PairLists,K)
    Priority = zeros(K,1);
    for s = 1 : K
        Count = numel(PairLists{s}.FeasibleIndex);
        if Count > 0
            Priority(s) = log(1+Count)/(eps + max(PairLists{s}.FrontGap(1),0));
        end
    end
end

function Quota = AllocateSectorQuota(PairLists,Priority,NBPair,FeRatio)
    K = numel(PairLists);
    Quota = zeros(K,1);
    Available = find(cellfun(@(P)~isempty(P.FeasibleIndex),PairLists));
    if isempty(Available) || NBPair <= 0
        return;
    end
    if nargin < 4 || isempty(FeRatio)
        FeRatio = 0;
    end

    if numel(Available) > NBPair
        [~,ord] = sort(Priority(Available),'descend');
        Available = Available(ord(1:NBPair));
    end
    [~,ord] = sort(Priority(Available),'descend');
    Available = Available(ord);
    CoverCount = min(numel(Available),ceil((1-min(max(FeRatio,0),1))*numel(Available)));
    if CoverCount > 0
        Quota(Available(1:CoverCount)) = 1;
    end
    Remain = NBPair - sum(Quota);
    while Remain > 0
        Utility = -inf(K,1);
        for s = Available(:)'
            if Quota(s) < numel(PairLists{s}.FeasibleIndex)
                Utility(s) = Priority(s)/(Quota(s)+1);
            end
        end
        [BestUtility,BestSector] = max(Utility);
        if ~isfinite(BestUtility)
            break;
        end
        Quota(BestSector) = Quota(BestSector) + 1;
        Remain = Remain - 1;
    end
end

function Target = SelectTargetPairsFromQuota(A,Source,PairLists,Quota,Priority,NBPair)
    Target = InitTargetArchive(A,NBPair);
    if isempty(A) || isempty(PairLists)
        return;
    end
    Source = MatchLength(Source,numel(A),BoundarySourceCode('b'));

    K = numel(PairLists);
    [~,SectorOrder] = sort(Priority(:),'descend');
    Used = false(numel(A),1);
    PairF = zeros(NBPair,1);
    PairI = zeros(NBPair,1);
    PairS = zeros(NBPair,1);
    PairGap = zeros(NBPair,1);
    FrontGap = zeros(NBPair,1);
    PairFSource = zeros(NBPair,1);
    PairISource = zeros(NBPair,1);
    Count = 0;
    for ss = 1 : numel(SectorOrder)
        s = SectorOrder(ss);
        if s < 1 || s > K || Quota(s) <= 0
            continue;
        end
        Taken = 0;
        List = PairLists{s};
        for p = 1 : numel(List.FeasibleIndex)
            if Taken >= Quota(s) || Count >= NBPair
                break;
            end
            f = List.FeasibleIndex(p);
            i = List.InfeasibleIndex(p);
            if Used(f) || Used(i)
                continue;
            end
            Count = Count + 1;
            Taken = Taken + 1;
            Used(f) = true;
            Used(i) = true;
            PairF(Count) = f;
            PairI(Count) = i;
            PairS(Count) = s;
            PairGap(Count) = List.PairGap(p);
            FrontGap(Count) = List.FrontGap(p);
            PairFSource(Count) = Source(f);
            PairISource(Count) = Source(i);
        end
    end
    if Count <= 0
        return;
    end

    PairF = PairF(1:Count);
    PairI = PairI(1:Count);
    PairS = PairS(1:Count);
    PairGap = PairGap(1:Count);
    FrontGap = FrontGap(1:Count);
    PairFSource = PairFSource(1:Count);
    PairISource = PairISource(1:Count);
    Key = [PairGap,FrontGap,PairS,(1:Count)'];
    [~,ord] = sortrows(Key,[1 2 3 4]);
    PairF = PairF(ord);
    PairI = PairI(ord);
    PairS = PairS(ord);
    PairGap = PairGap(ord);
    FrontGap = FrontGap(ord);
    PairFSource = PairFSource(ord);
    PairISource = PairISource(ord);

    Target.Population = flattenPairPopulation(A,PairF,PairI);
    Target.Sector = PairS;
    Target.PairGap = PairGap;
    Target.FrontGap = FrontGap;
    Target.FeasibleSource = PairFSource;
    Target.InfeasibleSource = PairISource;
    Target.Quota = MatchLength(Quota,NBPair,0);
    Target.Priority = MatchLength(Priority,NBPair,0);
end

function Population = flattenPairPopulation(A,FeasibleIndex,InfeasibleIndex)
    Rows = reshape([FeasibleIndex(:),InfeasibleIndex(:)]',1,[]);
    Population = A(Rows);
end

function [B,BInfo,Diag] = BlendBoundaryArchive(B,BInfo,Target,etaB,NBPair)
    Diag = struct('added',0,'replaced',0,'dropped',0,'kept',0,'budget',ceil(max(etaB,0)*NBPair));
    Current = PopulationToPairStruct(B,BInfo);
    TargetPairs = TargetToPairStruct(Target);
    if TargetPairs.Count <= 0
        Current.Age = Current.Age + 1;
        [B,BInfo] = PairStructToPopulation(Current,B,NBPair);
        Diag.kept = Current.Count;
        return;
    end

    Budget = ceil(max(etaB,0)*NBPair);
    if Current.Count <= 0
        Take = 1:min([NBPair,TargetPairs.Count,Budget]);
        if isempty(Take)
            Take = zeros(0,1);
        end
        if isempty(Take)
            [B,BInfo] = PairStructToPopulation(Current,B,NBPair);
            return;
        end
        NewPairs = SlicePairStruct(TargetPairs,Take);
        NewPairs.Age(:) = 0;
        [B,BInfo] = PairStructToPopulation(NewPairs,Target.Population,NBPair);
        Diag.added = NewPairs.Count;
        return;
    end

    Current.Age = Current.Age + 1;
    CurrentKey = PairKeyMatrix(Current.Population);
    TargetKey = PairKeyMatrix(TargetPairs.Population);
    [InTarget,TargetLoc] = ismember(CurrentKey,TargetKey,'rows');
    [InCurrent,~] = ismember(TargetKey,CurrentKey,'rows');

    for c = find(InTarget(:))'
        t = TargetLoc(c);
        Current.Sector(c) = TargetPairs.Sector(t);
        Current.FrontGap(c) = TargetPairs.FrontGap(t);
        Current.PairGap(c) = TargetPairs.PairGap(t);
        Current.FeasibleSource(c) = TargetPairs.FeasibleSource(t);
        Current.InfeasibleSource(c) = TargetPairs.InfeasibleSource(t);
    end

    AddIdx = find(~InCurrent);
    if ~isempty(AddIdx)
        [~,ord] = sortrows([TargetPairs.PairGap(AddIdx),TargetPairs.FrontGap(AddIdx),AddIdx(:)],[1 2 3]);
        AddIdx = AddIdx(ord);
    end
    DropIdx = find(~InTarget);
    if ~isempty(DropIdx)
        [~,ord] = sortrows([Current.PairGap(DropIdx),Current.FrontGap(DropIdx),Current.Age(DropIdx),DropIdx(:)],[-1 -2 -3 4]);
        DropIdx = DropIdx(ord);
    end

    NewPairs = Current;
    FillCount = min([NBPair-NewPairs.Count,numel(AddIdx),Budget]);
    if FillCount > 0
        NewPairs = AppendPairStruct(NewPairs,SlicePairStruct(TargetPairs,AddIdx(1:FillCount)));
        NewPairs.Age(end-FillCount+1:end) = 0;
        Diag.added = FillCount;
    end
    Budget = Budget - FillCount;
    AddPtr = FillCount + 1;
    ReplaceCount = min([Budget,numel(AddIdx)-FillCount,numel(DropIdx)]);
    for k = 1 : ReplaceCount
        d = DropIdx(k);
        a = AddIdx(AddPtr);
        NewPairs = ReplacePairStruct(NewPairs,d,TargetPairs,a);
        AddPtr = AddPtr + 1;
    end
    Diag.replaced = ReplaceCount;
    Diag.dropped = ReplaceCount;

    if NewPairs.Count > NBPair
        [~,ord] = sortrows([NewPairs.PairGap,NewPairs.FrontGap,NewPairs.Age,(1:NewPairs.Count)'],[1 2 3 4]);
        NewPairs = SlicePairStruct(NewPairs,sort(ord(1:NBPair)));
    end
    Diag.kept = max(0,NewPairs.Count - Diag.added - Diag.replaced);
    [B,BInfo] = PairStructToPopulation(NewPairs,B,NBPair);
end

function Pairs = InitPairStruct(Prototype)
    Pairs = struct( ...
        'Population',Prototype([]), ...
        'Sector',zeros(0,1), ...
        'FrontGap',zeros(0,1), ...
        'PairGap',zeros(0,1), ...
        'FeasibleSource',zeros(0,1), ...
        'InfeasibleSource',zeros(0,1), ...
        'Age',zeros(0,1), ...
        'Count',0);
end

function Pairs = PopulationToPairStruct(B,BInfo)
    Pairs = InitPairStruct(B);
    Count = min([floor(numel(B)/2),BInfo.PairCount,numel(BInfo.Active)]);
    if Count <= 0
        return;
    end
    Active = find(BInfo.Active(1:Count));
    if isempty(Active)
        Active = (1:Count)';
    end
    KeepRows = reshape([2*Active(:)-1,2*Active(:)]',1,[]);
    Pairs.Population = B(KeepRows);
    Pairs.Sector = BInfo.Sector(Active);
    Pairs.FrontGap = BInfo.FrontGap(Active);
    Pairs.PairGap = BInfo.PairGap(Active);
    [FeasibleSource,InfeasibleSource] = BoundaryPairSources(BInfo,Count);
    Pairs.FeasibleSource = FeasibleSource(Active);
    Pairs.InfeasibleSource = InfeasibleSource(Active);
    Pairs.Age = BInfo.Age(Active);
    Pairs.Count = numel(Active);
end

function Pairs = TargetToPairStruct(Target)
    Pairs = InitPairStruct(Target.Population);
    Count = min([floor(numel(Target.Population)/2),numel(Target.Sector), ...
        numel(Target.FrontGap),numel(Target.PairGap), ...
        numel(Target.FeasibleSource),numel(Target.InfeasibleSource)]);
    if Count <= 0
        return;
    end
    Pairs.Population = Target.Population(1:2*Count);
    Pairs.Sector = Target.Sector(1:Count);
    Pairs.FrontGap = Target.FrontGap(1:Count);
    Pairs.PairGap = Target.PairGap(1:Count);
    Pairs.FeasibleSource = Target.FeasibleSource(1:Count);
    Pairs.InfeasibleSource = Target.InfeasibleSource(1:Count);
    Pairs.Age = zeros(Count,1);
    Pairs.Count = Count;
end

function Key = PairKeyMatrix(Population)
    if isempty(Population)
        Key = zeros(0,0);
        return;
    end
    Count = floor(numel(Population)/2);
    Dec = double(Population.decs);
    D = size(Dec,2);
    Key = reshape(Dec(1:2*Count,:)',2*D,Count)';
end

function Pairs = SlicePairStruct(Pairs,Keep)
    Keep = Keep(:);
    if isempty(Keep)
        Pairs = InitPairStruct(Pairs.Population);
        return;
    end
    Rows = reshape([2*Keep-1,2*Keep]',1,[]);
    Pairs.Population = Pairs.Population(Rows);
    Pairs.Sector = Pairs.Sector(Keep);
    Pairs.FrontGap = Pairs.FrontGap(Keep);
    Pairs.PairGap = Pairs.PairGap(Keep);
    Pairs.FeasibleSource = Pairs.FeasibleSource(Keep);
    Pairs.InfeasibleSource = Pairs.InfeasibleSource(Keep);
    Pairs.Age = Pairs.Age(Keep);
    Pairs.Count = numel(Keep);
end

function Pairs = AppendPairStruct(Pairs,Add)
    if Add.Count <= 0
        return;
    end
    if Pairs.Count <= 0
        Pairs = Add;
        return;
    end
    Pairs.Population = [Pairs.Population,Add.Population];
    Pairs.Sector = [Pairs.Sector;Add.Sector];
    Pairs.FrontGap = [Pairs.FrontGap;Add.FrontGap];
    Pairs.PairGap = [Pairs.PairGap;Add.PairGap];
    Pairs.FeasibleSource = [Pairs.FeasibleSource;Add.FeasibleSource];
    Pairs.InfeasibleSource = [Pairs.InfeasibleSource;Add.InfeasibleSource];
    Pairs.Age = [Pairs.Age;Add.Age];
    Pairs.Count = Pairs.Count + Add.Count;
end

function Pairs = ReplacePairStruct(Pairs,DropIndex,Source,SourceIndex)
    Rows = [2*DropIndex-1,2*DropIndex];
    SourceRows = [2*SourceIndex-1,2*SourceIndex];
    Pairs.Population(Rows) = Source.Population(SourceRows);
    Pairs.Sector(DropIndex) = Source.Sector(SourceIndex);
    Pairs.FrontGap(DropIndex) = Source.FrontGap(SourceIndex);
    Pairs.PairGap(DropIndex) = Source.PairGap(SourceIndex);
    Pairs.FeasibleSource(DropIndex) = Source.FeasibleSource(SourceIndex);
    Pairs.InfeasibleSource(DropIndex) = Source.InfeasibleSource(SourceIndex);
    Pairs.Age(DropIndex) = 0;
end

function [B,BInfo] = PairStructToPopulation(Pairs,Prototype,NBPair)
    if Pairs.Count <= 0
        B = Prototype([]);
        BInfo = InitBoundaryArchiveInfo(NBPair);
        return;
    end
    B = Pairs.Population(1:2*Pairs.Count);
    BInfo = InitBoundaryArchiveInfo(NBPair);
    Count = min(Pairs.Count,NBPair);
    BInfo.PairCount = Count;
    BInfo.Sector(1:Count) = Pairs.Sector(1:Count);
    BInfo.FrontGap(1:Count) = Pairs.FrontGap(1:Count);
    BInfo.PairGap(1:Count) = Pairs.PairGap(1:Count);
    BInfo.FeasibleSource(1:Count) = Pairs.FeasibleSource(1:Count);
    BInfo.InfeasibleSource(1:Count) = Pairs.InfeasibleSource(1:Count);
    BInfo.Age(1:Count) = Pairs.Age(1:Count);
    BInfo.Active(1:Count) = true;
end

%% Boundary metadata and environmental selection

function [Population,Diag] = EnvironmentalSelectionC_ObjectBoundary(Population,N,W,Model,B,BInfo,PreviousC)
    Diag = InitInfeasibleSelectionDiag();
    Population = KeepUniquePopulation(Population);
    if isempty(Population)
        return;
    end

    Feasible = FilterFeasiblePopulation(Population);
    Infeasible = Population(any(Population.cons>0,2));
    Diag.carry_pool_size = numel(Infeasible);

    Carry = 0;
    EnoughB = BInfo.PairCount >= MinReadyBoundaryPairs(Population,W);
    if EnoughB && ~isempty(Infeasible)
        Carry = min(max(1,round(0.01*N)),numel(Infeasible));
    end

    FeasibleNeed = max(N - Carry,0);
    Next = ObjectiveSelectionWithLastSectorTruncation( ...
        Feasible,min(FeasibleNeed,numel(Feasible)),W);

    if Carry > 0
        [CarryPick,CarryDiag] = SelectInfeasibleByBoundaryMeta( ...
            Infeasible,Carry,W,Model,B,BInfo,PreviousC);
        Diag.carry_applicable_count = CarryDiag.applicable_count;
        Diag.carry_selected = numel(CarryPick);
        Diag.carry_pool_mean_prob = CarryDiag.pool_mean_prob;
        Diag.carry_selected_mean_prob = CarryDiag.selected_mean_prob;
        Diag.carry_prob_gain = CarryDiag.prob_gain;
        Next = [Next,CarryPick];
        Infeasible = RemovePopulationByDecision(Infeasible,CarryPick);
    end

    Need = N - numel(Next);
    if Need > 0 && ~isempty(Infeasible)
        [Pick,Diag] = SelectInfeasibleByBoundaryMeta( ...
            Infeasible,Need,W,Model,B,BInfo,PreviousC,Diag);
        Next = [Next,Pick];
    end

    if numel(Next) < N
        Rest = RemovePopulationByDecision(Population,Next);
        Rest = ObjectiveSelectionWithLastSectorTruncation(Rest,min(N-numel(Next),numel(Rest)),W);
        Next = [Next,Rest];
    end
    Population = PadPopulation(Next,N);
end

function [Pick,Diag] = SelectInfeasibleByBoundaryMeta(Population,N,W,Model,B,BInfo,PopulationC,Diag)
    if nargin < 8 || isempty(Diag)
        Diag = InitInfeasibleSelectionDiag();
    end
    Pick = Population([]);
    if isempty(Population) || N <= 0
        return;
    end

    Population = Population(any(Population.cons>0,2));
    if isempty(Population)
        return;
    end
    N = min(N,numel(Population));
    Meta = BuildBoundaryMetaFromB(Population,W,Model,B,BInfo,PopulationC);
    Applicable = FindModelApplicableInfeasible(Meta);
    Diag.pool_size = numel(Population);
    Diag.applicable_count = sum(Applicable);
    Diag.pool_mean_prob = MeanOrNaN(Meta.prob);
    if ~any(Applicable)
        return;
    end

    if isempty(W)
        Order = SortBySelectionKey(Meta,find(Applicable));
        PickIndex = Order(1:min(N,numel(Order)));
    else
        Ranked = cell(size(W,1),1);
        for s = 1 : size(W,1)
            idx = find(Meta.sector == s & Applicable);
            if ~isempty(idx)
                Ranked{s} = SortBySelectionKey(Meta,idx);
            end
        end
        Order = SectorRoundRobinPick(Ranked,N);
        if isempty(Order)
            Global = SortBySelectionKey(Meta,find(Applicable));
            PickIndex = Global(1:min(N,numel(Global)));
        else
            PickIndex = Order(:,2);
        end
    end
    Pick = Population(PickIndex);
    Diag.selected = numel(Pick);
    Diag.selected_mean_prob = MeanOrNaN(Meta.prob(PickIndex));
    Diag.prob_gain = Diag.selected_mean_prob - Diag.pool_mean_prob;
end

function Count = MinReadyBoundaryPairs(Population,W)
    if ~isempty(W)
        M = size(W,2);
    elseif ~isempty(Population)
        M = size(Population.objs,2);
    else
        M = 1;
    end
    Count = max(4,2*M);
end

function Meta = BuildBoundaryMetaFromB(Candidates,W,Model,B,BInfo,PopulationC)
    N = numel(Candidates);
    Meta = InitBoundaryMeta(N);
    if isempty(Candidates)
        return;
    end
    [CandObjN,BObjN,PopCObjN] = NormalizeCandidateBObjects(Candidates,B,PopulationC);
    if isempty(W)
        Sector = ones(N,1);
        Neighbors = cell(1,1);
    else
        Sector = AssociateSectorsLocal(CandObjN,W,zeros(1,size(CandObjN,2)));
        Neighbors = BuildSectorNeighbors(W,min(3,max(size(W,1)-1,0)));
    end
    Prob = PredictBoundaryMLP(Model,Candidates.decs);
    Feasible = double(all(Candidates.cons<=0,2));
    SupportDist = ComputeSupportDistObjToB(CandObjN,Sector,BObjN,BInfo,Neighbors);
    PopCLabel = zeros(0,1);
    if ~isempty(PopulationC)
        PopCLabel = double(all(PopulationC.cons<=0,2));
    end
    FrontGap = ComputeBoundaryFrontGapObj(CandObjN,Sector,PopCObjN,PopCLabel,W,Neighbors);

    Meta.sector = Sector;
    Meta.feasible = Feasible;
    Meta.prob = Prob;
    Meta.supportDistObjToB = SupportDist;
    Meta.frontGap = FrontGap;
end

function [CandObjN,BObjN,PopCObjN] = NormalizeCandidateBObjects(Candidates,B,PopulationC)
    CandObj = double(Candidates.objs);
    M = size(CandObj,2);
    BObj = zeros(0,M);
    PopCObj = zeros(0,M);
    if ~isempty(B)
        BObj = double(B.objs);
    end
    if ~isempty(PopulationC)
        PopCObj = double(PopulationC.objs);
    end
    AllObj = [CandObj;BObj;PopCObj];
    [AllObjN,~,~] = NormalizeObjectives(AllObj);
    nC = size(CandObj,1);
    nB = size(BObj,1);
    CandObjN = AllObjN(1:nC,:);
    BObjN = AllObjN(nC+1:nC+nB,:);
    PopCObjN = AllObjN(nC+nB+1:end,:);
end

function Dist = ComputeSupportDistObjToB(CandObjN,Sector,BObjN,BInfo,Neighbors)
    Dist = inf(size(CandObjN,1),1);
    PairCount = min([BInfo.PairCount,floor(size(BObjN,1)/2)]);
    if PairCount <= 0
        return;
    end
    Active = find(BInfo.Active(1:PairCount));
    if isempty(Active)
        return;
    end
    PairSector = BInfo.Sector(Active);
    SegmentA = BObjN(2*Active-1,:);
    SegmentB = BObjN(2*Active,:);
    K = max([numel(Neighbors),max(BInfo.Sector(Active)),1]);
    for i = 1 : size(CandObjN,1)
        Local = ResolveLocalSectorSet(Sector(i),Neighbors,K);
        Mask = ismember(PairSector,Local);
        if any(Mask)
            Dist(i) = PointToSegmentsDistanceObj(CandObjN(i,:),SegmentA(Mask,:),SegmentB(Mask,:));
        end
    end
end

function d = PointToSegmentsDistanceObj(p,A,B)
    if isempty(A)
        d = inf;
        return;
    end
    V = B - A;
    Denom = sum(V.^2,2);
    SafeDenom = max(Denom,1e-12);
    T = sum(bsxfun(@minus,p,A).*V,2)./SafeDenom;
    T = min(max(T,0),1);
    Proj = A + bsxfun(@times,T,V);
    Dist = sqrt(sum(bsxfun(@minus,p,Proj).^2,2));
    Degenerate = Denom <= 1e-12;
    if any(Degenerate)
        Dist(Degenerate) = sqrt(sum(bsxfun(@minus,p,A(Degenerate,:)).^2,2));
    end
    d = min(Dist);
end

function FrontGap = ComputeBoundaryFrontGapObj(CandObjN,Sector,PopCObjN,PopCLabel,W,Neighbors)
    if isempty(CandObjN)
        FrontGap = zeros(0,1);
        return;
    end
    Best = ComputeBestFeasibleScalar(PopCObjN,PopCLabel,W,Neighbors,PopCObjN,PopCLabel);
    Scalar = ComputeSectorScalar(CandObjN,W,zeros(1,size(CandObjN,2)),Sector);
    FrontGap = zeros(size(CandObjN,1),1);
    for i = 1 : numel(FrontGap)
        s = min(max(Sector(i),1),numel(Best));
        FrontGap(i) = max(0,Scalar(i) - Best(s));
    end
end

function Applicable = FindModelApplicableInfeasible(Meta)
    Applicable = (Meta.feasible == 0);
end

function Meta = InitBoundaryMeta(Count)
    Meta = struct( ...
        'sector',zeros(Count,1), ...
        'feasible',zeros(Count,1), ...
        'prob',0.5*ones(Count,1), ...
        'supportDistObjToB',inf(Count,1), ...
        'frontGap',zeros(Count,1));
end

function Order = SortBySelectionKey(Meta,idx)
    idx = idx(:);
    if isempty(idx)
        Order = idx;
        return;
    end
    Key = [Meta.frontGap(idx),-Meta.prob(idx),Meta.supportDistObjToB(idx),idx];
    [~,ord] = sortrows(Key,1:size(Key,2));
    Order = idx(ord);
end

function Pick = SelectBoundaryBandTrainingPoints(Population,B,BInfo,PopulationC,W,MaxPick)
    Pick = Population([]);
    if isempty(Population) || isempty(B) || BInfo.PairCount <= 0 || MaxPick <= 0
        return;
    end
    Population = KeepUniquePopulation(Population);
    Meta = BuildBoundaryMetaFromB(Population,W,[],B,BInfo,PopulationC);
    Candidate = find(isfinite(Meta.supportDistObjToB) & isfinite(Meta.frontGap));
    if isempty(Candidate)
        return;
    end
    Key = [Meta.supportDistObjToB(Candidate),Meta.frontGap(Candidate),Candidate(:)];
    [~,ord] = sortrows(Key,[1 2 3]);
    Candidate = Candidate(ord(1:min(MaxPick,numel(ord))));
    Pick = Population(Candidate);
end

%% T-buffer MLP

function [Model,Diag] = UpdateBoundaryMLPPeriodically( ...
    Model,B,BInfo,TrainBuffer,hidden,epoch,lr,Generation,Gstart,Tretrain,Problem)

    Dataset = BuildTrainingBufferT(B,BInfo,TrainBuffer);
    Balanced = BalanceTrainingDataset(Dataset);
    Diag = InitMLPDiag(Generation,Problem.FE,Model,Dataset,BInfo);
    Diag.due = Generation >= Gstart && mod(Generation-Gstart,Tretrain) == 0;
    MinPairs = max(4,2*Problem.M);
    Diag.can_train = Diag.due && Dataset.pair_count >= MinPairs && ...
        Balanced.pos_count > 0 && Balanced.neg_count > 0;
    if ~Diag.due
        Diag.skip_reason = "not_due";
        return;
    end
    if ~Diag.can_train
        Diag.skip_reason = "insufficient_b";
        return;
    end

    if isempty(Model)
        Model = TrainBoundaryMLPColdStart(Balanced,Problem,hidden,epoch,lr);
    else
        WarmEpoch = max(1,round(0.25*epoch));
        WarmLR = max(1e-4,0.2*lr);
        Model = ContinueTrainBoundaryMLP(Model,Balanced,Problem,hidden,WarmEpoch,WarmLR);
    end
    Diag.trained = true;
    Diag.acc_after = EvaluateBinaryAccuracy(Model,Dataset.Dec,Dataset.Label);
    Diag.degraded = IsAccuracyDegraded(Diag.acc_before,Diag.acc_after);
    Diag.skip_reason = "trained";
end

function TrainBuffer = InitTrainingBuffer(D)
    TrainBuffer = struct( ...
        'Dec',zeros(0,D), ...
        'Label',zeros(0,1), ...
        'Source',zeros(0,1), ...
        'Time',zeros(0,1));
end

function TrainBuffer = UpdateTrainingBufferT(TrainBuffer,Refinement,BoundaryBand,Generation,MaxTrain)
    if nargin < 5 || isempty(MaxTrain)
        MaxTrain = inf;
    end
    TrainBuffer = AppendTrainingSamples(TrainBuffer,BoundaryBand,3,Generation);
    TrainBuffer = AppendTrainingSamples(TrainBuffer,Refinement,2,Generation);
    TrainBuffer = KeepLatestTrainingRows(TrainBuffer);
    TrainBuffer = TrimTrainingBufferNonB(TrainBuffer,MaxTrain);
end

function TrainBuffer = AppendTrainingSamples(TrainBuffer,Population,Source,Generation)
    if isempty(Population)
        return;
    end
    Count = numel(Population);
    TrainBuffer.Dec = [TrainBuffer.Dec;double(Population.decs)];
    TrainBuffer.Label = [TrainBuffer.Label;double(all(Population.cons<=0,2))];
    TrainBuffer.Source = [TrainBuffer.Source;Source*ones(Count,1)];
    TrainBuffer.Time = [TrainBuffer.Time;Generation*ones(Count,1)];
end

function TrainBuffer = KeepLatestTrainingRows(TrainBuffer)
    Keep = KeepLatestDecisionRowsLocal(TrainBuffer.Dec);
    TrainBuffer.Dec = TrainBuffer.Dec(Keep,:);
    TrainBuffer.Label = TrainBuffer.Label(Keep);
    TrainBuffer.Source = TrainBuffer.Source(Keep);
    TrainBuffer.Time = TrainBuffer.Time(Keep);
end

function TrainBuffer = TrimTrainingBufferNonB(TrainBuffer,MaxTrain)
    if isempty(TrainBuffer.Dec) || ~isfinite(MaxTrain)
        return;
    end
    Keep = SelectNewestBySource(TrainBuffer,2,MaxTrain);
    Remain = max(0,MaxTrain-numel(Keep));
    Keep = [Keep;SelectNewestBySource(TrainBuffer,3,Remain)];
    Keep = sort(Keep);
    TrainBuffer.Dec = TrainBuffer.Dec(Keep,:);
    TrainBuffer.Label = TrainBuffer.Label(Keep);
    TrainBuffer.Source = TrainBuffer.Source(Keep);
    TrainBuffer.Time = TrainBuffer.Time(Keep);
end

function Keep = SelectNewestBySource(TrainBuffer,Source,MaxCount)
    Keep = zeros(0,1);
    if MaxCount <= 0
        return;
    end
    Candidate = find(TrainBuffer.Source == Source);
    if isempty(Candidate)
        return;
    end
    [~,ord] = sortrows([-TrainBuffer.Time(Candidate),Candidate(:)],[1 2]);
    Keep = Candidate(ord(1:min(MaxCount,numel(ord))));
end

function Dataset = BuildTrainingBufferT(B,BInfo,TrainBuffer)
    Dataset = struct( ...
        'Dec',zeros(0,0), ...
        'Label',zeros(0,1), ...
        'Source',zeros(0,1), ...
        'train_size',0, ...
        'pos_count',0, ...
        'neg_count',0, ...
        'src_b',0, ...
        'src_refinement',0, ...
        'src_boundary_band',0, ...
        'pair_count',BInfo.PairCount);
    if ~isempty(B)
        Dataset.Dec = double(B.decs);
        Dataset.Label = double(all(B.cons<=0,2));
        Dataset.Source = ones(size(Dataset.Label));
    end
    if isstruct(TrainBuffer) && ~isempty(TrainBuffer.Dec)
        Dataset.Dec = [Dataset.Dec;double(TrainBuffer.Dec)];
        Dataset.Label = [Dataset.Label;double(TrainBuffer.Label(:) > 0)];
        Dataset.Source = [Dataset.Source;double(TrainBuffer.Source(:))];
    end
    Dataset.train_size = size(Dataset.Dec,1);
    Dataset.pos_count = sum(Dataset.Label == 1);
    Dataset.neg_count = sum(Dataset.Label == 0);
    Dataset.src_b = sum(Dataset.Source == 1);
    Dataset.src_refinement = sum(Dataset.Source == 2);
    Dataset.src_boundary_band = sum(Dataset.Source == 3);
end

function Balanced = BalanceTrainingDataset(Dataset)
    Balanced = Dataset;
    Pos = find(Dataset.Label == 1);
    Neg = find(Dataset.Label == 0);
    Count = min(numel(Pos),numel(Neg));
    if Count <= 0
        Balanced.Dec = zeros(0,size(Dataset.Dec,2));
        Balanced.Label = zeros(0,1);
        Balanced.Source = zeros(0,1);
    else
        Keep = sort([Pos(1:Count);Neg(1:Count)]);
        Balanced.Dec = Dataset.Dec(Keep,:);
        Balanced.Label = Dataset.Label(Keep);
        Balanced.Source = Dataset.Source(Keep);
    end
    Balanced.train_size = size(Balanced.Dec,1);
    Balanced.pos_count = sum(Balanced.Label == 1);
    Balanced.neg_count = sum(Balanced.Label == 0);
    Balanced.src_b = sum(Balanced.Source == 1);
    Balanced.src_refinement = sum(Balanced.Source == 2);
    Balanced.src_boundary_band = sum(Balanced.Source == 3);
end

function Model = TrainBoundaryMLPColdStart(Dataset,Problem,Hidden,Epoch,LR)
    [Xn,Y,Lower,Upper,Ready] = PrepareBoundaryMLPTrainingData(Dataset,Problem);
    if ~Ready
        Model = [];
        return;
    end
    [~,D] = size(Xn);
    W1 = 0.1*randn(D,Hidden);
    b1 = zeros(1,Hidden);
    W2 = 0.1*randn(Hidden,1);
    b2 = 0;
    Model = struct('Lower',Lower,'Upper',Upper,'W1',W1,'b1',b1,'W2',W2,'b2',b2);
    Model = RunBoundaryMLPTraining(Model,Xn,Y,Epoch,LR);
end

function Model = ContinueTrainBoundaryMLP(Model,Dataset,Problem,Hidden,Epoch,LR)
    [Xn,Y,Lower,Upper,Ready] = PrepareBoundaryMLPTrainingData(Dataset,Problem);
    if ~Ready
        return;
    end
    if ~IsBoundaryMLPCompatible(Model,size(Xn,2))
        Model = TrainBoundaryMLPColdStart(Dataset,Problem,Hidden,Epoch,LR);
        return;
    end
    Model.Lower = Lower;
    Model.Upper = Upper;
    Model = RunBoundaryMLPTraining(Model,Xn,Y,Epoch,LR);
end

function Model = RunBoundaryMLPTraining(Model,Xn,Y,Epoch,LR)
    W1 = Model.W1;
    b1 = Model.b1;
    W2 = Model.W2;
    b2 = Model.b2;
    for e = 1 : Epoch
        H = tanh(Xn*W1 + b1);
        Z = H*W2 + b2;
        P = 1./(1+exp(-Z));
        Delta2 = (P-Y)./max(size(Xn,1),1);
        D1 = (Delta2*W2').*(1-H.^2);
        Step = LR/sqrt(e);
        W2 = W2 - Step*(H'*Delta2);
        b2 = b2 - Step*sum(Delta2);
        W1 = W1 - Step*(Xn'*D1);
        b1 = b1 - Step*sum(D1,1);
    end

    Model.W1 = W1;
    Model.b1 = b1;
    Model.W2 = W2;
    Model.b2 = b2;
end

function Prob = PredictBoundaryMLP(Model,X)
    if nargin < 2 || isempty(X)
        Prob = zeros(0,1);
        return;
    end
    if isempty(Model) || ~isfield(Model,'Lower') || ~isfield(Model,'Upper') || ...
            ~IsBoundaryMLPCompatible(Model,size(X,2))
        Prob = 0.5*ones(size(X,1),1);
        return;
    end
    Xn = NormalizeDecisionByBounds(double(X),Model.Lower,Model.Upper);
    H = tanh(Xn*Model.W1 + Model.b1);
    Z = H*Model.W2 + Model.b2;
    Prob = min(max(1./(1+exp(-Z)),1e-6),1-1e-6);
end

function [Xn,Y,Lower,Upper,Ready] = PrepareBoundaryMLPTrainingData(Dataset,Problem)
    X = double(Dataset.Dec);
    Y = double(Dataset.Label(:) > 0);
    Lower = double(Problem.lower(:)');
    Upper = double(Problem.upper(:)');
    Ready = ~(isempty(X) || size(X,1) < 2 || numel(unique(Y)) < 2);
    if ~Ready
        Xn = zeros(0,numel(Lower));
        return;
    end
    Xn = NormalizeDecisionByBounds(X,Lower,Upper);
end

function Xn = NormalizeDecisionByBounds(X,Lower,Upper)
    Range = max(Upper-Lower,1e-12);
    Xn = 2*bsxfun(@rdivide,bsxfun(@minus,double(X),Lower),Range) - 1;
end

function Decision = ClipDecisionsToProblemBounds(Decision,Lower,Upper)
    Decision = bsxfun(@min,bsxfun(@max,double(Decision),double(Lower)),double(Upper));
end

function Dist = DecisionDistanceNormalized(A,B,Lower,Upper)
    An = NormalizeDecisionByBounds(A,double(Lower(:)'),double(Upper(:)'));
    Bn = NormalizeDecisionByBounds(B,double(Lower(:)'),double(Upper(:)'));
    Dist = sqrt(sum((An-Bn).^2,2));
end

function tf = IsBoundaryMLPCompatible(Model,D)
    tf = ~isempty(Model) && isfield(Model,'W1') && isfield(Model,'W2') && ...
        size(Model.W1,1) == D && size(Model.W1,2) == size(Model.W2,1);
end

function Flag = IsAccuracyDegraded(AccBefore,AccAfter)
    if ~isfinite(AccAfter)
        Flag = false;
    elseif isfinite(AccBefore)
        Flag = AccAfter < AccBefore - 1e-12;
    else
        Flag = AccAfter < 0.5 - 1e-12;
    end
end

%% Objective selection and population utilities

function Population = EnvironmentalSelectionU(Population,N,W)
    Population = KeepUniquePopulation(Population);
    Population = ObjectiveSelectionWithLastSectorTruncation(Population,min(N,numel(Population)),W);
    Population = PadPopulation(Population,N);
end

function Population = ObjectiveSelectionWithLastSectorTruncation(Population,N,W)
    if isempty(Population)
        return;
    end
    if N <= 0
        Population = Population([]);
        return;
    end

    N = min(N,numel(Population));
    if numel(Population) <= N
        return;
    end

    [FrontNo,MaxFNo] = NDSort(Population.objs,N);
    Next = FrontNo < MaxFNo;
    Last = find(FrontNo == MaxFNo);
    Need = N - sum(Next);
    if Need > 0 && ~isempty(Last)
        PickLast = SelectLastFrontBySector(Population(Last),Need,W,Population.objs);
        Next(Last(PickLast)) = true;
    end
    Population = Population(Next);
end

function Pick = SelectLastFrontBySector(LastPopulation,N,W,RefObj)
    Pick = zeros(0,1);
    if isempty(LastPopulation) || N <= 0
        return;
    end

    N = min(N,numel(LastPopulation));
    if isempty(W)
        CrowdDis = CrowdingDistance(LastPopulation.objs,ones(1,numel(LastPopulation)));
        [~,ord] = sort(CrowdDis,'descend');
        Pick = ord(1:N);
        return;
    end

    Sector = AssociateSectorsLocal(LastPopulation.objs,W,RefObj);
    Scalar = ComputeSectorScalar(LastPopulation.objs,W,RefObj,Sector);
    CrowdDis = CrowdingDistance(LastPopulation.objs,ones(1,numel(LastPopulation)));
    Ranked = cell(size(W,1),1);
    for s = 1 : size(W,1)
        idx = find(Sector == s);
        if isempty(idx)
            continue;
        end
        LocalCrowd = CrowdDis(idx);
        Key = [-LocalCrowd(:),Scalar(idx(:)),idx(:)];
        [~,ord] = sortrows(Key,[1 2 3]);
        Ranked{s} = idx(ord);
    end
    Order = SectorRoundRobinPick(Ranked,N);
    Pick = Order(:,2);
end

function Population = PadPopulation(Population,N)
    if isempty(Population)
        return;
    end
    if numel(Population) < N
        Population = [Population,Population(mod(0:N-numel(Population)-1,numel(Population))+1)];
    else
        Population = Population(1:N);
    end
end

function [Flag,FrontNo,CrowdDis] = ConstraintSideIndicator(Population)
    [FrontNo,CrowdDis] = ObjectiveSideIndicator(Population);
    Flag = sum(max(0,Population.cons),2);
end

function [FrontNo,CrowdDis] = ObjectiveSideIndicator(Population)
    [FrontNo,~] = NDSort(Population.objs,numel(Population));
    CrowdDis = CrowdingDistance(Population.objs,FrontNo);
end

function Population = FilterFeasiblePopulation(Population)
    if isempty(Population)
        return;
    end
    Population = Population(all(Population.cons<=0,2));
end

function Population = KeepUniquePopulation(Population)
    if isempty(Population)
        return;
    end
    Keep = KeepLatestDecisionRowsLocal(Population.decs);
    Population = Population(Keep);
end

function [Population,Source] = KeepUniquePopulationWithSource(Population,Source)
    Source = double(Source(:));
    if isempty(Population)
        Source = zeros(0,1);
        return;
    end
    Source = MatchLength(Source,numel(Population),BoundarySourceCode('b'));
    Keep = KeepLatestDecisionRowsLocal(Population.decs);
    Population = Population(Keep);
    Source = Source(Keep);
end

function Population = RemovePopulationByDecision(Population,Remove)
    if isempty(Population) || isempty(Remove)
        return;
    end
    Keep = ~ismember(Population.decs,Remove.decs,'rows');
    Population = Population(Keep);
end

function Keep = KeepLatestDecisionRowsLocal(Dec)
    if isempty(Dec)
        Keep = zeros(0,1);
        return;
    end
    [~,Keep] = unique(double(Dec),'rows','last');
    Keep = sort(Keep);
end

%% Objective-space geometry

function [ObjN,zmin,zmax] = NormalizeObjectives(Obj)
    if isempty(Obj)
        ObjN = Obj;
        zmin = zeros(1,0);
        zmax = zeros(1,0);
        return;
    end
    zmin = min(Obj,[],1);
    zmax = max(Obj,[],1);
    ObjN = NormalizeObjectivesWithBounds(Obj,zmin,zmax);
end

function ObjN = NormalizeObjectivesWithBounds(Obj,zmin,zmax)
    if isempty(Obj)
        ObjN = Obj;
        return;
    end
    ObjN = (Obj - zmin)./max(zmax-zmin,1e-12);
end

function Best = ComputeBestFeasibleScalar(RefObjN,RefLabel,W,Neighbors,FallbackObjN,FallbackLabel)
    if isempty(W)
        K = 1;
    else
        K = size(W,1);
    end
    Best = inf(K,1);
    Obj = RefObjN;
    Label = RefLabel;
    if isempty(Obj) || ~any(Label == 1)
        Obj = FallbackObjN;
        Label = FallbackLabel;
    end
    if isempty(Obj) || ~any(Label == 1)
        Best(:) = 0;
        return;
    end

    FeasibleObj = Obj(Label == 1,:);
    if isempty(W)
        Best(:) = min(sum(FeasibleObj,2));
        return;
    end
    FeasibleSector = AssociateSectorsLocal(FeasibleObj,W,zeros(1,size(FeasibleObj,2)));
    FeasibleScalar = ComputeSectorScalar(FeasibleObj,W,zeros(1,size(FeasibleObj,2)),FeasibleSector);
    for s = 1 : K
        Local = ResolveLocalSectorSet(s,Neighbors,K);
        idx = ismember(FeasibleSector,Local);
        if any(idx)
            Best(s) = min(FeasibleScalar(idx));
        else
            Best(s) = min(FeasibleScalar);
        end
    end
    Best(~isfinite(Best)) = 0;
end

function [Sector,Scalar] = AssociateSectorsLocal(Obj,W,RefObj)
    if isempty(Obj)
        Sector = zeros(0,1);
        if nargout > 1
            Scalar = zeros(0,1);
        end
        return;
    end
    if isempty(W)
        Sector = ones(size(Obj,1),1);
        if nargout > 1
            Scalar = sum(max(Obj,0),2);
        end
        return;
    end
    if nargin < 3 || isempty(RefObj)
        RefObj = min(Obj,[],1);
    else
        RefObj = min(RefObj,[],1);
    end
    Shift = max(Obj - RefObj,0);
    ShiftNorm = sqrt(sum(Shift.^2,2));
    ShiftNorm(ShiftNorm < 1e-12) = 1;
    WNorm = sqrt(sum(W.^2,2));
    WNorm(WNorm < 1e-12) = 1;
    Cosine = (Shift./ShiftNorm)*(W./WNorm)';
    [~,Sector] = max(Cosine,[],2);
    if nargout > 1
        Scalar = ComputeSectorScalar(Obj,W,RefObj,Sector);
    end
end

function Scalar = ComputeSectorScalar(Obj,W,RefObj,Sector)
    if isempty(Obj)
        Scalar = zeros(0,1);
        return;
    end
    if nargin < 3 || isempty(RefObj)
        RefObj = min(Obj,[],1);
    else
        RefObj = min(RefObj,[],1);
    end
    if nargin < 4 || isempty(Sector)
        Sector = AssociateSectorsLocal(Obj,W,RefObj);
    end
    Shift = max(Obj - RefObj,0);
    if isempty(W)
        Scalar = sum(Shift,2);
        return;
    end
    Scalar = zeros(size(Obj,1),1);
    MeanW = max(mean(W,1),1e-6);
    for i = 1 : size(Obj,1)
        if Sector(i) > 0 && Sector(i) <= size(W,1)
            Weight = max(W(Sector(i),:),1e-6);
        else
            Weight = MeanW;
        end
        Scalar(i) = max(Shift(i,:).*Weight,[],2);
    end
end

function Neighbors = BuildSectorNeighbors(W,NeighborCount)
    K = size(W,1);
    Neighbors = cell(K,1);
    if K <= 1 || NeighborCount <= 0
        return;
    end
    WNorm = sqrt(sum(W.^2,2));
    WNorm(WNorm < 1e-12) = 1;
    Wn = W./WNorm;
    Cosine = Wn*Wn';
    Cosine(1:K+1:end) = -inf;
    for i = 1 : K
        [~,ord] = sort(Cosine(i,:),'descend');
        Neighbors{i} = ord(1:min(NeighborCount,K-1));
    end
end

function Local = ResolveLocalSectorSet(Sector,Neighbors,K)
    if nargin < 3 || isempty(K)
        K = numel(Neighbors);
    end
    if K <= 0
        Local = 1;
        return;
    end
    if isempty(Neighbors)
        Local = min(max(Sector,1),K);
        return;
    end
    if Sector <= 0 || Sector > K
        Local = (1:K)';
        return;
    end
    Local = unique([Sector;Neighbors{Sector}(:)],'stable');
end

function Pick = SectorRoundRobinPick(Ranked,MaxPick)
    Pick = zeros(MaxPick,2);
    if isempty(Ranked) || MaxPick <= 0
        Pick = zeros(0,2);
        return;
    end
    Ptr = ones(numel(Ranked),1);
    PickCount = 0;
    while PickCount < MaxPick
        Changed = false;
        for s = 1 : numel(Ranked)
            if Ptr(s) <= numel(Ranked{s})
                PickCount = PickCount + 1;
                Pick(PickCount,:) = [s,Ranked{s}(Ptr(s))];
                Ptr(s) = Ptr(s) + 1;
                Changed = true;
                if PickCount >= MaxPick
                    break;
                end
            end
        end
        if ~Changed
            break;
        end
    end
    Pick = Pick(1:PickCount,:);
end

function Value = MatchLength(Value,Count,Fill)
    Value = Value(:);
    if isempty(Value)
        Value = Fill*ones(Count,1);
    elseif numel(Value) > Count
        Value = Value(1:Count);
    elseif numel(Value) < Count
        Value(end+1:Count,1) = Fill;
    end
end

%% Diagnostics

function BDiag = BuildBoundaryArchiveDiagnostics(B,BInfo,PopulationC,PopulationU,Problem)
    BDiag = EmptyBoundaryArchiveDiagnostics();
    PairCount = min([StructFieldOr(BInfo,'PairCount',0),floor(numel(B)/2),numel(BInfo.Active)]);
    if PairCount <= 0
        return;
    end
    Active = find(BInfo.Active(1:PairCount));
    if isempty(Active)
        return;
    end

    FeasibleRows = 2*Active(:)-1;
    InfeasibleRows = 2*Active(:);
    Rows = reshape([FeasibleRows,InfeasibleRows]',1,[]);
    Endpoints = B(Rows);
    EndpointObj = double(Endpoints.objs);

    [FeasibleSource,InfeasibleSource] = BoundaryPairSources(BInfo,PairCount);
    FeasibleSource = FeasibleSource(Active);
    InfeasibleSource = InfeasibleSource(Active);
    EndpointSource = reshape([FeasibleSource(:),InfeasibleSource(:)]',[],1);

    BDiag.src_b = sum(EndpointSource == BoundarySourceCode('b'));
    BDiag.src_popc = sum(EndpointSource == BoundarySourceCode('population_c'));
    BDiag.src_popu = sum(EndpointSource == BoundarySourceCode('population_u'));
    BDiag.src_offc = sum(EndpointSource == BoundarySourceCode('offspring_c'));
    BDiag.src_offu = sum(EndpointSource == BoundarySourceCode('offspring_u'));
    BDiag.src_refinement = sum(EndpointSource == BoundarySourceCode('refinement'));

    BDiag.inf_src_b = sum(InfeasibleSource == BoundarySourceCode('b'));
    BDiag.inf_src_popc = sum(InfeasibleSource == BoundarySourceCode('population_c'));
    BDiag.inf_src_popu = sum(InfeasibleSource == BoundarySourceCode('population_u'));
    BDiag.inf_src_offc = sum(InfeasibleSource == BoundarySourceCode('offspring_c'));
    BDiag.inf_src_offu = sum(InfeasibleSource == BoundarySourceCode('offspring_u'));
    BDiag.inf_src_refinement = sum(InfeasibleSource == BoundarySourceCode('refinement'));
    BDiag.inf_unconstrained_src_ratio = ...
        (BDiag.inf_src_popu + BDiag.inf_src_offu)/max(numel(InfeasibleSource),1);

    [OverlapC,OverlapU,OverlapAny] = ComputeBoundaryPopulationOverlap(Endpoints,PopulationC,PopulationU);
    BDiag.overlap_popc_count = sum(OverlapC);
    BDiag.overlap_popc_ratio = mean(double(OverlapC));
    BDiag.overlap_popu_count = sum(OverlapU);
    BDiag.overlap_popu_ratio = mean(double(OverlapU));
    BDiag.overlap_any_population_ratio = mean(double(OverlapAny));

    if size(EndpointObj,2) ~= 2
        return;
    end
    BoundaryPoints = ExtractObjectiveBoundaryPoints(Problem);
    if isempty(BoundaryPoints)
        return;
    end

    FeasibleObj = double(B(FeasibleRows).objs);
    InfeasibleObj = double(B(InfeasibleRows).objs);
    MidObj = 0.5*(FeasibleObj + InfeasibleObj);
    MidDist = DistanceToPointSet(MidObj,BoundaryPoints);
    InfeasibleDist = DistanceToPointSet(InfeasibleObj,BoundaryPoints);
    FarThreshold = ObjectiveBoundaryFarThreshold([BoundaryPoints;EndpointObj(:,1:2)]);

    BDiag.mean_mid_boundary_dist = MeanOrNaN(MidDist);
    BDiag.p90_mid_boundary_dist = PercentileOrNaN(MidDist,90);
    BDiag.mean_infeasible_boundary_dist = MeanOrNaN(InfeasibleDist);
    BDiag.p90_infeasible_boundary_dist = PercentileOrNaN(InfeasibleDist,90);
    BDiag.max_infeasible_boundary_dist = MaxOrNaN(InfeasibleDist);
    BDiag.far_infeasible_boundary_ratio = mean(double(InfeasibleDist > FarThreshold));
end

function BDiag = EmptyBoundaryArchiveDiagnostics()
    BDiag = struct( ...
        'mean_mid_boundary_dist',NaN, ...
        'p90_mid_boundary_dist',NaN, ...
        'mean_infeasible_boundary_dist',NaN, ...
        'p90_infeasible_boundary_dist',NaN, ...
        'max_infeasible_boundary_dist',NaN, ...
        'far_infeasible_boundary_ratio',NaN, ...
        'overlap_popc_count',0, ...
        'overlap_popc_ratio',NaN, ...
        'overlap_popu_count',0, ...
        'overlap_popu_ratio',NaN, ...
        'overlap_any_population_ratio',NaN, ...
        'src_b',0, ...
        'src_popc',0, ...
        'src_popu',0, ...
        'src_offc',0, ...
        'src_offu',0, ...
        'src_refinement',0, ...
        'inf_src_b',0, ...
        'inf_src_popc',0, ...
        'inf_src_popu',0, ...
        'inf_src_offc',0, ...
        'inf_src_offu',0, ...
        'inf_src_refinement',0, ...
        'inf_unconstrained_src_ratio',NaN);
end

function [OverlapC,OverlapU,OverlapAny] = ComputeBoundaryPopulationOverlap(BoundaryPopulation,PopulationC,PopulationU)
    EndpointCount = numel(BoundaryPopulation);
    OverlapC = false(EndpointCount,1);
    OverlapU = false(EndpointCount,1);
    if EndpointCount <= 0
        OverlapAny = false(0,1);
        return;
    end
    Dec = double(BoundaryPopulation.decs);
    if ~isempty(PopulationC)
        OverlapC = ismember(Dec,double(PopulationC.decs),'rows');
    end
    if ~isempty(PopulationU)
        OverlapU = ismember(Dec,double(PopulationU.decs),'rows');
    end
    OverlapAny = OverlapC | OverlapU;
end

function Points = ExtractObjectiveBoundaryPoints(Problem)
    Points = zeros(0,2);
    try
        PF = Problem.PF;
    catch
        return;
    end
    if iscell(PF)
        Points = ExtractObjectiveBoundaryPointsFromCell(PF);
    elseif isnumeric(PF) && size(PF,2) >= 2
        Points = double(PF(:,1:2));
        Points = Points(all(isfinite(Points),2),:);
    end
    if ~isempty(Points)
        Points = unique(Points,'rows');
    end
end

function Points = ExtractObjectiveBoundaryPointsFromCell(PF)
    Points = zeros(0,2);
    if numel(PF) < 3
        return;
    end
    X = double(PF{1});
    Y = double(PF{2});
    Z = double(PF{3});
    if isempty(X) || isempty(Y) || isempty(Z) || ~isequal(size(X),size(Y),size(Z))
        return;
    end
    Mask = isfinite(Z);
    Edge = false(size(Mask));
    if size(Mask,1) > 1
        Change = Mask(1:end-1,:) ~= Mask(2:end,:);
        Edge(1:end-1,:) = Edge(1:end-1,:) | Change;
        Edge(2:end,:) = Edge(2:end,:) | Change;
    end
    if size(Mask,2) > 1
        Change = Mask(:,1:end-1) ~= Mask(:,2:end);
        Edge(:,1:end-1) = Edge(:,1:end-1) | Change;
        Edge(:,2:end) = Edge(:,2:end) | Change;
    end
    Points = [X(Edge),Y(Edge)];
    Points = Points(all(isfinite(Points),2),:);
end

function Dist = DistanceToPointSet(Points,Reference)
    Points = double(Points);
    Reference = double(Reference);
    Dist = NaN(size(Points,1),1);
    if isempty(Points) || isempty(Reference)
        return;
    end
    for i = 1 : size(Points,1)
        Delta = bsxfun(@minus,Reference,Points(i,:));
        Dist(i) = sqrt(min(sum(Delta.^2,2)));
    end
end

function Threshold = ObjectiveBoundaryFarThreshold(Points)
    Points = Points(all(isfinite(Points),2),:);
    if isempty(Points)
        Threshold = inf;
        return;
    end
    Span = max(Points,[],1) - min(Points,[],1);
    Threshold = 0.05*sqrt(sum(Span.^2));
    if Threshold <= 0 || ~isfinite(Threshold)
        Threshold = inf;
    end
end

function Observer = InitObserver(Algorithm,Problem,Params)
    Params = normalizeParams(Params);
    RootDir = fileparts(which('platemo'));
    BaseFolder = fullfile(RootDir,'Data','PRBCCMO_t');
    [~,~] = mkdir(BaseFolder);
    [~,Token] = fileparts(tempname(BaseFolder));
    RunFolder = fullfile(BaseFolder,sprintf('%s_%s_run%d_%s', ...
        class(Algorithm),class(Problem),resolveRunId(Algorithm),Token));
    [~,~] = mkdir(RunFolder);

    Observer = struct( ...
        'folder',RunFolder, ...
        'meta_file',fullfile(RunFolder,'run_meta.csv'), ...
        'core_file',fullfile(RunFolder,'core_metrics.csv'));
    if Params(8) > 0
        Observer.boundary_folder = fullfile(RunFolder,'boundary_snapshots');
        [~,~] = mkdir(Observer.boundary_folder);
        Observer.boundary_file = fullfile(RunFolder,'boundary_snapshot.csv');
        Observer.boundary_manifest_file = fullfile(RunFolder,'boundary_snapshots.csv');
        Observer.boundary_checkpoint_fes = unique(max(1,round([0.25,0.50,0.75,1.00]*Problem.maxFE)));
        Observer.boundary_next_checkpoint = 1;
        WriteCsvHeader(Observer.boundary_manifest_file,{ ...
            'target_fe','actual_fe','generation','b_pair_count','snapshot_file'});
    end

    WriteCsvHeader(Observer.meta_file,{ ...
        'algorithm','problem','family','run','M','D','N','maxFE', ...
        'hidden','epoch','lr','betaB','etaB','Tretrain','Gstart','output_folder'});
    AppendCsvRows(Observer.meta_file,{ ...
        class(Algorithm),class(Problem),familyOfProblem(class(Problem)), ...
        resolveRunId(Algorithm),Problem.M,Problem.D,Problem.N,Problem.maxFE, ...
        Params(1),Params(2),Params(3),Params(4),Params(5),Params(6),Params(7),RunFolder});

    WriteCsvHeader(Observer.core_file,{ ...
        'generation','fe','fe_ratio', ...
        'b_pair_count','b_mean_pair_gap','b_p90_pair_gap','b_mean_front_gap','b_p90_front_gap', ...
        'b_active_sector_ratio','b_changed_pair_count','b_changed_pair_ratio', ...
        'b_added_pair_count','b_replaced_pair_count','b_dropped_pair_count','b_contracted_pair_count', ...
        'b_global_change_pair_ratio','b_contraction_pair_ratio', ...
        'b_mean_mid_boundary_dist','b_p90_mid_boundary_dist', ...
        'b_mean_infeasible_boundary_dist','b_p90_infeasible_boundary_dist', ...
        'b_max_infeasible_boundary_dist','b_far_infeasible_boundary_ratio', ...
        'b_overlap_popc_count','b_overlap_popc_ratio','b_overlap_popu_count','b_overlap_popu_ratio', ...
        'b_overlap_any_population_ratio', ...
        'b_src_b','b_src_popc','b_src_popu','b_src_offc','b_src_offu','b_src_refinement', ...
        'b_inf_src_b','b_inf_src_popc','b_inf_src_popu','b_inf_src_offc','b_inf_src_offu','b_inf_src_refinement', ...
        'b_inf_unconstrained_src_ratio', ...
        'mlp_due','mlp_can_train','mlp_trained','mlp_acc_before','mlp_train_acc','mlp_degraded', ...
        'train_size','train_pos','train_neg', ...
        'train_src_b','train_src_refinement','train_src_boundary_band', ...
        'inf_pool_size','inf_selected','inf_pool_mean_prob','inf_selected_mean_prob','inf_prob_gain', ...
        'inf_carry_pool_size','inf_carry_applicable_count', ...
        'inf_carry_selected','inf_carry_pool_mean_prob','inf_carry_selected_mean_prob','inf_carry_prob_gain'});
end

function Observer = LogGenerationDiagnostics(Observer,Problem,Generation, ...
    B,BInfo,PopulationC,PopulationU,BlendDiag,MLPDiag,SelectionDiag)
    Active = BInfo.Active;
    BDiag = BuildBoundaryArchiveDiagnostics(B,BInfo,PopulationC,PopulationU,Problem);
    ChangedPairCount = double(BlendDiag.added) + double(BlendDiag.replaced) + ...
        double(StructFieldOr(BlendDiag,'contracted',0));
    ActiveSectors = unique(BInfo.Sector(Active));
    AppendCsvRows(Observer.core_file,{ ...
        Generation,Problem.FE,min(Problem.FE/max(Problem.maxFE,1),1), ...
        BInfo.PairCount,MeanOrNaN(BInfo.PairGap(Active)),PercentileOrNaN(BInfo.PairGap(Active),90), ...
        MeanOrNaN(BInfo.FrontGap(Active)),PercentileOrNaN(BInfo.FrontGap(Active),90), ...
        numel(ActiveSectors)/max(Problem.N,1),ChangedPairCount,ChangedPairCount/max(numel(BInfo.Active),1), ...
        BlendDiag.added,BlendDiag.replaced,BlendDiag.dropped,StructFieldOr(BlendDiag,'contracted',0), ...
        (double(BlendDiag.added)+double(BlendDiag.replaced))/max(BInfo.PairCount,1), ...
        double(StructFieldOr(BlendDiag,'contracted',0))/max(BInfo.PairCount,1), ...
        BDiag.mean_mid_boundary_dist,BDiag.p90_mid_boundary_dist, ...
        BDiag.mean_infeasible_boundary_dist,BDiag.p90_infeasible_boundary_dist, ...
        BDiag.max_infeasible_boundary_dist,BDiag.far_infeasible_boundary_ratio, ...
        BDiag.overlap_popc_count,BDiag.overlap_popc_ratio,BDiag.overlap_popu_count,BDiag.overlap_popu_ratio, ...
        BDiag.overlap_any_population_ratio, ...
        BDiag.src_b,BDiag.src_popc,BDiag.src_popu,BDiag.src_offc,BDiag.src_offu,BDiag.src_refinement, ...
        BDiag.inf_src_b,BDiag.inf_src_popc,BDiag.inf_src_popu,BDiag.inf_src_offc,BDiag.inf_src_offu,BDiag.inf_src_refinement, ...
        BDiag.inf_unconstrained_src_ratio, ...
        MLPDiag.due,MLPDiag.can_train,MLPDiag.trained,MLPDiag.acc_before,MLPDiag.acc_after,MLPDiag.degraded, ...
        MLPDiag.train_size,MLPDiag.pos_count,MLPDiag.neg_count, ...
        MLPDiag.src_b,MLPDiag.src_refinement,MLPDiag.src_boundary_band, ...
        SelectionDiag.pool_size,SelectionDiag.selected,SelectionDiag.pool_mean_prob,SelectionDiag.selected_mean_prob,SelectionDiag.prob_gain, ...
        SelectionDiag.carry_pool_size,SelectionDiag.carry_applicable_count, ...
        SelectionDiag.carry_selected,SelectionDiag.carry_pool_mean_prob, ...
        SelectionDiag.carry_selected_mean_prob,SelectionDiag.carry_prob_gain});
end

function Observer = WriteBoundarySnapshotsIfDue(Observer,Problem,Generation,B,BInfo)
    if ~isstruct(Observer) || ~isfield(Observer,'boundary_manifest_file')
        return;
    end
    Checkpoints = Observer.boundary_checkpoint_fes;
    Next = Observer.boundary_next_checkpoint;
    while Next <= numel(Checkpoints) && Problem.FE >= Checkpoints(Next)
        TargetFE = Checkpoints(Next);
        SnapshotFile = fullfile(Observer.boundary_folder, ...
            sprintf('boundary_snapshot_fe%06d.csv',TargetFE));
        WriteBoundarySnapshotFile(SnapshotFile,Problem,B,BInfo,Generation,Problem.FE,TargetFE);
        AppendCsvRows(Observer.boundary_manifest_file,{ ...
            TargetFE,Problem.FE,Generation,BInfo.PairCount,SnapshotFile});
        if TargetFE >= Problem.maxFE
            WriteBoundarySnapshotFile(Observer.boundary_file,Problem,B,BInfo, ...
                Generation,Problem.FE,TargetFE);
        end
        Next = Next + 1;
    end
    Observer.boundary_next_checkpoint = Next;
end

function WriteBoundarySnapshotFile(FilePath,Problem,B,BInfo,Generation,ActualFE,TargetFE)
    if isempty(FilePath)
        return;
    end

    PairCount = min([BInfo.PairCount,floor(numel(B)/2),numel(BInfo.Active)]);
    if PairCount <= 0
        Active = [];
    else
        Active = find(BInfo.Active(1:PairCount));
    end

    ObjHeader = arrayfun(@(m)sprintf('obj%d',m),1:Problem.M,'UniformOutput',false);
    Header = [{'target_fe','actual_fe','generation', ...
        'pair_index','pair_side','feasible','source_code','sector','pair_gap','front_gap'},ObjHeader];
    WriteCsvHeader(FilePath,Header);

    if ~isempty(Active)
        Rows = cell(numel(Active)*2,numel(Header));
        SideNames = {'feasible_endpoint','infeasible_endpoint'};
        [FeasibleSource,InfeasibleSource] = BoundaryPairSources(BInfo,PairCount);
        r = 0;
        for i = 1 : numel(Active)
            p = Active(i);
            for side = 1 : 2
                idx = 2*p + side - 2;
                if idx > numel(B)
                    continue;
                end
                Obj = nan(1,Problem.M);
                CurrentObj = double(B(idx).objs);
                Obj(1:min(Problem.M,numel(CurrentObj))) = CurrentObj(1:min(Problem.M,numel(CurrentObj)));
                Cons = double(B(idx).cons);
                IsFeasible = isempty(Cons) || all(Cons <= 0);

                r = r + 1;
                if side == 1
                    SourceCode = FeasibleSource(p);
                else
                    SourceCode = InfeasibleSource(p);
                end
                Rows(r,1:10) = {TargetFE,ActualFE,Generation, ...
                    p,SideNames{side},IsFeasible, ...
                    SourceCode,BInfo.Sector(p),BInfo.PairGap(p),BInfo.FrontGap(p)};
                for m = 1 : Problem.M
                    Rows{r,10+m} = Obj(m);
                end
            end
        end
        AppendCsvRows(FilePath,Rows(1:r,:));
    end
end

%% Diagnostic structs and scalar helpers

function Diag = InitMLPDiag(Generation,FE,Model,Dataset,BInfo)
    Diag = struct( ...
        'generation',Generation, ...
        'fe',FE, ...
        'due',false, ...
        'can_train',false, ...
        'trained',false, ...
        'skip_reason',"not_due", ...
        'train_size',Dataset.train_size, ...
        'pos_count',Dataset.pos_count, ...
        'neg_count',Dataset.neg_count, ...
        'src_b',Dataset.src_b, ...
        'src_refinement',Dataset.src_refinement, ...
        'src_boundary_band',Dataset.src_boundary_band, ...
        'pair_count',BInfo.PairCount, ...
        'acc_before',EvaluateBinaryAccuracy(Model,Dataset.Dec,Dataset.Label), ...
        'degraded',false, ...
        'acc_after',NaN);
end

function Diag = InitInfeasibleSelectionDiag()
    Diag = struct( ...
        'pool_size',0, ...
        'applicable_count',0, ...
        'selected',0, ...
        'pool_mean_prob',NaN, ...
        'selected_mean_prob',NaN, ...
        'prob_gain',NaN, ...
        'carry_pool_size',0, ...
        'carry_applicable_count',0, ...
        'carry_selected',0, ...
        'carry_pool_mean_prob',NaN, ...
        'carry_selected_mean_prob',NaN, ...
        'carry_prob_gain',NaN);
end

function Acc = EvaluateBinaryAccuracy(Model,X,Y)
    if isempty(Model) || isempty(X) || isempty(Y) || numel(unique(Y)) < 2
        Acc = NaN;
        return;
    end
    Prob = PredictBoundaryMLP(Model,X);
    Y = double(Y(:) > 0);
    Pred = double(Prob(:) >= 0.5);
    Acc = mean(Pred == Y);
end

function Value = MeanOrNaN(Value)
    Value = double(Value(:));
    Value = Value(isfinite(Value));
    if isempty(Value)
        Value = NaN;
    else
        Value = mean(Value);
    end
end

function Value = MaxOrNaN(Value)
    Value = double(Value(:));
    Value = Value(isfinite(Value));
    if isempty(Value)
        Value = NaN;
    else
        Value = max(Value);
    end
end

function Value = PercentileOrNaN(Value,Percentile)
    Value = double(Value(:));
    Value = sort(Value(isfinite(Value)));
    if isempty(Value)
        Value = NaN;
    else
        idx = max(1,min(numel(Value),ceil(Percentile/100*numel(Value))));
        Value = Value(idx);
    end
end

function Value = StructFieldOr(S,Name,Default)
    if isstruct(S) && isfield(S,Name)
        Value = S.(Name);
    else
        Value = Default;
    end
end

function RunId = resolveRunId(Algorithm)
    if isempty(Algorithm.run)
        RunId = 1;
    else
        RunId = Algorithm.run;
    end
end

function Family = familyOfProblem(problem)
    name = char(string(problem));
    if startsWith(name,'DASCMOP')
        Family = 'DASCMOP_BC';
    elseif startsWith(name,'LIRCMOP')
        Family = 'LIRCMOP_BC';
    else
        Family = 'other';
    end
end

function WriteCsvHeader(FilePath,Header)
    fid = fopen(FilePath,'w');
    cleaner = onCleanup(@()fclose(fid));
    fprintf(fid,'%s\n',strjoin(Header,','));
end

function AppendCsvRows(FilePath,Rows)
    if isempty(Rows)
        return;
    end
    if ~iscell(Rows)
        Rows = {Rows};
    end
    if isvector(Rows) && ~isempty(Rows) && ~iscell(Rows{1})
        Rows = reshape(Rows,1,[]);
    end
    fid = fopen(FilePath,'a');
    cleaner = onCleanup(@()fclose(fid));
    for r = 1 : size(Rows,1)
        Line = cell(1,size(Rows,2));
        for c = 1 : size(Rows,2)
            Line{c} = ToCsvValue(Rows{r,c});
        end
        fprintf(fid,'%s\n',strjoin(Line,','));
    end
end

function S = ToCsvValue(Value)
    if isstring(Value) || ischar(Value)
        S = char(string(Value));
        S = strrep(S,'"','""');
        if contains(S,',') || contains(S,'"') || contains(S,newline)
            S = ['"',S,'"'];
        end
    elseif islogical(Value)
        S = sprintf('%d',double(Value));
    elseif isnumeric(Value)
        if isempty(Value) || (isscalar(Value) && isnan(Value))
            S = 'NaN';
        elseif isscalar(Value)
            S = sprintf('%.15g',double(Value));
        else
            S = ['"',strjoin(arrayfun(@(x)sprintf('%.15g',x),double(Value(:))','UniformOutput',false),' '),'"'];
        end
    else
        S = char(string(Value));
    end
end
