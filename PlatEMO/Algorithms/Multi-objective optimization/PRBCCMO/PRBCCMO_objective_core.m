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
    UseMLP   = Params(9) > 0;
    rhoRef   = 0.10;
    Kbis     = 3;

    NBPair = max(1,ceil(betaB*Problem.N));
    MaxRefinementBufPerClass = round(0.5*Problem.N);

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

        [OffspringC,MatingDiag] = GenerateDEOffspring( ...
            Problem,PopulationC,true,W,Model,B,BInfo,PopulationC,UseMLP);
        OffspringUDec = GenerateDEOffspringDecision(Problem,PopulationU,false);
        [OffspringU,B,BInfo,ContractionDiag] = EvaluateHelperOffspringWithBoundaryRefinement( ...
            Problem,OffspringUDec,B,BInfo,rhoRef,Kbis,PopulationC,W,NBPair);
        [B,BInfo,~,BlendDiag] = UpdateBoundaryArchiveObjective( ...
            B,BInfo,PopulationC,PopulationU,OffspringC,OffspringU,W,NBPair,etaB,Problem);
        BInfo = ResetContractedPairAges(B,BInfo,ContractionDiag.accepted_key);
        BlendDiag.contracted = ContractionDiag.accepted;

        TrainBuffer = UpdateTrainingBufferT(TrainBuffer,ContractionDiag.accepted_refined,ContractionDiag.rejected_refined,Generation,MaxRefinementBufPerClass);
        [Model,MLPDiag] = UpdateBoundaryMLPPeriodically( ...
            Model,B,BInfo,TrainBuffer,PopulationC,PopulationU,OffspringU,W, ...
            hidden,epoch,lr,Generation,Gstart,Tretrain,Problem,ContractionDiag);

        QC = KeepUniquePopulation([PopulationC,OffspringC,OffspringU]);
        QU = KeepUniquePopulation([PopulationU,OffspringC,OffspringU]);
        [PopulationC,SelectionDiag] = EnvironmentalSelectionC_ObjectBoundary( ...
            QC,Problem.N,W,Model,B,BInfo,PopulationC,UseMLP);
        PopulationU = EnvironmentalSelectionU(QU,Problem.N,W);
        ResultPopulationC = BuildExternalResultPopulation(PopulationC,ResultPopulationC);

        if TraceMode
            Observer = LogGenerationDiagnostics(Observer,Problem,Generation, ...
                B,BInfo,PopulationC,PopulationU,BlendDiag,MLPDiag,MatingDiag,SelectionDiag);
            Observer = WriteBoundarySnapshotsIfDue(Observer,Problem,Generation,B,BInfo);
        end
    end
end

function Params = normalizeParams(Params)
    Defaults = [64,200,1e-3,3,0.1,10,0,0,1];
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
    Params(3) = max(1e-6,Params(3));
    Params(4) = max(1,Params(4));
    Params(5) = max(0,min(1,Params(5)));
    Params(6) = max(1,round(Params(6)));
    Params(7) = max(0,round(Params(7)));
    Params(8) = double(Params(8) > 0);
    Params(9) = double(Params(9) > 0);
end

%% Main-population offspring

function [Offspring,MatingDiag] = GenerateDEOffspring(Problem,Population,useConstraintIndicator,W,Model,B,BInfo,PopulationC,UseMLP)
    MatingDiag = InitMatingDiag();
    if isempty(Population)
        Offspring = Population;
        return;
    end
    if nargin < 4
        W = [];
    end
    if nargin < 5
        Model = [];
    end
    if nargin < 6
        B = Population([]);
    end
    if nargin < 7 || isempty(BInfo)
        BInfo = InitBoundaryArchiveInfo(0);
    end
    if nargin < 8
        PopulationC = Population([]);
    end
    if nargin < 9
        UseMLP = true;
    end
    [OffspringDec,MatingDiag] = GenerateDEOffspringDecision( ...
        Problem,Population,useConstraintIndicator,W,Model,B,BInfo,PopulationC,UseMLP);
    Offspring = Problem.Evaluation(OffspringDec);
end

function [OffspringDec,MatingDiag] = GenerateDEOffspringDecision(Problem,Population,useConstraintIndicator,W,Model,B,BInfo,PopulationC,UseMLP)
    MatingDiag = InitMatingDiag();
    if isempty(Population)
        OffspringDec = zeros(0,Problem.D);
        return;
    end
    if nargin < 4
        W = [];
    end
    if nargin < 5
        Model = [];
    end
    if nargin < 6
        B = Population([]);
    end
    if nargin < 7 || isempty(BInfo)
        BInfo = InitBoundaryArchiveInfo(0);
    end
    if nargin < 8
        PopulationC = Population([]);
    end
    if nargin < 9
        UseMLP = true;
    end

    N = numel(Population);
    if useConstraintIndicator
        [MatingPool,MatingDiag] = TournamentSelectionConstraintMLP( ...
            Population,2*N,W,Model,B,BInfo,PopulationC,UseMLP);
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
    StepIndices = 1 : Kbis;
    RefinedCells = cell(1,numel(StepIndices));
    for step = 1 : numel(StepIndices) %#ok<NASGU>
        BatchDec = zeros(nRef,Problem.D);
        [BatchDec,RefineInfo] = InjectBoundaryRefinementIntoHelperOffspring( ...
            Problem,BatchDec,B,BInfo,PairPlan);
        Refined = Problem.Evaluation(BatchDec);
        [B,BInfo,StepDiag] = ContractBoundaryPairsByRefinedSamples( ...
            B,BInfo,Refined,RefineInfo,PopulationC,W,NBPair,Problem);
        Diag = MergeContractionDiag(Diag,StepDiag);
        RefinedCells{step} = Refined;
    end
    RefinedAll = [RefinedCells{:}];

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

    PairGap = BInfo.PairGap(Active);
    PairAge = BInfo.Age(Active);
    Valid = isfinite(PairGap(:)) & isfinite(PairAge(:));
    Active = Active(Valid);
    PairGap = PairGap(Valid);
    PairAge = PairAge(Valid);
    if isempty(Active)
        return;
    end

    [~,ord] = sortrows([-PairGap(:),-PairAge(:),Active(:)],[1 2 3]);
    Pick = Active(ord(1:min(MaxPairs,numel(ord))));
end

%% Objective-space boundary archive

function BInfo = InitBoundaryArchiveInfo(NBPair)
    BInfo = struct( ...
        'PairCount',0, ...
        'Sector',zeros(NBPair,1), ...
        'MidScalar',inf(NBPair,1), ...
        'PairGap',inf(NBPair,1), ...
        'FeasibleSource',ones(NBPair,1), ...
        'InfeasibleSource',ones(NBPair,1), ...
        'Age',inf(NBPair,1), ...
        'Active',false(NBPair,1));
end

function [B,BInfo,Target,BlendDiag] = UpdateBoundaryArchiveObjective( ...
    B,BInfo,PopulationC,PopulationU,OffspringC,OffspringU,W,NBPair,etaB,Problem)

    %#ok<INUSD>
    [A,Source] = BuildBoundaryCandidatePool(B,BInfo,OffspringC,OffspringU);
    Target = BuildTargetBoundaryArchive(A,Source,W,NBPair);
    [B,BInfo,BlendDiag] = BlendBoundaryArchive(B,BInfo,Target,etaB,NBPair);
end

function [A,Source] = BuildBoundaryCandidatePool(B,BInfo,OffspringC,OffspringU)
    A = [B,OffspringC,OffspringU];
    Source = [BoundaryEndpointSources(B,BInfo); ...
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
        if Accept
            B(2*p-1) = CandF;
            B(2*p) = CandI;
            BInfo.Sector(p) = CandSector;
            BInfo.PairGap(p) = CandKey(1);
            BInfo.MidScalar(p) = CandKey(2);
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
            Diag.accepted_refined = [Diag.accepted_refined,Refined]; %#ok<AGROW>
        else
            Diag.rejected = Diag.rejected + 1;
            Diag.rejected_refined = [Diag.rejected_refined,Refined]; %#ok<AGROW>
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
        'accepted_refined',Prototype([]), ...
        'rejected_refined',Prototype([]), ...
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
    Diag.accepted_refined = [Diag.accepted_refined,StepDiag.accepted_refined];
    Diag.rejected_refined = [Diag.rejected_refined,StepDiag.rejected_refined];
    Diag.refined = [Diag.accepted_refined,Diag.rejected_refined];
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

    [OldKey,CandKey,CandSector] = BuildBoundaryPairObjectiveKeys( ...
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

    Flag = CompareBoundaryPairObjectiveKeys(CandKey,OldKey) < 0 || ...
        (CompareBoundaryPairObjectiveKeys(CandKey,OldKey) == 0 && NewDecGap < OldDecGap - 1e-12);
end

function [OldKey,CandKey,CandSector] = BuildBoundaryPairObjectiveKeys( ...
    OldF,OldI,CandF,CandI,PopulationC,W)

    PairObj = [double(OldF.objs);double(OldI.objs);double(CandF.objs);double(CandI.objs)];
    PopCObj = zeros(0,size(PairObj,2));
    if ~isempty(PopulationC)
        PopCObj = double(PopulationC.objs);
    end
    AllObj = [PairObj;PopCObj];
    [AllObjN,~,~] = NormalizeObjectives(AllObj);
    PairObjN = AllObjN(1:4,:);

    [OldKey,~] = EvaluateBoundaryPairKeyFromNorm( ...
        PairObjN(1,:),PairObjN(2,:),W);
    [CandKey,CandSector] = EvaluateBoundaryPairKeyFromNorm( ...
        PairObjN(3,:),PairObjN(4,:),W);
end

function Cmp = CompareBoundaryPairObjectiveKeys(CandKey,OldKey)
    CandKey = double(CandKey(:)');
    OldKey = double(OldKey(:)');
    if any(~isfinite(CandKey))
        Cmp = 1;
        return;
    end
    if any(~isfinite(OldKey))
        Cmp = -1;
        return;
    end
    Tol = 1e-12;
    Width = min(numel(CandKey),numel(OldKey));
    for i = 1 : Width
        if CandKey(i) < OldKey(i) - Tol
            Cmp = -1;
            return;
        elseif CandKey(i) > OldKey(i) + Tol
            Cmp = 1;
            return;
        end
    end
    Cmp = 0;
end

function [Key,Sector] = EvaluateBoundaryPairKeyFromNorm(FObjN,IObjN,W)
    MidObj = 0.5*(FObjN + IObjN);
    if isempty(W)
        Sector = 1;
    else
        Sector = AssociateSectorsLocal(MidObj,W,zeros(1,size(MidObj,2)));
    end
    PairGap = sqrt(sum((FObjN - IObjN).^2,2));
    MidScalar = ComputeSectorScalar(MidObj,W,zeros(1,size(MidObj,2)),Sector);
    Key = [PairGap,MidScalar];
end

function Target = InitTargetArchive(Prototype,NBPair)
    Target = struct( ...
        'Population',Prototype([]), ...
        'Sector',zeros(0,1), ...
        'MidScalar',zeros(0,1), ...
        'PairGap',zeros(0,1), ...
        'FeasibleSource',zeros(0,1), ...
        'InfeasibleSource',zeros(0,1), ...
        'AvailableSectorCount',0, ...
        'Quota',zeros(NBPair,1), ...
        'Priority',zeros(NBPair,1));
end

function Target = BuildTargetBoundaryArchive(A,Source,W,NBPair)
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

    [ObjN,~,~] = NormalizeObjectives(Obj);
    K = max(1,size(W,1));
    if isempty(W)
        SectorAll = ones(size(ObjN,1),1);
        Neighbors = cell(1,1);
    else
        SectorAll = AssociateSectorsLocal(ObjN,W,zeros(1,size(ObjN,2)));
        Neighbors = BuildSectorNeighbors(W,min(3,max(K-1,0)));
    end

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
            ObjN,Label,LocalFeasible,LocalInfeasible,s,W);
    end

    Priority = ComputeSectorPriority(PairLists,K);
    Quota = AllocateSectorQuota(PairLists,Priority,NBPair);
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
        'MidScalar',zeros(0,1));
end

function PairList = BuildSectorCandidatePairsObjective( ...
    ObjN,~,LocalFeasible,LocalInfeasible,Sector,W)
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

    Key = [KeepGap(:),MidScalar(:),(1:Count)'];
    [~,ord] = sortrows(Key,[1 2 3]);
    PairList.FeasibleIndex = KeepI(ord);
    PairList.InfeasibleIndex = KeepJ(ord);
    PairList.Sector = Sector*ones(Count,1);
    PairList.PairGap = KeepGap(ord);
    PairList.MidScalar = MidScalar(ord);
end

function Priority = ComputeSectorPriority(PairLists,K)
    Priority = zeros(K,1);
    for s = 1 : K
        Count = numel(PairLists{s}.FeasibleIndex);
        if Count > 0
            BestPairGap = max(PairLists{s}.PairGap(1),0);
            Priority(s) = log(1+Count)/(eps + BestPairGap);
        end
    end
end

function Quota = AllocateSectorQuota(PairLists,Priority,NBPair)
    K = numel(PairLists);
    Quota = zeros(K,1);
    Available = find(cellfun(@(P)~isempty(P.FeasibleIndex),PairLists));
    if isempty(Available) || NBPair <= 0
        return;
    end

    if numel(Available) > NBPair
        [~,ord] = sort(Priority(Available),'descend');
        Available = Available(ord(1:NBPair));
    end
    [~,ord] = sort(Priority(Available),'descend');
    Available = Available(ord);
    Quota(Available)=1;
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
    MidScalar = zeros(NBPair,1);
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
            MidScalar(Count) = List.MidScalar(p);
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
    MidScalar = MidScalar(1:Count);
    PairFSource = PairFSource(1:Count);
    PairISource = PairISource(1:Count);
    Key = [PairGap,MidScalar,PairS,(1:Count)'];
    [~,ord] = sortrows(Key,[1 2 3 4]);
    PairF = PairF(ord);
    PairI = PairI(ord);
    PairS = PairS(ord);
    PairGap = PairGap(ord);
    MidScalar = MidScalar(ord);
    PairFSource = PairFSource(ord);
    PairISource = PairISource(ord);

    Target.Population = flattenPairPopulation(A,PairF,PairI);
    Target.Sector = PairS;
    Target.PairGap = PairGap;
    Target.MidScalar = MidScalar;
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
        Current.MidScalar(c) = TargetPairs.MidScalar(t);
        Current.PairGap(c) = TargetPairs.PairGap(t);
        Current.FeasibleSource(c) = TargetPairs.FeasibleSource(t);
        Current.InfeasibleSource(c) = TargetPairs.InfeasibleSource(t);
    end

    AddIdx = find(~InCurrent);
    if ~isempty(AddIdx)
        [~,ord] = sortrows([TargetPairs.PairGap(AddIdx),AddIdx(:)],[1 2]);
        AddIdx = AddIdx(ord);
    end
    DropIdx = find(~InTarget);
    if ~isempty(DropIdx)
        [~,ord] = sortrows([Current.PairGap(DropIdx),Current.Age(DropIdx),DropIdx(:)],[-1 -2 3]);
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
        [~,ord] = sortrows([NewPairs.PairGap,NewPairs.Age,(1:NewPairs.Count)'],[1 2 3]);
        NewPairs = SlicePairStruct(NewPairs,sort(ord(1:NBPair)));
    end
    Diag.kept = max(0,NewPairs.Count - Diag.added - Diag.replaced);
    [B,BInfo] = PairStructToPopulation(NewPairs,B,NBPair);
end

function Pairs = InitPairStruct(Prototype)
    Pairs = struct( ...
        'Population',Prototype([]), ...
        'Sector',zeros(0,1), ...
        'MidScalar',zeros(0,1), ...
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
    Pairs.MidScalar = BInfo.MidScalar(Active);
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
        numel(Target.MidScalar),numel(Target.PairGap), ...
        numel(Target.FeasibleSource),numel(Target.InfeasibleSource)]);
    if Count <= 0
        return;
    end
    Pairs.Population = Target.Population(1:2*Count);
    Pairs.Sector = Target.Sector(1:Count);
    Pairs.MidScalar = Target.MidScalar(1:Count);
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
    Pairs.MidScalar = Pairs.MidScalar(Keep);
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
    Pairs.MidScalar = [Pairs.MidScalar;Add.MidScalar];
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
    Pairs.MidScalar(DropIndex) = Source.MidScalar(SourceIndex);
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
    BInfo.MidScalar(1:Count) = Pairs.MidScalar(1:Count);
    BInfo.PairGap(1:Count) = Pairs.PairGap(1:Count);
    BInfo.FeasibleSource(1:Count) = Pairs.FeasibleSource(1:Count);
    BInfo.InfeasibleSource(1:Count) = Pairs.InfeasibleSource(1:Count);
    BInfo.Age(1:Count) = Pairs.Age(1:Count);
    BInfo.Active(1:Count) = true;
end

%% Boundary metadata and environmental selection

function [Population,Diag] = EnvironmentalSelectionC_ObjectBoundary(Population,N,W,Model,B,BInfo,PreviousC,UseMLP)
    Diag = InitInfeasibleSelectionDiag();
    Population = KeepUniquePopulation(Population);
    if isempty(Population)
        return;
    end
    if nargin < 8
        UseMLP = true;
    end

    Feasible = FilterFeasiblePopulation(Population);
    Infeasible = Population(any(Population.cons>0,2));
    Diag.pool_size = numel(Infeasible);

    Next = ObjectiveSelectionWithLastSectorTruncation( ...
        Feasible,min(N,numel(Feasible)),W);

    Need = N - numel(Next);
    Diag.feasible_deficit = max(Need,0);
    if Need > 0
        [Pick,Diag] = SelectTopInfeasibleByUtilityMeta( ...
            Infeasible,Need,W,Model,B,BInfo,PreviousC,UseMLP,Diag);
        Next = [Next,Pick];
    end

    if numel(Next) < N
        Rest = RemovePopulationByDecision(Population,Next);
        Rest = ObjectiveSelectionWithLastSectorTruncation(Rest,min(N-numel(Next),numel(Rest)),W);
        Next = [Next,Rest];
    end
    Population = PadPopulation(Next,N);
end

function [Pick,Diag] = SelectTopInfeasibleByUtilityMeta(Population,N,W,Model,B,BInfo,PopulationC,UseMLP,Diag)
    if nargin < 8
        UseMLP = true;
    end
    if nargin < 9 || isempty(Diag)
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
    UtilityMeta = ComputeInfeasibleUtilityMeta(Meta,UseMLP);
    CandidateIndex = find(Meta.feasible == 0);
    Diag.pool_size = numel(Population);
    Diag.ranked_count = numel(CandidateIndex);
    if isempty(CandidateIndex)
        return;
    end

    Diag.mlp_used = double(UseMLP > 0 && ~isempty(Model) && ...
        isfield(Model,'Lower') && isfield(Model,'Upper') && ...
        IsBoundaryMLPCompatible(Model,size(Population.decs,2)));
    Diag.pool_mean_utility = MeanOrNaN(UtilityMeta.utility(CandidateIndex));
    Diag.pool_mean_prob = MeanOrNaN(Meta.prob(CandidateIndex));
    Order = RankInfeasibleByUtilityMeta(UtilityMeta,CandidateIndex);
    PickIndex = Order(1:min(N,numel(Order)));
    Pick = Population(PickIndex);
    Diag.selected = numel(Pick);
    Diag.selected_mean_utility = MeanOrNaN(UtilityMeta.utility(PickIndex));
    Diag.utility_gain = DifferenceOrNaN(Diag.selected_mean_utility,Diag.pool_mean_utility);
    Diag.selected_mean_prob = MeanOrNaN(Meta.prob(PickIndex));
    Diag.selected_prob_gain = DifferenceOrNaN(Diag.selected_mean_prob,Diag.pool_mean_prob);
end

function Meta = BuildBoundaryMetaFromB(Candidates,W,Model,B,BInfo,PopulationC)
    N = numel(Candidates);
    Meta = InitBoundaryMeta(N);
    if isempty(Candidates)
        return;
    end
    [CandObjN,BObjN,~,~] = NormalizeCandidateBObjects(Candidates,B,PopulationC);
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
    ScalarObj = ComputeSectorScalar(CandObjN,W,zeros(1,size(CandObjN,2)),Sector);

    Meta.sector = Sector;
    Meta.feasible = Feasible;
    Meta.prob = Prob;
    Meta.supportDistObjToB = SupportDist;
    Meta.scalarObj = ScalarObj;
end

function [CandObjN,BObjN,PopCObjN,PopCFeasible] = NormalizeCandidateBObjects(Candidates,B,PopulationC)
    CandObj = double(Candidates.objs);
    M = size(CandObj,2);
    BObj = zeros(0,M);
    PopCObj = zeros(0,M);
    PopCFeasible = false(0,1);
    if ~isempty(B)
        BObj = double(B.objs);
    end
    if ~isempty(PopulationC)
        PopCObj = double(PopulationC.objs);
        PopCFeasible = all(PopulationC.cons<=0,2);
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
        if ~isfinite(Dist(i))
            Dist(i) = PointToSegmentsDistanceObj(CandObjN(i,:),SegmentA,SegmentB);
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

function Meta = InitBoundaryMeta(Count)
    Meta = struct( ...
        'sector',zeros(Count,1), ...
        'feasible',zeros(Count,1), ...
        'prob',0.5*ones(Count,1), ...
        'supportDistObjToB',inf(Count,1), ...
        'scalarObj',inf(Count,1));
end

function UtilityMeta = ComputeInfeasibleUtilityMeta(Meta,UseMLP)
    if nargin < 2
        UseMLP = true;
    end
    SupportDist = double(Meta.supportDistObjToB(:));
    SupportDist(~isfinite(SupportDist)) = inf;
    ScalarObj = double(Meta.scalarObj(:));
    ScalarObj(~isfinite(ScalarObj)) = inf;
    ScalarObj = max(ScalarObj,0);
    BRank = NormalizeLexRankAsc([SupportDist,ScalarObj]);
    Prob = double(Meta.prob(:));
    if ~UseMLP
        Prob = 0.5*ones(size(Prob));
    end
    tauProb = 0.02;
    ProbRank = NormalizeTieAwareRankDesc(Prob,tauProb);
    Utility = ProbRank;
    Utility(~isfinite(Utility)) = -inf;
    UtilityMeta = Meta;
    UtilityMeta.b_rank = BRank;
    UtilityMeta.prob_rank = ProbRank;
    UtilityMeta.utility = Utility;
end

function Score = NormalizeLexRankAsc(Keys)
    if isempty(Keys)
        Score = zeros(0,1);
        return;
    end
    Keys = double(Keys);
    n = size(Keys,1);
    Keys(~isfinite(Keys)) = inf;
    [~,ord] = sortrows([Keys,(1:n)'],1:size(Keys,2)+1);
    Rank = zeros(n,1);
    Rank(ord) = (1:n)';
    if n <= 1
        Score = ones(n,1);
    else
        Score = (n - Rank) ./ (n - 1);
    end
end

function Score = NormalizeTieAwareRankDesc(Value,Tol)
    if nargin < 2 || isempty(Tol)
        Tol = 1e-12;
    end
    Tol = max(0,double(Tol));
    Value = double(Value(:));
    n = numel(Value);
    if n <= 0
        Score = zeros(0,1);
        return;
    end
    Value(~isfinite(Value)) = -inf;
    [Sorted,ord] = sort(Value,'descend');
    Rank = zeros(n,1);
    left = 1;
    while left <= n
        right = left;
        while right < n && abs(Sorted(right+1)-Sorted(left)) <= Tol
            right = right + 1;
        end
        Rank(ord(left:right)) = mean(left:right);
        left = right + 1;
    end
    if n <= 1
        Score = ones(n,1);
    else
        Score = (n - Rank) ./ (n - 1);
    end
end

function Order = RankInfeasibleByUtilityMeta(UtilityMeta,idx)
    idx = idx(:);
    if isempty(idx)
        Order = idx;
        return;
    end
    Utility = double(UtilityMeta.utility(idx));
    Utility(~isfinite(Utility)) = -inf;
    SupportDist = double(UtilityMeta.supportDistObjToB(idx));
    SupportDist(~isfinite(SupportDist)) = inf;
    ScalarObj = double(UtilityMeta.scalarObj(idx));
    ScalarObj(~isfinite(ScalarObj)) = inf;
    Key=[-Utility,SupportDist,ScalarObj,idx];
    [~,ord] = sortrows(Key,1:size(Key,2));
    Order = idx(ord);
end

function Cmp = CompareInfeasibleByUtility(UtilityMeta,A,B)
    KeyA = InfeasibleUtilityKey(UtilityMeta,A);
    KeyB = InfeasibleUtilityKey(UtilityMeta,B);
    Cmp = CompareLexicographicKey(KeyA,KeyB);
end

function Key = InfeasibleUtilityKey(UtilityMeta,Index)
    Utility = double(UtilityMeta.utility(Index));
    if ~isfinite(Utility)
        Utility = -inf;
    end
    SupportDist = double(UtilityMeta.supportDistObjToB(Index));
    if ~isfinite(SupportDist)
        SupportDist = inf;
    end
    ScalarObj = double(UtilityMeta.scalarObj(Index));
    if ~isfinite(ScalarObj)
        ScalarObj = inf;
    end
    Key = [-Utility,SupportDist,ScalarObj,Index];
end

function Cmp = CompareLexicographicKey(A,B)
    Tol = 1e-12;
    Width = min(numel(A),numel(B));
    Cmp = 0;
    for i = 1 : Width
        if A(i) < B(i) - Tol
            Cmp = -1;
            return;
        elseif A(i) > B(i) + Tol
            Cmp = 1;
            return;
        end
    end
end

%% T-buffer MLP

function [Model,Diag] = UpdateBoundaryMLPPeriodically( ...
    Model,B,BInfo,TrainBuffer,PopulationC,PopulationU,OffspringU,W, ...
    hidden,epoch,lr,Generation,Gstart,Tretrain,Problem,ContractionDiag)

    if nargin < 16
        ContractionDiag = InitContractionDiag(B);
    end
    Dataset = BuildBoundaryDrivenDataset(B,BInfo,TrainBuffer,PopulationC,PopulationU,OffspringU,W,Problem);
    Diag = InitMLPDiag(Generation,Problem.FE,Model,Dataset,BInfo);
    Diag.due = Generation >= Gstart && mod(Generation-Gstart,Tretrain) == 0;
    MinTrain = MinimumRecoverabilityTrainingSize(Problem);
    MinPairs = max(4,2*Problem.M);
    Diag.can_train = Diag.due && Dataset.train_size >= MinTrain && ...
        Dataset.pos_count > 0 && Dataset.neg_count > 0 && ...
        Dataset.pair_count >= MinPairs;
    if ~Diag.due
        Diag.skip_reason = "not_due";
        Diag = EvaluateRefinementProbabilityDiag(Diag,Model,ContractionDiag);
        return;
    end
    if ~Diag.can_train
        Diag.skip_reason = "insufficient_boundary_driven_data";
        Diag = EvaluateRefinementProbabilityDiag(Diag,Model,ContractionDiag);
        return;
    end

    if isempty(Model)
        Model = TrainBoundaryMLPColdStart(Dataset,Problem,hidden,epoch,lr);
    else
        WarmEpoch = 80;
        WarmLR = 3e-4;
        Model = ContinueTrainBoundaryMLP(Model,Dataset,Problem,hidden,WarmEpoch,WarmLR);
    end
    Diag.trained = true;
    Diag.acc_after = EvaluateBinaryAccuracy(Model,Dataset.Dec,Dataset.Label);
    Diag.degraded = IsAccuracyDegraded(Diag.acc_before,Diag.acc_after);
    Diag = EvaluateBoundaryMLPProbabilityGroups(Diag,Model,Dataset);
    Diag = EvaluateRefinementProbabilityDiag(Diag,Model,ContractionDiag);
    Diag.skip_reason = "trained";
end

function TrainBuffer = InitTrainingBuffer(D)
    TrainBuffer = struct( ...
        'Dec',zeros(0,D), ...
        'Label',zeros(0,1), ...
        'Time',zeros(0,1));
end

function TrainBuffer = UpdateTrainingBufferT(TrainBuffer,AcceptedRefinement,RejectedRefinement,Generation,MaxRefinementBufPerClass)
    if nargin < 5 || isempty(MaxRefinementBufPerClass)
        MaxRefinementBufPerClass = inf;
    end
    TrainBuffer = AppendTrainingSamples(TrainBuffer,AcceptedRefinement,Generation,1);
    TrainBuffer = AppendTrainingSamples(TrainBuffer,RejectedRefinement,Generation,0);
    TrainBuffer = KeepLatestTrainingRows(TrainBuffer);
    TrainBuffer = TrimTrainingBufferByClassTime(TrainBuffer,MaxRefinementBufPerClass);
end

function TrainBuffer = AppendTrainingSamples(TrainBuffer,Population,Generation,LabelValue)
    if isempty(Population)
        return;
    end
    Count = numel(Population);
    TrainBuffer.Dec = [TrainBuffer.Dec;double(Population.decs)];
    TrainBuffer.Label = [TrainBuffer.Label;double(LabelValue > 0)*ones(Count,1)];
    TrainBuffer.Time = [TrainBuffer.Time;Generation*ones(Count,1)];
end

function TrainBuffer = KeepLatestTrainingRows(TrainBuffer)
    Keep = KeepLatestDecisionRowsLocal(TrainBuffer.Dec);
    TrainBuffer.Dec = TrainBuffer.Dec(Keep,:);
    TrainBuffer.Label = TrainBuffer.Label(Keep);
    TrainBuffer.Time = TrainBuffer.Time(Keep);
end

function TrainBuffer = TrimTrainingBufferByClassTime(TrainBuffer,MaxPerClass)
    if isempty(TrainBuffer.Dec) || ~isfinite(MaxPerClass)
        return;
    end
    MaxPerClass = max(0,round(MaxPerClass));
    Keep = false(size(TrainBuffer.Label));
    for LabelValue = [1 0]
        Rows = find(TrainBuffer.Label == LabelValue);
        if isempty(Rows)
            continue;
        end
        [~,ord] = sortrows([-TrainBuffer.Time(Rows),Rows(:)],[1 2]);
        Keep(Rows(ord(1:min(MaxPerClass,numel(ord))))) = true;
    end
    Keep = find(Keep);
    TrainBuffer.Dec = TrainBuffer.Dec(Keep,:);
    TrainBuffer.Label = TrainBuffer.Label(Keep);
    TrainBuffer.Time = TrainBuffer.Time(Keep);
end

function Dataset = BuildBoundaryDrivenDataset(B,BInfo,TrainBuffer,PopulationC,PopulationU,OffspringU,W,Problem)
    Dataset = InitTrainingDataset(BInfo,Problem.D);
    TrainPairs = SelectBoundaryPairsForTraining(B,BInfo,Problem);
    Dataset.pair_count = numel(TrainPairs);
    Dataset = CollectRefinementOutcomeSamples(Dataset,TrainBuffer);
    if TrainingDataNeedsSupplement(Dataset,Problem)
        Dataset = CollectBoundaryEndpointAnchors(Dataset,B,BInfo,TrainPairs);
    end
    if TrainingDataNeedsSupplement(Dataset,Problem)
        Dataset = CollectCurrentPopulationAnchors( ...
            Dataset,PopulationC,PopulationU,OffspringU,B,BInfo,W,PopulationC,Problem);
    end
    Dataset = FinalizeBoundaryDrivenDataset(Dataset,Problem);
end

function MinTrain = MinimumRecoverabilityTrainingSize(Problem)
    MinTrain = max(64,round(0.8*Problem.N));
end

function tf = TrainingDataNeedsSupplement(Dataset,Problem)
    Label = double(Dataset.Label(:) > 0);
    tf = numel(Label) < MinimumRecoverabilityTrainingSize(Problem) || ...
        ~any(Label == 1) || ~any(Label == 0);
end

function Dataset = InitTrainingDataset(BInfo,D)
    if nargin < 2 || isempty(D)
        D = 0;
    end
    Dataset = struct( ...
        'Dec',zeros(0,D), ...
        'Label',zeros(0,1), ...
        'Source',zeros(0,1), ...
        'Time',zeros(0,1), ...
        'Pair',zeros(0,1), ...
        'Priority',zeros(0,1), ...
        'train_size',0, ...
        'pos_count',0, ...
        'neg_count',0, ...
        'b_core_count',0, ...
        'refinement_count',0, ...
        'refinement_pos_count',0, ...
        'refinement_neg_count',0, ...
        'refinement_ratio',0, ...
        'current_pop_count',0, ...
        'pair_count',BInfo.PairCount);
end

function Pick = SelectBoundaryPairsForTraining(B,BInfo,Problem)
    Pick = zeros(0,1);
    if isempty(B) || BInfo.PairCount <= 0
        return;
    end

    PairCount = min([BInfo.PairCount,floor(numel(B)/2),numel(BInfo.Active)]);
    Active = find(BInfo.Active(1:PairCount));
    if isempty(Active)
        return;
    end

    PairGap = double(BInfo.PairGap(Active));
    MidScalar = double(BInfo.MidScalar(Active));
    Sector = double(BInfo.Sector(Active));
    Valid = isfinite(PairGap(:)) & isfinite(MidScalar(:)) & isfinite(Sector(:)) & Sector(:) > 0;
    Active = Active(Valid);
    PairGap = PairGap(Valid);
    MidScalar = MidScalar(Valid);
    Sector = Sector(Valid);
    if isempty(Active)
        return;
    end

    K = max([max(Sector),Problem.N,1]);
    Ranked = cell(K,1);
    for s = 1 : K
        Local = find(Sector == s);
        if isempty(Local)
            Ranked{s} = zeros(0,1);
            continue;
        end
        [~,ord] = sortrows([PairGap(Local),MidScalar(Local),Active(Local)],[1 2 3]);
        Ranked{s} = Active(Local(ord));
    end

    MaxPairs = max(4,ceil(0.5*Problem.N));
    PickWithSector = SectorRoundRobinPick(Ranked,MaxPairs);
    if ~isempty(PickWithSector)
        Pick = PickWithSector(:,2);
    end
end

function Dataset = CollectRefinementOutcomeSamples(Dataset,TrainBuffer)
    if isstruct(TrainBuffer) && isfield(TrainBuffer,'Dec') && ~isempty(TrainBuffer.Dec)
        Dataset = AppendTrainingDatasetRows( ...
            Dataset,double(TrainBuffer.Dec),double(TrainBuffer.Label > 0), ...
            2*ones(size(TrainBuffer.Label)),double(TrainBuffer.Time), ...
            zeros(size(TrainBuffer.Label)),ones(size(TrainBuffer.Label)));
    end
end

function Dataset = CollectBoundaryEndpointAnchors(Dataset,B,BInfo,PairPick)
    %#ok<INUSD>
    PairPick = double(PairPick(:));
    if ~isempty(B) && ~isempty(PairPick)
        Rows = reshape([2*PairPick(:)-1,2*PairPick(:)]',[],1);
        Rows = Rows(Rows >= 1 & Rows <= numel(B));
        if ~isempty(Rows)
            PairId = ceil(Rows(:)/2);
            Dataset = AppendTrainingDatasetRows( ...
                Dataset,double(B(Rows).decs),double(all(B(Rows).cons<=0,2)), ...
                ones(numel(Rows),1),inf(numel(Rows),1),PairId,2*ones(numel(Rows),1));
        end
    end
end

function Dataset = CollectCurrentPopulationAnchors( ...
    Dataset,PopulationC,PopulationU,OffspringU,B,BInfo,W,PopulationCContext,Problem)

    PerClass = max(1,ceil(0.8*Problem.N));
    Feasible = FilterFeasiblePopulation(PopulationC);
    FeasiblePick = SelectCurrentPopulationRows( ...
        Feasible,PerClass,B,BInfo,W,PopulationCContext);
    if ~isempty(FeasiblePick)
        Dataset = AppendTrainingDatasetRows( ...
            Dataset,double(FeasiblePick.decs),ones(numel(FeasiblePick),1), ...
            3*ones(numel(FeasiblePick),1),inf(numel(FeasiblePick),1), ...
            zeros(numel(FeasiblePick),1),3*ones(numel(FeasiblePick),1));
    end

    Infeasible = [PopulationU,OffspringU];
    if ~isempty(Infeasible)
        Infeasible = Infeasible(any(Infeasible.cons>0,2));
    end
    InfeasiblePick = SelectCurrentPopulationRows( ...
        Infeasible,PerClass,B,BInfo,W,PopulationCContext);
    if ~isempty(InfeasiblePick)
        Dataset = AppendTrainingDatasetRows( ...
            Dataset,double(InfeasiblePick.decs),zeros(numel(InfeasiblePick),1), ...
            3*ones(numel(InfeasiblePick),1),inf(numel(InfeasiblePick),1), ...
            zeros(numel(InfeasiblePick),1),3*ones(numel(InfeasiblePick),1));
    end
end

function Pick = SelectCurrentPopulationRows(Population,MaxPick,B,BInfo,W,PopulationC)
    Pick = Population([]);
    if isempty(Population) || MaxPick <= 0
        return;
    end
    Population = KeepUniquePopulation(Population);
    if isempty(Population)
        return;
    end
    Meta = BuildBoundaryMetaFromB(Population,W,[],B,BInfo,PopulationC);
    Candidate = (1:numel(Population))';
    SupportDist = double(Meta.supportDistObjToB(:));
    ScalarObj = double(Meta.scalarObj(:));
    Near = Candidate(isfinite(SupportDist));
    if isempty(Near)
        Ranked = zeros(0,1);
    else
        [~,ord] = sortrows([SupportDist(Near),ScalarObj(Near),Near],[1 2 3]);
        Ranked = Near(ord);
    end
    Remaining = setdiff(Candidate,Ranked,'stable');
    Fill = UniformSupplementRows(Remaining,max(0,MaxPick-numel(Ranked)));
    Rows = [Ranked(:);Fill(:)];
    Rows = Rows(1:min(MaxPick,numel(Rows)));
    Pick = Population(Rows);
end

function Rows = UniformSupplementRows(Candidate,MaxPick)
    Candidate = Candidate(:);
    if isempty(Candidate) || MaxPick <= 0
        Rows = zeros(0,1);
        return;
    end
    Count = min(MaxPick,numel(Candidate));
    if Count == numel(Candidate)
        Rows = Candidate;
    else
        Pos = unique(round(linspace(1,numel(Candidate),Count)),'stable');
        Rows = Candidate(Pos(:));
    end
end

function Dataset = AppendTrainingDatasetRows(Dataset,Dec,Label,Source,Time,Pair,Priority)
    if isempty(Dec)
        return;
    end
    Count = size(Dec,1);
    if Count ~= numel(Label) || Count ~= numel(Source) || Count ~= numel(Time) || ...
            Count ~= numel(Pair) || Count ~= numel(Priority)
        error('PRBCCMO:InvalidTrainingDatasetRows','Training dataset row fields must have matching lengths.');
    end
    Dataset.Dec = [Dataset.Dec;double(Dec)];
    Dataset.Label = [Dataset.Label;double(Label(:) > 0)];
    Dataset.Source = [Dataset.Source;double(Source(:))];
    Dataset.Time = [Dataset.Time;double(Time(:))];
    Dataset.Pair = [Dataset.Pair;double(Pair(:))];
    Dataset.Priority = [Dataset.Priority;double(Priority(:))];
end

function Dataset = FinalizeBoundaryDrivenDataset(Dataset,Problem)
    if ~isempty(Dataset.Dec)
        Keep = KeepBestTrainingDatasetRows(Dataset);
        Dataset = SliceTrainingDataset(Dataset,Keep);
    end
    Pos = find(Dataset.Label == 1);
    Neg = find(Dataset.Label == 0);
    MaxTotal = max(96,min(192,round(1.6*Problem.N)));
    Count = min([numel(Pos),numel(Neg),floor(MaxTotal/2)]);
    if Count <= 0
        Dataset = SliceTrainingDataset(Dataset,zeros(0,1));
    else
        PosKeep = SelectDatasetRowsByPriority(Dataset,Pos,Count);
        NegKeep = SelectDatasetRowsByPriority(Dataset,Neg,Count);
        Keep = sort([PosKeep;NegKeep]);
        Dataset = SliceTrainingDataset(Dataset,Keep);
    end
    Dataset = UpdateTrainingDatasetCounts(Dataset);
end

function Keep = KeepBestTrainingDatasetRows(Dataset)
    if isempty(Dataset.Dec)
        Keep = zeros(0,1);
        return;
    end
    n = size(Dataset.Dec,1);
    [~,ord] = sortrows([Dataset.Priority(:),-Dataset.Time(:),(1:n)'],[1 2 3]);
    [~,ia] = unique(Dataset.Dec(ord,:),'rows','stable');
    Keep = sort(ord(ia));
end

function Keep = SelectDatasetRowsByPriority(Dataset,Rows,Count)
    Rows = Rows(:);
    if isempty(Rows) || Count <= 0
        Keep = zeros(0,1);
        return;
    end
    [~,ord] = sortrows([Dataset.Priority(Rows),-Dataset.Time(Rows),Dataset.Pair(Rows),Rows],[1 2 3 4]);
    Keep = Rows(ord(1:min(Count,numel(ord))));
end

function Dataset = UpdateTrainingDatasetCounts(Dataset)
    Dataset.train_size = size(Dataset.Dec,1);
    Dataset.pos_count = sum(Dataset.Label == 1);
    Dataset.neg_count = sum(Dataset.Label == 0);
    Dataset.b_core_count = sum(Dataset.Source == 1);
    Dataset.refinement_count = sum(Dataset.Source == 2);
    Dataset.refinement_pos_count = sum(Dataset.Source == 2 & Dataset.Label == 1);
    Dataset.refinement_neg_count = sum(Dataset.Source == 2 & Dataset.Label == 0);
    Dataset.refinement_ratio = Dataset.refinement_count / max(Dataset.train_size,1);
    Dataset.current_pop_count = sum(Dataset.Source == 3);
end

function Dataset = SliceTrainingDataset(Dataset,Keep)
    Keep = Keep(:);
    D = size(Dataset.Dec,2);
    Dataset.Dec = Dataset.Dec(Keep,:);
    if isempty(Dataset.Dec)
        Dataset.Dec = zeros(0,D);
    end
    Dataset.Label = Dataset.Label(Keep);
    Dataset.Source = Dataset.Source(Keep);
    Dataset.Time = Dataset.Time(Keep);
    Dataset.Pair = Dataset.Pair(Keep);
    Dataset.Priority = Dataset.Priority(Keep);
end

function Model = TrainBoundaryMLPColdStart(Dataset,Problem,Hidden,Epoch,LR)
    [Xn,Y,Lower,Upper,Ready] = PrepareBoundaryMLPTrainingData(Dataset,Problem);
    if ~Ready
        Model = [];
        return;
    end
    Model = TrainBoundaryMLPWithDeepLearningToolbox([],Xn,Y,Lower,Upper,Hidden,Epoch,LR,false);
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
    Model = TrainBoundaryMLPWithDeepLearningToolbox(Model.Net,Xn,Y,Lower,Upper,Hidden,Epoch,LR,true);
end

function Model = TrainBoundaryMLPWithDeepLearningToolbox(PreviousNet,Xn,Y,Lower,Upper,Hidden,Epoch,LR,IsWarm)
    assert(exist('trainnet','file') == 2, ...
        'PRBCCMO:MissingTrainnet', ...
        'MATLAB Deep Learning Toolbox trainnet is required for PRBCCMO MLP training.');
    Hidden1 = max(2,round(Hidden));
    Hidden2 = max(16,round(Hidden1/2));
    Xn = single(Xn);
    Y = single(Y(:));
    [TrainIdx,ValIdx] = StratifiedValidationSplit(Y,0.20);
    XTrain = Xn(TrainIdx,:);
    YTrain = Y(TrainIdx,:);
    XVal = Xn(ValIdx,:);
    YVal = Y(ValIdx,:);
    MiniBatch = min(64,max(1,size(XTrain,1)));

    if isempty(PreviousNet)
        NetOrLayers = [
            featureInputLayer(size(XTrain,2),Normalization="none")
            fullyConnectedLayer(Hidden1)
            reluLayer
            fullyConnectedLayer(Hidden2)
            reluLayer
            fullyConnectedLayer(1)
            sigmoidLayer];
    else
        NetOrLayers = PreviousNet;
    end

    if isempty(ValIdx)
        Options = trainingOptions("adam", ...
            InitialLearnRate=LR, ...
            MaxEpochs=Epoch, ...
            MiniBatchSize=MiniBatch, ...
            Shuffle="every-epoch", ...
            Verbose=false, ...
            Plots="none", ...
            ExecutionEnvironment="cpu");
    else
        Options = trainingOptions("adam", ...
            InitialLearnRate=LR, ...
            MaxEpochs=Epoch, ...
            MiniBatchSize=MiniBatch, ...
            Shuffle="every-epoch", ...
            ValidationData={XVal,YVal}, ...
            ValidationFrequency=max(1,floor(size(XTrain,1)/MiniBatch)), ...
            ValidationPatience=15, ...
            OutputNetwork="best-validation", ...
            Verbose=false, ...
            Plots="none", ...
            ExecutionEnvironment="cpu");
    end

    Net = trainnet(XTrain,YTrain,NetOrLayers,"binary-crossentropy",Options);
    Model = struct( ...
        'Kind',"deep_learning_toolbox", ...
        'Lower',Lower, ...
        'Upper',Upper, ...
        'Net',Net, ...
        'Hidden',[Hidden1,Hidden2], ...
        'InputSize',size(XTrain,2), ...
        'IsWarm',logical(IsWarm));
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
    Xn = single(NormalizeDecisionByBounds(double(X),Model.Lower,Model.Upper));
    Scores = minibatchpredict(Model.Net,Xn,MiniBatchSize=1024, ...
        ExecutionEnvironment="cpu",Acceleration="auto");
    if isa(Scores,'dlarray')
        Scores = extractdata(Scores);
    end
    Scores = gather(Scores);
    Prob = min(max(double(Scores(:)),1e-6),1-1e-6);
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

function [TrainIdx,ValIdx] = StratifiedValidationSplit(Y,Holdout)
    Y = double(Y(:) > 0);
    TrainIdx = (1:numel(Y))';
    ValIdx = zeros(0,1);
    if numel(Y) < 10 || numel(unique(Y)) < 2
        return;
    end
    ValCells = cell(2,1);
    cellCount = 0;
    for Label = [0 1]
        Rows = find(Y == Label);
        if numel(Rows) < 5
            ValIdx = zeros(0,1);
            return;
        end
        Count = max(1,floor(Holdout*numel(Rows)));
        Count = min(Count,numel(Rows)-1);
        cellCount = cellCount + 1;
        ValCells{cellCount} = Rows(end-Count+1:end);
    end
    ValIdx = vertcat(ValCells{1:cellCount});
    TrainIdx = setdiff(TrainIdx,ValIdx,'stable');
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
    tf = ~isempty(Model) && isstruct(Model) && isfield(Model,'Net') && ...
        isfield(Model,'InputSize') && double(Model.InputSize) == D;
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

function [MatingPool,Diag] = TournamentSelectionConstraintMLP(Population,K,W,Model,B,BInfo,PopulationC,UseMLP)
    N = numel(Population);
    MatingPool = zeros(1,K);
    Diag = InitMatingDiag();
    if N <= 0 || K <= 0
        return;
    end
    if nargin < 8
        UseMLP = true;
    end
    [FrontNo,CrowdDis] = ObjectiveSideIndicator(Population);
    Meta = BuildBoundaryMetaFromB(Population,W,Model,B,BInfo,PopulationC);
    UtilityMeta = ComputeInfeasibleUtilityMeta(Meta,UseMLP);
    BaselineUtilityMeta = ComputeInfeasibleUtilityMeta(Meta,false);
    IsFeasible = all(Population.cons<=0,2);
    Diag.mlp_used = double(UseMLP > 0 && ~isempty(Model) && ...
        IsBoundaryMLPCompatible(Model,size(Population.decs,2)));
    TwoInfCount = 0;
    EffectiveCount = 0;
    PrimaryResolutionCount = 0;
    BFallbackCount = 0;
    for i = 1 : K
        Candidate = randi(N,1,2);
        MatingPool(i) = SelectConstraintTournamentWinner( ...
            Candidate(1),Candidate(2),IsFeasible,FrontNo,CrowdDis,UtilityMeta);
        if ~IsFeasible(Candidate(1)) && ~IsFeasible(Candidate(2))
            TwoInfCount = TwoInfCount + 1;
            if abs(UtilityMeta.prob_rank(Candidate(1)) - UtilityMeta.prob_rank(Candidate(2))) > 1e-12
                PrimaryResolutionCount = PrimaryResolutionCount + 1;
            else
                BFallbackCount = BFallbackCount + 1;
            end
            BaselineWinner = SelectConstraintTournamentWinner( ...
                Candidate(1),Candidate(2),IsFeasible,FrontNo,CrowdDis,BaselineUtilityMeta);
            EffectiveCount = EffectiveCount + double(MatingPool(i) ~= BaselineWinner);
        end
    end
    Diag.two_inf_total = TwoInfCount;
    Diag.effective_total = EffectiveCount;
    Diag.primary_resolution_total = PrimaryResolutionCount;
    Diag.b_fallback_total = BFallbackCount;
    Diag.two_inf_tournament_rate = TwoInfCount / max(K,1);
    if TwoInfCount > 0
        Diag.effective_win_rate = EffectiveCount / TwoInfCount;
        Diag.mlp_primary_resolution_rate = PrimaryResolutionCount / TwoInfCount;
        Diag.b_fallback_rate = BFallbackCount / TwoInfCount;
    end
end

function Winner = SelectConstraintTournamentWinner(A,B,IsFeasible,FrontNo,CrowdDis,UtilityMeta)
    if IsFeasible(A) && ~IsFeasible(B)
        Winner = A;
    elseif ~IsFeasible(A) && IsFeasible(B)
        Winner = B;
    elseif IsFeasible(A) && IsFeasible(B)
        Winner = SelectObjectiveTournamentWinner(A,B,FrontNo,CrowdDis);
    else
        if CompareInfeasibleByUtility(UtilityMeta,A,B) <= 0
            Winner = A;
        else
            Winner = B;
        end
    end
end

function Winner = SelectObjectiveTournamentWinner(A,B,FrontNo,CrowdDis)
    if FrontNo(A) < FrontNo(B)
        Winner = A;
    elseif FrontNo(A) > FrontNo(B)
        Winner = B;
    elseif CrowdDis(A) > CrowdDis(B)
        Winner = A;
    elseif CrowdDis(A) < CrowdDis(B)
        Winner = B;
    else
        Winner = A;
    end
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
        'hidden','epoch','lr','betaB','etaB','Tretrain','Gstart','saveB','useMLP','output_folder'});
    AppendCsvRows(Observer.meta_file,{ ...
        class(Algorithm),class(Problem),familyOfProblem(class(Problem)), ...
        resolveRunId(Algorithm),Problem.M,Problem.D,Problem.N,Problem.maxFE, ...
        Params(1),Params(2),Params(3),Params(4),Params(5),Params(6),Params(7),Params(8),Params(9),RunFolder});

    WriteCsvHeader(Observer.core_file,[ ...
        { ...
        'generation','fe','fe_ratio', ...
        'b_pair_count','b_mean_pair_gap','b_p90_pair_gap','b_mean_mid_scalar','b_p90_mid_scalar', ...
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
        'mlp_due','mlp_can_train','mlp_trained', ...
        'train_size','train_pos','train_neg', ...
        'train_b_core_count','train_refinement_count','train_current_pop_count', ...
        'train_refinement_pos_count','train_refinement_neg_count','train_refinement_ratio', ...
        'mlp_prob_bcore_feasible','mlp_prob_bcore_infeasible', ...
        'mlp_prob_curr_feasible','mlp_prob_curr_infeasible', ...
        'prob_accepted_refinement','prob_rejected_refinement', ...
        'mlp_prob_bcore_gap','mlp_prob_current_gap','prob_refinement_gap'}, ...
        MLPProbabilityHistogramColumnNames(), ...
        { ...
        'mlp_two_inf_tournament_total','mlp_effective_win_total', ...
        'mlp_primary_resolution_total','b_fallback_total', ...
        'mlp_two_inf_tournament_rate','mlp_effective_win_rate', ...
        'mlp_primary_resolution_rate','b_fallback_rate', ...
        'inf_mlp_used','inf_feasible_deficit', ...
        'inf_pool_size','inf_ranked_count','inf_selected', ...
        'inf_pool_mean_utility','inf_selected_mean_utility','inf_utility_gain', ...
        'inf_pool_mean_prob','inf_selected_mean_prob','inf_selected_prob_gain'}]);
end

function Observer = LogGenerationDiagnostics(Observer,Problem,Generation, ...
    B,BInfo,PopulationC,PopulationU,BlendDiag,MLPDiag,MatingDiag,SelectionDiag)
    Active = BInfo.Active;
    BDiag = BuildBoundaryArchiveDiagnostics(B,BInfo,PopulationC,PopulationU,Problem);
    ChangedPairCount = double(BlendDiag.added) + double(BlendDiag.replaced) + ...
        double(StructFieldOr(BlendDiag,'contracted',0));
    ActiveSectors = unique(BInfo.Sector(Active));
    Row = [ ...
        { ...
        Generation,Problem.FE,min(Problem.FE/max(Problem.maxFE,1),1), ...
        BInfo.PairCount,MeanOrNaN(BInfo.PairGap(Active)),PercentileOrNaN(BInfo.PairGap(Active),90), ...
        MeanOrNaN(BInfo.MidScalar(Active)),PercentileOrNaN(BInfo.MidScalar(Active),90), ...
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
        MLPDiag.due,MLPDiag.can_train,MLPDiag.trained, ...
        MLPDiag.train_size,MLPDiag.pos_count,MLPDiag.neg_count, ...
        MLPDiag.b_core_count,MLPDiag.refinement_count,MLPDiag.current_pop_count, ...
        MLPDiag.refinement_pos_count,MLPDiag.refinement_neg_count,MLPDiag.refinement_ratio, ...
        MLPDiag.prob_bcore_feasible,MLPDiag.prob_bcore_infeasible, ...
        MLPDiag.prob_curr_feasible,MLPDiag.prob_curr_infeasible, ...
        MLPDiag.prob_accepted_refinement,MLPDiag.prob_rejected_refinement, ...
        MLPDiag.mlp_prob_bcore_gap,MLPDiag.mlp_prob_current_gap,MLPDiag.prob_refinement_gap}, ...
        MLPProbabilityHistogramRow(MLPDiag), ...
        { ...
        MatingDiag.two_inf_total,MatingDiag.effective_total, ...
        MatingDiag.primary_resolution_total,MatingDiag.b_fallback_total, ...
        MatingDiag.two_inf_tournament_rate,MatingDiag.effective_win_rate, ...
        MatingDiag.mlp_primary_resolution_rate,MatingDiag.b_fallback_rate, ...
        SelectionDiag.mlp_used,SelectionDiag.feasible_deficit, ...
        SelectionDiag.pool_size,SelectionDiag.ranked_count,SelectionDiag.selected, ...
        SelectionDiag.pool_mean_utility,SelectionDiag.selected_mean_utility,SelectionDiag.utility_gain, ...
        SelectionDiag.pool_mean_prob,SelectionDiag.selected_mean_prob,SelectionDiag.selected_prob_gain}];
    AppendCsvRows(Observer.core_file,Row);
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

function Names = MLPProbabilityHistogramColumnNames()
    Names = [ ...
        probabilityHistogramColumns('mlp_prob_train_hist'), ...
        probabilityHistogramColumns('mlp_prob_bcore_feasible_hist'), ...
        probabilityHistogramColumns('mlp_prob_bcore_infeasible_hist'), ...
        probabilityHistogramColumns('mlp_prob_curr_feasible_hist'), ...
        probabilityHistogramColumns('mlp_prob_curr_infeasible_hist')];
end

function Values = MLPProbabilityHistogramRow(Diag)
    Values = [ ...
        num2cell(Diag.prob_train_hist), ...
        num2cell(Diag.prob_bcore_feasible_hist), ...
        num2cell(Diag.prob_bcore_infeasible_hist), ...
        num2cell(Diag.prob_curr_feasible_hist), ...
        num2cell(Diag.prob_curr_infeasible_hist)];
end

function Columns = probabilityHistogramColumns(Prefix)
    switch char(Prefix)
        case 'mlp_prob_train_hist'
            Columns = {'mlp_prob_train_hist_00_10','mlp_prob_train_hist_10_20', ...
                'mlp_prob_train_hist_20_30','mlp_prob_train_hist_30_40', ...
                'mlp_prob_train_hist_40_50','mlp_prob_train_hist_50_60', ...
                'mlp_prob_train_hist_60_70','mlp_prob_train_hist_70_80', ...
                'mlp_prob_train_hist_80_90','mlp_prob_train_hist_90_100'};
        case 'mlp_prob_bcore_feasible_hist'
            Columns = {'mlp_prob_bcore_feasible_hist_00_10','mlp_prob_bcore_feasible_hist_10_20', ...
                'mlp_prob_bcore_feasible_hist_20_30','mlp_prob_bcore_feasible_hist_30_40', ...
                'mlp_prob_bcore_feasible_hist_40_50','mlp_prob_bcore_feasible_hist_50_60', ...
                'mlp_prob_bcore_feasible_hist_60_70','mlp_prob_bcore_feasible_hist_70_80', ...
                'mlp_prob_bcore_feasible_hist_80_90','mlp_prob_bcore_feasible_hist_90_100'};
        case 'mlp_prob_bcore_infeasible_hist'
            Columns = {'mlp_prob_bcore_infeasible_hist_00_10','mlp_prob_bcore_infeasible_hist_10_20', ...
                'mlp_prob_bcore_infeasible_hist_20_30','mlp_prob_bcore_infeasible_hist_30_40', ...
                'mlp_prob_bcore_infeasible_hist_40_50','mlp_prob_bcore_infeasible_hist_50_60', ...
                'mlp_prob_bcore_infeasible_hist_60_70','mlp_prob_bcore_infeasible_hist_70_80', ...
                'mlp_prob_bcore_infeasible_hist_80_90','mlp_prob_bcore_infeasible_hist_90_100'};
        case 'mlp_prob_curr_feasible_hist'
            Columns = {'mlp_prob_curr_feasible_hist_00_10','mlp_prob_curr_feasible_hist_10_20', ...
                'mlp_prob_curr_feasible_hist_20_30','mlp_prob_curr_feasible_hist_30_40', ...
                'mlp_prob_curr_feasible_hist_40_50','mlp_prob_curr_feasible_hist_50_60', ...
                'mlp_prob_curr_feasible_hist_60_70','mlp_prob_curr_feasible_hist_70_80', ...
                'mlp_prob_curr_feasible_hist_80_90','mlp_prob_curr_feasible_hist_90_100'};
        case 'mlp_prob_curr_infeasible_hist'
            Columns = {'mlp_prob_curr_infeasible_hist_00_10','mlp_prob_curr_infeasible_hist_10_20', ...
                'mlp_prob_curr_infeasible_hist_20_30','mlp_prob_curr_infeasible_hist_30_40', ...
                'mlp_prob_curr_infeasible_hist_40_50','mlp_prob_curr_infeasible_hist_50_60', ...
                'mlp_prob_curr_infeasible_hist_60_70','mlp_prob_curr_infeasible_hist_70_80', ...
                'mlp_prob_curr_infeasible_hist_80_90','mlp_prob_curr_infeasible_hist_90_100'};
        otherwise
            error('PRBCCMO:UnknownProbabilityHistogramPrefix', ...
                'Unknown MLP probability histogram prefix: %s',char(Prefix));
    end
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
        'pair_index','pair_side','feasible','source_code','sector','pair_gap','mid_scalar'},ObjHeader];
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
                    SourceCode,BInfo.Sector(p),BInfo.PairGap(p),BInfo.MidScalar(p)};
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
        'b_core_count',Dataset.b_core_count, ...
        'refinement_count',Dataset.refinement_count, ...
        'refinement_pos_count',Dataset.refinement_pos_count, ...
        'refinement_neg_count',Dataset.refinement_neg_count, ...
        'refinement_ratio',Dataset.refinement_ratio, ...
        'current_pop_count',Dataset.current_pop_count, ...
        'prob_bcore_feasible',NaN, ...
        'prob_bcore_infeasible',NaN, ...
        'prob_curr_feasible',NaN, ...
        'prob_curr_infeasible',NaN, ...
        'prob_accepted_refinement',NaN, ...
        'prob_rejected_refinement',NaN, ...
        'mlp_prob_bcore_gap',NaN, ...
        'mlp_prob_current_gap',NaN, ...
        'prob_refinement_gap',NaN, ...
        'prob_train_hist',zeros(1,10), ...
        'prob_bcore_feasible_hist',zeros(1,10), ...
        'prob_bcore_infeasible_hist',zeros(1,10), ...
        'prob_curr_feasible_hist',zeros(1,10), ...
        'prob_curr_infeasible_hist',zeros(1,10), ...
        'pair_count',Dataset.pair_count, ...
        'acc_before',EvaluateBinaryAccuracy(Model,Dataset.Dec,Dataset.Label), ...
        'degraded',false, ...
        'acc_after',NaN);
end

function Diag = InitMatingDiag()
    Diag = struct( ...
        'mlp_used',0, ...
        'two_inf_total',0, ...
        'effective_total',0, ...
        'primary_resolution_total',0, ...
        'b_fallback_total',0, ...
        'two_inf_tournament_rate',0, ...
        'effective_win_rate',NaN, ...
        'mlp_primary_resolution_rate',NaN, ...
        'b_fallback_rate',NaN);
end

function Diag = InitInfeasibleSelectionDiag()
    Diag = struct( ...
        'mlp_used',0, ...
        'feasible_deficit',0, ...
        'pool_size',0, ...
        'ranked_count',0, ...
        'selected',0, ...
        'pool_mean_utility',NaN, ...
        'selected_mean_utility',NaN, ...
        'utility_gain',NaN, ...
        'pool_mean_prob',NaN, ...
        'selected_mean_prob',NaN, ...
        'selected_prob_gain',NaN);
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

function Diag = EvaluateBoundaryMLPProbabilityGroups(Diag,Model,Dataset)
    if isempty(Model) || isempty(Dataset.Dec)
        return;
    end
    Prob = PredictBoundaryMLP(Model,Dataset.Dec);
    BCoreFeasible = Dataset.Source == 1 & Dataset.Label == 1;
    BCoreInfeasible = Dataset.Source == 1 & Dataset.Label == 0;
    CurrFeasible = Dataset.Source == 3 & Dataset.Label == 1;
    CurrInfeasible = Dataset.Source == 3 & Dataset.Label == 0;

    Diag.prob_bcore_feasible = MeanOrNaN(Prob(BCoreFeasible));
    Diag.prob_bcore_infeasible = MeanOrNaN(Prob(BCoreInfeasible));
    Diag.prob_curr_feasible = MeanOrNaN(Prob(CurrFeasible));
    Diag.prob_curr_infeasible = MeanOrNaN(Prob(CurrInfeasible));
    Diag.mlp_prob_bcore_gap = DifferenceOrNaN(Diag.prob_bcore_feasible,Diag.prob_bcore_infeasible);
    Diag.mlp_prob_current_gap = DifferenceOrNaN(Diag.prob_curr_feasible,Diag.prob_curr_infeasible);
    Diag.prob_train_hist = ProbabilityDecileHistogram(Prob);
    Diag.prob_bcore_feasible_hist = ProbabilityDecileHistogram(Prob(BCoreFeasible));
    Diag.prob_bcore_infeasible_hist = ProbabilityDecileHistogram(Prob(BCoreInfeasible));
    Diag.prob_curr_feasible_hist = ProbabilityDecileHistogram(Prob(CurrFeasible));
    Diag.prob_curr_infeasible_hist = ProbabilityDecileHistogram(Prob(CurrInfeasible));
end

function Diag = EvaluateRefinementProbabilityDiag(Diag,Model,ContractionDiag)
    if isempty(Model) || ~isstruct(ContractionDiag)
        return;
    end
    if isfield(ContractionDiag,'accepted_refined') && ~isempty(ContractionDiag.accepted_refined)
        Diag.prob_accepted_refinement = MeanOrNaN( ...
            PredictBoundaryMLP(Model,ContractionDiag.accepted_refined.decs));
    end
    if isfield(ContractionDiag,'rejected_refined') && ~isempty(ContractionDiag.rejected_refined)
        Diag.prob_rejected_refinement = MeanOrNaN( ...
            PredictBoundaryMLP(Model,ContractionDiag.rejected_refined.decs));
    end
    Diag.prob_refinement_gap = DifferenceOrNaN(Diag.prob_accepted_refinement,Diag.prob_rejected_refinement);
end

function Hist = ProbabilityDecileHistogram(Prob)
    Prob = double(Prob(:));
    Prob = Prob(isfinite(Prob));
    if isempty(Prob)
        Hist = zeros(1,10);
        return;
    end
    Prob = min(max(Prob,0),1);
    Hist = histcounts(Prob,0:0.1:1);
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

function Value = DifferenceOrNaN(A,B)
    A = double(A);
    B = double(B);
    if isscalar(A) && isscalar(B) && isfinite(A) && isfinite(B)
        Value = A - B;
    else
        Value = NaN;
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
