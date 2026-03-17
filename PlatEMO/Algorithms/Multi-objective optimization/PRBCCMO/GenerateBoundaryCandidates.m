function Pool = GenerateBoundaryCandidates(Problem,PopulationC,PopulationU,ArchiveF,ArchiveI,FitnessC,FitnessU,type,W,NumCross,NumLocal,NumMate)
% Generate the unevaluated boundary candidate pool.

    Pool.decs      = zeros(0,Problem.D);
    Pool.proxyObjs = zeros(0,Problem.M);
    Pool.source    = zeros(0,1);

    [Decs,Proxy] = GenerateCrossCandidates(Problem,PopulationC,PopulationU,FitnessC,FitnessU,type,W,NumCross);
    Pool = AppendPool(Pool,Decs,Proxy,1);

    [NumLocalF,NumLocalI] = SplitLocalBudget(NumLocal,numel(ArchiveF),numel(ArchiveI));
    [Decs,Proxy] = LocalBoundaryPerturbation(Problem,ArchiveF,NumLocalF);
    Pool = AppendPool(Pool,Decs,Proxy,2);

    [Decs,Proxy] = LocalBoundaryPerturbation(Problem,ArchiveI,NumLocalI);
    Pool = AppendPool(Pool,Decs,Proxy,3);

    [Decs,Proxy] = GenerateBoundaryMatingCandidates(Problem,ArchiveI,PopulationC,FitnessC,type,W,NumMate);
    Pool = AppendPool(Pool,Decs,Proxy,4);
end

function Pool = AppendPool(Pool,Decs,ProxyObjs,Source)
    if isempty(Decs)
        return;
    end
    Pool.decs      = [Pool.decs;Decs];
    Pool.proxyObjs = [Pool.proxyObjs;ProxyObjs];
    Pool.source    = [Pool.source;repmat(Source,size(Decs,1),1)];
end

function [Decs,ProxyObjs] = GenerateCrossCandidates(Problem,PopulationC,PopulationU,FitnessC,~,type,W,N)
    Decs      = zeros(0,Problem.D);
    ProxyObjs = zeros(0,Problem.M);
    if N <= 0 || isempty(PopulationC) || isempty(PopulationU)
        return;
    end

    if type == 1
        ParentC   = TournamentSelection(2,N,FitnessC);
        ParentU   = MatchPartnersBySector(PopulationC(ParentC).objs,PopulationU.objs,W);
        ParentDec = [PopulationC(ParentC).decs;PopulationU(ParentU).decs];
        Decs      = OperatorGAhalf(Problem,ParentDec);
        ProxyObjs = 0.5*(PopulationC(ParentC).objs + PopulationU(ParentU).objs);
    else
        Base      = TournamentSelection(2,N,FitnessC);
        BaseObj    = PopulationC(Base).objs;
        Donor1    = MatchPartnersBySector(BaseObj,PopulationU.objs,W);
        Donor2    = MatchPartnersBySector(BaseObj,PopulationU.objs,W,Donor1);
        Decs      = OperatorDE(Problem,PopulationC(Base).decs, ...
            PopulationU(Donor1).decs,PopulationU(Donor2).decs);
        ProxyObjs = (PopulationC(Base).objs + PopulationU(Donor1).objs + PopulationU(Donor2).objs)/3;
    end
end

function [Decs,ProxyObjs] = GenerateBoundaryMatingCandidates(Problem,ArchiveI,PopulationC,~,type,W,N)
% Generate A_I x P_C candidates for the next constrained search step.

    Decs      = zeros(0,Problem.D);
    ProxyObjs = zeros(0,Problem.M);
    if N <= 0 || isempty(ArchiveI) || isempty(PopulationC)
        return;
    end

    PickI = randi(numel(ArchiveI),1,N);
    if type == 1
        PickC     = MatchPartnersBySector(ArchiveI(PickI).objs,PopulationC.objs,W);
        ParentDec = [ArchiveI(PickI).decs;PopulationC(PickC).decs];
        Decs      = OperatorGAhalf(Problem,ParentDec);
        ProxyObjs = 0.5*(ArchiveI(PickI).objs + PopulationC(PickC).objs);
    else
        SeedObj    = ArchiveI(PickI).objs;
        PickC1    = MatchPartnersBySector(SeedObj,PopulationC.objs,W);
        PickC2    = MatchPartnersBySector(SeedObj,PopulationC.objs,W,PickC1);
        Decs      = OperatorDE(Problem,ArchiveI(PickI).decs, ...
            PopulationC(PickC1).decs,PopulationC(PickC2).decs);
        ProxyObjs = (ArchiveI(PickI).objs + PopulationC(PickC1).objs + PopulationC(PickC2).objs)/3;
    end
end

function [NumLocalF,NumLocalI] = SplitLocalBudget(NumLocal,NumArchiveF,NumArchiveI)
    NumLocalF = 0;
    NumLocalI = 0;
    if NumLocal <= 0
        return;
    end

    if NumArchiveF == 0 && NumArchiveI == 0
        return;
    elseif NumArchiveF == 0
        NumLocalI = NumLocal;
        return;
    elseif NumArchiveI == 0
        NumLocalF = NumLocal;
        return;
    end

    if NumLocal == 1
        if NumArchiveF >= NumArchiveI
            NumLocalF = 1;
        else
            NumLocalI = 1;
        end
        return;
    end

    RatioF    = NumArchiveF/(NumArchiveF+NumArchiveI);
    NumLocalF = round(NumLocal*RatioF);
    NumLocalF = min(max(1,NumLocalF),NumLocal-1);
    NumLocalI = NumLocal - NumLocalF;
end

function Match = MatchPartnersBySector(AnchorObj,CandidateObj,W,Exclude)
    N = size(AnchorObj,1);
    Match = zeros(1,N);
    if N == 0 || isempty(CandidateObj)
        Match = zeros(1,0);
        return;
    end
    if nargin < 4 || isempty(Exclude)
        Exclude = zeros(N,1);
    else
        Exclude = Exclude(:);
    end

    RefObj = [AnchorObj;CandidateObj];
    SectorA = AssociateSectors(AnchorObj,W,RefObj);
    SectorC = AssociateSectors(CandidateObj,W,RefObj);
    MinObj  = min(RefObj,[],1);
    Range   = max(RefObj,[],1) - MinObj;
    Range(Range<1e-12) = 1;

    for i = 1 : N
        SameSector = find(SectorC==SectorA(i));
        SameSector = RemoveExcluded(SameSector,Exclude(i));
        if isempty(SameSector)
            SameSector = RemoveExcluded((1:size(CandidateObj,1))',Exclude(i));
        end
        if isempty(SameSector)
            SameSector = Exclude(i);
        end
        AnchorNorm = (AnchorObj(i,:)-MinObj)./Range;
        CandNorm   = (CandidateObj(SameSector,:)-MinObj)./Range;
        Dist       = sum((CandNorm-AnchorNorm).^2,2);
        [~,Best]   = min(Dist);
        Match(i)   = SameSector(Best);
    end
end

function Pool = RemoveExcluded(Pool,Exclude)
    if isempty(Pool) || Exclude <= 0 || numel(Pool) <= 1
        return;
    end
    Pool = Pool(Pool~=Exclude);
end
