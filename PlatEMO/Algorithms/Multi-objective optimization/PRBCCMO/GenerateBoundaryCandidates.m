function Pool = GenerateBoundaryCandidates(Problem,PopulationC,PopulationU,ArchiveF,ArchiveI,type,W,NumBridge,NumLocal,RuntimeOptions)
% Generate the unevaluated boundary candidate pool from bridge and local sources.

    if nargin < 10 || ~isstruct(RuntimeOptions)
        RuntimeOptions = struct();
    end
    Pool.decs      = zeros(0,Problem.D);
    Pool.proxyObjs = zeros(0,Problem.M);
    Pool.source    = zeros(0,1);

    FeasibleC   = PopulationC(all(PopulationC.cons<=0,2));
    InfeasibleU = PopulationU(~all(PopulationU.cons<=0,2));
    FeasiblePool   = [FeasibleC,ArchiveF];
    InfeasiblePool = [InfeasibleU,ArchiveI];

    [Decs,Proxy] = GenerateBridgeCandidates(Problem,FeasiblePool,InfeasiblePool,type,W,NumBridge);
    Pool = AppendPool(Pool,Decs,Proxy,1);

    [Decs,Proxy] = GenerateLocalCandidates( ...
        Problem,ArchiveF,ArchiveI,FeasiblePool,InfeasiblePool,W,NumLocal,RuntimeOptions);
    Pool = AppendPool(Pool,Decs,Proxy,2);
end

function Pool = AppendPool(Pool,Decs,ProxyObjs,Source)
    if isempty(Decs)
        return;
    end
    Pool.decs      = [Pool.decs;Decs];
    Pool.proxyObjs = [Pool.proxyObjs;ProxyObjs];
    Pool.source    = [Pool.source;repmat(Source,size(Decs,1),1)];
end

function [Decs,ProxyObjs] = GenerateBridgeCandidates(Problem,FeasiblePool,InfeasiblePool,type,W,N)
    Decs      = zeros(0,Problem.D);
    ProxyObjs = zeros(0,Problem.M);
    if N <= 0 || isempty(FeasiblePool) || isempty(InfeasiblePool)
        return;
    end

    FitnessF = CalFitness(FeasiblePool.objs);
    if type == 1
        ParentF   = TournamentSelection(2,N,FitnessF);
        ParentI   = MatchPartnersBySector(FeasiblePool(ParentF).objs,InfeasiblePool.objs,W);
        ParentDec = [FeasiblePool(ParentF).decs;InfeasiblePool(ParentI).decs];
        Decs      = OperatorGAhalf(Problem,ParentDec);
        ProxyObjs = 0.5*(FeasiblePool(ParentF).objs + InfeasiblePool(ParentI).objs);
    else
        Base      = TournamentSelection(2,N,FitnessF);
        BaseObj   = FeasiblePool(Base).objs;
        Donor1    = MatchPartnersBySector(BaseObj,InfeasiblePool.objs,W);
        Donor2    = MatchPartnersBySector(BaseObj,InfeasiblePool.objs,W,Donor1);
        Decs      = OperatorDE(Problem,FeasiblePool(Base).decs, ...
            InfeasiblePool(Donor1).decs,InfeasiblePool(Donor2).decs);
        ProxyObjs = (FeasiblePool(Base).objs + InfeasiblePool(Donor1).objs + InfeasiblePool(Donor2).objs)/3;
    end
end

function [Decs,ProxyObjs] = GenerateLocalCandidates(Problem,ArchiveF,ArchiveI,FeasiblePool,InfeasiblePool,W,N,RuntimeOptions)
    Decs      = zeros(0,Problem.D);
    ProxyObjs = zeros(0,Problem.M);
    if N <= 0 || (isempty(ArchiveF) && isempty(ArchiveI))
        return;
    end

    [NumLocalF,NumLocalI] = SplitLocalBudget(N,numel(ArchiveF),numel(ArchiveI));
    [DecF,ProxyF] = LocalBoundaryPerturbation(Problem,ArchiveF,InfeasiblePool,NumLocalF,W,true,RuntimeOptions);
    [DecI,ProxyI] = LocalBoundaryPerturbation(Problem,ArchiveI,FeasiblePool,NumLocalI,W,false,RuntimeOptions);
    Decs      = [DecF;DecI];
    ProxyObjs = [ProxyF;ProxyI];
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
