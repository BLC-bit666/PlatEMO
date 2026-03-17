function Pool = GenerateBoundaryCandidates(Problem,PopulationC,PopulationU,ArchiveF,ArchiveI,FitnessC,FitnessU,type,NumCross,NumLocal,NumMate)
% Generate the unevaluated boundary candidate pool.

    Pool.decs      = zeros(0,Problem.D);
    Pool.proxyObjs = zeros(0,Problem.M);
    Pool.source    = zeros(0,1);

    [Decs,Proxy] = GenerateCrossCandidates(Problem,PopulationC,PopulationU,FitnessC,FitnessU,type,NumCross);
    Pool = AppendPool(Pool,Decs,Proxy,1);

    [Decs,Proxy] = LocalBoundaryPerturbation(Problem,[ArchiveF,ArchiveI],NumLocal);
    Pool = AppendPool(Pool,Decs,Proxy,2);

    [Decs,Proxy] = GenerateBoundaryMatingCandidates(Problem,ArchiveI,PopulationC,FitnessC,type,NumMate);
    Pool = AppendPool(Pool,Decs,Proxy,3);
end

function Pool = AppendPool(Pool,Decs,ProxyObjs,Source)
    if isempty(Decs)
        return;
    end
    Pool.decs      = [Pool.decs;Decs];
    Pool.proxyObjs = [Pool.proxyObjs;ProxyObjs];
    Pool.source    = [Pool.source;repmat(Source,size(Decs,1),1)];
end

function [Decs,ProxyObjs] = GenerateCrossCandidates(Problem,PopulationC,PopulationU,FitnessC,FitnessU,type,N)
    Decs      = zeros(0,Problem.D);
    ProxyObjs = zeros(0,Problem.M);
    if N <= 0 || isempty(PopulationC) || isempty(PopulationU)
        return;
    end

    if type == 1
        ParentC   = TournamentSelection(2,N,FitnessC);
        ParentU   = TournamentSelection(2,N,FitnessU);
        ParentDec = [PopulationC(ParentC).decs;PopulationU(ParentU).decs];
        Decs      = OperatorGAhalf(Problem,ParentDec);
        ProxyObjs = 0.5*(PopulationC(ParentC).objs + PopulationU(ParentU).objs);
    else
        Base      = TournamentSelection(2,N,FitnessC);
        Donor1    = TournamentSelection(2,N,FitnessU);
        Donor2    = TournamentSelection(2,N,FitnessU);
        Decs      = OperatorDE(Problem,PopulationC(Base).decs, ...
            PopulationU(Donor1).decs,PopulationU(Donor2).decs);
        ProxyObjs = (PopulationC(Base).objs + PopulationU(Donor1).objs + PopulationU(Donor2).objs)/3;
    end
end

function [Decs,ProxyObjs] = GenerateBoundaryMatingCandidates(Problem,ArchiveI,PopulationC,FitnessC,type,N)
% Generate A_I x P_C candidates for the next constrained search step.

    Decs      = zeros(0,Problem.D);
    ProxyObjs = zeros(0,Problem.M);
    if N <= 0 || isempty(ArchiveI) || isempty(PopulationC)
        return;
    end

    PickI = randi(numel(ArchiveI),1,N);
    if type == 1
        PickC     = TournamentSelection(2,N,FitnessC);
        ParentDec = [ArchiveI(PickI).decs;PopulationC(PickC).decs];
        Decs      = OperatorGAhalf(Problem,ParentDec);
        ProxyObjs = 0.5*(ArchiveI(PickI).objs + PopulationC(PickC).objs);
    else
        PickC1    = TournamentSelection(2,N,FitnessC);
        PickC2    = TournamentSelection(2,N,FitnessC);
        Decs      = OperatorDE(Problem,ArchiveI(PickI).decs, ...
            PopulationC(PickC1).decs,PopulationC(PickC2).decs);
        ProxyObjs = (ArchiveI(PickI).objs + PopulationC(PickC1).objs + PopulationC(PickC2).objs)/3;
    end
end
