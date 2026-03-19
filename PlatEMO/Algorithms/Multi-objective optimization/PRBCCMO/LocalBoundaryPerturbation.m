function [Decs,ProxyObjs,SeedIdx] = LocalBoundaryPerturbation(Problem,Seeds,Anchors,N,W,IsFeasibleSeed,RuntimeOptions)
% Generate local boundary candidates around archived seeds.

    Decs      = zeros(0,Problem.D);
    ProxyObjs = zeros(0,Problem.M);
    SeedIdx   = zeros(0,1);
    if N <= 0 || isempty(Seeds)
        return;
    end
    if nargin < 5
        W = [];
    end
    if nargin < 6
        IsFeasibleSeed = true;
    end
    if nargin < 7 || ~isstruct(RuntimeOptions)
        RuntimeOptions = struct();
    end

    SeedIdx   = randi(numel(Seeds),1,N);
    Opposite  = [];
    if ~isempty(Anchors)
        Opposite = MatchPartnersBySector(Seeds(SeedIdx).objs,Anchors.objs,W);
    end

    Decs      = zeros(N,Problem.D);
    ProxyObjs = zeros(N,Problem.M);
    for i = 1 : N
        Seed = Seeds(SeedIdx(i));
        if isempty(Opposite)
            Anchor = [];
        else
            Anchor = Anchors(Opposite(i));
        end
        if UseIsotropicLocal(RuntimeOptions)
            if IsFeasibleSeed
                Decs(i,:) = GenerateIsotropicNeighbor(Problem,Seed,0.12);
                ProxyObjs(i,:) = Seed.objs;
            else
                Decs(i,:) = GenerateIsotropicNeighbor(Problem,Seed,0.08);
                ProxyObjs(i,:) = InferProxyObjective(Seed,Anchor);
            end
        else
            if IsFeasibleSeed
                Decs(i,:) = ExpandFromFeasibleSeed(Problem,Seed,Anchor);
                ProxyObjs(i,:) = Seed.objs;
            else
                Decs(i,:) = SearchFromInfeasibleSeed(Problem,Seed,Anchor);
                ProxyObjs(i,:) = InferProxyObjective(Seed,Anchor);
            end
        end
    end
end

function Flag = UseIsotropicLocal(RuntimeOptions)
    Flag = false;
    if isstruct(RuntimeOptions) && isfield(RuntimeOptions,'LocalMode') && ~isempty(RuntimeOptions.LocalMode)
        Flag = round(RuntimeOptions.LocalMode) == 2;
    end
end

function Dec = ExpandFromFeasibleSeed(Problem,Seed,Anchor)
    Dec = BaseLabelAwareChild(Problem,Seed,Anchor);
    Idx = find(Problem.encoding<=2);
    if isempty(Idx)
        return;
    end
    if isempty(Anchor)
        Dec(Idx) = Dec(Idx) + 0.05*randn(1,numel(Idx)).*ScaleRange(Problem,Idx);
    else
        [Dir,Ortho,Range] = BuildDirectionFrame(Problem,Seed.dec(Idx),Anchor.dec(Idx),Idx);
        Step = (0.10 + 0.20*rand).*Dir + (0.03 + 0.08*rand).*Ortho;
        Dec(Idx) = Seed.dec(Idx) + Step.*Range;
    end
    Dec = RepairDecision(Problem,Dec);
end

function Dec = SearchFromInfeasibleSeed(Problem,Seed,Anchor)
    Dec = BaseLabelAwareChild(Problem,Seed,Anchor);
    Idx = find(Problem.encoding<=2);
    if isempty(Idx)
        return;
    end
    Range = ScaleRange(Problem,Idx);
    if isempty(Anchor)
        Dec(Idx) = Seed.dec(Idx) + (0.02 + 0.06*rand).*randn(1,numel(Idx)).*Range;
    else
        [~,Ortho,Range] = BuildDirectionFrame(Problem,Seed.dec(Idx),Anchor.dec(Idx),Idx);
        Lambda  = 0.35 + 0.30*rand;
        Midpoint = Anchor.dec(Idx) + Lambda*(Seed.dec(Idx)-Anchor.dec(Idx));
        Dec(Idx) = Midpoint + (0.01 + 0.04*rand).*Ortho.*Range;
    end
    Dec = RepairDecision(Problem,Dec);
end

function Dec = BaseLabelAwareChild(Problem,Seed,Anchor)
    if isempty(Anchor)
        Parent = [Seed.dec;Seed.dec];
        Dec    = OperatorGAhalf(Problem,Parent,{0,20,1,20});
    else
        Parent = [Seed.dec;Anchor.dec];
        Dec    = OperatorGAhalf(Problem,Parent,{1,20,1,20});
    end
    Dec = Dec(1,:);
end

function Proxy = InferProxyObjective(Seed,Anchor)
    if isempty(Anchor)
        Proxy = Seed.objs;
    else
        Proxy = 0.5*(Seed.objs + Anchor.objs);
    end
end

function [Dir,Ortho,Range] = BuildDirectionFrame(Problem,SeedDec,AnchorDec,Idx)
    Range = ScaleRange(Problem,Idx);
    Dir   = (SeedDec-AnchorDec)./Range;
    if norm(Dir) < 1e-12
        Dir = randn(1,numel(Idx));
    end
    Dir = Dir./max(norm(Dir),1e-12);

    Ortho = randn(1,numel(Idx));
    Ortho = Ortho - (Ortho*Dir')*Dir;
    if norm(Ortho) < 1e-12
        Ortho = circshift(Dir,1);
        Ortho = Ortho - (Ortho*Dir')*Dir;
    end
    Ortho = Ortho./max(norm(Ortho),1e-12);
end

function Range = ScaleRange(Problem,Idx)
    Range = Problem.upper(Idx) - Problem.lower(Idx);
    Range(Range<1e-12) = 1;
end

function Dec = GenerateIsotropicNeighbor(Problem,Seed,StepScale)
    Dec = BaseLabelAwareChild(Problem,Seed,[]);
    RealIdx = find(Problem.encoding<=2);
    if ~isempty(RealIdx)
        Range = ScaleRange(Problem,RealIdx);
        Noise = randn(1,numel(RealIdx));
        Noise = Noise./max(norm(Noise),1e-12);
        Dec(RealIdx) = Seed.dec(RealIdx) + StepScale*Noise.*Range;
    end
    Dec = RepairDecision(Problem,Dec);
end

function Dec = RepairDecision(Problem,Dec)
    Dec = Problem.CalDec(Dec);
end
