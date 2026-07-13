function test_CBS_region_bmem_coherent()
%TEST_CBS_REGION_BMEM_COHERENT Verify the coherent BMem semantics bundle.

    repoRoot = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
    addpath(genpath(repoRoot));
    rng(1,'twister');

    testPairingPrecedesReferenceCap();
    testPreviousAnchorHasStrictOneRefreshTTL();
    testPreviousAnchorCannotChangeCurrentNormalization();
    testDominatingFeasiblePairIsRetained();
    fprintf('CBS RegionGAN coherent-BMem regressions passed.\n');
end

function testPairingPrecedesReferenceCap()
    Problem = DASCMOP1_BC('N',8,'D',6,'maxFE',100);
    W = [0.5 0.5];
    Yf = [1 9;5 5;9 1];
    Yi = [1.5 9.5;5.05 5.05;9.8 1.8];
    Pop = makePopulation(Problem,Yf,Yi);
    Empty = Pop([]);
    Options = coherentOptions(1);

    BMem = UpdateBoundaryMemory_RC([],Pop,Empty,Empty,Empty,W,Options);

    assert(size(BMem.y_f,1) == 1, ...
        'Coherent BMem must apply the per-ref cap after pairing.');
    assert(all(abs(BMem.y_f - [5 5]) <= 1e-12), ...
        'Pair-before-cap must retain the feasible anchor with the best gap.');
end

function testPreviousAnchorHasStrictOneRefreshTTL()
    Problem = DASCMOP1_BC('N',8,'D',6,'maxFE',100);
    W = [0.5 0.5];
    Xi = makeDecisionRows(Problem,1);
    Current = SOLUTION(Xi,[2 2],1);
    Empty = Current([]);
    Prev = previousMemory(Problem,[1 1],0);
    Options = coherentOptions(5);

    BMem1 = UpdateBoundaryMemory_RC(Prev,Current,Empty,Empty,Empty,W,Options);
    BMem2 = UpdateBoundaryMemory_RC(BMem1,Current,Empty,Empty,Empty,W,Options);

    assert(size(BMem1.y_f,1) == 1 && BMem1.age_f == 1, ...
        'An age-zero previous anchor must be eligible for exactly one refresh.');
    assert(isempty(BMem2.y_f), ...
        'An age-one previous anchor must expire instead of recursively surviving.');
end

function testPreviousAnchorCannotChangeCurrentNormalization()
    Problem = DASCMOP1_BC('N',8,'D',6,'maxFE',100);
    W = [1 0;0 1];
    Yf = [1 4;4 1];
    Yi = [2 5;5 2];
    Pop = makePopulation(Problem,Yf,Yi);
    Empty = Pop([]);
    Options = coherentOptions(10);
    Prev = previousMemory(Problem,[1000 1000],0);

    Base = UpdateBoundaryMemory_RC([],Pop,Empty,Empty,Empty,W,Options);
    WithPrev = UpdateBoundaryMemory_RC( ...
        Prev,Pop,Empty,Empty,Empty,W,Options);
    CurrentRows = WithPrev.source_f == 0;

    BaseRows = sortrows([Base.ref,Base.y_f,Base.gap],[1 2 3 4]);
    Current = sortrows([WithPrev.ref(CurrentRows), ...
        WithPrev.y_f(CurrentRows,:),WithPrev.gap(CurrentRows)],[1 2 3 4]);
    assert(isequal(size(BaseRows),size(Current)) && ...
            max(abs(BaseRows - Current),[],'all') <= 1e-12, ...
        'Previous anchors must not alter current-row normalization, refs, or gaps.');
end

function testDominatingFeasiblePairIsRetained()
    Problem = DASCMOP1_BC('N',8,'D',6,'maxFE',100);
    W = [0.5 0.5];
    Pop = makePopulation(Problem,[1 1],[2 2]);
    Empty = Pop([]);

    Coherent = UpdateBoundaryMemory_RC([],Pop,Empty,Empty,Empty,W, ...
        coherentOptions(5));
    LegacyOptions = coherentOptions(5);
    LegacyOptions.bmemMode = "legacy";
    Legacy = UpdateBoundaryMemory_RC([],Pop,Empty,Empty,Empty,W, ...
        LegacyOptions);

    assert(size(Coherent.y_f,1) == 1, ...
        'Coherent BMem must retain a valid local pair despite objective dominance.');
    assert(isempty(Legacy.y_f), ...
        'Legacy control must preserve its objective-dominance skip.');
end

function Options = coherentOptions(maxAnchorsPerRef)
    Options = struct( ...
        'frontDepth',2, ...
        'pairNeighborRefRadius',0, ...
        'minBoundaryLength',100, ...
        'maxAnchorsPerRef',maxAnchorsPerRef, ...
        'bmemMode',"coherent");
end

function Pop = makePopulation(Problem,Yf,Yi)
    Xf = makeDecisionRows(Problem,size(Yf,1));
    Xi = makeDecisionRows(Problem,size(Yi,1));
    Pop = SOLUTION([Xf;Xi],[Yf;Yi], ...
        [zeros(size(Yf,1),1);ones(size(Yi,1),1)]);
end

function Prev = previousMemory(Problem,Y,age)
    X = makeDecisionRows(Problem,1);
    Prev = struct( ...
        'ref',1, ...
        'y_b',Y, ...
        'gap',0.01, ...
        'x_b',X, ...
        'x_f',X, ...
        'y_f',Y, ...
        'x_i',X, ...
        'y_i',Y + 0.1, ...
        'source_f',1, ...
        'age_f',age, ...
        'front_rank_f',1, ...
        'candidate_row_f',1, ...
        'candidate_row_i',1, ...
        'pair_normal',zeros(1,size(Y,2)));
end

function X = makeDecisionRows(Problem,n)
    lower = double(Problem.lower);
    upper = double(Problem.upper);
    X = repmat(lower,n,1) + rand(n,Problem.D).*repmat(upper-lower,n,1);
end
