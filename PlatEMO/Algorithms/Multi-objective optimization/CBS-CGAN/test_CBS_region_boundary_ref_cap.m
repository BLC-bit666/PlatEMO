function test_CBS_region_boundary_ref_cap()
%TEST_CBS_REGION_BOUNDARY_REF_CAP Verify RegionGAN boundary data gates.

    repoRoot = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
    addpath(genpath(repoRoot));

    testBoundaryMemoryCapsFeasibleAnchorsPerRefBeforePairing();
    testBoundaryMemoryPFSeedBandKeepsLocalAnchors();
    testBoundaryMemoryPFSeedBandMinCoverSupplementsAnchors();
    testPreviousBoundaryMemoryCompetesFairlyWhenEnabled();
    testRegionRunnerSkipsSmallTrainingSet();
    fprintf('CBS RegionGAN boundary ref-cap regressions passed.\n');
end

function testBoundaryMemoryCapsFeasibleAnchorsPerRefBeforePairing()
    Problem = DASCMOP1_BC('N',8,'D',6,'maxFE',100);
    W = [0.5 0.5];
    nFeasible = 10;
    Xf = makeDecisionRows(Problem,nFeasible);
    Yf = [(1:nFeasible)' (nFeasible:-1:1)'];
    Cf = zeros(nFeasible,1);
    Xi = makeDecisionRows(Problem,1);
    Yi = [0 0];
    Ci = 1;
    Pop = SOLUTION([Xf;Xi],[Yf;Yi],[Cf;Ci]);
    Empty = Pop([]);
    Options = struct( ...
        'frontDepth',2, ...
        'pairNeighborRefRadius',0, ...
        'minBoundaryLength',100, ...
        'maxAnchorsPerRef',5);

    [BMem,Diag] = UpdateBoundaryMemory_RC([],Pop,Empty,Empty,Empty,W,Options);

    Fitness = CalFitness_CBS(Yf,Cf);
    [~,ord] = sort(Fitness,'ascend');
    expected = sortrows(Yf(ord(1:5),:));
    actual = sortrows(BMem.y_b);
    assert(size(BMem.y_b,1) == 5, ...
        'Boundary memory must keep at most five feasible anchors per ref.');
    assert(isequal(actual,expected), ...
        'Per-ref anchor cap must keep the best feasible anchors by fitness.');
    assert(Diag.bmem_count == 5, ...
        'Boundary diagnostics must report the capped memory size.');
end

function testBoundaryMemoryPFSeedBandKeepsLocalAnchors()
    Problem = DASCMOP1_BC('N',8,'D',6,'maxFE',100);
    W = [0.5 0.5];
    nFeasible = 10;
    Xf = makeDecisionRows(Problem,nFeasible);
    Yf = [(1:nFeasible)' (nFeasible:-1:1)'];
    Cf = zeros(nFeasible,1);
    Xi = makeDecisionRows(Problem,1);
    Yi = [0 0];
    Ci = 1;
    Pop = SOLUTION([Xf;Xi],[Yf;Yi],[Cf;Ci]);
    Empty = Pop([]);
    Options = struct( ...
        'frontDepth',2, ...
        'pairNeighborRefRadius',0, ...
        'minBoundaryLength',100, ...
        'maxAnchorsPerRef',5, ...
        'bmemBandMode',"pfseed_band", ...
        'bandMaxAnchorsPerRef',3);

    [BMem,Diag] = UpdateBoundaryMemory_RC([],Pop,Empty,Empty,Empty,W,Options);

    expected = expectedPFSeedBand(Yf,Yi,5,3);
    actual = sortrows(BMem.y_b);
    assert(size(BMem.y_b,1) == 3, ...
        'PF-seed band mode must keep only the local seed band per ref.');
    assert(isequal(actual,expected), ...
        'PF-seed band mode must keep the best anchor and nearest objective neighbours.');
    assert(Diag.bmem_count == 3, ...
        'Boundary diagnostics must report the PF-seed local-band size.');
end

function testBoundaryMemoryPFSeedBandMinCoverSupplementsAnchors()
    Problem = DASCMOP1_BC('N',8,'D',6,'maxFE',100);
    W = [0.5 0.5];
    nFeasible = 10;
    Xf = makeDecisionRows(Problem,nFeasible);
    Yf = [(1:nFeasible)' (nFeasible:-1:1)'];
    Cf = zeros(nFeasible,1);
    Xi = makeDecisionRows(Problem,1);
    Yi = [0 0];
    Ci = 1;
    Pop = SOLUTION([Xf;Xi],[Yf;Yi],[Cf;Ci]);
    Empty = Pop([]);
    Options = struct( ...
        'frontDepth',2, ...
        'pairNeighborRefRadius',0, ...
        'minBoundaryLength',5, ...
        'maxAnchorsPerRef',6, ...
        'bmemBandMode',"pfseed_band_mincover", ...
        'bandMaxAnchorsPerRef',2, ...
        'minGANTrainCount',5);

    [BMem,Diag] = UpdateBoundaryMemory_RC([],Pop,Empty,Empty,Empty,W,Options);

    localBand = expectedPFSeedBand(Yf,Yi,6,2);
    assert(size(BMem.y_b,1) == 5, ...
        'PF-seed min-cover mode must supplement local bands up to the GAN train gate.');
    assert(all(ismember(localBand,BMem.y_b,'rows')), ...
        'PF-seed min-cover mode must retain the local seed band while supplementing.');
    assert(Diag.bmem_count == 5, ...
        'Boundary diagnostics must report the supplemented BMem size.');
end

function testPreviousBoundaryMemoryCompetesFairlyWhenEnabled()
    Problem = DASCMOP1_BC('N',8,'D',6,'maxFE',100);
    W = [0.5 0.5];
    Xf = makeDecisionRows(Problem,1);
    Yf = [10 10];
    Cf = 0;
    Xi = makeDecisionRows(Problem,1);
    Yi = [0 0];
    Ci = 1;
    Pop = SOLUTION([Xf;Xi],[Yf;Yi],[Cf;Ci]);
    Empty = Pop([]);
    PrevBMem = struct( ...
        'ref',1, ...
        'y_b',[1 1], ...
        'gap',0.01, ...
        'x_b',makeDecisionRows(Problem,1), ...
        'x_f',makeDecisionRows(Problem,1), ...
        'y_f',[1 1], ...
        'x_i',makeDecisionRows(Problem,1), ...
        'y_i',[0.9 0.9]);
    Options = struct( ...
        'frontDepth',2, ...
        'pairNeighborRefRadius',0, ...
        'minBoundaryLength',100, ...
        'maxAnchorsPerRef',1, ...
        'prevBMemMode',"prev1_fair_union");

    [BMem,Diag] = UpdateBoundaryMemory_RC(PrevBMem,Pop,Empty,Empty,Empty, ...
        W,Options);

    assert(size(BMem.y_b,1) == 1, ...
        'Fair previous-memory union must respect the per-ref cap.');
    assert(isequal(BMem.y_b,[1 1]), ...
        'Previous BMem candidates must compete by the same quality gate, not be ignored.');
    assert(isequal(BMem.y_i,[0 0]), ...
        'Previous feasible anchors must be re-paired with current infeasible samples.');
    assert(Diag.prev_bmem_candidate_count == 1 && ...
            Diag.prev_bmem_survivor_count == 1, ...
        'Diagnostics must report previous-memory candidates and survivors.');
end

function testRegionRunnerSkipsSmallTrainingSet()
    Problem = DASCMOP1_BC('N',8,'D',6,'maxFE',100);
    TrainX = makeDecisionRows(Problem,31);
    TrainC = rand(31,2);
    QueryC = rand(2,2);
    Options = struct( ...
        'zDim',6, ...
        'iter',1, ...
        'miniBatch',8, ...
        'lrD',1e-4, ...
        'lrG',1e-4, ...
        'minTrainCount',32, ...
        'generatorHidden',[32 32], ...
        'discriminatorHidden',[32 32]);

    [GAN,RawDec] = RunRegionGAN_RC('trainandsample',"cgan",[], ...
        TrainX,TrainC,QueryC,1,Problem,Options);

    assert(isempty(GAN), ...
        'Runner must not train CGAN when TrainX has fewer than minTrainCount rows.');
    assert(isempty(RawDec) && size(RawDec,2) == Problem.D, ...
        'Runner must not generate decisions when training data is too small.');
end

function X = makeDecisionRows(Problem,n)
    lower = double(Problem.lower);
    upper = double(Problem.upper);
    X = repmat(lower,n,1) + rand(n,Problem.D).*repmat(upper-lower,n,1);
end

function Expected = expectedPFSeedBand(Yf,Yi,maxPerRef,bandMax)
    Fitness = CalFitness_CBS(Yf,zeros(size(Yf,1),1));
    [~,fitOrd] = sort(Fitness,'ascend');
    capped = fitOrd(1:maxPerRef);
    Yall = [Yf;Yi];
    minY = min(Yall,[],1);
    spanY = max(Yall,[],1) - minY;
    spanY(spanY <= eps) = 1;
    Yn = (Yall - minY)./spanY;
    cappedYn = Yn(capped,:);
    cappedFitness = Fitness(capped);
    [~,seedLocal] = min(cappedFitness);
    seedYn = cappedYn(seedLocal,:);
    dist = sqrt(sum((cappedYn - seedYn).^2,2));
    order = sortrows([(1:numel(capped))' dist(:) cappedFitness(:)],[2 3 1]);
    Expected = sortrows(Yf(capped(order(1:bandMax,1)),:));
end
