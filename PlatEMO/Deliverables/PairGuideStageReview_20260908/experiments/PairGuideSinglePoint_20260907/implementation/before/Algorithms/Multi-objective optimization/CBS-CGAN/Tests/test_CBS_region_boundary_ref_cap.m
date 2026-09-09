function test_CBS_region_boundary_ref_cap()
%TEST_CBS_REGION_BOUNDARY_REF_CAP Verify fixed BMem and TrainX gates.

    repoRoot = fileparts(which('platemo'));
    addCBSPaths(repoRoot);
    testAnchorCap();
    testPreviousAnchorCompetition();
    testNearestLegalPartner();
    testSharedReferenceScale();
    testSmallTrainingSetIsSkipped();
    testMinimalWGANTraining();
    fprintf('CBS RegionGAN boundary regressions passed.\n');
end

function testSharedReferenceScale()
    W = [0 1;0.5 0.5;1 0];
    Y = [0 0;10 100;9 10;8 20];
    [allRefs,Scale] = AssignReferenceVectors_CBS(Y,W);
    sharedRefs = AssignReferenceVectors_CBS(Y(3:4,:),W,Scale);
    localRefs = AssignReferenceVectors_CBS(Y(3:4,:),W);
    assert(isequal(Scale.minimum,[0 0]) && ...
        isequal(Scale.span,[10 100]));
    assert(isequal(sharedRefs,allRefs(3:4)));
    assert(~isequal(localRefs,sharedRefs), ...
        'The fixture must expose the cross-generation scale mismatch.');
end

function testAnchorCap()
    Problem = DASCMOP1_BC('N',8,'D',6,'maxFE',100);
    W = [0.5 0.5];
    Xf = makeDecisionRows(Problem,10);
    Yf = [(1:10)' (10:-1:1)'];
    Xi = makeDecisionRows(Problem,1);
    Pop = SOLUTION([Xf;Xi],[Yf;0 0],[zeros(10,1);1]);
    Empty = Pop([]);
    Options = struct('frontDepth',2,'pairNeighborRefRadius',0, ...
        'minBoundaryLength',100,'maxAnchorsPerRef',5);
    BMem = UpdateBoundaryMemory_RC([],Pop,Empty,Empty,Empty,W,Options);
    Fitness = CalFitness_CBS(Yf,zeros(10,1));
    [~,order] = sort(Fitness,'ascend');
    assert(isequal(sortrows(BMem.y_b),sortrows(Yf(order(1:5),:))));
    assert(isequal(string(fieldnames(BMem)), ...
        ["ref";"gap";"x_b";"y_b";"x_i"]));
end

function testPreviousAnchorCompetition()
    Problem = DASCMOP1_BC('N',8,'D',6,'maxFE',100);
    W = [0.5 0.5];
    Pop = SOLUTION([makeDecisionRows(Problem,1); ...
        makeDecisionRows(Problem,1)],[10 10;0 0],[0;1]);
    Empty = Pop([]);
    x = makeDecisionRows(Problem,1);
    Previous = struct('ref',1,'gap',0.01,'x_b',x,'y_b',[1 1]);
    Options = struct('frontDepth',2,'pairNeighborRefRadius',0, ...
        'minBoundaryLength',100,'maxAnchorsPerRef',1);
    BMem = UpdateBoundaryMemory_RC(Previous,Pop,Empty,Empty,Empty,W,Options);
    assert(isequal(BMem.x_b,x) && isequal(BMem.y_b,[1 1]));
end

function testNearestLegalPartner()
    Problem = DASCMOP1_BC('N',8,'D',6,'maxFE',100);
    W = [0.5 0.5];
    X = makeDecisionRows(Problem,3);
    Y = [1 1;1.1 1.1;0.5 2];
    Pop = SOLUTION(X,Y,[0;1;1]);
    Empty = Pop([]);
    Options = struct('frontDepth',2,'pairNeighborRefRadius',0, ...
        'minBoundaryLength',100,'maxAnchorsPerRef',5);
    BMem = UpdateBoundaryMemory_RC([],Pop,Empty,Empty,Empty,W,Options);
    assert(size(BMem.y_b,1) == 1);
    minimum = min(Y,[],1);
    span = max(Y,[],1)-minimum;
    expectedGap = norm((Y(1,:)-minimum)./span - ...
        (Y(3,:)-minimum)./span);
    assert(isequal(BMem.y_b,Y(1,:)) && ...
        abs(BMem.gap-expectedGap) <= 10*eps(expectedGap));
end

function testSmallTrainingSetIsSkipped()
    Problem = DASCMOP1_BC('N',8,'D',6,'maxFE',100);
    Options = struct('minTrainCount',32);
    [GAN,RawDec] = RunRegionGAN_RC('trainandsample',[], ...
        makeDecisionRows(Problem,31),rand(31,3),rand(2,3),Problem,Options);
    assert(isempty(GAN));
    assert(isempty(RawDec) && size(RawDec,2) == Problem.D);
end

function testMinimalWGANTraining()
    Problem = LIRCMOP5_BC('N',10,'D',5,'maxFE',100);
    span = Problem.upper-Problem.lower;
    TrainX = Problem.lower + [0.25;0.75].*span;
    TrainC = [0.25 0.75 1;0.75 0.25 0];
    SampleC = [0.5 0.5 1;0.2 0.8 1];
    Options = struct('minTrainCount',2,'zDim',2,'iter',1, ...
        'miniBatch',2,'nCritic',1,'generatorHidden',[4 4], ...
        'criticHidden',[4 4],'sampleSigma',0.1);
    rng(1907,'twister');
    [GAN,RawDec] = RunRegionGAN_RC('trainandsample',[], ...
        TrainX,TrainC,SampleC,Problem,Options);
    assert(GAN.iterG == 1 && GAN.iterC == 1);
    assert(isequal(size(RawDec),[2 Problem.D]) && ...
        all(isfinite(RawDec),'all') && ...
        all(RawDec >= Problem.lower,'all') && ...
        all(RawDec <= Problem.upper,'all'));
    assert(Problem.FE == 0, ...
        'Raw generator rows must remain unevaluated guides.');
end

function X = makeDecisionRows(Problem,n)
    X = Problem.lower + rand(n,Problem.D).*(Problem.upper-Problem.lower);
end
