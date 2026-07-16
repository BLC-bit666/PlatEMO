function test_CBS_region_boundary_ref_cap()
%TEST_CBS_REGION_BOUNDARY_REF_CAP Verify fixed BMem and TrainX gates.

    repoRoot = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
    addpath(genpath(repoRoot));
    testAnchorCap();
    testPreviousAnchorCompetition();
    testSmallTrainingSetIsSkipped();
    fprintf('CBS RegionGAN boundary regressions passed.\n');
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
    assert(all(BMem.source_f == 0) && all(BMem.age_f == 0));
end

function testPreviousAnchorCompetition()
    Problem = DASCMOP1_BC('N',8,'D',6,'maxFE',100);
    W = [0.5 0.5];
    Pop = SOLUTION([makeDecisionRows(Problem,1); ...
        makeDecisionRows(Problem,1)],[10 10;0 0],[0;1]);
    Empty = Pop([]);
    x = makeDecisionRows(Problem,1);
    Previous = struct('ref',1,'gap',0.01,'x_b',x,'y_b',[1 1], ...
        'x_f',x,'y_f',[1 1],'x_i',x,'y_i',[0.9 0.9], ...
        'source_f',0,'age_f',0);
    Options = struct('frontDepth',2,'pairNeighborRefRadius',0, ...
        'minBoundaryLength',100,'maxAnchorsPerRef',1);
    BMem = UpdateBoundaryMemory_RC(Previous,Pop,Empty,Empty,Empty,W,Options);
    assert(isequal(BMem.y_b,[1 1]) && isequal(BMem.y_i,[0 0]));
    assert(BMem.source_f == 1 && BMem.age_f == 1);
end

function testSmallTrainingSetIsSkipped()
    Problem = DASCMOP1_BC('N',8,'D',6,'maxFE',100);
    Options = struct('minTrainCount',32);
    [GAN,RawDec] = RunRegionGAN_RC('trainandsample',[], ...
        makeDecisionRows(Problem,31),rand(31,2),rand(2,2),Problem,Options);
    assert(isempty(GAN));
    assert(isempty(RawDec) && size(RawDec,2) == Problem.D);
end

function X = makeDecisionRows(Problem,n)
    X = Problem.lower + rand(n,Problem.D).*(Problem.upper-Problem.lower);
end
