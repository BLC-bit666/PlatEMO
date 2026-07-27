function test_CBS_pairflag_dataset()
%TEST_CBS_PAIRFLAG_DATASET Verify the sole CGAN training-data definition.

    repoRoot = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
    addpath(genpath(repoRoot));
    testDatasetShape();
    testEmptyMemoryWidth();
    testMinimalTraining();
    fprintf('CBS pairflag data regressions passed.\n');
end

function testDatasetShape()
    Problem = DASCMOP1_BC('N',8,'D',6,'maxFE',100);
    W = [0.25 0.75;0.75 0.25];
    Xb = makeDecisionRows(Problem,3);
    Xi = makeDecisionRows(Problem,3);
    Xi(3,2) = NaN;
    BMem = struct('ref',[1;2;1],'gap',[0.1;0.2;0.3], ...
        'x_b',Xb,'y_b',[1 2;2 1;1.5 1.5],'x_i',Xi);
    [TrainX,TrainC,QueryRefs] = BuildBoundaryDataset_RC(BMem,W,Problem);

    assert(isequal(TrainX(1:3,:),Xb));
    assert(isequal(TrainC(1:3,:),[W([1;2;1],:),ones(3,1)]));
    assert(isequal(TrainX(4:5,:),Xi(1:2,:)));
    assert(isequal(TrainC(4:5,:),[W([1;2],:),zeros(2,1)]));
    assert(isequal(QueryRefs,[1;2]));
end

function testEmptyMemoryWidth()
    Problem = DASCMOP1_BC('N',8,'D',6,'maxFE',100);
    W = [0.25 0.75;0.75 0.25];
    [TrainX,TrainC,QueryRefs] = BuildBoundaryDataset_RC([],W,Problem);
    assert(isempty(TrainX) && size(TrainX,2) == Problem.D);
    assert(isempty(TrainC) && size(TrainC,2) == size(W,2)+1);
    assert(isempty(QueryRefs));
end

function testMinimalTraining()
    Problem = LIRCMOP5_BC('N',10,'D',5,'maxFE',100);
    span = Problem.upper-Problem.lower;
    TrainX = Problem.lower+[0.25;0.75;0.4;0.6].*span;
    TrainC = [0.25 0.75 1;0.75 0.25 1; ...
        0.25 0.75 0;0.75 0.25 0];
    QueryC = [0.5 0.5 1;0.2 0.8 1];
    Options = struct('minTrainCount',2,'zDim',2,'iter',1, ...
        'miniBatch',2,'nCritic',1,'generatorHidden',[4 4], ...
        'criticHidden',[4 4],'sampleSigma',0.1);
    rng(1907,'twister');
    [GAN,RawDec] = RunRegionGAN_RC('trainandsample',[], ...
        TrainX,TrainC,QueryC,Problem,Options);
    assert(GAN.iterG == 1 && GAN.iterC == 1 && GAN.M == 3);
    assert(isequal(size(RawDec),[2 Problem.D]) && ...
        all(isfinite(RawDec),'all') && ...
        all(RawDec >= Problem.lower,'all') && ...
        all(RawDec <= Problem.upper,'all'));
end

function X = makeDecisionRows(Problem,n)
    X = Problem.lower+rand(n,Problem.D).*(Problem.upper-Problem.lower);
end
