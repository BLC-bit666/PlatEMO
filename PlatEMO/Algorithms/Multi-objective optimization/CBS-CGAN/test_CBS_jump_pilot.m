function test_CBS_jump_pilot
%TEST_CBS_JUMP_PILOT Guards for the landing-zone jump machinery.
%   Covers the landing dataset construction (exact one-row-per-pair
%   lambda formula with box clamping), the half query allocation with
%   logical backward compatibility, and fixed-seed smoke runs proving the
%   JX/TJ arms change the trajectory, trigger events, and keep protection
%   active.

    rootDir = fileparts(which('platemo'));
    if isempty(rootDir)
        error('test_CBS_jump_pilot:PlatEMOOnPath', ...
            'platemo must be on the MATLAB path.');
    end
    testLandingDataset();
    testHalfAllocation();
    testJumpSmoke();
    fprintf('test_CBS_jump_pilot: all checks passed.\n');
end

function testLandingDataset()
    Problem = LIRCMOP6_BC('N',20,'maxFE',5000);
    D = Problem.D;
    rng(31,'twister');
    BMem = struct( ...
        'ref',[1;1;2;3], ...
        'gap',[0.1;0.2;0.1;0.3], ...
        'x_b',rand(4,D).*0.4 + 0.3, ...
        'y_b',rand(4,2), ...
        'x_i',rand(4,D).*0.4 + 0.3);
    [W,~] = UniformPoint(10,2);

    [AX,AC,AQ] = BuildBoundaryDataset_RC(BMem,W,Problem,'anchor');
    [LX,LC,LQ] = BuildBoundaryDataset_RC(BMem,W,Problem,'landing');
    assert(isequal(AX,double(BMem.x_b)), ...
        'anchor mode must return the anchor rows unchanged.');
    assert(size(LX,1) == size(AX,1), ...
        'landing mode must keep one row per pair.');
    assert(isequal(AC,LC) && isequal(AQ,LQ), ...
        'landing mode must keep conditions and query refs unchanged.');
    ladder = [1.5;2;3;1.5];
    expected = double(BMem.x_b) + ladder.*(double(BMem.x_i) - ...
        double(BMem.x_b));
    expected = min(max(expected,repmat(double(Problem.lower),4,1)), ...
        repmat(double(Problem.upper),4,1));
    assert(max(abs(LX(:)-expected(:))) < 1e-12, ...
        'landing rows must follow the tiled-lambda formula with clamp.');
    assert(all(LX(:) >= min(double(Problem.lower)) - 1e-12) && ...
        all(all(LX <= repmat(double(Problem.upper),4,1) + 1e-12)), ...
        'landing rows must respect the box constraints.');
    Legacy = BuildBoundaryDataset_RC(BMem,W,Problem);
    assert(isequal(Legacy,AX), ...
        'omitting the mode must reproduce the anchor path.');
end

function testHalfAllocation()
    [W,~] = UniformPoint(50,2);
    populated = (5:10)';

    rng(41,'twister');
    [C,R,T] = RunRegionGAN_RC('regionquerysamples',populated,W,20,'half');
    assert(numel(R) == 20 && size(C,1) == 20, ...
        'half allocation must spend the full budget.');
    assert(sum(T == 1) == 10 && sum(T == 2) == 10 && ~any(T == 3), ...
        'half allocation must split 10/10 with one-hop only.');
    assert(all(~ismember(R(T == 2),populated)), ...
        'half-mode frontier rows must target empty references.');

    rng(41,'twister');
    [~,R0,T0] = RunRegionGAN_RC('regionquerysamples',populated,W,30,false);
    assert(numel(R0) == 30 && sum(T0 == 1) == 25 && sum(T0 == 2) == 5, ...
        'logical false must still map to the legacy allocation.');
    rng(41,'twister');
    [~,~,T1] = RunRegionGAN_RC('regionquerysamples',populated,W,20,true);
    assert(any(T1 == 3),'logical true must still map to scout mode.');
end

function testJumpSmoke()
    Defaults = CBS_RegionWGAN_GP.mainlineDefaults();
    jumpPar = {20,Defaults.zDim,Defaults.ganIter,Defaults.ganMiniBatch, ...
        Defaults.nCritic,Defaults.minGANTrainCount,1.0};

    [jxDecs,jxMx] = runArmOnce({'scoutMode','nofrontier', ...
        'generatorMode','jump','metricsMode','on','parameter',jumpPar});
    assert(~isempty(jxMx) && sum(jxMx.ganEvents) > 0, ...
        'JX smoke run must trigger CGAN events.');
    assert(sum(jxMx.protUsed) > 0, ...
        'JX must protect its jump offspring.');
    assert(sum(jxMx.ganHop2Gen) == 0, ...
        'JX must use the legacy one-hop allocation.');

    [tjDecs,tjMx] = runArmOnce({'scoutMode','nofrontier', ...
        'generatorMode','jumptrivial','metricsMode','on', ...
        'parameter',jumpPar});
    assert(sum(tjMx.ganEvents) > 0 && sum(tjMx.protUsed) > 0, ...
        'TJ smoke run must trigger events with protection.');
    assert(~isequal(jxDecs,tjDecs), ...
        'JX and TJ must produce different trajectories.');

    [defDecs,~] = runArmOnce({'metricsMode','on'});
    assert(~isequal(defDecs,jxDecs) && ~isequal(defDecs,tjDecs), ...
        'jump arms must differ from the default mainline.');
end

function [decs,Mx] = runArmOnce(extraArgs)
    rng(4242,'twister');
    Problem = LIRCMOP6_BC('N',100,'maxFE',20000);
    Algorithm = CBS_RegionWGAN_GP('save',0, ...
        'outputFcn',@quietOutput,'guideMode','off',extraArgs{:});
    Algorithm.Solve(Problem);
    decs = Algorithm.result{end,2}.decs;
    Mx = Algorithm.collectedScoutMetrics();
end

function quietOutput(~,~)
end
