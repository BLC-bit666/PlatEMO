function test_CBS_scout_pilot
%TEST_CBS_SCOUT_PILOT Guards for the scout-mode study machinery.
%   Covers the protected selection contract (carry-out fitness, guaranteed
%   slots, byte-identical no-protect path), the scout query allocation, and
%   a fixed-seed smoke run proving that scout mode changes the trajectory,
%   that protection is active, and that metrics recording is neutral.

    rootDir = fileparts(which('platemo'));
    if isempty(rootDir)
        error('test_CBS_scout_pilot:PlatEMOOnPath', ...
            'platemo must be on the MATLAB path.');
    end
    testProtectedSelection();
    testScoutQueryAllocation();
    testScoutSmokeAndNeutrality();
    fprintf('test_CBS_scout_pilot: all checks passed.\n');
end

function testProtectedSelection()
    rng(11,'twister');
    Problem = LIRCMOP6_BC('N',40,'maxFE',10000);
    span = Problem.upper - Problem.lower;
    Decs = repmat(Problem.lower,120,1) + ...
        rand(120,Problem.D).*repmat(span,120,1);
    Union = Problem.Evaluation(Decs);

    [P0,F0,sel0,uF0] = EnvironmentalSelection_CBS(Union,40,true);
    assert(numel(P0) == 40 && numel(sel0) == 40, ...
        'legacy path must return N rows with a selection index.');
    assert(isequal(P0.decs,Union(sel0).decs), ...
        'selIdx must map output rows back to the union.');
    assert(isequal(F0,uF0(sel0)), ...
        'carried fitness must equal the union fitness values.');
    [P00,F00] = EnvironmentalSelection_CBS(Union,40,true,[]);
    assert(isequal(P00.decs,P0.decs) && isequal(F00,F0), ...
        'empty protect list must reproduce the legacy path.');

    protect = 101:110;
    [P1,F1,sel1,uF1] = EnvironmentalSelection_CBS(Union,40,true,protect);
    assert(numel(P1) == 40,'protected path must still return N rows.');
    assert(all(ismember(protect,sel1)), ...
        'every protected row must receive a slot.');
    assert(numel(setdiff(sel1,protect)) == 30, ...
        'competitive slots must shrink to N minus protected count.');
    assert(isequal(F1,uF1(sel1)), ...
        'protected path must carry union fitness, not recompute.');
    assert(isequal(P1.decs,Union(sel1).decs), ...
        'protected selIdx must map output rows back to the union.');
end

function testScoutQueryAllocation()
    [W,~] = UniformPoint(50,2);
    populated = (5:10)';

    rng(21,'twister');
    [C,R,T] = RunRegionGAN_RC('regionquerysamples',populated,W,20,true);
    assert(numel(R) == 20 && numel(T) == 20 && size(C,1) == 20, ...
        'scout allocation must spend the full budget.');
    assert(sum(T == 1) == 7 && sum(T == 2) == 9 && sum(T == 3) == 4, ...
        'scout allocation must split 7/9/4 when all pools exist.');
    assert(all(ismember(R(T == 1),populated)), ...
        'populated rows must target populated references.');
    assert(all(~ismember(R(T >= 2),populated)), ...
        'frontier rows must target empty references.');

    rng(21,'twister');
    [~,R0,T0] = RunRegionGAN_RC('regionquerysamples',populated,W,30,false);
    assert(numel(R0) == 30 && sum(T0 == 2) == 5 && ...
        sum(T0 == 1) == 25 && ~any(T0 == 3), ...
        'legacy allocation must keep the one-sixth one-hop split.');
end

function testScoutSmokeAndNeutrality()
    Defaults = CBS_RegionWGAN_GP.mainlineDefaults();
    par = {20,Defaults.zDim,Defaults.ganIter,Defaults.ganMiniBatch, ...
        Defaults.nCritic,Defaults.minGANTrainCount,Defaults.sampleSigma};

    [scDecs,scMx] = runScoutOnce({'scoutMode','on', ...
        'metricsMode','on','parameter',par});
    [sc2Decs,sc2Mx] = runScoutOnce({'scoutMode','on', ...
        'metricsMode','off','parameter',par});
    assert(isequal(scDecs,sc2Decs), ...
        'metrics recording must be trajectory-neutral.');
    assert(isempty(sc2Mx),'metricsMode=off must not record.');
    assert(~isempty(scMx) && sum(scMx.ganEvents) > 0, ...
        'scout smoke run must trigger CGAN events.');
    assert(sum(scMx.protUsed) > 0, ...
        'protected injection must be active in scout mode.');
    assert(sum(scMx.ganFrontGen) > sum(scMx.ganPopGen), ...
        'scout mode must allocate a frontier majority.');
    assert(~isempty(scMx.covFE) && all(diff(scMx.covFE) > 0), ...
        'coverage curve must be recorded each generation.');

    [defDecs,defMx] = runScoutOnce({'scoutMode','off','metricsMode','on'});
    assert(~isequal(defDecs,scDecs), ...
        'scout mode must change the search trajectory.');
    assert(sum(defMx.protUsed) == 0, ...
        'scoutMode=off must never protect offspring.');
    assert(sum(defMx.ganHop2Gen) == 0, ...
        'scoutMode=off must never issue two-hop queries.');
end

function [decs,Mx] = runScoutOnce(extraArgs)
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
