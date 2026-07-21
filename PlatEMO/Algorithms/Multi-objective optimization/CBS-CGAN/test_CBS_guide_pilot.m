function test_CBS_guide_pilot
%TEST_CBS_GUIDE_PILOT Guards for the unevaluated-guide mechanism.
%   Covers the wide (6:14) query allocation and a fixed-seed smoke run
%   proving that guide mode changes the trajectory, produces guided
%   offspring, and never evaluates or protects raw CGAN solutions.

    rootDir = fileparts(which('platemo'));
    if isempty(rootDir)
        error('test_CBS_guide_pilot:PlatEMOOnPath', ...
            'platemo must be on the MATLAB path.');
    end
    testWideAllocation();
    testGuideSmoke();
    fprintf('test_CBS_guide_pilot: all checks passed.\n');
end

function testWideAllocation()
    [W,~] = UniformPoint(50,2);
    populated = (5:10)';
    rng(51,'twister');
    [C,R,T] = RunRegionGAN_RC('regionquerysamples',populated,W,20,'wide');
    assert(numel(R) == 20 && size(C,1) == 20, ...
        'wide allocation must spend the full budget.');
    assert(sum(T == 1) == 6 && sum(T == 2) == 10 && sum(T == 3) == 4, ...
        'wide allocation must split 6/10/4 when all pools exist.');
    assert(all(ismember(R(T == 1),populated)) && ...
        all(~ismember(R(T >= 2),populated)), ...
        'wide rows must respect populated/empty pool membership.');
end

function testGuideSmoke()
    [gdDecs,gdMx] = runGuideOnce({'guideMode','on', ...
        'scoutMode','off','metricsMode','on'});
    assert(~isempty(gdMx) && sum(gdMx.guideEvents) > 0, ...
        'guide smoke run must cache guides.');
    assert(sum(gdMx.gdGen) > 0, ...
        'guide mode must produce guided offspring.');
    assert(sum(gdMx.ganPopGen) + sum(gdMx.ganFrontGen) == 0, ...
        'guide mode must never evaluate raw CGAN solutions.');
    assert(sum(gdMx.protUsed) == 0, ...
        'guide mode must not protect anything.');
    assert(isequal(sum(gdMx.gdFGen(:)),sum(gdMx.gdGen)), ...
        'per-F counters must partition the guided offspring.');
    assert(all(sum(gdMx.gdFGen,1) > 0), ...
        'all three F rungs must be exercised.');

    [defDecs,~] = runGuideOnce({'guideMode','off', ...
        'scoutMode','off','metricsMode','on'});
    assert(~isequal(defDecs,gdDecs), ...
        'guide mode must change the search trajectory.');

    Defaults = CBS_RegionWGAN_GP.mainlineDefaults();
    mixPar = {0,Defaults.zDim,Defaults.ganIter,Defaults.ganMiniBatch, ...
        Defaults.nCritic,Defaults.minGANTrainCount,Defaults.sampleSigma};
    [mixDecs,mixMx] = runGuideOnce({'guideMode','mix', ...
        'scoutMode','off','metricsMode','on','parameter',mixPar});
    assert(sum(mixMx.guideEvents) + sum(mixMx.gdGen) + ...
        sum(mixMx.ganEvents) == 0, ...
        'mix mode must run without guides or CGAN events.');
    assert(~isequal(mixDecs,defDecs) && ~isequal(mixDecs,gdDecs), ...
        'mix mode must differ from both default and guide mode.');

    Probe = CBS_RegionWGAN_GP('save',0,'outputFcn',@quietOutput, ...
        'guideMode','on','guideShare',0.2,'guideWindow','full');
    assert(Probe.effectiveGuideShare() == 0.2 && ...
        Probe.effectiveGuideWindow() == "full", ...
        'share/window switches must be readable.');
    didThrow = false;
    try
        CBS_RegionWGAN_GP('save',0,'outputFcn',@quietOutput, ...
            'guideShare',0.9);
    catch Error
        didThrow = strcmp(Error.identifier,'CBSRegionGAN:BadGuideShare');
    end
    assert(didThrow,'invalid guideShare must be rejected.');
end

function [decs,Mx] = runGuideOnce(extraArgs)
    rng(4242,'twister');
    Problem = LIRCMOP6_BC('N',100,'maxFE',20000);
    Algorithm = CBS_RegionWGAN_GP('save',0, ...
        'outputFcn',@quietOutput,extraArgs{:});
    Algorithm.Solve(Problem);
    decs = Algorithm.result{end,2}.decs;
    Mx = Algorithm.collectedScoutMetrics();
end

function quietOutput(~,~)
end
