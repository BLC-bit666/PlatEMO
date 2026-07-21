function test_CBS_bls_fusion
%TEST_CBS_BLS_FUSION Guards for the BLS-window and BMem-feed switches.
%   The default path (blsWindow=late, blsFeed=off) must keep the recorded
%   mainline fingerprints (verified by test_CBS_operator_modes); here we
%   check switch validation, and that the fusion configuration changes
%   the trajectory while spending the exact evaluation budget.

    rootDir = fileparts(which('platemo'));
    if isempty(rootDir)
        error('test_CBS_bls_fusion:PlatEMOOnPath', ...
            'platemo must be on the MATLAB path.');
    end
    testSwitchValidation();
    testFusionSmoke();
    fprintf('test_CBS_bls_fusion: all checks passed.\n');
end

function testSwitchValidation()
    Default = CBS_RegionWGAN_GP('save',0,'outputFcn',@quietOutput);
    assert(Default.effectiveBlsWindow() == "full" && ...
        Default.effectiveBlsFeed() == "on", ...
        'fusion mainline defaults must be full/on.');
    Probe = CBS_RegionWGAN_GP('save',0,'outputFcn',@quietOutput, ...
        'blsWindow','late','blsFeed','off');
    assert(Probe.effectiveBlsWindow() == "late" && ...
        Probe.effectiveBlsFeed() == "off", ...
        'switches must be readable.');
    for bad = {{'blsWindow','early'},{'blsFeed','maybe'}}
        didThrow = false;
        try
            CBS_RegionWGAN_GP('save',0,'outputFcn',@quietOutput, ...
                bad{1}{:});
        catch Error
            didThrow = startsWith(Error.identifier,'CBSRegionGAN:BadBls');
        end
        assert(didThrow,'invalid bls switch values must be rejected.');
    end
end

function testFusionSmoke()
    fusionDecs = runOnce({});
    preDecs = runOnce({'blsWindow','late','blsFeed','off'});
    noFeedDecs = runOnce({'blsFeed','off'});
    assert(~isequal(fusionDecs,preDecs), ...
        'the fusion mainline must differ from the pre-fusion path.');
    assert(~isequal(fusionDecs,noFeedDecs), ...
        'the harvest feed must change the trajectory.');
    assert(~isequal(preDecs,noFeedDecs), ...
        'the calibration window must change the trajectory.');
end

function decs = runOnce(extraArgs)
    rng(4242,'twister');
    Problem = LIRCMOP6_BC('N',100,'maxFE',20000);
    Algorithm = CBS_RegionWGAN_GP('save',0, ...
        'outputFcn',@quietOutput,extraArgs{:});
    Algorithm.Solve(Problem);
    assert(Problem.FE == 20000, ...
        'the evaluation budget must be spent exactly.');
    decs = Algorithm.result{end,2}.decs;
end

function quietOutput(~,~)
end
