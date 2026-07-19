function test_CBS_operator_modes()
%TEST_CBS_OPERATOR_MODES Verify the engineering offspring-operator switch.

    repoRoot = fileparts(fileparts(fileparts(fileparts( ...
        mfilename('fullpath')))));
    addpath(genpath(repoRoot));
    testModeValidation();
    testDefaultTrajectoryUnchanged();
    testAlternativeModesRunAndDiffer();
    testTriageRunnerSmoke();
    fprintf('CBS operator-mode regressions passed.\n');
end

function testModeValidation()
    Default = CBS_RegionWGAN_GP('save',0,'outputFcn',@(varargin)[]);
    assert(Default.effectiveOperatorMode() == "ga_de_half");
    Legacy = CBS_RegionWGAN_GP('save',0,'outputFcn',@(varargin)[], ...
        'operatorMode',"de");
    assert(Legacy.effectiveOperatorMode() == "de");
    S1 = CBS_RegionWGAN_GP('save',0,'outputFcn',@(varargin)[], ...
        'operatorMode',"imtcmo_de");
    assert(S1.effectiveOperatorMode() == "imtcmo_de");
    S2 = CBS_RegionWGAN_GP('save',0,'outputFcn',@(varargin)[], ...
        'operatorMode','ga_de_half');
    assert(S2.effectiveOperatorMode() == "ga_de_half");
    didThrow = false;
    try
        CBS_RegionWGAN_GP('save',0,'outputFcn',@(varargin)[], ...
            'operatorMode',"bogus");
    catch Error
        didThrow = strcmp(Error.identifier,'CBSRegionGAN:BadOperatorMode');
    end
    assert(didThrow);
end

function testDefaultTrajectoryUnchanged()
%   Both fingerprints use LIRCMOP6_BC, N=100, D=30, maxFE=20000,
%   rng(4242,'twister'). The mainline fingerprint pins the S2+BLS default
%   path; the legacy fingerprint proves the pre-switch trajectory stays
%   reachable bit-for-bit through explicit switches.
    rng(4242,'twister');
    Problem = LIRCMOP6_BC('N',100,'D',30,'maxFE',20000);
    Algorithm = CBS_RegionWGAN_GP('save',0,'outputFcn',@(varargin)[]);
    Algorithm.Solve(Problem);
    Population = Algorithm.result{end,2};
    decs = Population.decs;
    assert(isequal(Problem.CalMetric('IGD',Population), ...
        1.3470807122527642));
    assert(isequal(sum(decs(:)),1919.0968011138757));
    state = rng;
    assert(state.State(1) == 368524342);

    rng(4242,'twister');
    Legacy = LIRCMOP6_BC('N',100,'D',30,'maxFE',20000);
    LegacyAlgorithm = CBS_RegionWGAN_GP('save',0, ...
        'outputFcn',@(varargin)[], ...
        'operatorMode','de','boundarySearch','off');
    LegacyAlgorithm.Solve(Legacy);
    LegacyPopulation = LegacyAlgorithm.result{end,2};
    legacyDecs = LegacyPopulation.decs;
    assert(isequal(Legacy.CalMetric('IGD',LegacyPopulation), ...
        1.3653447524657205));
    assert(isequal(sum(legacyDecs(:)),1729.4377215220884));
    state = rng;
    assert(state.State(1) == 415301093);
end

function testAlternativeModesRunAndDiffer()
    reference = shortRunDecSum("de");
    for mode = ["imtcmo_de","ga_de_half"]
        value = shortRunDecSum(mode);
        assert(~isequal(value,reference), ...
            'Mode %s did not change the search trajectory.',mode);
    end
end

function value = shortRunDecSum(mode)
    rng(4242,'twister');
    Problem = LIRCMOP6_BC('N',20,'D',10,'maxFE',2000);
    Algorithm = CBS_RegionWGAN_GP('save',0,'outputFcn',@(varargin)[], ...
        'operatorMode',char(mode));
    Algorithm.Solve(Problem);
    assert(Problem.FE == 2000);
    Population = Algorithm.result{end,2};
    decs = Population.decs;
    value = sum(decs(:));
end

function testTriageRunnerSmoke()
    outDir = string(tempname);
    cleanup = onCleanup(@()removeTree(outDir));
    Options = struct('problems',"LIRCMOP6_BC",'seeds',1, ...
        'maxFE',600,'N',10,'arms',["S1","S2"]);
    Summary = run_CBS_operator_triage(outDir,1,Options);
    assert(height(Summary) == 2);
    assert(all(Summary.status == "ok"));
    assert(all(Summary.finalFE == 600));
    Reused = run_CBS_operator_triage(outDir,1,Options);
    assert(all(Reused.reused == 1));
    assert(isequaln(Reused.IGD,Summary.IGD));
end

function removeTree(pathValue)
    if isfolder(pathValue)
        rmdir(pathValue,'s');
    end
end
