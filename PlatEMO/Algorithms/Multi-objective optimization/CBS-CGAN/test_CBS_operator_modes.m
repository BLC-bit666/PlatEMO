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
%   All fingerprints use LIRCMOP6_BC, N=100, D=30, maxFE=20000,
%   rng(4242,'twister'). The mainline fingerprint pins the GD20 guide
%   default (2026-07-20 decision: 40% GA + 40% plain DE + 20% guided DE
%   toward unevaluated CGAN guides). The protected-injection mainline,
%   the plain S2+BLS trajectory, and the pure-DE legacy trajectory stay
%   reachable bit-for-bit through explicit switches.
    rng(4242,'twister');
    Problem = LIRCMOP6_BC('N',100,'D',30,'maxFE',20000);
    Algorithm = CBS_RegionWGAN_GP('save',0,'outputFcn',@(varargin)[]);
    assert(Algorithm.effectiveGuideMode() == "on");
    assert(Algorithm.effectiveGuideShare() == 0.2);
    assert(Algorithm.effectiveBlsWindow() == "full");
    assert(Algorithm.effectiveBlsFeed() == "on");
    Algorithm.Solve(Problem);
    Population = Algorithm.result{end,2};
    decs = Population.decs;
    assert(isequal(Problem.CalMetric('IGD',Population), ...
        1.3484288502689699));
    assert(isequal(sum(decs(:)),1885.893899968843));
    state = rng;
    assert(state.State(1) == 3538802550);

    rng(4242,'twister');
    Pre = LIRCMOP6_BC('N',100,'D',30,'maxFE',20000);
    PreAlgorithm = CBS_RegionWGAN_GP('save',0, ...
        'outputFcn',@(varargin)[], ...
        'blsWindow','late','blsFeed','off');
    PreAlgorithm.Solve(Pre);
    PrePopulation = PreAlgorithm.result{end,2};
    preDecs = PrePopulation.decs;
    assert(isequal(Pre.CalMetric('IGD',PrePopulation), ...
        1.3490149458128178));
    assert(isequal(sum(preDecs(:)),1881.2224281584283));
    state = rng;
    assert(state.State(1) == 2349263053);

    rng(4242,'twister');
    Guard = LIRCMOP6_BC('N',100,'D',30,'maxFE',20000);
    GuardAlgorithm = CBS_RegionWGAN_GP('save',0, ...
        'outputFcn',@(varargin)[], ...
        'guideMode','off','scoutMode','nofrontier', ...
        'blsWindow','late','blsFeed','off');
    GuardAlgorithm.Solve(Guard);
    GuardPopulation = GuardAlgorithm.result{end,2};
    guardDecs = GuardPopulation.decs;
    assert(isequal(Guard.CalMetric('IGD',GuardPopulation), ...
        1.3471946005101978));
    assert(isequal(sum(guardDecs(:)),1910.3749679275759));
    state = rng;
    assert(state.State(1) == 2297882816);

    oldParameter = {30,6,100,32,4,32,0.3};
    rng(4242,'twister');
    Prior = LIRCMOP6_BC('N',100,'D',30,'maxFE',20000);
    PriorAlgorithm = CBS_RegionWGAN_GP('save',0, ...
        'outputFcn',@(varargin)[], ...
        'guideMode','off','scoutMode','off', ...
        'blsWindow','late','blsFeed','off','parameter',oldParameter);
    PriorAlgorithm.Solve(Prior);
    PriorPopulation = PriorAlgorithm.result{end,2};
    priorDecs = PriorPopulation.decs;
    assert(isequal(Prior.CalMetric('IGD',PriorPopulation), ...
        1.3470807122527642));
    assert(isequal(sum(priorDecs(:)),1919.0968011138757));
    state = rng;
    assert(state.State(1) == 368524342);

    rng(4242,'twister');
    Legacy = LIRCMOP6_BC('N',100,'D',30,'maxFE',20000);
    LegacyAlgorithm = CBS_RegionWGAN_GP('save',0, ...
        'outputFcn',@(varargin)[], ...
        'operatorMode','de','boundarySearch','off', ...
        'guideMode','off','scoutMode','off', ...
        'blsWindow','late','blsFeed','off','parameter',oldParameter);
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
