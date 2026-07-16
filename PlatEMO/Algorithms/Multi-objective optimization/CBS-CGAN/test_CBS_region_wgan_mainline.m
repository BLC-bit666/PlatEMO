function test_CBS_region_wgan_mainline()
%TEST_CBS_REGION_WGAN_MAINLINE Verify the sole supported algorithm path.

    repoRoot = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
    addpath(genpath(repoRoot));
    assert(exist('CBS_RegionWGAN_GP','class') == 8);
    Defaults = CBS_RegionWGAN_GP.mainlineDefaults();
    assert(Defaults.trainGap == 1 && Defaults.archiveGap == 1);
    assert(Defaults.ganMiniBatch == 32 && Defaults.ganIter == 100);
    assert(Defaults.nCritic == 4 && ...
        isequal(Defaults.generatorHidden,[32 32]) && ...
        isequal(Defaults.criticHidden,[32 32]));
    forbidden = {'schedulePolicy','trainingAmountUnit', ...
        'trainingAmountPhaseValues','mappingDiagnostics', ...
        'structuredZMode','bmemMode','trainDedupMode'};
    assert(~any(isfield(Defaults,forbidden)));
    assertThrows(@()RunRegionGAN_RC('metricnames'), ...
        'CBSRegionGAN:BadRunnerAction');
    Algorithm = CBS_RegionWGAN_GP('parameter',{50},'save',0, ...
        'outputFcn',@(varargin)[]);
    Problem = DASCMOP1_BC('N',10,'D',5,'maxFE',100);
    assertThrows(@()Algorithm.Solve(Problem), ...
        'CBSRegionGAN:FixedMainlineParameters');
    fprintf('CBS RegionWGAN-GP mainline regressions passed.\n');
end

function assertThrows(F,identifier)
    didThrow = false;
    try
        F();
    catch Error
        didThrow = strcmp(Error.identifier,identifier);
    end
    assert(didThrow);
end
