function test_CBS_region_wgan_mainline()
%TEST_CBS_REGION_WGAN_MAINLINE Verify the sole supported algorithm path.

    repoRoot = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
    addpath(genpath(repoRoot));
    assert(exist('CBS_RegionWGAN_GP','class') == 8);
    Defaults = CBS_RegionWGAN_GP.mainlineDefaults();
    assert(Defaults.ganMiniBatch == 32 && Defaults.ganIter == 100);
    assert(Defaults.nCritic == 4 && ...
        Defaults.nGen == 20 && Defaults.zDim == 6 && ...
        Defaults.minGANTrainCount == 32 && ...
        Defaults.sampleSigma == 0.3 && ...
        Defaults.ganStopFraction == 0.5 && ...
        isequal(Defaults.generatorHidden,[32 32]) && ...
        isequal(Defaults.criticHidden,[32 32]));
    assert(~isfield(Defaults,'trainGap') && ...
        ~isfield(Defaults,'archiveGap'));
    forbidden = {'schedulePolicy','trainingAmountUnit', ...
        'trainingAmountPhaseValues','mappingDiagnostics', ...
        'structuredZMode','bmemMode','trainDedupMode'};
    assert(~any(isfield(Defaults,forbidden)));
    source = fileread(which('CBS_RegionWGAN_GP'));
    tokens = regexp(source, ...
        '(?m)^%\s*(\w+)\s*---\s*([^\r\n]+?)\s*---', ...
        'tokens');
    names = string(cellfun(@(x)x{1},tokens,'UniformOutput',false));
    assert(isequal(names,["nGen","zDim","ganIter","ganMiniBatch", ...
        "nCritic","minGANTrainCount","sampleSigma"]));
    wganSource = fileread(which('BoundaryWGAN_RC'));
    removedTokens = ["splitTrainingRows","HoldoutIdx", ...
        "advanceLegacyDiagnosticRNG"];
    assert(~any(contains(wganSource,removedTokens)));
    removedStudyTokens = ["studyObserver","figureObserver", ...
        "studyOptions","mechanismAblationArms","ganPoolPerSlot", ...
        "scorebycondition"];
    sourceTree = string(source) + newline + string(wganSource);
    assert(~any(contains(sourceTree,removedStudyTokens)));
    assertThrows(@()RunRegionGAN_RC('metricnames'), ...
        'CBSRegionGAN:BadRunnerAction');
    Parameters = {0,7,50,8,5,9,0.2};
    Algorithm = CBS_RegionWGAN_GP('parameter',Parameters,'save',0, ...
        'outputFcn',@(varargin)[]);
    assert(isequal(Algorithm.parameter,Parameters));
    Problem = DASCMOP1_BC('N',10,'D',5,'maxFE',100);
    Algorithm.Solve(Problem);
    assert(Algorithm.result{end,1} == Problem.maxFE);
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
