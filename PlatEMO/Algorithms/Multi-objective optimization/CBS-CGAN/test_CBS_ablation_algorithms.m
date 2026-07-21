function test_CBS_ablation_algorithms
%TEST_CBS_ABLATION_ALGORITHMS Guards for the two ablation companions.
%   CBS_RegionWGAN_GP_A00 (module removal) and CBS_RegionWGAN_GP_CNB
%   (learning necessity) must be byte-identical to the mainline class
%   constructed with the corresponding explicit switches, the generalized
%   native runner must accept them and write native files with save=2,
%   and the campaign launcher must pass its dry-run validation.

    rootDir = fileparts(which('platemo'));
    if isempty(rootDir)
        error('test_CBS_ablation_algorithms:PlatEMOOnPath', ...
            'platemo must be on the MATLAB path.');
    end
    testReadbacks();
    testBitEquivalence();
    testRunnerWithAblationAlgorithm(rootDir);
    testLauncherDryRun(rootDir);
    fprintf('CBS ablation-algorithm guards passed.\n');
end

function testReadbacks()
    A = CBS_RegionWGAN_GP_A00('save',0,'outputFcn',@quietOutput);
    assert(A.effectiveGuideMode() == "off" && ...
        A.effectiveBoundarySearch() == "off" && ...
        A.effectiveScoutMode() == "off" && ...
        A.effectiveGeneratorMode() == "wgan", ...
        'A00 must pin the module-removal switches.');
    C = CBS_RegionWGAN_GP_CNB('save',0,'outputFcn',@quietOutput);
    assert(C.effectiveGeneratorMode() == "copynoise" && ...
        C.effectiveGuideMode() == "on" && ...
        C.effectiveGuideShare() == 0.2 && ...
        C.effectiveBoundarySearch() == "on" && ...
        C.effectiveBlsWindow() == "full" && ...
        C.effectiveBlsFeed() == "on", ...
        'CNB must differ from the mainline only in the generator.');
end

function testBitEquivalence()
    Defaults = CBS_RegionWGAN_GP.mainlineDefaults();
    zeroPar = {0,Defaults.zDim,Defaults.ganIter,Defaults.ganMiniBatch, ...
        Defaults.nCritic,Defaults.minGANTrainCount,Defaults.sampleSigma};

    a00 = runOnce(@(varargin)CBS_RegionWGAN_GP_A00(varargin{:}),{});
    twinA = runOnce(@(varargin)CBS_RegionWGAN_GP(varargin{:}), ...
        {'guideMode','off','boundarySearch','off','scoutMode','off', ...
        'blsWindow','late','blsFeed','off','parameter',zeroPar});
    assert(isequal(a00,twinA), ...
        'A00 must be bit-identical to the manually switched mainline.');

    cnb = runOnce(@(varargin)CBS_RegionWGAN_GP_CNB(varargin{:}),{});
    twinC = runOnce(@(varargin)CBS_RegionWGAN_GP(varargin{:}), ...
        {'generatorMode','copynoise'});
    assert(isequal(cnb,twinC), ...
        'CNB must be bit-identical to the manually switched mainline.');
    assert(~isequal(a00,cnb), ...
        'the two ablations must have distinct trajectories.');
end

function decs = runOnce(construct,extraArgs)
    rng(4242,'twister');
    Problem = LIRCMOP6_BC('N',100,'maxFE',20000);
    Algorithm = construct('save',0,'outputFcn',@quietOutput,extraArgs{:});
    Algorithm.Solve(Problem);
    assert(Problem.FE == 20000, ...
        'the evaluation budget must be spent exactly.');
    decs = Algorithm.result{end,2}.decs;
end

function testRunnerWithAblationAlgorithm(rootDir) %#ok<INUSD>
    scratch = tempname;
    cleanup = onCleanup(@()removeTree(scratch));
    outDir = fullfile(scratch,'Data','CBS_RegionWGAN_GP_A00');
    Options = struct('resume',true,'algorithm',"CBS_RegionWGAN_GP_A00");
    Summary = run_CBS_RegionWGAN_GP_mainline( ...
        outDir,1,"LIRCMOP8_BC",10,[],600,991,Options);
    assert(height(Summary) == 1 && Summary.status == "ok" && ...
        Summary.finalFE == 600,'runner must complete the ablation task.');
    expectedFile = fullfile(outDir,sprintf( ...
        'CBS_RegionWGAN_GP_A00_LIRCMOP8_BC_M%d_D%d_991.mat', ...
        Summary.M,Summary.D));
    assert(isfile(expectedFile), ...
        'the native file must carry the ablation class name.');
    Variables = whos('-file',expectedFile);
    assert(isequal(sort(string({Variables.name})),["metric","result"]), ...
        'native files must contain exactly result and metric.');
    Saved = load(expectedFile,'result','metric');
    assert(size(Saved.result,1) == 2 && numel(Saved.metric.IGD) == 2, ...
        'save=2 must record the mid-run and final snapshots.');
    Reused = run_CBS_RegionWGAN_GP_mainline( ...
        outDir,1,"LIRCMOP8_BC",10,[],600,991,Options);
    assert(Reused.reused == 1 && isequaln(Reused.IGD,Summary.IGD), ...
        'valid native files must be reused untouched.');
end

function testLauncherDryRun(rootDir)
    previous = getenv('CBS_MAINLINE_DRY_RUN');
    cleanup = onCleanup(@()setenv('CBS_MAINLINE_DRY_RUN',previous));
    setenv('CBS_MAINLINE_DRY_RUN','1');
    output = evalc( ...
        "run(fullfile('" + string(rootDir) + "'," + ...
        "'run_CBS_RegionWGAN_GP_DAS_LIR_full.m'))");
    assert(contains(output,'Dry-run validation passed'), ...
        'the launcher dry run must pass its preflight.');
    assert(contains(output,'CBS_RegionWGAN_GP_A00') && ...
        contains(output,'CBS_RegionWGAN_GP_CNB'), ...
        'the manifest must list both ablation campaigns.');
end

function removeTree(pathValue)
    if isfolder(pathValue)
        rmdir(pathValue,'s');
    end
end

function quietOutput(~,~)
end
