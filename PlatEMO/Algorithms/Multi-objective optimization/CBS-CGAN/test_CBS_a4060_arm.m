function test_CBS_a4060_arm
%TEST_CBS_A4060_ARM Guards for the operator-ratio control CBS_RegionWGAN_GP_A4060.
%   The control must pin "module off, guideMode on" (40/60 fallback
%   composition), be bit-identical to the mainline class constructed with
%   the same explicit switches, differ from A00 (proving the ratio changes
%   the trajectory), and pass the generalized native runner contract.

    testReadbacks();
    testBitEquivalenceAndDistinctness();
    testRunnerContract();
    fprintf('CBS A4060 operator-ratio guards passed.\n');
end

function testReadbacks()
    A = CBS_RegionWGAN_GP_A4060('save',0,'outputFcn',@quietOutput);
    assert(A.effectiveGuideMode() == "on" && ...
        A.effectiveBoundarySearch() == "off" && ...
        A.effectiveScoutMode() == "off" && ...
        A.effectiveGeneratorMode() == "wgan", ...
        'A4060 must pin module-off switches with guideMode on.');
end

function testBitEquivalenceAndDistinctness()
    Defaults = CBS_RegionWGAN_GP.mainlineDefaults();
    zeroPar = {0,Defaults.zDim,Defaults.ganIter,Defaults.ganMiniBatch, ...
        Defaults.nCritic,Defaults.minGANTrainCount,Defaults.sampleSigma};
    a4060 = runOnce(@(varargin)CBS_RegionWGAN_GP_A4060(varargin{:}),{});
    twin = runOnce(@(varargin)CBS_RegionWGAN_GP(varargin{:}), ...
        {'guideMode','on','boundarySearch','off','scoutMode','off', ...
        'blsWindow','late','blsFeed','off','parameter',zeroPar});
    assert(isequal(a4060,twin), ...
        'A4060 must be bit-identical to the manually switched mainline.');
    a00 = runOnce(@(varargin)CBS_RegionWGAN_GP_A00(varargin{:}),{});
    assert(~isequal(a4060,a00), ...
        'the 40/60 and 50/50 backbones must have distinct trajectories.');
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

function testRunnerContract()
    scratch = tempname;
    cleanup = onCleanup(@()removeTree(scratch));
    outDir = fullfile(scratch,'Data','CBS_RegionWGAN_GP_A4060');
    Options = struct('resume',true,'algorithm',"CBS_RegionWGAN_GP_A4060");
    Summary = run_CBS_RegionWGAN_GP_mainline( ...
        outDir,1,"LIRCMOP8_BC",20,[],1200,992,Options);
    assert(height(Summary) == 1 && Summary.status == "ok" && ...
        Summary.finalFE == 1200,'runner must complete the A4060 task.');
    expectedFile = fullfile(outDir,sprintf( ...
        'CBS_RegionWGAN_GP_A4060_LIRCMOP8_BC_M%d_D%d_992.mat', ...
        Summary.M,Summary.D));
    assert(isfile(expectedFile), ...
        'the native file must carry the A4060 class name.');
    Saved = load(expectedFile,'result','metric');
    assert(size(Saved.result,1) == 2 && numel(Saved.metric.IGD) == 2, ...
        'save=2 must record the mid-run and final snapshots.');
end

function removeTree(pathValue)
    if isfolder(pathValue)
        rmdir(pathValue,'s');
    end
end

function quietOutput(~,~)
end
