function test_CBS_no_cgan_ablation()
%TEST_CBS_NO_CGAN_ABLATION Verify the random-slot removal control.

    repoRoot = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
    addpath(genpath(repoRoot));
    assert(exist('CBS_RegionWGAN_GP_NoCGAN','class') == 8);

    %% The shared 100K cutoff is strict and deterministic
    assert(CBS_RegionWGAN_GP_NoCGAN.randomSlotsAtFE(0,100000));
    assert(CBS_RegionWGAN_GP_NoCGAN.randomSlotsAtFE(99999,100000));
    assert(~CBS_RegionWGAN_GP_NoCGAN.randomSlotsAtFE(100000,100000));
    assert(~CBS_RegionWGAN_GP_NoCGAN.randomSlotsAtFE(200000,100000));

    %% Audit the fixed 40/40/20-to-40/60 wiring
    mainSource = string(fileread(which('CBS_RegionWGAN_GP')));
    branchSource = string(fileread(which('CBS_RegionWGAN_GP_NoCGAN')));
    assert(contains(mainSource, ...
        'plainShare = (1-double(guidedShare))/2') && ...
        contains(mainSource,'Problem.Initialization(guidedCount)') && ...
        contains(mainSource,'if cganEnabled && Problem.FE < ganFELimit'));
    forbidden = ["BoundaryWGAN_RC","BuildBoundaryDataset_RC", ...
        "RunRegionGAN_RC","UpdateBoundaryMemory_RC"];
    assert(~any(contains(branchSource,forbidden)));

    %% A complete short run must never enter any CGAN function
    rng(4242,'twister');
    Problem = DASCMOP1_BC('N',20,'D',10,'maxFE',2400);
    Algorithm = CBS_RegionWGAN_GP_NoCGAN( ...
        'save',0,'outputFcn',@(varargin)[]);
    profile clear;
    profile on;
    Algorithm.Solve(Problem);
    Profile = profile('info');
    profile off;
    cleanup = onCleanup(@()profile('clear'));
    names = string({Profile.FunctionTable.FunctionName});
    assert(~any(contains(names,forbidden)));
    assert(Problem.FE == Problem.maxFE);
    Population = Algorithm.result{end,2};
    assert(numel(Population) == Problem.N && ...
        all(isfinite(Population.decs),'all') && ...
        all(isfinite(Population.objs),'all'));
    clear cleanup
    profile clear;
    fprintf('CBS no-CGAN random-slot ablation passed.\n');
end
