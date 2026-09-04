function test_CBS_mainline_fingerprint()
%TEST_CBS_MAINLINE_FINGERPRINT Pin the deterministic unique mainline.

    repoRoot = fileparts(which('platemo'));
    addCBSPaths(repoRoot);
    rng(4242,'twister');
    Problem = LIRCMOP6_BC('N',100,'D',30,'maxFE',20000);
    Algorithm = CBS_RegionWGAN_GP('save',0,'outputFcn',@(varargin)[]);
    Algorithm.Solve(Problem);
    Population = Algorithm.result{end,2};
    state = rng;

    assert(Problem.FE == 20000);
    assert(isequal(Problem.CalMetric('IGD',Population), ...
        1.3475758015641228));
    assert(isequal(sum(Population.decs,'all'),1885.976923973072));
    assert(isequal(sum(Population.objs,'all'),463.058790530907));
    assert(isequal(sum(Population.cons,'all'),0));
    assert(state.State(1) == 4002897975);
    fprintf('CBS unique-mainline deterministic fingerprint passed.\n');
end
