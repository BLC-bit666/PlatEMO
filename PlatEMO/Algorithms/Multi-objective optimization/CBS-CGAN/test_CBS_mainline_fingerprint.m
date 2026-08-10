function test_CBS_mainline_fingerprint()
%TEST_CBS_MAINLINE_FINGERPRINT Pin the deterministic unique mainline.

    repoRoot = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
    addCBSPaths(repoRoot);
    rng(4242,'twister');
    Problem = LIRCMOP6_BC('N',100,'D',30,'maxFE',20000);
    Algorithm = CBS_RegionWGAN_GP('save',0,'outputFcn',@(varargin)[]);
    Algorithm.Solve(Problem);
    Population = Algorithm.result{end,2};
    state = rng;
    Snapshot = Algorithm.boundaryTransferSnapshot();

    assert(Problem.FE == 20000);
    assert(isequal(Problem.CalMetric('IGD',Population), ...
        1.3468921272381897));
    assert(isequal(sum(Population.decs,'all'),1729.8887087629678));
    assert(isequal(sum(Population.objs,'all'),1056.8965815545916));
    assert(isequal(sum(Population.cons,'all'),0));
    assert(state.State(1) == 281257120);
    assert(Snapshot.bracketRows == 222 && ...
        Snapshot.lateRequested == 904 && Snapshot.lateUsed == 904 && ...
        Snapshot.lateFallback == 0);
    fprintf('CBS unique-mainline deterministic fingerprint passed.\n');
end
