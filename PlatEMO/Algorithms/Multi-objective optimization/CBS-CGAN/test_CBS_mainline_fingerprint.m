function test_CBS_mainline_fingerprint()
%TEST_CBS_MAINLINE_FINGERPRINT Pin the deterministic unique mainline.

    repoRoot = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
    addpath(genpath(repoRoot));
    rng(4242,'twister');
    Problem = LIRCMOP6_BC('N',100,'D',30,'maxFE',20000);
    Algorithm = CBS_RegionWGAN_GP('save',0,'outputFcn',@(varargin)[]);
    Algorithm.Solve(Problem);
    Population = Algorithm.result{end,2};
    state = rng;

    assert(Problem.FE == 20000);
    assert(isequal(Problem.CalMetric('IGD',Population), ...
        1.346550324710176));
    assert(isequal(sum(Population.decs,'all'),1883.8784633646248));
    assert(isequal(sum(Population.objs,'all'),443.43527478704067));
    assert(isequal(sum(Population.cons,'all'),0));
    assert(state.State(1) == 3522559217);
    fprintf('CBS unique-mainline deterministic fingerprint passed.\n');
end
