function test_CBS_mainline_fingerprint()
%TEST_CBS_MAINLINE_FINGERPRINT Pin both retained algorithms independently.

    repoRoot = fileparts(which('platemo'));
    addCBSPaths(repoRoot);

    %% Current PairGuide
    parameters = {500,6,0,64,5,32,1};
    rng(4242,'twister');
    PairProblem = LIRCMOP6_BC('N',30,'D',10,'maxFE',600);
    Current = PairGuide('parameter',parameters, ...
        'save',0,'outputFcn',@(varargin)[]);
    Current.Solve(PairProblem);
    PairPopulation = Current.result{end,2};
    PairSnapshot = Current.guideExperimentSnapshot();
    pairState = rng;

    assert(PairProblem.FE == 600 && ...
        PairSnapshot.arm == 7 && ...
        PairSnapshot.generationMode == "pair_guide" && ...
        PairSnapshot.useMode == "pair_guide" && ...
        PairSnapshot.pairGanEpoch == 0 && ...
        PairSnapshot.pairGuideSchema == "PairGuide");
    rng(4242,'twister');
    ReplayProblem = LIRCMOP6_BC('N',30,'D',10,'maxFE',600);
    Replay = PairGuide('parameter',parameters,'save',0,'outputFcn',@(varargin)[]);
    Replay.Solve(ReplayProblem);
    assert(isequal(PairPopulation.decs,Replay.result{end,2}.decs) && ...
        isequal(PairPopulation.objs,Replay.result{end,2}.objs) && ...
        isequaln(pairState,rng));

    %% Previous CBS_RegionWGAN_GP
    rng(4242,'twister');
    PreviousProblem = LIRCMOP6_BC('N',100,'D',30,'maxFE',20000);
    Previous = CBS_RegionWGAN_GP('save',0,'outputFcn',@(varargin)[]);
    Previous.Solve(PreviousProblem);
    PreviousPopulation = Previous.result{end,2};
    previousState = rng;

    assert(PreviousProblem.FE == 20000);
    assert(isequal(PreviousProblem.CalMetric('IGD',PreviousPopulation), ...
        1.3473816458691643));
    assert(isequal(sum(PreviousPopulation.decs,'all'), ...
        1828.5991360038395));
    assert(isequal(sum(PreviousPopulation.objs,'all'), ...
        625.98054078737562));
    assert(isequal(sum(PreviousPopulation.cons,'all'),0));
    assert(previousState.State(1) == 274547642);

    fprintf('PairGuide and previous-mainline fingerprints passed.\n');
end
