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
    assert(isequal(PairProblem.CalMetric('IGD',PairPopulation), ...
        1.3996342555488801) && ...
        isequal(sum(PairPopulation.decs,'all'),177.89171819126977) && ...
        isequal(sum(PairPopulation.objs,'all'),264.07215397883789) && ...
        isequal(sum(PairPopulation.cons,'all'),0) && ...
        pairState.State(1) == 3724164776);

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
