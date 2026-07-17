function test_CBS_region_one_sixth_query()
%TEST_CBS_REGION_ONE_SIXTH_QUERY Verify the sole query policy.

    repoRoot = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
    addpath(genpath(repoRoot));
    W = [1 0;0.75 0.25;0.5 0.5;0.25 0.75;0 1];
    populated = [2;4];

    rng(1709,'twister');
    [SampleC,SampleRefs] = RunRegionGAN_RC( ...
        'regionquerysamples',populated,W,10);
    assert(isequal(SampleRefs,[5;2;2;4;2;4;2;2;5;4]));
    assert(isequal(SampleC,W(SampleRefs,:)));
    assert(sum(ismember(SampleRefs,[1;3;5])) == 2);

    budgets = [12 30 60];
    for i = 1 : numel(budgets)
        rng(6100+i,'twister');
        [SampleC,SampleRefs] = RunRegionGAN_RC( ...
            'regionquerysamples',populated,W,budgets(i));
        assert(size(SampleC,1) == budgets(i));
        assert(sum(ismember(SampleRefs,[1;3;5])) == round(budgets(i)/6));
    end

    rng(1301,'twister');
    [~,SampleRefs] = RunRegionGAN_RC( ...
        'regionquerysamples',(1:size(W,1))',W,30);
    assert(numel(SampleRefs) == 30 && ...
        all(ismember(SampleRefs,(1:size(W,1))')));

    assertThrows(@()RunRegionGAN_RC( ...
        'regionquerysamples',[2;NaN],W,1), ...
        'CBSRegionGAN:BadSampleRef');
    Defaults = CBS_RegionWGAN_GP.mainlineDefaults();
    assert(Defaults.ganIter == 100 && Defaults.nCritic == 4 && ...
        Defaults.ganMiniBatch == 32 && Defaults.nGen == 30 && ...
        Defaults.ganStopFraction == 0.5);
    fprintf('CBS one-sixth frontier query regressions passed.\n');
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
