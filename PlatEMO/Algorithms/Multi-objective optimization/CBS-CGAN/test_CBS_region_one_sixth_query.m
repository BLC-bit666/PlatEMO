function test_CBS_region_one_sixth_query()
%TEST_CBS_REGION_ONE_SIXTH_QUERY Verify the sole query policy.

    repoRoot = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
    addpath(genpath(repoRoot));

    testFixedSeedFixture();
    testBudgetScalesWithNGen();
    testPopulatedFallback();
    testInvalidReferenceFails();
    testMainlineDefaults();
    fprintf('CBS one-sixth frontier query regressions passed.\n');
end

function testFixedSeedFixture()
    [W,QueryC,Info] = queryFixture();
    rng(1709,'twister');
    [SampleC,Counts,SampleRefs,DiagC,DiagRefs,Groups] = ...
        RunRegionGAN_RC('regionquerysamples',QueryC,Info,W,10);

    assert(isequal(SampleRefs,[5;2;2;4;2;4;2;2;5;4]));
    assert(isequal(Counts,[5;3;0;0;2]));
    assert(isequal(Groups,[2;1;1;1;1;1;1;1;2;1]));
    assert(isequal(SampleC,W(SampleRefs,:)));
    assert(isequal(DiagRefs,[2;4;1;3;5]));
    assert(isequal(DiagC,W(DiagRefs,:)));
    assert(RunRegionGAN_RC('regionquerypoolcount',QueryC,Info,W) == 5);
end

function testBudgetScalesWithNGen()
    [W,QueryC,Info] = queryFixture();
    budgets = [12 30 60];
    expectedFrontier = [2 5 10];
    for i = 1 : numel(budgets)
        rng(6100 + i,'twister');
        [SampleC,Counts,SampleRefs,DiagC,DiagRefs,Groups] = ...
            RunRegionGAN_RC('regionquerysamples', ...
            QueryC,Info,W,budgets(i));
        assert(size(SampleC,1) == budgets(i));
        assert(numel(SampleRefs) == budgets(i));
        assert(sum(Groups == 2) == expectedFrontier(i));
        assert(sum(Groups == 1) == budgets(i) - expectedFrontier(i));
        assert(~any(Groups == 3));
        assert(sum(Counts) == budgets(i));
        assert(numel(Counts) == size(DiagC,1));
        assert(numel(DiagRefs) == size(DiagC,1));
    end
end

function testPopulatedFallback()
    [W,QueryC,Info] = queryFixture();
    Info.allQueryConditions = QueryC;
    Info.allQueryRegions = Info.queryRegions;
    rng(1301,'twister');
    [~,Counts,SampleRefs,~,~,Groups] = RunRegionGAN_RC( ...
        'regionquerysamples',QueryC,Info,W,30);
    assert(numel(SampleRefs) == 30 && sum(Counts) == 30);
    assert(all(Groups == 1));
end

function testInvalidReferenceFails()
    [W,QueryC,Info] = queryFixture();
    Info.allQueryRegions(1) = NaN;
    didThrow = false;
    try
        RunRegionGAN_RC('regionquerysamples',QueryC,Info,W,1);
    catch Error
        didThrow = strcmp(Error.identifier,'CBSRegionGAN:BadSampleRef');
    end
    assert(didThrow);
end

function testMainlineDefaults()
    Defaults = CBS_RegionWGAN_GP.mainlineDefaults();
    assert(Defaults.ganIter == 100 && Defaults.nCritic == 2);
    assert(Defaults.sampleSigma == 0.3);
end

function [W,QueryC,Info] = queryFixture()
    W = [1 0;0.75 0.25;0.5 0.5;0.25 0.75;0 1];
    queryRefs = [2;4];
    QueryC = W(queryRefs,:);
    Info = struct( ...
        'queryRegions',queryRefs, ...
        'allQueryConditions',W, ...
        'allQueryRegions',(1:size(W,1))');
end
