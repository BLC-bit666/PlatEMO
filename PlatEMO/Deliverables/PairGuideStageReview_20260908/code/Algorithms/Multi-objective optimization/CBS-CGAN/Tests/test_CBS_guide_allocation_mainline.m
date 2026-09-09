function test_CBS_guide_allocation_mainline()
%TEST_CBS_GUIDE_ALLOCATION_MAINLINE Verify retained legacy critic utilities.

    repoRoot = fileparts(which('platemo'));
    addCBSPaths(repoRoot);
    [W,~] = UniformPoint(50,2);
    populated = (5:10)';

    rng(51,'twister');
    [C,R] = RunRegionGAN_RC('regionquerysamples',populated,W,20);
    isPopulated = ismember(R,populated);
    assert(size(C,1) == 20 && numel(R) == 20);
    assert(sum(isPopulated) == 14 && sum(~isPopulated) == 6);
    assert(isequal(C,W(R,:)));

    [~,Rall] = RunRegionGAN_RC('regionquerysamples', ...
        (1:size(W,1))',W,20);
    assert(all(ismember(Rall,(1:size(W,1))')), ...
        'Frontier slots must reflow when no empty neighbor exists.');

    rng(52,'twister');
    [Cbalanced,Rbalanced] = RunRegionGAN_RC( ...
        'balancedquerysamples',W,500);
    counts = accumarray(Rbalanced,1,[size(W,1),1]);
    assert(size(Cbalanced,1) == 500 && ...
        isequal(Cbalanced,W(Rbalanced,:)) && all(counts > 0) && ...
        max(counts)-min(counts) <= 1);

    rawRefs = repelem((1:3)',4);
    rawScore = [5;5;4;3;-10;-20;-30;-40;0.1;0.2;0.3;0.4];
    rawDec = reshape(1:24,12,2);
    [filteredDec,filteredRefs,keepIdx,percentile] = RunRegionGAN_RC( ...
        'conditioncriticfilter',rawDec,rawRefs,rawScore,6,W(1:3,:));
    assert(isequal(keepIdx,[1;2;5;6;11;12]) && ...
        isequal(filteredDec,rawDec(keepIdx,:)) && ...
        isequal(filteredRefs,rawRefs(keepIdx)) && ...
        all(percentile(keepIdx) >= 0.625));
    fprintf('CBS retained legacy critic utility regressions passed.\n');
end
