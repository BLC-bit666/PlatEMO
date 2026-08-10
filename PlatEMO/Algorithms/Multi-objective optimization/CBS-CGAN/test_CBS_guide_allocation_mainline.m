function test_CBS_guide_allocation_mainline()
%TEST_CBS_GUIDE_ALLOCATION_MAINLINE Verify the fixed 14+6 query policy.

    repoRoot = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
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
    fprintf('CBS mainline 14+6 query allocation passed.\n');
end
