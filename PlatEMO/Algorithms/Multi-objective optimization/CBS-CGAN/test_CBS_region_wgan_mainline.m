function test_CBS_region_wgan_mainline()
%TEST_CBS_REGION_WGAN_MAINLINE Verify the single supported algorithm path.

    repoRoot = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
    addpath(genpath(repoRoot));
    assert(exist('CBS_RegionWGAN_GP','class') == 8);
    assert(exist('CBS_RegionCGAN','class') == 0);
    assert(exist('CBS_CGAN','class') == 0);

    Defaults = CBS_RegionWGAN_GP.mainlineDefaults();
    required = {'trainGap','archiveGap','nGen','zDim','ganIter', ...
        'ganMiniBatch','ganLrD','ganLrG','frontDepth', ...
        'pairNeighborRefRadius','refDivisor','minBoundaryLength', ...
        'gpLambda','nCritic','maxAnchorsPerRef','minGANTrainCount', ...
        'sampleSigma'};
    assert(all(isfield(Defaults,required)));
    forbidden = {'conditionMode','queryPerCondition', ...
        'bandMaxAnchorsPerRef','anchorConsistencyMode'};
    assert(~any(isfield(Defaults,forbidden)));

    [lastMetric,historyMetric,cloudMetric] = ...
        RunRegionGAN_RC('metricnames');
    assert(strcmp(lastMetric,'region_wgan_gp_last'));
    assert(strcmp(historyMetric,'region_wgan_gp_history'));
    assert(strcmp(cloudMetric,'region_wgan_gp_cloud'));
    fprintf('CBS RegionWGAN-GP mainline regressions passed.\n');
end
