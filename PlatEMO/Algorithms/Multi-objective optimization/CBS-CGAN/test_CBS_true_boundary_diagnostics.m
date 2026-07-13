function test_CBS_true_boundary_diagnostics()
%TEST_CBS_TRUE_BOUNDARY_DIAGNOSTICS Verify true-boundary diagnostic metrics.

    repoRoot = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
    addpath(genpath(repoRoot));

    testTrueBoundaryMetricsFromFeasibleMask();
    testCoverageUsesBoundaryArcOrder();
    testRunnerSchemasExposeTrueBoundaryFields();
    fprintf('CBS true-boundary diagnostics regressions passed.\n');
end

function testTrueBoundaryMetricsFromFeasibleMask()
    [x,y] = meshgrid(linspace(0,1,11),linspace(0,1,11));
    z = nan(size(x));
    z(x <= 0.6) = 0;
    PF = {x,y,z};
    GeneratedObj = [0.6 0.2;0.5 0.5;0.8 0.8];
    GeneratedCon = [0;0;1];

    Metrics = RunRegionGAN_RC('trueboundarydiagnostics', ...
        GeneratedObj,GeneratedCon,PF,struct('coverBinCount',10));

    assert(isfield(Metrics,'bdist50_true') && ...
        isfield(Metrics,'bwidth90_10_true') && ...
        isfield(Metrics,'bcover_eps_true'), ...
        'True-boundary diagnostics must expose the three core metrics.');
    SignedDist = [0;0.1;-0.2];
    assert(abs(Metrics.bdist50_true - median(abs(SignedDist))) < 1e-12, ...
        'bdist50_true must be the median absolute distance to the PF boundary.');
    expectedWidth = prctile(SignedDist,90) - prctile(SignedDist,10);
    assert(abs(Metrics.bwidth90_10_true - expectedWidth) < 1e-12, ...
        'bwidth90_10_true must be the signed 90-10 percentile width.');
    assert(Metrics.bcover_eps_true > 0 && Metrics.bcover_eps_true <= 1, ...
        'Feasible near-boundary samples must cover at least one boundary bin.');
end

function testCoverageUsesBoundaryArcOrder()
    PF = [0 0;1 0;1 1;0 1];
    GeneratedObj = [0 0;1 0];
    GeneratedCon = [0;0];

    Metrics = RunRegionGAN_RC('trueboundarydiagnostics', ...
        GeneratedObj,GeneratedCon,PF,struct( ...
            'coverBinCount',2,'coverEpsilon',1e-12));

    assert(abs(Metrics.bcover_eps_true - 0.5) < 1e-12, ...
        'bcover_eps_true must bin by boundary arc order, not coordinate order.');
end

function testRunnerSchemasExposeTrueBoundaryFields()
    outDir = fullfile(tempdir,sprintf('cbs_true_boundary_schema_%s', ...
        char(java.util.UUID.randomUUID)));
    Options = struct( ...
        'captureRun',1, ...
        'stageTargets',220);

    [~,EventSummary,~,~,StageSnapshots] = ...
        run_CBS_RegionWGAN_GP_mainline( ...
        outDir,1,"DASCMOP1_BC",20,[],600,1,Options);

    expected = ["bdist50_true","bwidth90_10_true","bcover_eps_true"];
    assert(all(ismember(expected,string(EventSummary.Properties.VariableNames))), ...
        'Event summary must export true-boundary diagnostic fields.');
    assert(all(ismember(expected,string(StageSnapshots.Properties.VariableNames))), ...
        'Stage snapshots must export true-boundary diagnostic fields.');
    if isfolder(outDir)
        rmdir(outDir,'s');
    end
end
