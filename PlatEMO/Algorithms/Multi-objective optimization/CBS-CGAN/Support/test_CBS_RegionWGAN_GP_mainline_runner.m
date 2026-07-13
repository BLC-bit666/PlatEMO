function test_CBS_RegionWGAN_GP_mainline_runner()
%TEST_CBS_REGIONWGAN_GP_MAINLINE_RUNNER Smoke-test the non-visual runner.

    repoRoot = fileparts(which('platemo'));
    addpath(genpath(repoRoot));
    outDir = fullfile(tempdir,'cbs_region_wgan_mainline_runner_test');
    if isfolder(outDir)
        rmdir(outDir,'s');
    end
    cleanup = onCleanup(@()removeFolder(outDir));
    Options = struct( ...
        'captureRun',0, ...
        'captureWGANTrainHistory',false, ...
        'wganMappingDiagnostics',false, ...
        'bmemLearnabilityDiagnostics',false);
    [Summary,Events,History,~,Stages] = ...
        run_CBS_RegionWGAN_GP_mainline( ...
        outDir,1,"LIRCMOP8_BC",20,10,1000,1,Options);

    assert(height(Summary) == 1 && Summary.status == "ok");
    assert(Summary.finalFE == Summary.maxFE, ...
        'Mainline runner must enforce exact maxFE without overshoot.');
    assert(~isempty(Events) && isempty(History));
    assert(isempty(Stages));
    required = ["run_summary.csv","event_summary_all.csv", ...
        "train_history_all.csv","provenance.csv","source_manifest.csv"];
    for i = 1 : numel(required)
        assert(isfile(fullfile(outDir,required(i))));
    end
    assert(~isfile(fullfile(outDir,'figure_manifest.csv')));
    clear cleanup
    removeFolder(outDir);
    fprintf('CBS RegionWGAN-GP mainline runner smoke test passed.\n');
end

function removeFolder(path)
    if isfolder(path)
        rmdir(path,'s');
    end
end
