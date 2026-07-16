function test_CBS_RegionWGAN_GP_mainline_runner()
%TEST_CBS_REGIONWGAN_GP_MAINLINE_RUNNER Verify IGD-only output and resume.

    repoRoot = fileparts(which('platemo'));
    addpath(genpath(repoRoot));
    outDir = fullfile(tempdir,'cbs_region_wgan_igd_mainline_test');
    if isfolder(outDir); rmdir(outDir,'s'); end
    cleanup = onCleanup(@()removeFolder(outDir));

    [First,returnedDir] = run_CBS_RegionWGAN_GP_mainline( ...
        outDir,1,'LIRCMOP8_BC',20,10,600,1,struct('resume',true));
    assert(string(returnedDir) == string(outDir));
    assert(height(First) == 1 && First.status == "ok");
    assert(First.finalFE == First.maxFE && isfinite(First.IGD));
    assert(isfile(First.task_result_file));
    Loaded = load(First.task_result_file,'TaskResult');
    assert(isequal(fieldnames(Loaded.TaskResult),{'row'}));
    required = ["run_summary.csv","provenance.csv", ...
        "source_manifest.csv","mainline_config.json"];
    assert(all(arrayfun(@(f)isfile(fullfile(outDir,f)),required)));
    forbidden = ["event_summary_all.csv","train_history_all.csv", ...
        "stage_snapshots_all.csv","metric.mat"];
    assert(~any(arrayfun(@(f)isfile(fullfile(outDir,f)),forbidden)));

    Second = run_CBS_RegionWGAN_GP_mainline( ...
        outDir,1,'LIRCMOP8_BC',20,10,600,1,struct('resume',true));
    assert(Second.reused == 1 && Second.IGD == First.IGD);
    assert(isscalar(dir(fullfile(outDir,'LIRCMOP8_BC_run1','attempt_*'))));
    assertThrows(@()run_CBS_RegionWGAN_GP_mainline( ...
        outDir,1,'LIRCMOP8_BC',20,10,601,1,struct('resume',true)), ...
        'CBSRegionGAN:OutputConfigurationMismatch');

    clear cleanup
    removeFolder(outDir);
    fprintf('CBS RegionWGAN-GP IGD runner test passed.\n');
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

function removeFolder(path)
    if isfolder(path); rmdir(path,'s'); end
end
