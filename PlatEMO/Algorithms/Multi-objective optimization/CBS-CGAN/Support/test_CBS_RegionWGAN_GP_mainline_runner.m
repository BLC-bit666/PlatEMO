function test_CBS_RegionWGAN_GP_mainline_runner()
%TEST_CBS_REGIONWGAN_GP_MAINLINE_RUNNER Verify native PlatEMO result files.

    repoRoot = fileparts(which('platemo'));
    addpath(genpath(repoRoot));
    testRoot = fullfile(tempdir,'cbs_region_wgan_platemo_result_test');
    outDir = fullfile(testRoot,'Data','CBS_RegionWGAN_GP');
    if isfolder(testRoot); rmdir(testRoot,'s'); end
    cleanup = onCleanup(@()removeFolder(testRoot));

    assertThrows(@()run_CBS_RegionWGAN_GP_mainline( ...
        outDir,9,'LIRCMOP8_BC',10,[],120,991,struct('resume',true)), ...
        'CBSRegionGAN:MainlineWorkerCount');
    assertThrows(@()run_CBS_RegionWGAN_GP_mainline( ...
        outDir,10,"",10,[],120,991,struct('resume',true)), ...
        'CBSRegionGAN:BadProblemNames');

    [First,returnedDir] = run_CBS_RegionWGAN_GP_mainline( ...
        outDir,1,'LIRCMOP8_BC',10,[],120,991,struct('resume',true));
    assert(string(returnedDir) == string(outDir));
    assert(height(First) == 1 && First.status == "ok" && ...
        First.reused == 0 && First.D == 30 && First.M == 2 && ...
        First.finalFE == First.maxFE);

    expectedFile = fullfile(outDir, ...
        'CBS_RegionWGAN_GP_LIRCMOP8_BC_M2_D30_991.mat');
    assert(string(First.result_file) == string(expectedFile));
    assert(isfile(expectedFile), ...
        'The runner did not create the native PlatEMO result file.');
    Variables = whos('-file',expectedFile);
    assert(isequal(sort(string({Variables.name})),["metric","result"]), ...
        'A standard result file must contain only result and metric.');
    Saved = load(expectedFile,'result','metric');
    assert(iscell(Saved.result) && isequal(size(Saved.result),[2,2]), ...
        'save=2 must record the mid-run and final snapshots.');
    assert(Saved.result{end,1} == 120 && ...
        isa(Saved.result{end,2},'SOLUTION'));
    Population = Saved.result{end,2};
    assert(size(Population.decs,2) == 30 && ...
        size(Population.objs,2) == 2 && numel(Population) == 10);
    assert(isequal(sort(string(fieldnames(Saved.metric))), ...
        ["IGD";"runtime"]));
    assert(isequaln(double(Saved.metric.IGD(end)),First.IGD));

    before = dir(expectedFile);
    Second = run_CBS_RegionWGAN_GP_mainline( ...
        outDir,1,'LIRCMOP8_BC',10,[],120,991,struct('resume',true));
    after = dir(expectedFile);
    assert(Second.reused == 1 && isequaln(Second.IGD,First.IGD));
    assert(before.bytes == after.bytes && before.datenum == after.datenum, ...
        'A reusable native result file must not be rewritten.');
    assert(isempty(dir(fullfile(outDir,'**','task_result.mat'))) && ...
        ~isfile(fullfile(outDir,'run_summary.csv')) && ...
        ~isfile(fullfile(outDir,'provenance.csv')));

    clear cleanup
    removeFolder(testRoot);
    fprintf('CBS RegionWGAN-GP native PlatEMO runner test passed.\n');
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
