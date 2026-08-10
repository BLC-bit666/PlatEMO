function test_CBS_RegionWGAN_GP_mainline_runner()
%TEST_CBS_REGIONWGAN_GP_MAINLINE_RUNNER Verify the 100K/200K contract.

    repoRoot = fileparts(which('platemo'));
    addpath(fileparts(fileparts(mfilename('fullpath'))),'-begin');
    addCBSPaths(repoRoot);
    testRoot = fullfile(tempdir,'cbs_region_wgan_mainline_runner_test');
    outDir = fullfile(testRoot,'Data','CBS_RegionWGAN_GP');
    if isfolder(testRoot); rmdir(testRoot,'s'); end
    mkdir(outDir);
    cleanup = onCleanup(@()removeFolder(testRoot));

    assertThrows(@()run_CBS_RegionWGAN_GP_mainline( ...
        outDir,9,'LIRCMOP8_BC',10,[],200000,991, ...
        struct('resume',true)),'CBSRegionGAN:MainlineWorkerCount');
    assertThrows(@()run_CBS_RegionWGAN_GP_mainline( ...
        outDir,1,'LIRCMOP8_BC',10,[],100000,991, ...
        struct('resume',true)),'CBSRegionGAN:MainlineMaxFE');
    assertThrows(@()run_CBS_RegionWGAN_GP_mainline( ...
        outDir,1,"",10,[],200000,991, ...
        struct('resume',true)),'CBSRegionGAN:BadProblemNames');
    assertThrows(@()run_CBS_RegionWGAN_GP_mainline( ...
        outDir,1,'LIRCMOP8_BC',10,[],200000,991, ...
        struct('algorithm','anything')), ...
        'CBSRegionGAN:BadMainlineOptions');

    %% Supply a native fixture so resume validation needs no 200K run
    Population = SOLUTION(zeros(10,30),zeros(10,2),zeros(10,1));
    result = {99960,Population;200000,Population};
    metric = struct('runtime',[1 2],'IGD',[0.5 0.25]);
    resultFile = fullfile(outDir, ...
        'CBS_RegionWGAN_GP_LIRCMOP8_BC_M2_D30_991.mat');
    save(resultFile,'result','metric');

    [Summary,returnedDir] = run_CBS_RegionWGAN_GP_mainline( ...
        outDir,1,'LIRCMOP8_BC',10,[],200000,991, ...
        struct('resume',true));
    assert(string(returnedDir) == string(outDir));
    assert(height(Summary) == 1 && Summary.status == "ok" && ...
        Summary.reused == 1 && Summary.FE100K == 99960 && ...
        Summary.IGD100K == 0.5 && Summary.FE200K == 200000 && ...
        Summary.IGD200K == 0.25);
    expected = ["problem","run","seed","N","D","M", ...
        "FE100K","IGD100K","FE200K","IGD200K","status", ...
        "reused","result_file","error_identifier","error_message"];
    assert(isequal(string(Summary.Properties.VariableNames),expected));

    clear cleanup
    removeFolder(testRoot);
    fprintf('CBS mainline runner 100K/200K contract passed.\n');
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
