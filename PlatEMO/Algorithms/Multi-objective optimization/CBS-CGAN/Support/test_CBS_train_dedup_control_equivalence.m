function test_CBS_train_dedup_control_equivalence()
%TEST_CBS_TRAIN_DEDUP_CONTROL_EQUIVALENCE Default and explicit off agree.

    repoRoot = fileparts(which('platemo'));
    addpath(genpath(repoRoot));
    root = fullfile(tempdir,'cbs_train_dedup_control_equivalence_test');
    if isfolder(root); rmdir(root,'s'); end
    cleanup = onCleanup(@()removeFolder(root));
    Common = struct('captureRun',0,'captureWGANTrainHistory',false, ...
        'wganMappingDiagnostics',false, ...
        'bmemLearnabilityDiagnostics',false, ...
        'schemaVersion',"dedup_control_equivalence_v1");
    [S0,E0] = run_CBS_RegionWGAN_GP_mainline( ...
        fullfile(root,'default'),1,"LIRCMOP8_BC",20,10,600,1,Common);
    Explicit = Common;
    Explicit.trainDedupMode = "off";
    [S1,E1] = run_CBS_RegionWGAN_GP_mainline( ...
        fullfile(root,'explicit_off'),1,"LIRCMOP8_BC",20,10,600,1,Explicit);

    timeFields = startsWith(string(E0.Properties.VariableNames),'time_');
    assert(isequaln(E0(:,~timeFields),E1(:,~timeFields)), ...
        'Default and explicit-off event semantics must match exactly.');
    files = {'final_p1_file','final_p2_file','final_feasible_nd_file', ...
        'query_samples_file'};
    for i = 1 : numel(files)
        A = readtable(char(S0.(files{i})),'Delimiter',',','TextType','string');
        B = readtable(char(S1.(files{i})),'Delimiter',',','TextType','string');
        assert(isequaln(A,B),'Core export differs: %s.',files{i});
    end
    clear cleanup
    removeFolder(root);
    fprintf('CBS train-dedup control equivalence test passed.\n');
end

function removeFolder(path)
    if isfolder(path); rmdir(path,'s'); end
end
