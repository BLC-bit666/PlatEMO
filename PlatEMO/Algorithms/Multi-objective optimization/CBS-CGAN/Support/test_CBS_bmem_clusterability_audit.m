function test_CBS_bmem_clusterability_audit()
%TEST_CBS_BMEM_CLUSTERABILITY_AUDIT Integration test with locked toy modes.

    root = fullfile(tempdir,'cbs_bmem_clusterability_audit_test');
    if isfolder(root)
        rmdir(root,'s');
    end
    inputRoot = fullfile(root,'input');
    outputRoot = fullfile(root,'output');
    runFolder = fullfile(inputRoot,'LIRCMOP5_BC_run1');
    mkdir(runFolder);
    cleanup = onCleanup(@()removeFolder(root));

    T = syntheticHistory();
    historyFile = fullfile(runFolder,'bmem_history.csv');
    writetable(T,historyFile);
    Summary = table("LIRCMOP5_BC",1,1,100,30,2,100000,"ok", ...
        string(historyFile),'VariableNames',{'problem','run','seed','N','D', ...
        'M','maxFE','status','bmem_history_file'});
    writetable(Summary,fullfile(inputRoot,'run_summary.csv'));

    [G,S,~,~,Decision] = ...
        analyze_CBS_RegionWGAN_GP_bmem_clusterability(inputRoot,outputRoot);
    target = G(G.ref == 10,:);
    assert(height(target) == 2);
    assert(all(target.duplicate_count == 1));
    assert(all(target.separated_candidate));
    assert(~target.prune_eligible(1) && target.prune_eligible(2));
    assert(S.dedup_count(2) == 4 && S.gated_count(2) == 3);
    assert(Decision.next_step == "validate_exact_sample_id_dedup_only");
    assert(isfile(fullfile(outputRoot,'audit_design.csv')));
    assert(isfile(fullfile(outputRoot,'audit_manifest.csv')));
    clear cleanup
    removeFolder(root);
    fprintf('CBS BMem clusterability audit integration test passed.\n');
end

function T = syntheticHistory()
    patternRef = [9;10;10;10;11];
    ref = [patternRef;patternRef];
    snapshot_index = [ones(5,1);2*ones(5,1)];
    generation = snapshot_index;
    fe = snapshot_index*100;
    bmem_row = repmat((1:5)',2,1);
    sample_id_f = repmat(["N1";"A";"A";"B";"N2"],2,1);
    gap = repmat([0.2;0.1;0.1;0.5;0.2],2,1);
    age_f = repmat([0;0;0;1;0],2,1);
    front_rank_f = repmat([1;1;1;2;1],2,1);
    candidate_row_f = (1:10)';
    T = table(snapshot_index,generation,fe,bmem_row,sample_id_f,ref, ...
        gap,age_f,front_rank_f,candidate_row_f);

    A = zeros(1,30);
    B = ones(1,30);
    N1 = 0.01*ones(1,30);
    N2 = 0.99*ones(1,30);
    X = repmat([N1;A;A;B;N2],2,1);
    Y = repmat([1.0,2.0;1.1,2.1;1.1,2.1;3.0,3.0;2.9,3.1],2,1);
    for j = 1 : 30
        T.(sprintf('x_f%d',j)) = X(:,j);
    end
    T.y_f1 = Y(:,1);
    T.y_f2 = Y(:,2);
end

function removeFolder(path)
    if isfolder(path)
        rmdir(path,'s');
    end
end
