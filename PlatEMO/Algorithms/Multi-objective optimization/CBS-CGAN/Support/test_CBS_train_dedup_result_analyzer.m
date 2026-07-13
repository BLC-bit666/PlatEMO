function test_CBS_train_dedup_result_analyzer()
%TEST_CBS_TRAIN_DEDUP_RESULT_ANALYZER Verify paired promotion bookkeeping.

    root = fullfile(tempdir,'cbs_train_dedup_analyzer_test');
    if isfolder(root); rmdir(root,'s'); end
    mkdir(fullfile(root,'A_control'));
    mkdir(fullfile(root,'B_exact_ref_x'));
    cleanup = onCleanup(@()removeFolder(root));
    [Runs,A,B] = syntheticResults();
    writetable(Runs,fullfile(root,'A_control','run_summary.csv'));
    writetable(Runs,fullfile(root,'B_exact_ref_x','run_summary.csv'));
    writetable(A,fullfile(root,'A_control','event_summary_all.csv'));
    writetable(B,fullfile(root,'B_exact_ref_x','event_summary_all.csv'));
    [C,S,D] = analyze_CBS_RegionWGAN_GP_train_dedup_results(root);
    assert(height(C) == 18 && height(S) == 13);
    assert(D.status == "promote" && D.primary_gate && D.mechanism_integrity);
    assert(isfile(fullfile(root,'promotion_decision.csv')));
    clear cleanup
    removeFolder(root);
    fprintf('CBS train-dedup result analyzer test passed.\n');
end

function [Runs,A,B] = syntheticResults()
    problems = repelem(["LIRCMOP5_BC";"LIRCMOP6_BC"; ...
        "LIRCMOP7_BC";"LIRCMOP8_BC";"LIRCMOP9_BC"; ...
        "LIRCMOP10_BC"],3);
    run = repmat((1:3)',6,1);
    seed = run;
    status = repmat("ok",18,1);
    finalFE = 100000*ones(18,1);
    maxFE = finalFE;
    runtime = 100*ones(18,1);
    Runs = table(problems,run,seed,status,finalFE,maxFE,runtime, ...
        'VariableNames',{'problem','run','seed','status','finalFE', ...
        'maxFE','runtime'});
    A = eventTable(problems,run,seed,false);
    B = eventTable(problems,run,seed,true);
end

function T = eventTable(problem,run,seed,dedup)
    n = numel(run);
    train_count_raw = 50*ones(n,1);
    train_count = train_count_raw - 5*double(dedup);
    train_exact_duplicate_count = 5*ones(n,1);
    train_removed_duplicate_count = 5*double(dedup)*ones(n,1);
    train_dedup_enabled = double(dedup)*ones(n,1);
    raw_generated_count = 30*ones(n,1);
    if dedup
        bdist50_true = 0.09*ones(n,1);
        bwidth90_10_true = 0.18*ones(n,1);
        diversity = 1.5*ones(n,1);
        anchor = 0.4*ones(n,1);
        survival = 0.02*ones(n,1);
        groupDist = 0.09*ones(n,1);
        groupWidth = 0.18*ones(n,1);
    else
        bdist50_true = 0.10*ones(n,1);
        bwidth90_10_true = 0.20*ones(n,1);
        diversity = 2.0*ones(n,1);
        anchor = 0.3*ones(n,1);
        survival = 0.01*ones(n,1);
        groupDist = 0.10*ones(n,1);
        groupWidth = 0.20*ones(n,1);
    end
    bcover_eps_true = 0.5*ones(n,1);
    query_populated_bdist50_true = groupDist;
    query_populated_bwidth90_10_true = groupWidth;
    query_populated_bcover_eps_true = bcover_eps_true;
    query_frontier_bdist50_true = groupDist;
    query_frontier_bwidth90_10_true = groupWidth;
    query_frontier_bcover_eps_true = bcover_eps_true;
    generated_train_dec_diversity_ratio90 = diversity;
    generated_queried_anchor_utilization_rate = anchor;
    offspringG_survive_union_rate = survival;
    T = table(problem,run,seed,train_count,train_count_raw, ...
        train_exact_duplicate_count,train_removed_duplicate_count, ...
        train_dedup_enabled,raw_generated_count,bdist50_true, ...
        bwidth90_10_true,bcover_eps_true, ...
        generated_train_dec_diversity_ratio90, ...
        generated_queried_anchor_utilization_rate, ...
        offspringG_survive_union_rate, ...
        query_populated_bdist50_true,query_populated_bwidth90_10_true, ...
        query_populated_bcover_eps_true,query_frontier_bdist50_true, ...
        query_frontier_bwidth90_10_true,query_frontier_bcover_eps_true);
end

function removeFolder(path)
    if isfolder(path); rmdir(path,'s'); end
end
