function test_CBS_CGAN_smoke()
%TEST_CBS_CGAN_SMOKE Minimal executable check for the CBS-CGAN mainline.

    repoRoot = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
    addpath(genpath(repoRoot));

    rng(7,'twister');
    Algorithm = CBS_CGAN('save',0,'outputFcn',@silentOutput, ...
        'parameter',{1,1,20,2,3,16,1e-4,2e-4,0,1,1,4,1,2,1,0.10,1});
    Problem = DASCMOP1_BC('N',20,'D',30,'maxFE',180);
    Algorithm.Solve(Problem);

    Diag = Algorithm.metric.cbs_cgan_last;
    required = {'bmem_count','boundary_count','train_count','query_count', ...
        'finite_gap_count','inf_gap_count','median_gap','max_gap', ...
        'raw_generated_count','feasible_generated_count', ...
        'segment_distance_min','segment_distance_mean','segment_distance_max', ...
        'segment_width90','segment_width90_ratio','side_rate', ...
        'pair_margin50','ref_cover','query_obj_dist50','query_obj_dist90'};
    for i = 1 : numel(required)
        assert(isfield(Diag,required{i}), ...
            'CBS-CGAN smoke missing diagnostic field: %s',required{i});
    end
    forbidden = {'train_dec_rmse','train_obj_dist50','train_obj_dist90', ...
        'train_feasible_rate','generated_new_ref_hit_rate', ...
        'holdout_ref_hit_rate','novel_useful_rate'};
    for i = 1 : numel(forbidden)
        assert(~isfield(Diag,forbidden{i}), ...
            'CBS-CGAN smoke should not expose misleading field: %s', ...
            forbidden{i});
    end
    assert(Problem.FE >= Problem.maxFE, ...
        'CBS-CGAN smoke should consume the requested short budget.');
    assert(Diag.raw_generated_count <= Diag.query_count, ...
        'With queryPerCondition=1, raw generated count must not exceed QueryC count.');

    fprintf('CBS-CGAN smoke diagnostics:\n');
    fprintf('  BMem nodes: %d\n',Diag.bmem_count);
    fprintf('  current boundaries: %d\n',Diag.boundary_count);
    fprintf('  TrainX/TrainC: %d\n',Diag.train_count);
    fprintf('  QueryC: %d\n',Diag.query_count);
    fprintf('  raw generated: %d\n',Diag.raw_generated_count);
    fprintf('  feasible generated: %d\n',Diag.feasible_generated_count);
    fprintf('  generated-to-boundary distance min/mean/max: %.6g / %.6g / %.6g\n', ...
        Diag.segment_distance_min,Diag.segment_distance_mean, ...
        Diag.segment_distance_max);
end

function silentOutput(~,~)
end
