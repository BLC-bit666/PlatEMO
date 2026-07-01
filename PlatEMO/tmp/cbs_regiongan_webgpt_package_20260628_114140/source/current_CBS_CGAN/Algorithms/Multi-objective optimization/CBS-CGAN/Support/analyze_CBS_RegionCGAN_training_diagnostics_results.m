function [ProblemSummary,StageSummary,StepSummary] = ...
    analyze_CBS_RegionCGAN_training_diagnostics_results(outDir)
%ANALYZE_CBS_REGIONCGAN_TRAINING_DIAGNOSTICS_RESULTS Summarize D/G diagnostics.

    if nargin < 1 || isempty(outDir)
        error('CBSRegionCGAN:MissingOutDir', ...
            'An experiment output directory is required.');
    end
    EventFile = fullfile(outDir,'event_summary_all.csv');
    TrainFile = fullfile(outDir,'train_history_all.csv');
    RunFile = fullfile(outDir,'run_summary.csv');
    E = readtable(EventFile,'Delimiter',',','ReadVariableNames',true, ...
        'TextType','string');
    H = readtable(TrainFile,'Delimiter',',','ReadVariableNames',true, ...
        'TextType','string');
    R = readtable(RunFile,'Delimiter',',','ReadVariableNames',true, ...
        'TextType','string');

    E.stage = assignStages(E);
    ProblemSummary = summarizeByProblem(E,R);
    StageSummary = summarizeByProblemStage(E);
    StepSummary = summarizeByProblemStep(H,[0 10 25 50]);

    writetable(ProblemSummary,fullfile(outDir,'analysis_problem_summary.csv'));
    writetable(StageSummary,fullfile(outDir,'analysis_stage_summary.csv'));
    writetable(StepSummary,fullfile(outDir,'analysis_step_summary.csv'));
end

function Stage = assignStages(E)
    Stage = strings(height(E),1);
    Keys = unique(strcat(E.problem,"#run",string(E.run)),'stable');
    for i = 1 : numel(Keys)
        key = strcat(E.problem,"#run",string(E.run)) == Keys(i);
        rows = find(key);
        maxGen = max(E.generation(rows));
        ratio = E.generation(rows)./maxGen;
        Stage(rows(ratio <= 0.30)) = "early";
        Stage(rows(ratio > 0.30 & ratio <= 0.70)) = "middle";
        Stage(rows(ratio > 0.70)) = "late";
    end
end

function T = summarizeByProblem(E,R)
    Problems = unique(E.problem,'stable');
    Rows = repmat(emptyProblemRow(),numel(Problems)+1,1);
    for i = 1 : numel(Problems)
        idx = E.problem == Problems(i);
        ridx = R.problem == Problems(i);
        Rows(i) = problemRow(Problems(i),E(idx,:),R(ridx,:));
    end
    Rows(end) = problemRow("ALL",E,R);
    T = struct2table(Rows);
end

function Row = problemRow(problem,E,R)
    Row = emptyProblemRow();
    Row.problem = string(problem);
    Row.runs = height(R);
    Row.events = height(E);
    Row.train_rows = sum(E.history_rows);
    Row.runtime_min_median = medianFinite(R.runtime)/60;
    Row.train_count_median = medianFinite(E.train_count);
    Row.feasible_rate_median = medianFinite(E.feasible_rate);
    startDReal = 2*E.start_d_bal_acc - E.start_d_fake_acc;
    finalDReal = 2*E.final_d_bal_acc - E.final_d_fake_acc;
    Row.start_d_bal_median = medianFinite(E.start_d_bal_acc);
    Row.final_d_bal_median = medianFinite(E.final_d_bal_acc);
    Row.final_d_bal_iqr = iqrFinite(E.final_d_bal_acc);
    Row.start_d_real_median = medianFinite(startDReal);
    Row.final_d_real_median = medianFinite(finalDReal);
    Row.final_d_fake_median = medianFinite(E.final_d_fake_acc);
    Row.start_g_fool_median = medianFinite(E.start_g_fool_rate);
    Row.final_g_fool_median = medianFinite(E.final_g_fool_rate);
    Row.final_g_fool_iqr = iqrFinite(E.final_g_fool_rate);
    Row.delta_g_fool_median = medianFinite(E.delta_g_fool_rate);
    Row.delta_d_fake_median = medianFinite(E.delta_d_fake_acc);
    Row.score_gap_median = medianFinite(E.score_gap);
    Row.loss_d_median = medianFinite(E.final_loss_d);
    Row.loss_g_median = medianFinite(E.final_loss_g);
    Row.g_improve_rate = rate(E.delta_g_fool_rate > 0.05);
    Row.g_worse_rate = rate(E.delta_g_fool_rate < -0.05);
    Row.high_fool_rate = rate(E.final_g_fool_rate >= 0.50);
    Row.fake_bias_rate = rate(finalDReal <= 0.25 & ...
        E.final_d_fake_acc >= 0.75);
    Row.strong_d_rate = rate(E.final_d_bal_acc >= 0.75 & ...
        E.final_g_fool_rate <= 0.25);
    Row.balance_like_rate = rate(finalDReal >= 0.40 & ...
        finalDReal <= 0.60 & E.final_d_fake_acc >= 0.40 & ...
        E.final_d_fake_acc <= 0.60 & E.final_g_fool_rate >= 0.40 & ...
        E.final_g_fool_rate <= 0.60 & abs(E.score_gap) <= 0.05);
    Row.diagnosis = diagnose(Row);
end

function T = summarizeByProblemStage(E)
    Problems = unique(E.problem,'stable');
    Stages = ["early","middle","late"];
    Rows = repmat(emptyStageRow(),numel(Problems)*numel(Stages),1);
    row = 0;
    for i = 1 : numel(Problems)
        for s = 1 : numel(Stages)
            row = row + 1;
            idx = E.problem == Problems(i) & E.stage == Stages(s);
            Rows(row) = stageRow(Problems(i),Stages(s),E(idx,:));
        end
    end
    T = struct2table(Rows);
end

function Row = stageRow(problem,stage,E)
    Row = emptyStageRow();
    Row.problem = string(problem);
    Row.stage = string(stage);
    Row.events = height(E);
    Row.train_count_median = medianFinite(E.train_count);
    Row.final_d_bal_median = medianFinite(E.final_d_bal_acc);
    finalDReal = 2*E.final_d_bal_acc - E.final_d_fake_acc;
    Row.final_d_real_median = medianFinite(finalDReal);
    Row.final_d_fake_median = medianFinite(E.final_d_fake_acc);
    Row.final_g_fool_median = medianFinite(E.final_g_fool_rate);
    Row.delta_g_fool_median = medianFinite(E.delta_g_fool_rate);
    Row.score_gap_median = medianFinite(E.score_gap);
    Row.strong_d_rate = rate(E.final_d_bal_acc >= 0.75 & ...
        E.final_g_fool_rate <= 0.25);
    Row.fake_bias_rate = rate(finalDReal <= 0.25 & ...
        E.final_d_fake_acc >= 0.75);
    Row.balance_like_rate = rate(finalDReal >= 0.40 & ...
        finalDReal <= 0.60 & E.final_d_fake_acc >= 0.40 & ...
        E.final_d_fake_acc <= 0.60 & E.final_g_fool_rate >= 0.40 & ...
        E.final_g_fool_rate <= 0.60 & abs(E.score_gap) <= 0.05);
end

function T = summarizeByProblemStep(H,Steps)
    Problems = unique(H.problem,'stable');
    Rows = repmat(emptyStepRow(),numel(Problems)*numel(Steps),1);
    row = 0;
    for i = 1 : numel(Problems)
        for s = 1 : numel(Steps)
            row = row + 1;
            idx = H.problem == Problems(i) & H.step == Steps(s);
            Rows(row) = stepRow(Problems(i),Steps(s),H(idx,:));
        end
    end
    T = struct2table(Rows);
end

function Row = stepRow(problem,step,H)
    Row = emptyStepRow();
    Row.problem = string(problem);
    Row.step = double(step);
    Row.rows = height(H);
    Row.loss_d_median = medianFinite(H.loss_d);
    Row.loss_g_median = medianFinite(H.loss_g);
    Row.d_real_acc_median = medianFinite(H.d_real_acc);
    Row.d_fake_acc_median = medianFinite(H.d_fake_acc);
    Row.d_bal_acc_median = medianFinite(H.d_bal_acc);
    Row.g_fool_rate_median = medianFinite(H.g_fool_rate);
    Row.score_real_mean_median = medianFinite(H.score_real_mean);
    Row.score_fake_mean_median = medianFinite(H.score_fake_mean);
    Row.score_gap_median = Row.score_real_mean_median - ...
        Row.score_fake_mean_median;
end

function text = diagnose(Row)
    if Row.fake_bias_rate > 0.50 && Row.final_g_fool_median < 0.10
        text = "D_fake_bias_G_not_fooling";
    elseif Row.strong_d_rate > 0.25 && Row.final_g_fool_median < 0.30
        text = "D_overpowering";
    elseif Row.balance_like_rate > 0.35 && Row.high_fool_rate > 0.35
        text = "near_balance";
    elseif Row.g_improve_rate > 0.45 && Row.final_g_fool_median >= 0.35
        text = "G_learning";
    elseif Row.g_worse_rate > Row.g_improve_rate
        text = "G_regresses";
    else
        text = "mixed";
    end
end

function Row = emptyProblemRow()
    Row = struct( ...
        'problem',"", ...
        'runs',0, ...
        'events',0, ...
        'train_rows',0, ...
        'runtime_min_median',NaN, ...
        'train_count_median',NaN, ...
        'feasible_rate_median',NaN, ...
        'start_d_bal_median',NaN, ...
        'final_d_bal_median',NaN, ...
        'final_d_bal_iqr',NaN, ...
        'start_d_real_median',NaN, ...
        'final_d_real_median',NaN, ...
        'final_d_fake_median',NaN, ...
        'start_g_fool_median',NaN, ...
        'final_g_fool_median',NaN, ...
        'final_g_fool_iqr',NaN, ...
        'delta_g_fool_median',NaN, ...
        'delta_d_fake_median',NaN, ...
        'score_gap_median',NaN, ...
        'loss_d_median',NaN, ...
        'loss_g_median',NaN, ...
        'g_improve_rate',NaN, ...
        'g_worse_rate',NaN, ...
        'high_fool_rate',NaN, ...
        'fake_bias_rate',NaN, ...
        'strong_d_rate',NaN, ...
        'balance_like_rate',NaN, ...
        'diagnosis',"");
end

function Row = emptyStageRow()
    Row = struct( ...
        'problem',"", ...
        'stage',"", ...
        'events',0, ...
        'train_count_median',NaN, ...
        'final_d_bal_median',NaN, ...
        'final_d_real_median',NaN, ...
        'final_d_fake_median',NaN, ...
        'final_g_fool_median',NaN, ...
        'delta_g_fool_median',NaN, ...
        'score_gap_median',NaN, ...
        'fake_bias_rate',NaN, ...
        'strong_d_rate',NaN, ...
        'balance_like_rate',NaN);
end

function Row = emptyStepRow()
    Row = struct( ...
        'problem',"", ...
        'step',NaN, ...
        'rows',0, ...
        'loss_d_median',NaN, ...
        'loss_g_median',NaN, ...
        'd_real_acc_median',NaN, ...
        'd_fake_acc_median',NaN, ...
        'd_bal_acc_median',NaN, ...
        'g_fool_rate_median',NaN, ...
        'score_real_mean_median',NaN, ...
        'score_fake_mean_median',NaN, ...
        'score_gap_median',NaN);
end

function value = medianFinite(X)
    X = double(X(:));
    X = X(isfinite(X));
    if isempty(X)
        value = NaN;
    else
        value = median(X);
    end
end

function value = iqrFinite(X)
    X = double(X(:));
    X = X(isfinite(X));
    if isempty(X)
        value = NaN;
    else
        value = percentileFinite(X,75) - percentileFinite(X,25);
    end
end

function value = percentileFinite(X,p)
    X = sort(double(X(:)));
    X = X(isfinite(X));
    if isempty(X)
        value = NaN;
        return;
    end
    pos = 1 + (numel(X) - 1)*double(p)/100;
    lo = floor(pos);
    hi = ceil(pos);
    if lo == hi
        value = X(lo);
    else
        value = X(lo) + (pos - lo)*(X(hi) - X(lo));
    end
end

function value = rate(mask)
    if isempty(mask)
        value = NaN;
    else
        value = mean(double(mask(:)));
    end
end
