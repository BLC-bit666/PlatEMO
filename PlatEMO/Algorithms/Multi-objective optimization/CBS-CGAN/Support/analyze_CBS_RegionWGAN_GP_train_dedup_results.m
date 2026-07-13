function [Comparison,Stats,Decision] = ...
        analyze_CBS_RegionWGAN_GP_train_dedup_results(rootDir)
%ANALYZE_CBS_REGIONWGAN_GP_TRAIN_DEDUP_RESULTS Analyze the locked A/B test.
%   Uses paired problem-run medians. No figures or problem evaluations.

    if nargin < 1 || isempty(rootDir)
        error('CBSRegionGAN:MissingDedupExperimentRoot', ...
            'Provide the completed train-dedup experiment root.');
    end
    rootDir = char(string(rootDir));
    A = loadArm(fullfile(rootDir,'A_control'));
    B = loadArm(fullfile(rootDir,'B_exact_ref_x'));
    assertArmIntegrity(A,"A_control");
    assertArmIntegrity(B,"B_exact_ref_x");

    Keys = innerjoin(A.runs(:,{'problem','run','seed'}), ...
        B.runs(:,{'problem','run','seed'}), ...
        'Keys',{'problem','run','seed'});
    Keys = unique(Keys(:,{'problem','run','seed'}),'rows','stable');
    if height(Keys) ~= 18
        error('CBSRegionGAN:BadDedupPairCount', ...
            'Expected 18 paired problem-runs, found %d.',height(Keys));
    end
    rows = repmat(emptyComparisonRow(),height(Keys),1);
    for i = 1 : height(Keys)
        rows(i) = summarizePair(Keys(i,:),A,B);
    end
    Comparison = struct2table(rows);
    Stats = pairedStatistics(Comparison);
    Decision = decide(Comparison,Stats,A.runs,B.runs);
    writetable(Comparison,fullfile(rootDir,'run_comparison.csv'));
    writetable(Stats,fullfile(rootDir,'paired_statistics.csv'));
    writetable(Decision,fullfile(rootDir,'promotion_decision.csv'));
end

function Arm = loadArm(path)
    Arm = struct();
    Arm.runs = readCSV(fullfile(path,'run_summary.csv'));
    Arm.events = readCSV(fullfile(path,'event_summary_all.csv'));
end

function T = readCSV(path)
    if ~isfile(path)
        error('CBSRegionGAN:MissingDedupResult','Missing result file: %s',path);
    end
    T = readtable(path,'Delimiter',',','TextType','string', ...
        'VariableNamingRule','preserve');
end

function assertArmIntegrity(Arm,name)
    requiredRuns = ["problem","run","seed","status","finalFE", ...
        "maxFE","runtime"];
    requiredEvents = ["problem","run","seed","train_count", ...
        "train_count_raw","train_exact_duplicate_count", ...
        "train_removed_duplicate_count","train_dedup_enabled", ...
        "raw_generated_count","bdist50_true","bwidth90_10_true", ...
        "bcover_eps_true"];
    requireColumns(Arm.runs,requiredRuns);
    requireColumns(Arm.events,requiredEvents);
    if height(Arm.runs) ~= 18 || any(Arm.runs.status ~= "ok") || ...
            any(Arm.runs.finalFE ~= Arm.runs.maxFE)
        error('CBSRegionGAN:BadDedupArmIntegrity', ...
            '%s must contain 18 status=ok exact-FE runs.',name);
    end
end

function Row = summarizePair(Key,A,B)
    Row = emptyComparisonRow();
    Row.problem = string(Key.problem);
    Row.run = double(Key.run);
    Row.seed = double(Key.seed);
    EA = selectEvents(A.events,Key);
    EB = selectEvents(B.events,Key);
    RA = selectRun(A.runs,Key);
    RB = selectRun(B.runs,Key);
    Row.a_event_count = height(EA);
    Row.b_event_count = height(EB);
    Row.a_generated_event_count = sum(EA.raw_generated_count > 0);
    Row.b_generated_event_count = sum(EB.raw_generated_count > 0);
    Row.a_trainable_event_count = sum(EA.raw_generated_count > 0);
    Row.b_trainable_event_count = sum(EB.raw_generated_count > 0);
    Row.a_exact_duplicate_sum = sumFinite(EA.train_exact_duplicate_count);
    Row.b_exact_duplicate_sum = sumFinite(EB.train_exact_duplicate_count);
    Row.a_removed_duplicate_sum = sumFinite(EA.train_removed_duplicate_count);
    Row.b_removed_duplicate_sum = sumFinite(EB.train_removed_duplicate_count);
    Row.a_runtime = double(RA.runtime);
    Row.b_runtime = double(RB.runtime);
    Row.runtime_delta = Row.b_runtime - Row.a_runtime;

    metricNames = comparisonMetrics();
    for i = 1 : numel(metricNames)
        name = metricNames(i);
        a = eventMetricMedian(EA,name);
        b = eventMetricMedian(EB,name);
        Row.("a_" + name) = a;
        Row.("b_" + name) = b;
        Row.("delta_" + name) = b - a;
    end
end

function E = selectEvents(Events,Key)
    E = Events(Events.problem == string(Key.problem) & ...
        Events.run == double(Key.run) & Events.seed == double(Key.seed),:);
end

function R = selectRun(Runs,Key)
    R = Runs(Runs.problem == string(Key.problem) & ...
        Runs.run == double(Key.run) & Runs.seed == double(Key.seed),:);
    if height(R) ~= 1
        error('CBSRegionGAN:BadDedupRunKey','Expected one run-summary row.');
    end
end

function value = eventMetricMedian(E,name)
    generated = E.raw_generated_count > 0;
    if name == "diversity_deviation90"
        source = numericColumn(E,'generated_train_dec_diversity_ratio90');
        source = abs(log(source));
    elseif name == "runtime"
        value = NaN;
        return;
    else
        source = numericColumn(E,char(name));
    end
    source = source(generated & isfinite(source));
    if isempty(source)
        value = NaN;
    else
        value = median(source);
    end
end

function X = numericColumn(T,name)
    if ismember(name,T.Properties.VariableNames)
        X = double(T.(name));
    else
        X = nan(height(T),1);
    end
end

function names = comparisonMetrics()
    names = [ ...
        "bdist50_true","bwidth90_10_true","bcover_eps_true", ...
        "diversity_deviation90", ...
        "generated_queried_anchor_utilization_rate", ...
        "offspringG_survive_union_rate", ...
        "query_populated_bdist50_true", ...
        "query_populated_bwidth90_10_true", ...
        "query_populated_bcover_eps_true", ...
        "query_frontier_bdist50_true", ...
        "query_frontier_bwidth90_10_true", ...
        "query_frontier_bcover_eps_true"];
end

function Stats = pairedStatistics(C)
    names = [comparisonMetrics(),"runtime"];
    lowerBetter = [true,true,false,true,false,false, ...
        true,true,false,true,true,false,true];
    rows = repmat(struct('metric',"",'better_direction',"",'n',0, ...
        'median_delta',NaN,'wins',0,'ties',0,'losses',0, ...
        'p_value',NaN,'p_holm_primary',NaN),numel(names),1);
    for i = 1 : numel(names)
        name = names(i);
        if name == "runtime"
            a = C.a_runtime;
            b = C.b_runtime;
        else
            a = C.("a_" + name);
            b = C.("b_" + name);
        end
        valid = isfinite(a) & isfinite(b);
        delta = b(valid) - a(valid);
        rows(i).metric = name;
        rows(i).better_direction = string(ternary(lowerBetter(i), ...
            'lower','higher'));
        rows(i).n = sum(valid);
        rows(i).median_delta = medianFinite(delta);
        if lowerBetter(i)
            rows(i).wins = sum(delta < 0);
            rows(i).losses = sum(delta > 0);
        else
            rows(i).wins = sum(delta > 0);
            rows(i).losses = sum(delta < 0);
        end
        rows(i).ties = sum(delta == 0);
        rows(i).p_value = signedRankP(a(valid),b(valid));
    end
    adjusted = holmAdjust([rows(1:2).p_value]);
    rows(1).p_holm_primary = adjusted(1);
    rows(2).p_holm_primary = adjusted(2);
    Stats = struct2table(rows);
end

function D = decide(C,Stats,ARuns,BRuns)
    distance = metricRow(Stats,"bdist50_true");
    width = metricRow(Stats,"bwidth90_10_true");
    primaryGate = distance.median_delta < 0 && width.median_delta < 0 && ...
        distance.p_holm_primary < 0.05 && width.p_holm_primary < 0.05;
    positiveSignal = distance.median_delta < 0 && width.median_delta < 0 && ...
        distance.wins > distance.losses && width.wins > width.losses;
    coverageGuard = nonWorse(Stats,"bcover_eps_true",false);
    diversityGuard = nonWorse(Stats,"diversity_deviation90",true);
    anchorGuard = nonWorse(Stats, ...
        "generated_queried_anchor_utilization_rate",false);
    survivalGuard = nonWorse(Stats,"offspringG_survive_union_rate",false);
    groupGeometryGuard = ...
        nonWorse(Stats,"query_populated_bdist50_true",true) && ...
        nonWorse(Stats,"query_populated_bwidth90_10_true",true) && ...
        nonWorse(Stats,"query_frontier_bdist50_true",true) && ...
        nonWorse(Stats,"query_frontier_bwidth90_10_true",true);
    exactFE = all(ARuns.finalFE == ARuns.maxFE) && ...
        all(BRuns.finalFE == BRuns.maxFE);
    mechanismIntegrity = all(C.a_removed_duplicate_sum == 0) && ...
        all(C.b_removed_duplicate_sum > 0) && exactFE;
    allGuards = coverageGuard && diversityGuard && anchorGuard && ...
        survivalGuard && groupGeometryGuard;
    if mechanismIntegrity && primaryGate && allGuards
        status = "promote";
        nextStep = "start_50x2_plus_signature_skip_speed_experiment";
    elseif mechanismIntegrity && positiveSignal && allGuards
        status = "hold";
        nextStep = "add_only_seeds_4_5_for_exact_dedup";
    else
        status = "reject";
        nextStep = "restore_row_weighting_and_stop_dedup_clustering_route";
    end
    D = table(status,nextStep,mechanismIntegrity,primaryGate, ...
        positiveSignal,coverageGuard,diversityGuard,anchorGuard, ...
        survivalGuard,groupGeometryGuard,exactFE, ...
        'VariableNames',{'status','next_step','mechanism_integrity', ...
        'primary_gate','positive_signal','coverage_guard', ...
        'diversity_guard','anchor_guard','survival_guard', ...
        'group_geometry_guard','exact_fe'});
end

function tf = nonWorse(Stats,name,lowerBetter)
    row = metricRow(Stats,name);
    tf = height(row) == 1 && isfinite(row.median_delta) && ...
        ((lowerBetter && row.median_delta <= 0) || ...
        (~lowerBetter && row.median_delta >= 0));
end

function row = metricRow(Stats,name)
    row = Stats(Stats.metric == name,:);
end

function requireColumns(T,names)
    names = string(names(:));
    missing = names(~ismember(names,string(T.Properties.VariableNames)));
    if ~isempty(missing)
        error('CBSRegionGAN:MissingDedupColumns','Missing columns: %s', ...
            strjoin(missing,', '));
    end
end

function p = signedRankP(a,b)
    delta = double(b(:)-a(:));
    delta = delta(isfinite(delta) & delta ~= 0);
    if isempty(delta)
        p = 1;
    else
        p = signrank(a,b);
    end
end

function adjusted = holmAdjust(p)
    p = double(p(:));
    [sorted,order] = sort(p);
    m = numel(p);
    scaled = zeros(m,1);
    running = 0;
    for i = 1 : m
        running = max(running,(m-i+1)*sorted(i));
        scaled(i) = min(1,running);
    end
    adjusted = zeros(m,1);
    adjusted(order) = scaled;
end

function value = medianFinite(X)
    X = double(X(:));
    X = X(isfinite(X));
    if isempty(X); value = NaN; else; value = median(X); end
end

function value = sumFinite(X)
    X = double(X(:));
    value = sum(X(isfinite(X)));
end

function value = ternary(condition,a,b)
    if condition; value = a; else; value = b; end
end

function R = emptyComparisonRow()
    R = struct('problem',"",'run',0,'seed',0, ...
        'a_event_count',0,'b_event_count',0, ...
        'a_generated_event_count',0,'b_generated_event_count',0, ...
        'a_trainable_event_count',0,'b_trainable_event_count',0, ...
        'a_exact_duplicate_sum',0,'b_exact_duplicate_sum',0, ...
        'a_removed_duplicate_sum',0,'b_removed_duplicate_sum',0, ...
        'a_runtime',NaN,'b_runtime',NaN,'runtime_delta',NaN);
    names = comparisonMetrics();
    for i = 1 : numel(names)
        R.("a_" + names(i)) = NaN;
        R.("b_" + names(i)) = NaN;
        R.("delta_" + names(i)) = NaN;
    end
end
