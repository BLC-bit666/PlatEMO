function [AllRuns,ScopeSummary,ProblemSummary] = analyze_PRBCCMO_t_innovation_suite(suiteDir)
% Analyze PRBCCMO_t innovation evidence from benchmark shards and trace CSVs.

    if nargin < 1 || isempty(suiteDir)
        rootDir = fileparts(which('platemo'));
        suiteDir = fullfile(rootDir,'Data','PRBCCMO_t','innovation_runs_20260428_5runs');
    end
    suiteDir = char(string(suiteDir));
    assert(isfolder(suiteDir), ...
        'analyze_PRBCCMO_t_innovation_suite:MissingSuiteDir', ...
        'Suite directory not found: %s', suiteDir);

    Files = dir(fullfile(suiteDir,'run_*.csv'));
    assert(~isempty(Files), ...
        'analyze_PRBCCMO_t_innovation_suite:MissingCsv', ...
        'No run_*.csv benchmark shards found in %s.', suiteDir);

    AllRuns = table();
    for i = 1 : numel(Files)
        T = readtable(fullfile(Files(i).folder,Files(i).name),'TextType','string');
        AllRuns = [AllRuns;T]; %#ok<AGROW>
    end
    AllRuns = sortrows(AllRuns,{'family','problem','run'});
    AllRuns = appendTraceEvidence(AllRuns);
    writetable(AllRuns,fullfile(suiteDir,'benchmark_all.csv'));
    writetable(AllRuns,fullfile(suiteDir,'innovation_trace_runs.csv'));

    ScopeSummary = buildScopeSummary(AllRuns);
    writetable(ScopeSummary,fullfile(suiteDir,'innovation_summary.csv'));

    ProblemSummary = buildProblemSummary(AllRuns);
    writetable(ProblemSummary,fullfile(suiteDir,'innovation_problem_summary.csv'));
end

function Runs = appendTraceEvidence(Runs)
    N = height(Runs);
    Runs.trace_generation_count = zeros(N,1);
    Runs.trace_boundary_event_count = zeros(N,1);
    Runs.trace_boundary_evidence_count_mean = NaN(N,1);
    Runs.trace_boundary_selection_yield_mean = NaN(N,1);
    Runs.trace_boundary_corr_mean = NaN(N,1);
    Runs.trace_boundary_corr_pos_rate = NaN(N,1);
    Runs.trace_boundary_lowmargin_oppdist_gain_mean = NaN(N,1);
    Runs.trace_boundary_lowmargin_oppdist_gain_pos_rate = NaN(N,1);
    Runs.trace_b_corr_mean = NaN(N,1);
    Runs.trace_b_corr_pos_rate = NaN(N,1);
    Runs.trace_archive_hit_rate_eps_mean = NaN(N,1);
    Runs.trace_b_lowmargin_oppdist_gain_mean = NaN(N,1);
    Runs.trace_b_lowmargin_oppdist_gain_pos_rate = NaN(N,1);
    Runs.trace_inf_lowmargin_ratio_gain_mean = NaN(N,1);
    Runs.trace_inf_lowmargin_ratio_gain_pos_rate = NaN(N,1);
    Runs.trace_inf_margin_gain_mean = NaN(N,1);
    Runs.trace_inf_margin_gain_pos_rate = NaN(N,1);
    Runs.trace_inf_obj_score_gain_mean = NaN(N,1);
    Runs.trace_inf_obj_score_gain_pos_rate = NaN(N,1);
    Runs.event_margin_oppdist_corr = NaN(N,1);
    Runs.event_lowmargin_oppdist_gain = NaN(N,1);
    Runs.event_lowmargin_mean_opp_dist = NaN(N,1);
    Runs.event_highmargin_mean_opp_dist = NaN(N,1);
    Runs.event_lowmargin_survive_c_rate = NaN(N,1);
    Runs.event_lowmargin_survive_u_rate = NaN(N,1);

    for i = 1 : N
        folder = char(string(Runs.analysis_folder(i)));
        Gen = readtable(fullfile(folder,'generation_summary.csv'));
        Event = readtable(fullfile(folder,'boundary_event.csv'));

        Runs.trace_generation_count(i) = height(Gen);
        Runs.trace_boundary_event_count(i) = height(Event);
        Runs.trace_boundary_evidence_count_mean(i) = meanFinite(Gen.boundary_evidence_count);
        Runs.trace_boundary_selection_yield_mean(i) = meanFinite( ...
            double(Gen.boundary_off_count)./max(double(Gen.boundary_evidence_count),1));
        Runs.trace_boundary_corr_mean(i) = meanFinite(Gen.boundary_margin_oppdist_corr);
        Runs.trace_boundary_corr_pos_rate(i) = positiveRate(Gen.boundary_margin_oppdist_corr);
        Runs.trace_boundary_lowmargin_oppdist_gain_mean(i) = meanFinite(Gen.boundary_lowmargin_oppdist_gain);
        Runs.trace_boundary_lowmargin_oppdist_gain_pos_rate(i) = positiveRate(Gen.boundary_lowmargin_oppdist_gain);
        Runs.trace_b_corr_mean(i) = meanFinite(Gen.b_margin_true_boundary_corr);
        Runs.trace_b_corr_pos_rate(i) = positiveRate(Gen.b_margin_true_boundary_corr);
        Runs.trace_archive_hit_rate_eps_mean(i) = meanFinite(Gen.archive_hit_rate_eps);
        Runs.trace_b_lowmargin_oppdist_gain_mean(i) = meanFinite(Gen.b_lowmargin_oppdist_gain);
        Runs.trace_b_lowmargin_oppdist_gain_pos_rate(i) = positiveRate(Gen.b_lowmargin_oppdist_gain);
        Runs.trace_inf_lowmargin_ratio_gain_mean(i) = meanFinite(Gen.inf_lowmargin_ratio_gain);
        Runs.trace_inf_lowmargin_ratio_gain_pos_rate(i) = positiveRate(Gen.inf_lowmargin_ratio_gain);
        Runs.trace_inf_margin_gain_mean(i) = meanFinite(Gen.inf_margin_gain);
        Runs.trace_inf_margin_gain_pos_rate(i) = positiveRate(Gen.inf_margin_gain);
        Runs.trace_inf_obj_score_gain_mean(i) = meanFinite(Gen.inf_obj_score_gain);
        Runs.trace_inf_obj_score_gain_pos_rate(i) = positiveRate(Gen.inf_obj_score_gain);

        if ~isempty(Event)
            Margin = double(Event.margin);
            OppDist = double(Event.opp_dist);
            Low = Margin <= 0.10;
            High = Margin > 0.10;
            Runs.event_margin_oppdist_corr(i) = safeCorr(Margin,OppDist);
            Runs.event_lowmargin_mean_opp_dist(i) = meanFinite(OppDist(Low));
            Runs.event_highmargin_mean_opp_dist(i) = meanFinite(OppDist(High));
            Runs.event_lowmargin_oppdist_gain(i) = ...
                Runs.event_highmargin_mean_opp_dist(i) - Runs.event_lowmargin_mean_opp_dist(i);
            Runs.event_lowmargin_survive_c_rate(i) = meanFinite(double(Event.survive_c(Low)));
            Runs.event_lowmargin_survive_u_rate(i) = meanFinite(double(Event.survive_u(Low)));
        end
    end
end

function Summary = buildScopeSummary(AllRuns)
    Rows = summarizeGroup(AllRuns,"overall","all");
    Families = unique(string(AllRuns.family),'stable')';
    for family = Families
        Mask = string(AllRuns.family) == family;
        Rows = [Rows;summarizeGroup(AllRuns(Mask,:), "family", family)]; %#ok<AGROW>
    end
    Summary = rowsToSummaryTable(Rows);
end

function Summary = buildProblemSummary(AllRuns)
    Problems = unique(string(AllRuns.problem),'stable')';
    Rows = cell(numel(Problems),1);
    for i = 1 : numel(Problems)
        Mask = string(AllRuns.problem) == Problems(i);
        Rows{i} = summarizeGroup(AllRuns(Mask,:), "problem", Problems(i));
        Rows{i}{3} = string(AllRuns.family(find(Mask,1,'first')));
    end
    Summary = rowsToSummaryTable(vertcat(Rows{:}));
end

function Row = summarizeGroup(T,Scope,Name)
    Row = { ...
        string(Scope),string(Name),string(""),height(T),sum(double(T.trace_boundary_event_count)), ...
        meanFinite(T.trace_boundary_evidence_count_mean),meanFinite(T.trace_boundary_selection_yield_mean), ...
        meanFinite(T.trace_boundary_corr_mean),positiveRate(T.trace_boundary_corr_mean), ...
        meanFinite(T.trace_boundary_lowmargin_oppdist_gain_mean),positiveRate(T.trace_boundary_lowmargin_oppdist_gain_mean), ...
        meanFinite(T.trace_b_corr_mean),positiveRate(T.trace_b_corr_mean), ...
        meanFinite(T.trace_archive_hit_rate_eps_mean), ...
        meanFinite(T.trace_b_lowmargin_oppdist_gain_mean),positiveRate(T.trace_b_lowmargin_oppdist_gain_mean), ...
        meanFinite(T.trace_inf_lowmargin_ratio_gain_mean),positiveRate(T.trace_inf_lowmargin_ratio_gain_mean), ...
        meanFinite(T.trace_inf_margin_gain_mean),positiveRate(T.trace_inf_margin_gain_mean), ...
        meanFinite(T.trace_inf_obj_score_gain_mean),positiveRate(T.trace_inf_obj_score_gain_mean), ...
        meanFinite(T.event_margin_oppdist_corr),positiveRate(T.event_margin_oppdist_corr), ...
        meanFinite(T.event_lowmargin_oppdist_gain),positiveRate(T.event_lowmargin_oppdist_gain), ...
        meanFinite(T.event_lowmargin_survive_c_rate),meanFinite(T.event_lowmargin_survive_u_rate), ...
        meanFinite(T.final_boundary_survive_c_rate),meanFinite(T.final_boundary_survive_u_rate), ...
        meanFinite(T.igd),meanFinite(T.hv),meanFinite(T.feasible_rate), ...
        mean(~isnan(double(T.stable3_generation)))};
end

function Summary = rowsToSummaryTable(Rows)
    Names = { ...
        'scope','name','family','n_runs','total_boundary_events', ...
        'mean_trace_boundary_evidence_count','mean_trace_boundary_selection_yield', ...
        'mean_trace_boundary_corr','pos_rate_trace_boundary_corr', ...
        'mean_trace_boundary_lowmargin_oppdist_gain','pos_rate_trace_boundary_lowmargin_oppdist_gain', ...
        'mean_trace_b_corr','pos_rate_trace_b_corr', ...
        'mean_trace_archive_hit_rate_eps', ...
        'mean_trace_b_lowmargin_oppdist_gain','pos_rate_trace_b_lowmargin_oppdist_gain', ...
        'mean_trace_inf_lowmargin_ratio_gain','pos_rate_trace_inf_lowmargin_ratio_gain', ...
        'mean_trace_inf_margin_gain','pos_rate_trace_inf_margin_gain', ...
        'mean_trace_inf_obj_score_gain','pos_rate_trace_inf_obj_score_gain', ...
        'mean_event_margin_oppdist_corr','pos_rate_event_margin_oppdist_corr', ...
        'mean_event_lowmargin_oppdist_gain','pos_rate_event_lowmargin_oppdist_gain', ...
        'mean_event_lowmargin_survive_c_rate','mean_event_lowmargin_survive_u_rate', ...
        'mean_final_boundary_survive_c_rate','mean_final_boundary_survive_u_rate', ...
        'mean_igd','mean_hv','mean_feasible_rate','stable3_hit_rate'};
    Summary = cell2table(Rows,'VariableNames',Names);
end

function Value = meanFinite(Value)
    Value = double(Value);
    Value = Value(isfinite(Value));
    if isempty(Value)
        Value = NaN;
    else
        Value = mean(Value);
    end
end

function Value = positiveRate(Value)
    Value = double(Value);
    Value = Value(isfinite(Value));
    if isempty(Value)
        Value = NaN;
    else
        Value = mean(Value > 0);
    end
end

function Value = safeCorr(X,Y)
    X = double(X(:));
    Y = double(Y(:));
    Mask = isfinite(X) & isfinite(Y);
    X = X(Mask);
    Y = Y(Mask);
    if numel(X) < 3 || std(X) < 1e-12 || std(Y) < 1e-12
        Value = NaN;
        return;
    end
    C = corrcoef(X,Y);
    Value = C(1,2);
end
