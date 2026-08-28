function [RunMetrics,ProblemComparisons,ContrastSummary] = ...
        analyze_CBS_CGAN_cutoff_diagnostics(rootPath,campaignName,phase)
%ANALYZE_CBS_CGAN_CUTOFF_DIAGNOSTICS Compare run-level cutoff evidence.
% Nonfinite observations are removed before means and independent ranksum
% tests. Candidate rows are never treated as independent replicates.

    if nargin < 1 || isempty(rootPath)
        rootPath = fileparts(which('platemo'));
    end
    if nargin < 2 || isempty(campaignName)
        campaignName = "CBS_CGAN_cutoff_diagnostics_runs1_5";
    end
    if nargin < 3 || isempty(phase)
        phase = "phase1";
    end
    phase = lower(string(phase));
    campaignDir = fullfile(rootPath,'Data',char(campaignName));
    manifest = load(fullfile(campaignDir, ...
        "campaign_manifest_"+phase+".mat"),'Protocol','Tasks');
    Protocol = manifest.Protocol;
    Tasks = manifest.Tasks;
    if phase == "phase2"
        phase1 = load(fullfile(campaignDir, ...
            'campaign_manifest_phase1.mat'),'Tasks');
        Tasks = [phase1.Tasks([phase1.Tasks.arm] == "E0");Tasks];
    end
    Definitions = metricDefinitions();
    RunMetrics = loadRunMetrics(Tasks,Definitions);
    Contrasts = contrastDefinitions(phase);
    ProblemComparisons = compareProblems( ...
        RunMetrics,Contrasts,Definitions,Protocol.problems);
    ContrastSummary = summarizeContrasts( ...
        ProblemComparisons,Contrasts,Definitions);

    analysisDir = fullfile(campaignDir,'analysis');
    mkdir(analysisDir);
    writetable(RunMetrics,fullfile(analysisDir,phase+"_run_metrics.csv"));
    writetable(ProblemComparisons,fullfile(analysisDir, ...
        phase+"_problem_comparisons.csv"));
    writetable(ContrastSummary,fullfile(analysisDir, ...
        phase+"_contrast_summary.csv"));
    save(fullfile(analysisDir,phase+"_analysis.mat"), ...
        'RunMetrics','ProblemComparisons','ContrastSummary', ...
        'Definitions','Contrasts','Protocol','-v7.3');
end

function Definitions = metricDefinitions()
    Definitions = [ ...
        metric("cganEndIGD","lower"), ...
        metric("cganEndHV","higher"), ...
        metric("frontHalfIGDAUC","lower"), ...
        metric("frontHalfHVAUC","higher"), ...
        metric("cganEndFeasibleCount","higher"), ...
        metric("cganEndNondominatedFeasibleCount","higher"), ...
        metric("cganEndRefCoverage","higher"), ...
        metric("cganEndRefEntropy","higher"), ...
        metric("rawBoundaryDistanceMedian","lower"), ...
        metric("rawBoundaryDistanceP90","lower"), ...
        metric("rawBoundaryBandRate","higher"), ...
        metric("rawBoundarySupportRate","diagnostic"), ...
        metric("keptBoundaryDistanceMedian","lower"), ...
        metric("keptBoundaryDistanceP90","lower"), ...
        metric("keptBoundaryBandRate","higher"), ...
        metric("keptBoundarySupportRate","diagnostic"), ...
        metric("rejectedBoundaryDistanceMedian","lower"), ...
        metric("selectedBoundaryDistanceMedian","lower"), ...
        metric("selectedBoundaryDistanceP90","lower"), ...
        metric("selectedBoundaryBandRate","higher"), ...
        metric("centerBoundaryDistanceMedian","lower"), ...
        metric("childBoundaryDistanceMedian","lower"), ...
        metric("criticBoundarySpearman","higher"), ...
        metric("rawDirectionCoverage","higher"), ...
        metric("rawDirectionEntropy","higher"), ...
        metric("rawNearDuplicateRate","lower"), ...
        metric("keptDirectionCoverage","higher"), ...
        metric("keptDirectionEntropy","higher"), ...
        metric("keptNearDuplicateRate","lower"), ...
        metric("selectedDirectionCoverage","higher"), ...
        metric("selectedDirectionEntropy","higher"), ...
        metric("selectedNearDuplicateRate","lower"), ...
        metric("selectedToCenterBoundaryChangeMean","lower"), ...
        metric("centerToChildBoundaryChangeMean","lower"), ...
        metric("frontHalfIGDAUCCoverage","diagnostic"), ...
        metric("frontHalfHVAUCCoverage","diagnostic")];
end

function Definition = metric(name,direction)
    Definition = struct('name',string(name),'direction',string(direction));
end

function Contrasts = contrastDefinitions(phase)
    if phase == "phase1"
        Contrasts = [ ...
            contrast("A1_vs_A0","A0","A1"), ...
            contrast("Current_vs_A1","A1","Current"), ...
            contrast("Current_vs_A0","A0","Current"), ...
            contrast("E0_vs_A0","A0","E0"), ...
            contrast("Current_vs_E0","E0","Current"), ...
            contrast("Current_vs_Random20","Random20","Current"), ...
            contrast("Current_vs_DE20","DE20","Current"), ...
            contrast("Current_vs_GA20","GA20","Current")];
    else
        Contrasts = [ ...
            contrast("E1_vs_E0","E0","E1"), ...
            contrast("K10_vs_E0","E0","K10"), ...
            contrast("KAll_vs_E0","E0","KAll"), ...
            contrast("Cap10_vs_E0","E0","Cap10")];
    end
end

function Contrast = contrast(name,control,candidate)
    Contrast = struct('name',string(name),'control',string(control), ...
        'candidate',string(candidate));
end

function T = loadRunMetrics(Tasks,Definitions)
    template = struct('phase',"",'arm',"",'className',"", ...
        'problem',"",'seed',NaN,'status',"missing", ...
        'cutoffFE',NaN,'metric',"",'direction',"",'value',NaN);
    rows = repmat(template,numel(Tasks)*numel(Definitions),1);
    next = 0;
    for t = 1 : numel(Tasks)
        status = "missing";
        cutoffFE = NaN;
        Audit = struct();
        try
            Data = load(char(Tasks(t).outputFile),'Record','Audit');
            if isfield(Data,'Record') && isfield(Data,'Audit') && ...
                    string(Data.Record.status) == "ok"
                status = "ok";
                cutoffFE = double(Data.Audit.cganEndFE);
                Audit = Data.Audit;
            else
                status = "invalid";
            end
        catch
            if exist(char(Tasks(t).outputFile),'file') == 2
                status = "invalid";
            end
        end
        for d = 1 : numel(Definitions)
            next = next+1;
            rows(next).phase = string(Tasks(t).phase);
            rows(next).arm = string(Tasks(t).arm);
            rows(next).className = string(Tasks(t).className);
            rows(next).problem = string(Tasks(t).problem);
            rows(next).seed = double(Tasks(t).seed);
            rows(next).status = status;
            rows(next).cutoffFE = cutoffFE;
            rows(next).metric = Definitions(d).name;
            rows(next).direction = Definitions(d).direction;
            if status == "ok" && isfield(Audit,Definitions(d).name)
                value = double(Audit.(Definitions(d).name));
                if isscalar(value)
                    rows(next).value = value;
                end
            end
        end
    end
    T = struct2table(rows);
end

function T = compareProblems(Runs,Contrasts,Definitions,problems)
    template = struct('contrast',"",'control',"",'candidate',"", ...
        'problem',"",'metric',"",'direction',"", ...
        'controlN',0,'candidateN',0,'controlMean',NaN, ...
        'candidateMean',NaN,'candidateToControlRatio',NaN, ...
        'candidateMinusControl',NaN,'orientedImprovement',NaN, ...
        'rankSumP',NaN,'practicalEffect',"=", ...
        'significantBetter',false,'significantWorse',false);
    rows = repmat(template,numel(Contrasts)*numel(problems)* ...
        numel(Definitions),1);
    next = 0;
    for c = 1 : numel(Contrasts)
        for problem = problems
            for d = 1 : numel(Definitions)
                next = next+1;
                definition = Definitions(d);
                base = finiteValues(Runs.value( ...
                    Runs.arm == Contrasts(c).control & ...
                    Runs.problem == problem & Runs.metric == definition.name & ...
                    Runs.status == "ok"));
                candidate = finiteValues(Runs.value( ...
                    Runs.arm == Contrasts(c).candidate & ...
                    Runs.problem == problem & Runs.metric == definition.name & ...
                    Runs.status == "ok"));
                row = template;
                row.contrast = Contrasts(c).name;
                row.control = Contrasts(c).control;
                row.candidate = Contrasts(c).candidate;
                row.problem = problem;
                row.metric = definition.name;
                row.direction = definition.direction;
                row.controlN = numel(base);
                row.candidateN = numel(candidate);
                row.controlMean = finiteMean(base);
                row.candidateMean = finiteMean(candidate);
                row.candidateToControlRatio = safeRatio( ...
                    row.candidateMean,row.controlMean);
                row.candidateMinusControl = ...
                    row.candidateMean-row.controlMean;
                if definition.direction == "lower"
                    row.orientedImprovement = ...
                        row.controlMean-row.candidateMean;
                elseif definition.direction == "higher"
                    row.orientedImprovement = ...
                        row.candidateMean-row.controlMean;
                end
                row.rankSumP = independentRankSum(base,candidate);
                row.practicalEffect = practicalEffect(row,definition.direction);
                row.significantBetter = row.rankSumP < 0.05 && ...
                    row.orientedImprovement > 0;
                row.significantWorse = row.rankSumP < 0.05 && ...
                    row.orientedImprovement < 0;
                rows(next) = row;
            end
        end
    end
    T = struct2table(rows);
end

function T = summarizeContrasts(Problems,Contrasts,Definitions)
    template = struct('contrast',"",'control',"",'candidate',"", ...
        'metric',"",'direction',"",'problemsWithFiniteMeans',0, ...
        'meanCandidateToControlRatio',NaN, ...
        'meanOrientedImprovement',NaN,'wins',0,'ties',0,'losses',0, ...
        'significantWins',0,'significantLosses',0);
    rows = repmat(template,numel(Contrasts)*numel(Definitions),1);
    next = 0;
    for c = 1 : numel(Contrasts)
        for d = 1 : numel(Definitions)
            next = next+1;
            mask = Problems.contrast == Contrasts(c).name & ...
                Problems.metric == Definitions(d).name;
            mine = Problems(mask,:);
            row = template;
            row.contrast = Contrasts(c).name;
            row.control = Contrasts(c).control;
            row.candidate = Contrasts(c).candidate;
            row.metric = Definitions(d).name;
            row.direction = Definitions(d).direction;
            valid = isfinite(mine.controlMean) & isfinite(mine.candidateMean);
            row.problemsWithFiniteMeans = sum(valid);
            row.meanCandidateToControlRatio = finiteMean( ...
                mine.candidateToControlRatio);
            row.meanOrientedImprovement = finiteMean(mine.orientedImprovement);
            row.wins = sum(mine.practicalEffect == "+");
            row.ties = sum(mine.practicalEffect == "=");
            row.losses = sum(mine.practicalEffect == "-");
            row.significantWins = sum(mine.significantBetter);
            row.significantLosses = sum(mine.significantWorse);
            rows(next) = row;
        end
    end
    T = struct2table(rows);
end

function values = finiteValues(values)
    values = double(values(:));
    values = values(isfinite(values));
end

function value = finiteMean(values)
    values = finiteValues(values);
    if isempty(values)
        value = NaN;
    else
        value = mean(values);
    end
end

function value = safeRatio(numerator,denominator)
    if isfinite(numerator) && isfinite(denominator) && denominator ~= 0
        value = numerator/denominator;
    else
        value = NaN;
    end
end

function p = independentRankSum(a,b)
    if isempty(a) || isempty(b) || exist('ranksum','file') ~= 2
        p = NaN;
        return;
    end
    try
        p = ranksum(a,b);
    catch
        p = NaN;
    end
end

function effect = practicalEffect(Row,direction)
    if direction == "diagnostic" || ...
            ~isfinite(Row.controlMean) || ~isfinite(Row.candidateMean)
        effect = "=";
        return;
    end
    ratio = Row.candidateToControlRatio;
    if isfinite(ratio) && Row.controlMean > 0 && Row.candidateMean >= 0
        if direction == "lower"
            improvement = 1-ratio;
        else
            improvement = ratio-1;
        end
        if improvement >= 0.02
            effect = "+";
        elseif improvement <= -0.02
            effect = "-";
        else
            effect = "=";
        end
    elseif Row.orientedImprovement > 0
        effect = "+";
    elseif Row.orientedImprovement < 0
        effect = "-";
    else
        effect = "=";
    end
end
