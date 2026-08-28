function [CandidateSummary,ProblemSummary,RunSummary] = ...
        analyze_CBS_CGAN_screening_stage( ...
        rootPath,controlAlgorithm,candidateAlgorithms,problems,runs)
%ANALYZE_CBS_CGAN_SCREENING_STAGE Compare independent repeated runs.
%   NaN IGD values are removed before means and rank-sum tests, matching
%   the project convention. Run identifiers are not treated as paired seeds.

    algorithms = [string(controlAlgorithm),string(candidateAlgorithms(:)')];
    problems = string(problems(:)');
    runs = double(runs(:)');
    rows = repmat(emptyRunRow(),numel(algorithms)*numel(problems)* ...
        numel(runs),1);
    next = 0;
    for algorithm = algorithms
        for problem = problems
            for run = runs
                next = next+1;
                rows(next) = loadRun(rootPath,algorithm,problem,run);
            end
        end
    end
    RunSummary = struct2table(rows);

    comparisonRows = repmat(emptyProblemRow(), ...
        numel(candidateAlgorithms)*numel(problems),1);
    candidateRows = repmat(emptyCandidateRow(), ...
        numel(candidateAlgorithms),1);
    next = 0;
    for c = 1 : numel(candidateAlgorithms)
        candidate = string(candidateAlgorithms(c));
        ratios100 = nan(numel(problems),1);
        ratios200 = nan(numel(problems),1);
        for p = 1 : numel(problems)
            problem = problems(p);
            baseRows = RunSummary.algorithm == string(controlAlgorithm) & ...
                RunSummary.problem == problem & RunSummary.status == "ok";
            candidateRowsMask = RunSummary.algorithm == candidate & ...
                RunSummary.problem == problem & RunSummary.status == "ok";
            base100 = finiteValues(RunSummary.IGD100K(baseRows));
            base200 = finiteValues(RunSummary.IGD200K(baseRows));
            test100 = finiteValues(RunSummary.IGD100K(candidateRowsMask));
            test200 = finiteValues(RunSummary.IGD200K(candidateRowsMask));
            next = next+1;
            row = emptyProblemRow();
            row.control = string(controlAlgorithm);
            row.candidate = candidate;
            row.problem = problem;
            row.controlN = numel(base200);
            row.candidateN = numel(test200);
            row.controlNaN = sum(baseRows)-numel(base200);
            row.candidateNaN = sum(candidateRowsMask)-numel(test200);
            row.controlMean100K = finiteMean(base100);
            row.candidateMean100K = finiteMean(test100);
            row.controlMean200K = finiteMean(base200);
            row.candidateMean200K = finiteMean(test200);
            row.ratio100K = safeRatio( ...
                row.candidateMean100K,row.controlMean100K);
            row.ratio200K = safeRatio( ...
                row.candidateMean200K,row.controlMean200K);
            row.rankSumP100K = independentRankSum(base100,test100);
            row.rankSumP200K = independentRankSum(base200,test200);
            row.effect = practicalEffect(row.ratio200K);
            comparisonRows(next) = row;
            ratios100(p) = row.ratio100K;
            ratios200(p) = row.ratio200K;
        end
        mine = comparisonRows([comparisonRows.candidate] == candidate);
        effects = string({mine.effect});
        problemNames = string({mine.problem});
        row = emptyCandidateRow();
        row.control = string(controlAlgorithm);
        row.candidate = candidate;
        row.gmeanRatio100K = geometricMean(ratios100);
        row.gmeanRatio200K = geometricMean(ratios200);
        row.wins = sum(effects == "+");
        row.ties = sum(effects == "=");
        row.losses = sum(effects == "-");
        row.dasRatio200K = geometricMean( ...
            ratios200(startsWith(problemNames,"DASCMOP")));
        row.lirRatio200K = geometricMean( ...
            ratios200(startsWith(problemNames,"LIRCMOP")));
        row.allTasksComplete = all(RunSummary.status( ...
            RunSummary.algorithm == candidate) == "ok") && ...
            sum(RunSummary.algorithm == candidate) == ...
            numel(problems)*numel(runs);
        row.pass = row.allTasksComplete && ...
            row.gmeanRatio200K <= 0.98 && row.wins > row.losses && ...
            row.dasRatio200K <= 1.02 && row.lirRatio200K <= 1.02;
        candidateRows(c) = row;
    end
    CandidateSummary = struct2table(candidateRows);
    ProblemSummary = struct2table(comparisonRows);
end

function Row = loadRun(rootPath,algorithm,problem,run)
    Row = emptyRunRow();
    Row.algorithm = algorithm;
    Row.problem = problem;
    Row.run = run;
    pattern = sprintf('%s_%s_M*_D*_%d.mat',algorithm,problem,run);
    files = dir(fullfile(rootPath,'Data',algorithm,pattern));
    if numel(files) ~= 1
        Row.status = "missing";
        return;
    end
    try
        Data = load(fullfile(files(1).folder,files(1).name), ...
            'result','metric');
        if ~isfield(Data,'result') || isempty(Data.result) || ...
                ~isfield(Data,'metric') || ~isfield(Data.metric,'IGD')
            Row.status = "invalid";
            return;
        end
        FE = cell2mat(Data.result(:,1));
        IGD = reshape(double(Data.metric.IGD),[],1);
        count = min(numel(FE),numel(IGD));
        FE = FE(1:count);
        IGD = IGD(1:count);
        first100 = find(FE >= 1e5,1,'first');
        if isempty(first100)
            [~,first100] = min(abs(FE-1e5));
        end
        Row.FE100K = FE(first100);
        Row.IGD100K = IGD(first100);
        Row.FE200K = FE(end);
        Row.IGD200K = IGD(end);
        if isfield(Data.metric,'CBSAudit') && ...
                isstruct(Data.metric.CBSAudit)
            Row.audit = {Data.metric.CBSAudit};
        end
        if Row.FE200K >= 2e5
            Row.status = "ok";
        else
            Row.status = "incomplete";
        end
    catch
        Row.status = "invalid";
    end
end

function values = finiteValues(values)
    values = double(values(:));
    values = values(isfinite(values));
end

function value = finiteMean(values)
    if isempty(values)
        value = NaN;
    else
        value = mean(values);
    end
end

function value = safeRatio(numerator,denominator)
    if isfinite(numerator) && isfinite(denominator) && denominator > 0
        value = numerator/denominator;
    elseif numerator == 0 && denominator == 0
        value = 1;
    else
        value = Inf;
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

function effect = practicalEffect(ratio)
    if ratio <= 0.98
        effect = "+";
    elseif ratio > 1.02
        effect = "-";
    else
        effect = "=";
    end
end

function value = geometricMean(values)
    values = double(values(:));
    if isempty(values) || any(~isfinite(values) | values <= 0)
        value = Inf;
    else
        value = exp(mean(log(values)));
    end
end

function Row = emptyRunRow()
    Row = struct('algorithm',"",'problem',"",'run',NaN, ...
        'status',"missing",'FE100K',NaN,'IGD100K',NaN, ...
        'FE200K',NaN,'IGD200K',NaN,'audit',{{struct()}});
end

function Row = emptyProblemRow()
    Row = struct('control',"",'candidate',"",'problem',"", ...
        'controlN',0,'candidateN',0,'controlNaN',0,'candidateNaN',0, ...
        'controlMean100K',NaN,'candidateMean100K',NaN, ...
        'controlMean200K',NaN,'candidateMean200K',NaN, ...
        'ratio100K',NaN,'ratio200K',NaN,'rankSumP100K',NaN, ...
        'rankSumP200K',NaN,'effect',"=");
end

function Row = emptyCandidateRow()
    Row = struct('control',"",'candidate',"", ...
        'gmeanRatio100K',NaN,'gmeanRatio200K',NaN, ...
        'wins',0,'ties',0,'losses',0,'dasRatio200K',NaN, ...
        'lirRatio200K',NaN,'allTasksComplete',false,'pass',false);
end
