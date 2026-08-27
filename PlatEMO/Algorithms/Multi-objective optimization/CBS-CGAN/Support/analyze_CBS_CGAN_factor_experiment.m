function [Effects,Pairwise] = analyze_CBS_CGAN_factor_experiment(RunSummary)
%ANALYZE_CBS_CGAN_FACTOR_EXPERIMENT Quantify sequential package effects.
%   A1/A0 estimates the generation-and-screening package; A2/A1 estimates
%   the utilization package; A2/A0 is the end-to-end effect. Lower IGD is
%   better, so ratios below one favor the numerator arm.

    required = ["arm","problem","run","IGD100K","IGD200K","status"];
    if ~istable(RunSummary) || ...
            ~all(ismember(required,string(RunSummary.Properties.VariableNames)))
        error('CBSRegionGAN:BadFactorSummary', ...
            'RunSummary is missing required experiment columns.');
    end
    RunSummary = RunSummary(RunSummary.status == "ok",:);
    comparisons = [0 1;1 2;0 2];
    labels = ["generation_screening","utilization","end_to_end"];
    effectRows = repmat(emptyEffectRow(),3,1);
    pairTables = cell(3,1);
    for c = 1 : 3
        baseArm = comparisons(c,1);
        candidateArm = comparisons(c,2);
        Base = armTable(RunSummary,baseArm,"base");
        Candidate = armTable(RunSummary,candidateArm,"candidate");
        P = innerjoin(Base,Candidate,'Keys',{'problem','run'});
        P.effect = repmat(labels(c),height(P),1);
        P.baseArm = repmat(baseArm,height(P),1);
        P.candidateArm = repmat(candidateArm,height(P),1);
        P.ratio100K = igdRatio(P.IGD100K_candidate,P.IGD100K_base);
        P.ratio200K = igdRatio(P.IGD200K_candidate,P.IGD200K_base);
        P = movevars(P,{'effect','baseArm','candidateArm'},'Before',1);
        pairTables{c} = P;

        problems = unique(P.problem,'stable');
        problemRatio100K = nan(numel(problems),1);
        problemRatio200K = nan(numel(problems),1);
        for p = 1 : numel(problems)
            rows = P.problem == problems(p);
            problemRatio100K(p) = median(P.ratio100K(rows));
            problemRatio200K(p) = median(P.ratio200K(rows));
        end
        finitePrimary = all(isfinite(P.ratio200K) & P.ratio200K > 0) && ...
            height(P) == height(Base) && height(P) == height(Candidate);
        gmean100K = geometricMean(problemRatio100K);
        gmean200K = geometricMean(problemRatio200K);
        wins = sum(problemRatio200K <= 0.98);
        losses = sum(problemRatio200K > 1.02);
        familyNoHarm = familyGate(problems,problemRatio200K);
        effectRows(c) = struct('effect',labels(c),'baseArm',baseArm, ...
            'candidateArm',candidateArm,'pairedRuns',height(P), ...
            'problemCount',numel(problems),'gmeanRatio100K',gmean100K, ...
            'gmeanRatio200K',gmean200K,'wins',wins,'losses',losses, ...
            'allFinite',finitePrimary,'familyNoHarm',familyNoHarm, ...
            'pass',finitePrimary && gmean200K <= 0.98 && ...
            wins > losses && familyNoHarm);
    end
    Effects = struct2table(effectRows);
    Pairwise = vertcat(pairTables{:});
end

function T = armTable(Summary,arm,suffix)
    rows = Summary.arm == arm;
    names = ["problem","run","IGD100K_"+suffix,"IGD200K_"+suffix];
    T = table(Summary.problem(rows),Summary.run(rows), ...
        Summary.IGD100K(rows),Summary.IGD200K(rows), ...
        'VariableNames',names);
end

function ratio = igdRatio(candidate,base)
    ratio = candidate./base;
    bothZero = candidate == 0 & base == 0;
    ratio(bothZero) = 1;
    ratio(candidate > 0 & base == 0) = inf;
    invalid = candidate < 0 | base < 0 | ~isfinite(candidate) | ...
        ~isfinite(base);
    ratio(invalid) = NaN;
end

function value = geometricMean(values)
    if isempty(values) || any(~isfinite(values) | values <= 0)
        value = inf;
    else
        value = exp(mean(log(values)));
    end
end

function passed = familyGate(problems,ratios)
    families = strings(numel(problems),1);
    families(startsWith(problems,"DASCMOP")) = "DASCMOP";
    families(startsWith(problems,"LIRCMOP")) = "LIRCMOP";
    uniqueFamilies = unique(families(families ~= ""));
    passed = ~isempty(uniqueFamilies);
    for family = reshape(uniqueFamilies,1,[])
        passed = passed && geometricMean(ratios(families == family)) <= 1;
    end
end

function Row = emptyEffectRow()
    Row = struct('effect',"",'baseArm',NaN,'candidateArm',NaN, ...
        'pairedRuns',0,'problemCount',0,'gmeanRatio100K',NaN, ...
        'gmeanRatio200K',NaN,'wins',0,'losses',0,'allFinite',false, ...
        'familyNoHarm',false,'pass',false);
end
