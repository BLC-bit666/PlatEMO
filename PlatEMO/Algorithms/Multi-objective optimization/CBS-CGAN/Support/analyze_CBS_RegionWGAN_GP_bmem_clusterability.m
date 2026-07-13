function [GroupAudit,SnapshotAudit,RunAudit,Stats,Decision] = ...
        analyze_CBS_RegionWGAN_GP_bmem_clusterability(InputRoot,OutputDir)
%ANALYZE_CBS_REGIONWGAN_GP_BMEM_CLUSTERABILITY Offline legacy-BMem audit.
%   The audit consumes only previously exported bmem_history.csv files. It
%   performs no problem Evaluation, no GAN training, and no evolutionary
%   selection. Exact duplicate weighting is measured before any candidate
%   mode analysis so that duplication cannot be mistaken for multimodality.

    repoRoot = fileparts(which('platemo'));
    if nargin < 1 || strlength(string(InputRoot)) == 0
        InputRoot = fullfile(repoRoot,'Data','CBS_RegionGAN_compare', ...
            'structured_z_mi_20260712_211947','A_control');
    end
    if nargin < 2 || strlength(string(OutputDir)) == 0
        stamp = char(datetime('now','Format','yyyyMMdd_HHmmss'));
        OutputDir = fullfile(repoRoot,'Data','CBS_RegionGAN_compare', ...
            ['bmem_clusterability_audit_' stamp]);
    end
    InputRoot = char(string(InputRoot));
    OutputDir = char(string(OutputDir));
    if ~isfolder(InputRoot)
        error('CBSBMemAudit:MissingInput','Input root does not exist: %s',InputRoot);
    end
    summaryFile = fullfile(InputRoot,'run_summary.csv');
    if ~isfile(summaryFile)
        error('CBSBMemAudit:MissingRunSummary', ...
            'Missing run_summary.csv under %s.',InputRoot);
    end
    if ~isfolder(OutputDir)
        mkdir(OutputDir);
    end

    Design = auditDesign();
    writetable(Design,fullfile(OutputDir,'audit_design.csv'));
    Runs = readtable(summaryFile,'Delimiter',',','TextType','string', ...
        'VariableNamingRule','preserve');
    requireColumns(Runs,["problem","run","seed","N","D","M", ...
        "maxFE","status"]);
    Runs = Runs(lower(strtrim(Runs.status)) == "ok",:);
    Runs = sortrows(Runs,{'problem','run'});
    if isempty(Runs)
        error('CBSBMemAudit:NoCompleteRuns','No status=ok runs were found.');
    end

    GroupParts = cell(height(Runs),1);
    SnapshotParts = cell(height(Runs),1);
    RunRows = repmat(emptyRunRow(),height(Runs),1);
    for i = 1 : height(Runs)
        problem = string(Runs.problem(i));
        run = double(Runs.run(i));
        runFolder = fullfile(InputRoot,sprintf('%s_run%d',problem,run));
        historyFile = fullfile(runFolder,'bmem_history.csv');
        if ~isfile(historyFile) && ismember('bmem_history_file', ...
                Runs.Properties.VariableNames)
            historyFile = char(Runs.bmem_history_file(i));
        end
        if ~isfile(historyFile)
            error('CBSBMemAudit:MissingHistory', ...
                'Missing BMem history for %s run %d.',problem,run);
        end
        fprintf('Auditing %s run %d (%d/%d)\n',problem,run,i,height(Runs));
        T = readtable(historyFile,'Delimiter',',','TextType','string', ...
            'VariableNamingRule','preserve');
        [GroupParts{i},SnapshotParts{i},RunRows(i)] = ...
            auditOneRun(T,Runs(i,:));

        outRun = fullfile(OutputDir,sprintf('%s_run%d',problem,run));
        if ~isfolder(outRun)
            mkdir(outRun);
        end
        writetable(GroupParts{i},fullfile(outRun,'group_audit.csv'));
        writetable(SnapshotParts{i},fullfile(outRun,'snapshot_audit.csv'));
    end

    GroupAudit = vertcat(GroupParts{:});
    SnapshotAudit = vertcat(SnapshotParts{:});
    RunAudit = struct2table(RunRows);
    ProblemAudit = summarizeProblems(RunAudit);
    TrainCountAudit = summarizeTrainCounts(SnapshotAudit, ...
        CBS_RegionWGAN_GP.mainlineDefaults().minGANTrainCount);
    Stats = pairedStatistics(RunAudit);
    Decision = auditDecision(GroupAudit,RunAudit,Stats);

    writetable(GroupAudit,fullfile(OutputDir,'group_audit_all.csv'));
    writetable(SnapshotAudit,fullfile(OutputDir,'snapshot_audit_all.csv'));
    writetable(RunAudit,fullfile(OutputDir,'run_audit.csv'));
    writetable(ProblemAudit,fullfile(OutputDir,'problem_audit.csv'));
    writetable(TrainCountAudit,fullfile(OutputDir,'train_count_audit.csv'));
    writetable(Stats,fullfile(OutputDir,'paired_statistics.csv'));
    writetable(Decision,fullfile(OutputDir,'audit_decision.csv'));
    Manifest = auditManifest(InputRoot,OutputDir,Runs,GroupAudit,SnapshotAudit);
    writetable(Manifest,fullfile(OutputDir,'audit_manifest.csv'));
    fprintf('BMem clusterability audit written to %s\n',OutputDir);
end

function [GroupAudit,SnapshotAudit,RunRow] = auditOneRun(T,Run)
    required = ["snapshot_index","generation","fe","bmem_row", ...
        "sample_id_f","ref","gap","age_f","front_rank_f", ...
        "candidate_row_f"];
    requireColumns(T,required);
    problem = string(Run.problem(1));
    run = double(Run.run(1));
    seed = double(Run.seed(1));
    N = double(Run.N(1));
    D = double(Run.D(1));
    M = double(Run.M(1));
    maxFE = double(Run.maxFE(1));
    xNames = "x_f" + (1:D);
    yNames = "y_f" + (1:M);
    requireColumns(T,[xNames,yNames]);
    T = sortrows(T,{'snapshot_index','ref','bmem_row'});

    Problem = feval(char(problem),'N',N,'D',D,'maxFE',maxFE);
    if Problem.FE ~= 0
        error('CBSBMemAudit:UnexpectedFE', ...
            'Problem construction consumed FE before the audit.');
    end
    PF = Problem.PF;
    lower = double(Problem.lower(:)');
    span = double(Problem.upper(:)' - Problem.lower(:)');
    span(span <= eps) = 1;
    Defaults = CBS_RegionWGAN_GP.mainlineDefaults();
    nRef = max(2,round(N/Defaults.refDivisor));
    [W,~] = UniformPoint(nRef,M);

    snapshots = double(T.snapshot_index);
    refsAll = double(T.ref);
    newGroup = [true;diff(snapshots) ~= 0 | diff(refsAll) ~= 0];
    nGroup = sum(newGroup);
    newSnapshot = [true;diff(snapshots) ~= 0];
    snapshotStart = find(newSnapshot);
    snapshotEnd = [snapshotStart(2:end)-1;height(T)];
    nSnapshot = numel(snapshotStart);
    GroupRows = repmat(emptyGroupRow(),nGroup,1);
    SnapshotRows = repmat(emptySnapshotRow(),nSnapshot,1);
    Previous = cell(size(W,1),1);
    groupCounter = 0;

    for s = 1 : nSnapshot
        first = snapshotStart(s);
        last = snapshotEnd(s);
        snapRows = (first:last)';
        snapRefs = refsAll(snapRows);
        refStartLocal = find([true;diff(snapRefs) ~= 0]);
        refEndLocal = [refStartLocal(2:end)-1;numel(snapRows)];
        groups = repmat(emptyWorkingGroup(),numel(refStartLocal),1);
        dedupMask = false(numel(snapRows),1);
        gatedMask = false(numel(snapRows),1);

        for j = 1 : numel(refStartLocal)
            local = refStartLocal(j):refEndLocal(j);
            rows = snapRows(local);
            uniqueRows = deduplicateRows(T,rows);
            X = double(T{uniqueRows,cellstr(xNames)});
            X = (X - lower)./span;
            ids = normalizedIDs(T.sample_id_f(uniqueRows),uniqueRows);
            groups(j).ref = refsAll(rows(1));
            groups(j).rawRows = rows;
            groups(j).uniqueRows = uniqueRows;
            groups(j).X = X;
            groups(j).ids = ids;
            dedupMask(uniqueRows-first+1) = true;
            gatedMask(uniqueRows-first+1) = true;
        end

        groupRefs = [groups.ref]';
        for j = 1 : numel(groups)
            groupCounter = groupCounter + 1;
            g = groups(j);
            neighborRefs = oneHopNeighbors(W,g.ref);
            neighborGroups = find(ismember(groupRefs,neighborRefs));
            NeighborX = zeros(0,D);
            for k = neighborGroups(:)'
                NeighborX = [NeighborX;groups(k).X]; %#ok<AGROW>
            end
            [labels,largestEdge,secondEdge] = strongestMSTSplit(g.X);
            localScale = localContinuityScale(g.X,NeighborX);
            separationRatio = safeRatio(largestEdge,localScale);
            separated = numel(labels) >= 2 && isfinite(largestEdge) && ...
                isfinite(localScale) && largestEdge > localScale;

            [quality1,quality2] = componentQualities(T,g.uniqueRows,labels);
            dominated = dominatedComponent(quality1,quality2);
            valueDominated = dominated > 0;
            current = struct('snapshot',snapshots(first), ...
                'ids',g.ids,'labels',labels, ...
                'separated',separated,'dominated',dominated);
            previous = [];
            if isfinite(g.ref) && g.ref >= 1 && g.ref <= numel(Previous)
                previous = Previous{g.ref};
            end
            temporal = temporalEvidence(current,previous);
            pruneEligible = temporal.valueStable;
            removed = 0;
            if pruneEligible
                removeRows = g.uniqueRows(labels == dominated);
                gatedMask(removeRows-first+1) = false;
                removed = numel(removeRows);
            end
            if isfinite(g.ref) && g.ref >= 1 && g.ref <= numel(Previous)
                Previous{g.ref} = current;
            end

            row = emptyGroupRow();
            row.problem = problem;
            row.run = run;
            row.seed = seed;
            row.snapshot_index = snapshots(first);
            row.generation = double(T.generation(first));
            row.fe = double(T.fe(first));
            row.ref = g.ref;
            row.raw_count = numel(g.rawRows);
            row.unique_count = numel(g.uniqueRows);
            row.duplicate_count = row.raw_count - row.unique_count;
            row.duplicate_fraction = safeRatio(row.duplicate_count,row.raw_count);
            origin = duplicateOrigins(T,g.rawRows);
            row.duplicate_current_only = origin.currentOnly;
            row.duplicate_previous_only = origin.previousOnly;
            row.duplicate_mixed = origin.mixed;
            row.duplicate_unknown = origin.unknown;
            row.largest_mst_edge = largestEdge;
            row.second_mst_edge = secondEdge;
            row.local_continuity_scale = localScale;
            row.separation_ratio = separationRatio;
            row.separated_candidate = separated;
            row.component1_count = sum(labels == 1);
            row.component2_count = sum(labels == 2);
            row.component1_gap = quality1(1);
            row.component1_front_rank = quality1(2);
            row.component1_age = quality1(3);
            row.component2_gap = quality2(1);
            row.component2_front_rank = quality2(2);
            row.component2_age = quality2(3);
            row.dominated_component = dominated;
            row.value_dominated = valueDominated;
            row.previous_overlap_ids = temporal.overlapIDs;
            row.previous_snapshot_gap = temporal.snapshotGap;
            row.previous_pair_count = temporal.pairCount;
            row.coassignment_agreement = temporal.agreement;
            row.temporal_split_stable = temporal.splitStable;
            row.temporal_value_stable = temporal.valueStable;
            row.prune_eligible = pruneEligible;
            row.removed_count = removed;
            GroupRows(groupCounter) = row;
        end

        Obj = double(T{snapRows,cellstr(yNames)});
        Masks = [true(numel(snapRows),1),dedupMask,gatedMask];
        Metrics = RunRegionGAN_RC('trueboundarysubsetdiagnostics', ...
            Obj,[],PF,Masks,struct());
        row = emptySnapshotRow();
        row.problem = problem;
        row.run = run;
        row.seed = seed;
        row.snapshot_index = snapshots(first);
        row.generation = double(T.generation(first));
        row.fe = double(T.fe(first));
        row.raw_count = numel(snapRows);
        row.dedup_count = sum(dedupMask);
        row.gated_count = sum(gatedMask);
        row.duplicate_count = row.raw_count - row.dedup_count;
        row.removed_count = row.dedup_count - row.gated_count;
        row.populated_ref_count = numel(groups);
        row.raw_bdist50_true = Metrics(1).bdist50_true;
        row.raw_bwidth90_10_true = Metrics(1).bwidth90_10_true;
        row.raw_bcover_eps_true = Metrics(1).bcover_eps_true;
        row.dedup_bdist50_true = Metrics(2).bdist50_true;
        row.dedup_bwidth90_10_true = Metrics(2).bwidth90_10_true;
        row.dedup_bcover_eps_true = Metrics(2).bcover_eps_true;
        row.gated_bdist50_true = Metrics(3).bdist50_true;
        row.gated_bwidth90_10_true = Metrics(3).bwidth90_10_true;
        row.gated_bcover_eps_true = Metrics(3).bcover_eps_true;
        SnapshotRows(s) = row;
    end
    if Problem.FE ~= 0
        error('CBSBMemAudit:HiddenEvaluation', ...
            'The offline audit changed Problem.FE from zero.');
    end
    GroupAudit = struct2table(GroupRows);
    SnapshotAudit = struct2table(SnapshotRows);
    RunRow = summarizeRun(GroupAudit,SnapshotAudit);
end

function uniqueRows = deduplicateRows(T,rows)
    ids = normalizedIDs(T.sample_id_f(rows),rows);
    uniqueIDs = unique(ids,'stable');
    uniqueRows = zeros(numel(uniqueIDs),1);
    for i = 1 : numel(uniqueIDs)
        candidates = rows(ids == uniqueIDs(i));
        score = [finiteOrInf(double(T.gap(candidates))), ...
            finiteOrInf(double(T.front_rank_f(candidates))), ...
            finiteOrInf(double(T.age_f(candidates))), ...
            finiteOrInf(double(T.candidate_row_f(candidates))), ...
            double(candidates)];
        [~,ord] = sortrows(score,[1 2 3 4 5]);
        uniqueRows(i) = candidates(ord(1));
    end
    uniqueRows = sort(uniqueRows);
end

function ids = normalizedIDs(ids,rows)
    ids = string(ids(:));
    missing = ismissing(ids) | strlength(strtrim(ids)) == 0;
    if any(missing)
        ids(missing) = "missing_row_" + string(rows(missing));
    end
end

function [labels,largestEdge,secondEdge] = strongestMSTSplit(X)
    n = size(X,1);
    labels = ones(n,1);
    largestEdge = NaN;
    secondEdge = NaN;
    if n < 2
        return;
    end
    pairs = zeros(n*(n-1)/2,3);
    q = 0;
    for i = 1 : n-1
        for j = i+1 : n
            q = q + 1;
            pairs(q,:) = [sqrt(sum((X(i,:) - X(j,:)).^2)),i,j];
        end
    end
    pairs = sortrows(pairs,[1 2 3]);
    parent = 1:n;
    edges = zeros(n-1,3);
    count = 0;
    for i = 1 : size(pairs,1)
        a = findRoot(parent,pairs(i,2));
        b = findRoot(parent,pairs(i,3));
        if a == b
            continue;
        end
        count = count + 1;
        edges(count,:) = pairs(i,:);
        parent(b) = a;
        if count == n-1
            break;
        end
    end
    edges = edges(1:count,:);
    [~,cut] = max(edges(:,1));
    largestEdge = edges(cut,1);
    remaining = edges(:,1);
    remaining(cut) = [];
    if ~isempty(remaining)
        secondEdge = max(remaining);
    end
    adjacency = false(n,n);
    for i = 1 : size(edges,1)
        if i == cut
            continue;
        end
        a = edges(i,2);
        b = edges(i,3);
        adjacency(a,b) = true;
        adjacency(b,a) = true;
    end
    component1 = false(n,1);
    queue = zeros(n,1);
    queue(1) = 1;
    component1(1) = true;
    head = 1;
    tail = 1;
    while head <= tail
        node = queue(head);
        head = head + 1;
        next = find(adjacency(node,:) & ~component1');
        for k = next
            tail = tail + 1;
            queue(tail) = k;
            component1(k) = true;
        end
    end
    labels(~component1) = 2;
end

function root = findRoot(parent,node)
    root = node;
    while parent(root) ~= root
        root = parent(root);
    end
end

function scale = localContinuityScale(X,NeighborX)
    scale = NaN;
    if isempty(X) || isempty(NeighborX)
        return;
    end
    D2 = max(sum(X.^2,2) + sum(NeighborX.^2,2)' - 2*(X*NeighborX'),0);
    scale = median(sqrt(min(D2,[],2)),'omitnan');
end

function refs = oneHopNeighbors(W,ref)
    refs = zeros(0,1);
    ref = round(double(ref));
    if isempty(W) || ~isfinite(ref) || ref < 1 || ref > size(W,1)
        return;
    end
    d = sqrt(sum((double(W) - double(W(ref,:))).^2,2));
    [~,ord] = sortrows([d,(1:size(W,1))'],[1 2]);
    ord = ord(1:min(3,numel(ord)));
    refs = ord(ord ~= ref);
end

function [q1,q2] = componentQualities(T,rows,labels)
    q1 = qualityVector(T,rows(labels == 1));
    q2 = qualityVector(T,rows(labels == 2));
end

function q = qualityVector(T,rows)
    if isempty(rows)
        q = [NaN,NaN,NaN];
        return;
    end
    q = [medianFinite(double(T.gap(rows))), ...
        medianFinite(double(T.front_rank_f(rows))), ...
        medianFinite(double(T.age_f(rows)))];
end

function component = dominatedComponent(q1,q2)
    component = 0;
    if any(~isfinite([q1,q2]))
        return;
    end
    if all(q1 <= q2) && any(q1 < q2)
        component = 2;
    elseif all(q2 <= q1) && any(q2 < q1)
        component = 1;
    end
end

function E = temporalEvidence(current,previous)
    E = struct('snapshotGap',NaN,'overlapIDs',0,'pairCount',0,'agreement',NaN, ...
        'splitStable',false,'valueStable',false);
    if isempty(previous) || ~isstruct(previous)
        return;
    end
    E.snapshotGap = current.snapshot - previous.snapshot;
    if E.snapshotGap ~= 1
        return;
    end
    overlap = intersect(current.ids,previous.ids,'stable');
    E.overlapIDs = numel(overlap);
    if numel(overlap) < 2
        return;
    end
    [~,cur] = ismember(overlap,current.ids);
    [~,old] = ismember(overlap,previous.ids);
    agreement = zeros(numel(overlap)*(numel(overlap)-1)/2,1);
    q = 0;
    for i = 1 : numel(overlap)-1
        for j = i+1 : numel(overlap)
            q = q + 1;
            sameNow = current.labels(cur(i)) == current.labels(cur(j));
            sameOld = previous.labels(old(i)) == previous.labels(old(j));
            agreement(q) = sameNow == sameOld;
        end
    end
    E.pairCount = q;
    E.agreement = mean(agreement);
    E.splitStable = current.separated && previous.separated && ...
        E.agreement == 1;
    if ~E.splitStable || current.dominated == 0 || previous.dominated == 0
        return;
    end
    currentBad = current.ids(current.labels == current.dominated);
    previousBad = previous.ids(previous.labels == previous.dominated);
    currentGood = current.ids(current.labels ~= current.dominated);
    previousGood = previous.ids(previous.labels ~= previous.dominated);
    E.valueStable = ~isempty(intersect(currentBad,previousBad)) && ...
        ~isempty(intersect(currentGood,previousGood));
end

function RunRow = summarizeRun(G,S)
    RunRow = emptyRunRow();
    RunRow.problem = string(G.problem(1));
    RunRow.run = double(G.run(1));
    RunRow.seed = double(G.seed(1));
    RunRow.snapshot_count = height(S);
    RunRow.group_count = height(G);
    RunRow.raw_row_count = sum(G.raw_count);
    RunRow.unique_row_count = sum(G.unique_count);
    RunRow.duplicate_row_count = sum(G.duplicate_count);
    RunRow.duplicate_group_count = sum(G.duplicate_count > 0);
    RunRow.duplicate_current_only = sum(G.duplicate_current_only);
    RunRow.duplicate_previous_only = sum(G.duplicate_previous_only);
    RunRow.duplicate_mixed = sum(G.duplicate_mixed);
    RunRow.duplicate_unknown = sum(G.duplicate_unknown);
    RunRow.separated_candidate_count = sum(G.separated_candidate);
    RunRow.value_dominated_count = sum(G.value_dominated);
    RunRow.temporal_assessable_count = sum(G.previous_pair_count > 0);
    RunRow.temporal_split_stable_count = sum(G.temporal_split_stable);
    RunRow.prune_eligible_count = sum(G.prune_eligible);
    RunRow.removed_row_count = sum(G.removed_count);
    RunRow.duplicate_row_fraction = safeRatio( ...
        RunRow.duplicate_row_count,RunRow.raw_row_count);
    RunRow.separated_candidate_fraction = safeRatio( ...
        RunRow.separated_candidate_count,RunRow.group_count);
    RunRow.prune_eligible_fraction = safeRatio( ...
        RunRow.prune_eligible_count,RunRow.group_count);
    RunRow.removed_row_fraction = safeRatio( ...
        RunRow.removed_row_count,RunRow.unique_row_count);
    metrics = ["raw_bdist50_true","raw_bwidth90_10_true", ...
        "raw_bcover_eps_true","dedup_bdist50_true", ...
        "dedup_bwidth90_10_true","dedup_bcover_eps_true", ...
        "gated_bdist50_true","gated_bwidth90_10_true", ...
        "gated_bcover_eps_true"];
    for name = metrics
        RunRow.("median_" + name) = medianFinite(S.(name));
    end
end

function P = summarizeProblems(R)
    problems = unique(R.problem,'stable');
    rows = repmat(struct('problem',"",'run_count',0,'group_count',0, ...
        'duplicate_row_count',0,'duplicate_group_count',0, ...
        'duplicate_current_only',0,'duplicate_previous_only',0, ...
        'duplicate_mixed',0,'duplicate_unknown',0, ...
        'separated_candidate_count',0,'prune_eligible_count',0, ...
        'duplicate_row_fraction',NaN,'prune_eligible_fraction',NaN), ...
        numel(problems),1);
    for i = 1 : numel(problems)
        X = R(R.problem == problems(i),:);
        rows(i).problem = problems(i);
        rows(i).run_count = height(X);
        rows(i).group_count = sum(X.group_count);
        rows(i).duplicate_row_count = sum(X.duplicate_row_count);
        rows(i).duplicate_group_count = sum(X.duplicate_group_count);
        rows(i).duplicate_current_only = sum(X.duplicate_current_only);
        rows(i).duplicate_previous_only = sum(X.duplicate_previous_only);
        rows(i).duplicate_mixed = sum(X.duplicate_mixed);
        rows(i).duplicate_unknown = sum(X.duplicate_unknown);
        rows(i).separated_candidate_count = sum(X.separated_candidate_count);
        rows(i).prune_eligible_count = sum(X.prune_eligible_count);
        rows(i).duplicate_row_fraction = safeRatio( ...
            sum(X.duplicate_row_count),sum(X.raw_row_count));
        rows(i).prune_eligible_fraction = safeRatio( ...
            sum(X.prune_eligible_count),sum(X.group_count));
    end
    P = struct2table(rows);
end

function T = summarizeTrainCounts(S,minTrainCount)
    problems = unique(S.problem,'stable');
    rows = repmat(struct('problem',"",'snapshot_count',0, ...
        'raw_train_count_median',NaN,'dedup_train_count_median',NaN, ...
        'dedup_to_raw_count_ratio',NaN,'raw_trainable_count',0, ...
        'dedup_trainable_count',0,'crossed_below_min_count',0, ...
        'min_gan_train_count',minTrainCount),numel(problems),1);
    for i = 1 : numel(problems)
        X = S(S.problem == problems(i),:);
        rows(i).problem = problems(i);
        rows(i).snapshot_count = height(X);
        rows(i).raw_train_count_median = medianFinite(X.raw_count);
        rows(i).dedup_train_count_median = medianFinite(X.dedup_count);
        rows(i).dedup_to_raw_count_ratio = safeRatio( ...
            sum(X.dedup_count),sum(X.raw_count));
        rows(i).raw_trainable_count = sum(X.raw_count >= minTrainCount);
        rows(i).dedup_trainable_count = sum(X.dedup_count >= minTrainCount);
        rows(i).crossed_below_min_count = sum(X.raw_count >= minTrainCount & ...
            X.dedup_count < minTrainCount);
    end
    T = struct2table(rows);
end

function Origin = duplicateOrigins(T,rows)
    Origin = struct('currentOnly',0,'previousOnly',0,'mixed',0,'unknown',0);
    ids = normalizedIDs(T.sample_id_f(rows),rows);
    source = nan(numel(rows),1);
    if ismember('source_f',T.Properties.VariableNames)
        source = double(T.source_f(rows));
    end
    uniqueIDs = unique(ids,'stable');
    for i = 1 : numel(uniqueIDs)
        members = find(ids == uniqueIDs(i));
        excess = numel(members) - 1;
        if excess <= 0
            continue;
        end
        values = source(members);
        if all(values == 0)
            Origin.currentOnly = Origin.currentOnly + excess;
        elseif all(values == 1)
            Origin.previousOnly = Origin.previousOnly + excess;
        elseif all(ismember(values,[0,1])) && any(values == 0) && any(values == 1)
            Origin.mixed = Origin.mixed + excess;
        else
            Origin.unknown = Origin.unknown + excess;
        end
    end
end

function Stats = pairedStatistics(R)
    contrasts = ["dedup_minus_raw","gated_minus_dedup"];
    metrics = ["bdist50_true","bwidth90_10_true","bcover_eps_true"];
    rows = repmat(struct('contrast',"",'metric',"",'better_direction',"", ...
        'n',0,'median_delta',NaN,'wins',0,'ties',0,'losses',0, ...
        'p_value',NaN,'p_holm_primary',NaN),numel(contrasts)*numel(metrics),1);
    q = 0;
    for c = 1 : numel(contrasts)
        primaryRows = zeros(2,1);
        for m = 1 : numel(metrics)
            q = q + 1;
            if contrasts(c) == "dedup_minus_raw"
                control = R.("median_raw_" + metrics(m));
                variant = R.("median_dedup_" + metrics(m));
            else
                control = R.("median_dedup_" + metrics(m));
                variant = R.("median_gated_" + metrics(m));
            end
            valid = isfinite(control) & isfinite(variant);
            delta = variant(valid) - control(valid);
            lowerBetter = metrics(m) ~= "bcover_eps_true";
            if lowerBetter
                wins = sum(delta < 0);
                losses = sum(delta > 0);
                direction = "lower";
            else
                wins = sum(delta > 0);
                losses = sum(delta < 0);
                direction = "higher";
            end
            rows(q).contrast = contrasts(c);
            rows(q).metric = metrics(m);
            rows(q).better_direction = direction;
            rows(q).n = sum(valid);
            rows(q).median_delta = medianFinite(delta);
            rows(q).wins = wins;
            rows(q).ties = sum(delta == 0);
            rows(q).losses = losses;
            rows(q).p_value = signedRankP(control(valid),variant(valid));
            if m <= 2
                primaryRows(m) = q;
            end
        end
        adjusted = holmAdjust([rows(primaryRows).p_value]);
        for m = 1 : 2
            rows(primaryRows(m)).p_holm_primary = adjusted(m);
        end
    end
    Stats = struct2table(rows);
end

function p = signedRankP(control,variant)
    delta = double(variant(:) - control(:));
    delta = delta(isfinite(delta) & delta ~= 0);
    if isempty(delta)
        p = 1;
        return;
    end
    if exist('signrank','file') ~= 2
        error('CBSBMemAudit:MissingSignrank', ...
            'Statistics and Machine Learning Toolbox signrank is required.');
    end
    p = signrank(control,variant);
end

function adjusted = holmAdjust(p)
    p = double(p(:));
    adjusted = NaN(size(p));
    valid = isfinite(p);
    values = p(valid);
    [sorted,order] = sort(values);
    m = numel(sorted);
    scaled = zeros(m,1);
    running = 0;
    for i = 1 : m
        running = max(running,(m-i+1)*sorted(i));
        scaled(i) = min(1,running);
    end
    restored = zeros(m,1);
    restored(order) = scaled;
    adjusted(valid) = restored;
    adjusted = adjusted(:)';
end

function D = auditDecision(G,R,Stats)
    problems = unique(R.problem);
    duplicateProblems = unique(G.problem(G.duplicate_count > 0));
    eligibleProblems = unique(G.problem(G.prune_eligible));
    dedupGate = numel(duplicateProblems) == numel(problems);
    clusterPhenomenon = numel(eligibleProblems) == numel(problems);
    distance = Stats(Stats.contrast == "gated_minus_dedup" & ...
        Stats.metric == "bdist50_true",:);
    width = Stats(Stats.contrast == "gated_minus_dedup" & ...
        Stats.metric == "bwidth90_10_true",:);
    coverage = Stats(Stats.contrast == "gated_minus_dedup" & ...
        Stats.metric == "bcover_eps_true",:);
    clusterGeometry = height(distance) == 1 && height(width) == 1 && ...
        distance.median_delta < 0 && width.median_delta < 0 && ...
        distance.p_holm_primary < 0.05 && width.p_holm_primary < 0.05;
    coverageGuard = height(coverage) == 1 && coverage.median_delta >= 0;
    clusterGate = clusterPhenomenon && clusterGeometry && coverageGuard;
    if dedupGate
        nextStep = "validate_exact_sample_id_dedup_only";
        status = "dedup_prerequisite_supported";
    elseif clusterGate
        nextStep = "validate_stable_value_mode_pruning_only";
        status = "cluster_prerequisite_supported";
    else
        nextStep = "stop_mode_filtering_route";
        status = "mode_filtering_prerequisite_rejected";
    end
    D = table(status,nextStep,dedupGate,clusterPhenomenon, ...
        clusterGeometry,coverageGuard,clusterGate, ...
        'VariableNames',{'status','next_step','dedup_all_problems', ...
        'cluster_eligible_all_problems','cluster_geometry_gate', ...
        'cluster_coverage_guard','cluster_gate'});
end

function T = auditDesign()
    stage = ["scope";"dedup";"candidate_split";"local_scale"; ...
        "separation";"value";"temporal";"offline_simulation"; ...
        "dedup_gate";"cluster_gate"];
    rule = [ ...
        "read_only"; ...
        "exact_sample_id_within_snapshot_ref"; ...
        "cut_largest_mst_edge"; ...
        "median_nearest_distance_to_one_hop_refs"; ...
        "largest_mst_edge_gt_local_scale"; ...
        "pareto_dominance_of_median_gap_rank_age"; ...
        "overlap_pair_coassignment_and_bad_good_id_persistence"; ...
        "raw_vs_dedup_vs_temporally_gated_subsets"; ...
        "duplicates_present_in_every_problem"; ...
        "eligible_every_problem_and_holm_geometry_and_coverage_guard"];
    online_inputs = [ ...
        "exported_real_evaluations_only";"sample_id,gap,front_rank,age"; ...
        "normalized_x_f";"normalized_x_f,W";"normalized_x_f,W"; ...
        "gap,front_rank,age";"adjacent_snapshot_sample_id,split,value"; ...
        "offline_true_boundary_only_for_evaluation";"sample_id"; ...
        "online_gate_plus_offline_evaluation"];
    numeric_parameters = zeros(numel(stage),1);
    failure = [ ...
        "any_FE_change_or_Evaluation_call"; ...
        "no_cross_problem_exact_duplicates"; ...
        "candidate_alone_is_not_mode_evidence"; ...
        "missing_neighbor_support_is_not_separated"; ...
        "within_gap_not_larger_than_local_continuity"; ...
        "neither_component_dominates"; ...
        "split_or_value_membership_does_not_persist"; ...
        "geometry_or_coverage_guard_fails"; ...
        "duplicates_not_observed_in_every_problem"; ...
        "any_locked_cluster_gate_fails"];
    T = table(stage,rule,online_inputs,numeric_parameters,failure);
end

function T = auditManifest(InputRoot,OutputDir,Runs,G,S)
    schema_version = "cbs_bmem_clusterability_audit_v1";
    created_at = string(datetime('now','TimeZone','local', ...
        'Format','yyyy-MM-dd''T''HH:mm:ssXXX'));
    input_root = string(InputRoot);
    output_root = string(OutputDir);
    input_run_count = height(Runs);
    group_row_count = height(G);
    snapshot_row_count = height(S);
    evaluation_calls = 0;
    git_sha = "";
    source_tree_sha256 = "";
    if ismember('git_sha',Runs.Properties.VariableNames)
        git_sha = strjoin(unique(Runs.git_sha),";");
    end
    if ismember('source_tree_sha256',Runs.Properties.VariableNames)
        source_tree_sha256 = strjoin(unique(Runs.source_tree_sha256),";");
    end
    T = table(schema_version,created_at,input_root,output_root, ...
        input_run_count,group_row_count,snapshot_row_count,evaluation_calls, ...
        git_sha,source_tree_sha256);
end

function requireColumns(T,names)
    names = string(names(:));
    missing = names(~ismember(names,string(T.Properties.VariableNames)));
    if ~isempty(missing)
        error('CBSBMemAudit:MissingColumns','Missing columns: %s', ...
            strjoin(missing,', '));
    end
end

function x = finiteOrInf(x)
    x = double(x(:));
    x(~isfinite(x)) = inf;
end

function value = medianFinite(x)
    x = double(x(:));
    x = x(isfinite(x));
    if isempty(x)
        value = NaN;
    else
        value = median(x);
    end
end

function value = safeRatio(a,b)
    a = double(a);
    b = double(b);
    if isempty(a) || isempty(b) || ~isfinite(a(1)) || ...
            ~isfinite(b(1)) || b(1) == 0
        if ~isempty(a) && isfinite(a(1)) && a(1) > 0 && ...
                ~isempty(b) && b(1) == 0
            value = inf;
        else
            value = NaN;
        end
    else
        value = a(1)/b(1);
    end
end

function G = emptyWorkingGroup()
    G = struct('ref',NaN,'rawRows',zeros(0,1), ...
        'uniqueRows',zeros(0,1),'X',zeros(0,0),'ids',strings(0,1));
end

function R = emptyGroupRow()
    R = struct('problem',"",'run',0,'seed',0,'snapshot_index',0, ...
        'generation',0,'fe',0,'ref',0,'raw_count',0,'unique_count',0, ...
        'duplicate_count',0,'duplicate_fraction',NaN, ...
        'duplicate_current_only',0,'duplicate_previous_only',0, ...
        'duplicate_mixed',0,'duplicate_unknown',0, ...
        'largest_mst_edge',NaN,'second_mst_edge',NaN, ...
        'local_continuity_scale',NaN,'separation_ratio',NaN, ...
        'separated_candidate',false,'component1_count',0, ...
        'component2_count',0,'component1_gap',NaN, ...
        'component1_front_rank',NaN,'component1_age',NaN, ...
        'component2_gap',NaN,'component2_front_rank',NaN, ...
        'component2_age',NaN,'dominated_component',0, ...
        'value_dominated',false,'previous_overlap_ids',0, ...
        'previous_snapshot_gap',NaN,'previous_pair_count',0, ...
        'coassignment_agreement',NaN, ...
        'temporal_split_stable',false,'temporal_value_stable',false, ...
        'prune_eligible',false,'removed_count',0);
end

function R = emptySnapshotRow()
    R = struct('problem',"",'run',0,'seed',0,'snapshot_index',0, ...
        'generation',0,'fe',0,'raw_count',0,'dedup_count',0, ...
        'gated_count',0,'duplicate_count',0,'removed_count',0, ...
        'populated_ref_count',0,'raw_bdist50_true',NaN, ...
        'raw_bwidth90_10_true',NaN,'raw_bcover_eps_true',NaN, ...
        'dedup_bdist50_true',NaN,'dedup_bwidth90_10_true',NaN, ...
        'dedup_bcover_eps_true',NaN,'gated_bdist50_true',NaN, ...
        'gated_bwidth90_10_true',NaN,'gated_bcover_eps_true',NaN);
end

function R = emptyRunRow()
    R = struct('problem',"",'run',0,'seed',0,'snapshot_count',0, ...
        'group_count',0,'raw_row_count',0,'unique_row_count',0, ...
        'duplicate_row_count',0,'duplicate_group_count',0, ...
        'duplicate_current_only',0,'duplicate_previous_only',0, ...
        'duplicate_mixed',0,'duplicate_unknown',0, ...
        'separated_candidate_count',0,'value_dominated_count',0, ...
        'temporal_assessable_count',0,'temporal_split_stable_count',0, ...
        'prune_eligible_count',0,'removed_row_count',0, ...
        'duplicate_row_fraction',NaN,'separated_candidate_fraction',NaN, ...
        'prune_eligible_fraction',NaN,'removed_row_fraction',NaN, ...
        'median_raw_bdist50_true',NaN, ...
        'median_raw_bwidth90_10_true',NaN, ...
        'median_raw_bcover_eps_true',NaN, ...
        'median_dedup_bdist50_true',NaN, ...
        'median_dedup_bwidth90_10_true',NaN, ...
        'median_dedup_bcover_eps_true',NaN, ...
        'median_gated_bdist50_true',NaN, ...
        'median_gated_bwidth90_10_true',NaN, ...
        'median_gated_bcover_eps_true',NaN);
end
