function T = PairGuide_checkpoint_metrics(Problem,E,targets,Record)
%PAIRGUIDE_CHECKPOINT_METRICS Real completed states; first use and selected FE.
%   No search oracle calls. The global random stream is restored after metrics.
    times = cellfun(@(p)p.observationFE,E.population);
    first = find(cellfun(@(g)g.use.selected>0,E.generations),1);
    labels = "first_true_use"; requested = NaN;
    if ~isempty(first); requested = times(first+1); end
    reached = reshape(targets(targets<=times(end)),1,[]);
    labels = [labels,compose('FE%06d',reached)]; requested = [requested,reached];
    saved = rng; cleanup = onCleanup(@()rng(saved));
    rng(314159,'twister');
    rows = struct([]);
    for k = 1:numel(requested)
        row = struct('problem',Record.problem,'seed',Record.seed,'mode',Record.mode, ...
            'stage',labels(k),'targetFE',requested(k),'actualFE',NaN,'missing',true, ...
            'IGD',NaN,'HV',NaN,'p1Feasible',NaN,'p2Feasible',NaN, ...
            'populationOverlap',NaN,'archivePairs',NaN,'activePairs',NaN, ...
            'inactivePairs',NaN,'eligiblePairs',NaN,'activeRefCoverage',NaN, ...
            'frontRejectedPairs',NaN,'infeasibleDominatedPairs',NaN,'eligibilityRemoved',NaN, ...
            'feasiblePoolEligible',NaN,'infeasiblePoolEligible',NaN, ...
            'gapMedian',NaN,'gapP90',NaN,'trainingEvents',0,'generatorUpdates',0, ...
            'criticUpdates',0,'trainingSeconds',0,'lastTrainingFE',NaN, ...
            'lastTrainingPairs',NaN,'lastTrainingEpochs',NaN,'lastTrainingUpdates',NaN, ...
            'lastTrainingEndpointRMSE',NaN,'modelVersion',NaN, ...
            'consumedTrainingFE',NaN,'consumedTrainingPairs',NaN, ...
            'consumedTrainingEpochs',NaN,'consumedTrainingUpdates',NaN,'consumedTrainingRMSE',NaN, ...
            'productionFE',NaN,'selectedCount',0,'selectedFeasibleRate',NaN, ...
            'survivedP1',0,'survivedP2',0,'cumulativeGuided',0, ...
            'nativeBandRate',NaN,'rebuildInputPoints',NaN,'netGapReduction',NaN);
        p = find(times<=requested(k),1,'last');
        if ~isempty(p)
            S = E.population{p}; A = S.archive;
            row.actualFE = times(p); row.missing = false;
            Pop = SOLUTION(S.p1Decs,S.p1Objs,S.p1Cons);
            row.IGD = Problem.CalMetric('IGD',Pop); row.HV = Problem.CalMetric('HV',Pop);
            row.p1Feasible = nnz(all(S.p1Cons<=0,2)); row.p2Feasible = nnz(all(S.p2Cons<=0,2));
            row.populationOverlap = nnz(ismember(S.p1Decs,S.p2Decs,'rows'));
            row.archivePairs = numel(A.id); row.activePairs = nnz(A.active);
            row.inactivePairs = nnz(~A.active); row.activeRefCoverage = row.activePairs/size(E.W,1);
            if any(A.active)
                row.gapMedian = median(A.gap(A.active)); row.gapP90 = prctile(A.gap(A.active),90);
            end
            trained = E.training(cellfun(@(t)t.trained && t.generation<=p-1,E.training));
            row.trainingEvents = numel(trained);
            if ~isempty(trained)
                t = trained{end}; row.lastTrainingFE = times(t.generation+1);
                row.lastTrainingPairs = t.trainingPairs; row.lastTrainingEpochs = t.epochs;
                row.lastTrainingUpdates = t.updates;
                row.lastTrainingEndpointRMSE = t.postDiagnostics.allEndpointRMSE;
                row.generatorUpdates = sum(cellfun(@(t)t.updates,trained));
                row.criticUpdates = sum(cellfun(@(t)t.criticUpdates,trained));
                row.trainingSeconds = sum(cellfun(@(t)t.trainingSeconds,trained));
            end
            if p>1
                G = E.generations{p-1}; U = G.use;
                row.eligiblePairs = G.archive.eligiblePairs;
                for field = ["frontRejectedPairs","infeasibleDominatedPairs", ...
                        "eligibilityRemoved","feasiblePoolEligible","infeasiblePoolEligible"]
                    if isfield(G.archive,field); row.(field) = G.archive.(field); end
                end
                row.rebuildInputPoints = G.archive.candidateInputPoints;
                row.netGapReduction = G.archive.netGapReduction;
                row.productionFE = U.productionFE; row.selectedCount = U.selected;
                if U.selected>0; row.selectedFeasibleRate = mean(all(U.childCons<=0,2)); end
                row.survivedP1 = nnz(U.survivedP1); row.survivedP2 = nnz(U.survivedP2);
                row.cumulativeGuided = sum(cellfun(@(g)g.use.selected,E.generations(1:p-1)));
                q = find(cellfun(@(q)q.productionFE==U.productionFE,E.queries),1);
                if ~isempty(q)
                    row.modelVersion = E.queries{q}.modelVersion;
                    row.nativeBandRate = mean(E.queries{q}.pool.nativeInBand);
                    history=trained(cellfun(@(t)t.generation<=E.queries{q}.generation,trained));
                    if ~isempty(history)
                        t=history{end}; row.consumedTrainingFE=times(t.generation+1);
                        row.consumedTrainingPairs=t.trainingPairs; row.consumedTrainingEpochs=t.epochs;
                        row.consumedTrainingUpdates=t.updates;
                        row.consumedTrainingRMSE=t.postDiagnostics.allEndpointRMSE;
                    end
                end
            end
        end
        rows = [rows;row]; %#ok<AGROW>
    end
    T = struct2table(rows);
end
