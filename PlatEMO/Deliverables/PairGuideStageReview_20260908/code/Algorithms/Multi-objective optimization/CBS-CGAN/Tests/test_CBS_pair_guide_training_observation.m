function test_CBS_pair_guide_training_observation
%TEST_CBS_PAIR_GUIDE_TRAINING_OBSERVATION Delayed direct evaluation and pure logging.
    addCBSPaths(fileparts(which('platemo')));
    results = cell(1,2); states = cell(1,2);
    for observed = 1:2
        rng(1,'twister');
        % The final single FE has no P1 guide slot; 10% checkpoints also fall
        % between completed generations and must use the preceding state.
        P = LIRCMOP5_BC('N',100,'maxFE',6001);
        A = PairGuide('save',1,'outputFcn',@(varargin)[]);
        A.configurePairGuideTrainingExperiment(struct('initialEpoch',2, ...
            'retrainEpoch',1,'nCritic',1));
        if observed == 2
            % The new default must replay the explicit C configuration exactly.
            A.configureBoundaryExperiment(struct('selectionPool',"ccmo",'archiveFrontDepth',1));
            A.configureObjectiveSpaceSnapshots(struct('enabled',true, ...
                'targetFE',1000:1000:6000,'expectedRawCount',500,'expectedGuidedCount',20));
        end
        A.Solve(P);
        assert(P.FE == 6001);
        results{observed} = A.guideExperimentSnapshot();
        checkpoints = results{observed};
        assert(all(checkpoints.checkpointFE <= checkpoints.checkpointTargets));
        states{observed} = rng;
        if observed == 2
            before = P.FE;
            Snapshots = A.objectiveSpaceSnapshots();
            assert(P.FE == before && all([Snapshots.observationFE] <= [Snapshots.targetFE]) && ...
                all(arrayfun(@(s)isempty(s.rawObjs),Snapshots)));
        end
    end
    E = results{1}.evidence;
    assert(isequaln(states{1},states{2}) && ...
        isequaln(E.population,results{2}.evidence.population) && ...
        isequaln(E.evaluations,results{2}.evidence.evaluations));
    assert(E.fullFE == 6001 && E.objectiveOnlyFE == 0 && E.constraintOnlyFE == 0 && ...
        E.oracleCalls.CalObjRows == 12002 && E.oracleCalls.CalConRows == 6001);
    assert(~isempty(E.queries) && ~isempty(E.training));
    assert(E.networkEndpointRows == 500*numel(E.queries));
    assert(E.schema == "PairGuide-single-v3");
    for k = 1:numel(E.population)
        P = E.population{k}; F = P.p1Objs(all(P.p1Cons <= 0,2),:);
        for y = P.archive.yf(P.archive.active,:)'
            assert(~any(all(F <= y'+1e-12,2) & any(F < y'-1e-12,2)), ...
                'Active feasible endpoints must pass the P1 at this same observation.');
        end
        for y = P.archive.yi'
            assert(~any(all(F <= y'+1e-12,2) & any(F < y'-1e-12,2)), ...
                'No stored infeasible endpoint may be dominated by current feasible P1.');
        end
    end
    positive = 0;
    for k = 1:numel(E.generations)
        G = E.generations{k}; U = G.use;
        assert(G.archive.archiveFrontDepth == 1 && ...
            all(G.archive.retainedFrontRanks == 1) && G.archive.eligibilityRejectedPairs == 0);
        assert(U.selected+U.fallback == U.requested);
        assert(U.requested == 20 || (k == numel(E.generations) && U.requested == 0));
        if U.selected == 0; continue; end
        positive = positive+1;
        assert(U.productionGeneration == U.consumptionGeneration-1 && ...
            U.productionFE < G.consumptionFE);
        query = E.queries{find(cellfun(@(q)q.productionFE == U.productionFE,E.queries),1)};
        assert(all(ismember(U.childDecs,query.pending.decs,'rows')));
        assert(all(U.matchedPairIds==0) && numel(U.requestedSides)==U.selected);
        assert(all(U.evalIDs > U.productionFE) && ...
            all(U.evalIDs <= G.consumptionFE));
        for j = 1:numel(G.archive.events)
            event = G.archive.events(j);
            assert(event.afterGap < event.beforeGap-1e-12);
        end
    end
    assert(positive > 0 && all(cellfun(@(q)q.productionFE < 6000,E.queries)));
    for k = 1:numel(E.queries)
        Q = E.queries{k};
        assert(isequal(Q.pending.decs,Q.rawDecs(Q.pool.keepIdx,:)) && ...
            isequal(Q.pool.candidateDecs,Q.rawDecs) && ~isfield(Q.pool,'constructedDecs'));
        nextCount1 = min(100,ceil(min(200,6001-Q.productionFE)/2));
        assert(round(0.20*nextCount1) > 0 && ...
            size(Q.pending.decs,1) <= round(0.20*nextCount1));
    end
    assert(all(cellfun(@(s)s.generation < E.generations{end-1}.generation,E.training)));
    fprintf('PairGuide direct-use and observation contract passed.\n');
end
