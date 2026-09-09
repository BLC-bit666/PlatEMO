function Summary = analyze_PairGuide_interval_validation(folder)
%ANALYZE_PAIRGUIDE_INTERVAL_VALIDATION Cached geometry, use, feedback and AUC.
%   This analysis never calls CalObj/CalCon. Boundary-distance certification
%   is a separate, explicitly costed offline audit.
    files = dir(fullfile(folder,'*_seed*_*.mat'));
    rows = cell(numel(files),1);
    for k = 1:numel(files)
        R = load(fullfile(files(k).folder,files(k).name),'Record','Audit','Trajectory');
        E = R.Audit.evidence;
        accepted=find(cellfun(@(s)s.trained,E.training),1);
        firstModelGeneration=Inf; firstModelFE=Inf;
        if ~isempty(accepted)
            firstModelGeneration=E.training{accepted}.generation;
            firstModelFE=E.generations{firstModelGeneration}.consumptionFE;
        end
        firstGuideFE=Inf; postQuota=0; postGuided=0;
        requested=0; guided=0; positive=0; survived1=0; survived2=0;
        useEvents=0; budgetGenerations=0;
        tightenedIDs=[]; logShrink=0; netShrink=0; native=0; proposals=0;
        coarse=0; certified=0; nativeCertified=0; pending=0;
        correction=[]; retention=zeros(1,3);
        horizons=[1 5 10];
        for j=1:numel(E.generations)
            G=E.generations{j}; U=G.use;
            useEvents=useEvents+(U.requested>0);
            remaining=min(2*R.Record.N,max(0,R.Record.finalFE-G.consumptionFE));
            budgetGenerations=budgetGenerations+ ...
                (round(0.20*min(R.Record.N,ceil(remaining/2)))>0);
            requested=requested+U.requested; guided=guided+U.selected;
            if U.selected>0; firstGuideFE=min(firstGuideFE,min(U.evalIDs)); end
            if G.generation>firstModelGeneration
                postQuota=postQuota+U.requested; postGuided=postGuided+U.selected;
            end
            positive=positive+(U.selected>0);
            survived1=survived1+nnz(U.survivedP1);
            survived2=survived2+nnz(U.survivedP2);
            netShrink=netShrink+G.archive.netGapReduction;
            events=G.archive.events;
            for h=1:numel(events)
                event=events(h);
                if event.source ~= "guided"; continue; end
                logShrink=logShrink+log(event.beforeGap/event.afterGap);
                if isfield(event,'evalID'); tightenedIDs(end+1)=event.evalID; end %#ok<AGROW>
            end
            for h=1:3
                state=j+1+horizons(h);
                if state<=numel(E.population) && ~isempty(U.childDecs)
                    retention(h)=retention(h)+nnz(ismember(U.childDecs, ...
                        E.population{state}.p1Decs,'rows'));
                end
            end
        end
        for j=1:numel(E.queries)
            Q=E.queries{j}.pool;
            proposals=proposals+Q.rawCount;
            native=native+nnz(Q.nativeInBand);
            if isfield(Q,'candidateDecs')
                valid=all(isfinite(Q.candidateDecs),2) & isfinite(Q.gaps);
                bound=Q.boundaryDistanceUpperBoundRMS;
            else
                % Frozen v1 records used a geometrically projected interval.
                valid=all(isfinite(Q.constructedDecs),2) & isfinite(Q.gaps);
                bound=sqrt(0.3625)*Q.gaps/sqrt(R.Record.D);
            end
            small=valid & bound<=0.003;
            coarse=coarse+nnz(valid & ~small);
            certified=certified+nnz(small);
            nativeCertified=nativeCertified+nnz(small & Q.nativeInBand);
            pending=pending+Q.keptCount;
            if isfield(Q,'correction')
                correction=[correction;Q.correction(isfinite(Q.correction))]; %#ok<AGROW>
            end
        end
        refCount=size(E.W,1);
        time=cellfun(@(p)p.observationFE,E.population)';
        gaps=zeros(size(time)); covered=zeros(size(time));
        gstar=0.003*sqrt(R.Record.D)/sqrt(0.3625);
        for j=1:numel(time)
            A=E.population{j}.archive;
            perRef=sqrt(R.Record.D)*ones(refCount,1); % Fixed missing-direction penalty.
            perRef(A.ref(A.active))=A.gap(A.active);
            gaps(j)=mean(perRef);
            covered(j)=nnz(perRef<=gstar)/refCount;
        end
        duration=max(1,time(end)-time(1));
        training=E.training;
        service=nnz(cellfun(@(s)s.useModel,training));
        rows{k}=struct('problem',R.Record.problem,'seed',R.Record.seed, ...
            'mode',R.Record.mode,'guideQuota',requested,'guideFull',guided, ...
            'firstModelFE',firstModelFE,'firstGuidedFE',firstGuideFE, ...
            'trainingEvents',nnz(cellfun(@(s)s.trained,E.training)), ...
            'postModelQuotaOccupancy',ratio(postGuided,postQuota), ...
            'quotaOccupancy',ratio(guided,requested),'fallbackRate',ratio(requested-guided,requested), ...
            'positiveUseRate',ratio(positive,useEvents), ...
            'modelServiceRate',ratio(service,budgetGenerations), ...
            'queryRate',ratio(numel(E.queries),budgetGenerations), ...
            'nativeBandRate',ratio(native,proposals),'projectionMedian',median(correction), ...
            'coarseIntervalProposals',coarse,'certifiedIntervalProposals',certified, ...
            'nativeCertifiedBandRate',ratio(nativeCertified,proposals), ...
            'pendingPassRate',ratio(pending,proposals),'fullEvaluationRate',ratio(guided,pending), ...
            'uniqueGuidedUpdateRate',ratio(numel(unique(tightenedIDs)),guided), ...
            'logShrinkPerGuideFE',ratio(logShrink,guided),'netGapReduction',netShrink, ...
            'survivedP1',ratio(survived1,guided),'survivedP2',ratio(survived2,guided), ...
            'retainedP1After1',retention(1),'retainedP1After5',retention(2), ...
            'retainedP1After10',retention(3), ...
            'penalizedGapAUC',trapz(time,gaps)/duration, ...
            'pairCoverageAUC',trapz(time,covered)/duration, ...
            'IGDAUC',metricAUC(R.Trajectory,'IGD'),'HVAUC',metricAUC(R.Trajectory,'HV'), ...
            'finalIGD',R.Trajectory.IGD(end),'finalHV',R.Trajectory.HV(end), ...
            'trainingSeconds',E.trainingSeconds,'wallSeconds',R.Record.wallSeconds, ...
            'searchFE',E.fullFE,'CalObjRows',E.oracleCalls.CalObjRows, ...
            'CalConRows',E.oracleCalls.CalConRows);
    end
    if isempty(rows); Summary=table(); return; end
    Summary=struct2table(vertcat(rows{:}));
    writetable(Summary,fullfile(folder,'interval_summary.csv'));
end

function value=ratio(a,b)
    value=NaN;
    if b>0; value=a/b; end
end

function value=metricAUC(T,name)
    valid=isfinite(T.(name)) & isfinite(T.actualFE);
    value=NaN;
    if ~all(valid); return; end % Missing/infeasible phases must not improve AUC.
    x=T.actualFE; y=T.(name);
    if numel(x)>1 && x(end)>x(1); value=trapz(x,y)/(x(end)-x(1)); end
end
