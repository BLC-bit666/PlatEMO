function audit_search_arm(arm)
% Independently check model frames, first-use identity, FE and raw consumption.
warning('off','all'); maxNumCompThreads(1);
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
folder=fullfile(root,'Data','PairGuideGPTValidation_20260908');
addpath(folder); proxyDir=fullfile(folder,'boundary_proxies'); if ~isfolder(proxyDir); mkdir(proxyDir); end
source=fullfile(folder,char(arm)); files=dir(fullfile(source,'LIRCMOP*_seed*_cgan.mat'));
assert(numel(files)==12);
rows=cell(12,1);
for k=1:12
    R=load(fullfile(files(k).folder,files(k).name),'Record','Audit'); E=R.Audit.evidence;
    assert(R.Record.finalFE==100000 && E.fullFE==100000 && R.Record.D==30);
    assert(E.oracleCalls.CalObjRows==200000 && E.oracleCalls.CalConRows==100000);
    assert(E.objectiveOnlyFE==0 && E.constraintOnlyFE==0);
    trained=E.training(cellfun(@(t)t.trained,E.training));
    assert(trained{1}.updates==1000 && all(cellfun(@(t)t.updates==20,trained(2:end))));
    assert(E.lastModel.iterG==sum(cellfun(@(t)t.updates,trained)));
    assert(E.lastModel.iterC==5*E.lastModel.iterG && ~isempty(E.lastModel.avgG) && ~isempty(E.lastModel.avgC));
    span=E.firstModel.conditionSpan;
    assert(all(cellfun(@(t)isequal(t.conditionScale.span,span),trained)));
    label=E.firstModel.useSideCondition;
    for j=1:numel(trained)
        T=trained{j}; P=E.population{T.generation+1};
        assert(isequal(T.conditionScale.minimum,P.referenceScale.minimum));
        if ~label; assert(all(T.actualCF(:,end)==0) && all(T.actualCI(:,end)==0)); end
    end
    first=load(fullfile(root,'Data','PairGuideSinglePoint_20260907','first_use', ...
        sprintf('%s_seed%02d_lrzero1000.mat',R.Record.problem,R.Record.seed)),'Q','Prefix');
    prefix=E.population{E.queries{1}.generation+1}; assert(isequaln(prefix,first.Prefix));
    initialRawEqual=isequal(E.queries{1}.rawDecs,first.Q.rawDecs);
    if label; assert(initialRawEqual); end
    for j=1:numel(E.queries)
        Q=E.queries{j}; U=E.generations{Q.generation+1}.use;
        assert(isequal(Q.pending.decs,Q.rawDecs(Q.pool.keepIdx,:)) && all(Q.pending.ids==0));
        assert(all(ismember(U.childDecs,Q.pending.decs,'rows')) && U.selected+U.fallback==U.requested);
        assert(isequal(Q.conditionScale.span,span));
        assert(isequal(Q.conditionScale.minimum,E.population{Q.generation+1}.referenceScale.minimum));
        if ~label; assert(all(Q.networkConditions(:,end)==0)); end
    end
    assert(all(cellfun(@(g)g.archive.eligibilityRejectedPairs==0 && g.archive.archiveFrontDepth==1,E.generations)));
    assert(all(cellfun(@(p)numel(p.archive.id)<=500,E.population)));
    selected=sum(cellfun(@(g)g.use.selected,E.generations));
    survived=sum(cellfun(@(g)nnz(g.use.survivedP1),E.generations));
    newI=sum(cellfun(@(g)g.archive.guidedTightenedInfeasible,E.generations));
    rows{k}=struct('problem',R.Record.problem,'seed',R.Record.seed,'arm',string(arm), ...
        'verified',true,'initialPrefixExact',true,'initialRawExact',initialRawEqual, ...
        'useSideCondition',label,'trainingEvents',numel(trained),'totalGUpdates',E.lastModel.iterG, ...
        'trainingSeconds',E.trainingSeconds,'guidedSelected',selected,'guidedSurvivedP1',survived, ...
        'guidedInfeasibleReplacements',newI,'queries',numel(E.queries));
    Proxy=boundary_proxy_rows(E,R.Record.problem,R.Record.seed,arm);
    writetable(Proxy,fullfile(proxyDir,sprintf('%s_seed%02d_%s.csv',R.Record.problem,R.Record.seed,arm)));
    fprintf('AUDITED %s seed=%d %s\n',R.Record.problem,R.Record.seed,arm);
    clear R E;
end
writetable(struct2table(vertcat(rows{:})),fullfile(folder,char(string(arm)+"_verification.csv")));
end
