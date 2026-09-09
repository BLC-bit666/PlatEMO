function prepare_reset_fixtures
% Exact six panels in the user's figure; copy cached state without evaluation.
warning('off','all'); maxNumCompThreads(1);
folder=fileparts(mfilename('fullpath')); root=fileparts(fileparts(folder));
previous=fullfile(root,'Data','PairGuideGPTValidation_20260908');
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support'));
addCBSPaths(root); addpath(folder);
out=fullfile(folder,'fixtures'); if ~isfolder(out); mkdir(out); end
arms={'original','stable_side','stable_no_side'};
sources={fullfile(root,'Data','PairGuideSinglePoint_20260907','full_run'), ...
    fullfile(previous,'stable_side'),fullfile(previous,'stable_no_side')};
for a=1:3
    for n=7:8
        stem=sprintf('%s_LIRCMOP%d_BC',arms{a},n); target=fullfile(out,[stem,'.mat']);
        if isfile(target); continue; end
        fprintf('RESET_FIXTURE_LOAD %s\n',stem);
        source=fullfile(sources{a},sprintf('LIRCMOP%d_BC_seed01_cgan.mat',n));
        R=load(source,'Audit'); E=R.Audit.evidence; Model=E.lastModel; W=E.W;
        trained=E.training(cellfun(@(t)t.trained,E.training)); T=trained{end};
        State=E.population{T.generation+1}; assert(State.observationFE==99800);
        Q=E.queries{end}; assert(Q.productionFE==99800 && Q.modelVersion==Model.version);
        RawData=Model.lastData;
        [present,idx]=ismember(RawData.id,State.archive.id); assert(all(present));
        RawData.yF=State.archive.yf(idx,:); RawData.yI=State.archive.yi(idx,:); RawData.W=W;
        assert(isequal(RawData.xF,State.archive.xf(idx,:)) && isequal(RawData.xI,State.archive.xi(idx,:)));
        ActualData=RawData;
        if isfield(Model,'lastTrainingData'); ActualData=Model.lastTrainingData; end
        ActualData.yF=RawData.yF; ActualData.yI=RawData.yI; ActualData.W=W;
        useSide=~strcmp(arms{a},'stable_no_side'); stable=~strcmp(arms{a},'original');
        Model.useSideCondition=useSide; Model.stableConditionSpan=stable;
        ctor=str2func(sprintf('LIRCMOP%d_BC',n)); P=ctor('N',100,'maxFE',100000);
        [X,Info]=PairBoundaryWGAN_RC('sample',Model,Q.conditions,P,struct('sampleSigma',0,'referenceScale',ActualData.referenceScale));
        assert(isequal(X,Q.rawDecs)); assert(P.FE==0);
        [QueryC,keep]=unique(Info.conditions,'rows','stable'); CachedRaw=X(keep,:);
        Plot=load(fullfile(sources{a},'analysis','figures',sprintf('LIRCMOP%d_BC',n),'run_01','plot_data.mat'),'States');
        S=Plot.States{end}; assert(S.productionFE==99800 && S.trainingFE==99800);
        assert(isequal(S.xf,RawData.yF) && isequal(S.xi,RawData.yI));
        CachedObjectives=S.native(keep,:);
        save(target,'Model','RawData','ActualData','QueryC','CachedRaw','CachedObjectives','W','State','source','useSide','stable','-v7.3');
        fprintf('RESET_FIXTURE_SAVED %s pairs=%d queryConditions=%d exactOriginal=true\n',stem,RawData.count,size(QueryC,1));
        clear R E Plot;
    end
end
fprintf('RESET_FIXTURES_COMPLETE\n');
end
