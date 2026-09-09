function inspect_frozen_geometry
% Algebraic condition audit and an offline conditional-centroid diagnostic.
% Centroids are not proposed as a generator or used by search/training.
warning('off','all'); maxNumCompThreads(1); folder=fileparts(mfilename('fullpath'));
root=fileparts(fileparts(folder)); addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support')); addCBSPaths(root); addpath(folder);
out=fullfile(folder,'geometry'); if ~isfolder(out); mkdir(out); end
rows=cell(6,1); k=0;
for arm={'original','stable_side','stable_no_side'}
    for n=7:8
        stem=sprintf('%s_LIRCMOP%d_BC',arm{1},n); F=load(fullfile(folder,'fixtures',[stem,'.mat']));
        Data=F.ActualData; X=[Data.xF;Data.xI]; Y=[Data.yF;Data.yI]; C=[Data.cF;Data.cI];
        [~,keep]=unique([X,C(:,end)],'rows','stable'); X=X(keep,:); Y=Y(keep,:); C=C(keep,:);
        [~,~,trueGroups]=unique(C,'rows'); weights=zeros(size(X,1),1);
        for side=0:1
            groups=unique(trueGroups(C(:,end)==side));
            for g=groups'; ix=trueGroups==g; weights(ix)=.5/numel(groups)/nnz(ix); end
        end
        if ~F.useSide; C(:,end)=0; end
        [codes,~,groups]=unique(C,'rows'); means=zeros(size(codes,1),size(X,2)); meanY=zeros(size(codes,1),2);
        for g=1:size(codes,1)
            ix=groups==g; w=weights(ix)/sum(weights(ix)); means(g,:)=sum(X(ix,:).*w,1); meanY(g,:)=sum(Y(ix,:).*w,1);
        end
        ctor=str2func(sprintf('LIRCMOP%d_BC',n)); P=ctor('N',100,'maxFE',100000);
        PairGuideCost_RC('start',P); accounting=onCleanup(@()PairGuideCost_RC('stop'));
        Pop=P.Evaluation(means); meanObj=Pop.objs; meanCons=Pop.cons; cost=PairGuideCost_RC('snapshot');
        assert(P.FE==size(means,1) && cost.CalObjRows==2*P.FE && cost.CalConRows==P.FE);
        seen=ismember(F.QueryC,codes,'rows'); [found,at]=ismember(codes,F.QueryC,'rows'); assert(all(found));
        Gx=F.CachedRaw(at,:); Gy=F.CachedObjectives(at,:);
        if n==7; baseReal=1-sqrt(X(:,1)); baseG=1-sqrt(Gx(:,1));
        else; baseReal=1-X(:,1).^2; baseG=1-Gx(:,1).^2; end
        energyReal=[Y(:,1)-X(:,1)-.7057,Y(:,2)-baseReal-.7057];
        energyG=[Gy(:,1)-Gx(:,1)-.7057,Gy(:,2)-baseG-.7057];
        meanEnergy=zeros(size(energyG));
        for g=1:size(codes,1); ix=groups==g; w=weights(ix)/sum(weights(ix)); meanEnergy(g,:)=sum(energyReal(ix,:).*w,1); end
        gapG=min(pdist2(Gy,Y),[],2); gapMean=min(pdist2(meanObj,Y),[],2);
        k=k+1; rows{k}=struct('arm',string(arm{1}),'problem',sprintf('LIRCMOP%d_BC',n), ...
            'deduplicatedTrainingEndpoints',size(X,1),'observedConditions',size(codes,1), ...
            'queriedConditions',size(F.QueryC,1),'observedQueryConditions',nnz(seen), ...
            'multiEndpointConditions',nnz(accumarray(groups,1)>1), ...
            'warmSeenObjectiveGap',mean(gapG),'centroidObjectiveGap',mean(gapMean), ...
            'warmToConditionalMeanDecisionRMSE',sqrt(mean((Gx-means).^2,'all')), ...
            'meanSignedOddEnergyDifference',mean(energyG(:,1)-meanEnergy(:,1)), ...
            'meanSignedEvenEnergyDifference',mean(energyG(:,2)-meanEnergy(:,2)), ...
            'offlineFullFE',P.FE);
        save(fullfile(out,[stem,'.mat']),'codes','means','meanY','meanObj','meanCons','Gx','Gy','energyG','meanEnergy','gapG','gapMean');
        payload=struct('Data',Data,'queryConditions',F.QueryC,'cachedGeneratedX',F.CachedRaw,'cachedGeneratedObjectives',F.CachedObjectives, ...
            'codes',codes,'conditionalMeanX',means,'conditionalMeanObjectives',meanObj,'prototypeOfflineFE',P.FE);
        fid=fopen(fullfile(out,[stem,'.json']),'w','n','UTF-8'); assert(fid>=0); fprintf(fid,'%s\n',jsonencode(payload)); fclose(fid);
        clear accounting;
    end
end
writetable(struct2table(vertcat(rows{:})),fullfile(folder,'geometry_summary.csv'));
fprintf('FROZEN_GEOMETRY_COMPLETE 6 panels; centroid evaluations separately recorded\n');
end
