function extract_case(number)
% Extract fixed datasets and actual query evidence, with separately billed oracles.
warning('off','all'); restoredefaultpath; maxNumCompThreads(1);
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support')); addCBSPaths(root);
out=fullfile(root,'Data','PairGuideTrainingDiagnosis_20260907'); addpath(out);
folder=fullfile(root,'Data','PairGuideFilteredArchive_5to8_R3_20260907');
name=sprintf('LIRCMOP%d_BC',number); ctor=str2func(name); P=ctor();
R=load(fullfile(folder,[name,'_seed01_cgan.mat']),'Audit'); E=R.Audit.evidence;
Events=cell(size(E.training)); rows={}; previous=[];
for k=1:numel(E.training)
 T=E.training{k}; Pop=E.population{T.generation+1};
 [D,Gate]=PairBoundaryArchive_RC('trainingdata',Pop.archive,E.W,P,struct());
 Events{k}=struct('T',T,'Data',D,'Gate',Gate,'FE',Pop.observationFE);
 if ~T.trained, continue; end
 assert(D.count==T.trainingPairs);
 weights=1./max(sum(D.delta.^2,2),1e-6*P.D); w=sort(weights,'descend');
 movement=NaN; common=0;
 if ~isempty(previous)
  [~,now,old]=intersect(D.ref,previous.ref); common=numel(now);
  if common, movement=sqrt(mean([D.xF(now,:)-previous.xF(old,:);D.xI(now,:)-previous.xI(old,:)].^2,'all')); end
 end
 [~,order]=sort(D.w(:,1)); jump=vecnorm(diff([D.xF(order,:),D.xI(order,:)]),2,2)/sqrt(2*P.D);
 row=struct('problem',string(name),'FE',Pop.observationFE,'pairs',D.count,'updates',T.updates, ...
  'epochs',T.epochs,'movementRMSE',movement,'commonRefs',common,'adjacentTargetJump',median(jump), ...
  'top10WeightShare',sum(w(1:ceil(.1*D.count)))/sum(w),'effectiveWeightedPairs',sum(w)^2/sum(w.^2), ...
  'preRMSE',T.preDiagnostics.allEndpointRMSE,'postRMSE',T.postDiagnostics.allEndpointRMSE, ...
  'relativeError',T.postDiagnostics.relativeEndpointError,'noiseRMSE',T.postDiagnostics.sameConditionThickness);
 rows{end+1}=row; previous=D; %#ok<AGROW>
end
Training=struct2table(vertcat(rows{:})); W=E.W; seed=E.seed;
save(fullfile(out,[name,'_events.mat']),'Events','W','seed','Training','-v7.3');
writetable(Training,fullfile(out,[name,'_training.csv']));
fprintf('EXTRACTED %s events=%d successful=%d\n',name,numel(Events),height(Training));
cache=load(fullfile(folder,'analysis','figures',name,'run_01','plot_data.mat'),'States');
PairGuideCost_RC('start',P); rows={}; Nodes=cell(size(cache.States));
for k=1:numel(cache.States)
 S=cache.States{k};
 if S.missingGeneration, Nodes{k}=struct('FE',S.observationFE,'missing',true); continue; end
 q=find(cellfun(@(q)q.productionFE==S.productionFE,E.queries),1); Q=E.queries{q};
 t=find(cellfun(@(e)e.T.trained && e.FE==S.trainingFE,Events),1); D=Events{t}.Data;
 [known,idx]=ismember(Q.pool.rawRefs,D.ref); assert(any(known)); idx=idx(known);
 trueF=D.xF(idx,:); trueI=D.xI(idx,:); alpha=Q.sample.alpha(known);
 genF=Q.sample.generatedF(known,:); genI=Q.sample.generatedI(known,:);
 mid=(1-alpha).*trueF+alpha.*trueI; gen=Q.rawDecs(known,:);
 yf=P.CalObj(trueF); yi=P.CalObj(trueI); ym=P.CalObj(mid);
 ygf=P.CalObj(genF); ygi=P.CalObj(genI); yg=P.CalObj(gen);
 cf=P.CalCon(genF); ci=P.CalCon(genI); cm=P.CalCon(mid); cg=P.CalCon(gen);
 [~,near]=sort(pdist2(D.w,D.w),2); near=near(:,1:min(5,D.count));
 smoothF=zeros(size(D.xF)); smoothI=smoothF;
 for j=1:D.count, smoothF(j,:)=mean(D.xF(near(j,:),:),1); smoothI(j,:)=mean(D.xI(near(j,:),:),1); end
 yt=P.CalObj([D.xF;D.xI]); ys=P.CalObj([smoothF;smoothI]);
 row=struct('problem',string(name),'FE',S.observationFE,'trainingFE',S.trainingFE, ...
  'modelAgeFE',S.productionFE-S.trainingFE,'pairs',D.count,'knownConditionRate',mean(known), ...
  'endpointDecisionRMSE',sqrt(mean([genF-trueF;genI-trueI].^2,'all')), ...
  'endpointObjectiveRMSE',sqrt(mean([ygf-yf;ygi-yi].^2,'all')), ...
  'midpointObjectiveRMSE',sqrt(mean((yg-ym).^2,'all')), ...
  'exactInterpolationNonlinearity',sqrt(mean((ym-((1-alpha).*yf+alpha.*yi)).^2,'all')), ...
  'generatedFLabelAccuracy',mean(all(cf<=0,2)),'generatedILabelAccuracy',mean(any(ci>0,2)), ...
  'trueMidFeasibleRate',mean(all(cm<=0,2)),'generatedMidFeasibleRate',mean(all(cg<=0,2)), ...
  'neighborAverageDecisionRMSE',sqrt(mean([smoothF-D.xF;smoothI-D.xI].^2,'all')), ...
  'neighborAverageObjectiveRMSE',sqrt(mean((ys-yt).^2,'all')));
 rows{end+1}=row; %#ok<AGROW>
 Nodes{k}=struct('FE',S.observationFE,'missing',false,'Data',D,'query',Q, ...
  'trueF',trueF,'trueI',trueI,'genF',genF,'genI',genI,'trueMid',mid,'yTrueF',yf,'yTrueI',yi, ...
  'yGenF',ygf,'yGenI',ygi,'yTrueMid',ym,'yGenMid',yg,'smoothF',smoothF,'smoothI',smoothI, ...
  'ySmooth',ys,'yTargets',yt,'row',row);
end
Geometry=struct2table(vertcat(rows{:})); Oracle=PairGuideCost_RC('snapshot'); PairGuideCost_RC('stop');
assert(P.FE==0); writetable(Geometry,fullfile(out,[name,'_geometry.csv']));
save(fullfile(out,[name,'_nodes.mat']),'Nodes','Geometry','Oracle','-v7.3');
fprintf('GEOMETRY_COMPLETE %s nodes=%d oracle=%s\n',name,height(Geometry),jsonencode(Oracle));
end
