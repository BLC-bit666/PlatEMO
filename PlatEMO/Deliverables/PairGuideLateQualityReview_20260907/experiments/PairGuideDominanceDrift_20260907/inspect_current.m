function inspect_current
warning('off','all'); restoredefaultpath; maxNumCompThreads(1);
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support')); addCBSPaths(root);
out='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Data/PairGuideDominanceDrift_20260907';
folder=fullfile(root,'Data','PairGuideCCMOFront1_5to9_20260907');
trainingRows={}; archiveRows={};
for number=5:9
 name=sprintf('LIRCMOP%d_BC',number); ctor=str2func(name); P=ctor();
 R=load(fullfile(folder,[name,'_seed01_cgan.mat']),'Audit'); E=R.Audit.evidence;
 trained=find(cellfun(@(s)s.trained,E.training)); previous=[];
 for k=reshape(trained,1,[])
  T=E.training{k}; Pop=E.population{T.generation+1};
  [D,~]=PairBoundaryArchive_RC('trainingdata',Pop.archive,E.W,P,struct());
  assert(D.count==T.trainingPairs);
  gap2=max(sum(D.delta.^2,2),1e-6*P.D); weight=1./gap2;
  h=sort(weight,'descend'); top=max(1,ceil(.1*D.count));
  [~,order]=sort(D.w(:,1)); x=[D.xF,D.xI];
  jump=vecnorm(diff(x(order,:),1,1),2,2)/sqrt(2*P.D);
  movement=NaN; common=0;
  if ~isempty(previous)
   [refs,now,old]=intersect(D.ref,previous.ref); common=numel(refs);
   if common>0
    movement=sqrt(mean([D.xF(now,:)-previous.xF(old,:);D.xI(now,:)-previous.xI(old,:)].^2,'all'));
   end
  end
  row=struct('problem',string(name),'FE',Pop.observationFE,'generation',T.generation, ...
   'pairs',D.count,'updates',T.updates,'epochs',T.epochs,'contentChange',T.contentChange, ...
   'commonRefs',common,'targetMovementRMSE',movement,'adjacentTargetJumpMedian',median(jump), ...
   'weightMaxMin',max(weight)/min(weight),'top10WeightShare',sum(h(1:top))/sum(h), ...
   'preRMSE',T.preDiagnostics.allEndpointRMSE,'postRMSE',T.postDiagnostics.allEndpointRMSE, ...
   'preRelativeError',T.preDiagnostics.relativeEndpointError,'postRelativeError',T.postDiagnostics.relativeEndpointError, ...
   'postThickness',T.postDiagnostics.sameConditionThickness,'postRelativeThickness',T.postDiagnostics.relativeThickness);
  trainingRows{end+1}=row; previous=D; %#ok<AGROW>
 end
 first=E.training{trained(1)}.generation;
 stages=unique([first,arrayfun(@(fe)find(cellfun(@(s)s.observationFE,E.population)<=fe,1,'last')-1,[10000 30000 50000 70000 100000])]);
 for g=stages
  before=E.population{g}; now=E.population{g+1}; A=now.archive;
  O1=E.evaluations{2*g}; O2=E.evaluations{2*g+1};
  X=[before.p1Decs;before.p2Decs;O1.decisions;O2.decisions];
  Y=[before.p1Objs;before.p2Objs;O1.objectives;O2.objectives];
  C=[before.p1Cons;before.p2Cons;O1.constraints;O2.constraints];
  f=all(C<=0,2); fx=[X(f,:);before.archive.xf]; fy=[Y(f,:);before.archive.yf];
  ix=[X(~f,:);before.archive.xi]; iy=[Y(~f,:);before.archive.yi];
  [fx,k]=unique(fx,'rows','stable'); fy=fy(k,:);
  [ix,k]=unique(ix,'rows','stable'); iy=iy(k,:);
  ranks=NDSort(fy,inf); F=fy(ranks==1,:); XF=fx(ranks==1,:);
  idom=dominated(F,iy); own=all(A.yf<=A.yi,2)&any(A.yf<A.yi,2);
  globaldom=dominated(F,A.yi);
  [~,Scale]=AssignReferenceVectors_CBS([Y;before.archive.yf;before.archive.yi],E.W);
  refs=AssignReferenceVectors_CBS(F,E.W,Scale); ir=AssignReferenceVectors_CBS(iy(~idom,:),E.W,Scale);
  keptI=ix(~idom,:); partners=zeros(size(XF,1),1); newrefs=[];
  cosine=(E.W*E.W')./(vecnorm(E.W,2,2)*vecnorm(E.W,2,2)');
  for frow=1:size(XF,1)
   distance=1-cosine(refs(frow),:); distance(refs(frow))=-Inf;
   [~,n]=sort(distance,'ascend'); n=n(1:min(5,numel(n)));
   local=find(ismember(ir,n));
   gaps=vecnorm((keptI(local,:)-XF(frow,:))./(P.upper-P.lower),2,2);
   valid=gaps>1e-12; local=local(valid); gaps=gaps(valid);
   if ~isempty(local), [~,j]=min(gaps); partners(frow)=local(j); newrefs(end+1)=refs(frow); end %#ok<AGROW>
  end
  row=struct('problem',string(name),'FE',now.observationFE,'archivePairs',numel(A.id), ...
   'activePairs',nnz(A.active),'ownDominatedActive',nnz(A.active&own), ...
   'globalDominatedActive',nnz(A.active&globaldom),'feasiblePool',size(fx,1), ...
   'frontOne',nnz(ranks==1),'infeasiblePool',size(ix,1),'infeasibleSurviving',nnz(~idom), ...
   'freshPairableFeasible',nnz(partners),'freshActiveRefs',numel(unique(newrefs)), ...
   'reusedInfeasibleEndpoints',nnz(partners)-numel(unique(partners(partners>0))));
  archiveRows{end+1}=row; %#ok<AGROW>
 end
 fprintf('INSPECT_COMPLETE %s training=%d\n',name,numel(trained));
end
writetable(struct2table(vertcat(trainingRows{:})),fullfile(out,'training.csv'));
writetable(struct2table(vertcat(archiveRows{:})),fullfile(out,'archive.csv'));
fprintf('INSPECTION_COMPLETE\n');
end

function result=dominated(F,Y)
result=false(size(Y,1),1);
for k=1:size(Y,1), result(k)=any(all(F<=Y(k,:),2)&any(F<Y(k,:),2)); end
end
