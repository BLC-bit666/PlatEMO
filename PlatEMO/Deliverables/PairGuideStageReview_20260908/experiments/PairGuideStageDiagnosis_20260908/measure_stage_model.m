function [R,S]=measure_stage_model(M,F)
% Full evaluation on an isolated problem; never feeds search or training.
ctor=str2func(F.Meta.problem); P=ctor('N',100,'maxFE',100000);
saved=rng; cleanup=onCleanup(@()rng(saved)); rng(13579,'twister');
[X,Info]=PairBoundaryWGAN_RC('sample',M,F.QueryC,P,struct('sampleSigma',0,'referenceScale',F.Data.referenceScale));
PairGuideCost_RC('start',P); accounting=onCleanup(@()PairGuideCost_RC('stop'));
Pop=P.Evaluation(X); Y=Pop.objs; Cons=Pop.cons; cost=PairGuideCost_RC('snapshot');
assert(P.FE==size(X,1) && cost.CalObjRows==2*P.FE && cost.CalConRows==P.FE);
D=F.Data; realX=[D.xF;D.xI]; realY=[D.yF;D.yI]; C=[D.cF;D.cI];
[~,keep]=unique([realX,C(:,end)],'rows','stable'); realX=realX(keep,:);realY=realY(keep,:);C=C(keep,:);
[codes,~,group]=unique(C,'rows'); counts=accumarray(group,1);
[seen,at]=ismember(Info.conditions,codes,'rows');
single=false(size(seen)); single(seen)=counts(at(seen))==1;
multi=seen & ~single;
A=F.TrainState.archive;
arF=AssignReferenceVectors_CBS(A.yf,F.W,D.referenceScale); arI=AssignReferenceVectors_CBS(A.yi,F.W,D.referenceScale);
archiveC=[F.W(arF,:),ones(numel(arF),1);F.W(arI,:),zeros(numel(arI),1)];
archiveSeen=ismember(Info.conditions,archiveC,'rows');
excluded=~seen & archiveSeen; unsupported=~seen & ~archiveSeen;
assert(all(double(single)+double(multi)+double(excluded)+double(unsupported)==1));
distances=pdist2(Y,realY); nearest=min(distances,[],2); reverse=min(distances,[],1);
conditional=nan(size(seen));
for k=find(seen)'; conditional(k)=min(distances(k,group==at(k))); end
[~,~,scaled]=AssignReferenceVectors_CBS(Y,F.W,D.referenceScale);
directions=scaled./max(vecnorm(scaled,2,2),eps); target=Info.conditions(:,1:2);target=target./vecnorm(target,2,2);
angle=acosd(max(-1,min(1,sum(directions.*target,2))));
[Audit,~]=PairBoundaryWGAN_StageProbe('gradientaudit',M,D);
R=struct('exactDistance',Audit.distance,'weightedDecisionMSE',Audit.weightedDecisionMSE, ...
    'conditionalVariance',Audit.conditionalVariance,'varianceFraction',Audit.varianceFraction, ...
    'gradientCosine',Audit.gradientCosine,'objectiveGap',mean(nearest),'reverseCoverageGap',mean(reverse), ...
    'seenObjectiveGap',mean(nearest(seen)),'conditionalSeenGap',mean(conditional(seen)), ...
    'singletonObjectiveGap',mean(nearest(single)),'multipleObjectiveGap',mean(nearest(multi)), ...
    'archiveExcludedObjectiveGap',mean(nearest(excluded)),'archiveUnseenObjectiveGap',mean(nearest(unsupported)), ...
    'directionError',mean(angle),'seenDirectionError',mean(angle(seen)), ...
    'singletonConditions',nnz(single),'multipleConditions',nnz(multi), ...
    'archiveExcludedConditions',nnz(excluded),'archiveUnseenConditions',nnz(unsupported), ...
    'queryConditions',size(X,1),'generatedFeasibleRate',mean(all(Cons<=0,2)), ...
    'offlineFE',P.FE,'offlineObjRows',cost.CalObjRows,'offlineConRows',cost.CalConRows);
S=struct('X',X,'Y',Y,'constraints',Cons,'conditions',Info.conditions, ...
    'seen',seen,'singleton',single,'multiple',multi,'archiveExcluded',excluded,'archiveUnseen',unsupported, ...
    'nearestObjectiveDistance',nearest,'conditionalObjectiveDistance',conditional,'angle',angle);
end
