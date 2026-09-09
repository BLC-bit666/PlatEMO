function test_CBS_pair_guide_condition_span
% Stable model frame must not change first training, gate data, or side mass.
root=fileparts(which('platemo')); addCBSPaths(root);
P=LIRCMOP7_BC('N',100,'maxFE',100000);
source=fullfile(root,'Data','PairGuideSinglePoint_20260907','full_run', ...
    'analysis','figures','LIRCMOP7_BC','run_01','plot_data.mat');
R=load(source,'States'); S=R.States{end};
[W,~]=UniformPoint(100,2);
O=struct('initialEpoch',2,'retrainEpoch',2,'nCritic',1,'generation',1, ...
    'referenceScale',S.trainingState.referenceScale,'useSideCondition',true);
[Data,Gate]=PairBoundaryArchive_RC('trainingdata',S.trainingState.archive,W,P,O);
PairGuideCost_RC('start',P); cleanup=onCleanup(@()PairGuideCost_RC('stop'));
rng(20260908); [Original,A]=PairBoundaryWGAN_RC('trainifneeded',[],Data,Gate,P,O);
O.stableConditionSpan=true;
rng(20260908); [Stable,B]=PairBoundaryWGAN_RC('trainifneeded',[],Data,Gate,P,O);
assert(isequaln(Original.netG.Learnables,Stable.netG.Learnables));
assert(isequaln(Original.netC.Learnables,Stable.netC.Learnables));
assert(isequal(A.actualCF,B.actualCF) && isequal(A.actualCI,B.actualCI));
Moved=Data; Moved.referenceScale.span=Data.referenceScale.span.*[4 .25];
rF=AssignReferenceVectors_CBS(Moved.yF,W,Moved.referenceScale);
rI=AssignReferenceVectors_CBS(Moved.yI,W,Moved.referenceScale);
Moved.cF=[W(rF,:),ones(Data.count,1)]; Moved.cI=[W(rI,:),zeros(Data.count,1)];
O.generation=11; O.referenceScale=Moved.referenceScale;
[Next,C]=PairBoundaryWGAN_RC('trainifneeded',Stable,Moved,Gate,P,O);
assert(C.trained && Next.iterG==4 && isequal(Next.lastData,Moved));
assert(isequal(C.actualCF,B.actualCF) && isequal(C.actualCI,B.actualCI));
assert(isequal(C.conditionScale.span,Data.referenceScale.span));
assert(isequal(C.conditionScale.minimum,Moved.referenceScale.minimum));
O.useSideCondition=false; O.generation=21;
[NoSide,D]=PairBoundaryWGAN_RC('trainifneeded',Next,Moved,Gate,P,O);
assert(D.trainingSamples==C.trainingSamples && D.endpointVisits==C.endpointVisits);
Q=[W,zeros(100,1);W,ones(100,1)]; O.sampleSigma=0;
[X,Info]=PairBoundaryWGAN_RC('sample',NoSide,Q,P,O);
assert(isequal(X(1:100,:),X(101:200,:)) && all(Info.conditions(:,end)==0));
assert(~Info.useSideCondition && isequal(Info.conditionScale.span,Data.referenceScale.span));
Cost=PairGuideCost_RC('snapshot');
assert(Cost.CalObjRows==0 && Cost.CalConRows==0 && P.FE==0);
fprintf('CONDITION_SPAN_TEST_PASS first identity / unchanged gates / fixed span / side ablation / zero oracle\n');
end
