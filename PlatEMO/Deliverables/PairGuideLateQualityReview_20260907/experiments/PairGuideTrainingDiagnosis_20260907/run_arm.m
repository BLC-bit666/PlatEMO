function run_arm(number,targetFE,arm)
% One-variable fixed-data comparison. Nothing is fed back to the search.
warning('off','all'); restoredefaultpath; maxNumCompThreads(1);
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support')); addCBSPaths(root);
out=fullfile(root,'Data','PairGuideTrainingDiagnosis_20260907'); addpath(out);
name=sprintf('LIRCMOP%d_BC',number); ctor=str2func(name); P=ctor(); arm=string(arm);
R=load(fullfile(out,sprintf('%s_FE%06d_model.mat',name,targetFE))); B=R.B;
D=B.Data; O=B.Options; M=B.BaseModel; updates=1000; startRNG=B.BaseRNG; priorUpdates=0;
if arm=="production20", updates=20;
elseif arm=="initial200", updates=200;
elseif arm=="noAdv1000", O.noAdv=true;
elseif arm=="uniformGap1000", O.uniformGap=true;
elseif arm=="lr1e3_1000", O.lrG=1e-3;
elseif arm=="noNoise1000", O.trainingSigma=0; O.sampleSigma=0;
elseif arm=="fresh32_1000", M=[];
elseif arm=="fresh128_1000", M=[]; O.generatorHidden=[128 128];
elseif arm=="fresh32uniform1000", M=[]; O.uniformGap=true;
elseif arm=="fresh128uniform1000", M=[]; O.generatorHidden=[128 128]; O.uniformGap=true;
elseif arm=="onehot32uniform1000"
 M=[]; O.uniformGap=true; code=eye(100); D.cF=[code(D.ref,:),ones(D.count,1)]; D.cI=[code(D.ref,:),zeros(D.count,1)];
elseif arm=="onehot32_1000"
 M=[]; code=eye(100); D.cF=[code(D.ref,:),ones(D.count,1)]; D.cI=[code(D.ref,:),zeros(D.count,1)];
elseif endsWith(arm,"5000") || endsWith(arm,"10000")
 if endsWith(arm,"10000"), baseArm=regexprep(arm,'10000$','5000'); priorUpdates=5000; updates=5000;
 else, baseArm=regexprep(arm,'5000$','1000'); priorUpdates=1000; updates=4000; end
 V=load(fullfile(out,sprintf('%s_FE%06d_%s.mat',name,targetFE,baseArm)));
 M=V.M; D=V.D; O=V.O; O.generation=M.lastTrainGeneration+1; O.retrainChange=0;
 if priorUpdates>1000 && isfield(V,'TrainingRNG'), startRNG=V.TrainingRNG;
 else, rng(20260908,'twister'); startRNG=rng; end
elseif arm~="warm1000", error('Unknown arm %s',arm);
end
O.initialEpoch=updates; O.retrainEpoch=updates;
rng(startRNG); timer=tic;
[M,T]=PairBoundaryWGAN_Diagnostic('trainifneeded',M,D,B.Gate,P,O);
seconds=toc(timer); TrainingRNG=rng; assert(T.trained && T.updates==updates);
if ismember(arm,["production20","initial200"]), assert(abs(T.postDiagnostics.allEndpointRMSE-B.Expected.postDiagnostics.allEndpointRMSE)<1e-9); end
rng(20260907,'twister'); repeat=64;
[native,Q]=PairBoundaryWGAN_RC('sample',M,repelem(D.cF,repeat,1),P,O);
trueF=repelem(D.xF,repeat,1); trueI=repelem(D.xI,repeat,1);
trueMid=(1-Q.alpha).*trueF+Q.alpha.*trueI;
gf=reshape(Q.generatedF,repeat,D.count,P.D); gi=reshape(Q.generatedI,repeat,D.count,P.D);
mf=reshape(mean(gf,1),D.count,P.D); mi=reshape(mean(gi,1),D.count,P.D);
bias=mean([mf-D.xF;mi-D.xI].^2,'all');
noise=mean(cat(2,gf-reshape(mf,1,D.count,P.D),gi-reshape(mi,1,D.count,P.D)).^2,'all');
PairGuideCost_RC('start',P);
yf=P.CalObj(trueF); yi=P.CalObj(trueI); ym=P.CalObj(trueMid);
ygf=P.CalObj(Q.generatedF); ygi=P.CalObj(Q.generatedI); yg=P.CalObj(native);
cf=P.CalCon(Q.generatedF); ci=P.CalCon(Q.generatedI); cg=P.CalCon(native);
Oracle=PairGuideCost_RC('snapshot'); PairGuideCost_RC('stop'); assert(P.FE==0);
Probe=PairBoundaryWGAN_Diagnostic('lossprobe',M,D);
[~,order]=sort(D.w(:,1));
targetJump=median(vecnorm(diff([D.xF(order,:),D.xI(order,:)]),2,2)/sqrt(2*P.D));
outputJump=median(vecnorm(diff([mf(order,:),mi(order,:)]),2,2)/sqrt(2*P.D));
row=struct('problem',string(name),'FE',targetFE,'arm',arm,'pairs',D.count, ...
 'updates',priorUpdates+updates,'seconds',seconds,'parameters',Probe.generatorParameters,'RMSE',T.postDiagnostics.allEndpointRMSE, ...
 'biasRMSE',sqrt(bias),'noiseRMSE',sqrt(noise),'biasMSEShare',bias/(bias+noise), ...
 'relativeError',T.postDiagnostics.relativeEndpointError,'targetJump',targetJump,'outputJump',outputJump, ...
 'endpointObjectiveRMSE',sqrt(mean([ygf-yf;ygi-yi].^2,'all')), ...
 'midpointObjectiveRMSE',sqrt(mean((yg-ym).^2,'all')), ...
 'feasibleLabelAccuracy',mean(all(cf<=0,2)),'infeasibleLabelAccuracy',mean(any(ci>0,2)), ...
 'midpointFeasibleRate',mean(all(cg<=0,2)), ...
 'adversarialGradientNorm',Probe.adversarialGradientNorm,'geometryGradientNorm',Probe.geometryGradientNorm, ...
 'advGeometryCosine',Probe.advGeometryCosine);
base=fullfile(out,sprintf('%s_FE%06d_%s',name,targetFE,arm));
writetable(struct2table(row),[base,'.csv']);
save([base,'.mat'],'row','M','T','D','O','Q','mf','mi','yf','yi','ym','ygf','ygi','yg','Oracle','Probe','TrainingRNG','-v7.3');
fprintf('ARM_COMPLETE %s\n',jsonencode(row));
end
