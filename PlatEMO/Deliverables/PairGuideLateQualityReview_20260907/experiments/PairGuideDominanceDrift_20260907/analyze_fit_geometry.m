function analyze_fit_geometry
warning('off','all'); restoredefaultpath; maxNumCompThreads(1);
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support')); addCBSPaths(root);
out='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Data/PairGuideDominanceDrift_20260907'; rows={};
for number=[7 9]
 name=sprintf('LIRCMOP%d_BC',number); ctor=str2func(name); P=ctor();
 B=load(fullfile(out,[name,'_FE030000_checkpoint.mat'])); D=B.Data;
 rng(B.BaseRNG); [M,S]=PairBoundaryWGAN_RC('trainifneeded',B.BaseModel,D,B.Gate,P,B.Options);
 assert(abs(S.postDiagnostics.allEndpointRMSE-B.Expected.postDiagnostics.allEndpointRMSE)<1e-9);
 rng(20260907,'twister'); repeat=128; C=repelem(D.cF,repeat,1);
 [~,Q]=PairBoundaryWGAN_RC('sample',M,C,P,B.Options);
 gf=reshape(Q.generatedF,repeat,D.count,P.D); gi=reshape(Q.generatedI,repeat,D.count,P.D);
 mf=reshape(mean(gf,1),D.count,P.D); mi=reshape(mean(gi,1),D.count,P.D);
 nf=gf-reshape(mf,1,D.count,P.D); ni=gi-reshape(mi,1,D.count,P.D);
 bias=sqrt(mean([mf-D.xF;mi-D.xI].^2,'all'));
 noise=sqrt(mean(cat(2,nf,ni).^2,'all'));
 coord=mean([mf-D.xF;mi-D.xI].^2,1); [high,which]=sort(coord,'descend');
 targetStd=sqrt(mean([D.xF-mean(D.xF,1);D.xI-mean(D.xI,1)].^2,'all'));
 outputStd=sqrt(mean([mf-mean(mf,1);mi-mean(mi,1)].^2,'all'));
 [~,order]=sort(D.w(:,1));
 targetJump=vecnorm(diff([D.xF(order,:),D.xI(order,:)],1,1),2,2)/sqrt(2*P.D);
 outputJump=vecnorm(diff([mf(order,:),mi(order,:)],1,1),2,2)/sqrt(2*P.D);
 row=struct('problem',string(name),'FE',B.Pop.observationFE,'pairs',D.count, ...
  'biasRMSE',bias,'noiseRMSE',noise,'biasMSEShare',bias^2/(bias^2+noise^2), ...
  'targetConditionStd',targetStd,'generatedConditionStd',outputStd, ...
  'adjacentTargetJumpMedian',median(targetJump),'adjacentGeneratedJumpMedian',median(outputJump), ...
  'largestErrorCoordinate',which(1),'largestCoordinateShare',high(1)/sum(coord));
 rows{end+1}=row; fprintf('FIT_GEOMETRY %s\n',jsonencode(row)); %#ok<AGROW>
 inputDistance=vecnorm(diff(D.w(order,:),1,1),2,2);
 Details=table(D.ref(order(1:end-1)),D.ref(order(2:end)),inputDistance,targetJump,outputJump, ...
  'VariableNames',{'refA','refB','conditionDistance','targetJumpRMS','generatedJumpRMS'});
 writetable(Details,fullfile(out,[name,'_condition_jumps.csv']));
 save(fullfile(out,[name,'_geometry.mat']),'D','mf','mi','coord','Q','row','-v7.3');
 F=figure('Visible','off','Color','w','Position',[50 50 1500 780]); L=tiledlayout(2,2,'Padding','compact');
 nexttile; plot(D.w(order,1),D.xF(order,1),'o-'); hold on; plot(D.w(order,1),mf(order,1),'s-');
 xlabel('Reference weight w_1'); ylabel('Normalized x_1'); title('Feasible endpoint: condition response'); legend('Actual target','Mean generated'); grid on;
 nexttile; plot(D.w(order,1),D.xI(order,1),'o-'); hold on; plot(D.w(order,1),mi(order,1),'s-');
 xlabel('Reference weight w_1'); ylabel('Normalized x_1'); title('Infeasible endpoint: condition response'); legend('Actual target','Mean generated'); grid on;
 targetF=P.CalObj(P.lower+D.xF.*(P.upper-P.lower)); targetI=P.CalObj(P.lower+D.xI.*(P.upper-P.lower));
 generatedF=P.CalObj(P.lower+Q.generatedF.*(P.upper-P.lower)); generatedI=P.CalObj(P.lower+Q.generatedI.*(P.upper-P.lower));
 nexttile; scatter(targetF(:,1),targetF(:,2),25,[0 .5 .15],'o'); hold on; scatter(targetI(:,1),targetI(:,2),25,[.8 .15 .1],'s');
 scatter(generatedF(:,1),generatedF(:,2),5,[.15 .4 .85],'.'); scatter(generatedI(:,1),generatedI(:,2),5,[.65 .2 .7],'.');
 xlabel('f_1'); ylabel('f_2'); title('Exact fixed training set and generated endpoints'); legend('Actual F','Actual I','Generated F','Generated I'); grid on;
 nexttile; bar(sqrt(coord)); xlabel('Decision variable'); ylabel('Mean-output bias RMSE'); title(sprintf('Bias contributes %.1f%% of probe MSE',100*row.biasMSEShare)); grid on;
 title(L,sprintf('%s | FE %d | %d fixed training pairs | production 20-update retrain',name,B.Pop.observationFE,D.count),'Interpreter','none');
 exportgraphics(F,fullfile(out,[name,'_fit_geometry.png']),'Resolution',130); close(F);
 assert(P.FE==0);
end
writetable(struct2table(vertcat(rows{:})),fullfile(out,'fit_geometry.csv'));
fprintf('FIT_GEOMETRY_COMPLETE\n');
end
