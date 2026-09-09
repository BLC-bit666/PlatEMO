function step_size_probe
warning('off','all'); restoredefaultpath; maxNumCompThreads(1);
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support')); addCBSPaths(root);
out='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Data/PairGuideDominanceDrift_20260907'; rows={};
for number=[7 9]
 name=sprintf('LIRCMOP%d_BC',number); ctor=str2func(name); P=ctor();
 B=load(fullfile(out,[name,'_FE030000_checkpoint.mat']));
 rng(B.BaseRNG); O=B.Options; O.retrainEpoch=1000; O.lrG=1e-3;
 clock=tic; [M,T]=PairBoundaryWGAN_RC('trainifneeded',B.BaseModel,B.Data,B.Gate,P,O); seconds=toc(clock);
 rng(20260907,'twister'); [~,Q]=PairBoundaryWGAN_RC('sample',M,repelem(B.Data.cF,8,1),P,O);
 yf=P.CalObj(P.lower+Q.generatedF.*(P.upper-P.lower)); cf=P.CalCon(P.lower+Q.generatedF.*(P.upper-P.lower));
 yi=P.CalObj(P.lower+Q.generatedI.*(P.upper-P.lower)); ci=P.CalCon(P.lower+Q.generatedI.*(P.upper-P.lower));
 tf=P.CalObj(P.lower+repelem(B.Data.xF,8,1).*(P.upper-P.lower)); ti=P.CalObj(P.lower+repelem(B.Data.xI,8,1).*(P.upper-P.lower));
 row=struct('problem',string(name),'FE',B.Pop.observationFE,'generatorLR',1e-3,'updates',T.updates, ...
  'RMSE',T.postDiagnostics.allEndpointRMSE,'relativeError',T.postDiagnostics.relativeEndpointError, ...
  'thickness',T.postDiagnostics.sameConditionThickness,'objectiveRMSE',sqrt(mean([yf-tf;yi-ti].^2,'all')), ...
  'feasibleLabelAccuracy',mean(all(cf<=0,2)),'infeasibleLabelAccuracy',mean(any(ci>0,2)),'seconds',seconds);
 rows{end+1}=row; fprintf('STEP_SIZE %s\n',jsonencode(row)); %#ok<AGROW>
 writetable(struct2table(vertcat(rows{:})),fullfile(out,'step_size.csv'));
 assert(P.FE==0);
end
fprintf('STEP_SIZE_COMPLETE\n');
end
