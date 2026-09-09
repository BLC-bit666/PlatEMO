function replay_case(number)
warning('off','all'); restoredefaultpath; maxNumCompThreads(1);
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support')); addCBSPaths(root);
out=fullfile(root,'Data','PairGuideTrainingDiagnosis_20260907'); addpath(out);
name=sprintf('LIRCMOP%d_BC',number); ctor=str2func(name); P=ctor();
R=load(fullfile(out,[name,'_events.mat'])); Events=R.Events;
good=find(cellfun(@(e)e.T.trained,Events));
near30=good(find(cellfun(@(e)e.FE<=30000,Events(good)),1,'last'));
targets=unique([good(1),near30,good(end)]);
stream=RandStream('mt19937ar','Seed',mod(double(R.seed)+104729,2^32-1));
Model=[]; verified=0;
for k=1:good(end)
 E=Events{k}; Data=E.Data; Gate=E.Gate; Options=struct('generation',E.T.generation);
 RandStream.setGlobalStream(stream);
 if ismember(k,targets), BaseModel=Model; BaseRNG=rng; end
 [Model,T]=PairBoundaryWGAN_RC('trainifneeded',Model,Data,Gate,P,Options);
 assert(T.trained==E.T.trained);
 if T.trained
  assert(abs(T.postDiagnostics.allEndpointRMSE-E.T.postDiagnostics.allEndpointRMSE)<1e-9, ...
   'Replay mismatch at %d',E.FE);
  verified=verified+1;
  if mod(verified,50)==0, fprintf('REPLAY_EXACT %s events=%d FE=%d\n',name,verified,E.FE); end
 end
 if ismember(k,targets)
  B=struct('BaseModel',BaseModel,'BaseRNG',BaseRNG,'Model',Model,'Data',Data,'Gate',Gate, ...
   'Options',Options,'Expected',E.T,'FE',E.FE,'verified',verified);
  save(fullfile(out,sprintf('%s_FE%06d_model.mat',name,E.FE)),'B','-v7.3');
  fprintf('MODEL_SAVED %s FE=%d pairs=%d RMSE=%.9f\n',name,E.FE,Data.count,T.postDiagnostics.allEndpointRMSE);
 end
end
fprintf('REPLAY_COMPLETE %s successful=%d\n',name,verified);
end
