function replay_training(number,targetFE)
warning('off','all'); restoredefaultpath; maxNumCompThreads(1);
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support')); addCBSPaths(root);
out='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Data/PairGuideDominanceDrift_20260907'; addpath(out);
name=sprintf('LIRCMOP%d_BC',number); ctor=str2func(name); P=ctor();
file=fullfile(out,sprintf('%s_FE%06d_checkpoint.mat',name,targetFE));
if ~isfile(file)
 R=load(fullfile(root,'Data','PairGuideCCMOFront1_5to9_20260907',[name,'_seed01_cgan.mat']),'Audit'); E=R.Audit.evidence;
 good=find(cellfun(@(s)s.trained && E.population{s.generation+1}.observationFE<=targetFE,E.training));
 target=good(end); seed=mod(double(E.seed)+104729,2^32-1);
 stream=RandStream('mt19937ar','Seed',seed); RandStream.setGlobalStream(stream);
 Model=[]; verified=0;
 for k=1:target
  recorded=E.training{k}; g=recorded.generation; Pop=E.population{g+1};
  [Data,Gate]=PairBoundaryArchive_RC('trainingdata',Pop.archive,E.W,P,struct());
  Options=struct('generation',g);
  RandStream.setGlobalStream(stream);
  if k==target
   BaseModel=Model; BaseRNG=rng; Expected=recorded;
   save(file,'BaseModel','BaseRNG','Expected','Data','Gate','Options','Pop','verified','-v7.3');
   break;
  end
  [Model,S]=PairBoundaryWGAN_RC('trainifneeded',Model,Data,Gate,P,Options);
  assert(S.trained==recorded.trained);
  if S.trained
   assert(abs(S.postDiagnostics.allEndpointRMSE-recorded.postDiagnostics.allEndpointRMSE)<1e-9, ...
    'Training replay mismatch at generation %d: %.16g versus %.16g',g,S.postDiagnostics.allEndpointRMSE,recorded.postDiagnostics.allEndpointRMSE);
   verified=verified+1;
   if mod(verified,25)==0, fprintf('REPLAY %s verified=%d FE=%d\n',name,verified,Pop.observationFE); end
  end
 end
 fprintf('REPLAY_EXACT %s previousEvents=%d\n',name,verified);
end
B=load(file); rows={}; outputs={};
arms=["warm20","warm200","warm1000","resetAdam1000","fresh1000","noAdv1000","uniformGap1000"];
for arm=arms
 rng(B.BaseRNG); Model=B.BaseModel; O=B.Options; fn=@PairBoundaryWGAN_RC;
 if arm=="warm20", updates=20; elseif arm=="warm200", updates=200; else, updates=1000; end
 O.retrainEpoch=updates; O.initialEpoch=updates;
 if arm=="resetAdam1000"
  Model.avgG=[]; Model.avgSqG=[]; Model.avgC=[]; Model.avgSqC=[]; Model.iterG=0; Model.iterC=0;
 elseif arm=="fresh1000"
  Model=[];
 elseif arm=="noAdv1000"
  fn=@PairBoundaryWGAN_NoAdvProbe;
 elseif arm=="uniformGap1000"
  fn=@PairBoundaryWGAN_UniformGapProbe;
 end
 clock=tic; [Model,T]=fn('trainifneeded',Model,B.Data,B.Gate,P,O); seconds=toc(clock);
 assert(T.trained && T.updates==updates);
 if arm=="warm20", assert(abs(T.postDiagnostics.allEndpointRMSE-B.Expected.postDiagnostics.allEndpointRMSE)<1e-9); end
 rng(20260907,'twister'); Conditions=repelem(B.Data.cF,8,1);
 [~,Q]=PairBoundaryWGAN_RC('sample',Model,Conditions,P,O);
 trueF=repelem(B.Data.xF,8,1); trueI=repelem(B.Data.xI,8,1);
 xf=P.lower+Q.generatedF.*(P.upper-P.lower); xi=P.lower+Q.generatedI.*(P.upper-P.lower);
 yf=P.CalObj(xf); yi=P.CalObj(xi); cf=P.CalCon(xf); ci=P.CalCon(xi);
 TF=P.CalObj(P.lower+trueF.*(P.upper-P.lower)); TI=P.CalObj(P.lower+trueI.*(P.upper-P.lower));
 row=struct('problem',string(name),'FE',B.Pop.observationFE,'arm',arm,'pairs',B.Data.count, ...
  'updates',updates,'epochs',T.epochs,'RMSE',T.postDiagnostics.allEndpointRMSE, ...
  'relativeError',T.postDiagnostics.relativeEndpointError,'thickness',T.postDiagnostics.sameConditionThickness, ...
  'probeRMSE',sqrt(mean([Q.generatedF-trueF;Q.generatedI-trueI].^2,'all')), ...
  'objectiveRMSE',sqrt(mean([yf-TF;yi-TI].^2,'all')), ...
  'feasibleLabelAccuracy',mean(all(cf<=0,2)),'infeasibleLabelAccuracy',mean(any(ci>0,2)), ...
  'seconds',seconds);
 rows{end+1}=row; outputs{end+1}=struct('arm',arm,'yf',yf,'yi',yi,'T',T); %#ok<AGROW>
 fprintf('ARM %s %s\n',name,jsonencode(row));
 writetable(struct2table(vertcat(rows{:})),fullfile(out,sprintf('%s_FE%06d_arms.csv',name,targetFE)));
 save(fullfile(out,sprintf('%s_FE%06d_outputs.mat',name,targetFE)),'outputs','B','-v7.3');
 assert(P.FE==0,'Offline diagnosis must not spend search FE.');
end
fprintf('ARMS_COMPLETE %s\n',name);
end
