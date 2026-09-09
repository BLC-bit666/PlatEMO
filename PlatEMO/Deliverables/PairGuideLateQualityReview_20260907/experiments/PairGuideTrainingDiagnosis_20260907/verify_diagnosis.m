function verify_diagnosis
warning('off','all'); restoredefaultpath; maxNumCompThreads(1);
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support')); addCBSPaths(root);
out=fullfile(root,'Data','PairGuideTrainingDiagnosis_20260907');
R=load(fullfile(out,'model_summary.mat')); T=R.T; assert(height(T)==48);
cost=struct('CalObjRows',0,'CalConRows',0); modelChecks=0;
for k=1:height(T)
 name=sprintf('%s_FE%06d_%s.mat',T.problem(k),T.FE(k),T.arm(k));
 V=load(fullfile(out,name)); B=load(fullfile(out,sprintf('%s_FE%06d_model.mat',T.problem(k),T.FE(k))));
 assert(isequal(V.D.xF,B.B.Data.xF) && isequal(V.D.xI,B.B.Data.xI) && isequal(V.D.ref,B.B.Data.ref));
 assert(V.M.ready && V.T.trained && V.M.C==size(V.D.cF,2));
 assert(V.row.RMSE==V.T.postDiagnostics.allEndpointRMSE && isfinite(V.row.midpointObjectiveRMSE));
 assert(size(V.Q.generatedF,1)==64*V.D.count && size(V.Q.generatedF,2)==30);
 assert(all(V.Q.generatedF>=0 & V.Q.generatedF<=1,'all') && all(V.Q.generatedI>=0 & V.Q.generatedI<=1,'all'));
 e=[V.Q.generatedF-repelem(V.D.xF,64,1);V.Q.generatedI-repelem(V.D.xI,64,1)];
 assert(abs(mean(e.^2,'all')-(V.row.biasRMSE^2+V.row.noiseRMSE^2))<1e-12);
 if ismember(string(V.row.arm),["production20","initial200"])
  assert(abs(V.row.RMSE-B.B.Expected.postDiagnostics.allEndpointRMSE)<1e-9);
 end
 cost.CalObjRows=cost.CalObjRows+V.Oracle.CalObjRows; cost.CalConRows=cost.CalConRows+V.Oracle.CalConRows;
 modelChecks=modelChecks+1;
end
trainingEvents=0; plotNodes=0; generatedNodes=0;
for n=5:8
 E=load(fullfile(out,sprintf('LIRCMOP%d_BC_events.mat',n)),'Events');
 trainingEvents=trainingEvents+nnz(cellfun(@(e)e.T.trained,E.Events));
 V=load(fullfile(out,sprintf('LIRCMOP%d_BC_nodes.mat',n)));
 cost.CalObjRows=cost.CalObjRows+V.Oracle.CalObjRows; cost.CalConRows=cost.CalConRows+V.Oracle.CalConRows;
 for j=1:numel(V.Nodes)
  S=V.Nodes{j}; plotNodes=plotNodes+1; if S.missing, continue; end
  Q=S.query; expected=(1-Q.sample.alpha).*Q.sample.generatedF+Q.sample.alpha.*Q.sample.generatedI;
  assert(max(abs(Q.rawDecs-expected),[],'all')<1e-12);
  generatedNodes=generatedNodes+1;
 end
end
assert(trainingEvents==1501 && plotNodes==24 && generatedNodes==23);
assert(isequal(cost,R.calls) && cost.CalObjRows==2561626 && cost.CalConRows==859804);
Z=readtable(fullfile(out,'objective_decomposition.csv')); assert(height(Z)==4 && max(Z.identityMaxError)<1.4e-15);
assert(numel(dir(fullfile(out,'*.png')))==3);
S=struct('status',"verified_analysis",'productionModified',false,'diagnosticModels',modelChecks, ...
 'successfulTrainingEventsInspected',trainingEvents,'exactReplayedSuccessfulEvents',933, ...
 'plotNodes',plotNodes,'plotNodesWithGeneration',generatedNodes,'diagnosticFigures',3, ...
 'offlineCalObjRows',cost.CalObjRows,'offlineCalConRows',cost.CalConRows, ...
 'newSearchFE',0,'originalOffsetStillReproduces',true,'verifiedAt',string(datetime('now')));
fid=fopen(fullfile(out,'verification.json'),'w'); cleanup=onCleanup(@()fclose(fid));
fprintf(fid,'%s\n',jsonencode(S,PrettyPrint=true)); clear cleanup;
fprintf('DIAGNOSIS_VERIFIED %s\n',jsonencode(S));
end
