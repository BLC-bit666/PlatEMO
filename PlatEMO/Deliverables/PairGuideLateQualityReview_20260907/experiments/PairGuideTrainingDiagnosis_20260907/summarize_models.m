function summarize_models
warning('off','all'); restoredefaultpath; maxNumCompThreads(1);
out='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Data/PairGuideTrainingDiagnosis_20260907';
files=dir(fullfile(out,'LIRCMOP*_FE*_*000.mat')); files=[files;dir(fullfile(out,'LIRCMOP*_FE*_production20.mat'));dir(fullfile(out,'LIRCMOP*_FE*_initial200.mat'))];
rows={}; calls=struct('CalObjRows',0,'CalConRows',0);
for k=1:numel(files)
 R=load(fullfile(files(k).folder,files(k).name),'row','Probe','Oracle','Q','D','mf','mi');
 if ~isfield(R,'row'), continue; end
 a=R.row; d=R.D.delta; gd=R.mi-R.mf;
 a.trueGapMean=mean(vecnorm(d,2,2)); a.generatedMeanGap=mean(vecnorm(gd,2,2));
 a.deltaRMSE=sqrt(mean((gd-d).^2,'all'));
 a.deltaCosineMean=mean(sum(gd.*d,2)./max(eps,vecnorm(gd,2,2).*vecnorm(d,2,2)));
 a.endpointWeightedLoss=R.Probe.endpointLoss; a.deltaWeightedLoss=R.Probe.directionLoss;
 a.geometryLoss=10*R.Probe.endpointLoss+R.Probe.directionLoss;
 a.adversarialLoss=R.Probe.adversarialLoss;
 rows{end+1}=a; %#ok<AGROW>
 calls.CalObjRows=calls.CalObjRows+R.Oracle.CalObjRows;
 calls.CalConRows=calls.CalConRows+R.Oracle.CalConRows;
end
for n=5:8
 R=load(fullfile(out,sprintf('LIRCMOP%d_BC_nodes.mat',n)),'Oracle');
 calls.CalObjRows=calls.CalObjRows+R.Oracle.CalObjRows;
 calls.CalConRows=calls.CalConRows+R.Oracle.CalConRows;
end
T=struct2table(vertcat(rows{:})); writetable(T,fullfile(out,'model_summary.csv'));
save(fullfile(out,'model_summary.mat'),'T','calls');
fprintf('SUMMARY models=%d oracle=%s\n',height(T),jsonencode(calls));
end
