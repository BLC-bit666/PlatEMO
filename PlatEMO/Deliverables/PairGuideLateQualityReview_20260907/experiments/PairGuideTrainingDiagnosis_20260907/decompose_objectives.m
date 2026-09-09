function decompose_objectives
% Exact algebra on saved outputs; no new objective or constraint oracle calls.
warning('off','all'); restoredefaultpath; maxNumCompThreads(1);
out='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Data/PairGuideTrainingDiagnosis_20260907'; rows={};
for n=[7 8]
 for arm=["production20","onehot32_10000"]
  R=load(fullfile(out,sprintf('LIRCMOP%d_BC_FE099800_%s.mat',n,arm)));
  actual=repelem([R.D.xF;R.D.xI],64,1); generated=[R.Q.generatedF;R.Q.generatedI];
  dim=size(actual,2); cross=zeros(size(actual,1),2); square=cross;
  for j=2:dim
   a=.5*j/dim*pi;
   if mod(j,2), old=actual(:,j)-sin(a*actual(:,1)); now=generated(:,j)-sin(a*generated(:,1)); col=1;
   else, old=actual(:,j)-cos(a*actual(:,1)); now=generated(:,j)-cos(a*generated(:,1)); col=2; end
   change=now-old; cross(:,col)=cross(:,col)+20*old.*change; square(:,col)=square(:,col)+10*change.^2;
  end
  base=zeros(size(cross)); base(:,1)=generated(:,1)-actual(:,1);
  if n==7, base(:,2)=sqrt(actual(:,1))-sqrt(generated(:,1)); else, base(:,2)=actual(:,1).^2-generated(:,1).^2; end
  observed=[R.ygf-R.yf;R.ygi-R.yi]; err=max(abs(base+cross+square-observed),[],'all'); assert(err<1e-10);
  row=struct('problem',sprintf('LIRCMOP%d_BC',n),'arm',arm,'endpointRows',size(actual,1), ...
   'objectiveRMSE',sqrt(mean(observed.^2,'all')),'baseTermRMS',sqrt(mean(base.^2,'all')), ...
   'crossTermRMS',sqrt(mean(cross.^2,'all')),'squareTermRMS',sqrt(mean(square.^2,'all')), ...
   'crossTermMean',mean(cross,'all'),'squareTermMean',mean(square,'all'),'identityMaxError',err);
  rows{end+1}=row; %#ok<AGROW>
 end
end
writetable(struct2table(vertcat(rows{:})),fullfile(out,'objective_decomposition.csv'));
fprintf('EXACT_OBJECTIVE_DECOMPOSITION_VERIFIED\n');
end
