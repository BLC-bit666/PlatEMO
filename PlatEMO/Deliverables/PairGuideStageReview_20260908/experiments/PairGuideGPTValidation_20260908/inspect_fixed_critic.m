function inspect_fixed_critic
% Forward diagnostics on saved models only; no training or oracle evaluation.
warning('off','all');maxNumCompThreads(1);folder=fileparts(mfilename('fullpath'));
files=dir(fullfile(folder,'fixed_conditions','LIRCMOP*.mat'));rows=struct([]);
for k=1:numel(files)
 R=load(fullfile(files(k).folder,files(k).name),'Snapshots','task');
 for j=1:numel(R.Snapshots)
  S=R.Snapshots{j};M=S.Model;D=S.Data;C=single([D.cF;D.cI]');
  Z=zeros(M.zDim,size(C,2),'single');
  X=single(2*[D.xF;D.xI]'-1);
  Q=forward(M.netG,dlarray([Z;C],'CB'));
  real=extractdata(forward(M.netC,dlarray([X;C],'CB')));
  fake=extractdata(forward(M.netC,[Q;dlarray(C,'CB')]));
  row=struct('problem',string(sprintf('LIRCMOP%d_BC',R.task.number)), ...
   'seed',R.task.seed,'moving',R.task.moving,'addedUpdates',S.measurement.addedUpdates, ...
   'criticGap',double(mean(real,'all')-mean(fake,'all')), ...
   'realMean',double(mean(real,'all')),'fakeMean',double(mean(fake,'all')));
  rows=[rows;row]; %#ok<AGROW>
 end
end
writetable(struct2table(rows),fullfile(folder,'fixed_critic_diagnostics.csv'));
fprintf('FIXED_CRITIC_DIAGNOSTICS_COMPLETE zero oracle calls\n');
end
