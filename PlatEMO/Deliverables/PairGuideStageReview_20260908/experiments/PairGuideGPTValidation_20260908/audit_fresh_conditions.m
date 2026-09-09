function audit_fresh_conditions
warning('off','all');maxNumCompThreads(1);folder=fileparts(mfilename('fullpath'));rows=struct([]);allRecords=cell(6,1);k=0;
for n=7:8
 for seed=1:3
  name=sprintf('LIRCMOP%d_BC_seed%02d',n,seed);
  A=load(fullfile(folder,'fixed_conditions',[name,'_moving0.mat']));
  B=load(fullfile(folder,'fresh_conditions',[name,'_fresh.mat']));
  assert(isequal(A.Schedule,B.Schedule)&&B.base==0);
  M=B.Snapshots{1}.Model;
  assert(isempty(M.avgG)&&isempty(M.avgC)&&isempty(M.avgSqG)&&isempty(M.avgSqC));
  assert(M.iterG==0&&M.iterC==0&&isequal(M.generatorHidden,A.Snapshots{1}.Model.generatorHidden));
  assert(~isequaln(M.netG.Learnables,A.Snapshots{1}.Model.netG.Learnables));
  for j=1:5
   sa=A.Snapshots{j};sb=B.Snapshots{j};
   assert(isequal(sa.Data.xF,sb.Data.xF)&&isequal(sa.Data.xI,sb.Data.xI));
   assert(isequal(sa.Data.cF,sb.Data.cF)&&isequal(sa.Data.cI,sb.Data.cI));
   assert(sb.Model.iterG==B.Table.addedUpdates(j)&&sb.Model.iterC==5*B.Table.addedUpdates(j));
   assert(B.Table.offlineFullFE(j)==200&&B.Table.offlineCalConRows(j)==200);
  end
  k=k+1;T=B.Table;T.initialization=repmat("fresh",height(T),1);allRecords{k}=T;
  rows=[rows;struct('problem',string(sprintf('LIRCMOP%d_BC',n)),'seed',seed, ...
   'sameDataConditionsAndBatchIDs',true,'sameArchitecture',true,'freshAdamAtStart',true, ...
   'warmFinalRMSE',A.Table.endpointRMSE(end),'freshFinalRMSE',B.Table.endpointRMSE(end), ...
   'warmFinalDirectionError',A.Table.directionError(end),'freshFinalDirectionError',B.Table.directionError(end), ...
   'warmFinalObjectiveDistance',A.Table.nearestTrainingObjective(end), ...
   'freshFinalObjectiveDistance',B.Table.nearestTrainingObjective(end),'offlineFullFE',1000)]; %#ok<AGROW>
 end
end
writetable(struct2table(rows),fullfile(folder,'fresh_condition_verification.csv'));
writetable(vertcat(allRecords{:}),fullfile(folder,'fresh_condition_records.csv'));
fprintf('FRESH_CONDITION_AUDIT_COMPLETE 6 cases same batch IDs/data/conditions/architecture\n');
end
