function collect_plot_evidence
% Reuse offline evaluated plot caches; zero new oracle rows.
warning('off','all');maxNumCompThreads(1);folder=fileparts(mfilename('fullpath'));
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
campaigns={fullfile(root,'Data','PairGuideSinglePoint_20260907','full_run'),'original'; ...
 fullfile(folder,'stable_side'),'stable_side';fullfile(folder,'stable_no_side'),'stable_no_side'};
rows=struct([]);costs=struct([]);
for a=1:size(campaigns,1)
 for n=5:8
  problem=sprintf('LIRCMOP%d_BC',n);
  file=fullfile(campaigns{a,1},'analysis','figures',problem,'run_01','plot_data.mat');
  if ~isfile(file);continue;end
  R=load(file,'States','Report');
  c=struct('arm',string(campaigns{a,2}),'problem',string(problem), ...
   'offlineFullFE',R.Report.offlineFullFE,'CalObjRows',R.Report.oracleCalls.CalObjRows, ...
   'CalConRows',R.Report.oracleCalls.CalConRows,'figures',R.Report.figureCount);
  costs=[costs;c]; %#ok<AGROW>
  for k=1:numel(R.States)
   S=R.States{k};if isempty(S.native);continue;end
   T=[S.xf;S.xi];dist=min(pdist2(S.native,T),[],2);
   row=struct('problem',string(problem),'seed',1,'arm',string(campaigns{a,2}), ...
    'targetFE',S.targetFE,'productionFE',S.productionFE,'trainingFE',S.trainingFE, ...
    'nativeCount',size(S.native,1),'nativeUniqueObjectives',size(unique(S.native,'rows'),1), ...
    'trainingPairs',S.trainingPairs,'trainingEndpointRMSE',S.trainingRMSE, ...
    'meanNearestTrainingObjective',mean(dist),'medianNearestTrainingObjective',median(dist), ...
    'requestedDirectionError',S.directionError,'sideAccuracy',S.labelAccuracy, ...
    'nativeFeasibleFraction',mean(S.trueFeasible==1));
   rows=[rows;row]; %#ok<AGROW>
  end
 end
end
writetable(struct2table(rows),fullfile(folder,'plot_evidence.csv'));
writetable(struct2table(costs),fullfile(folder,'plot_oracle_cost.csv'));
fprintf('COLLECTED_PLOT_CACHES %d\n',numel(costs));
end
