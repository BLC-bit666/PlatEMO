function export_fixed_schedules
warning('off','all');folder=fileparts(mfilename('fullpath'));
output=fullfile(folder,'batch_schedules');if ~isfolder(output);mkdir(output);end
for n=7:8
 for seed=1:3
  name=sprintf('LIRCMOP%d_BC_seed%02d',n,seed);
  A=load(fullfile(folder,'fixed_conditions',[name,'_moving0.mat']),'Schedule');
  B=load(fullfile(folder,'fixed_conditions',[name,'_moving1.mat']),'Schedule');
  assert(isequal(A.Schedule,B.Schedule));writematrix(A.Schedule,fullfile(output,[name,'.csv']));
 end
end
fprintf('EXPORTED_SIX_IDENTICAL_PAIRED_BATCH_SCHEDULES\n');
end
