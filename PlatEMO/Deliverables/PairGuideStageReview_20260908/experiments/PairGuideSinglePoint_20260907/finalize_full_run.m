warning('off','all'); restoredefaultpath; maxNumCompThreads(1);
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support')); addCBSPaths(root);
folder=fullfile(root,'Data','PairGuideSinglePoint_20260907','full_run');
S=load(fullfile(folder,'state.mat'),'State'); State=S.State; assert(State.completed==36 && State.failed==0);
ledger=fullfile(folder,'analysis','renderer_v6_cost.mat');
if ~isfile(ledger)
 PreviousReports=cell(4,1);
 for number=5:8; R=load(fullfile(folder,'analysis','figures',sprintf('LIRCMOP%d_BC',number),'run_01','plot_data.mat'),'Report'); PreviousReports{number-4}=R.Report; end
 save(ledger,'PreviousReports');
end
for number=5:8
 render_PairGuide_interval_distribution(fullfile(folder,sprintf('LIRCMOP%d_BC_seed01_cgan.mat',number)));
end
report_PairGuide_interval_validation(folder);
State.status="complete"; State.offlineRevision="single-v3 summary fix and renderer-v7"; save(fullfile(folder,'state.mat'),'State');
for name=["analyze_PairGuide_interval_validation","render_PairGuide_interval_distribution"]; assert(isempty(checkcode(which(name),'-id'))); end
disp('FINAL_REPORTING_COMPLETE; 36 saved searches reused without retraining.');
