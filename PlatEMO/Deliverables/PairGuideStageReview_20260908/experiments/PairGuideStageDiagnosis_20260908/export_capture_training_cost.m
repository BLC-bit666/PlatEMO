function export_capture_training_cost
% The replay check proves these source final weights equal this round's replays.
warning('off','all');folder=fileparts(mfilename('fullpath'));root=fileparts(fileparts(folder));
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support'));addCBSPaths(root);
rows={};
for n=5:8
    check=readtable(fullfile(folder,sprintf('LIRCMOP%d_BC_seed01_capture_check.csv',n)));
    assert(check.sameG && check.sameD && check.sameP1 && check.sameP2 && check.sameLastQuery);
    source=fullfile(root,'Data','PairGuideSinglePoint_20260907','full_run',sprintf('LIRCMOP%d_BC_seed01_cgan.mat',n));
    R=load(source,'Audit');M=R.Audit.evidence.lastModel;
    rows{end+1}=struct('problemNumber',n,'searchSeed',1,'searchFE',100000,'GUpdates',M.iterG,'DUpdates',M.iterC);
end
writetable(struct2table(vertcat(rows{:})),fullfile(folder,'capture_training_cost.csv'));
fprintf('CAPTURE_TRAINING_COST_EXPORTED; cached final model iteration counters, no oracle calls\n');
end
