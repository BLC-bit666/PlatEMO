warning('off','all'); restoredefaultpath; maxNumCompThreads(1);
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support')); addCBSPaths(root);
folder=fullfile(root,'Data','PairGuideSinglePoint_20260907','first_use');
Tasks=struct('problem',{},'updates',{});
for u=[200 1000 4000]
    for problem=5:8
        Tasks(end+1)=struct('problem',problem,'updates',u);
    end
end
pool=parpool('Processes',10);
errors=strings(numel(Tasks),1);
parfor k=1:numel(Tasks)
    try
        task=Tasks(k); O=struct('initialEpoch',task.updates,'retrainEpoch',20,'nCritic',5);
        run_PairGuide_single_first_use(root,folder,task.problem,1,sprintf('u%04d',task.updates),O);
    catch err
        errors(k)=string(getReport(err,'extended','hyperlinks','off'));
        fprintf('FAILED %d %s\n',k,errors(k));
    end
end
save(fullfile(folder,'stage1_status.mat'),'Tasks','errors'); delete(pool);
assert(all(strlength(errors)==0)); disp('STAGE1_COMPLETE');
