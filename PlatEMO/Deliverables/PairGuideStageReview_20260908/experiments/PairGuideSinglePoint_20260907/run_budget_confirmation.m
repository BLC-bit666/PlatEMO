warning('off','all'); restoredefaultpath; maxNumCompThreads(1);
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support')); addCBSPaths(root);
folder=fullfile(root,'Data','PairGuideSinglePoint_20260907','first_use');
Tasks=struct('problem',{},'config',{},'arm',{},'seed',{});
for seed=1:3
    for updates=[200 500 1000 2000 4000]
        for problem=5:8
            O=struct('initialEpoch',updates,'retrainEpoch',20,'nCritic',5,'lrG',1e-3,'lrD',1e-3, ...
                'trainingSigma',0,'sampleSigma',0);
            Tasks(end+1)=struct('problem',problem,'config',O,'arm',string(sprintf('lrzero%04d',updates)),'seed',seed);
        end
    end
end
pool=parpool('Processes',10); errors=strings(numel(Tasks),1);
parfor k=1:numel(Tasks)
    try
        task=Tasks(k);
        run_PairGuide_single_first_use(root,folder,task.problem,task.seed,task.arm,task.config);
    catch err
        errors(k)=string(getReport(err,'extended','hyperlinks','off')); fprintf('FAILED %d %s\n',k,errors(k));
    end
end
save(fullfile(folder,'confirmation_status.mat'),'Tasks','errors'); delete(pool);
assert(all(strlength(errors)==0)); disp('CONFIRMATION_COMPLETE');
