warning('off','all'); restoredefaultpath; maxNumCompThreads(1);
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support')); addCBSPaths(root);
folder=fullfile(root,'Data','PairGuideSinglePoint_20260907','first_use');
base=struct('initialEpoch',2000,'retrainEpoch',20,'nCritic',5,'lrG',1e-3,'lrD',1e-3);
configs={base,base,base,base}; configs{1}.initialEpoch=500; configs{3}.initialEpoch=4000;
configs{4}.trainingSigma=0; configs{4}.sampleSigma=0;
names=["lr0500","lr2000","lr4000","lrzero2000"];
Tasks=struct('problem',{},'config',{},'arm',{});
for c=1:numel(configs)
    for problem=5:8
        Tasks(end+1)=struct('problem',problem,'config',configs{c},'arm',names(c));
    end
end
pool=parpool('Processes',10); errors=strings(numel(Tasks),1);
parfor k=1:numel(Tasks)
    try
        task=Tasks(k);
        run_PairGuide_single_first_use(root,folder,task.problem,1,task.arm,task.config);
    catch err
        errors(k)=string(getReport(err,'extended','hyperlinks','off')); fprintf('FAILED %d %s\n',k,errors(k));
    end
end
save(fullfile(folder,'stage3_status.mat'),'Tasks','errors'); delete(pool);
assert(all(strlength(errors)==0)); disp('STAGE3_COMPLETE');
