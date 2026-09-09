warning('off','all'); restoredefaultpath; maxNumCompThreads(1);
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support')); addCBSPaths(root);
folder=fullfile(root,'Data','PairGuideSinglePoint_20260907','first_use');
base=struct('initialEpoch',1000,'retrainEpoch',20,'nCritic',5);
configs={base,base,base,base};
configs{1}.lrG=1e-3; configs{1}.lrD=1e-3;
configs{2}.trainingSigma=0; configs{2}.sampleSigma=0;
configs{3}.generatorHidden=[64 64]; configs{3}.criticHidden=[64 64];
configs{4}.nCritic=1; configs{4}.lrG=3e-4; configs{4}.lrD=3e-4;
names=["lr1000","zero1000","wide1000","c1lr1000"];
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
save(fullfile(folder,'stage2_status.mat'),'Tasks','errors'); delete(pool);
assert(all(strlength(errors)==0)); disp('STAGE2_COMPLETE');
