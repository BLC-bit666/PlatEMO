function validate_archive
warning('off','all'); restoredefaultpath; maxNumCompThreads(1);
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support')); addCBSPaths(root);
addpath('/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Data/PairGuideDominanceDrift_20260907');
tests={'reproduce_dominance','test_CBS_pair_guide_pool_front','test_CBS_pair_guide_archive_capacity', ...
 'test_CBS_pair_guide','test_CBS_pair_guide_training_observation','test_CBS_platemo_compliance','test_CBS_region_wgan_mainline'};
failed={};
for k=1:numel(tests)
 try
  feval(tests{k}); fprintf('TEST_PASS %s\n',tests{k});
 catch e
  failed{end+1}=tests{k}; fprintf('TEST_FAIL %s\n%s\n',tests{k},getReport(e,'extended','hyperlinks','off')); %#ok<AGROW>
 end
end
assert(isempty(failed),'Tests failed: %s',strjoin(failed,', '));
fprintf('ALL_TESTS_PASS\n');
end
