function run_comparisons
warning('off','all'); restoredefaultpath; maxNumCompThreads(1);
out='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Data/PairGuideTrainingDiagnosis_20260907'; addpath(out);
arms=["production20","warm1000","noAdv1000","uniformGap1000","lr1e3_1000", ...
 "noNoise1000","fresh32_1000","fresh128_1000","onehot32_1000"];
pool=parpool('Processes',4); cleanup=onCleanup(@()delete(pool));
options=parforOptions(pool,'RangePartitionMethod','fixed','SubrangeSize',1);
parfor (k=1:2*numel(arms),options)
 n=7+floor((k-1)/numel(arms)); a=arms(1+mod(k-1,numel(arms)));
 run_arm(n,99800,a);
end
fprintf('ALL_FIXED_DATA_COMPARISONS_COMPLETE\n');
end
