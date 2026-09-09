function run_extended
warning('off','all'); restoredefaultpath; maxNumCompThreads(1);
out='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Data/PairGuideTrainingDiagnosis_20260907'; addpath(out);
arms=["warm","uniformGap","fresh32_","fresh128_","onehot32_"];
pool=parpool('Processes',4); cleanup=onCleanup(@()delete(pool));
options=parforOptions(pool,'RangePartitionMethod','fixed','SubrangeSize',1);
parfor (k=1:2*numel(arms),options)
 n=7+floor((k-1)/numel(arms)); a=arms(1+mod(k-1,numel(arms)));
 run_arm(n,99800,a+"5000");
end
fprintf('EXTENDED_COMPARISONS_COMPLETE\n');
end
