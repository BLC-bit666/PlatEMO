function run_factorial
warning('off','all'); restoredefaultpath; maxNumCompThreads(1);
out='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Data/PairGuideTrainingDiagnosis_20260907'; addpath(out);
arms=["fresh32uniform","fresh128uniform","onehot32uniform"];
pool=parpool('Processes',2); cleanup=onCleanup(@()delete(pool));
options=parforOptions(pool,'RangePartitionMethod','fixed','SubrangeSize',1);
parfor (k=1:6,options)
 n=7+floor((k-1)/3); a=arms(1+mod(k-1,3));
 run_arm(n,99800,a+"1000"); run_arm(n,99800,a+"5000");
end
fprintf('FACTORIAL_COMPARISONS_COMPLETE\n');
end
