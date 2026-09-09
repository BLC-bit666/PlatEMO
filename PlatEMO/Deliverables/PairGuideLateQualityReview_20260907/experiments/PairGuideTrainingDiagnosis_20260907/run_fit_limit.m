function run_fit_limit
warning('off','all'); restoredefaultpath; maxNumCompThreads(1);
out='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Data/PairGuideTrainingDiagnosis_20260907'; addpath(out);
arms=["fresh32_10000","onehot32_10000"];
pool=parpool('Processes',4); cleanup=onCleanup(@()delete(pool));
parfor k=1:4
 n=7+floor((k-1)/2); a=arms(1+mod(k-1,2));
 run_arm(n,99800,a);
end
fprintf('FIT_LIMIT_COMPLETE\n');
end
