warning('off','all'); restoredefaultpath; maxNumCompThreads(1);
folder='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Data/PairGuideSinglePoint_20260907'; addpath(folder);
C=jsondecode(fileread(fullfile(folder,'selected_configuration.json')));
C.trainingOptions.generatorHidden=C.trainingOptions.generatorHidden'; C.trainingOptions.criticHidden=C.trainingOptions.criticHidden';
run_direction_holdout(C.trainingOptions,'selected1000'); disp('HOLDOUT_COMPLETE');
