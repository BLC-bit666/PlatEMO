warning('off','all'); restoredefaultpath; maxNumCompThreads(1);
folder='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Data/PairGuideSinglePoint_20260907'; addpath(folder);
C=jsondecode(fileread(fullfile(folder,'selected_configuration.json')));
C.trainingOptions.generatorHidden=C.trainingOptions.generatorHidden'; C.trainingOptions.criticHidden=C.trainingOptions.criticHidden';
C.trainingOptions.initialEpoch=4000; run_direction_holdout(C.trainingOptions,'continued4000','selected1000'); disp('HOLDOUT_CONTINUATION_COMPLETE');
