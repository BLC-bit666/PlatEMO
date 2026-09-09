function run_search_arm(useSide,nWorker,seeds)
% Same search protocol; one explicit condition intervention per campaign.
if nargin<2; nWorker=8; end
if nargin<3; seeds=1:3; end
warning('off','all'); restoredefaultpath; maxNumCompThreads(1);
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support')); addCBSPaths(root);
saved=jsondecode(fileread(fullfile(root,'Data','PairGuideSinglePoint_20260907','selected_configuration.json')));
O=saved.trainingOptions; O.generatorHidden=O.generatorHidden'; O.criticHidden=O.criticHidden';
O.stableConditionSpan=true; O.useSideCondition=logical(useSide);
arm="stable_side"; if ~useSide; arm="stable_no_side"; end
folder=fullfile(root,'Data','PairGuideGPTValidation_20260908',char(arm));
if ~isequal(seeds,1:3); folder=folder+"_confirmation"; end
Options=struct('problems',["LIRCMOP5_BC","LIRCMOP6_BC","LIRCMOP7_BC","LIRCMOP8_BC"], ...
    'seeds',seeds,'N',100,'maxFE',100000,'modes',"cgan", ...
    'checkpointFE',[10000 30000 50000 70000 100000],'distributionLayout',"training_pair", ...
    'outputDir',string(folder),'trainingOptions',O);
State=run_PairGuide_validation(root,nWorker,Options);
assert(State.status=="complete"); fprintf('SEARCH_ARM_COMPLETE %s\n',arm);
end
