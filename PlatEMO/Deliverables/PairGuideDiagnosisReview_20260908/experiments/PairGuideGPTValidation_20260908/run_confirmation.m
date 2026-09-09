function run_confirmation(nWorker,selectedOnly)
% Locked new seeds, only after the preregistered development criterion passes.
if nargin<1;nWorker=8;end
if nargin<2;selectedOnly=false;end
warning('off','all');restoredefaultpath;maxNumCompThreads(1);
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';folder=fullfile(root,'Data','PairGuideGPTValidation_20260908');
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support'));addCBSPaths(root);
S=jsondecode(fileread(fullfile(folder,'development_selection.json')));
assert(S.independentConfirmationRequired && ~isempty(S.selected));
saved=jsondecode(fileread(fullfile(root,'Data','PairGuideSinglePoint_20260907','selected_configuration.json')));
O=saved.trainingOptions;O.generatorHidden=O.generatorHidden';O.criticHidden=O.criticHidden';
Options=struct('problems',["LIRCMOP5_BC","LIRCMOP6_BC","LIRCMOP7_BC","LIRCMOP8_BC"], ...
 'seeds',11:15,'N',100,'maxFE',100000,'checkpointFE',[10000 30000 50000 70000 100000], ...
 'distributionLayout',"training_pair");
% A common original/DE/pair-only campaign (60 cases).
O.stableConditionSpan=false;O.useSideCondition=true;
Options.modes=["cgan","fallback_only","pair_only"];Options.trainingOptions=O;
Options.outputDir=string(fullfile(folder,'confirmation_controls'));
if ~selectedOnly
 State=run_PairGuide_validation(root,nWorker,Options);assert(State.status=="complete");
end
% Locked selected candidate (20 cases); no retuning against new seeds.
O.stableConditionSpan=true;O.useSideCondition=strcmp(S.selected,'stable_side');
Options.modes="cgan";Options.trainingOptions=O;
Options.outputDir=string(fullfile(folder,'confirmation_selected'));
State=run_PairGuide_validation(root,nWorker,Options);assert(State.status=="complete");
fprintf('CONFIRMATION_COMPLETE 80 runs selected=%s\n',S.selected);
end
