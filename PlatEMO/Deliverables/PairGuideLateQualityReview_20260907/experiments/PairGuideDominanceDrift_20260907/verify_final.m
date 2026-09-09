function verify_final
warning('off','all'); restoredefaultpath; maxNumCompThreads(1);
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support')); addCBSPaths(root);
test_CBS_pair_guide_training_observation;
fprintf('UPDATED_OBSERVATION_PASS\n');
rng(1,'twister'); P=LIRCMOP5_BC('N',100,'maxFE',6000);
A=PairGuide('save',1,'outputFcn',@(varargin)[]);
A.configurePairGuideTrainingExperiment(struct('initialEpoch',2,'retrainEpoch',1,'nCritic',1)); A.Solve(P);
R=A.guideExperimentSnapshot(); E=R.evidence;
Record=struct('problem',"LIRCMOP5_BC",'seed',1,'mode',"cgan");
before=rng; T=PairGuide_checkpoint_metrics(P,E,[1000 3000 6000],Record);
assert(isequal(before,rng));
last=E.generations{end}.archive; row=T(T.actualFE==6000,:);
assert(row.frontRejectedPairs==last.frontRejectedPairs && row.infeasibleDominatedPairs==last.infeasibleDominatedPairs && row.eligibilityRemoved==last.eligibilityRemoved);
for k=1:numel(E.generations)
 for field=["frontRejectedPairs","infeasibleDominatedPairs","eligibilityRemoved","feasiblePoolEligible","infeasiblePoolEligible"]
  E.generations{k}.archive=rmfield(E.generations{k}.archive,field);
 end
end
Old=PairGuide_checkpoint_metrics(P,E,[6000],Record);
assert(all(isnan(Old.eligibilityRemoved)) && all(isnan(Old.infeasiblePoolEligible)));
fprintf('NEW_AND_LEGACY_CHECKPOINT_FIELDS_PASS\n');
names={'PairBoundaryArchive_RC','PairGuide_checkpoint_metrics','run_PairGuide_pool_front_first_use', ...
 'test_CBS_pair_guide','test_CBS_pair_guide_pool_front','test_CBS_pair_guide_archive_capacity','test_CBS_pair_guide_training_observation'};
for k=1:numel(names)
 messages=checkcode(which(names{k}),'-id');
 fprintf('STATIC %s messages=%d\n',names{k},numel(messages));
 for j=1:numel(messages), fprintf('  %d %s %s\n',messages(j).line,messages(j).id,messages(j).message); end
end
fprintf('FINAL_CHECKS_COMPLETE\n');
end
