function rebuild_recorded_states
warning('off','all'); restoredefaultpath; maxNumCompThreads(1);
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support')); addCBSPaths(root);
out='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO/Data/PairGuideDominanceDrift_20260907'; rows={};
for number=5:9
 name=sprintf('LIRCMOP%d_BC',number); ctor=str2func(name); P=ctor();
 R=load(fullfile(root,'Data','PairGuideCCMOFront1_5to9_20260907',[name,'_seed01_cgan.mat']),'Audit'); E=R.Audit.evidence;
 first=find(cellfun(@(s)s.trained,E.training),1); first=E.training{first}.generation;
 stages=unique([first,arrayfun(@(fe)find(cellfun(@(s)s.observationFE,E.population)<=fe,1,'last')-1,[10000 30000 50000 70000 100000])]);
 for g=stages
  before=E.population{g}; now=E.population{g+1};
  O1=E.evaluations{2*g}; O2=E.evaluations{2*g+1};
  U=[SOLUTION(before.p1Decs,before.p1Objs,before.p1Cons),SOLUTION(before.p2Decs,before.p2Objs,before.p2Cons), ...
   SOLUTION(O1.decisions,O1.objectives,O1.constraints),SOLUTION(O2.decisions,O2.objectives,O2.constraints)];
  P1=SOLUTION(now.p1Decs,now.p1Objs,now.p1Cons); previousRNG=rng;
  [A,~,T]=PairBoundaryArchive_RC('update',before.archive,P1,U,E.W,P,struct(),E.generations{g}.use,now.observationFE);
  assert(isequal(previousRNG,rng) && P.FE==0 && numel(A.id)<=500);
  fx=[U(all(U.cons<=0,2)).decs;before.archive.xf]; fy=[U(all(U.cons<=0,2)).objs;before.archive.yf];
  [~,k]=unique(fx,'rows','stable'); fy=fy(k,:);
  for y=A.yf', assert(~any(all(fy<=y',2)&any(fy<y',2))); end
  for y=A.yi', assert(~any(all(fy<=y',2)&any(fy<y',2))); end
  assert(all(T.retainedFrontRanks==1) && T.eligibilityRejectedPairs==0 && numel(unique(A.ref(A.active)))==nnz(A.active));
  rows{end+1}=struct('problem',string(name),'FE',now.observationFE,'oldPairs',numel(now.archive.id), ...
   'newPairs',numel(A.id),'oldActive',nnz(now.archive.active),'newActive',nnz(A.active), ...
   'rejectedPairs',T.eligibilityRemoved,'infeasibleDominatedPairs',T.infeasibleDominatedPairs); %#ok<AGROW>
 end
 fprintf('REAL_STATE_PASS %s nodes=%d\n',name,numel(stages));
end
writetable(struct2table(vertcat(rows{:})),fullfile(out,'rebuilt_states.csv'));
fprintf('ALL_30_REAL_STATES_PASS\n');
end
