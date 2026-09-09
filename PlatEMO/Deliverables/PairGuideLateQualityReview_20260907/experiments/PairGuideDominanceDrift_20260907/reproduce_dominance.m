function reproduce_dominance
warning('off','all'); restoredefaultpath;
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support')); addCBSPaths(root);
P=LIRCMOP5_BC('N',100,'D',2,'maxFE',1000);
U=SOLUTION([.1 .1;.11 .1;.3 .1],[1 1;2 2;.5 2],[0;1;1]);
A=PairBoundaryArchive_RC('update',[],U(1),U,[1 1],P,struct(),struct(),0);
assert(~isempty(A.id));
assert(~any(all(A.yf(A.active,:)<=A.yi(A.active,:),2)&any(A.yf(A.active,:)<A.yi(A.active,:),2)), ...
 'Reproduced: nearest dominated infeasible endpoint enters an active boundary pair.');
assert(isequal(A.xi(A.active,:),[.3 .1]),'The admissible farther endpoint must be selected.');
fprintf('DOMINANCE_REPRO_PASS\n');
end
