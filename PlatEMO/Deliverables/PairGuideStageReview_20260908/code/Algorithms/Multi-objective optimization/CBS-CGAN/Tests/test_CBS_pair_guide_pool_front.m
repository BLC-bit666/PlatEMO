function test_CBS_pair_guide_pool_front
%TEST_CBS_PAIR_GUIDE_POOL_FRONT Joint fronts and private-parent selection.
root = fileparts(which('platemo')); addCBSPaths(root);
D = PairGuide.mainlineDefaults();
assert(D.selectionPool=="ccmo" && D.archiveFrontDepth==1 && D.frontDepth==1);
P = LIRCMOP5_BC('N',100,'D',2,'maxFE',1000);
W = [1 1];
A = struct('id',(1:3)','ref',ones(3,1), ...
    'xf',[.1 .1;.4 .4;.7 .7],'xi',[.1 .100001;.4 .400001;.7 .700001], ...
    'yf',[1 1;2 2;3 3],'yi',[4 0;4 0;4 0], ...
    'gap',1e-6*ones(3,1),'active',true(3,1),'nextId',4);
for depth = 1:2
    [R,~,T] = PairBoundaryArchive_RC('update',A,[],[],W,P, ...
        struct('archiveFrontDepth',depth),struct(),0);
    assert(isequal(T.retainedFrontRanks,(1:depth)') && T.eligiblePairs==depth);
    assert(numel(R.id)==depth && nnz(R.active)==1 && T.frontRejectedPairs==3-depth);
end
% All feasible Union rows matter, even if a row has no local opposite partner.
W = [linspace(0,1,100)',linspace(1,0,100)'];
A.id=1; A.nextId=2; A.ref=1; A.xf=[.2 .2]; A.xi=[.200001 .2];
A.yf=[2 2]; A.yi=[3 0]; A.gap=1e-6; A.active=true;
U = SOLUTION([.9 .9],[1 1],0);
[R,~,T] = PairBoundaryArchive_RC('update',A,U,U,W,P, ...
    struct('archiveFrontDepth',1),struct(),0);
assert(isempty(R.id) && T.frontRejectedPairs==1);
assert(T.eligiblePairs==0,'Unpaired feasible points must affect the joint front.');
[Default,~,DT] = PairBoundaryArchive_RC('update',A,U,U,W,P,struct(),struct(),0);
assert(isequaln(Default,R) && isequaln(DT,T),'The default archive must use global front one.');
[R,~,T] = PairBoundaryArchive_RC('update',A,U,U,W,P, ...
    struct('archiveFrontDepth',2),struct(),0);
assert(R.active && T.eligiblePairs==1 && T.p1DominatedPairs==1, ...
    'Front two must replace, rather than stack with, the P1 dominance gate.');
assert(P.FE==0,'Archive classification must not evaluate objectives or constraints.');
% A globally dominated nearer xi cannot hide a farther admissible partner.
U=SOLUTION([.1 .1;.8 .8;.11 .1;.3 .1], ...
    [1 2;2 1;3 1;.5 3],[0;0;1;1]);
[R,~,T]=PairBoundaryArchive_RC('update',[],U(1:2),U,[1 0;0 1],P,struct(),struct(),0);
assert(numel(R.id)==2 && nnz(R.active)==2 && T.infeasiblePoolEligible==1);
assert(all(R.xi==[.3 .1],'all') && size(unique(R.xf,'rows'),1)==2, ...
    'One admissible infeasible endpoint may serve multiple feasible endpoints.');
assert(~any(all([1 2]<=[3 1],2)),'The rejected xi is dominated by another feasible endpoint.');
% Real feedback is credited first, but its dominated xi must then be deleted.
id=R.id(1); q=R.xf(1,:)+.1*(R.xi(1,:)-R.xf(1,:));
F=struct('childDecs',q,'childObjs',[3 3],'childCons',1,'matchedPairIds',id);
[S,~,T]=PairBoundaryArchive_RC('update',R,[],SOLUTION(q,[3 3],1), ...
    [1 0;0 1],P,struct(),F,1);
assert(any(arrayfun(@(e)e.pairId==id && e.originalPair,T.events)));
assert(~ismember(id,S.id) && ~any(ismember(S.xi,q,'rows')) && T.infeasibleDominatedPairs>0);
% Equal objectives do not constitute strict Pareto dominance.
U=SOLUTION([.1 .1;.2 .1],[1 1;1 1],[0;1]);
R=PairBoundaryArchive_RC('update',[],U(1),U,[1 1],P,struct(),struct(),0);
assert(isscalar(R.id) && R.active && P.FE==0);
% Run only a one-generation, zero-training selection check, not experiment A.
rng(1,'twister'); P=LIRCMOP5_BC('N',100,'maxFE',400);
G=PairGuide('save',0,'outputFcn',@(varargin)[]);
G.configureComparison('fallback_only');
G.Solve(P); S=G.guideExperimentSnapshot(); E=S.evidence;
assert(E.boundaryExperiment.selectionPool=="ccmo" && ...
    E.boundaryExperiment.archiveFrontDepth==1 && S.frontDepth==1);
I=E.population{1}; F=E.evaluations{2}; H=E.evaluations{3};
P1=SOLUTION(I.p1Decs,I.p1Objs,I.p1Cons); P2=SOLUTION(I.p2Decs,I.p2Objs,I.p2Cons);
O1=SOLUTION(F.decisions,F.objectives,F.constraints); O2=SOLUTION(H.decisions,H.objectives,H.constraints);
R1=EnvironmentalSelection_CBS([P1,O1,O2],100,true);
R2=EnvironmentalSelection_CBS([P2,O1,O2],100,false);
assert(isequal(R1.decs,E.population{2}.p1Decs) && isequal(R2.decs,E.population{2}.p2Decs));
assert(~isequal(R1.decs,R2.decs) && E.generations{1}.archive.evaluatedInputPoints==400);
assert(isempty(E.queries) && isempty(E.training));
fprintf('VERIFIED joint front depth, unpaired feasible inputs, no stacked P1 gate, CCMO pools, complete Union.\n');
end
