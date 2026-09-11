function test_CBS_pair_guide_exploration_selection
% Global box, 16+4 allocation, uncovered outside requests and exact costs.
    addCBSPaths(fileparts(which('platemo')));
    P=LIRCMOP5_BC('N',100,'D',2,'maxFE',100);
    W=[0 1;.25 .75;.5 .5;.75 .25;1 0];
    Scale=struct('minimum',[0 0],'span',[1 1]);
    A=struct('id',[1;2],'ref',[1;5],'xf',[.2 .2;.4 .6], ...
        'xi',[.6 .4;.8 .8],'yf',[0 1;1 0],'yi',[0 1;1 0], ...
        'active',[true;false],'gap',sqrt([.2;.2]),'nextId',3);
    O=struct('W',W,'guideQuota',20,'currentDecs',[.3 .3], ...
        'p1AllObjs',[.5 .5],'p1Objs',zeros(0,2));
    % P1's infeasible point still means reference3 is not empty. Inactive
    % archive endpoints contribute to both the global box and nearest F.
    X=[.22 .22;.9 .1;.41 .61;.95 .9;.7 .21;.9 .6;.3 .3;NaN .5;-.1 .4];
    I=struct('refs',[1;2;5;4;1;3;1;2;4],'sides',[0;1;0;1;0;1;0;1;0], ...
        'pairIds',zeros(9,1));
    PairGuideCost_RC('start',P);rng(19,'twister');
    [Q,R,Ids,T]=PairBoundaryArchive_RC('selectcandidates',X,I,A,Scale,P,O);
    assert(isequal(T.keepIdx(1:3),[3;1;5]) && isequal(sort(T.keepIdx(4:5)),[2;4]));
    assert(isequal(Q,X(T.keepIdx,:)) && isequal(R,I.refs(T.keepIdx)) && all(Ids==0));
    assert(isequal(T.knownRefs,[1;3;5]) && isequal(T.uncoveredRefs,[2;4]));
    assert(isequal(T.archiveBoxLower,[.2 .2]) && isequal(T.archiveBoxUpper,[.8 .8]));
    assert(~T.insideArchiveBox(2) && ~T.insideArchiveBox(4) && T.keptUncoveredCount==2);
    assert(T.insideQuota==16 && T.outsideQuota==4 && T.keptInsideCount==3 && T.keptOutsideCount==2);
    assert(~T.requestedUncovered(6) && ~T.insideArchiveBox(6) && ~ismember(6,T.keepIdx));
    O.guideQuota=4;rng(19,'twister');
    [~,~,~,All]=PairBoundaryArchive_RC('selectcandidates',X,I,A,Scale,P,O);
    assert(isequal(All.keepIdx,[3;1;5]) && All.keptOutsideCount==0 && ~ismember(7,All.keepIdx));
    % Row5 lies outside both individual diameter balls, but inside the user's
    % global endpoint box and therefore must remain eligible.
    assert(all(vecnorm(X(5,:)-(A.xf+A.xi)/2,2,2)>vecnorm(A.xi-A.xf,2,2)/2));
    assert(All.insideArchiveBox(5) && ismember(5,All.keepIdx));
    O.guideQuota=20;
    [C,Info]=PairBoundaryArchive_RC('querycontexts',A,W,O,500);
    assert(isequal(unique(Info.refs),(1:5)') && all(ismember([2;4],Info.refs)));
    assert(nnz(C(:,end)==0)==250 && nnz(C(:,end)==1)==250);
    % A corrupt/unknown numeric ID does not remove a point with a valid
    % actual query vector. Explicit external vectors get their own IDs.
    Repair=struct('refs',[NaN;999],'sides',[0;1],'pairIds',[0;0], ...
        'conditions',[W(2,:) 0;-.1 1.1 1]);
    [Recovered,Refs,~,RT]=PairBoundaryArchive_RC('selectcandidates',X([2;4],:),Repair,A,Scale,P,O);
    assert(size(Recovered,1)==2 && isequal(sort(Refs),[2;6]) && RT.recoveredReferenceCount==2);
    assert(all(RT.requestedUncovered) && isequal(RT.queryVectors(end,:),[-.1 1.1]));
    % Enough candidates: unknown in-box points share the inside pool; outside
    % points cannot exceed four slots or repeat a requested direction.
    Dense=O;Dense.W=[linspace(0,1,9)',linspace(1,0,9)'];
    Box=[linspace(.21,.39,24)',.25*ones(24,1)];
    Outside=[.9*ones(30,1),linspace(.1,.9,30)'];
    Candidates=[Box;Outside;.95 .3;.95 .4];
    Info=struct('refs',[2;ones(23,1);repmat([2;3;4;6;7;8],5,1);5;9], ...
        'sides',zeros(56,1),'pairIds',zeros(56,1));
    rng(19,'twister');
    [Q,~,~,T]=PairBoundaryArchive_RC('selectcandidates',Candidates,Info,A,Scale,P,Dense);
    at=T.keepIdx;outside=at(~T.insideArchiveBox(at));
    assert(size(Q,1)==20 && isequal(at(1:16),(1:16)'));
    assert(T.keptInsideCount==16 && T.keptOutsideCount==4 && numel(unique(Info.refs(outside)))==4);
    assert(all(T.requestedUncovered(outside)) && ~any(ismember([55;56],at)));
    assert(T.requestedUncovered(1) && T.insideArchiveBox(1) && ismember(1,at));
    % One uncovered outside direction supplies one slot, not all four;
    % unused exploration slots return to the nearest in-box candidates.
    One=Info;One.refs(25:54)=2;
    [~,~,~,T]=PairBoundaryArchive_RC('selectcandidates',Candidates,One,A,Scale,P,Dense);
    at=T.keepIdx;
    assert(T.keptInsideCount==19 && T.keptOutsideCount==1 && ...
        isequal(at(T.insideArchiveBox(at)),(1:19)'));
    % When P1 covers every direction, all twenty must come from the box.
    Covered=Dense;Covered.p1AllObjs=Dense.W;
    [~,~,~,T]=PairBoundaryArchive_RC('selectcandidates',Candidates,Info,A,Scale,P,Covered);
    assert(isequal(T.keepIdx,(1:20)') && T.keptOutsideCount==0);
    % Missing in-box candidates do not release their slots to exploration.
    Sparse=struct('refs',Info.refs(25:54),'sides',zeros(30,1),'pairIds',zeros(30,1));
    [Q,~,~,T]=PairBoundaryArchive_RC('selectcandidates',Outside,Sparse,A,Scale,P,Dense);
    assert(size(Q,1)==4 && T.keptInsideCount==0 && T.keptOutsideCount==4);
    Dense.guideQuota=5;
    [~,~,~,T]=PairBoundaryArchive_RC('selectcandidates',Candidates,Info,A,Scale,P,Dense);
    assert(T.keptInsideCount==4 && T.keptOutsideCount==1);
    O.guideQuota=0;[Empty,~,~,Zero]=PairBoundaryArchive_RC('selectcandidates',X,I,A,Scale,P,O);
    assert(isempty(Empty) && isempty(Zero.keepIdx));
    Cost=PairGuideCost_RC('snapshot');PairGuideCost_RC('stop');
    assert(P.FE==0 && Cost.CalObjRows==0 && Cost.CalConRows==0);
    fprintf('EXPLORATION_SELECTION_PASS global box / 16+4 cap / nearest F / one outside per direction / backfill / zero oracle\n');
end
