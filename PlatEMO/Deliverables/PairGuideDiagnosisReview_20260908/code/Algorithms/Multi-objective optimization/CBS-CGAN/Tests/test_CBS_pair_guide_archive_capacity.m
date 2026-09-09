function test_CBS_pair_guide_archive_capacity
%TEST_CBS_PAIR_GUIDE_ARCHIVE_CAPACITY Bounded competition, not a truncated Union.
    addCBSPaths(fileparts(which('platemo')));
    P = LIRCMOP5_BC('N',100,'D',2,'maxFE',100000);
    W = [linspace(0,1,100)',linspace(1,0,100)'];
    n = 600;
    refs = repmat((1:100)',6,1);
    xf = [linspace(0.1,0.7,n)',0.2*ones(n,1)];
    xi = xf+[0 1e-6];
    A = struct('id',(1:n)','ref',refs,'xf',xf,'yf',W(refs,:), ...
        'xi',xi,'yi',W(refs,:),'gap',sqrt(sum((xi-xf).^2,2)), ...
        'active',[(true(100,1));false(n-100,1)],'nextId',n+1);
    % Capacity is applied after feedback, even for a record later evicted.
    A.xi(end,2) = A.xf(end,2)+4e-6;
    F = struct('childDecs',A.xf(end,:)+[0 1e-6], ...
        'childObjs',A.yf(end,:),'childCons',0,'matchedPairIds',n);
    before = rng;
    FeedbackUnion = SOLUTION(F.childDecs,F.childObjs,F.childCons);
    [B,~,T] = PairBoundaryArchive_RC('update',A,[],FeedbackUnion,W,P,struct(),F,200);
    assert(numel(B.id) == 500,'The complete archive must retain 500 pairs.');
    assert(nnz(B.active) == 100 && numel(unique(B.ref(B.active))) == 100);
    assert(T.capDropped > 0 && T.retained == 500 && T.afterCap == 500);
    assert(any(arrayfun(@(e)e.pairId == n && e.originalPair,T.events)));
    assert(~ismember(n,B.id),'The original feedback must precede capacity eviction.');
    assert(isequal(before,rng) && P.FE == 0);
    % Rejected history is deleted even when the archive was previously full.
    Legacy = struct('archiveFrontDepth',0);
    [Inactive,~,IT] = PairBoundaryArchive_RC('update',B, ...
        SOLUTION([0.9 0.9],[-1 -1],0),[],W,P,Legacy,struct(),201);
    assert(isempty(Inactive.id) && IT.frontRejectedPairs == 500);
    [Reactivated,~,~] = PairBoundaryArchive_RC('update',Inactive,[],[],W,P,Legacy,struct(),202);
    assert(isempty(Reactivated.id));
    % All 400 current points join the 1000 stored endpoints before competition.
    X = [linspace(0.1001,0.6999,400)',0.2*ones(400,1)];
    Y = W(repmat((1:100)',4,1),:);
    U = SOLUTION(X,Y,zeros(400,1));
    for generation = 1:12
        [B,~,T] = PairBoundaryArchive_RC('update',B,[],U,W,P,struct(),struct(),200+200*generation);
        assert(T.candidateInputPoints == 1400 && T.evaluatedInputPoints == 400);
        assert(numel(B.id) == 500 && nnz(B.active) == 100 && T.afterCap == 500);
        assert(T.archiveFrontDepth == 1 && all(T.retainedFrontRanks(B.active) == 1));
        assert(size(unique([B.ref,B.xf,B.xi],'rows'),1) == 500);
        assert(all(B.gap > 0) && numel(unique(B.id)) == 500);
    end
    % The last Union row can improve its original pair; no first-1000 cutoff.
    target = find(B.active,1,'last'); id = B.id(target);
    q = B.xf(target,:)+0.25*(B.xi(target,:)-B.xf(target,:));
    U = [U(1:399),SOLUTION(q,B.yf(target,:),0)];
    F = struct('childDecs',q,'childObjs',B.yf(target,:), ...
        'childCons',0,'matchedPairIds',id);
    [~,~,T] = PairBoundaryArchive_RC('update',B,[],U,W,P,struct(),F,3000);
    assert(T.candidateInputPoints == 1400 && ...
        any(arrayfun(@(e)e.pairId == id && e.originalPair && isequal(e.decision,q),T.events)));
    % A fully occupied archive stays full even when tightening merges lineages.
    B.id = (1:500)'; B.nextId = 501; B.ref(:) = 1;
    B.xf = repmat([0.1 0.1],500,1);
    B.xi = B.xf+[linspace(0.001,0.5,500)',zeros(500,1)];
    B.yf = repmat([0 1],500,1); B.yi = B.yf;
    B.gap = sqrt(sum((B.xi-B.xf).^2,2));
    [Merged,~,T] = PairBoundaryArchive_RC('update',B,[],[],[1 1],P,struct(),struct(),3200);
    assert(numel(Merged.id)==500 && T.restoredCapacity>0 && ...
        size(unique([Merged.xf,Merged.xi],'rows'),1)==500 && nnz(Merged.active)==1);
    assert(all(ismember(Merged.xf,B.xf,'rows')) && all(ismember(Merged.xi,B.xi,'rows')));
    % Shortage at initialization is not filled with invented or duplicate truth.
    [Empty,~,~] = PairBoundaryArchive_RC('update',[],[],[],W,P,struct(),struct(),0);
    assert(isempty(Empty.id));
    fprintf('PairGuide archive capacity/competition/feedback contract passed.\n');
end
