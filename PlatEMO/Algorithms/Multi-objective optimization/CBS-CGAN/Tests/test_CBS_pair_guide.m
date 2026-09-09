function test_CBS_pair_guide()
%TEST_CBS_PAIR_GUIDE Final shared-conversation contract, including real autodiff.
    addCBSPaths(fileparts(which('platemo')));
    P = LIRCMOP5_BC('N',10,'D',2,'maxFE',100);
    O = struct('pairMinPairs',8,'guideQuota',20,'generation',1, ...
        'initialEpoch',2,'retrainEpoch',1,'nCritic',1,'miniBatch',32);
    W = [1 1];
    A = archive([0.2 0.2],[0.8 0.8],[0 1],[1 0],1);
    P1 = SOLUTION([0.05 0.05],[-1 -1],0);
    % Unqualified historical endpoints are deleted, not merely deactivated.
    [Retained,~] = PairBoundaryArchive_RC('update',A,P1,P1,W,P,O,struct(),10);
    assert(isempty(Retained.id) && ~isfield(Retained,'resumeEligible') && ...
        ~isfield(Retained,'lastFE'));
    % Explicit legacy control: P1-only qualification remains available.
    Legacy = O; Legacy.archiveFrontDepth = 0;
    Candidates = archive([0.1 0.1;0.8 0.8], ...
        [0.1 0.10001;0.8 0.80001],[0.2 0.8;0.7 0.7],[0 1;1 0],[1;1]);
    CurrentP1 = SOLUTION([0.5 0.5],[0.6 0.6],0);
    [Eligible,~,ET] = PairBoundaryArchive_RC('update',Candidates,CurrentP1,[],W,P,Legacy,struct(),10);
    assert(Eligible.active(Eligible.id == 1) && ~ismember(2,Eligible.id) && ...
        ET.frontRejectedPairs == 1 && isscalar(Eligible.id));
    [Reactivated,~] = PairBoundaryArchive_RC('update',Eligible, ...
        SOLUTION([0.5 0.5],[-10 -10],1),[],W,P,Legacy,struct(),11);
    assert(nnz(Reactivated.active)==1 && ~ismember(2,Reactivated.id), ...
        'An infeasible P1 cannot reject a valid pair or resurrect deleted history.');
    % Global front one rejects dominated endpoints across reference directions.
    Candidates.yf = [1 2;2 3]; Candidates.yi = [0 4;4 0];
    [Internal,~,IT] = PairBoundaryArchive_RC('update',Candidates, ...
        SOLUTION([0.5 0.5],[5 5],0),[],[1 2;2 3],P,O,struct(),12);
    assert(nnz(Internal.active) == 1 && isscalar(Internal.id) && ...
        IT.eligiblePairs == 1 && IT.frontRejectedPairs == 1, ...
        'Only qualified feasible endpoints may remain in the archive.');
    % Feasible guided feedback need not survive P1. Accept even tiny shrinkage.
    q = [0.200001 0.200001];
    F = struct('childDecs',q,'childObjs',[0 1],'childCons',0,'matchedPairIds',1);
    [Updated,~,T] = PairBoundaryArchive_RC('update',A,[], ...
        SOLUTION(q,[0 1],0),W,P,O,F,20);
    row = find(Updated.id == 1);
    assert(norm(Updated.xf(row,:)-q) < 1e-12 && Updated.gap(row) < A.gap);
    assert(T.guidedTightenedFeasible == 1 && T.events(1).originalPair);
    assert(T.events(1).afterGap/T.events(1).beforeGap > 0.99);
    % A distinct ordinary result near a guided result is still real evidence.
    nearQ = q+1e-7;
    Union = [SOLUTION(q,[0 1],0),SOLUTION(nearQ,[0 1],0)];
    [Updated,~,T] = PairBoundaryArchive_RC('update',A,[],Union,W,P,O,F,21);
    row = find(Updated.id == 1);
    assert(isequal(Updated.xf(row,:),nearQ) && ...
        any(arrayfun(@(event)event.source == "ordinary",T.events)));
    % Migration changes the input condition, preserving the lineage ID.
    A.ref = 3;
    [Migrated,~] = PairBoundaryArchive_RC('update',A,[],[], ...
        [0 1;0.5 0.5;1 0],P,O,struct(),30);
    assert(any(Migrated.id == 1) && Migrated.nextId >= 2);
    % Angular nearest-neighbor relations are not symmetric. A middle source
    % must not update all nine pairs through the inverse neighborhood.
    W9 = [linspace(0,1,9)',linspace(1,0,9)'];
    A9 = archive(zeros(9,2),ones(9,2),W9,W9,(1:9)');
    Q9 = SOLUTION([0.25 0.25],W9(5,:),0);
    [~,~,T9] = PairBoundaryArchive_RC('update',A9,[],Q9,W9,P,O,struct(),31);
    assert(isequal(sort([T9.events.pairId]),3:7));
    % Original-pair attribution remains exempt from instantaneous ref drift.
    F9 = struct('childDecs',[0.25 0.25],'childObjs',W9(5,:), ...
        'childCons',0,'matchedPairIds',1);
    [~,~,T9] = PairBoundaryArchive_RC('update',A9,[],Q9,W9,P,O,F9,32);
    assert(T9.events(1).pairId == 1 && T9.events(1).originalPair && ...
        isequal(sort([T9.events(2:end).pairId]),3:7));
    O.pairOnly = true; % Preserve the explicit real-pair control.
    % Native proposals survive selection unchanged, including far-off-band rows.
    A = archive([0.2 0.2],[0.8 0.8],[0 1],[1 0],1);
    Raw = [0 1;1 0;0.5 0.5;NaN 0;-0.1 0.5];
    I = struct('refs',ones(5,1),'pairIds',ones(5,1));
    Scale = struct('minimum',[0 0],'span',[1 1]);
    PairGuideCost_RC('start',P);
    [Q,~,ids,T] = PairBoundaryArchive_RC('selectcandidates',Raw,I,A,Scale,P,O);
    Cost = PairGuideCost_RC('snapshot'); PairGuideCost_RC('stop');
    assert(size(Q,1) == 2 && all(ids == 1) && Cost.CalObjRows == 0 && ...
        Cost.CalConRows == 0 && T.objectiveFE == 0);
    assert(isequal(Q,Raw(T.keepIdx,:)), ...
        'Selection must not project or otherwise move a native CGAN proposal.');
    assert(isequal(Q,Raw(1:2,:)) && ~any(T.nativeInBand(T.keepIdx)) && T.invalidCount == 2);
    assert(isequaln(T.candidateDecs,Raw) && ~isfield(T,'constructedDecs') && ...
        ~isfield(T,'correction'));
    assert(all(Q >= 0 & Q <= 1,'all'));
    assert(all(T.coarseInterval(1:3)) && ~any(T.boundaryCertified));
    Narrow = archive([0.2 0.2],[0.20001 0.20001],[0 1],[1 0],1);
    [~,~,~,TN] = PairBoundaryArchive_RC('selectcandidates',[0.200005 0.200005], ...
        struct('refs',1,'pairIds',1),Narrow,Scale,P,O);
    assert(TN.boundaryCertified && ~TN.coarseInterval && ...
        TN.boundaryDistanceUpperBoundRMS <= 0.003);
    [Far,~,~,TF] = PairBoundaryArchive_RC('selectcandidates',[0.9 0.9], ...
        struct('refs',1,'pairIds',1),Narrow,Scale,P,O);
    assert(isequal(Far,[0.9 0.9]) && ~TF.boundaryCertified && TF.coarseInterval, ...
        'A tiny source gap must not certify a distant native candidate.');
    % Round-robin covers pairs first, with a strict cap of two per pair.
    refs = (1:8)';
    xF = [linspace(0.05,0.4,8)',0.1*ones(8,1)];
    xI = xF+0.4;
    weights = [linspace(0.05,0.95,8)',linspace(0.95,0.05,8)'];
    A = archive(xF,xI,weights,weights,refs);
    [Data,Gate] = PairBoundaryArchive_RC('trainingdata',A,weights,P,O);
    rng(21);
    [C,I] = PairBoundaryArchive_RC('querycontexts',A,weights,O,500);
    assert(size(C,1) == 500 && all(C(:,end) == 0));
    [~,rows] = ismember(I.pairIds,A.id);
    Raw = A.xf(rows,:)+(0.4+0.2*rand(500,1)).*(A.xi(rows,:)-A.xf(rows,:));
    [Q,~,ids] = PairBoundaryArchive_RC('selectcandidates',Raw,I,A,Scale,P,O);
    assert(size(Q,1) == 16 && numel(unique(ids(1:8))) == 8 && ...
        all(arrayfun(@(id)nnz(ids == id),unique(ids)) == 2));
    % Two generator updates, not two epochs; both sides share z in sampling.
    [M,S] = PairBoundaryWGAN_RC('trainifneeded',[],Data,Gate,P,O);
    assert(M.ready && S.updates == 2 && S.criticUpdates == 2 && ...
        S.pairVisits == 0 && S.endpointVisits == 32 && S.trainingSamples == 16 && isfinite(S.preDiagnostics.allEndpointRMSE));
    [Raw,Sample] = PairBoundaryWGAN_RC('sample',M,C,P,O);
    expected = P.lower+Sample.normalized.*(P.upper-P.lower);
    assert(isequal(Raw,expected) && Sample.endpointForwardRows == 500 && ...
        size(Sample.z,1) == 500 && isequal(Sample.sides,C(:,end)));
    assert(~isfield(Sample,'alpha') && ~isfield(Sample,'generatedF'));
    O.pairOnly = false; O.W = [weights;0.123 0.877];
    [C,I] = PairBoundaryArchive_RC('querycontexts',A,O.W,O,500);
    assert(nnz(C(:,end)==1)==250 && nnz(C(:,end)==0)==250 && ...
        numel(unique(I.refs))==9 && all(I.pairIds==0));
    Raw = rand(500,2);
    [Q,refs,ids,T] = PairBoundaryArchive_RC('selectcandidates',Raw,I,A,Scale,P,O);
    assert(size(Q,1)==20 && all(ids==0));
    assert(isequal(Q,Raw(T.keepIdx,:)) && all(isnan(T.nativeInBand)));
    assert(all(T.requestedUncovered(T.keepIdx)), ...
        'Enough valid uncovered requests must take priority over known requests.');
    F.matchedPairIds = 0;
    [~,~,T] = PairBoundaryArchive_RC('update',archive([.2 .2],[.8 .8],[0 1],[1 0],1), ...
        [],SOLUTION(q,[0 1],0),[1 1],P,O,F,33);
    assert(T.guidedTightenedFeasible>0 && ~any([T.events.originalPair]));
    % A shared real endpoint is one sample, and owns its own direction label.
    Shared = A; Shared.xi(:,:) = repmat(A.xi(1,:),8,1);
    Shared.yi(:,:) = repmat(A.yi(1,:),8,1);
    [SharedData,SharedGate] = PairBoundaryArchive_RC('trainingdata',Shared,weights,P,O);
    assert(size(unique(SharedData.cI,'rows'),1)==1);
    rng(71);
    [~,SharedStatus] = PairBoundaryWGAN_RC('trainifneeded',[],SharedData,SharedGate,P,O);
    assert(SharedStatus.trainingSamples==9 && SharedStatus.pairVisits==0);
    % Binary conditions are required; intermediate side=0.5 has no meaning.
    bad = C; bad(1,end)=0.5; rejected=false;
    try
        PairBoundaryWGAN_RC('sample',M,bad,P,O);
    catch err
        rejected=strcmp(err.identifier,'CBSPairGuide:BadQueryCondition');
    end
    assert(rejected);
    % ID-only changes are zero; accumulated content and generation triggers differ.
    Same = Data; Same.id = Same.id+1000; O.generation = 2;
    [~,S] = PairBoundaryWGAN_RC('trainifneeded',M,Same,Gate,P,O);
    assert(~S.trained && S.useModel && S.contentChange == 0 && S.reason == "current");
    O.generation = 11;
    [~,S] = PairBoundaryWGAN_RC('trainifneeded',M,Same,Gate,P,O);
    assert(S.trained && S.trigger == "generation" && S.updates == 1);
    Changed = Data; Changed.xI(:,1) = Changed.xI(:,1)-0.2;
    Changed.delta = Changed.xI-Changed.xF; O.generation = 2;
    [~,S] = PairBoundaryWGAN_RC('trainifneeded',M,Changed,Gate,P,O);
    assert(S.trained && S.trigger == "content" && S.contentChange >= 0.2);
    One = Data;
    fields = {'xF','xI','w','ref','id','delta','cF','cI'};
    for k=1:numel(fields); One.(fields{k}) = One.(fields{k})(1,:); end
    One.count = 1;
    [~,S] = PairBoundaryWGAN_RC('trainifneeded',M,One,struct('eligible',false),P,O);
    assert(S.useModel && ~S.trained);
    % Real full evaluation counts nested objectives transparently.
    PairGuideCost_RC('start',P);
    P.Evaluation([0.1 0.2;0.3 0.4]);
    Cost = PairGuideCost_RC('snapshot'); PairGuideCost_RC('stop');
    assert(Cost.CalObjRows == 4 && Cost.CalConRows == 2 && ...
        Cost.CalObjBatches == 2 && Cost.CalConBatches == 1);
    fprintf('PairGuide final mechanism contract passed.\n');
end

function A = archive(xf,xi,yf,yi,refs)
    n = size(xf,1);
    A = struct('id',(1:n)','ref',refs(:),'xf',xf,'xi',xi,'yf',yf,'yi',yi, ...
        'gap',vecnorm(xi-xf,2,2),'active',true(n,1),'nextId',n+1);
end
