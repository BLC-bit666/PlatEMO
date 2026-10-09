function test_GANCMO_mainline
% Runtime contracts: direct DE/G offspring, FE boundaries, and archive updates.
    root=fileparts(fileparts(mfilename('fullpath'))); oldPath=path; oldRng=rng;
    clean=onCleanup(@()restore(oldPath,oldRng));
    addpath(fullfile(root,'Algorithms'),fullfile(root,'Algorithms','Utility functions'), ...
        fullfile(root,'Algorithms','Multi-objective optimization','GANCMO'), ...
        fullfile(root,'Problems'),fullfile(root,'Problems','Multi-objective optimization','LIR-CMOP_BC'));
    rng(123,'twister'); P=LIRCMOP5_BC('N',100,'D',30,'maxFE',10000);
    P1=P.Initialization(); P2=P.Initialization();
    F1=GANCMOFitness(P1.objs,P1.cons); F2=GANCMOFitness(P2.objs);
    for counts=[50 50 0;50 50 45;5 4 3;1 0 1;0 0 0]'
        n1=counts(1); n2=counts(2); m=counts(3);
        q=P.lower+(.3+.4*rand(m,P.D)).*(P.upper-P.lower); before=P.FE; beforeRng=rng;
        [O1,O2,G]=GANCMOOffspring(P,P1,P2,F1,F2,n1,n2,q,17,3);
        assert(P.FE-before==n1+n2 && isequal(rng,beforeRng));
        assert(numel(O1)==n1 && numel(O2)==n2 && numel(G)==m);
        if m>0
            assert(isequal(G.decs,q) && isequal(G,O1(n1-m+1:n1)));
            [~,plain]=GANCMOOffspring(P,P1,P2,F1,F2,n1,n2,zeros(0,P.D),17,3);
            assert(isequal(O2.decs,plain.decs));
        end
    end
    [progress,quota]=GANCMOControl(200,200,150000,50); assert(progress==0 && quota==5);
    [progress,quota]=GANCMOControl(75100,200,150000,50); assert(progress==.5 && quota==25);
    for budget=[24 25 73]
        recorded=[]; state=[];
        for record=[true false]
            rng(1,'twister'); P=LIRCMOP5_BC('N',12,'D',30,'maxFE',budget,'maxRuntime',Inf);
            A=GANCMO('save',-2,'outputFcn',@(varargin)[]); A.RecordTrace=record; A.Solve(P);
            assert(P.FE==budget && A.result{end,1}==budget && numel(A.result{end,2})==P.N);
            if record; recorded=A.result{end,2}.decs; state=rng;
            else; assert(isequal(recorded,A.result{end,2}.decs) && isequal(state,rng)); end
        end
    end
    S=GANCMOSettings; [M,I]=GANCMOModel.train([],struct('count',7),30,1,S);
    assert(isempty(M) && ~I.attempted,'The initial gate must be eight pairs.');
    % Empty direction, a shorter I that removes the old same-F pair from the
    % pair front, an exact tie, then a longer reconstruction after invalidation.
    W=[1 0;0 1]; fixed=struct('minimum',[0 0],'span',[1 1]);
    P=struct('D',2,'M',2,'lower',[0 0],'upper',[1 1]);
    pop=SOLUTION([.2 .2;.8 .8],[1 1;.9 .9],[0;1]);
    [A,scale]=GANCMOArchive.update([],pop,W,P,fixed,pop([]));
    assert(nnz(A.active)==1 && A.transition.action(1)==1 && isequal(scale,fixed));
    old=A.id(A.active); near=SOLUTION([.7 .7],[.95 .95],1);
    B=GANCMOArchive.update(A,near,W,P,fixed,pop([]));
    assert(B.transition.action(1)==4 && B.id(B.active)~=old && ...
        B.transition.newGap2(1)<B.transition.oldGap2(1));
    assert(isequal(B.xf(B.active,:),[.2 .2]) && isequal(B.xi(B.active,:),[.7 .7]));
    tie=SOLUTION([.6 .6],[.95 .95],1);
    C=GANCMOArchive.update(B,tie,W,P,fixed,pop([]));
    assert(C.transition.action(1)==3 && C.id(C.active)==B.id(B.active));
    strong=SOLUTION([.1 .1;.85 .85],[.5 .5;.3 .3],[0;1]);
    D=GANCMOArchive.update(C,strong,W,P,fixed,pop([]));
    assert(D.transition.action(1)==4 && D.transition.newGap2(1)>D.transition.oldGap2(1));
    data=GANCMOArchive.training(D,W,scale,P);
    assert(data.count==1 && all(data.cF(:,end)==1) && all(data.cI(:,end)==0));
    testArchiveRules;
    testCandidateSelection(S);
    % Joint-input training at two decision and objective dimensions.
    % Production settings cannot be overridden; these are short component runs.
    for shape=[3 2;3 3;30 2;30 3]'
        dimensions=shape(1); objectives=shape(2);
        rng(915,'twister'); W=rand(8,objectives);
        data=struct('count',8,'ref',(1:8)','xF',rand(8,dimensions),'xI',rand(8,dimensions), ...
            'cF',[W,ones(8,1)],'cI',[W,zeros(8,1)]);
        P=struct('D',dimensions,'M',objectives,'lower',zeros(1,dimensions),'upper',ones(1,dimensions));
        component=S; component.initialG=2; component.updateG=1;
        [M,I]=GANCMOModel.train([],data,P.D,1,component);
        assert(I.updatesG==2 && I.updatesC==4 && M.ready && ~I.failed);
        assert(I.batchSize==2*data.count);
        assert(M.netG.Layers(1).InputSize==P.D+P.M+1 && M.netC.Layers(1).InputSize==P.D+P.M+1);
        assert(isequal(size(M.netG.Learnables.Value{1}),[S.generatorHidden P.D+P.M+1]), ...
            'Noise and condition must both enter the first learned layer.');
        assert(all(M.lastData.cF(:,end)==1) && all(M.lastData.cI(:,end)==0));
        C=[data.cF;data.cI]; rng(916,'twister'); X=GANCMOModel.sample(M,C,P);
        assert(isequal(size(X),[16 P.D]) && all(isfinite(X),'all') && all(X>=0 & X<=1,'all'));
        % The network graph independently checks the direct joint-input path.
        rng(916,'twister'); referenceZ=S.noiseSigma*randn(size(C,1),P.D);
        referenceC=C; referenceC(:,1:end-1)=C(:,1:end-1)./max(vecnorm(C(:,1:end-1),2,2),eps);
        reference=forward(M.netG,dlarray(single([referenceZ,referenceC]'),'CB'));
        expected=min(1,max(0,(double(extractdata(reference))'+1)/2));
        assert(isequal(X,expected),'Direct generator forward differs from the network graph.');
        beforeScore=rng; scores=GANCMOModel.score(M,X,C,P);
        reference=forward(M.netC,dlarray(single([2*X-1,referenceC]'),'CB'));
        assert(isequal(scores,double(extractdata(reference))') && isequal(rng,beforeScore), ...
            'Critic scoring must match the trained graph and consume no random numbers.');
        C(:,1:end-1)=7*C(:,1:end-1); rng(916,'twister'); ray=GANCMOModel.sample(M,C,P);
        assert(max(abs(X-ray),[],'all')<1e-6);
        % Both true sides must have a nonlinear, differentiable noise path.
        % An affine noise term added to the output has a constant Jacobian.
        for side=[0 1]
            condition=referenceC(1,:)'; condition(end)=side;
            condition=dlarray(single(condition),'CB');
            g0=dlfeval(@noiseGradient,M.netG,dlarray(zeros(P.D,1,'single'),'CB'),condition);
            g1=dlfeval(@noiseGradient,M.netG,dlarray(.5*ones(P.D,1,'single'),'CB'),condition);
            assert(all(isfinite([g0;g1])) && all(abs(g0)>1e-8));
            assert(norm(g1-g0)>1e-5,'Noise must pass through the nonlinear hidden layer on both sides.');
        end
        [same,skipped]=GANCMOModel.train(M,data,P.D,2,component);
        assert(~skipped.attempted && same.iterG==M.iterG);
        beforeNoise=extractdata(M.netG.Learnables.Value{1}(:,1:P.D));
        [continued,updated]=GANCMOModel.train(M,data,P.D,21,component);
        afterNoise=extractdata(continued.netG.Learnables.Value{1}(:,1:P.D));
        assert(updated.updatesG==1 && updated.updatesC==2 && continued.iterG==M.iterG+1);
        assert(any(beforeNoise~=afterNoise,'all'),'Continuation must train the noise input weights.');
    end
    fprintf('PASS GANCMO: joint nonlinear noise input, continuation, direct offspring, independent P2, exact FE/RNG, gate=8 and relative archive replacement\n');
end

function testArchiveRules
% Endpoint eligibility is independent of P1 membership; pair fronts always apply.
    P=struct('D',2,'M',2,'lower',[0 0],'upper',[1 1]);
    fixed=struct('minimum',[0 0],'span',[1 1]); W=[1 2;2 3];
    pool=SOLUTION([.1 .1;.2 .2;.7 .7;.8 .8], ...
        [1 2;2 3;.5 3;1.01 2.01],[0;0;1;1]);
    A=GANCMOArchive.update([],pool,W,P,fixed,pool(1:2));
    assert(numel(A.id)==1 && isequal(A.xf,pool(1).dec), ...
        'A dominated F must remain ineligible even when it belongs to P1.');
    assert(isequal(A.xi,pool(3).dec), ...
        'An I dominated by the feasible front must be excluded, even if closer.');
    % The pair front removes a worse-quality backup but preserves the shorter
    % non-P1 pair as a tradeoff; nearest distance still decides the active pair.
    W=[.5 .5];
    pool=SOLUTION([.1 .1;.2 .2;.3 .3;.6 .1;.6 .2;.6 .3], ...
        [1 3;2 2;3 1;.9 3;1.5 2;2.5 1],[0;0;0;1;1;1]);
    A=GANCMOArchive.update([],pool,W,P,fixed,pool(2:3));
    assert(numel(A.id)==2 && all(ismember(pool(1:2).decs,A.xf,'rows')) && ...
        ~ismember(pool(3).dec,A.xf,'rows'),'Pair-front filtering must apply from initialization.');
    assert(isequal(A.xf(A.active,:),pool(1).dec), ...
        'A shorter non-P1 pair must survive a longer current-P1 pair.');
    old=GANCMOArchive.update([],pool([2 5]),W,P,fixed,pool(2));
    B=GANCMOArchive.update(old,pool,W,P,fixed,pool(2:3));
    assert(B.transition.action==2 && B.transition.oldLegal && any(B.id==old.id) && ...
        B.transition.newGap2<B.transition.oldGap2, ...
        'A surviving incumbent must only be replaced by a strictly shorter pair.');
    % Equal-distance, equal-quality pairs differ only in current-P1 membership.
    % The old pair becomes ineligible at the pair-front step, so this is a rebuild.
    first=SOLUTION([.1 .1;.6 .1],[1 3;.5 3],[0;1]);
    A=GANCMOArchive.update([],first,W,P,fixed,first(1));
    next=SOLUTION([.3 .3;.6 .3],[3 1;2.5 1],[0;1]);
    B=GANCMOArchive.update(A,next,W,P,fixed,next(1));
    assert(numel(B.id)==1 && isequal(B.xf,next(1).dec) && ~ismember(A.id,B.id));
    assert(B.transition.action==4 && ~B.transition.oldLegal && ...
        B.transition.newGap2==B.transition.oldGap2, ...
        'Pair-front invalidation must precede the incumbent tie rule.');
    fprintf('PASS unified archive: strict F/I eligibility, pair fronts from initialization, distance priority and P1 tie rebuild\n');
end

function testCandidateSelection(S)
% A struct problem has no objective, constraint or Evaluation methods.
% Controlled networks test both ranking paths without any oracle queries.
    P=struct('D',2,'M',2,'lower',[-2 10],'upper',[2 30]);
    W=[1 0;0 1;.5 .5]; fixed=struct('minimum',[0 0],'span',[1 1]);
    A=struct('active',[true;true;false],'ref',[1;2;3], ...
        'xf',P.lower+[.25 .375;.375 .5;0 0].*(P.upper-P.lower), ...
        'xi',P.lower+[.75 .5;.5 .875;1 1].*(P.upper-P.lower), ...
        'yf',W,'yi',W);
    Data=GANCMOArchive.training(A,W,fixed,P);
    dummy=SOLUTION(P.lower,[0 0],0); empty=dummy([]);
    % I endpoints expand the box beyond the F-only bounds. Box faces count
    % as inside; these points remain far beyond the removed distance gate.
    inside=[.625 .75;.25 .75;.75 .75;.625 .375;.625 .875];
    for point=inside'
        M=candidateModel(point,S,0); expected=P.lower+point'.*(P.upper-P.lower);
        [q,info]=GANCMOCandidates(M,A,W,empty,empty,P,2,S,Data);
        assert(isequal(q,expected) && info.available==S.candidateCount);
        assert(all(info.distanceF(info.hasActivePair)>1e-4));
        assert(sum(info.queryCounts)==S.candidateCount && max(info.queryCounts)-min(info.queryCounts)==1);
        duplicate=SOLUTION(expected,[1 1],0);
        [q,info]=GANCMOCandidates(M,A,W,duplicate,empty,P,2,S,Data);
        assert(isempty(q) && info.available==0,'Existing-point deduplication must remain active.');
    end
    % The box applies only to active directions. Inactive backup endpoints
    % must not expand it; an unpaired request may retain the native point.
    for point=[.125 .75;.875 .75;.625 .25;.625 1]'
        M=candidateModel(point,S,0);
        [q,info]=GANCMOCandidates(M,A,W,empty,empty,P,2,S,Data);
        expected=P.lower+point'.*(P.upper-P.lower);
        assert(isequal(q,expected) && info.ref==3 && ~info.hasActivePair);
        assert(info.available==info.queryCounts(3) && isfinite(info.criticScore) && isnan(info.distanceF));
    end
    % A newly active direction is queried before the next training event.
    A.active(3)=true; Data=GANCMOArchive.training(A,W,fixed,P);
    M=candidateModel([.875;.75],S,.05);
    [q,info]=GANCMOCandidates(M,A,W,empty,empty,P,3,S,Data);
    assert(size(q,1)==3 && isequal(sort(info.ref),[1;2;3]) && all(info.hasActivePair));
    A.active(3)=false; Data=GANCMOArchive.training(A,W,fixed,P);
    M=candidateModel([.625;.75],S,.05);
    % Replay only query generation, then check the two ranking outcomes
    % independently: F lies below every proposal; this critic increases in x1.
    rng(122,'twister'); directions=randperm(3); order=randperm(S.candidateCount);
    requests=reshape(directions(mod(order-1,3)+1),[],1);
    raw=GANCMOModel.sample(M,[W(requests,:),ones(S.candidateCount,1)],P);
    expected=zeros(3,2);
    for j=1:3
        rows=find(requests==j);
        if j<3; [~,best]=min(raw(rows,1)); else; [~,best]=max(raw(rows,1)); end
        expected(j,:)=raw(rows(best),:);
    end
    rng(122,'twister'); [q,info]=GANCMOCandidates(M,A,W,empty,empty,P,3,S,Data);
    [~,order]=sort(info.ref);
    assert(isequal(q(order,:),expected) && info.available==S.candidateCount && info.diverse==3, ...
        'Paired directions must minimize F distance; unpaired directions must maximize critic score.');
    assert(isequal(info.hasActivePair(order),[true;true;false]));
    assert(all(isnan(info.criticScore(info.hasActivePair))) && info.criticScore(~info.hasActivePair)<0, ...
        'Negative critic scores remain eligible; paired requests do not use critic rank.');
    % Score native decisions using the training normalization and unit rays.
    before=rng; scores=GANCMOModel.score(M,q, ...
        [7*W(info.ref,:),ones(size(q,1),1)],P);
    unit=W(info.ref,:)./vecnorm(W(info.ref,:),2,2);
    normalized=(q-P.lower)./(P.upper-P.lower);
    linear=2*normalized(:,1)-1+10*unit(:,1)+.25;
    expectedScore=max(linear,.2*linear)-20;
    assert(max(abs(scores-expectedScore))<2e-6 && isequal(rng,before));
    % Both branches share one quota. Remainder directions and selected
    % directions must vary; there is no fixed extra-query or exploration cap.
    seen=false(3,1); remainder=false(3,1);
    for seed=1:12
        rng(seed,'twister');
        [q,info]=GANCMOCandidates(M,A,W,empty,empty,P,1,S,Data);
        assert(size(q,1)==1); seen(info.ref)=true;
        remainder(info.queryCounts==floor(S.candidateCount/3))=true;
    end
    assert(all(seen) && all(remainder));
    A.active(:)=false; Data=GANCMOArchive.training(A,W,fixed,P);
    [q,info]=GANCMOCandidates(M,A,W,empty,empty,P,3,S,Data);
    assert(size(q,1)==3 && ~any(info.hasActivePair) && all(info.targetRef==0));
    assert(all(isnan(info.distanceF)) && all(isfinite(info.criticScore)));
    [manyW,~]=UniformPoint(100,2); Data=GANCMOArchive.training(A,manyW,fixed,P);
    [q,info]=GANCMOCandidates(M,A,manyW,empty,empty,P,40,S,Data);
    assert(all(info.queryCounts==5) && size(q,1)==40 && numel(unique(info.ref))==40, ...
        'Unpaired directions must share the whole quota, with at most one point per request direction.');
    % Only an unavailable model or zero quota suppresses queries, not a
    % temporarily inactive archive after successful training.
    M.ready=false; before=rng;
    [q,info]=GANCMOCandidates(M,A,W,empty,empty,P,2,S,Data);
    assert(isempty(q) && info.queried==0 && isequal(rng,before));
    M.ready=true;
    [q,info]=GANCMOCandidates(M,A,W,empty,empty,P,0,S,Data);
    assert(isempty(q) && info.queried==0 && isequal(rng,before));
    fprintf('PASS candidates: uniform all-direction queries, active-only box, F/critic ranking, shared quota, deduplication and no oracle calls\n');
end

function M = candidateModel(point,S,spread)
    D=numel(point); weights=zeros(1,D+3,'single'); weights(1)=1;
    layers=[featureInputLayer(D+3,'Normalization','none','Name','in'); ...
        fullyConnectedLayer(1,'Weights',weights,'Bias',single(0),'Name','hidden'); ...
        functionLayer(@sin,'Formattable',true,'Name','sine'); ...
        fullyConnectedLayer(D,'Weights',single(spread)*ones(D,1,'single'), ...
        'Bias',single(2*point-1),'Name','out')];
    M=struct('ready',true,'lastData',struct('ref',[1;2]), ...
        'settings',S,'netG',dlnetwork(layers));
    weights=zeros(1,D+3,'single'); weights(1)=1; weights(D+1)=10; weights(end)=.25;
    layers=[featureInputLayer(D+3,'Normalization','none','Name','in'); ...
        fullyConnectedLayer(1,'Weights',weights,'Bias',single(0),'Name','hidden'); ...
        leakyReluLayer(.2,'Name','leaky'); ...
        fullyConnectedLayer(1,'Weights',single(1),'Bias',single(-20),'Name','out')];
    M.netC=dlnetwork(layers);
end

function gradient = noiseGradient(netG,Z,C)
    output=forward(netG,[Z;C]);
    gradient=extractdata(dlgradient(sum(output,'all'),Z));
end

function restore(oldPath,oldRng)
    path(oldPath); rng(oldRng);
end
