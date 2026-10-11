function test_GANCMO_mainline
% Runtime contracts: direct DE/G offspring, FE boundaries, and archive updates.
    root=fileparts(fileparts(mfilename('fullpath'))); oldPath=path; oldRng=rng;
    clean=onCleanup(@()restore(oldPath,oldRng));
    addpath(fullfile(root,'Algorithms'),fullfile(root,'Algorithms','Utility functions'), ...
        fullfile(root,'Algorithms','Multi-objective optimization','GANCMO'), ...
        fullfile(root,'Problems'),fullfile(root,'Problems','Multi-objective optimization','LIR-CMOP_BC'));
    rng(123,'twister'); P=LIRCMOP7_BC('N',100,'D',30,'maxFE',10000);
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
        for repeat=1:2
            rng(1,'twister'); P=LIRCMOP7_BC('N',12,'D',30,'maxFE',budget,'maxRuntime',Inf);
            A=GANCMO('save',-2,'outputFcn',@(varargin)[]); A.Solve(P);
            assert(P.FE==budget && A.result{end,1}==budget && numel(A.result{end,2})==P.N);
            if repeat==1; recorded=A.result{end,2}.decs; state=rng;
            else; assert(isequal(recorded,A.result{end,2}.decs) && isequal(state,rng)); end
        end
    end
    S=GANCMOSettings; before=rng;
    for count=[0 7]
        data=struct('count',5*count,'ref',repelem((1:count)',5));
        M=GANCMOModel.train([],data,30,1,S);
        assert(isempty(M) && isequal(rng,before), ...
            'Even 35 pairs on seven F directions must not start training or consume RNG.');
    end
    % Each stored pair is legal data; guides only choose candidate geometry.
    W=[1 0;0 1]; fixed=struct('minimum',[0 0],'span',[1 1]);
    P=struct('N',100,'D',2,'M',2,'lower',[0 0],'upper',[1 1]);
    pop=SOLUTION([.2 .2;.8 .8],[1 1;.9 .9],[0;1]);
    [A,scale,gA]=GANCMOArchive.update([],pop,W,P,fixed);
    assert(numel(gA)==1 && isequal(scale,fixed));
    near=SOLUTION([.7 .7],[.95 .95],1);
    [B,~,gB]=GANCMOArchive.update(A,near,W,P,fixed);
    assert(isequal(B.xf(gB,:),[.2 .2]) && isequal(B.xi(gB,:),[.7 .7]));
    tie=SOLUTION([.6 .6],[.95 .95],1);
    [C,~,gC]=GANCMOArchive.update(B,tie,W,P,fixed);
    assert(isequal(C.xi(gC,:),[.6 .6]),'Equal-distance I ties use decision order, not the old pair.');
    strong=SOLUTION([.1 .1;.85 .85],[.5 .5;.3 .3],[0;1]);
    [D,~,gD]=GANCMOArchive.update(C,strong,W,P,fixed);
    assert(isequal(D.xf(gD,:),[.1 .1]) && isequal(D.xi(gD,:),[.85 .85]));
    data=GANCMOArchive.training(D,W,scale,P);
    assert(data.count==numel(D.ref) && all(data.cF(:,end)==1) && all(data.cI(:,end)==0));
    assert(isequal(sort(fieldnames(D)),sort({'ref';'xf';'yf';'xi';'yi'})));
    testArchiveRules;
    testAllPairTriggers(S);
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
        M=GANCMOModel.train([],data,P.D,1,component);
        assert(M.iterG==2 && M.iterC==4);
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
        same=GANCMOModel.train(M,data,P.D,2,component);
        assert(same.iterG==M.iterG && same.lastTrainGeneration==M.lastTrainGeneration);
        beforeNoise=extractdata(M.netG.Learnables.Value{1}(:,1:P.D));
        continued=GANCMOModel.train(M,data,P.D,21,component);
        afterNoise=extractdata(continued.netG.Learnables.Value{1}(:,1:P.D));
        assert(continued.iterG==M.iterG+1 && continued.iterC==M.iterC+2);
        assert(any(beforeNoise~=afterNoise,'all'),'Continuation must train the noise input weights.');
    end
    fprintf('PASS GANCMO: joint nonlinear noise input, continuation, direct offspring, independent P2, exact FE/RNG, eight-direction initial gate, quality-first guides and all-pair training\n');
end

function testArchiveRules
% Endpoint eligibility, pair-independent rebuilding and hard direction limits.
    P=struct('N',100,'D',2,'M',2,'lower',[0 0],'upper',[1 1]);
    fixed=struct('minimum',[0 0],'span',[1 1]); W=[1 2;2 3];
    pool=SOLUTION([.1 .1;.2 .2;.7 .7;.8 .8], ...
        [1 2;2 3;.5 3;1.01 2.01],[0;0;1;1]);
    A=GANCMOArchive.update([],pool,W,P,fixed);
    assert(numel(A.ref)==1 && isequal(A.xf,pool(1).dec), ...
        'A dominated feasible endpoint is ineligible.');
    assert(isequal(A.xi,pool(3).dec),'An I dominated by the feasible front is excluded.');
    W=[.5 .5];
    pool=SOLUTION([.1 .1;.2 .2;.3 .3;.6 .1;.6 .2;.6 .3], ...
        [1 3;2 2;3 1;.9 3;1.5 2;2.5 1],[0;0;0;1;1;1]);
    [A,~,guide]=GANCMOArchive.update([],pool,W,P,fixed);
    assert(numel(A.ref)==3 && isequal(A.xf,pool(1:3).decs), ...
        'All nondominated F endpoints fit; pair-front filtering must not delete them.');
    assert(isequal(A.xf(guide,:),pool(2).dec), ...
        'A better F wins even when another legal pair has a smaller gap.');
    allData=GANCMOArchive.training(A,W,fixed,P);
    guideData=GANCMOArchive.training(A,W,fixed,P,guide);
    assert(allData.count==3 && guideData.count==1 && isequal(allData.xF,A.xf));
    % Alter only old pair links, row order and old direction identifiers.
    shuffled=A; shuffled.xi=A.xi(end:-1:1,:); shuffled.yi=A.yi(end:-1:1,:);
    shuffled.ref(:)=999; before=rng;
    [rebuilt,~,rebuiltGuide]=GANCMOArchive.update(shuffled,pool([]),W,P,fixed);
    [permuted,~,permutedGuide]=GANCMOArchive.update([],pool(end:-1:1),W,P,fixed);
    assert(isequal(rebuilt,A) && isequal(permuted,A) && ...
        isequal(rebuiltGuide,guide) && isequal(permutedGuide,guide) && isequal(rng,before), ...
        'The endpoint pool, not old pair identities or input order, determines the archive.');
    % Equal F quality: objective gap breaks the tie.
    tied=SOLUTION([.1 .1;.3 .3;.6 .1;.6 .3],[1 3;3 1;.8 3;2.5 1],[0;0;1;1]);
    [B,~,gB]=GANCMOArchive.update([],tied,W,P,fixed);
    assert(isequal(B.xf(gB,:),tied(1).dec));
    % The same eligible I may support distinct F endpoints.
    shared=SOLUTION([.1 .1;.3 .3;.6 .1],[1 3;3 1;.5 .5],[0;0;1]);
    B=GANCMOArchive.update([],shared,W,P,fixed);
    assert(numel(B.ref)==2 && isequal(B.xi,repmat(shared(3).dec,2,1)));
    % Invalid stored endpoints cannot bypass numerical/bound checks.
    broken=A; broken.xf(1,1)=NaN; broken.xi(2,1)=2;
    B=GANCMOArchive.update(broken,pool([]),W,P,fixed);
    assert(all(isfinite([B.xf,B.yf,B.xi,B.yi]),'all') && ...
        all([B.xf,B.xi]>=0 & [B.xf,B.xi]<=1,'all'));
    % A crowded direction is capped at five even when global space is available.
    n=600; k=(1:n)'; xf=[k/(n+1),.25*ones(n,1)]; xi=[k/(n+1),.75*ones(n,1)];
    yf=[k,n+1-k]; yi=yf-[1e-4*k,zeros(n,1)];
    pool=SOLUTION([xf;xi],[yf;yi],[zeros(n,1);ones(n,1)]); W=[1 .001];
    [A,~,guide]=GANCMOArchive.update([],pool,W,P,fixed);
    assert(numel(A.ref)==5 && numel(guide)==1 && isequal(A.xf,xf(end-4:end,:)) && ...
        isequal(A.xf(guide,:),xf(end,:)));
    data=GANCMOArchive.training(A,W,fixed,P); assert(data.count==5);
    % Six feasible endpoints in every direction: exactly five survive per ray.
    for shape=[2 3;12 2;12 3;100 2;100 3]'
        P.N=shape(1); P.M=shape(2); [W,~]=UniformPoint(P.N,P.M); R=size(W,1);
        frame=struct('minimum',zeros(1,P.M),'span',ones(1,P.M));
        yf=repelem(W,6,1); offset=repmat((1:6)',R,1)*1e-9;
        yf(:,1)=yf(:,1)+offset; yf(:,2)=yf(:,2)-offset; yi=yf-1e-7;
        n=size(yf,1); xf=[(1:n)'/(n+1),.25*ones(n,1)]; xi=[xf(:,1),.75*ones(n,1)];
        pool=SOLUTION([xf;xi],[yf;yi],[zeros(n,1);ones(n,1)]);
        [A,~,guide]=GANCMOArchive.update([],pool,W,P,frame);
        assert(numel(A.ref)==5*R && numel(guide)==R && ...
            all(accumarray(A.ref,1,[R 1])==5));
        if R>P.N
            assert(numel(A.ref)>5*P.N,'Actual direction count sets capacity; no separate 5N cap.');
        end
        sparse=GANCMOArchive.update([],pool([1:6,n+(1:6)]),W,P,frame);
        assert(numel(sparse.ref)==5 && numel(unique(sparse.ref))==1, ...
            'Empty directions must not lend slots to a crowded direction.');
        if P.N==100 && P.M==3; assert(numel(A.ref)==455); end
        if P.N==100 && P.M==2
            moved=frame; moved.minimum=[-1e6 -2e6];
            merged=GANCMOArchive.update(A,pool([]),W,P,moved);
            assert(numel(merged.ref)==5 && numel(unique(merged.ref))==1, ...
                'The hard limit must also hold after many directions merge under a new origin.');
        end
    end
    fprintf('PASS archive: F/I eligibility, rebuilt pairs, canonical ties, all-pair training and hard five-per-direction limit\n');
end

function testAllPairTriggers(S)
% Gates count occupied F directions; all pairs still train and trigger updates.
    D=3; W=UniformPoint(8,2); refs=repelem((1:8)',3);
    x=repmat(linspace(.05,.85,24)',1,D);
    data=struct('count',24,'ref',refs,'xF',x,'xI',x+.005, ...
        'cF',[W(refs,:),ones(24,1)],'cI',repmat([W(1,:),0],24,1));
    S.initialG=1; S.updateG=1;
    model=GANCMOModel.train([],data,D,1,S);
    assert(model.iterG==1 && model.iterC==2);
    assert(isequal(model.lastData,data));
    same=GANCMOModel.train(model,data,D,2,S); assert(same.iterG==model.iterG);
    changed=data; nonGuide=find(mod((1:24)',3)~=1);
    changed.xF(nonGuide,:)=changed.xF(nonGuide,:)+.025;
    changed.xI(nonGuide,:)=changed.xI(nonGuide,:)+.025;
    updated=GANCMOModel.train(model,changed,D,2,S);
    assert(updated.iterG==model.iterG+1 && updated.lastTrainGeneration==2 && isequal(updated.lastData,changed), ...
        'Changes to non-guide pairs must trigger continuation before the interval.');
    % Moving the origin can relabel a retained I while both condition groups
    % remain present. Endpoint geometry and the condition set alone miss this.
    labelled=data; labelled.cI(3:3:24,:)=repmat([W(2,:),0],8,1);
    labelModel=GANCMOModel.train([],labelled,D,1,S); assert(~isempty(labelModel));
    reordered=labelled;
    for field={'ref','xF','xI','cF','cI'}
        reordered.(field{1})=reordered.(field{1})(end:-1:1,:);
    end
    same=GANCMOModel.train(labelModel,reordered,D,2,S);
    assert(same.iterG==labelModel.iterG,'Row order must not count as an endpoint condition change.');
    relabelled=labelled; relabelled.cI([1 4],:)=repmat(labelled.cI(3,:),2,1);
    assert(isequal(unique(labelled.cI,'rows'),unique(relabelled.cI,'rows')));
    updated=GANCMOModel.train(labelModel,relabelled,D,2,S);
    assert(updated.iterG==labelModel.iterG+1 && isequal(updated.lastData,relabelled) && ...
        isequal(labelModel.lastData,labelled), ...
        'Retained endpoint relabelling must trigger training without rewriting its old snapshot.');
    for count=[1 2 7]
        small=data; rows=data.ref<=count;
        for field={'ref','xF','xI','cF','cI'}
            small.(field{1})=data.(field{1})(rows,:);
        end
        small.count=nnz(rows);
        before=rng; same=GANCMOModel.train(model,small,D,21,S);
        assert(isequal(same,model) && isequal(rng,before), ...
            'Fewer than eight F directions must preserve the model, Adam and RNG even after 20 generations.');
    end
    P=struct('D',D,'M',2,'lower',zeros(1,D),'upper',ones(1,D));
    before=rng; expected=GANCMOModel.sample(model,data.cF,P); expectedRng=rng;
    rng(before); actual=GANCMOModel.sample(same,data.cF,P);
    assert(isequal(expected,actual) && isequal(rng,expectedRng), ...
        'A model skipped by the direction gate must remain usable for identical queries.');
    for field={'ref','xF','xI','cF','cI'}; data.(field{1})=data.(field{1})(1:3:24,:); end
    data.count=8;
    updated=GANCMOModel.train(model,data,D,21,S);
    assert(updated.iterG==model.iterG+1 && isequal(updated.lastData,data), ...
        'Eight pairs on eight distinct F directions permit continuation.');
    before=rng; same=GANCMOModel.train(updated,data,D,22,S);
    assert(isequal(same,updated) && isequal(rng,before));
    continued=GANCMOModel.train(updated,data,D,41,S);
    assert(continued.iterG==updated.iterG+1 && continued.lastTrainGeneration==41, ...
        'Unchanged data on eight F directions still train after the 20-generation interval.');
    fprintf('PASS training: eight/eight-direction gates, all-pair geometric/condition changes and immutable training snapshots\n');
end

function testCandidateSelection(S)
% A struct problem has no evaluator. Check decisions against replayed queries.
    P=struct('D',2,'M',2,'lower',[-2 10],'upper',[2 30]);
    W=[1 0;0 1;.5 .5]; fixed=struct('minimum',[0 0],'span',[1 1]);
    A=struct('ref',[1;2;1], ...
        'xf',P.lower+[.25 .375;.375 .5;0 0].*(P.upper-P.lower), ...
        'xi',P.lower+[.75 .5;.5 .875;1 1].*(P.upper-P.lower), ...
        'yf',W([1 2 1],:),'yi',W);
    Data=GANCMOArchive.training(A,W,fixed,P,[1;2]);
    dummy=SOLUTION(P.lower,[0 0],0); empty=dummy([]);
    % Box faces remain valid. Repeated samples give exactly one decision.
    for point=[.625 .75;.25 .75;.75 .75;.625 .375;.625 .875]'
        M=candidateModel(point,S,0); expected=P.lower+point'.*(P.upper-P.lower);
        q=GANCMOCandidates(M,A,W,empty,empty,P,2,S,Data);
        assert(isequal(q,expected));
        duplicate=SOLUTION(expected,[1 1],0);
        q=GANCMOCandidates(M,A,W,duplicate,empty,P,2,S,Data);
        assert(isempty(q),'Existing-point deduplication must remain active.');
    end
    % The unpaired direction retains native proposals outside the guide box.
    for point=[.125 .75;.875 .75;.625 .25;.625 1]'
        M=candidateModel(point,S,0);
        q=GANCMOCandidates(M,A,W,empty,empty,P,2,S,Data);
        assert(isequal(q,P.lower+point'.*(P.upper-P.lower)));
    end
    % A newly represented direction is usable before another training event.
    A.ref(3)=3; A.yf(3,:)=W(3,:); Data=GANCMOArchive.training(A,W,fixed,P,[1;2;3]);
    M=candidateModel([.875;.75],S,.05);
    q=GANCMOCandidates(M,A,W,empty,empty,P,3,S,Data); assert(size(q,1)==3);
    A.ref(3)=1; A.yf(3,:)=W(1,:); Data=GANCMOArchive.training(A,W,fixed,P,[1;2]);
    M=candidateModel([.625;.75],S,.05);
    % Replay the uniformly allocated queries independently of the selector.
    % F lies below every proposal; the unpaired critic increases in x1.
    rng(122,'twister'); directions=randperm(3); order=randperm(S.candidateCount);
    requests=reshape(directions(mod(order-1,3)+1),[],1);
    raw=GANCMOModel.sample(M,[W(requests,:),ones(S.candidateCount,1)],P);
    expected=zeros(3,2);
    for j=1:3
        rows=find(requests==j);
        if j<3; [~,best]=min(raw(rows,1)); else; [~,best]=max(raw(rows,1)); end
        expected(j,:)=raw(rows(best),:);
    end
    rng(122,'twister'); q=GANCMOCandidates(M,A,W,empty,empty,P,3,S,Data);
    [found,ref]=ismember(q,expected,'rows');
    assert(all(found) && numel(unique(ref))==3, ...
        'Paired requests minimize F distance; unpaired requests maximize critic score.');
    before=rng; scores=GANCMOModel.score(M,q,[7*W(ref,:),ones(size(q,1),1)],P);
    unit=W(ref,:)./vecnorm(W(ref,:),2,2); normalized=(q-P.lower)./(P.upper-P.lower);
    linear=2*normalized(:,1)-1+10*unit(:,1)+.25;
    assert(max(abs(scores-(max(linear,.2*linear)-20)))<2e-6 && isequal(rng,before));
    assert(all(scores<0),'Negative critic scores remain eligible.');
    % Zero quota/unavailable model consume no randomness. An empty archive
    % does not suppress a trained model or impose an extra exploration cap.
    for field={'ref','xf','yf','xi','yi'}; A.(field{1})=A.(field{1})([], :); end
    [manyW,~]=UniformPoint(100,2); Data=GANCMOArchive.training(A,manyW,fixed,P);
    q=GANCMOCandidates(M,A,manyW,empty,empty,P,40,S,Data);
    assert(size(q,1)==40 && size(unique(q,'rows'),1)==40);
    before=rng;
    q=GANCMOCandidates([],A,manyW,empty,empty,P,2,S,Data);
    assert(isempty(q) && isequal(rng,before));
    q=GANCMOCandidates(M,A,manyW,empty,empty,P,0,S,Data);
    assert(isempty(q) && isequal(rng,before));
    fprintf('PASS candidates: guide box, F/critic ranking, shared quota, deduplication and no oracle calls\n');
end

function M = candidateModel(point,S,spread)
    D=numel(point); weights=zeros(1,D+3,'single'); weights(1)=1;
    layers=[featureInputLayer(D+3,'Normalization','none','Name','in'); ...
        fullyConnectedLayer(1,'Weights',weights,'Bias',single(0),'Name','hidden'); ...
        functionLayer(@sin,'Formattable',true,'Name','sine'); ...
        fullyConnectedLayer(D,'Weights',single(spread)*ones(D,1,'single'), ...
        'Bias',single(2*point-1),'Name','out')];
    M=struct('lastData',struct('ref',[1;2]), ...
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
