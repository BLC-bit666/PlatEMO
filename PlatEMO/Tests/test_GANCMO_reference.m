function test_GANCMO_reference
% Historical ideal points and direction migration under fixed reference rays.
    root=fileparts(fileparts(mfilename('fullpath'))); oldPath=path; oldRng=rng;
    clean=onCleanup(@()restore(oldPath,oldRng));
    addpath(fullfile(root,'Algorithms'),fullfile(root,'Algorithms','Utility functions'), ...
        fullfile(root,'Algorithms','Multi-objective optimization','GANCMO'),fullfile(root,'Problems'));
    testIdealPointRuns;
    testArchiveMigration;
    fprintf('PASS reference: historical ideal, batched updates, endpoint migration and fixed W/N/neighborhoods\n');
end

function testIdealPointRuns
% UserProblem supplies an exact evaluation stream; no search-quality metric.
    for shape=[6 2;100 3]'
        N=shape(1); M=shape(2); budget=3*N+1;
        Y=repmat(10+(1:budget)'/100,1,M); C=zeros(budget,1);
        Y(N+1,1)=4; % The second initial population contributes to the ideal.
        Y(2*N+1,1)=2; C(2*N+1)=1; % An infeasible O1 child lowers it again.
        secondBatch=2*N+ceil(N/2)+1;
        Y(secondBatch,M)=3; C(secondBatch)=1; % O2 is included too.
        [A,P,seen,state]=runSequence(N,Y,C,true);
        assert(P.FE==budget && P.N==N && isequal(seen,Y));
        assert(isequal(A.Frames{end}.scale.minimum,min(Y,[],1)) && all(A.Frames{end}.scale.span==1));
        final=A.result{end,2}; finalY=final.objs;
        assert(all(final.cons<=0,'all') && min(finalY(:,1))>2, ...
            'An infeasible child rejected by P1 must still update the ideal point.');
        [W,~]=UniformPoint(N,M); assert(isequal(A.Frames{end}.W,W));
        if M==3; assert(size(W,1)==91 && P.N==100); end
        [B,~,plain,plainState]=runSequence(N,Y,C,false);
        assert(isequal(plain,Y) && isequal(A.result{end,2}.decs,B.result{end,2}.decs) && ...
            isequal(state,plainState),'Reference diagnostics must not alter FE, RNG or search.');
        % No generation runs when initialization uses the entire budget.
        initial=runSequence(N,Y(1:2*N,:),C(1:2*N),true);
        assert(isequal(initial.Frames{end}.scale.minimum,min(Y(1:2*N,:),[],1)));
    end
    % Crowding removes the sole f1-minimum witness from both populations.
    % It is absent from the next evaluated union and from the empty archive.
    Y=5+[0 5 5;10 0 10;10 10 0;10 0 10;10 10 0;20 20 20; ...
        .1 4.9 5.1;.1 5.1 4.9;30 30 30;40 40 40;40 40 40;40 40 40];
    A=runSequence(3,Y,zeros(size(Y,1),1),true);
    assert(min(A.Frames{end}.populationY(:,1))>5 && isempty(A.Frames{end}.archive.ref));
    assert(isequal(A.Frames{end}.scale.minimum,[5 5 5]), ...
        'The ideal must retain a historical minimum after its witness is lost.');
    fprintf('PASS ideal point: both initial populations, all offspring, rejected/lost minima, tail budgets and N unchanged\n');
end

function [A,P,seen,endState] = runSequence(N,Y,C,record)
    cursor=0; lastFE=0; seen=zeros(0,size(Y,2)); rng(481,'twister');
    P=UserProblem('N',N,'D',3,'maxFE',size(Y,1),'maxRuntime',Inf, ...
        'lower',0,'upper',1,'evalFcn',@evaluate,'once',true);
    % UserProblem validates its evaluator with one initialization in its constructor.
    cursor=0; seen=zeros(0,size(Y,2));
    if record
        [A,probeCleanup]=make_GANCMO_probe('save',-1,'outputFcn',@checkFrame); %#ok<NASGU>
    else
        A=GANCMO('save',-1,'outputFcn',@checkFrame);
    end
    A.Solve(P);
    if record; clear probeCleanup; end
    endState=rng;

    function [X,F,V] = evaluate(X)
        rows=cursor+(1:size(X,1)); X=min(1,max(0,X));
        F=Y(rows,:); V=C(rows,:); seen=[seen;F]; cursor=cursor+numel(rows);
    end

    function checkFrame(Algorithm,Problem)
        assert(cursor==Problem.FE && isequal(seen,Y(1:Problem.FE,:)));
        if record
            expected=inf(1,size(Y,2));
            for row=1:Problem.FE; expected=min(expected,Y(row,:)); end
            assert(isequal(Algorithm.Frames{end}.scale.minimum,expected), ...
                'Batch aggregation must equal sequential MOEA/D minimum updates.');
            if lastFE>0
                frame=Algorithm.Frames{end-1}.scale;
                assert(isequal(frame.minimum,min(Y(1:lastFE,:),[],1)), ...
                    'Training and queries must share the frame available before evaluating offspring.');
            end
        end
        lastFE=Problem.FE;
    end
end

function testArchiveMigration
    P=struct('N',10,'D',2,'M',2,'lower',[0 0],'upper',[1 1]); [W,~]=UniformPoint(10,2);
    old=struct('minimum',[1 1],'span',[1 1]); moved=struct('minimum',[0 1],'span',[1 1]);
    pool=SOLUTION([.2 .2;.8 .8],[2 4;1.7 4.6],[0;1]); empty=pool([]); before=rng;
    A=GANCMOArchive.update([],pool,W,P,old);
    [B,scale,gB]=GANCMOArchive.update(A,empty,W,P,moved);
    assert(A.ref==3 && B.ref==5 && isscalar(gB) && isequal(scale,moved));
    assert(isequal(A.xf,B.xf) && isequal(A.xi,B.xi) && isequal(A.yf,B.yf) && isequal(A.yi,B.yi));
    data=GANCMOArchive.training(B,W,scale,P);
    assert(isequal(data.cF,[W(5,:),1]) && isequal(data.cI,[W(4,:),0]));
    [ref,~,relative]=GANCMOReference(moved.minimum,W,moved);
    [~,central]=min(sum(W.^2,2));
    assert(ref==central && all(relative==0),'The zero vector must have a deterministic direction.');

    % An old pair ceases to satisfy the unchanged five-direction neighborhood.
    old.minimum=[0 0]; moved.minimum=[-100 0];
    pool=SOLUTION([.2 .2;.8 .8],[100 100;1 1],[0;1]);
    A=GANCMOArchive.update([],pool,W,P,old);
    B=GANCMOArchive.update(A,empty,W,P,moved);
    assert(isempty(B.ref),'Rebuilding must not retain a pair outside the current neighborhood.');
    % Both historical endpoints can independently find new partners.
    incoming=SOLUTION([.3 .3;.7 .7],[100.1 .1;99.9 99.9],[0;1]);
    B=GANCMOArchive.update(A,incoming,W,P,moved);
    [found,rows]=ismember([.2 .2;.3 .3],B.xf,'rows');
    assert(all(found) && isequal(B.xi(rows,:),[.7 .7;.8 .8]));

    % Two F endpoints migrate to one direction; choose its best current guide.
    pool=SOLUTION([.1 .1;.2 .2;.7 .7;.8 .8],[1 3;3 1;.75 3;3 .875],[0;0;1;1]);
    [A,~,gA]=GANCMOArchive.update([],pool,W,P,old);
    assert(numel(gA)==2);
    moved.minimum=[-1000 -2000];
    [B,~,gB]=GANCMOArchive.update(A,empty,W,P,moved);
    assert(numel(B.ref)==2 && all(B.ref==4) && isscalar(gB) && isequal(B.xf(gB,:),[.1 .1]));
    [C,~,gC]=GANCMOArchive.update(B,empty,W,P,moved);
    assert(isequal(C,B) && isequal(gC,gB) && isequal(rng,before));
    fprintf('PASS archive migration: endpoint conditions, zero vector, current neighborhoods, rebuilt pairs and migrated guides\n');
end

function restore(oldPath,oldRng)
    path(oldPath); rng(oldRng);
end
