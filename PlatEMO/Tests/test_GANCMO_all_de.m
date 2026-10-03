function test_GANCMO_all_de
% Direct CGAN identity, no GA, independent P2, and exact search FE accounting.
    root = fileparts(fileparts(mfilename('fullpath')));
    oldPath = path; oldRng = rng;
    cleanup = onCleanup(@()restoreState(oldPath,oldRng));
    addpath(fullfile(root,'Algorithms'),fullfile(root,'Algorithms','Utility functions'), ...
        fullfile(root,'Algorithms','Multi-objective optimization','GANCMO'), ...
        fullfile(root,'Problems'), ...
        fullfile(root,'Problems','Multi-objective optimization','LIR-CMOP_BC'));
    rng(123,'twister');
    P = LIRCMOP5_BC('N',100,'D',30,'maxFE',10000);
    P1 = P.Initialization(); P2 = P.Initialization();
    F1 = GANCMOFitness(P1.objs,P1.cons); F2 = GANCMOFitness(P2.objs);
    cases = [100 100 0;100 100 90;10 9 10;5 4 3;1 0 1;0 0 0];
    profile clear; profile on;
    for k = 1:size(cases,1)
        n1 = cases(k,1); n2 = cases(k,2); m = cases(k,3);
        q = P.lower+(.3+.4*rand(m,P.D)).*(P.upper-P.lower);
        before = P.FE; beforeRng = rng;
        [O1,O2,G] = GANCMOOffspring(P,P1,P2,F1,F2,n1,n2,q,17,3);
        assert(isequal(rng,beforeRng));
        assert(P.FE-before==n1+n2 && numel(O1)==n1 && numel(O2)==n2);
        assert(numel(G)==m);
        if m>0
            assert(isequal(G.decs,q),'Generated points were not directly evaluated.');
            assert(isequal(G,O1(n1-m+1:n1)));
            [~,plain2] = GANCMOOffspring(P,P1,P2,F1,F2,n1,n2,zeros(0,P.D),17,3);
            assert(isequal(O2.decs,plain2.decs),'P2 changed with CGAN proposals.');
        end
        fprintf('PASS direct offspring n1=%d n2=%d generated=%d\n',n1,n2,m);
    end
    [p,k] = GANCMOControl(200,200,200000,100); assert(p==0 && k==10);
    [p,k] = GANCMOControl(100100,200,200000,100); assert(p==.5 && k==50);
    [~,k] = GANCMOControl(199999,200,200000,1); assert(k==1);
    [~,k] = GANCMOControl(200,200,200,0); assert(k==0);
    for budget = [24 25 73]
        rng(1,'twister');
        P = LIRCMOP5_BC('N',12,'D',30,'maxFE',budget,'maxRuntime',Inf);
        A = GANCMO('save',-2,'outputFcn',@(varargin)[]); A.RecordTrace = true;
        A.Solve(P);
        assert(P.FE==budget && A.result{end,1}==budget);
        assert(numel(A.result{end,2})==P.N);
        fprintf('PASS full solve maxFE=%d\n',budget);
    end
    S = GANCMOSettings(struct('initialG',2,'updateG',[1 1],'trainInterval',1));
    D = struct('count',8,'ref',(1:8)','xF',rand(8,30),'xI',rand(8,30), ...
        'cF',[linspace(.1,.9,8)',linspace(.9,.1,8)',ones(8,1)], ...
        'cI',[linspace(.1,.9,8)',linspace(.9,.1,8)',zeros(8,1)]);
    [M,T] = GANCMOModel.train([],D,30,1,0,S);
    assert(M.ready && T.updatesG==2 && T.updatesC==10);
    [M,T] = GANCMOModel.train(M,D,30,2,.5,S);
    assert(M.iterG==3 && T.updatesG==1);
    P = LIRCMOP5_BC('N',100,'D',30,'maxFE',100);
    X = GANCMOModel.sample(M,D.cF,P);
    assert(isequal(size(X),[8 30]) && all(isfinite(X),'all'));
    assert(all(X>=0 & X<=1,'all') && P.FE==0);
    D.count = 7; old = M;
    [M,T] = GANCMOModel.train(M,D,30,9,.9,S);
    assert(isequal(M,old) && ~T.attempted);
    fprintf('PASS training updates, state continuation, feasible queries, no extra FE\n');
    profile off; stats = profile('info');
    names = string({stats.FunctionTable.FunctionName});
    assert(any(names=="OperatorDE") && ~any(contains(names,"OperatorGA")));
    assert(~any(contains(names,"GANCMOBoundaryStep")));
    fprintf('PASS no GA or boundary contraction calls\n');
end
function restoreState(oldPath,oldRng)
    profile off; path(oldPath); rng(oldRng);
end
