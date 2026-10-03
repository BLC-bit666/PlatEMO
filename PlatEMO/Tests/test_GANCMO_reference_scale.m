function test_GANCMO_reference_scale
% Fixed objective coordinates resist unrelated outliers and retain eligibility.
    root = fileparts(fileparts(mfilename('fullpath'))); oldPath = path;
    cleanup = onCleanup(@()path(oldPath));
    addpath(fullfile(root,'Problems'),fullfile(root,'Algorithms'),fullfile(root,'Algorithms','Utility functions'), ...
        fullfile(root,'Algorithms','Multi-objective optimization','GANCMO'));
    W = UniformPoint(100,2); P = struct('D',30,'M',2,'lower',zeros(1,30),'upper',ones(1,30));
    fx = repmat([.1;.4;.7],1,30); ix = fx+.005;
    fy = [1 3;2 2;3 1]; iy = fy-.01;
    Pop = SOLUTION([fx;ix],[fy;iy],[zeros(3,1);ones(3,1)]);
    fixed = struct('minimum',[0 0],'span',[1 1]);
    [A,S] = GANCMOArchive.update([],Pop,Pop([]),W,P,1,'objective_gap',fixed);
    assert(isequal(S,fixed) && ~isempty(A.id));
    outlier = SOLUTION(ones(1,30),[300 200],1);
    [B,S] = GANCMOArchive.update(A,[Pop,outlier],Pop([]),W,P,1,'objective_gap',fixed);
    assert(isequal(S,fixed) && isequal(sortrows(A.xf),sortrows(B.xf)));
    assert(isequal(A.ref,GANCMOReference(A.yf,W,fixed)) && nnz(A.active)==3);
    strong = SOLUTION(zeros(1,30),[0 0],0);
    C = GANCMOArchive.update(B,[Pop,strong],Pop([]),W,P,1,'objective_gap',fixed);
    assert(isempty(C.id),'Fixed coordinates bypassed original Pareto deletion.');
    [C,S1] = GANCMOArchive.update([],Pop,Pop([]),W,P,1,'quality');
    [D,S2] = GANCMOArchive.update([],Pop,Pop([]),W,P,1,'quality',[]);
    assert(isequaln(C,D) && isequaln(S1,S2));
    fprintf('PASS fixed objective directions, outlier stability, K1, original Pareto deletion and adaptive default\n');
end
