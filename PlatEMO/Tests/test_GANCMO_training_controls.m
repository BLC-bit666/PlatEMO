function test_GANCMO_training_controls
% Real optimizer steps check sparse continuation, reset, and multiple pairs.
    root = fileparts(fileparts(mfilename('fullpath')));
    oldPath = path; oldRng = rng;
    cleanup = onCleanup(@()restoreState(oldPath,oldRng));
    addpath(fullfile(root,'Algorithms'),fullfile(root,'Algorithms','Utility functions'), ...
        fullfile(root,'Algorithms','Multi-objective optimization','GANCMO'), ...
        fullfile(root,'Problems'), ...
        fullfile(root,'Problems','Multi-objective optimization','LIR-CMOP_BC'));
    rng(213,'twister');
    S = GANCMOSettings(jsondecode( ...
        '{"initialG":2,"updateG":[1,1],"trainInterval":1,"generatorHidden":[8,8],"criticHidden":[16,16]}'));
    D = struct('count',8,'ref',(1:8)','xF',rand(8,30),'xI',rand(8,30), ...
        'cF',[linspace(.1,.9,8)',linspace(.9,.1,8)',ones(8,1)], ...
        'cI',[linspace(.1,.9,8)',linspace(.9,.1,8)',zeros(8,1)]);
    sparse = D; fields = {'ref','xF','xI','cF','cI'};
    for k = 1:numel(fields); sparse.(fields{k}) = D.(fields{k})(1,:); end
    sparse.count = 1;
    [M,T] = GANCMOModel.train([],sparse,30,1,0,S);
    assert(isempty(M) && ~T.attempted,'Initial sample minimum was bypassed.');
    [M,T] = GANCMOModel.train([],D,30,1,0,S);
    assert(M.iterG==2 && T.updatesC==10);
    old = M;
    [M,T] = GANCMOModel.train(M,sparse,30,2,.5,S);
    assert(isequal(M,old) && ~T.attempted,'Default behavior changed.');
    S.minimumPairsContinue = 1;
    [M,T] = GANCMOModel.train(M,sparse,30,2,.5,S);
    assert(M.iterG==3 && T.updatesG==1 && T.updatesC==5 && M.lastData.count==1);
    S.resetInterval = 3;
    [M,T] = GANCMOModel.train(M,sparse,30,4,.5,S);
    assert(~T.reset && T.updatesG==1 && M.iterG==4 && M.lastResetGeneration==1);
    [M,T] = GANCMOModel.train(M,D,30,5,.5,S);
    assert(T.reset && T.updatesG==2 && M.iterG==2 && M.lastResetGeneration==5);
    empty = sparse; empty.count = 0;
    for k = 1:numel(fields); empty.(fields{k}) = sparse.(fields{k})([],:); end
    old = M; [M,T] = GANCMOModel.train(M,empty,30,6,.6,S);
    assert(isequal(M,old) && ~T.attempted);
    fprintf('PASS initial minimum, sparse continuation, real reset, empty-data skip\n');

    S.learningRateDecay = .1;
    trainingStream = RandStream('mt19937ar','Seed',491);
    searchRng = rng; StreamModel = []; states = cell(3,1);
    for generation = 1:3
        before = trainingStream.State;
        previous = RandStream.getGlobalStream;
        scope = onCleanup(@()RandStream.setGlobalStream(previous));
        RandStream.setGlobalStream(trainingStream);
        [StreamModel,T] = GANCMOModel.train(StreamModel,D,30,generation,.5,S);
        active = RandStream.getGlobalStream;
        assert(isequal(active.State,trainingStream.State), ...
            'Training replaced the caller-owned random stream.');
        states{generation} = trainingStream.State;
        assert(T.updated && ~isequal(before,states{generation}), ...
            'The dedicated training stream did not advance.');
        clear scope;
        assert(isequal(rng,searchRng),'Training consumed the search random stream.');
    end
    assert(StreamModel.iterG==4 && ~isequal(states{2},states{3}));
    fprintf('PASS training stream advances across continuation; search RNG remains isolated\n');

    P = LIRCMOP5_BC('N',100,'D',30,'maxFE',1000);
    fx = repmat([.1;.4;.7],1,30); ix = fx+.005;
    fy = [1 3;2 2;3 1]; iy = fy-.1;
    synthetic = SOLUTION([fx;ix],[fy;iy],[zeros(3,1);ones(3,1)]);
    W = [.5 .5];
    [A1,~] = GANCMOArchive.update([],synthetic,synthetic([]),W,P,1);
    [A3,scale] = GANCMOArchive.update([],synthetic,synthetic([]),W,P,3);
    assert(nnz(A1.active)==1 && nnz(A3.active)==3);
    train = GANCMOArchive.training(A3,W,scale,P);
    assert(train.count==3 && size(unique(train.xF,'rows'),1)==3 && P.FE==0);
    assert(all(train.cF(:,end)==1) && all(train.cI(:,end)==0));
    fprintf('PASS three distinct active pairs on one direction, true side encoding\n');

    originalArchive = A3;
    narrow = GANCMOArchive.training(A3,W,scale,P,.05);
    assert(narrow.count==0 && isequaln(A3,originalArchive));
    S.maxTrainingObjectiveGap = .05;
    [candidate,query] = GANCMOCandidates(M,A3,W,scale,synthetic,synthetic([]),P,10,S);
    assert(isempty(candidate) && query.available==0 && P.FE==0 && nnz(A3.active)==3);
    assert(GANCMOArchive.training(A3,W,scale,P,.2).count==3);
    fprintf('PASS training readiness defers unsupported queries without deleting real archive pairs or spending FE\n');
    S.maxTrainingObjectiveGap = Inf; S.maxCandidateDistanceF = 1e-12;
    [candidate,query] = GANCMOCandidates(M,A3,W,scale,synthetic,synthetic([]),P,10,S);
    assert(isempty(candidate) && query.queried==S.candidateCount && query.available==0 && P.FE==0);
    fprintf('PASS unfitted proposals are counted and rejected before evaluation; no extra FE\n');

    ix = fx+repmat([.001;.1;.02],1,30);
    iy = fy-repmat([.1;.1;.01],1,2);
    synthetic = SOLUTION([fx;ix],[fy;iy],[zeros(3,1);ones(3,1)]);
    [quality,~] = GANCMOArchive.update([],synthetic,synthetic([]),W,P,1);
    [gap,~] = GANCMOArchive.update([],synthetic,synthetic([]),W,P,1,'gap');
    S = GANCMOSettings(struct('activePairPriority','objective_gap'));
    [objectiveGap,scale] = GANCMOArchive.update([],synthetic,synthetic([]),W,P,1,S.activePairPriority);
    assert(isequal(quality.xf(quality.active,:),fx(2,:)));
    assert(isequal(gap.xf(gap.active,:),fx(1,:)));
    assert(isequal(objectiveGap.xf(objectiveGap.active,:),fx(3,:)));
    assert(isequal(sortrows([quality.xf,quality.xi]),sortrows([gap.xf,gap.xi])));
    assert(isequal(sortrows([quality.xf,quality.xi]),sortrows([objectiveGap.xf,objectiveGap.xi])));
    assert(GANCMOArchive.training(objectiveGap,W,scale,P,.02).count==1 && ...
        GANCMOArchive.training(quality,W,scale,P,.02).count==0);
    superior = SOLUTION(zeros(1,30),[0 0],0);
    removed = GANCMOArchive.update(objectiveGap,superior,superior([]),W,P,1,'objective_gap');
    assert(isempty(removed.id),'Objective-gap priority bypassed original Pareto deletion.');
    assert(P.FE==0);
    fprintf('PASS decision/objective gap priority selects real narrow pairs, preserves pair pool and Pareto deletion\n');
end
function restoreState(oldPath,oldRng)
    path(oldPath); rng(oldRng);
end
