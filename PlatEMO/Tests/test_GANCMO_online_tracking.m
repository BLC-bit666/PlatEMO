function test_GANCMO_online_tracking
% Fit moved endpoints from identical trained weights and Adam state.
    root = fileparts(fileparts(mfilename('fullpath')));
    oldPath = path; oldRng = rng;
    cleanup = onCleanup(@()restoreState(oldPath,oldRng));
    addpath(fullfile(root,'Algorithms','Multi-objective optimization','GANCMO'));
    Data = struct('count',8,'ref',ones(8,1),'xF',repmat([.2 .3],8,1), ...
        'xI',repmat([.8 .7],8,1),'cF',repmat([.5 .5 1],8,1), ...
        'cI',repmat([.5 .5 0],8,1),'directions',[.5 .5]);
    S = GANCMOSettings(struct('initialG',400,'updateG',[200 200],'trainInterval',1, ...
        'generatorHidden',[16 16],'criticHidden',[16 16],'criticSteps',1, ...
        'endpointWeight',100,'conditionEncoding','unit','learningRateDecay',.01));
    rng(719,'twister'); [Initial,Info] = GANCMOModel.train([],Data,2,1,0,S);
    assert(Info.updated && Initial.iterG==400);
    initialWeights = cellfun(@extractdata,[Initial.netG.Learnables.Value;Initial.netC.Learnables.Value], ...
        'UniformOutput',false);
    Data.xF(:,:) = repmat([.4 .45],8,1); Data.xI(:,:) = repmat([.6 .55],8,1);
    target = [repmat(Data.xF(1,:),100,1);repmat(Data.xI(1,:),100,1)];
    P = struct('D',2,'M',2,'lower',[0 0],'upper',[1 1]);
    C = [repmat([.5 .5],200,1),[ones(100,1);zeros(100,1)]];
    schedules = {'cold_tail','event_tail'}; errors = zeros(1,2);
    for variant = 1:2
        M = Initial; S.learningRateSchedule = schedules{variant}; M.settings = S;
        rng(902,'twister'); [M,Info] = GANCMOModel.train(M,Data,2,2,.5,S);
        assert(Info.updated && ~Info.failed && Info.updatesG==200 && M.iterG==600 && M.iterC==600);
        assert(isequaln(initialWeights,cellfun(@extractdata, ...
            [Initial.netG.Learnables.Value;Initial.netC.Learnables.Value],'UniformOutput',false)), ...
            'GANCMO:MutatedCheckpoint','A continuation modified the common initial network.');
        rng(811,'twister'); X = GANCMOModel.sample(M,C,P);
        errors(variant) = mean(vecnorm(X-target,2,2));
    end
    assert(errors(2)<.04 && errors(2)<.5*errors(1), ...
        'GANCMO:OnlineTracking','Event refinement did not track the moved true-side targets.');
    fprintf('PASS moved endpoint tracking from identical checkpoint: cold_tail=%.7g event_tail=%.7g\n',errors);
end

function restoreState(oldPath,oldRng)
    path(oldPath); rng(oldRng);
end
