function test_GANCMO_endpoint_loss
% Test learning of distinct true-side targets, not an optimizer implementation.
    root = fileparts(fileparts(mfilename('fullpath')));
    oldPath = path; oldRng = rng;
    cleanup = onCleanup(@()restoreState(oldPath,oldRng));
    addpath(fullfile(root,'Algorithms','Multi-objective optimization','GANCMO'));
    Data = struct('count',8,'ref',ones(8,1),'xF',repmat([.2 .3],8,1), ...
        'xI',repmat([.8 .7],8,1),'cF',repmat([.5 .5 1],8,1), ...
        'cI',repmat([.5 .5 0],8,1),'directions',[.5 .5]);
    S = GANCMOSettings(struct('initialG',400,'generatorHidden',[16 16], ...
        'criticHidden',[16 16],'criticSteps',1,'endpointWeight',100));
    P = struct('D',2,'M',2,'lower',[0 0],'upper',[1 1]);
    errors = zeros(1,2); labels = [ones(100,1);zeros(100,1)];
    target = [repmat(Data.xF(1,:),100,1);repmat(Data.xI(1,:),100,1)];
    for variant = 1:2
        S.endpointWeight = 100*(variant>1);
        rng(719,'twister');
        [M,I] = GANCMOModel.train([],Data,2,1,0,S);
        assert(I.updated && ~I.failed && I.updatesG==400 && I.updatesC==400);
        rng(811,'twister'); X = GANCMOModel.sample(M,[repmat([.5 .5],200,1),labels],P);
        errors(variant) = mean(vecnorm(X-target,2,2));
    end
    assert(errors(2)<.03 && errors(2)<.5*errors(1), ...
        'GANCMO:EndpointLearning','Same-side endpoint loss did not learn the distinct targets.');
    for disallowed = [2 3 5]
        caught = false;
        try
            GANCMOSettings(struct('pairsPerDirection',disallowed));
        catch err
            caught = strcmp(err.identifier,'GANCMO:BadPairCount');
        end
        assert(caught,'GANCMO:UnrestrictedPairCount','Disallowed pair count was accepted.');
    end
    fprintf('PASS unit-condition endpoint fitting: adversarial=%.7g endpoint=%.7g; K other than 1 rejected\n',errors);
end

function restoreState(oldPath,oldRng)
    path(oldPath); rng(oldRng);
end
