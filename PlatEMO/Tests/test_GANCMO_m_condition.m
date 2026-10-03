function test_GANCMO_m_condition
% The single mainline encoding has M+1 inputs and preserves reference rays.
    root = fileparts(fileparts(mfilename('fullpath')));
    oldPath = path; oldRng = rng;
    cleanup = onCleanup(@()restoreState(oldPath,oldRng)); %#ok<NASGU>
    addpath(fullfile(root,'Algorithms','Multi-objective optimization','GANCMO'));
    W = [linspace(.05,.95,8)',linspace(.95,.05,8)'];
    Data = struct('count',8,'ref',(1:8)','xF',repmat([.2 .3 .4],8,1), ...
        'xI',repmat([.7 .8 .6],8,1),'cF',[W,ones(8,1)], ...
        'cI',[W,zeros(8,1)],'directions',W);
    P = struct('D',3,'M',2,'lower',zeros(1,3),'upper',ones(1,3));
    for objectives = [2 3]
        if objectives==3
            Data.cF = [W,.3*ones(8,1),ones(8,1)];
            Data.cI = [W,.3*ones(8,1),zeros(8,1)];
            Data.directions = [W,.3*ones(8,1)]; P.M = 3;
        end
        for activation = {'leakyrelu','sine'}
            S = GANCMOSettings(struct('initialG',2,'generatorHidden',8,'criticHidden',4, ...
                'criticSteps',2,'endpointWeight',1000,'conditionEncoding','unit', ...
                'generatorActivation',activation{1},'generatorFrequency',256, ...
                'learningRateSchedule','event_exp','learningRateDecay',.001));
            rng(915,'twister'); [M,I] = GANCMOModel.train([],Data,3,1,0,S);
            assert(I.updated && ~I.failed && I.updatesG==2 && I.updatesC==4);
            assert(M.netG.Layers(1).InputSize==12+P.M+1);
            assert(M.netC.Layers(1).InputSize==P.D+P.M+1);
            assert(~isfield(M,'directions'),'A hidden direction table entered the M-dimensional model.');
            assert(isequal(M.lastData.cF(:,end),ones(8,1)) && ...
                isequal(M.lastData.cI(:,end),zeros(8,1)));
            rng(916,'twister'); X = GANCMOModel.sample(M,[Data.cF;Data.cI],P);
            assert(isequal(size(X),[16 3]) && all(isfinite(X),'all') && all(X>=0 & X<=1,'all'));
            C = [Data.cF;Data.cI]; C(:,1:end-1) = 7*C(:,1:end-1);
            rng(916,'twister'); scaled = GANCMOModel.sample(M,C,P);
            assert(max(abs(X-scaled),[],'all')<1e-6,'Unit encoding changed a reference ray.');
            fprintf('PASS M+1 M=%d activation=%s Ginput=%d Dinput=%d true labels and rays retained\n', ...
                objectives,activation{1},12+P.M+1,P.D+P.M+1);
        end
    end
    for encoding = {'coordinates','signed_unit','onehot','joint'}
        caught = false;
        try; GANCMOSettings(struct('conditionEncoding',encoding{1}));
        catch err; caught = strcmp(err.identifier,'GANCMO:BadConditionEncoding'); end
        assert(caught,'A non-mainline condition encoding was accepted.');
    end
end
function restoreState(oldPath,oldRng)
    path(oldPath); rng(oldRng);
end
