function M = run_PairGuide_single_first_use(root,folder,number,seed,arm,O)
%RUN_PAIRGUIDE_SINGLE_FIRST_USE Train once, consume once, then stop search.
% All raw-cloud objective/label measurements happen after search termination.
    warning('off','all'); maxNumCompThreads(1); addCBSPaths(root);
    if ~isfolder(folder); mkdir(folder); end
    name=sprintf('LIRCMOP%d_BC',number);
    file=fullfile(folder,sprintf('%s_seed%02d_%s.mat',name,seed,arm));
    if isfile(file)
        old=load(file,'M','TrainingOptions'); assert(isequal(old.TrainingOptions,O));
        M=old.M; return;
    end
    rng(seed,'twister'); ctor=str2func(name);
    P=ctor('N',100,'maxFE',100000,'maxRuntime',Inf);
    G=PairGuide('save',1,'run',seed,'outputFcn',@stopAfterFirstUse);
    G.configurePairGuideTrainingExperiment(O);
    timer=tic; G.Solve(P); wall=toc(timer); Audit=G.guideExperimentSnapshot(); E=Audit.evidence;
    trained=E.training(cellfun(@(t)t.trained,E.training));
    assert(isscalar(trained) && isscalar(E.queries),'Exactly one training/query required.');
    T=trained{1}; Q=E.queries{1}; Use=E.generations{end}.use;
    Prefix=E.population{Q.generation+1}; FinalPopulation=E.population{end};
    Data=E.firstModel.lastData;
    assert(P.FE==Q.productionFE+200 && T.updates==O.initialEpoch && Use.selected>0);
    assert(isequal(Q.pending.decs,Q.rawDecs(Q.pool.keepIdx,:)) && ...
        isequal(Use.childDecs,Q.pending.decs) && all(Use.matchedPairIds==0));
    assert(numel(unique(Q.refs))==size(E.W,1) && nnz(Q.sides==1)==250 && nnz(Q.sides==0)==250);
    assert(E.oracleCalls.CalObjRows==2*P.FE && E.oracleCalls.CalConRows==P.FE);
    % No offline labels are available to search or model selection within a run.
    Eval=ctor(); PairGuideCost_RC('start',Eval); cleanup=onCleanup(@()PairGuideCost_RC('stop'));
    Y=Eval.CalObj(Q.rawDecs); C=Eval.CalCon(Q.rawDecs); feasible=all(C<=0,2);
    k=Q.pool.keepIdx;
    assert(isequal(Y(k,:),Use.childObjs) && isequal(C(k,:),Use.childCons));
    [actualRef,~,Yn]=AssignReferenceVectors_CBS(Y,E.W,Prefix.referenceScale);
    target=E.W(Q.refs,:); target=target./max(vecnorm(target,2,2),eps);
    direction=Yn./max(vecnorm(Yn,2,2),eps);
    angles=acosd(max(-1,min(1,sum(target.*direction,2))));
    F=Prefix.p1Objs(all(Prefix.p1Cons<=0,2),:);
    dominated=false(size(Y,1),1);
    for j=1:size(F,1)
        dominated=dominated | (all(F(j,:)<=Y+1e-12,2) & any(F(j,:)<Y-1e-12,2));
    end
    trainC=unique([Data.cF;Data.cI],'rows'); known=ismember(Q.conditions,trainC,'rows');
    TrainY=[Prefix.archive.yf(Prefix.archive.active,:);Prefix.archive.yi(Prefix.archive.active,:)];
    nearest=sqrt(min(pdist2(Y,TrainY,'squaredeuclidean'),[],2));
    Pop=SOLUTION(FinalPopulation.p1Decs,FinalPopulation.p1Objs,FinalPopulation.p1Cons);
    Before=SOLUTION(Prefix.p1Decs,Prefix.p1Objs,Prefix.p1Cons);
    M=struct('problem',string(name),'seed',seed,'arm',string(arm), ...
        'updates',T.updates,'trainingPairs',T.trainingPairs,'trainingSamples',T.trainingSamples, ...
        'productionFE',Q.productionFE,'firstUseFE',P.FE,'trainingSeconds',T.trainingSeconds, ...
        'wallSeconds',wall,'labelAccuracy',mean(feasible==Q.sides), ...
        'feasibleRequestAccuracy',mean(feasible(Q.sides==1)), ...
        'infeasibleRequestAccuracy',mean(~feasible(Q.sides==0)), ...
        'directionError',mean(angles),'knownDirectionError',mean(angles(known)), ...
        'unseenDirectionError',mean(angles(~known)),'referenceHitRate',mean(actualRef==Q.refs), ...
        'unseenRequestFraction',mean(~known),'rawUseful',mean(feasible & ~dominated), ...
        'rawFeasible',mean(feasible),'nearestObjectiveDistance',mean(nearest), ...
        'selected',Use.selected,'survivedP1',nnz(Use.survivedP1),'survivedP2',nnz(Use.survivedP2), ...
        'initialIGD',P.CalMetric('IGD',Before),'firstUseIGD',P.CalMetric('IGD',Pop), ...
        'firstUseHV',P.CalMetric('HV',Pop),'probeRMSE',T.postDiagnostics.allEndpointRMSE);
    OfflineCost=PairGuideCost_RC('snapshot'); assert(Eval.FE==0 && OfflineCost.CalConRows==500);
    clear cleanup;
    TrainingOptions=O; Record=struct('schema',E.schema,'mode',"cgan",'seed',seed, ...
        'problem',string(name),'D',P.D,'N',100,'finalFE',P.FE,'sourceHash',sourceHash(), ...
        'experimentArm',string(arm), ...
        'distributionLayout',"training_pair",'checkpointFE',[]);
    save(file,'M','TrainingOptions','Record','Audit','Q','T','Prefix','FinalPopulation', ...
        'Data','Y','C','angles','known','OfflineCost','-v7.3');
    writetable(struct2table(M),[file,'.csv']);
    fprintf('FIRST_USE %s seed=%d arm=%s updates=%d label=%.3f angle=%.2f useful=%.3f P1=%d IGD=%.5g seconds=%.1f\n', ...
        name,seed,arm,T.updates,M.labelAccuracy,M.directionError,M.rawUseful,M.survivedP1,M.firstUseIGD,M.trainingSeconds);
end

function stopAfterFirstUse(G,P)
    S=G.guideExperimentSnapshot(); E=S.evidence;
    if ~isempty(E.queries)
        assert(~isempty(E.queries{1}.pending.decs),'No candidates after first training.');
        P.maxFE=E.queries{1}.productionFE+2*P.N;
    elseif any(cellfun(@(t)t.reason=="numerical_failure",E.training))
        error('PairGuide:FirstTrainingFailed','First training failed numerically.');
    end
end

function h=sourceHash()
    names=["PairGuideCore","PairBoundaryArchive_RC","PairBoundaryWGAN_RC","run_PairGuide_single_first_use"];
    bytes=uint8([]);
    for name=names; bytes=[bytes,unicode2native(fileread(which(name)),'UTF-8')]; end %#ok<AGROW>
    digest=java.security.MessageDigest.getInstance('SHA-256'); digest.update(bytes);
    h=lower(string(reshape(dec2hex(typecast(digest.digest(),'uint8'),2)',1,[])));
end
