function capture_stage_fixtures(seeds,nWorker)
% Re-run the unchanged original algorithm; capture model states read-only.
% Full runs keep maxFE=100000; compare final populations/weights to saved runs.
if nargin<1; seeds=1; end
if nargin<2; nWorker=4; end
warning('off','all'); maxNumCompThreads(1);
folder=fileparts(mfilename('fullpath')); root=fileparts(fileparts(folder));
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support'));
addCBSPaths(root); out=fullfile(folder,'fixtures'); if ~isfolder(out); mkdir(out); end
tasks=[]; for seed=seeds; for n=5:8; tasks=[tasks;n seed]; end; end %#ok<AGROW>
pool=parpool('Processes',nWorker); cleanup=onCleanup(@()delete(pool));
errors=strings(size(tasks,1),1);
parfor k=1:size(tasks,1)
    try; captureOne(root,folder,tasks(k,1),tasks(k,2));
    catch err; errors(k)=string(getReport(err,'extended','hyperlinks','off')); fprintf('%s\n',errors(k)); end
end
save(fullfile(folder,sprintf('capture_status_seed%d.mat',seeds(1))),'tasks','errors');
assert(all(strlength(errors)==0)); fprintf('STAGE_CAPTURE_COMPLETE %d unchanged full searches\n',size(tasks,1));
end

function captureOne(root,folder,n,seed)
maxNumCompThreads(1); addCBSPaths(root);
stem=sprintf('LIRCMOP%d_BC_seed%02d',n,seed);
done=fullfile(folder,[stem,'_capture_check.csv']); if isfile(done); return; end
targets=[0 20000 50000 80000 99800]; labels={'first','early','middle','late','terminal'};
captured=false(size(targets)); lastBlock=-1;
rng(seed,'twister'); ctor=str2func(sprintf('LIRCMOP%d_BC',n));
P=ctor('N',100,'maxFE',100000,'maxRuntime',Inf);
A=PairGuide('save',1,'run',seed,'outputFcn',@observe); A.configureComparison('cgan');
A.configureObjectiveSpaceSnapshots(struct('enabled',seed==1, ...
    'targetFE',[10000 30000 50000 70000 100000],'expectedRawCount',500,'expectedGuidedCount',20));
timer=tic; A.Solve(P); seconds=toc(timer);
Audit=A.guideExperimentSnapshot(); E=Audit.evidence;
assert(P.FE==100000 && E.oracleCalls.CalObjRows==200000 && E.oracleCalls.CalConRows==100000);
source=fullfile(root,'Data','PairGuideSinglePoint_20260907','full_run',[stem,'_cgan.mat']);
Old=load(source,'Audit'); old=Old.Audit.evidence;
sameP1=isequal(E.population{end}.p1Decs,old.population{end}.p1Decs);
sameP2=isequal(E.population{end}.p2Decs,old.population{end}.p2Decs);
sameG=isequaln(E.lastModel.netG.Learnables,old.lastModel.netG.Learnables);
sameD=isequaln(E.lastModel.netC.Learnables,old.lastModel.netC.Learnables);
sameQueries=isequal(E.queries{end}.rawDecs,old.queries{end}.rawDecs);
assert(sameP1 && sameP2 && sameG && sameD && sameQueries && all(captured));
T=table(n,seed,P.FE,seconds,sameP1,sameP2,sameG,sameD,sameQueries, ...
    'VariableNames',{'problemNumber','searchSeed','searchFE','seconds','sameP1','sameP2','sameG','sameD','sameLastQuery'});
writetable(T,done); fprintf('CAPTURE_REPLAY_VERIFIED %s FE=100000 identical=true\n',stem);

    function observe(Algorithm,Problem)
        AuditNow=Algorithm.guideExperimentSnapshot(); Ev=AuditNow.evidence;
        if ~isfield(Ev,'lastModel') || isempty(Ev.lastModel) || ~isfield(Ev.lastModel,'netG'); return; end
        used=any(cellfun(@(g)g.use.selected>0,Ev.generations));
        for j=1:numel(targets)
            if captured(j) || (j==1 && ~used) || (j>1 && Problem.FE<targets(j)); continue; end
            Model=Ev.lastModel; W=Ev.W;
            trained=Ev.training(cellfun(@(t)t.trained,Ev.training)); Event=trained{end};
            TrainState=Ev.population{Event.generation+1}; CurrentState=Ev.population{end};
            RawData=Model.lastData;
            [found,idx]=ismember(RawData.id,TrainState.archive.id); assert(all(found));
            RawData.yF=TrainState.archive.yf(idx,:); RawData.yI=TrainState.archive.yi(idx,:); RawData.W=W;
            assert(isequal(RawData.xF,TrainState.archive.xf(idx,:)) && isequal(RawData.xI,TrainState.archive.xi(idx,:)));
            Data=Model.lastTrainingData; Data.yF=RawData.yF; Data.yI=RawData.yI; Data.W=W;
            QueryC=[W,ones(size(W,1),1);W,zeros(size(W,1),1)];
            Meta=struct('problem',sprintf('LIRCMOP%d_BC',n),'searchSeed',seed,'stage',labels{j}, ...
                'targetFE',targets(j),'observationFE',Problem.FE,'trainingFE',TrainState.observationFE, ...
                'trainingGeneration',Event.generation,'currentArchivePairs',numel(CurrentState.archive.id), ...
                'currentActivePairs',nnz(CurrentState.archive.active),'fixedTrainingPairs',Data.count);
            saved=rng;
            save(fullfile(folder,'fixtures',sprintf('%s_%s.mat',stem,labels{j})), ...
                'Model','Data','RawData','W','QueryC','TrainState','CurrentState','Meta','Event','-v7.3');
            assert(isequal(rng,saved)); captured(j)=true;
            fprintf('STAGE_FIXTURE_READY %s %s observed=%d trained=%d pairs=%d\n', ...
                stem,labels{j},Meta.observationFE,Meta.trainingFE,Data.count);
        end
        block=floor(Problem.FE/10000);
        if block>lastBlock; lastBlock=block; fprintf('CAPTURE_PROGRESS %s FE=%d\n',stem,Problem.FE); end
    end
end
