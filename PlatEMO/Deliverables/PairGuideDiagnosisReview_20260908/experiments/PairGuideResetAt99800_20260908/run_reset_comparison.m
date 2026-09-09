function run_reset_comparison(nWorker,probeMismatch)
% Frozen FE99800 data: original weights versus three complete G/D/Adam resets.
if nargin<1; nWorker=8; end
if nargin<2; probeMismatch=false; end
warning('off','all'); maxNumCompThreads(1);
folder=fileparts(mfilename('fullpath')); root=fileparts(fileparts(folder));
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support'));
addCBSPaths(root); addpath(folder); addpath(fullfile(root,'Data','PairGuideGPTValidation_20260908'));
directory='runs'; arms={'original','stable_side','stable_no_side'}; resets=0:3;
if probeMismatch; directory='mismatch_runs'; arms={'original'}; resets=1; end
out=fullfile(folder,directory); if ~isfolder(out); mkdir(out); end
Tasks=struct('arm',{},'number',{},'reset',{},'panel',{},'probeMismatch',{}); panel=0;
for arm=arms
    for number=7:8
        panel=panel+1;
        for reset=resets
            Tasks(end+1)=struct('arm',arm{1},'number',number,'reset',reset,'panel',panel,'probeMismatch',probeMismatch);
        end
    end
end
pool=parpool('Processes',nWorker); cleanup=onCleanup(@()delete(pool));
errors=strings(numel(Tasks),1);
parfor k=1:numel(Tasks)
    try; runCase(root,folder,out,Tasks(k));
    catch err; errors(k)=string(getReport(err,'extended','hyperlinks','off')); fprintf('%s\n',errors(k)); end
end
save(fullfile(out,'status.mat'),'Tasks','errors'); assert(all(strlength(errors)==0));
fprintf('RESET_COMPARISON_COMPLETE %d cases, 10000 G updates each, mismatch=%d\n',numel(Tasks),probeMismatch);
end

function runCase(root,folder,out,task)
maxNumCompThreads(1); addCBSPaths(root); addpath(folder); addpath(fullfile(root,'Data','PairGuideGPTValidation_20260908'));
trainer=@PairBoundaryWGAN_RC;
if task.probeMismatch; trainer=@PairBoundaryWGAN_MismatchProbe; end
stem=sprintf('%s_LIRCMOP%d_BC_reset%d',task.arm,task.number,task.reset);
target=fullfile(out,[stem,'.mat']);
if isfile(target)
    done=load(target,'Complete'); if isfield(done,'Complete') && done.Complete; return; end
end
F=load(fullfile(folder,'fixtures',sprintf('%s_LIRCMOP%d_BC.mat',task.arm,task.number)));
Data=F.RawData; Model=F.Model;
if task.reset>0
    Model=fresh_model(Model,600000+task.reset);
    assert(Model.iterG==0 && Model.iterC==0 && isempty(Model.avgG) && isempty(Model.avgC) && isempty(Model.avgSqG) && isempty(Model.avgSqC));
    assert(~isequaln(Model.netG.Learnables,F.Model.netG.Learnables) && ~isequaln(Model.netC.Learnables,F.Model.netC.Learnables));
end
ctor=str2func(sprintf('LIRCMOP%d_BC',task.number)); P=ctor('N',100,'maxFE',100000);
O=struct('initialEpoch',1000,'retrainEpoch',20,'nCritic',5,'lrG',.001,'lrD',.001, ...
    'gpLambda',10,'miniBatch',32,'generatorHidden',[32 32],'criticHidden',[32 32], ...
    'trainingSigma',0,'sampleSigma',0,'retrainChange',0,'retrainGenerations',1, ...
    'useSideCondition',F.useSide,'stableConditionSpan',F.stable);
Gate=struct('eligible',true); base=Model.iterG; startingGeneration=Model.lastTrainGeneration;
Snapshots=cell(5,1); Records=cell(5,1); Events=cell(500,1); RngEnds=zeros(625,500,'uint32');
[Records{1},Samples]=measure(P,Model,F,0,0);
Snapshots{1}=struct('Model',Model,'Samples',Samples); next=2; seconds=0; Complete=false;
for event=1:500
    O.generation=startingGeneration+event;
    rng(1000000+10000*task.panel+event,'twister');
    [Model,T]=trainer('trainifneeded',Model,Data,Gate,P,O);
    state=rng; RngEnds(:,event)=state.State;
    assert(T.trained && T.updates==20 && Model.iterG==base+20*event);
    assert(isequal(Model.lastTrainingData.cF,F.ActualData.cF) && isequal(Model.lastTrainingData.cI,F.ActualData.cI));
    actual=[F.ActualData.cF;F.ActualData.cI]; if ~F.useSide; actual(:,end)=0; end
    assert(isequal([T.actualCF;T.actualCI],actual));
    Events{event}=struct('updates',event*20,'pre',T.preDiagnostics,'post',T.postDiagnostics,'seconds',T.trainingSeconds);
    seconds=seconds+T.trainingSeconds;
    if ismember(event,[50 100 250 500])
        [Records{next},Samples]=measure(P,Model,F,event*20,seconds);
        assert(abs(Records{next}.endpointRMSE-T.postDiagnostics.allEndpointRMSE)<1e-8);
        Snapshots{next}=struct('Model',Model,'Samples',Samples); next=next+1;
        Table=struct2table(vertcat(Records{1:next-1}));
        Table.arm=repmat(string(task.arm),height(Table),1); Table.problem=repmat(string(class(P)),height(Table),1);
        Table.reset=repmat(task.reset,height(Table),1); Complete=event==500;
        writetable(Table,fullfile(out,[stem,'.csv']));
        save(target,'Snapshots','Table','Events','RngEnds','task','base','Complete','-v7.3');
        fprintf('RESET_PROGRESS %s updates=%d RMSE=%.5f objectiveGap=%.5f seen=%d/%d\n', ...
            stem,event*20,Records{next-1}.endpointRMSE,Records{next-1}.objectiveGap, ...
            Records{next-1}.seenConditions,Records{next-1}.queryConditions);
    end
end
assert(P.FE==0); fprintf('RESET_CASE_COMPLETE %s\n',stem);
end

function [R,S]=measure(P,Model,F,updates,seconds)
saved=rng; cleanup=onCleanup(@()rng(saved)); rng(13579,'twister');
O=struct('sampleSigma',0,'referenceScale',F.ActualData.referenceScale);
[X,Info]=PairBoundaryWGAN_RC('sample',Model,F.QueryC,P,O);
[FitX,~]=PairBoundaryWGAN_RC('sample',Model,[F.ActualData.cF;F.ActualData.cI],P,O);
ctor=str2func(class(P)); Eval=ctor('N',100,'maxFE',100000);
PairGuideCost_RC('start',Eval); accounting=onCleanup(@()PairGuideCost_RC('stop'));
Pop=Eval.Evaluation(X); Y=Pop.objs; constraints=Pop.cons; cost=PairGuideCost_RC('snapshot');
assert(Eval.FE==size(X,1) && cost.CalConRows==size(X,1) && cost.CalObjRows==2*size(X,1));
realX=[F.ActualData.xF;F.ActualData.xI]; realY=[F.ActualData.yF;F.ActualData.yI];
C=[F.ActualData.cF;F.ActualData.cI]; if ~F.useSide; C(:,end)=0; end
seen=ismember(Info.conditions,C,'rows'); distance=min(pdist2(Y,realY),[],2);
conditionalDistance=nan(size(Y,1),1);
for k=find(seen)'
    same=all(C==Info.conditions(k,:),2);
    conditionalDistance(k)=min(vecnorm(realY(same,:)-Y(k,:),2,2));
end
[~,~,Yn]=AssignReferenceVectors_CBS(Y,F.W,F.ActualData.referenceScale);
direction=Yn./max(vecnorm(Yn,2,2),eps); target=Info.conditions(:,1:2); target=target./vecnorm(target,2,2);
angle=acosd(max(-1,min(1,sum(direction.*target,2))));
R=struct('addedUpdates',updates,'trainingSeconds',seconds,'totalGUpdates',Model.iterG, ...
    'endpointRMSE',sqrt(mean((FitX-realX).^2,'all')), ...
    'objectiveGap',mean(distance),'objectiveGapP90',prctile(distance,90), ...
    'realCoverageGap',mean(min(pdist2(realY,Y),[],2)), ...
    'seenObjectiveGap',mean(distance(seen)),'unseenObjectiveGap',mean(distance(~seen)), ...
    'conditionalSeenObjectiveGap',mean(conditionalDistance(seen)), ...
    'directionError',mean(angle),'queryConditions',size(Y,1),'seenConditions',nnz(seen), ...
    'generatedFeasibleRate',mean(all(constraints<=0,2)), ...
    'offlineFullFE',Eval.FE,'offlineCalObjRows',cost.CalObjRows,'offlineCalConRows',cost.CalConRows);
S=struct('X',X,'Y',Y,'constraints',constraints,'conditions',Info.conditions, ...
    'conditionSeen',seen,'FitX',FitX,'rawObjectiveDistances',distance);
if updates==0 && Model.iterG==F.Model.iterG
    assert(isequal(X,F.CachedRaw)); assert(max(abs(Y-F.CachedObjectives),[],'all')<1e-10);
end
end
