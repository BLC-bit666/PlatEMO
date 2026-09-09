function run_fixed_conditions(nWorker)
% Same fixed real samples, optimizer start and batch IDs; only labels move.
if nargin<1; nWorker=2; end
warning('off','all'); maxNumCompThreads(1);
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support')); addCBSPaths(root);
folder=fullfile(root,'Data','PairGuideGPTValidation_20260908');
out=fullfile(folder,'fixed_conditions'); if ~isfolder(out); mkdir(out); end
Tasks=struct('number',{},'seed',{},'moving',{});
for number=[7 8]
    for seed=1:3
        for moving=[false true]
            Tasks(end+1)=struct('number',number,'seed',seed,'moving',moving);
        end
    end
end
pool=parpool('Processes',nWorker); cleanup=onCleanup(@()delete(pool));
errors=strings(numel(Tasks),1);
parfor k=1:numel(Tasks)
    try; runCase(root,folder,out,Tasks(k));
    catch err; errors(k)=string(getReport(err,'extended','hyperlinks','off')); fprintf('%s\n',errors(k)); end
end
save(fullfile(out,'status.mat'),'Tasks','errors');
assert(all(strlength(errors)==0)); fprintf('FIXED_CONDITIONS_COMPLETE\n');
end

function runCase(root,folder,out,task)
maxNumCompThreads(1); addCBSPaths(root);
name=sprintf('LIRCMOP%d_BC',task.number);
stem=sprintf('%s_seed%02d_moving%d',name,task.seed,task.moving);
target=fullfile(out,[stem,'.mat']); if isfile(target); return; end
F=load(fullfile(folder,'fixtures',sprintf('%s_seed%02d.mat',name,task.seed)));
Data=F.Data; Model=F.Model; W=F.W;
ctor=str2func(name); P=ctor('N',100,'maxFE',100000);
O=struct('initialEpoch',1000,'retrainEpoch',20,'nCritic',5,'generation',0, ...
    'retrainChange',0,'retrainGenerations',1,'useSideCondition',true,'stableConditionSpan',false);
% Fix equal side sample exposure independently of changing direction groups.
X=[Data.xF;Data.xI]; sides=[ones(Data.count,1);zeros(Data.count,1)];
[~,keep]=unique([X,sides],'rows','stable'); side=sides(keep);
batch=max(2,2*floor(min(32,numel(keep))/2)); half=batch/2;
saved=rng; rng(90000+task.seed);
Schedule=zeros(2000,batch);
for k=1:2000
    for s=0:1
        ids=find(side==s); order=zeros(0,1);
        while numel(order)<half; order=[order;ids(randperm(numel(ids)))]; end
        Schedule(k,s*half+(1:half))=order(1:half);
    end
end
rng(saved);
Gate=struct('eligible',true); Records=cell(5,1); Snapshots=cell(5,1);
base=Model.iterG; Records{1}=measure(P,Model,Data,W,0,0,NaN,NaN);
Snapshots{1}=struct('Model',Model,'Data',Data,'measurement',Records{1});
next=2; trainingSeconds=0; current=Data; startingGeneration=Model.lastTrainGeneration;
for event=1:100
    current=Data;
    if task.moving
        current.referenceScale.span=F.Spans(event,:);
        rF=AssignReferenceVectors_CBS(current.yF,W,current.referenceScale);
        rI=AssignReferenceVectors_CBS(current.yI,W,current.referenceScale);
        current.cF=[W(rF,:),ones(Data.count,1)]; current.cI=[W(rI,:),zeros(Data.count,1)];
    end
    O.generation=startingGeneration+event;
    O.diagnosticBatchIndices=Schedule((event-1)*20+(1:20),:);
    rng(100000+1000*task.seed+event,'twister');
    [Model,T]=PairBoundaryWGAN_RC('trainifneeded',Model,current,Gate,P,O);
    assert(T.trained && T.updates==20 && Model.iterG==base+20*event);
    trainingSeconds=trainingSeconds+T.trainingSeconds;
    if ismember(event,[1 10 50 100])
        Records{next}=measure(P,Model,current,W,20*event,trainingSeconds, ...
            T.postDiagnostics.allEndpointRMSE,T.preDiagnostics.allEndpointRMSE);
        Snapshots{next}=struct('Model',Model,'Data',current,'measurement',Records{next});
        fprintf('FIXED_CASE %s updates=%d RMSE=%.4g angle=%.2f\n',stem,20*event, ...
            Records{next}.endpointRMSE,Records{next}.directionError);
        next=next+1;
    end
end
assert(P.FE==0);
Table=struct2table(vertcat(Records{:})); Table.problem=repmat(string(name),height(Table),1);
Table.seed=repmat(task.seed,height(Table),1); Table.moving=repmat(task.moving,height(Table),1);
writetable(Table,fullfile(out,[stem,'.csv']));
save(target,'Snapshots','Table','Schedule','task','base','-v7.3');
end

function R=measure(P,Model,Data,W,updates,seconds,postRMSE,preRMSE)
% Offline complete evaluation is recorded separately; never feeds training.
C=[W,zeros(size(W,1),1);W,ones(size(W,1),1)];
O=struct('sampleSigma',0,'referenceScale',Data.referenceScale);
saved=rng; cleanup=onCleanup(@()rng(saved)); rng(13579,'twister');
[X,~]=PairBoundaryWGAN_RC('sample',Model,C,P,O);
[FitX,~]=PairBoundaryWGAN_RC('sample',Model,[Data.cF;Data.cI],P,O);
fitNormalized=(FitX-P.lower)./(P.upper-P.lower);
endpointRMSE=sqrt(mean((fitNormalized-[Data.xF;Data.xI]).^2,'all'));
ctor=str2func(class(P)); Eval=ctor('N',100,'maxFE',100000);
PairGuideCost_RC('start',Eval); accounting=onCleanup(@()PairGuideCost_RC('stop'));
Pop=Eval.Evaluation(X); Y=Pop.objs; constraints=Pop.cons;
cost=PairGuideCost_RC('snapshot'); assert(Eval.FE==200 && cost.CalConRows==200 && cost.CalObjRows==400);
[~,~,Yn]=AssignReferenceVectors_CBS(Y,W,Data.referenceScale);
target=C(:,1:2); target=target./vecnorm(target,2,2); direction=Yn./max(vecnorm(Yn,2,2),eps);
angle=acosd(max(-1,min(1,sum(target.*direction,2))));
distance=min(pdist2(Y,[Data.yF;Data.yI]),[],2);
R=struct('addedUpdates',updates,'trainingSeconds',seconds,'endpointRMSE',endpointRMSE, ...
    'loggedPostRMSE',postRMSE, ...
    'preEndpointRMSE',preRMSE,'directionError',mean(angle),'nearestTrainingObjective',mean(distance), ...
    'labelAccuracy',mean(all(constraints<=0,2)==C(:,end)), ...
    'offlineFullFE',Eval.FE,'offlineCalObjRows',cost.CalObjRows,'offlineCalConRows',cost.CalConRows);
end
