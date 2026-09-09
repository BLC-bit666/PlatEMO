function run_gradient_comparison(nWorker,stages,numbers,streams,generatorLR,runDirectory)
if nargin<1;nWorker=6;end
if nargin<2;stages={'early','middle','late'};end
if nargin<3;numbers=5:8;end
if nargin<4;streams=1:3;end
if nargin<5;generatorLR=.001;end
if nargin<6;runDirectory='runs';end
warning('off','all');maxNumCompThreads(1);
folder=fileparts(mfilename('fullpath'));root=fileparts(fileparts(folder));
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support'));addCBSPaths(root);addpath(folder);
out=fullfile(folder,runDirectory);if ~isfolder(out);mkdir(out);end
Tasks=struct('problemNumber',{},'stage',{},'stream',{},'loss',{},'generatorLR',{},'directory',{});
for n=numbers
    for stage=stages
        for stream=streams
            for loss={'adversarial','distance'}
                Tasks(end+1)=struct('problemNumber',n,'stage',stage{1},'stream',stream,'loss',loss{1},'generatorLR',generatorLR,'directory',runDirectory); %#ok<AGROW>
            end
        end
    end
end
pool=parpool('Processes',nWorker);cleanup=onCleanup(@()delete(pool));errors=strings(numel(Tasks),1);
parfor k=1:numel(Tasks)
    try;runOne(root,folder,Tasks(k));
    catch err;errors(k)=string(getReport(err,'extended','hyperlinks','off'));fprintf('%s\n',errors(k));end
end
save(fullfile(folder,sprintf('gradient_status_%s_%s_%d.mat',runDirectory,stages{1},numbers(1))),'Tasks','errors');
assert(all(strlength(errors)==0));fprintf('GRADIENT_COMPARISON_COMPLETE %d runs\n',numel(Tasks));
end

function runOne(root,folder,task)
maxNumCompThreads(1);addCBSPaths(root);addpath(folder);
stem=sprintf('LIRCMOP%d_BC_seed01_%s',task.problemNumber,task.stage);
name=sprintf('%s_%s_stream%d',stem,task.loss,task.stream);
file=fullfile(folder,task.directory,[name,'.mat']);
if isfile(file);R=load(file,'Complete');if R.Complete;return;end;end
F=load(fullfile(folder,'fixtures',[stem,'.mat']));B=load(fullfile(folder,'baselines',[stem,'.mat']));
M=F.Model;M.probeRecords=[];base=M.iterG;
ctor=str2func(F.Meta.problem);P=ctor('N',100,'maxFE',100000);
O=struct('retrainEpoch',20,'nCritic',5,'lrG',task.generatorLR,'lrD',.001,'gpLambda',10, ...
    'miniBatch',32,'trainingSigma',0,'sampleSigma',0,'retrainChange',0,'retrainGenerations',1, ...
    'useSideCondition',true,'stableConditionSpan',false,'probeLoss',task.loss,'probeBase',base);
Gate=struct('eligible',true);RngEnds=zeros(625,100,'uint32');
Records=cell(4,1);Snapshots=cell(4,1);Records{1}=B.R;Records{1}.addedUpdates=0;Records{1}.trainingSeconds=0;
Snapshots{1}=struct('Model',M,'Samples',B.S);index=2;seconds=0;Complete=false;
for event=1:100
    O.generation=F.Model.lastTrainGeneration+event;
    rng(2100000+10000*task.stream+event,'twister');
    [M,T]=PairBoundaryWGAN_StageProbe('trainifneeded',M,F.RawData,Gate,P,O);
    state=rng;RngEnds(:,event)=state.State;
    assert(T.trained && T.updates==20 && M.iterG==base+20*event);
    assert(isequal(M.lastTrainingData.cF,F.Data.cF) && isequal(M.lastTrainingData.cI,F.Data.cI));
    seconds=seconds+T.trainingSeconds;
    if ismember(event,[1 10 100])
        [row,S]=measure_stage_model(M,F);row.addedUpdates=20*event;row.trainingSeconds=seconds;
        Records{index}=row;Snapshots{index}=struct('Model',M,'Samples',S);index=index+1;
        Table=struct2table(vertcat(Records{1:index-1}));
        Table.problem=repmat(string(F.Meta.problem),height(Table),1);Table.stage=repmat(string(task.stage),height(Table),1);
        Table.stream=repmat(task.stream,height(Table),1);Table.loss=repmat(string(task.loss),height(Table),1);
        Complete=event==100;
        writetable(Table,fullfile(folder,task.directory,[name,'.csv']));
        writetable(struct2table(M.probeRecords),fullfile(folder,task.directory,[name,'_steps.csv']));
        save(file,'Snapshots','Table','RngEnds','task','Complete','base','-v7.3');
        fprintf('GRADIENT_PROGRESS %s updates=%d exact=%.5f seen=%.5f cover=%.5f\n', ...
            name,event*20,row.exactDistance,row.seenObjectiveGap,row.reverseCoverageGap);
    end
end
assert(P.FE==0);fprintf('GRADIENT_CASE_COMPLETE %s\n',name);
end
