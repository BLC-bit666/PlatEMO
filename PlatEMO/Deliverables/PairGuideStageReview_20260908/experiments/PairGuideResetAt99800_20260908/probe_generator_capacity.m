function probe_generator_capacity(hidden)
% Exploratory representability check, NOT a production loss recommendation.
% Default uses the same G architecture and fresh seed 600001; optional hidden
% width checks only original7/8. Full-batch centroid MSE differs from GAN
% training in several respects, so this cannot isolate a loss.
if nargin<1; hidden=[32 32]; end
warning('off','all'); maxNumCompThreads(1); folder=fileparts(mfilename('fullpath'));
root=fileparts(fileparts(folder)); addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support')); addCBSPaths(root); addpath(folder);
addpath(fullfile(root,'Data','PairGuideGPTValidation_20260908'));
arms={'original','stable_side','stable_no_side'}; directory='capacity';
if ~isequal(hidden,[32 32]); arms={'original'}; directory='capacity_wide'; end
out=fullfile(folder,directory); if ~isfolder(out); mkdir(out); end
for arm=arms
    for n=7:8
        stem=sprintf('%s_LIRCMOP%d_BC',arm{1},n); target=fullfile(out,[stem,'.mat']);
        if isfile(target); continue; end
        F=load(fullfile(folder,'fixtures',[stem,'.mat'])); H=load(fullfile(folder,'geometry',[stem,'.mat']),'codes','means');
        Template=F.Model; Template.generatorHidden=hidden;
        Model=fresh_model(Template,600001); avg=[]; avgSq=[];
        input=dlarray(single([zeros(size(H.codes,1),Model.zDim),H.codes]'),'CB');
        targets=dlarray(single(2*H.means'-1),'CB');
        Records=cell(4,1); Snapshots=cell(4,1); next=1;
        timer=tic;
        for step=0:10000
            if step>0
                gradients=dlfeval(@gradient,Model.netG,input,targets);
                [Model.netG,avg,avgSq]=adamupdate(Model.netG,gradients,avg,avgSq,step,.001,0,.9);
            end
            if ismember(step,[0 1000 5000 10000])
                [R,S]=measure(Model,F,H,n,step,toc(timer)); Records{next}=R;
                Snapshots{next}=struct('Model',Model,'Samples',S); next=next+1;
                fprintf('CAPACITY_PROGRESS %s steps=%d prototypeRMSE=%.5f seenGap=%.5f\n',stem,step,R.prototypeRMSE,R.seenObjectiveGap);
            end
        end
        Table=struct2table(vertcat(Records{:})); Table.arm=repmat(string(arm{1}),height(Table),1);
        Table.problem=repmat(string(sprintf('LIRCMOP%d_BC',n)),height(Table),1);
        Table.hiddenWidth=repmat(hidden(1),height(Table),1);
        save(target,'Table','Snapshots','avg','avgSq','-v7.3'); writetable(Table,fullfile(out,[stem,'.csv']));
    end
end
fprintf('CAPACITY_PROBE_COMPLETE %d cases hidden=%d; exploratory full-batch representability only\n',2*numel(arms),hidden(1));
end

function gradients=gradient(net,input,targets)
    prediction=forward(net,input); loss=mean((prediction-targets).^2,'all');
    gradients=dlgradient(loss,net.Learnables);
end

function [R,S]=measure(Model,F,H,n,step,seconds)
ctor=str2func(sprintf('LIRCMOP%d_BC',n)); P=ctor('N',100,'maxFE',100000);
O=struct('sampleSigma',0,'referenceScale',F.ActualData.referenceScale);
[X,Info]=PairBoundaryWGAN_RC('sample',Model,F.QueryC,P,O);
[Fit,~]=PairBoundaryWGAN_RC('sample',Model,H.codes,P,O);
PairGuideCost_RC('start',P); accounting=onCleanup(@()PairGuideCost_RC('stop'));
Pop=P.Evaluation(X); Y=Pop.objs; cost=PairGuideCost_RC('snapshot');
assert(P.FE==size(X,1) && cost.CalObjRows==2*P.FE && cost.CalConRows==P.FE);
seen=ismember(Info.conditions,H.codes,'rows'); distance=min(pdist2(Y,[F.ActualData.yF;F.ActualData.yI]),[],2);
R=struct('updates',step,'seconds',seconds,'prototypeRMSE',sqrt(mean((Fit-H.means).^2,'all')), ...
    'objectiveGap',mean(distance),'seenObjectiveGap',mean(distance(seen)), ...
    'unseenObjectiveGap',mean(distance(~seen)),'offlineFullFE',P.FE);
S=struct('X',X,'Y',Y,'conditions',Info.conditions,'conditionSeen',seen,'Fit',Fit);
end
