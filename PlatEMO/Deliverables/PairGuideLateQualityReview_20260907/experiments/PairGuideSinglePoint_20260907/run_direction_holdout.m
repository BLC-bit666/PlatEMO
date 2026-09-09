function run_direction_holdout(O,arm,baseArm)
% Fixed corrected-archive data; every network starts without held-out history.
if nargin<3; baseArm=""; end
warning('off','all'); maxNumCompThreads(1);
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support')); addCBSPaths(root);
folder=fullfile(root,'Data','PairGuideSinglePoint_20260907','holdout'); if ~isfolder(folder); mkdir(folder); end
rows=cell(4,1); next=0;
for number=[7 8]
    name=sprintf('LIRCMOP%d_BC',number); ctor=str2func(name); P=ctor('N',100,'maxFE',100000);
    source=fullfile(root,'Data','PairGuideFilteredArchive_5to8_R3_20260907','analysis','figures',name,'run_01','plot_data.mat');
    Saved=load(source,'States'); A=Saved.States{end}.trainingState.archive;
    [W,~]=UniformPoint(100,2); [Full,~]=PairBoundaryArchive_RC('trainingdata',A,W,P,struct());
    [~,rf]=ismember(Full.cF(:,1:2),W,'rows'); [~,ri]=ismember(Full.cI(:,1:2),W,'rows');
    available=unique([rf;ri]); center=available(ceil(numel(available)/2));
    heldRefs=max(1,center-8):min(100,center+8);
    removed=ismember(rf,heldRefs) | ismember(ri,heldRefs);
    assert(nnz(removed)>0 && nnz(~removed)>=8);
    [QueryC,Info]=PairBoundaryArchive_RC('querycontexts',A,W,struct(),500);
    for holdout=[false true]
        D=Full;
        if holdout
            for field=["xF","xI","delta","w","ref","id","cF","cI"]
                D.(field)=D.(field)(~removed,:);
            end
            D.count=nnz(~removed);
            assert(~any(ismember([D.cF(:,1:2);D.cI(:,1:2)],W(heldRefs,:),'rows')));
        end
        TrainO=O; TrainO.generation=1; Previous=[];
        if strlength(baseArm)>0
            Base=load(fullfile(folder,sprintf('%s_%s_held%d.mat',name,baseArm,holdout)),'Model','D');
            assert(isequaln(Base.D,D),'Continuation must use exactly the same full/held-out data.');
            Previous=Base.Model; TrainO.retrainEpoch=O.initialEpoch-Previous.iterG;
            assert(TrainO.retrainEpoch>0); TrainO.generation=Previous.lastTrainGeneration+10;
        end
        rng(20260907,'twister'); timer=tic;
        [Model,T]=PairBoundaryWGAN_RC('trainifneeded',Previous,D,struct('eligible',true),P,TrainO);
        assert(Model.ready && T.trained && Model.iterG==O.initialEpoch);
        rng(20260908,'twister'); [X,Sample]=PairBoundaryWGAN_RC('sample',Model,QueryC,P,TrainO);
        Eval=ctor(); PairGuideCost_RC('start',Eval); cleanup=onCleanup(@()PairGuideCost_RC('stop'));
        Y=Eval.CalObj(X); C=Eval.CalCon(X); correct=(all(C<=0,2)==QueryC(:,end));
        [~,~,Yn]=AssignReferenceVectors_CBS(Y,W,Full.referenceScale);
        target=QueryC(:,1:2); target=target./vecnorm(target,2,2);
        direction=Yn./max(vecnorm(Yn,2,2),eps);
        angles=acosd(max(-1,min(1,sum(target.*direction,2))));
        hidden=ismember(Info.refs,heldRefs);
        feasibleTargets=A.yf(A.active,:); dominated=false(500,1);
        for k=1:size(feasibleTargets,1)
            dominated=dominated | (all(feasibleTargets(k,:)<=Y+1e-12,2) & any(feasibleTargets(k,:)<Y-1e-12,2));
        end
        useful=all(C<=0,2) & ~dominated;
        next=next+1;
        M=struct('problem',string(name),'arm',string(arm),'holdout',holdout, ...
            'trainingPairs',D.count,'removedPairs',nnz(removed),'heldFirst',heldRefs(1),'heldLast',heldRefs(end), ...
            'totalUpdates',Model.iterG,'continuedFrom',string(baseArm), ...
            'heldAngle',mean(angles(hidden)),'otherAngle',mean(angles(~hidden)), ...
            'heldLabel',mean(correct(hidden)),'otherLabel',mean(correct(~hidden)), ...
            'heldJoint',mean(correct(hidden) & angles(hidden)<=5 & ~dominated(hidden)), ...
            'otherJoint',mean(correct(~hidden) & angles(~hidden)<=5 & ~dominated(~hidden)), ...
            'heldUseful',mean(useful(hidden)),'otherUseful',mean(useful(~hidden)), ...
            'trainingSeconds',T.trainingSeconds,'wallSeconds',toc(timer));
        OfflineCost=PairGuideCost_RC('snapshot'); assert(Eval.FE==0); clear cleanup;
        rows{next}=M;
        file=fullfile(folder,sprintf('%s_%s_held%d.mat',name,arm,holdout));
        save(file,'M','Model','T','D','Full','O','X','Y','C','Sample','QueryC','heldRefs','removed', ...
            'angles','correct','hidden','OfflineCost','source','-v7.3');
        fprintf('HOLDOUT %s held=%d pairs=%d angle=%.2f label=%.3f joint=%.3f\n', ...
            name,holdout,D.count,M.heldAngle,M.heldLabel,M.heldJoint);
    end
end
writetable(struct2table(vertcat(rows{:})),fullfile(folder,char(string(arm)+"_summary.csv")));
end
