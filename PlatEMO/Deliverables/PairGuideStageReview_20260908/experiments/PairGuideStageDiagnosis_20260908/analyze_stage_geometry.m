function analyze_stage_geometry(directory)
if nargin<1;directory='runs';end
warning('off','all');folder=fileparts(mfilename('fullpath'));rows={};
files=dir(fullfile(folder,directory,'*.mat'));
for k=1:numel(files)
    if strcmp(files(k).name,'status.mat');continue;end
    R=load(fullfile(files(k).folder,files(k).name),'Snapshots','Table','task','Complete');
    if ~isfield(R,'Complete') || ~R.Complete;continue;end
    n=R.task.problemNumber;stage=R.task.stage;
    F=load(fullfile(folder,'fixtures',sprintf('LIRCMOP%d_BC_seed01_%s.mat',n,stage)));
    D=F.Data;X=[D.xF;D.xI];Y=[D.yF;D.yI];C=[D.cF;D.cI];
    [~,keep]=unique([X,C(:,end)],'rows','stable');X=X(keep,:);Y=Y(keep,:);C=C(keep,:);
    realBase=basePart(X,n);energy=Y-realBase;
    for j=1:numel(R.Snapshots)
        S=R.Snapshots{j}.Samples;generatedBase=basePart(S.X,n);generatedEnergy=S.Y-generatedBase;
        targetY=zeros(size(S.Y));targetBase=zeros(size(S.Y));targetEnergy=zeros(size(S.Y));
        for t=find(S.seen)'
            same=all(C==S.conditions(t,:),2);
            targetY(t,:)=mean(Y(same,:),1);targetBase(t,:)=mean(realBase(same,:),1);targetEnergy(t,:)=mean(energy(same,:),1);
        end
        iy=find(S.seen);deltaY=mean(S.Y(iy,:)-targetY(iy,:),1);
        deltaBase=mean(generatedBase(iy,:)-targetBase(iy,:),1);
        deltaEnergy=mean(generatedEnergy(iy,:)-targetEnergy(iy,:),1);
        assert(max(abs(deltaY-deltaBase-deltaEnergy))<1e-10);
        f=C(:,end)==1;i=~f;
        reverseF=mean(min(pdist2(Y(f,:),S.Y),[],2));reverseI=mean(min(pdist2(Y(i,:),S.Y),[],2));
        realF=X(f,:);conditionsF=C(f,:);[~,order]=sort(conditionsF(:,1));realF=realF(order,:);conditionsF=conditionsF(order,:);
        [found,at]=ismember(conditionsF,S.conditions,'rows');assert(all(found));generatedF=S.X(at,:);
        realStep=sqrt(mean(diff(realF,1,1).^2,2));genStep=sqrt(mean(diff(generatedF,1,1).^2,2));
        rows{end+1}=struct('problem',sprintf('LIRCMOP%d_BC',n),'stage',string(stage), ...
            'loss',string(R.task.loss),'stream',R.task.stream,'addedUpdates',R.Table.addedUpdates(j), ...
            'reverseFeasibleCoverage',reverseF,'reverseInfeasibleCoverage',reverseI, ...
            'f1Offset',deltaY(1),'f2Offset',deltaY(2),'f1BaseOffset',deltaBase(1),'f2BaseOffset',deltaBase(2), ...
            'f1EnergyOffset',deltaEnergy(1),'f2EnergyOffset',deltaEnergy(2), ...
            'medianTrueFeasibleNeighborRMS',median(realStep),'medianGeneratedFeasibleNeighborRMS',median(genStep));
    end
end
writetable(struct2table(vertcat(rows{:})),fullfile(folder,[directory,'_geometry.csv']));
fprintf('STAGE_GEOMETRY_COMPLETE %s %d checkpoints; only cached X/Y, no oracle calls\n',directory,numel(rows));
end

function base=basePart(X,n)
if ismember(n,[5 7]);second=1-sqrt(X(:,1));else;second=1-X(:,1).^2;end
base=[X(:,1)+.7057,second+.7057];
end
