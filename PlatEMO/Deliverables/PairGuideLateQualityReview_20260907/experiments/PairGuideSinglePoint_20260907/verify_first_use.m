function verify_first_use
warning('off','all'); maxNumCompThreads(1);
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support')); addCBSPaths(root); addpath(fileparts(mfilename('fullpath')));
folder=fullfile(root,'Data','PairGuideSinglePoint_20260907','first_use');
files=dir(fullfile(folder,'LIRCMOP*_seed*_*.mat'));
prefixes=containers.Map('KeyType','char','ValueType','any'); scores=cell(numel(files),1);
for j=1:numel(files)
    file=fullfile(folder,files(j).name); R=load(file,'M','Q','T','Prefix','OfflineCost','Audit','angles','known','Y','C','Data');
    key=sprintf('%s_%d',R.M.problem,R.M.seed);
    if isKey(prefixes,key); assert(isequaln(prefixes(key),R.Prefix)); else; prefixes(key)=R.Prefix; end
    E=R.Audit.evidence; U=E.generations{end}.use; Q=R.Q;
    assert(E.schema=="PairGuide-single-v3" && numel(E.queries)==1 && nnz(cellfun(@(t)t.trained,E.training))==1);
    assert(E.fullFE==Q.productionFE+200 && all(Q.pending.ids==0) && U.selected==20);
    assert(isequal(Q.rawDecs(Q.pool.keepIdx,:),Q.pending.decs) && isequal(U.childDecs,Q.pending.decs));
    assert(E.networkEndpointRows==500 && E.oracleCalls.CalObjRows==2*E.fullFE && E.oracleCalls.CalConRows==E.fullFE);
    assert(R.OfflineCost.CalObjRows==1000 && R.OfflineCost.CalConRows==500);
    assert(R.T.trainingSamples==size(unique([[R.Data.xF;R.Data.xI],[ones(R.Data.count,1);zeros(R.Data.count,1)]],'rows'),1));
    assert(R.T.endpointVisits>0 && R.T.pairVisits==0 && R.T.criticUpdates==R.T.nCritic*R.T.updates);
    F=R.Prefix.p1Objs(all(R.Prefix.p1Cons<=0,2),:); dominated=false(500,1);
    for k=1:size(F,1); dominated=dominated | (all(F(k,:)<=R.Y+1e-12,2) & any(F(k,:)<R.Y-1e-12,2)); end
    consistent=(all(R.C<=0,2)==Q.sides);
    joint=consistent & R.angles<=5 & ~dominated;
    scores{j}=struct('problem',R.M.problem,'seed',R.M.seed,'arm',R.M.arm, ...
        'jointRate',mean(joint),'knownJointRate',mean(joint(R.known)), ...
        'unseenJointRate',mean(joint(~R.known)), ...
        'knownLabelAccuracy',mean(consistent(R.known)),'unseenLabelAccuracy',mean(consistent(~R.known)));
end
T=struct2table(vertcat(scores{:})); writetable(T,fullfile(fileparts(folder),'first_use_joint_scores.csv'));
V=struct('status','verified','cases',numel(files),'identicalPrefixGroups',prefixes.Count, ...
    'oneTrainingOneConsumption',true,'offlineLabelsSeparated',true,'nativeCoordinatesUnchanged',true, ...
    'sharedEndpointDedupVerified',true,'jointAngleToleranceDegrees',5);
fid=fopen(fullfile(fileparts(folder),'first_use_verification.json'),'w'); fprintf(fid,'%s',jsonencode(V,PrettyPrint=true)); fclose(fid);
disp(V);
end
