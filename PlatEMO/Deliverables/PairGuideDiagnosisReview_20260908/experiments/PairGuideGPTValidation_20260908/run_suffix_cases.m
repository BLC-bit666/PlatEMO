function run_suffix_cases(nWorker)
% One intervention batch, then common fallback-only backbone for 5000 FE.
if nargin<1; nWorker=2; end
warning('off','all'); maxNumCompThreads(1);
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support')); addCBSPaths(root);
folder=fileparts(mfilename('fullpath')); addpath(folder);
out=fullfile(folder,'suffix'); if ~isfolder(out); mkdir(out); end
Tasks=struct('number',{},'seed',{},'stage',{},'mode',{});
for number=[7 8]
    for seed=1:3
        for stage=1:2
            for mode=["raw","midpoint","de","pair_only"]
                Tasks(end+1)=struct('number',number,'seed',seed,'stage',stage,'mode',mode);
            end
        end
    end
end
if nWorker==0
    suffix_case(root,folder,out,Tasks(1)); fprintf('SUFFIX_SMOKE_COMPLETE\n'); return;
end
pool=parpool('Processes',nWorker); cleanup=onCleanup(@()delete(pool)); errors=strings(numel(Tasks),1);
parfor k=1:numel(Tasks)
    try; suffix_case(root,folder,out,Tasks(k));
    catch err; errors(k)=string(getReport(err,'extended','hyperlinks','off')); fprintf('%s\n',errors(k)); end
end
save(fullfile(out,'status.mat'),'Tasks','errors'); assert(all(strlength(errors)==0));
fprintf('SUFFIX_CASES_COMPLETE\n');
end

function suffix_case(root,folder,out,Task)
maxNumCompThreads(1); addCBSPaths(root); addpath(folder);
name=sprintf('LIRCMOP%d_BC',Task.number); fe=[50000 70000]; fe=fe(Task.stage);
stem=sprintf('%s_seed%02d_FE%06d_%s',name,Task.seed,fe,Task.mode);
target=fullfile(out,[stem,'.mat']); if isfile(target); return; end
F=load(fullfile(folder,'fixtures',sprintf('%s_seed%02d.mat',name,Task.seed)),'Prefixes','W');
Cached=load(fullfile(folder,'fixtures',sprintf('%s_seed%02d_suffix.mat',name,Task.seed)),'Suffix');
Prefix=F.Prefixes{Task.stage}; S=Prefix.state; W=F.W;
rng(Task.seed,'twister'); ctor=str2func(name); P=ctor('N',100,'maxFE',fe+5000); P.FE=fe;
state=rng; data=state.State; if iscell(data); data=data{1}; end
backboneSeed=mod(sum(double(data(1:min(16,numel(data))))),2^32-1);
P1=SOLUTION(S.p1Decs,S.p1Objs,S.p1Cons); P2=SOLUTION(S.p2Decs,S.p2Objs,S.p2Cons);
Fit1=Cached.Suffix{Task.stage}.Fitness1; Fit2=Cached.Suffix{Task.stage}.Fitness2;
Archive=S.archive; InitialArchive=Archive; Config=PairGuideCore.mainlineDefaults();
Options=Config; Options.W=W; Options.guideQuota=20; Options.referenceScale=S.referenceScale;
Pending=Prefix.query.pending; OriginalQ=Pending.decs; ParentIndices=[];
if Task.mode=="midpoint"
    [~,~,normalized]=AssignReferenceVectors_CBS(S.p1Objs,W,S.referenceScale);
    normalized=normalized./max(vecnorm(normalized,2,2),eps);
    request=W(Pending.refs,:); request=request./vecnorm(request,2,2);
    [~,ParentIndices]=max(normalized*request',[],1); ParentIndices=ParentIndices';
    Pending.decs=S.p1Decs(ParentIndices,:)+0.5*(OriginalQ-S.p1Decs(ParentIndices,:));
elseif Task.mode=="de"
    Pending=struct();
elseif Task.mode=="pair_only"
    Options.pairOnly=true; rng(700000+Task.seed+fe,'twister');
    [~,Info]=PairBoundaryArchive_RC('querycontexts',Archive,W,Options,500);
    [Raw,Sample]=suffix_native_helpers('pair_only',Archive,Info,P);
    Info.generatedF=Sample.generatedF; Info.generatedI=Sample.generatedI;
    Options.currentDecs=[S.p1Decs;S.p2Decs];
    [X,refs,ids,Pool]=PairBoundaryArchive_RC('selectcandidates',Raw,Info,Archive,S.referenceScale,P,Options);
    k=Pool.keepIdx;
    Pending=struct('decs',X,'refs',refs,'ids',ids,'xf',Pool.xf(k,:),'xi',Pool.xi(k,:), ...
        'yf',Pool.yf(k,:),'sides',nan(numel(k),1),'productionFE',fe, ...
        'productionGeneration',Prefix.generation,'modelVersion',0);
end
Options.pairOnly=false;
PairGuideCost_RC('start',P); cleanup=onCleanup(@()PairGuideCost_RC('stop'));
Trajectory=zeros(26,4); Trajectory(1,:)=[fe,metrics(P,P1),0];
History=cell(25,1); timer=tic;
for j=1:25
    generation=Prefix.generation+j;
    [O1,U]=suffix_native_helpers('p1',P,P1,Fit1,100,Pending,Config,backboneSeed,generation);
    O2=suffix_native_helpers('p2',P,P2,Fit2,100,backboneSeed,generation);
    Union=[P1,P2,O1,O2];
    [P1,Fit1]=EnvironmentalSelection_CBS([P1,O1,O2],100,true);
    [P2,Fit2]=EnvironmentalSelection_CBS([P2,O1,O2],100,false);
    [Archive,Scale,A]=PairBoundaryArchive_RC('update',Archive,P1,Union,W,P,Options,U,P.FE);
    Options.referenceScale=Scale;
    History{j}=struct('FE',P.FE,'use',U,'guidedInfeasible',A.guidedTightenedInfeasible, ...
        'guidedFeasible',A.guidedTightenedFeasible,'activePairs',nnz(Archive.active), ...
        'survivedP1',nnz(ismember(U.childDecs,P1.decs,'rows')));
    Trajectory(j+1,:)=[P.FE,metrics(P,P1),size(U.childDecs,1)];
    if j==1
        First=struct('p1Decs',P1.decs,'p1Objs',P1.objs,'p2Decs',P2.decs, ...
            'p2Objs',P2.objs,'archive',Archive,'use',U);
        if Task.mode=="raw"
            Expected=Cached.Suffix{Task.stage}.ExpectedNext;
            assert(isequal(P1.decs,Expected.p1Decs) && isequal(P2.decs,Expected.p2Decs) && ...
                isequaln(Archive,Expected.archive),'SuffixReplay:FirstStepMismatch', ...
                'The unmodified first step must exactly replay the original search.');
        end
    end
    Pending=struct(); % Isolate propagation of the first candidate batch.
end
Cost=PairGuideCost_RC('snapshot'); assert(P.FE==fe+5000 && Cost.CalConRows==5000 && Cost.CalObjRows==10000);
R=struct('problem',string(name),'seed',Task.seed,'prefixFE',fe,'mode',Task.mode,'backboneSeed',backboneSeed, ...
    'initialIGD',Trajectory(1,2),'finalIGD',Trajectory(end,2),'finalHV',Trajectory(end,3), ...
    'IGDAUC',trapz(Trajectory(:,1),Trajectory(:,2))/5000, ...
    'firstSelected',History{1}.use.selected,'firstSurvivedP1',History{1}.survivedP1, ...
    'firstGuidedInfeasible',History{1}.guidedInfeasible,'suffixFE',5000,'wallSeconds',toc(timer));
writetable(struct2table(R),fullfile(out,[stem,'.csv']));
Final=struct('p1Decs',P1.decs,'p1Objs',P1.objs,'p1Cons',P1.cons,'archive',Archive);
save(target,'R','Trajectory','History','First','Final','InitialArchive','OriginalQ','ParentIndices','Cost','-v7.3');
fprintf('SUFFIX %s IGD=%.6g AUC=%.6g\n',stem,R.finalIGD,R.IGDAUC);
end

function values=metrics(P,Population)
saved=rng; cleanup=onCleanup(@()rng(saved)); rng(314159,'twister');
values=[P.CalMetric('IGD',Population),P.CalMetric('HV',Population)];
end
