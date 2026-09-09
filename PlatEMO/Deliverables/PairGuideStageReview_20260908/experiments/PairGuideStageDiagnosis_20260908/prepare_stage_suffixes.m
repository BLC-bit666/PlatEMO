function prepare_stage_suffixes
% Cached original prefixes and exact survivor fitness, no oracle calls.
warning('off','all');maxNumCompThreads(1);
folder=fileparts(mfilename('fullpath'));root=fileparts(fileparts(folder));
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support'));addCBSPaths(root);
out=fullfile(folder,'suffix_fixtures');if ~isfolder(out);mkdir(out);end
rows={};
for n=5:8
    source=fullfile(root,'Data','PairGuideSinglePoint_20260907','full_run',sprintf('LIRCMOP%d_BC_seed01_cgan.mat',n));
    R=load(source,'Audit');E=R.Audit.evidence;times=cellfun(@(p)p.observationFE,E.population);W=E.W;
    stages={'early','middle','late'};fes=[20000 50000 80000];
    for j=1:3
        target=fullfile(out,sprintf('LIRCMOP%d_BC_seed01_%s.mat',n,stages{j}));
        fe=fes(j);k=find(times==fe,1);generation=k-1;previous=E.population{k-1};
        O1=E.evaluations{2*generation};O2=E.evaluations{2*generation+1};
        O=[SOLUTION(O1.decisions,O1.objectives,O1.constraints),SOLUTION(O2.decisions,O2.objectives,O2.constraints)];
        Prev1=SOLUTION(previous.p1Decs,previous.p1Objs,previous.p1Cons);Prev2=SOLUTION(previous.p2Decs,previous.p2Objs,previous.p2Cons);
        [P1,Fitness1]=EnvironmentalSelection_CBS([Prev1,O],100,true);
        [P2,Fitness2]=EnvironmentalSelection_CBS([Prev2,O],100,false);
        State=E.population{k};assert(isequal(P1.decs,State.p1Decs) && isequal(P2.decs,State.p2Decs));
        q=find(cellfun(@(q)q.productionFE==fe,E.queries),1);Query=struct();Pending=struct();queryAvailable=~isempty(q);
        if queryAvailable;Query=E.queries{q};Pending=Query.pending;end
        ExpectedNext=E.population{k+1};
        Prefix=struct('state',State,'generation',generation,'seed',1,'problem',sprintf('LIRCMOP%d_BC',n), ...
            'stage',stages{j},'FE',fe,'queryAvailable',queryAvailable,'query',Query,'pending',Pending);
        if ~isfile(target);save(target,'Prefix','W','Fitness1','Fitness2','ExpectedNext','source','-v7.3');end
        rows{end+1}=struct('problemNumber',n,'stage',string(stages{j}),'prefixFE',fe, ...
            'queryAvailable',queryAvailable,'activeArchivePairs',nnz(State.archive.active));
    end
end
writetable(struct2table(vertcat(rows{:})),fullfile(folder,'suffix_prefixes.csv'));
fprintf('STAGE_SUFFIX_FIXTURES_READY 12 exact prefixes; no oracle calls\n');
end
