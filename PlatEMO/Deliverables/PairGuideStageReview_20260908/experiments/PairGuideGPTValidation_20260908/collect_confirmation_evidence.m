function collect_confirmation_evidence(nWorker)
% Incremental read-only audit and boundary proxies for completed held-out runs.
if nargin<1;nWorker=2;end
warning('off','all');maxNumCompThreads(1);folder=fileparts(mfilename('fullpath'));addpath(folder);
lock=jsondecode(fileread(fullfile(folder,'confirmation_lock.json')));
Tasks=struct('file',{},'selected',{});
for selected=[false true]
 name='confirmation_controls';if selected;name='confirmation_selected';end
 files=dir(fullfile(folder,name,'LIRCMOP*_seed*_*.mat'));
 for k=1:numel(files);Tasks(end+1)=struct('file',fullfile(files(k).folder,files(k).name),'selected',selected);end %#ok<AGROW>
end
for name=["confirmation_verification","confirmation_boundary_proxies","confirmation_initials","confirmation_usage"]
 if ~isfolder(fullfile(folder,name));mkdir(fullfile(folder,name));end
end
pool=parpool('Processes',nWorker);cleanup=onCleanup(@()delete(pool));
parfor k=1:numel(Tasks)
 maxNumCompThreads(1);addpath(folder);
 [~,stem]=fileparts(Tasks(k).file);tag="control_";if Tasks(k).selected;tag="selected_";end
 output=fullfile(folder,'confirmation_verification',tag+stem+".csv");
 usage=fullfile(folder,'confirmation_usage',tag+stem+".csv");
 if isfile(output) && (~endsWith(stem,'_cgan') || isfile(usage));continue;end
 try
  audit_one(Tasks(k),folder,lock,output);
 catch err
  fprintf('DEFERRED %s: %s\n',Tasks(k).file,getReport(err,'extended','hyperlinks','off'));
 end
end
files=dir(fullfile(folder,'confirmation_verification','*.csv'));tables=cell(numel(files),1);
for k=1:numel(files);tables{k}=readtable(fullfile(files(k).folder,files(k).name),'TextType','string');end
if ~isempty(tables);writetable(vertcat(tables{:}),fullfile(folder,'confirmation_verification.csv'));end
fprintf('CONFIRMATION_EVIDENCE_COLLECTED %d\n',numel(files));
end
function audit_one(Task,folder,lock,output)
R=load(Task.file,'Record','Audit');E=R.Audit.evidence;
assert(R.Record.finalFE==100000&&R.Record.D==30&&E.fullFE==100000);
assert(E.oracleCalls.CalObjRows==200000&&E.oracleCalls.CalConRows==100000);
assert(E.objectiveOnlyFE==0&&E.constraintOnlyFE==0);
arm=R.Record.mode;if arm=="cgan";arm="original";end
if Task.selected;arm=string(lock.selected);end
trained=E.training(cellfun(@(t)t.trained,E.training));initialPrefixExact=NaN;
initial=fullfile(folder,'confirmation_initials',sprintf('%s_seed%02d.mat',R.Record.problem,R.Record.seed));
if R.Record.mode=="cgan" && ~isempty(trained)
 assert(trained{1}.updates==1000&&all(cellfun(@(t)t.updates==20,trained(2:end))));
 assert(E.lastModel.iterG==sum(cellfun(@(t)t.updates,trained))&&E.lastModel.iterC==5*E.lastModel.iterG);
 assert(~isempty(E.lastModel.avgG)&&~isempty(E.lastModel.avgC));
 if Task.selected;span=E.firstModel.conditionSpan;end
 for j=1:numel(trained)
  T=trained{j};P=E.population{T.generation+1};assert(size(T.actualCF,2)==3);
  assert(isequal(T.conditionScale.minimum,P.referenceScale.minimum));
  if Task.selected
   assert(isequal(T.conditionScale.span,span));
   if string(lock.selected)=="stable_no_side";assert(all(T.actualCF(:,end)==0)&&all(T.actualCI(:,end)==0));end
  else
   assert(isequal(T.conditionScale.span,P.referenceScale.span)&&T.useSideCondition);
  end
 end
 for j=1:numel(E.queries)
  Q=E.queries{j};U=E.generations{Q.generation+1}.use;
  assert(isequal(Q.pending.decs,Q.rawDecs(Q.pool.keepIdx,:))&&all(Q.pending.ids==0));
  assert(all(ismember(U.childDecs,Q.pending.decs,'rows'))&&U.selected+U.fallback==U.requested);
 end
 Prefix=E.population{E.queries{1}.generation+1};Raw=E.queries{1}.rawDecs;
 if Task.selected
  B=load(initial,'Prefix','Raw');assert(isequaln(Prefix,B.Prefix));initialPrefixExact=1;
  if string(lock.selected)=="stable_side";assert(isequal(Raw,B.Raw));end
 else
  save(initial,'Prefix','Raw');initialPrefixExact=1;
 end
elseif R.Record.mode=="fallback_only"
 assert(isempty(E.training)&&isempty(E.queries));
elseif R.Record.mode=="pair_only"
 assert(isempty(E.training)&&E.networkEndpointRows==0);
end
assert(all(cellfun(@(p)numel(p.archive.id)<=500,E.population)));
assert(all(cellfun(@(g)g.archive.eligibilityRejectedPairs==0&&g.archive.archiveFrontDepth==1,E.generations)));
Proxy=boundary_proxy_rows(E,R.Record.problem,R.Record.seed,arm);
writetable(Proxy,fullfile(folder,'confirmation_boundary_proxies',sprintf('%s_seed%02d_%s.csv',R.Record.problem,R.Record.seed,arm)));
late=E.generations(cellfun(@(g)g.startFE>=50000,E.generations));
row=struct('problem',R.Record.problem,'seed',R.Record.seed,'arm',arm,'verified',true, ...
 'fullFE',100000,'CalObjRows',200000,'CalConRows',100000,'initialPrefixExact',initialPrefixExact, ...
 'trainingEvents',numel(trained),'totalGUpdates',sum(cellfun(@(t)t.updates,trained)), ...
 'lateGuidedSelected',sum(cellfun(@(g)g.use.selected,late)), ...
 'lateGuidedSurvivedP1',sum(cellfun(@(g)nnz(g.use.survivedP1),late)), ...
 'lateGuidedInfeasibleReplacements',sum(cellfun(@(g)g.archive.guidedTightenedInfeasible,late)), ...
 'sourceHash',R.Record.sourceHash);
if R.Record.mode=="cgan"
 Usage=late_usage_record(E,R.Record,arm);[~,stem]=fileparts(Task.file);tag="control_";if Task.selected;tag="selected_";end
 writetable(struct2table(Usage),fullfile(folder,'confirmation_usage',tag+stem+".csv"));
end
writetable(struct2table(row),output);fprintf('CONFIRMED_AUDIT %s seed=%d %s\n',R.Record.problem,R.Record.seed,arm);
end
