function audit_auxiliary
warning('off','all');maxNumCompThreads(1);folder=fileparts(mfilename('fullpath'));addpath(folder);
fixed=dir(fullfile(folder,'fixed_conditions','*.csv'));assert(numel(fixed)==12);rows=struct([]);
for n=7:8
 for seed=1:3
  name=sprintf('LIRCMOP%d_BC_seed%02d',n,seed);
  A=load(fullfile(folder,'fixed_conditions',[name,'_moving0.mat']));
  B=load(fullfile(folder,'fixed_conditions',[name,'_moving1.mat']));
  assert(isequal(A.Schedule,B.Schedule));
  assert(isequaln(A.Snapshots{1}.Model,B.Snapshots{1}.Model));
  for j=1:5
   sa=A.Snapshots{j};sb=B.Snapshots{j};u=A.Table.addedUpdates(j);
   assert(isequal(sa.Data.xF,sb.Data.xF)&&isequal(sa.Data.xI,sb.Data.xI));
   assert(sa.Model.iterG==A.base+u&&sb.Model.iterG==A.base+u);
   assert(sa.Model.iterC==A.Snapshots{1}.Model.iterC+5*u);
   assert(A.Table.offlineFullFE(j)==200&&B.Table.offlineFullFE(j)==200);
  end
  row=struct('case',string(name),'sameInitialModelAdam',true,'sameBatchIDs',true, ...
   'sameTrueEndpoints',true,'addedGUpdatesPerArm',2000,'offlineFullFE',2000);
  rows=[rows;row]; %#ok<AGROW>
 end
end
writetable(struct2table(rows),fullfile(folder,'fixed_condition_verification.csv'));
Files=dir(fullfile(folder,'suffix','*.mat'));Files=Files(~strcmp({Files.name},'status.mat'));assert(numel(Files)==48);
rows=struct([]);boundary=struct([]);curves=cell(48,1);
for k=1:48
 S=load(fullfile(Files(k).folder,Files(k).name));R=S.R;
 assert(R.suffixFE==5000&&S.Cost.CalConRows==5000&&S.Cost.CalObjRows==10000);
 assert(all(cellfun(@(h)h.use.selected==0,S.History(2:end))));
 original=[];
 if R.mode=="raw"
  Original=load(fullfile(folder,'fixtures',sprintf('%s_seed%02d_suffix.mat',R.problem,R.seed)));
  stage=1+(R.prefixFE==70000);E=Original.Suffix{stage}.ExpectedNext;
  assert(isequal(S.First.p1Decs,E.p1Decs)&&isequal(S.First.p2Decs,E.p2Decs)&&isequaln(S.First.archive,E.archive));
  assert(all(ismember(S.First.use.childDecs,S.OriginalQ,'rows')));
 elseif R.mode=="midpoint"
  Original=load(fullfile(folder,'fixtures',sprintf('%s_seed%02d.mat',R.problem,R.seed)),'Prefixes');
  stage=1+(R.prefixFE==70000);X=Original.Prefixes{stage}.state.p1Decs(S.ParentIndices,:);
  expected=X+0.5*(S.OriginalQ-X);assert(all(ismember(S.First.use.childDecs,expected,'rows')));
 end
 A=struct('observationFE',50000,'archive',S.InitialArchive,'p1Decs',zeros(0,30));
 B=struct('observationFE',55000,'archive',S.Final.archive,'p1Decs',S.Final.p1Decs);
 E=struct('population',{{A,B}});T=boundary_proxy_rows(E,R.problem,R.seed,R.mode);T=T(T.FE==55000,:);
 T.FE(:)=R.prefixFE+5000;T.prefixFE=repmat(R.prefixFE,height(T),1);boundary=[boundary;table2struct(T)]; %#ok<AGROW>
 curves{k}=array2table(S.Trajectory,'VariableNames',{'FE','IGD','HV','guidedSelected'});
 curves{k}.problem=repmat(R.problem,26,1);curves{k}.seed=repmat(R.seed,26,1);
 curves{k}.prefixFE=repmat(R.prefixFE,26,1);curves{k}.mode=repmat(R.mode,26,1);
 row=struct('problem',R.problem,'seed',R.seed,'prefixFE',R.prefixFE,'mode',R.mode, ...
     'fullFE',5000,'firstStepExactReplay',R.mode=="raw",'laterGuidanceBatches',0);
 rows=[rows;row]; %#ok<AGROW>
end
writetable(struct2table(rows),fullfile(folder,'suffix_verification.csv'));
writetable(struct2table(boundary),fullfile(folder,'suffix_boundary_proxies.csv'));
writetable(vertcat(curves{:}),fullfile(folder,'suffix_trajectories.csv'));
fprintf('AUXILIARY_AUDIT_COMPLETE fixed=12 suffix=48 exactRawReplay=12\n');
end
