function audit_stage_suffixes
% Audit saved windows and calculate local short-pair witnesses without an oracle.
warning('off','all');maxNumCompThreads(1);folder=fileparts(mfilename('fullpath'));
rows={};checks={};proxies={};trajectories={};
for n=5:8
    for stage={'early','middle','late'}
        prefix=load(fullfile(folder,'suffix_fixtures',sprintf('LIRCMOP%d_BC_seed01_%s.mat',n,stage{1})));
        fe=prefix.Prefix.FE;stem=sprintf('LIRCMOP%d_BC_seed01_FE%06d_',n,fe);
        de=load(fullfile(folder,'suffix',[stem,'de.mat']));
        for mode=["raw","adversarial20","distance20","adversarial2000","distance2000","de","pair_only"]
            R=load(fullfile(folder,'suffix',[stem,char(mode),'.mat']));
            assert(R.Cost.CalObjRows==10000 && R.Cost.CalConRows==5000 && R.R.suffixFE==5000);
            assert(isequal(R.Trajectory(:,1),(fe:200:fe+5000)') && all(isfinite(R.Trajectory),'all'));
            assert(numel(R.History)==25 && R.History{1}.use.selected<=20);
            assert(all(cellfun(@(x)x.use.selected==0,R.History(2:end))));
            assert(isequaln(R.InitialArchive,prefix.Prefix.state.archive));
            assert(R.R.queryAvailable==prefix.Prefix.queryAvailable);
            if mode=="raw"
                E=prefix.ExpectedNext;
                assert(isequal(R.First.p1Decs,E.p1Decs) && isequal(R.First.p2Decs,E.p2Decs) && isequaln(R.First.archive,E.archive));
            end
            if ~prefix.Prefix.queryAvailable
                assert(R.History{1}.use.selected==0 && isequaln(R.Final,de.Final) && isequaln(R.Trajectory,de.Trajectory));
            end
            row=R.R;
            row.firstSurvivedP2=nnz(ismember(R.History{1}.use.childDecs,R.First.p2Decs,'rows'));
            row.firstGuidedFeasible=R.History{1}.guidedFeasible;
            row.firstRetainedInfeasibleEndpoints=nnz(ismember(R.History{1}.use.childDecs,R.First.archive.xi,'rows'));
            row.firstRetainedFeasibleEndpoints=nnz(ismember(R.History{1}.use.childDecs,R.First.archive.xf,'rows'));
            rows{end+1}=row;
            checks{end+1}=struct('problem',R.R.problem,'stage',string(stage{1}),'mode',mode,'FE',5000, ...
                'objRows',10000,'conRows',5000,'prefixVerified',true,'queryGateVerified',true,'oneBatchOnly',true, ...
                'rawExactReplay',mode=="raw");
            T=array2table(R.Trajectory,'VariableNames',{'FE','IGD','HV','guidedCount'});
            T.problem=repmat(R.R.problem,26,1);T.stage=repmat(string(stage{1}),26,1);T.mode=repmat(mode,26,1);
            trajectories{end+1}=T;
            for threshold=[.01 .02 .04]
                origin=midpoints(R.InitialArchive,threshold);current=midpoints(R.Final.archive,threshold);
                novel=current;
                if ~isempty(current) && ~isempty(origin)
                    novel=current(min(pdist2(current,origin)/sqrt(size(current,2)),[],2)>threshold,:);
                end
                novel=thin(novel,threshold);covered=0;
                if ~isempty(novel);covered=nnz(min(pdist2(novel,R.Final.p1Decs)/sqrt(size(novel,2)),[],2)<=threshold);end
                proxies{end+1}=struct('problem',R.R.problem,'stage',string(stage{1}),'mode',mode,'threshold',threshold, ...
                    'initialShortPairs',size(origin,1),'finalShortPairs',size(current,1), ...
                    'newSeparatedWitnesses',size(novel,1),'witnessesCoveredByP1',covered);
            end
        end
    end
end
assert(numel(rows)==84 && sum(cellfun(@(r)r.suffixFE,rows))==420000);
writetable(struct2table(vertcat(rows{:})),fullfile(folder,'suffix_records.csv'));
writetable(struct2table(vertcat(checks{:})),fullfile(folder,'suffix_verification.csv'));
writetable(struct2table(vertcat(proxies{:})),fullfile(folder,'suffix_boundary_witnesses.csv'));
writetable(vertcat(trajectories{:}),fullfile(folder,'suffix_trajectories.csv'));
fprintf('STAGE_SUFFIX_VERIFIED 84 cases,12 exact native first steps,420000FE; no new oracle calls\n');
end
function X=midpoints(A,t)
if isempty(A.xf);X=A.xf;return;end
short=vecnorm(A.xf-A.xi,2,2)/sqrt(size(A.xf,2))<=t;
X=unique((A.xf(short,:)+A.xi(short,:))/2,'rows');
end
function Y=thin(X,t)
Y=X([],:);
for k=1:size(X,1)
    if isempty(Y) || all(vecnorm(Y-X(k,:),2,2)/sqrt(size(X,2))>t);Y(end+1,:)=X(k,:);end %#ok<AGROW>
end
end
