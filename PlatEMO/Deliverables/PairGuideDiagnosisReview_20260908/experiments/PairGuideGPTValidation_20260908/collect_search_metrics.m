function collect_search_metrics(stage)
if nargin<1;stage="development";end
% Read cached trajectories only; never reevaluate objectives or constraints.
warning('off','all');
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO'; folder=fileparts(mfilename('fullpath'));
campaigns={fullfile(root,'Data','PairGuideSinglePoint_20260907','full_run'),'original'; ...
    fullfile(folder,'stable_side'),'stable_side';fullfile(folder,'stable_no_side'),'stable_no_side'};
prefix="";
if string(stage)=="confirmation"
    selected=jsondecode(fileread(fullfile(folder,'development_selection.json')));
    campaigns={fullfile(folder,'confirmation_controls'),'original'; ...
        fullfile(folder,'confirmation_selected'),selected.selected};
    prefix="confirmation_";
end
rows=cell(0,1); curves=cell(0,1); cached=table(); cachedCurves=table();
% Completed result files are immutable in this campaign; reuse exported curves.
if isfile(fullfile(folder,prefix+"search_metrics.csv")) && isfile(fullfile(folder,prefix+"search_trajectories.csv"))
    cached=readtable(fullfile(folder,prefix+"search_metrics.csv"),'TextType','string');
    cachedCurves=readtable(fullfile(folder,prefix+"search_trajectories.csv"),'TextType','string');
end
for c=1:size(campaigns,1)
    files=dir(fullfile(campaigns{c,1},'LIRCMOP*_seed*_*.mat'));
    for f=1:numel(files)
        tok=regexp(files(f).name,'^(LIRCMOP\d+_BC)_seed(\d+)_(.+)\.mat$','tokens','once');
        mode=string(campaigns{c,2}); if mode=="original" && string(tok{3})~="cgan";mode=string(tok{3});end
        if ~isempty(cached)
            hit=cached.problem==string(tok{1}) & cached.seed==str2double(tok{2}) & cached.arm==mode;
            if nnz(hit)==1
                curve=cachedCurves(cachedCurves.problem==string(tok{1}) & ...
                    cachedCurves.seed==str2double(tok{2}) & cachedCurves.arm==mode,:);
                assert(nnz(curve.targetFE>=50000)==51 && curve.actualFE(end)==100000);
                rows{end+1}=table2struct(cached(hit,:)); curves{end+1}=curve; continue;
            end
        end
        try; R=load(fullfile(files(f).folder,files(f).name),'Record','Trajectory');
        catch; continue; end
        if ~isfield(R,'Record') || ~isfield(R,'Trajectory'); continue; end
        T=R.Trajectory; mode=string(campaigns{c,2});
        if mode=="original" && R.Record.mode~="cgan"; mode=R.Record.mode; end
        late=T.targetFE>=50000; assert(nnz(late)==51);
        curve=T; curve.problem=repmat(R.Record.problem,height(T),1);
        curve.seed=repmat(R.Record.seed,height(T),1); curve.arm=repmat(mode,height(T),1);
        curves{end+1}=curve;
        row=struct('problem',R.Record.problem,'seed',R.Record.seed,'arm',mode, ...
            'finalIGD',T.IGD(end),'finalHV',T.HV(end), ...
            'lateIGDAUC',trapz(T.actualFE(late),T.IGD(late))/50000, ...
            'lateHVAUC',trapz(T.actualFE(late),T.HV(late))/50000, ...
            'wallSeconds',R.Record.wallSeconds,'maxFE',R.Record.finalFE,'sourceHash',R.Record.sourceHash);
        rows{end+1}=row;
    end
end
if ~isempty(rows)
    writetable(struct2table(vertcat(rows{:})),fullfile(folder,prefix+"search_metrics.csv"));
    writetable(vertcat(curves{:}),fullfile(folder,prefix+"search_trajectories.csv"));
end
fprintf('COLLECTED_SEARCH_RUNS %d\n',numel(rows));
end
