function render_PairGuide_pool_front_first_use(number)
%RENDER_PAIRGUIDE_POOL_FRONT_FIRST_USE Same axes for reused A and new B/C/D.
warning('off','all'); maxNumCompThreads(1);
root=fileparts(which('platemo')); addCBSPaths(root);
folder=fullfile(root,'Data','PairGuidePoolFrontFirstUse_20260907');
base=fullfile(root,'Data','PairGuideNativeFirstEpoch_20260906');
figures=fullfile(folder,'figures'); if ~isfolder(figures); mkdir(figures); end
name=sprintf('LIRCMOP%d_BC',number); R=cell(1,4); extent=[]; focus=[];
R{1}=load(fullfile(base,[name,'_epoch0800.mat'])); R{1}.M.arm="A";
R{1}.Trace=struct('retainedFrontRanks',nan(numel(R{1}.Pop.archive.id),1));
R{1}.M.eligiblePairs=legacyEligible(R{1}.Pop);
R{1}.M.populationOverlap=nnz(ismember(R{1}.Pop.p1Decs,R{1}.Pop.p2Decs,'rows'));
for j=2:4
    arm=char('A'+j-1);
    R{j}=load(fullfile(folder,sprintf('%s_%s_epoch0800.mat',name,arm)));
end
for j=1:4
    B=R{j}; A=B.Pop.archive; k=B.Q.pool.keepIdx;
    assert(B.T.epochs==800 && B.T.updates==800*B.T.batchesPerEpoch);
    assert(B.T.pairVisits==800*B.T.trainingPairs && B.Q.modelVersion==1);
    assert(B.M.firstUseFE==B.M.productionFE+200 && B.Pop.observationFE==B.Q.productionFE);
    assert(isequal(B.Use.childDecs,B.Q.rawDecs(k,:)) && isequal(B.Use.childObjs,B.Y(k,:)));
    assert(B.AlgorithmCost.CalConRows==B.M.firstUseFE && B.AlgorithmCost.CalObjRows==2*B.M.firstUseFE);
    assert(B.OfflineCost.CalConRows==1500 && B.OfflineCost.CalObjRows==3000);
    extent=[extent;B.Pop.p1Objs;B.Pop.p2Objs;A.yf;A.yi;B.Y]; %#ok<AGROW>
    focus=[focus;A.yf(A.active,:);A.yi(A.active,:);B.Y]; %#ok<AGROW>
end
ctor=str2func(name); P=ctor(); View=limits(extent); Focus=limits(prctile(focus,[1 99],1));
Summary=table();
for j=1:4
    B=R{j}; stem=sprintf('%s_%s_epoch0800',name,B.M.arm);
    drawPair(P,name,B,View,fullfile(figures,[stem,'.png']),false);
    drawPair(P,name,B,Focus,fullfile(figures,[stem,'_focus.png']),true);
    M=B.M;
    row=table(string(name),string(M.arm),M.productionFE,M.firstUseFE,M.archivePairs, ...
        M.eligiblePairs,M.trainingPairs,M.populationOverlap,M.updates,M.postRMSE, ...
        M.nativeFeasible,M.selectedCount,M.selectedFeasible,M.survivedP1,M.survivedP2,M.firstUseIGD, ...
        'VariableNames',{'problem','arm','productionFE','firstUseFE','archivePairs','eligiblePairs', ...
        'trainingPairs','populationOverlap','updates','postRMSE','nativeFeasible','selectedCount', ...
        'selectedFeasible','survivedP1','survivedP2','firstUseIGD'});
    Summary=[Summary;row]; %#ok<AGROW>
end
writetable(Summary,fullfile(folder,[name,'_comparison.csv']));
% A's full pre-training trajectory already exists from the earlier diagnosis.
% Reading it is sufficient; no replay or retraining of A is performed.
commonFE=min(Summary.productionFE); States=cell(1,4);
old=load(fullfile('/tmp/pairguide-activation-diagnosis-20260906',[name,'_diagnostic.mat']),'PrefixEvidence');
times=cellfun(@(p)p.observationFE,old.PrefixEvidence.population);
idx=find(times==commonFE,1); assert(~isempty(idx),'Missing cached A state at the common FE.');
States{1}=old.PrefixEvidence.population{idx}; clear old;
for j=2:4
    times=cellfun(@(p)p.observationFE,R{j}.E.population);
    idx=find(times==commonFE,1); assert(~isempty(idx)); States{j}=R{j}.E.population{idx};
end
drawCommon(P,name,States,commonFE,fullfile(figures,[name,'_same_FE.png']));
save(fullfile(folder,[name,'_same_FE.mat']),'States','commonFE','View','Focus');
fprintf('VERIFIED_FIGURES %s arms=4 full=4 focus=4 sameFE=%d\n',name,commonFE);
end

function drawPair(P,name,B,V,file,focused)
A=B.Pop.archive; k=B.Q.pool.keepIdx;
F=figure('Visible','off','Color','w','Position',[20 20 1900 850]);
cleanup=onCleanup(@()close(F)); L=tiledlayout(F,1,2,'Padding','compact','TileSpacing','compact');
ax=nexttile(L); drawState(ax,P,name,B.Pop,B.Trace.retainedFrontRanks,V);
title(ax,sprintf('First training | archive %d, eligible %d, active %d | P1/P2 overlap %d/100', ...
    numel(A.id),B.M.eligiblePairs,nnz(A.active),B.M.populationOverlap),'FontSize',12);
ax=nexttile(L); style(ax); background(ax,P,name,V);
hg=scatter(ax,B.Y(:,1),B.Y(:,2),24,[.18 .40 .74],'filled','MarkerFaceAlpha',.4,'MarkerEdgeAlpha',.2);
hs=scatter(ax,B.Y(k,1),B.Y(k,2),85,[1 .63 .08],'d','filled','MarkerEdgeColor',[.5 .28 .05]);
legend(ax,[hg hs],{sprintf('CGAN native candidates (%d)',size(B.Y,1)), ...
    sprintf('Selected and truly used (%d)',numel(k))},'Location','southoutside','FontSize',11);
axesLimits(ax,V);
title(ax,sprintf('Generated / selected feasible: %.1f%% / %.1f%% | P1/P2 survivors: %d / %d', ...
    100*B.M.nativeFeasible,100*B.M.selectedFeasible,B.M.survivedP1,B.M.survivedP2),'FontSize',12);
note='Full view: all P1/P2 points, archive endpoint slots and native candidates included';
if focused
    note=sprintf('Focus; outside axes: P1 %d, P2 %d, archive slots %d, native %d, selected %d; see full view', ...
        outside(B.Pop.p1Objs,V),outside(B.Pop.p2Objs,V),outside([A.yf;A.yi],V),outside(B.Y,V),outside(B.Y(k,:),V));
end
labels=["A: shared parents, P1 eligibility (reused)","B: CCMO parents, P1 eligibility", ...
    "C: CCMO parents, joint front 1","D: CCMO parents, joint fronts 1-2"];
j=double(char(B.M.arm))-'A'+1;
title(L,{sprintf('%s | %s | seed 1 | 800 epochs (%d G updates)',name,labels(j),B.T.updates), ...
    sprintf('Training FE %d | first-use FE %d | endpoint RMSE %.4f',B.M.productionFE,B.M.firstUseFE,B.M.postRMSE),note}, ...
    'Interpreter','none','FontSize',13);
exportgraphics(F,file,'Resolution',140); clear cleanup;
end

function drawState(ax,P,name,S,ranks,V)
style(ax); [hf,hi]=background(ax,P,name,V); A=S.archive; active=A.active; inactive=~active;
h0f=scatter(ax,A.yf(inactive,1),A.yf(inactive,2),18,[.40 .52 .42],'o','MarkerEdgeAlpha',.22);
h0i=scatter(ax,A.yi(inactive,1),A.yi(inactive,2),18,[.60 .43 .43],'s','MarkerEdgeAlpha',.22);
for row=reshape(find(active),1,[])
    lineStyle='-'; if ranks(row)==2; lineStyle='--'; end
    plot(ax,[A.yf(row,1),A.yi(row,1)],[A.yf(row,2),A.yi(row,2)], ...
        'Color',[.48 .48 .48],'LineStyle',lineStyle,'LineWidth',.8);
end
hp2=scatter(ax,S.p2Objs(:,1),S.p2Objs(:,2),42,[0 .51 .59],'^','LineWidth',1.2);
hp1=scatter(ax,S.p1Objs(:,1),S.p1Objs(:,2),42,[.51 .17 .70],'o','LineWidth',1.2);
second=active & ranks==2; first=active & ~second;
h1f=scatter(ax,A.yf(first,1),A.yf(first,2),62,[.07 .46 .16],'o','filled');
h2f=scatter(ax,A.yf(second,1),A.yf(second,2),65,[.26 .56 .13],'d','filled');
h1i=scatter(ax,A.yi(active,1),A.yi(active,2),60,[.79 .15 .16],'s','filled');
firstLabel='Active feasible endpoint (train)';
if any(isfinite(ranks)); firstLabel='Active feasible endpoint: front 1'; end
handles=[hp1 hp2 h1f h1i h0f h0i hf hi];
labels={sprintf('P1 constrained (%d)',size(S.p1Objs,1)),sprintf('P2 unconstrained (%d)',size(S.p2Objs,1)), ...
    firstLabel,'Active infeasible endpoint (train)','Inactive feasible endpoint', ...
    'Inactive infeasible endpoint','Feasible domain','Infeasible domain'};
if any(isfinite(ranks))
    handles=[handles(1:3),h2f,handles(4:end)];
    labels=[labels(1:3),{'Active feasible endpoint: front 2'},labels(4:end)];
end
legend(ax,handles,labels, ...
    'Location','southoutside','NumColumns',2,'FontSize',10);
axesLimits(ax,V);
end

function drawCommon(P,name,S,fe,file)
extent=[];
for j=1:4
    A=S{j}.archive; extent=[extent;S{j}.p1Objs;S{j}.p2Objs;A.yf;A.yi]; %#ok<AGROW>
end
V=limits(extent); F=figure('Visible','off','Color','w','Position',[20 20 1900 1500]);
cleanup=onCleanup(@()close(F)); L=tiledlayout(F,2,2,'Padding','compact','TileSpacing','compact');
for j=1:4
    ax=nexttile(L); A=S{j}.archive; drawState(ax,P,name,S{j},nan(numel(A.id),1),V);
    title(ax,sprintf('%c | archive %d pairs, active %d | P1/P2 overlap %d/100','A'+j-1, ...
        numel(A.id),nnz(A.active),nnz(ismember(S{j}.p1Decs,S{j}.p2Decs,'rows'))));
end
title(L,sprintf('%s | same FE %d | cached search states, no extra training',name,fe),'Interpreter','none');
exportgraphics(F,file,'Resolution',120); clear cleanup;
end

function style(ax)
hold(ax,'on'); grid(ax,'on'); box(ax,'on'); ax.FontSize=11;
xlabel(ax,'f_1'); ylabel(ax,'f_2');
end

function [hf,hi]=background(ax,P,name,V)
colors=struct('feasible',[.48 .77 .55],'infeasible',[.91 .53 .50]);
[hf,hi]=draw_CBS_CGAN_objective_region(ax,P,name,colors,V);
end

function axesLimits(ax,V)
xlim(ax,[V.lower(1),V.upper(1)]); ylim(ax,[V.lower(2),V.upper(2)]);
end

function V=limits(Y)
lo=min(Y,[],1); hi=max(Y,[],1); width=max(hi-lo,.1);
V=struct('lower',lo-.045*width,'upper',hi+.045*width);
end

function n=outside(Y,V)
n=nnz(any(Y<V.lower | Y>V.upper,2));
end

function n=legacyEligible(S)
F=S.p1Objs(all(S.p1Cons<=0,2),:); Y=S.archive.yf;
if isempty(F); n=size(Y,1); return; end
weak=all(reshape(F,1,size(F,1),2)<=reshape(Y,size(Y,1),1,2)+1e-12,3);
strict=any(reshape(F,1,size(F,1),2)<reshape(Y,size(Y,1),1,2)-1e-12,3);
n=nnz(~any(weak & strict,2));
end
