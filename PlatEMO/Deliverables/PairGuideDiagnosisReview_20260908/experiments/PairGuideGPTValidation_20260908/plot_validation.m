function plot_validation(stage)
if nargin<1;stage="development";end
% Standalone scientific figures from recorded CSV only; no oracle calls.
warning('off','all'); maxNumCompThreads(1); folder=fileparts(mfilename('fullpath'));
prefix="";if string(stage)=="confirmation";prefix="confirmation_";end
out=fullfile(folder,'figures');
if string(stage)=="confirmation";out=fullfile(out,'confirmation');end
if ~isfolder(out);mkdir(out);end
colors=[.25 .25 .25;.03 .50 .72;.85 .30 .14;.22 .58 .35;.56 .35 .65];
arms=["original","stable_side","stable_no_side","fallback_only","pair_only"];
labels=["Original","Fixed span + side","Fixed span, no side","DE control","Pair-only control"];
T=readtable(fullfile(folder,prefix+"search_trajectories.csv"),'TextType','string');
shortLabels=["Original","Fixed + s","Fixed, no s","DE","Pair-only"];
keep=ismember(arms,unique(T.arm));arms=arms(keep);labels=labels(keep);colors=colors(keep,:);shortLabels=shortLabels(keep);
seedCount=numel(unique(T.seed));
for measure=["IGD","HV"]
 F=figure('Visible','off','Color','w','Position',[80 80 1250 850]); L=tiledlayout(F,2,2,'TileSpacing','compact');
 for n=5:8
  ax=nexttile(L); hold(ax,'on'); box(ax,'on'); grid(ax,'on');
  for a=1:numel(arms)
   rows=T.problem==sprintf('LIRCMOP%d_BC',n) & T.arm==arms(a) & T.actualFE>=50000;
   S=T(rows,:); if isempty(S); continue; end
   x=unique(S.actualFE); y=arrayfun(@(v)mean(S.(measure)(S.actualFE==v)),x);
   plot(ax,x,y,'Color',colors(a,:),'LineWidth',1.8,'DisplayName',labels(a));
  end
  title(ax,sprintf('LIRCMOP%d_BC | mean of %d seeds',n,seedCount),'Interpreter','none');xlabel(ax,'Full search evaluations');ylabel(ax,measure);
  if measure=="IGD"; set(ax,'YScale','log'); end
  xlim(ax,[50000 100000]);
 end
 legend(ax,'Location','best','FontSize',8); title(L,measure+" in the second half of search");
 exportgraphics(F,fullfile(out,"late_"+measure+"_curves.png"),'Resolution',160);close(F);
end
T=readtable(fullfile(folder,prefix+"search_paired.csv"),'TextType','string');
F=figure('Visible','off','Color','w','Position',[80 80 1100 700]);L=tiledlayout(F,2,2,'TileSpacing','compact');
for n=5:8
 ax=nexttile(L);hold(ax,'on');box(ax,'on');grid(ax,'on');yline(ax,1,'--','Color',[.4 .4 .4]);
 for a=2:numel(arms)
  S=T(T.problem==sprintf('LIRCMOP%d_BC',n) & T.arm==arms(a),:);S=sortrows(S,'seed');
  plot(ax,a-1+linspace(-.1,.1,height(S)),S.lateIGDAUCRatio,'o','Color',colors(a,:),'MarkerFaceColor',colors(a,:));
  if ~isempty(S);plot(ax,[a-1-.18 a-1+.18],repmat(exp(mean(log(S.lateIGDAUCRatio))),1,2),'Color',colors(a,:),'LineWidth',2);end
 end
 xticks(ax,1:numel(arms)-1);xticklabels(ax,shortLabels(2:end));ylabel(ax,'Late IGD AUC / matched original');title(ax,sprintf('LIRCMOP%d_BC',n),'Interpreter','none');
end
title(L,'Matched-seed search ratios | below 1 is better | dots: seeds, bars: geometric mean');
exportgraphics(F,fullfile(out,'search_paired_ratios.png'),'Resolution',160);close(F);
if string(stage)=="confirmation";fprintf('CONFIRMATION_SEARCH_PLOTS_COMPLETE\n');return;end
T=readtable(fullfile(folder,'fixed_condition_summary.csv'),'TextType','string');
Fresh=readtable(fullfile(folder,'fresh_condition_records.csv'),'TextType','string');
F=figure('Visible','off','Color','w','Position',[80 80 1150 800]);L=tiledlayout(F,2,2,'TileSpacing','compact');
for n=7:8
 for metric=["endpointRMSE","directionError"]
  ax=nexttile(L);hold(ax,'on');box(ax,'on');grid(ax,'on');
  for moving=0:1
   S=T(T.problem==sprintf('LIRCMOP%d_BC',n)&T.moving==moving,:);S=sortrows(S,'addedUpdates');
   name="Fixed conditions";if moving;name="Replayed span changes";end
   plot(ax,S.addedUpdates,S.(metric),'-o','Color',colors(moving+1,:),'LineWidth',1.7,'DisplayName',name);
  end
  X=Fresh(Fresh.problem==sprintf('LIRCMOP%d_BC',n),:);xx=unique(X.addedUpdates);
  yy=arrayfun(@(u)mean(X.(metric)(X.addedUpdates==u)),xx);
  plot(ax,xx,yy,'-o','Color',colors(3,:),'LineWidth',1.7,'DisplayName','Fresh initialization, fixed conditions');
  xlabel(ax,'Additional generator updates');ylabel(ax,metric,'Interpreter','none');title(ax,sprintf('LIRCMOP%d_BC | mean of 3 late models',n),'Interpreter','none');legend(ax,'Location','best');
 end
end
title(L,'Frozen true samples and equal batch IDs | warm state and fresh initialization');
exportgraphics(F,fullfile(out,'fixed_condition_replay.png'),'Resolution',160);close(F);
T=readtable(fullfile(folder,'suffix_paired.csv'),'TextType','string');
F=figure('Visible','off','Color','w','Position',[80 80 1050 480]);L=tiledlayout(F,1,2,'TileSpacing','compact');
for n=7:8
 ax=nexttile(L);hold(ax,'on');box(ax,'on');grid(ax,'on');yline(ax,1,'--');
 modes=["midpoint","de","pair_only"];
 for a=1:3
  S=T(T.problem==sprintf('LIRCMOP%d_BC',n)&T.mode==modes(a),:);
  plot(ax,a+linspace(-.12,.12,height(S)),S.IGDAUCRatio,'o','Color',colors(a+1,:),'MarkerFaceColor',colors(a+1,:));
  plot(ax,[a-.2 a+.2],repmat(exp(mean(log(S.IGDAUCRatio))),1,2),'Color',colors(a+1,:),'LineWidth',2);
 end
 xlim(ax,[.5 3.5]);xticks(ax,1:3);xticklabels(ax,{'Midpoint','DE','Pair-only'});title(ax,sprintf('LIRCMOP%d_BC | 3 seeds x 2 prefixes',n),'Interpreter','none');ylabel(ax,'5000-FE IGD AUC / raw q');
end
title(L,'One intervention batch, then common DE backbone | below 1 is better');
exportgraphics(F,fullfile(out,'suffix_paired_ratios.png'),'Resolution',160);close(F);
fprintf('VALIDATION_PLOTS_COMPLETE\n');
end
