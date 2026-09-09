function plot_late_clouds
% Common objective axes and identical symbols for all generated candidates.
warning('off','all');maxNumCompThreads(1);folder=fileparts(mfilename('fullpath'));
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
folders={fullfile(root,'Data','PairGuideSinglePoint_20260907','full_run'), ...
 fullfile(folder,'stable_side'),fullfile(folder,'stable_no_side')};
labels=["Original","Fixed span + side","Fixed span, no side"];
F=figure('Visible','off','Color','w','Position',[60 60 1500 1000]);
L=tiledlayout(F,2,3,'TileSpacing','compact','Padding','compact');
for n=7:8
 for a=1:3
  R=load(fullfile(folders{a},'analysis','figures',sprintf('LIRCMOP%d_BC',n),'run_01','plot_data.mat'),'States');
  S=R.States{end};assert(S.targetFE==100000 && ~S.missingGeneration);
  ax=nexttile(L);hold(ax,'on');box(ax,'on');grid(ax,'on');
  plot(ax,[S.xf(:,1),S.xi(:,1)]',[S.xf(:,2),S.xi(:,2)]','Color',[.8 .8 .8],'HandleVisibility','off');
  h(1)=scatter(ax,S.xf(:,1),S.xf(:,2),12,[.05 .48 .30],'o','filled','DisplayName','Training feasible');
  h(2)=scatter(ax,S.xi(:,1),S.xi(:,2),12,[.75 .26 .24],'x','DisplayName','Training infeasible');
  Q=unique(S.native,'rows');
  h(3)=scatter(ax,Q(:,1),Q(:,2),14,[.05 .30 .85],'filled','DisplayName','Generated candidate');
  h(4)=scatter(ax,S.children(:,1),S.children(:,2),27,[.95 .55 .03],'d','filled','DisplayName','Selected for evaluation');
  outside=nnz(any(Q<.65|Q>2.6,2));
  xlim(ax,[.65 2.6]);ylim(ax,[.65 2.6]);axis(ax,'square');xlabel(ax,'f_1');ylabel(ax,'f_2');
  title(ax,{sprintf('LIRCMOP%d | %s',n,labels(a)), ...
   sprintf('Train FE %.0f | Query FE %.0f',S.trainingFE,S.productionFE)},'FontSize',11);
  text(ax,.68,.7,sprintf('%d / %d unique generated points outside view',outside,size(Q,1)),'FontSize',8);
 end
end
lg=legend(h,'Orientation','horizontal','FontSize',10);lg.Layout.Tile='south';
title(L,'Late training samples and actual CGAN outputs | seed 1 | common objective axes');
exportgraphics(F,fullfile(folder,'figures','late_clouds_common_axes.png'),'Resolution',180);close(F);
fprintf('COMMON_AXIS_CLOUDS_COMPLETE\n');
end
