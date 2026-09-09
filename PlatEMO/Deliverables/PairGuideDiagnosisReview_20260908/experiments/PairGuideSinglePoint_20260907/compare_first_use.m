function compare_first_use
warning('off','all'); maxNumCompThreads(1);
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support')); addCBSPaths(root); addpath(fileparts(mfilename('fullpath')));
folder=fullfile(root,'Data','PairGuideSinglePoint_20260907','first_use');
files=dir(fullfile(folder,'*.mat.csv')); tables=cell(numel(files),1);
for i=1:numel(files); tables{i}=readtable(fullfile(folder,files(i).name),'TextType','string'); end
T=vertcat(tables{:}); writetable(T,fullfile(fileparts(folder),'first_use_all.csv'));
metrics={'firstUseIGD','firstUseHV','survivedP1','rawUseful','labelAccuracy','directionError','trainingSeconds'};
S=groupsummary(T,'arm','mean',metrics); writetable(S,fullfile(fileparts(folder),'first_use_means.csv')); disp(S);
for problem=5:8
    name=sprintf('LIRCMOP%d_BC',problem); F=figure('Visible','off','Color','w','Position',[30 80 1350 860]);
    tl=tiledlayout(F,2,3,'TileSpacing','compact','Padding','compact');
    for j=1:3
        updates=[200 1000 4000]; file=fullfile(folder,sprintf('%s_seed01_u%04d.mat',name,updates(j)));
        R=load(file,'Q','Y','C','Prefix','M','angles','known');
        ax=nexttile(tl,j); hold(ax,'on');
        A=R.Prefix.archive; rows=A.active;
        scatter(ax,A.yf(rows,1),A.yf(rows,2),35,[.05 .45 .16],'filled','DisplayName','Real feasible');
        scatter(ax,A.yi(rows,1),A.yi(rows,2),35,[.78 .13 .14],'filled','DisplayName','Real infeasible');
        for side=0:1
            m=R.Q.sides==side; col=[.78 .13 .14]; mark='s'; if side==1; col=[.05 .45 .16]; mark='o'; end
            scatter(ax,R.Y(m,1),R.Y(m,2),12,col,mark,'MarkerEdgeAlpha',.35,'DisplayName',sprintf('Request s=%d',side));
        end
        k=R.Q.pool.keepIdx; scatter(ax,R.Y(k,1),R.Y(k,2),52,[1 .63 .08],'d','LineWidth',1.2,'DisplayName','Selected');
        xlabel(ax,'f_1'); ylabel(ax,'f_2'); grid(ax,'on'); axis(ax,'equal');
        title(ax,sprintf('%d updates | label %.1f%% | P1 survivors %d',updates(j),100*R.M.labelAccuracy,R.M.survivedP1));
        if j==1; legend(ax,'Location','best','FontSize',8); end
        ax=nexttile(tl,j+3); hold(ax,'on');
        scatter(ax,R.Q.refs(~R.known),R.angles(~R.known),12,[.50 .50 .50],'filled','DisplayName','Unseen condition');
        scatter(ax,R.Q.refs(R.known),R.angles(R.known),17,[.15 .35 .80],'filled','DisplayName','Training condition');
        xlabel(ax,'Requested reference index'); ylabel(ax,'True direction error (degrees)'); ylim(ax,[0 90]); grid(ax,'on');
        title(ax,sprintf('Mean angle %.1f deg | IGD after first use %.3f',R.M.directionError,R.M.firstUseIGD));
        if j==1; legend(ax,'Location','best','FontSize',8); end
    end
    title(tl,sprintf('%s | first-use budget comparison | raw labels evaluated only after stopping',name),'Interpreter','none');
    exportgraphics(F,fullfile(fileparts(folder),sprintf('%s_first_budget.png',name)),'Resolution',130); close(F);
end
end
