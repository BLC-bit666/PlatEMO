function plot_reset_metrics
warning('off','all'); folder=fileparts(mfilename('fullpath'));
T=readtable(fullfile(folder,'reset_records.csv'),'TextType','string');
out=fullfile(folder,'figures'); if ~isfolder(out); mkdir(out); end
for metric={'objectiveGap','endpointRMSE','seenObjectiveGap'}
    fig=figure('Visible','off','Color','w','Position',[50 50 1250 700]);
    layout=tiledlayout(fig,2,3,'TileSpacing','compact','Padding','compact');
    for n=7:8
        for arm={'original','stable_side','stable_no_side'}
            ax=nexttile(layout); hold(ax,'on'); grid(ax,'on'); box(ax,'on');
            for reset=0:3
                ix=T.problem==sprintf('LIRCMOP%d_BC',n) & T.arm==arm{1} & T.reset==reset & T.addedUpdates>0;
                X=T(ix,:); X=sortrows(X,'addedUpdates');
                if reset==0; color=[.15 .15 .15]; label='Warm continuation';
                else; colors=[.8 .25 .05;.05 .45 .75;.45 .2 .6];color=colors(reset,:);label=sprintf('Fresh init %d',reset);end
                plot(ax,X.addedUpdates,X.(metric{1}),'-o','Color',color,'DisplayName',label,'LineWidth',1.4,'MarkerSize',4);
            end
            base=T(T.problem==sprintf('LIRCMOP%d_BC',n) & T.arm==arm{1} & T.reset==0 & T.addedUpdates==0,:);
            if ~isempty(base); yline(ax,base.(metric{1}),':','Original FE99800','HandleVisibility','off'); end
            xlim(ax,[1000 10000]); xlabel(ax,'Additional G updates'); ylabel(ax,metric{1},'Interpreter','none');
            title(ax,sprintf('LIRCMOP%d BC | %s',n,arm{1}),'Interpreter','none');
        end
    end
    lg=legend(ax,'Orientation','horizontal'); lg.Layout.Tile='south';
    title(layout,'Frozen real data and conditions | three fresh initializations shown separately');
    exportgraphics(fig,fullfile(out,[metric{1},'_trajectories.png']),'Resolution',170); close(fig);
end
fprintf('RESET_METRIC_PLOTS_COMPLETE\n');
end
