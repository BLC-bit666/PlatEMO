function plot_reset_comparison(updates)
% Shared limits and unchanged true samples; fresh columns are never best-picked.
if nargin<1; updates=10000; end
warning('off','all'); folder=fileparts(mfilename('fullpath'));
out=fullfile(folder,'figures'); if ~isfolder(out); mkdir(out); end
arms={'original','stable_side','stable_no_side'};
for a=1:3
    models=cell(2,4); complete=true; warmRng=cell(2,1);
    for n=7:8
        for reset=0:3
            file=fullfile(folder,'runs',sprintf('%s_LIRCMOP%d_BC_reset%d.mat',arms{a},n,reset));
            if ~isfile(file); complete=false; continue; end
            R=load(file,'Snapshots','Table','RngEnds'); row=find(R.Table.addedUpdates==updates,1);
            if isempty(row); complete=false; else; models{n-6,reset+1}=R.Snapshots{row}.Samples; end
            if reset==0; warmRng{n-6}=R.RngEnds;
            elseif ~isempty(row) && ~isempty(warmRng{n-6}) && any(warmRng{n-6}(:,updates/20))
                assert(isequal(R.RngEnds(:,1:updates/20),warmRng{n-6}(:,1:updates/20)));
            end
        end
    end
    if ~complete; continue; end
    fig=figure('Visible','off','Color','w','Position',[30 30 1900 820]);
    layout=tiledlayout(fig,2,5,'TileSpacing','compact','Padding','compact');
    for n=7:8
        F=load(fullfile(folder,'fixtures',sprintf('%s_LIRCMOP%d_BC.mat',arms{a},n)),'ActualData','CachedObjectives');
        for col=1:5
            ax=nexttile(layout); hold(ax,'on'); grid(ax,'on'); box(ax,'on');
            if col==1; Y=F.CachedObjectives; label='Original FE99800';
            else
                Y=models{n-6,col-1}.Y;
                if col==2; label=sprintf('Warm +%d updates',updates);
                else; label=sprintf('Fresh init %d | %d updates',col-2,updates); end
            end
            h(1)=scatter(ax,F.ActualData.yF(:,1),F.ActualData.yF(:,2),12,[.05 .48 .3],'filled');
            h(2)=scatter(ax,F.ActualData.yI(:,1),F.ActualData.yI(:,2),12,[.75 .26 .24],'x');
            h(3)=scatter(ax,Y(:,1),Y(:,2),13,[.05 .30 .85],'filled');
            xlim(ax,[.65 2.6]); ylim(ax,[.65 2.6]); axis(ax,'square'); xlabel(ax,'f_1'); ylabel(ax,'f_2');
            title(ax,{sprintf('LIRCMOP%d BC',n),label},'Interpreter','none','FontSize',10);
            text(ax,.68,.70,sprintf('%d / %d outside view',nnz(any(Y<.65|Y>2.6,2)),size(Y,1)),'FontSize',8);
        end
    end
    lg=legend(h,{'Frozen feasible training','Frozen infeasible training','Generated candidates'},'Orientation','horizontal');
    lg.Layout.Tile='south'; title(layout,sprintf('%s | identical FE99800 data and conditions | no search or selection',arms{a}),'Interpreter','none');
    exportgraphics(fig,fullfile(out,sprintf('%s_reset_%05d.png',arms{a},updates)),'Resolution',160); close(fig);
    fprintf('RESET_PLOT_READY %s updates=%d\n',arms{a},updates);
end
end
