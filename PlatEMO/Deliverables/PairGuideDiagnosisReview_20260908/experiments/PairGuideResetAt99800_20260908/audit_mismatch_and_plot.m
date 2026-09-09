function audit_mismatch_and_plot(updates)
if nargin<1; updates=10000; end
warning('off','all'); folder=fileparts(mfilename('fullpath')); rows=cell(2,1);
fig=figure('Visible','off','Color','w','Position',[30 30 1250 780]);
layout=tiledlayout(fig,2,3,'TileSpacing','compact','Padding','compact');
for n=7:8
    stem=sprintf('original_LIRCMOP%d_BC_reset1',n);
    B=load(fullfile(folder,'runs',[stem,'.mat'])); R=load(fullfile(folder,'mismatch_runs',[stem,'.mat']));
    F=load(fullfile(folder,'fixtures',sprintf('original_LIRCMOP%d_BC.mat',n)));
    b=find(B.Table.addedUpdates==updates,1); at=find(R.Table.addedUpdates==updates,1); assert(~isempty(b) && ~isempty(at));
    assert(isequaln(B.Snapshots{1}.Model.netG.Learnables,R.Snapshots{1}.Model.netG.Learnables));
    assert(isequaln(B.Snapshots{1}.Model.netC.Learnables,R.Snapshots{1}.Model.netC.Learnables));
    assert(isequal(B.RngEnds(:,1:updates/20),R.RngEnds(:,1:updates/20)));
    assert(R.Snapshots{at}.Model.iterG==updates && R.Snapshots{at}.Model.iterC==5*updates);
    assert(isequal(R.Snapshots{at}.Model.lastData,F.RawData));
    if updates==10000; assert(R.Complete); end
    Y={F.CachedObjectives,B.Snapshots{b}.Samples.Y,R.Snapshots{at}.Samples.Y};
    names={'Original FE99800',sprintf('Fresh: original D | %d',updates),sprintf('Fresh: mixed D negatives | %d',updates)};
    for j=1:3
        ax=nexttile(layout); hold(ax,'on'); grid(ax,'on'); box(ax,'on');
        h(1)=scatter(ax,F.ActualData.yF(:,1),F.ActualData.yF(:,2),13,[.05 .48 .3],'filled');
        h(2)=scatter(ax,F.ActualData.yI(:,1),F.ActualData.yI(:,2),13,[.75 .26 .24],'x');
        h(3)=scatter(ax,Y{j}(:,1),Y{j}(:,2),14,[.05 .3 .85],'filled');
        xlim(ax,[.65 2.6]);ylim(ax,[.65 2.6]);axis(ax,'square');xlabel(ax,'f_1');ylabel(ax,'f_2');
        title(ax,{sprintf('LIRCMOP%d BC',n),names{j}},'Interpreter','none','FontSize',10);
        text(ax,.68,.7,sprintf('%d / %d outside view',nnz(any(Y{j}<.65|Y{j}>2.6,2)),size(Y{j},1)),'FontSize',8);
    end
    rows{n-6}=struct('problem',sprintf('LIRCMOP%d_BC',n),'updates',updates,'verified',true, ...
        'sameInitialWeights',true,'sameNativeRNGTrace',true, ...
        'baselineObjectiveGap',B.Table.objectiveGap(b),'mismatchObjectiveGap',R.Table.objectiveGap(at), ...
        'baselineDirectionError',B.Table.directionError(b),'mismatchDirectionError',R.Table.directionError(at), ...
        'baselineEndpointRMSE',B.Table.endpointRMSE(b),'mismatchEndpointRMSE',R.Table.endpointRMSE(at), ...
        'offlineFullFE',sum(R.Table.offlineFullFE(1:at)));
end
lg=legend(h,{'Frozen feasible training','Frozen infeasible training','Generated candidates'},'Orientation','horizontal');lg.Layout.Tile='south';
title(layout,'Same fresh initialization and RNG | critic-negative intervention only | offline diagnosis');
exportgraphics(fig,fullfile(folder,'figures',sprintf('mismatch_%05d.png',updates)),'Resolution',180);close(fig);
writetable(struct2table(vertcat(rows{:})),fullfile(folder,sprintf('mismatch_verification_%d.csv',updates)));
fprintf('MISMATCH_AUDIT_AND_PLOT_COMPLETE updates=%d\n',updates);
end
