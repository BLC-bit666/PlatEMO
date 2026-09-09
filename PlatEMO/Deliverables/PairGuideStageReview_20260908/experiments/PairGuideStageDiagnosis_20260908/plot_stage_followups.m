function plot_stage_followups
warning('off','all');folder=fileparts(mfilename('fullpath'));out=fullfile(folder,'figures');
for stage={'early','middle','late'}
    fig=figure('Visible','off','Color','w','Position',[50 50 1650 850]);
    tiledlayout(fig,2,4,'TileSpacing','compact','Padding','compact');
    for n=7:8
        stem=sprintf('LIRCMOP%d_BC_seed01_%s',n,stage{1});F=load(fullfile(folder,'fixtures',[stem,'.mat']));
        Y=[F.Data.yF;F.Data.yI];lo=min(Y,[],1);hi=max(Y,[],1);pad=max(.06,.08*(hi-lo));limits=[lo-pad;hi+pad];
        for arm={'adversarial','distance'}
            for directory={'runs','low_lr_runs'}
                nexttile;hold on;
                R=load(fullfile(folder,directory{1},sprintf('%s_%s_stream1.mat',stem,arm{1})),'Snapshots','Table');
                ix=find(R.Table.addedUpdates==2000,1);S=R.Snapshots{ix}.Samples;
                h1=scatter(F.Data.yF(:,1),F.Data.yF(:,2),14,[0 .48 .3],'filled');
                h2=scatter(F.Data.yI(:,1),F.Data.yI(:,2),14,[.84 .3 .29],'x');
                h3=scatter(S.Y(:,1),S.Y(:,2),13,[.05 .28 .88],'filled');
                xlim(limits(:,1)');ylim(limits(:,2)');grid on;box on;xlabel('f_1');ylabel('f_2');
                lr=.001;if strcmp(directory{1},'low_lr_runs');lr=.0001;end
                title(sprintf('LIRCMOP%d | %s | lrG=%g\nL*=%.3f | reverse gap=%.3f',n,arm{1},lr,R.Table.exactDistance(ix),R.Table.reverseCoverageGap(ix)),'Interpreter','none');
                outside=nnz(any(S.Y<limits(1,:) | S.Y>limits(2,:),2));
                text(.02,.02,sprintf('%d / %d outside view',outside,size(S.Y,1)),'Units','normalized','FontSize',9);
                if n==7 && strcmp(arm{1},'adversarial') && strcmp(directory{1},'runs');legend([h1 h2 h3],{'Training feasible','Training infeasible','Generated'},'Location','best');end
            end
        end
    end
    sgtitle(sprintf('%s stage | +2000 G updates | stream 1 fixed in advance | all 3 streams in CSV',stage{1}));
    exportgraphics(fig,fullfile(out,['learning_rate_',stage{1},'.png']),'Resolution',150);close(fig);
end
T=readtable(fullfile(folder,'suffix_trajectories.csv'),'TextType','string');
fig=figure('Visible','off','Color','w','Position',[50 50 1550 1040]);tiledlayout(fig,3,4,'TileSpacing','compact','Padding','compact');
stages={'early','middle','late'};modes=["raw","adversarial20","distance20","adversarial2000","distance2000","de","pair_only"];colors=lines(7);
for j=1:3
    for n=5:8
        nexttile;hold on;handles=[];
        for k=1:numel(modes)
            part=T(T.problem==sprintf('LIRCMOP%d_BC',n) & T.stage==stages{j} & T.mode==modes(k),:);part=sortrows(part,'FE');
            h=plot(part.FE-part.FE(1),part.IGD,'Color',colors(k,:),'LineWidth',1.2);handles=[handles h]; %#ok<AGROW>
        end
        grid on;box on;xlabel('Additional search FE');ylabel('Native IGD');title(sprintf('LIRCMOP%d | %s',n,stages{j}));
        if j==1 && n==5;legend(handles,modes,'Interpreter','none','Location','best','FontSize',7);end
    end
end
sgtitle('Same prefix, one candidate batch then 5000 FE; one search seed, not independent-run superiority');
exportgraphics(fig,fullfile(out,'suffix_IGD_all_stages.png'),'Resolution',150);close(fig);
fprintf('FOLLOWUP_PLOTS_COMPLETE:3 learning-rate clouds and1 search trajectory grid; no oracle calls\n');
end
