function plot_gradient_stages(updates)
if nargin<1;updates=2000;end
warning('off','all');folder=fileparts(mfilename('fullpath'));
out=fullfile(folder,'figures');if ~isfolder(out);mkdir(out);end
for n=5:8
    for stage={'early','middle','late'}
        stem=sprintf('LIRCMOP%d_BC_seed01_%s',n,stage{1});
        F=load(fullfile(folder,'fixtures',[stem,'.mat']));B=load(fullfile(folder,'baselines',[stem,'.mat']));
        Y=[F.Data.yF;F.Data.yI];lo=min(Y,[],1);hi=max(Y,[],1);pad=max(.06,.08*(hi-lo));limits=[lo-pad;hi+pad];
        fig=figure('Visible','off','Color','w','Position',[50 50 1680 850]);
        tiledlayout(fig,2,4,'TileSpacing','compact','Padding','compact');
        for arm=1:2
            names={'adversarial','distance'};
            for stream=0:3
                nexttile;S=B.S;metric=B.R;
                if stream>0
                    file=fullfile(folder,'runs',sprintf('%s_%s_stream%d.mat',stem,names{arm},stream));
                    if ~isfile(file);close(fig);error('StagePlot:Incomplete','Missing %s',file);end
                    R=load(file,'Snapshots','Table');ix=find(R.Table.addedUpdates==updates,1);assert(~isempty(ix));
                    S=R.Snapshots{ix}.Samples;metric=table2struct(R.Table(ix,:));
                end
                hold on;
                for k=1:F.Data.count
                    plot([F.Data.yF(k,1),F.Data.yI(k,1)],[F.Data.yF(k,2),F.Data.yI(k,2)],'-','Color',[.86 .86 .86],'HandleVisibility','off');
                end
                h1=scatter(F.Data.yF(:,1),F.Data.yF(:,2),14,[0 .48 .30],'filled');
                h2=scatter(F.Data.yI(:,1),F.Data.yI(:,2),14,[.84 .30 .29],'x');
                h3=scatter(S.Y(:,1),S.Y(:,2),13,[.05 .28 .88],'filled');
                xlim(limits(:,1)');ylim(limits(:,2)');grid on;box on;xlabel('f_1');ylabel('f_2');
                outside=nnz(any(S.Y<limits(1,:) | S.Y>limits(2,:),2));
                if stream==0;label='Captured original';else;label=sprintf('%s | stream %d',names{arm},stream);end
                title(sprintf('%s\nL*=%.3f | seen gap=%.3f',label,metric.exactDistance,metric.seenObjectiveGap),'Interpreter','none');
                text(.02,.02,sprintf('%d / %d outside view',outside,size(S.Y,1)),'Units','normalized','FontSize',9);
                if arm==1 && stream==0;legend([h1 h2 h3],{'Training feasible','Training infeasible','Generated'},'Location','best','FontSize',8);end
            end
        end
        sgtitle(sprintf('LIRCMOP%d_BC | %s | observed FE %d, trained FE %d | +%d G updates', ...
            n,stage{1},F.Meta.observationFE,F.Meta.trainingFE,updates),'Interpreter','none');
        exportgraphics(fig,fullfile(out,sprintf('%s_update%d.png',stem,updates)),'Resolution',150);close(fig);
    end
end
T=readtable(fullfile(folder,'gradient_records.csv'),'TextType','string');
for metric={'exactDistance','seenObjectiveGap','reverseCoverageGap'}
    fig=figure('Visible','off','Color','w','Position',[50 50 1450 1000]);
    tiledlayout(fig,3,4,'TileSpacing','compact','Padding','compact');
    stages={'early','middle','late'};colors=[.3 .3 .3;.82 .16 .08];
    for j=1:3
        for n=5:8
            nexttile;hold on;part=T(T.problem==sprintf('LIRCMOP%d_BC',n) & T.stage==stages{j},:);
            handles=[];
            for a=1:2
                arms={'adversarial','distance'};
                for stream=1:3
                    rows=part(part.loss==arms{a} & part.stream==stream,:);rows=sortrows(rows,'addedUpdates');
                    h=plot(rows.addedUpdates+1,rows.(metric{1}),'-o','Color',colors(a,:),'MarkerSize',3,'LineWidth',1);
                    if stream==1;handles=[handles h];else;set(h,'HandleVisibility','off');end %#ok<AGROW>
                end
            end
            set(gca,'XScale','log');grid on;box on;xlabel('Added G updates + 1');ylabel(metric{1},'Interpreter','none');
            title(sprintf('LIRCMOP%d | %s',n,stages{j}));
            if j==1 && n==5;legend(handles,{'Native adversarial','Exact distance'},'Location','best');end
        end
    end
    sgtitle('Every training stream is shown; frozen datasets at early / middle / late stages');
    exportgraphics(fig,fullfile(out,[metric{1},'_all_stages.png']),'Resolution',150);close(fig);
end
fprintf('STAGE_PLOTS_COMPLETE 12 cloud comparisons + 3 trajectories; zero oracle calls\n');
end
