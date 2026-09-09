function Summary = report_PairGuide_interval_validation(folder)
%REPORT_PAIRGUIDE_INTERVAL_VALIDATION One IGD/HV figure and paired tables per problem.
    Summary=analyze_PairGuide_interval_validation(folder);
    if isempty(Summary); return; end
    output=fullfile(folder,'analysis');
    if ~isfolder(output); mkdir(output); end
    checkpointFiles=dir(fullfile(folder,'*.mat.checkpoints.csv'));
    checkpoints=table();
    for k=1:numel(checkpointFiles)
        checkpoints=[checkpoints;readtable(fullfile(folder,checkpointFiles(k).name), ...
            'TextType','string')]; %#ok<AGROW>
    end
    if ~isempty(checkpoints)
        writetable(checkpoints,fullfile(output,'checkpoints_per_run.csv'));
        fields={'actualFE','IGD','HV','activePairs','eligiblePairs','archivePairs', ...
            'p1Feasible','p2Feasible','gapMedian','trainingEvents','generatorUpdates', ...
            'lastTrainingEndpointRMSE','selectedFeasibleRate','cumulativeGuided'};
        means=groupsummary(checkpoints,{'problem','mode','stage'},'mean',fields);
        deviations=groupsummary(checkpoints,{'problem','mode','stage'},'std',fields);
        writetable(means,fullfile(output,'checkpoints_mean.csv'));
        writetable(deviations,fullfile(output,'checkpoints_std.csv'));
    end
    names={'IGDAUC','finalIGD','finalHV','quotaOccupancy','postModelQuotaOccupancy', ...
        'nativeBandRate','projectionMedian','uniqueGuidedUpdateRate','trainingSeconds','wallSeconds'};
    modes=["fallback_only","pair_only","cgan"];
    colors=[0.45 0.45 0.45;0.05 0.55 0.45;0.12 0.36 0.78];
    styles=["--","-.","-"];
    groups=struct([]); pairs=struct([]);
    for problem=unique(Summary.problem)'
        F=figure('Visible','off','Color','w','Position',[100 100 1250 530]);
        cleanup=onCleanup(@()close(F));
        layout=tiledlayout(F,1,2,'TileSpacing','compact','Padding','compact');
        axesList=[nexttile(layout),nexttile(layout)];
        for a=axesList; hold(a,'on'); grid(a,'on'); box(a,'on'); xlabel(a,'Search FE'); end
        ylabel(axesList(1),'P1 IGD (lower is better)');
        ylabel(axesList(2),'P1 HV (higher is better)');
        for m=1:numel(modes)
            rows=Summary.problem==problem & Summary.mode==modes(m);
            S=sortrows(Summary(rows,:),'seed');
            if isempty(S); continue; end
            row=struct('problem',problem,'mode',modes(m),'runs',height(S));
            for j=1:numel(names)
                values=S.(names{j});
                row.([names{j},'Mean'])=mean(values);
                row.([names{j},'Std'])=std(values);
            end
            groups=[groups;row]; %#ok<AGROW>
            curves=cell(height(S),1);
            for j=1:height(S)
                file=fullfile(folder,sprintf('%s_seed%02d_%s.mat',problem,S.seed(j),modes(m)));
                R=load(file,'Trajectory'); curves{j}=R.Trajectory;
            end
            x=curves{1}.actualFE;
            assert(all(cellfun(@(t)isequaln(t.actualFE,x),curves)), ...
                'Same-mode trajectories must use the same FE grid.');
            for p=1:2
                metric='IGD'; if p==2; metric='HV'; end
                y=cell2mat(cellfun(@(t)t.(metric),curves','UniformOutput',false));
                center=mean(y,2); deviation=std(y,0,2);
                finite=isfinite(x)&isfinite(center)&isfinite(deviation);
                fill(axesList(p),[x(finite);flipud(x(finite))], ...
                    [center(finite)-deviation(finite);flipud(center(finite)+deviation(finite))], ...
                    colors(m,:),'FaceAlpha',0.12,'EdgeColor','none','HandleVisibility','off');
                plot(axesList(p),x,center,'Color',colors(m,:),'LineWidth',1.6, ...
                    'LineStyle',styles(m), ...
                    'DisplayName',sprintf('%s (n=%d)',modes(m),height(S)));
            end
        end
        for a=axesList; legend(a,'Location','best','Interpreter','none'); end
        title(layout,sprintf('%s | same-seed controls | mean +/- SD',problem),'Interpreter','none');
        exportgraphics(F,fullfile(output,char(problem+"_IGD_HV.png")),'Resolution',150);
        clear cleanup;
        C=Summary(Summary.problem==problem & Summary.mode=="cgan",:);
        for baseline=["fallback_only","pair_only"]
            B=Summary(Summary.problem==problem & Summary.mode==baseline,:);
            for j=1:height(C)
                b=find(B.seed==C.seed(j),1);
                if isempty(b); continue; end
                row=struct('problem',problem,'seed',C.seed(j),'baseline',baseline, ...
                    'cganIGDAUC',C.IGDAUC(j),'baselineIGDAUC',B.IGDAUC(b), ...
                    'IGDAUCImprovement',relativeGain(B.IGDAUC(b),C.IGDAUC(j)), ...
                    'finalIGDImprovement',relativeGain(B.finalIGD(b),C.finalIGD(j)), ...
                    'wallTimeRatio',C.wallSeconds(j)/B.wallSeconds(b));
                pairs=[pairs;row]; %#ok<AGROW>
            end
        end
    end
    writetable(struct2table(groups),fullfile(output,'problem_summary.csv'));
    if ~isempty(pairs); writetable(struct2table(pairs),fullfile(output,'paired_comparisons.csv')); end
    fprintf('REPORT %d runs: %s\n',height(Summary),output);
end

function gain=relativeGain(baseline,value)
    gain=NaN;
    if isfinite(baseline) && isfinite(value) && baseline>0
        gain=(baseline-value)/baseline;
    end
end
