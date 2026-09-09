function render_holdout_comparison
warning('off','all'); maxNumCompThreads(1);
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support')); addCBSPaths(root);
folder=fullfile(root,'Data','PairGuideSinglePoint_20260907','holdout');
for number=[7 8]
    name=sprintf('LIRCMOP%d_BC',number); ctor=str2func(name); P=ctor();
    sources=cell(4,1); values=[];
    for k=1:4
        arm="selected1000"; if k>2; arm="continued4000"; end
        held=mod(k-1,2);
        R=load(fullfile(folder,sprintf('%s_%s_held%d.mat',name,arm,held)), ...
            'M','D','Full','Y','QueryC','hidden','source');
        S=load(R.source,'States'); A=S.States{end}.trainingState.archive;
        [found,loc]=ismember(R.D.id,A.id); assert(all(found));
        R.trainY=[A.yf(loc,:);A.yi(loc,:)];
        sources{k}=R; values=[values;R.Y;R.trainY]; %#ok<AGROW>
    end
    low=min(values,[],1); high=max(values,[],1); span=max(high-low,.1);
    View=struct('lower',low-.05*span,'upper',high+.05*span);
    F=figure('Visible','off','Color','w','Position',[30 50 1350 950]);
    tl=tiledlayout(F,2,2,'TileSpacing','compact','Padding','compact');
    for k=1:4
        R=sources{k}; ax=nexttile(tl); hold(ax,'on');
        draw_CBS_CGAN_objective_region(ax,P,name,struct('feasible',[.78 .88 .79], ...
            'infeasible',[.96 .83 .83]),View);
        h1=scatter(ax,R.trainY(:,1),R.trainY(:,2),16,[.2 .2 .2],'x');
        h2=scatter(ax,R.Y(~R.hidden,1),R.Y(~R.hidden,2),10,[.25 .45 .75],'filled','MarkerFaceAlpha',.25);
        m=R.hidden & R.QueryC(:,end)==1;
        h3=scatter(ax,R.Y(m,1),R.Y(m,2),32,[.02 .45 .15],'o','LineWidth',1.1);
        m=R.hidden & R.QueryC(:,end)==0;
        h4=scatter(ax,R.Y(m,1),R.Y(m,2),32,[.80 .10 .10],'s','LineWidth',1.1);
        budget=1000; if k>2; budget=4000; end
        state='Full training directions'; if mod(k,2)==0; state='Directions 44-60 withheld'; end
        title(ax,sprintf('%s | %d total G updates\nInterval angle %.2f deg | label %.1f%% | joint %.1f%%', ...
            state,budget,R.M.heldAngle,100*R.M.heldLabel,100*R.M.heldJoint));
        xlim(ax,[View.lower(1),View.upper(1)]); ylim(ax,[View.lower(2),View.upper(2)]);
        xlabel(ax,'f_1'); ylabel(ax,'f_2'); box(ax,'on'); grid(ax,'on');
        legend(ax,[h1,h2,h3,h4],{'Real training endpoints','Other query directions', ...
            'Interval: requested feasible','Interval: requested infeasible'}, ...
            'Location','southoutside','NumColumns',2,'FontSize',8);
    end
    title(tl,sprintf('%s | same fixed archive; separate full/held-out models | 1000 -> 4000 keeps weights and Adam',name), ...
        'Interpreter','none');
    exportgraphics(F,fullfile(folder,sprintf('%s_direction_holdout.png',name)),'Resolution',135); close(F);
end
disp('HOLDOUT_PLOTS_PASS; cached objectives only, no additional CalObj/CalCon calls.');
end
