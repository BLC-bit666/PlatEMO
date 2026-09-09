function Report = render_PairGuide_interval_distribution(resultFile)
%RENDER_PAIRGUIDE_INTERVAL_DISTRIBUTION CGAN seed 1, offline objective plots.
%   First actual guided use and every 10000 FE. Search is already finished.
%   Additional oracle calls are cached and billed separately after search.
    R=load(resultFile,'Record','Audit');
    resultInfo=dir(resultFile);
    digest=java.security.MessageDigest.getInstance('SHA-256');
    digest.update(unicode2native(fileread([mfilename('fullpath'),'.m']),'UTF-8'));
    rendererHash=lower(string(reshape(dec2hex(typecast(digest.digest(),'uint8'),2)',1,[])));
    assert(R.Record.mode=="cgan" && R.Record.seed==1);
    paired=isfield(R.Record,'distributionLayout') && R.Record.distributionLayout=="training_pair";
    targetsFE=unique([10000:10000:R.Record.finalFE,R.Record.finalFE]);
    if isfield(R.Record,'checkpointFE'); targetsFE=R.Record.checkpointFE; end
    folder=fullfile(fileparts(resultFile),'analysis','figures',char(R.Record.problem),'run_01');
    if ~isfolder(folder); mkdir(folder); end
    cacheFile=fullfile(folder,'plot_data.mat');
    if isfile(cacheFile)
        old=load(cacheFile,'Report');
        if isfield(old.Report,'rendererVersion') && old.Report.rendererVersion==7 && ...
                old.Report.rendererSourceHash==rendererHash && ...
                old.Report.resultFile==string(resultFile) && ...
                old.Report.resultBytes==resultInfo.bytes && old.Report.resultDate==resultInfo.datenum && ...
                old.Report.sourceHash==R.Record.sourceHash && all(isfile(old.Report.files)) && ...
                (~paired || (isfield(old.Report,'focusFiles') && all(isfile(old.Report.focusFiles))))
            Report=old.Report; return;
        end
    end
    saved=rng; restore=onCleanup(@()rng(saved));
    constructor=str2func(char(R.Record.problem));
    Problem=constructor(); % Default dimension, independent of the search instance.
    assert(Problem.D==R.Record.D && Problem.M==2);
    PairGuideCost_RC('start',Problem);
    cleanup=onCleanup(@()PairGuideCost_RC('stop'));
    E=R.Audit.evidence;
    singleMode=E.schema=="PairGuide-single-v3";
    nativeMode=singleMode || E.schema=="PairGuide-native-v2";
    times=cellfun(@(p)p.observationFE,E.population);
    first=find(cellfun(@(g)g.use.selected>0,E.generations),1);
    indices=[]; targets=[]; labels=strings(0,1);
    if ~isempty(first)
        indices=first+1; targets=times(first+1); labels="first_true_use";
    end
    for target=targetsFE(targetsFE<=R.Record.finalFE)
        p=find(times<=target,1,'last');
        if isempty(p); continue; end
        indices(end+1)=p; targets(end+1)=target; %#ok<AGROW>
        labels(end+1)=sprintf('FE%06d',target); %#ok<AGROW>
    end
    queryCache=cell(size(E.queries));
    States=cell(numel(indices),1); files=strings(numel(indices),1); focusFiles=strings(0,1);
    timer=tic;
    for k=1:numel(indices)
        P=E.population{indices(k)}; active=P.archive.active;
        S=struct('targetFE',targets(k),'observationFE',P.observationFE, ...
            'productionFE',NaN,'consumptionFE',NaN,'modelVersion',NaN, ...
            'missingGeneration',true,'stale',P.observationFE<targets(k), ...
            'p1',P.p1Objs,'p2',P.p2Objs,'xf',P.archive.yf(active,:), ...
            'xi',P.archive.yi(active,:),'native',zeros(0,2), ...
            'candidates',zeros(0,2),'pending',zeros(0,2),'children',zeros(0,2), ...
            'nativeMode',nativeMode,'singleMode',singleMode,'sides',zeros(0,1), ...
            'trueFeasible',zeros(0,1),'labelAccuracy',NaN,'directionError',NaN, ...
            'nativeBandRate',NaN,'projectionMedian',NaN, ...
            'inactiveF',P.archive.yf(~active,:),'inactiveI',P.archive.yi(~active,:), ...
            'trainingFE',NaN,'trainingEpochs',NaN,'trainingUpdates',NaN, ...
            'trainingRMSE',NaN,'trainingPairs',NaN,'searchState',P);
        if indices(k)>1
            G=E.generations{indices(k)-1}; U=G.use;
            S.children=U.childObjs; S.consumptionFE=G.consumptionFE;
            q=find(cellfun(@(query)query.productionFE==U.productionFE,E.queries),1);
            if ~isempty(q)
                Q=E.queries{q}; S.productionFE=Q.productionFE;
                S.modelVersion=Q.modelVersion; S.missingGeneration=false;
                if paired
                    trained=E.training(cellfun(@(t)t.trained && t.generation<=Q.generation,E.training));
                    assert(numel(trained)==Q.modelVersion,'Model lineage must identify its actual training set.');
                    T=trained{end}; Train=E.population{T.generation+1}; A=Train.archive;
                    S.trainingState=Train; S.generationState=E.population{Q.generation+1};
                    S.p1=Train.p1Objs; S.p2=Train.p2Objs;
                    S.xf=A.yf(A.active,:); S.xi=A.yi(A.active,:);
                    S.inactiveF=A.yf(~A.active,:); S.inactiveI=A.yi(~A.active,:);
                    S.trainingFE=Train.observationFE; S.trainingEpochs=T.epochs;
                    S.trainingUpdates=T.updates; S.trainingRMSE=T.postDiagnostics.allEndpointRMSE;
                    S.trainingPairs=T.trainingPairs;
                    assert(size(S.xf,1)==T.trainingPairs);
                end
                if isempty(queryCache{q})
                    raw=double(Q.rawDecs);
                    if nativeMode; candidates=double(Q.pool.candidateDecs);
                    else; candidates=double(Q.pool.constructedDecs); end
                    X=[raw;candidates]; valid=all(isfinite(X),2);
                    Y=nan(size(X,1),Problem.M); feasible=nan(size(X,1),1);
                    [uniqueX,~,map]=unique(X(valid,:),'rows');
                    if ~isempty(uniqueX)
                        uniqueY=Problem.CalObj(uniqueX); Y(valid,:)=uniqueY(map,:);
                        if singleMode
                            uniqueC=Problem.CalCon(uniqueX); feasible(valid)=double(all(uniqueC(map,:)<=0,2));
                        end
                    end
                    queryCache{q}=struct('native',Y(1:size(raw,1),:), ...
                        'candidates',Y(size(raw,1)+1:end,:),'feasible',feasible(1:size(raw,1)));
                end
                S.native=queryCache{q}.native;
                if singleMode
                    S.sides=Q.sides; S.trueFeasible=queryCache{q}.feasible;
                    S.labelAccuracy=mean(S.trueFeasible==S.sides);
                    frame=E.population{Q.generation+1}.referenceScale;
                    [~,~,yn]=AssignReferenceVectors_CBS(S.native,E.W,frame);
                    target=Q.conditions(:,1:end-1); target=target./max(vecnorm(target,2,2),eps);
                    yn=yn./max(vecnorm(yn,2,2),eps);
                    S.directionError=mean(acosd(max(-1,min(1,sum(target.*yn,2)))));
                end
                S.candidates=queryCache{q}.candidates;
                S.pending=S.candidates(Q.pool.keepIdx,:);
                if nativeMode
                    assert(isequal(Q.pending.decs,Q.rawDecs(Q.pool.keepIdx,:)) && ...
                        isequal(S.pending,S.children),'Plotted selections must equal actual evaluated children.');
                end
                S.nativeBandRate=mean(Q.pool.nativeInBand);
                if ~nativeMode; S.projectionMedian=median(Q.pool.correction,'omitnan'); end
            end
        end
        files(k)=string(fullfile(folder,sprintf('%s_%s_actualFE%06d.png', ...
            R.Record.problem,labels(k),S.observationFE)));
        renderFigure(Problem,R.Record.problem,S,labels(k),files(k),paired,false);
        if paired
            focusFile=replace(files(k),'.png','_focus.png');
            renderFigure(Problem,R.Record.problem,S,labels(k),focusFile,true,true);
            S.focusFile=focusFile;
            focusFiles(end+1,1)=focusFile; %#ok<AGROW>
        end
        States{k}=S;
    end
    assert(Problem.FE==0,'Offline CalObj plotting changed full-evaluation FE.');
    Report=struct('rendererVersion',7,'rendererSourceHash',rendererHash, ...
        'sourceHash',R.Record.sourceHash,'problem',R.Record.problem, ...
        'resultFile',string(resultFile),'resultBytes',resultInfo.bytes,'resultDate',resultInfo.datenum, ...
        'seed',1,'mode',"cgan",'files',files,'focusFiles',focusFiles, ...
        'figureCount',numel(files)+numel(focusFiles), ...
        'searchFE',R.Record.finalFE,'offlineFullFE',Problem.FE, ...
        'oracleCalls',PairGuideCost_RC('snapshot'),'wallSeconds',toc(timer), ...
        'firstTrueUseFE',Inf,'label',"Offline visualization; never fed back to search");
    if ~isempty(first); Report.firstTrueUseFE=times(first+1); end
    save(cacheFile,'Report','States','-v7.3');
    fprintf('PLOTS %s run=1 figures=%d offlineCalObjRows=%d\n', ...
        R.Record.problem,Report.figureCount,Report.oracleCalls.CalObjRows);
end

function renderFigure(Problem,problem,S,label,file,paired,focused)
    C=struct('feasible',[0.36 0.72 0.47],'infeasible',[0.90 0.43 0.40]);
    values=[S.p1;S.p2;S.xf;S.xi;S.inactiveF;S.inactiveI;S.native;S.candidates;S.children];
    values=values(all(isfinite(values),2),:);
    if isempty(values); values=[0 0;1 1]; end
    low=min(values,[],1); high=max(values,[],1); span=max(high-low,0.1);
    View=struct('lower',low-0.06*span,'upper',high+0.06*span);
    if focused
        focus=[S.xf;S.xi;S.native]; focus=focus(all(isfinite(focus),2),:);
        if ~isempty(focus)
            limits=prctile(focus,[1 99],1); span=max(limits(2,:)-limits(1,:),0.1);
            View.lower=limits(1,:)-0.06*span; View.upper=limits(2,:)+0.06*span;
        end
    end
    F=figure('Visible','off','Color','w','Position',[50 100 1900 850]);
    cleanup=onCleanup(@()close(F));
    panes=3; if paired; panes=2; end
    layout=tiledlayout(F,1,panes,'TileSpacing','compact','Padding','compact');
    for pane=1:panes
        ax=nexttile(layout); hold(ax,'on');
        [hf,hi]=draw_CBS_CGAN_objective_region(ax,Problem,problem,C,View);
        if pane==1
            h0f=scatter(ax,S.inactiveF(:,1),S.inactiveF(:,2),18,[.40 .52 .42],'o','MarkerEdgeAlpha',.22);
            h0i=scatter(ax,S.inactiveI(:,1),S.inactiveI(:,2),18,[.60 .43 .43],'s','MarkerEdgeAlpha',.22);
            for j=1:size(S.xf,1)
                plot(ax,[S.xf(j,1),S.xi(j,1)],[S.xf(j,2),S.xi(j,2)], ...
                    '-','Color',[0.45 0.45 0.45],'LineWidth',0.5);
            end
            h2=scatter(ax,S.p2(:,1),S.p2(:,2),42,[0 .51 .59],'^','LineWidth',1.2);
            h1=scatter(ax,S.p1(:,1),S.p1(:,2),42,[.51 .17 .70],'o','LineWidth',1.2);
            h3=points(ax,S.xf,23,[0.05 0.45 0.16],'o',0.9);
            h4=points(ax,S.xi,23,[0.78 0.13 0.14],'s',0.9);
            title(ax,sprintf('Search state | active pairs %d',size(S.xf,1)));
            feasibleLabel='Active feasible endpoint';
            if paired; feasibleLabel='Active feasible endpoint: front 1'; end
            legend(ax,[h1 h2 h3 h4 h0f h0i hf hi],{sprintf('P1 constrained (%d)',size(S.p1,1)), ...
                sprintf('P2 unconstrained (%d)',size(S.p2,1)),feasibleLabel, ...
                'Active infeasible endpoint (train)','Inactive feasible endpoint', ...
                'Inactive infeasible endpoint','Feasible domain','Infeasible domain'}, ...
                'Location','southoutside','NumColumns',2,'FontSize',9);
            if paired && ~S.missingGeneration
                title(ax,sprintf('Model training FE %g | archive %d pairs, active %d', ...
                    S.trainingFE,size(S.xf,1)+size(S.inactiveF,1),size(S.xf,1)));
            end
        elseif pane==2
            h1=points(ax,S.native,14,[0.13 0.43 0.82],'o',0.45);
            if S.singleMode
                delete(h1);
                h1=points(ax,S.native(S.sides==1,:),17,[.05 .45 .16],'o',.55);
                h0=points(ax,S.native(S.sides==0,:),17,[.78 .13 .14],'s',.55);
            end
            title(ax,sprintf('Native c | relative in-band %.1f%%',100*S.nativeBandRate));
            if paired
                h2=points(ax,S.pending,75,[1 .63 .08],'d',.95);
                legend(ax,[h1 h2],{sprintf('CGAN native candidates (%d)',size(S.native,1)), ...
                    sprintf('Selected and truly used (%d)',size(S.pending,1))}, ...
                    'Location','southoutside','FontSize',11);
                title(ax,'CGAN generated / selected candidates | original coordinates');
                if S.singleMode
                    delete(h2); h2=points(ax,S.children,75,[1 .63 .08],'d',.95);
                    legend(ax,[h1 h0 h2],{'Requested feasible (s=1)', ...
                        'Requested infeasible (s=0)','Selected and truly evaluated'}, ...
                        'Location','southoutside','FontSize',10);
                    title(ax,'Single-point CGAN | colors are requests, not true feasibility');
                end
            else
                legend(ax,h1,{sprintf('native proposals (%d)',size(S.native,1))}, ...
                    'Location','southoutside','FontSize',9);
            end
        else
            if ~S.nativeMode
                h1=points(ax,S.candidates,13,[0.22 0.52 0.55],'o',0.4);
            end
            h2=points(ax,S.pending,40,[0.86 0.20 0.58],'v',0.9);
            h3=points(ax,S.children,55,[1 0.63 0.08],'d',0.95);
            if S.nativeMode
                title(ax,'Selected native c | unchanged coordinates');
                legend(ax,[h2 h3],{sprintf('Pending native c (%d)',size(S.pending,1)), ...
                    sprintf('true children (%d)',size(S.children,1))}, ...
                    'Location','southoutside','FontSize',9);
            else
                title(ax,sprintf('Constructed q | median correction / gap %.3g',S.projectionMedian));
                legend(ax,[h1 h2 h3],{sprintf('projected proposals (%d)',size(S.candidates,1)), ...
                    sprintf('Pending (%d)',size(S.pending,1)),sprintf('true children (%d)',size(S.children,1))}, ...
                    'Location','southoutside','FontSize',9);
            end
        end
        xlim(ax,[View.lower(1) View.upper(1)]); ylim(ax,[View.lower(2) View.upper(2)]);
        xlabel(ax,'f_1'); ylabel(ax,'f_2'); box(ax,'on'); grid(ax,'on');
        set(ax,'FontName','Helvetica','FontSize',10);
        if pane>1 && S.missingGeneration
            text(ax,0.5,0.5,'No matching generation event', ...
                'Units','normalized','HorizontalAlignment','center');
        end
    end
    note='Full view: all population points, archive endpoint slots and generated candidates included';
    if focused
        outside=@(X)nnz(any(X<View.lower | X>View.upper,2));
        note=sprintf('Focus; outside axes: P1 %d, P2 %d, archive slots %d, native %d, selected %d; see full view', ...
            outside(S.p1),outside(S.p2),outside([S.xf;S.xi;S.inactiveF;S.inactiveI]), ...
            outside(S.native),outside(S.pending));
    end
    trainingNote=sprintf('Last training: FE %g | %g pairs | %g epochs / %g G updates | endpoint RMSE %.4g', ...
        S.trainingFE,S.trainingPairs,S.trainingEpochs,S.trainingUpdates,S.trainingRMSE);
    if S.singleMode
        trainingNote=sprintf('Last training FE %g | %g pairs | %g G updates | true label accuracy %.1f%% | direction error %.1f deg', ...
            S.trainingFE,S.trainingPairs,S.trainingUpdates,100*S.labelAccuracy,S.directionError);
    end
    eventNote=sprintf('production FE %g | consumption FE %g | model %g | stale=%d | raw objectives: offline only', ...
        S.productionFE,S.consumptionFE,S.modelVersion,S.stale);
    if S.missingGeneration
        eventNote='No CGAN batch consumed at this checkpoint; missing data are not replaced by an older cloud';
        trainingNote=sprintf('Current search state: %d archive pairs, %d active | no matched training/query event', ...
            size(S.xf,1)+size(S.inactiveF,1),size(S.xf,1));
    end
    title(layout,{sprintf('%s | cgan run 1 | %s | target FE %g | observed FE %g', ...
        problem,label,S.targetFE,S.observationFE), ...
        eventNote, ...
        trainingNote,note}, ...
        'Interpreter','none','FontSize',12);
    exportgraphics(F,file,'Resolution',150);
end

function H=points(ax,X,sizeValue,color,marker,alpha)
    X=X(all(isfinite(X),2),:);
    H=scatter(ax,X(:,1),X(:,2),sizeValue,color,marker,'filled', ...
        'MarkerFaceAlpha',alpha,'MarkerEdgeColor',[0.15 0.15 0.15],'LineWidth',0.35);
end
