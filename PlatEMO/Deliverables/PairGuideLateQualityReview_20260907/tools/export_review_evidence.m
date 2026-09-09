function export_review_evidence(repoRoot,packageRoot)
% Export cached evidence only. No training, sampling or objective evaluation.
warning('off','all'); maxNumCompThreads(1);
campaign=fullfile(repoRoot,'Data','PairGuideSinglePoint_20260907');
out=fullfile(packageRoot,'numeric_evidence');
if ~isfolder(out); mkdir(out); end
P=load(fullfile(campaign,'full_run','protocol.mat'),'Protocol');
writeJSON(fullfile(out,'full_protocol.json'),P.Protocol);
for number=5:8
    name=sprintf('LIRCMOP%d_BC',number);
    source=fullfile(campaign,'full_run','analysis','figures',name,'run_01','plot_data.mat');
    Cached=load(source,'States','Report');
    for k=1:numel(Cached.States)
        S=Cached.States{k};
        if isfield(S,'focusFile'); S=rmfield(S,'focusFile'); end
        writeJSON(fullfile(out,sprintf('%s_run01_stage%02d_plot_state.json',name,k)),S);
        native=[S.native,S.sides,S.trueFeasible];
        T=array2table(native,'VariableNames',{'f1','f2','requested_side','true_feasible'});
        writetable(T,fullfile(out,sprintf('%s_run01_stage%02d_raw_objectives.csv',name,k)));
    end
end
files=dir(fullfile(campaign,'holdout','*_held*.mat'));
for k=1:numel(files)
    source=fullfile(files(k).folder,files(k).name);
    H=load(source,'M','D','Full','O','X','Y','C','QueryC','heldRefs','removed', ...
        'angles','correct','hidden','OfflineCost','Model');
    H.Model=plainModel(H.Model);
    [~,stem]=fileparts(source);
    writeJSON(fullfile(out,[stem,'.json']),H);
end
files=dir(fullfile(campaign,'full_run','LIRCMOP*_seed*_cgan.mat'));
assert(numel(files)==12);
for k=1:numel(files)
    source=fullfile(files(k).folder,files(k).name);
    [~,stem]=fileparts(source); timer=tic;
    fprintf('LOADING %s\n',stem);
    R=load(source,'Record','Audit'); E=R.Audit.evidence;
    assert(E.schema=="PairGuide-single-v3");
    writeJSON(fullfile(out,[stem,'_training_events.json']),E.training);
    traces=cell(size(E.generations));
    for j=1:numel(traces)
        G=E.generations{j}; traces{j}=scalarFields(G);
    end
    writeJSON(fullfile(out,[stem,'_generation_counters.json']),traces);
    population=cell(size(E.population));
    for j=1:numel(population)
        S=E.population{j};
        population{j}=struct('observationFE',S.observationFE, ...
            'archivePairs',numel(S.archive.id),'activePairs',nnz(S.archive.active), ...
            'feasibleP1',nnz(all(S.p1Cons<=0,2)),'referenceScale',S.referenceScale);
    end
    writeJSON(fullfile(out,[stem,'_population_counters.json']),population);
    if R.Record.seed==1
        name=char(R.Record.problem);
        Cached=load(fullfile(campaign,'full_run','analysis','figures',name,'run_01','plot_data.mat'),'States');
        for j=1:numel(Cached.States)
            S=Cached.States{j};
            if S.missingGeneration; continue; end
            q=find(cellfun(@(Q)Q.productionFE==S.productionFE,E.queries),1);
            assert(~isempty(q)); Q=E.queries{q};
            writeJSON(fullfile(out,sprintf('%s_run01_stage%02d_query.json',name,j)),Q);
        end
        writeJSON(fullfile(out,[stem,'_first_model.json']),plainModel(E.firstModel));
        writeJSON(fullfile(out,[stem,'_last_model.json']),plainModel(E.lastModel));
    end
    fprintf('EXPORTED %s %.1fs\n',stem,toc(timer));
    clear R E Cached;
end
writeJSON(fullfile(out,'export_status.json'),struct('complete',true, ...
    'cganRuns',12,'plotStates',24,'holdoutModels',numel(dir(fullfile(campaign,'holdout','*_held*.mat'))), ...
    'newTrainingCalls',0,'newObjectiveCalls',0,'newConstraintCalls',0));
fprintf('EXPORT_COMPLETE\n');
end

function out=scalarFields(S)
out=struct();
for field=string(fieldnames(S))'
    value=S.(field);
    if isstruct(value) && isscalar(value)
        out.(field)=scalarFields(value);
    elseif (isnumeric(value) || islogical(value)) && isscalar(value)
        out.(field)=value;
    elseif ischar(value) || (isstring(value) && isscalar(value))
        out.(field)=value;
    end
end
end

function out=plainModel(Model)
out=rmfield(Model,{'netG','netC','avgG','avgSqG','avgC','avgSqC'});
out.generator=plainTable(Model.netG.Learnables);
out.critic=plainTable(Model.netC.Learnables);
for field=["avgG","avgSqG","avgC","avgSqC"]
    if istable(Model.(field)); out.(field)=plainTable(Model.(field));
    else; out.(field)=[]; end
end
end

function out=plainTable(T)
out=cell(height(T),1);
for k=1:height(T)
    V=T.Value{k};
    if isa(V,'dlarray'); V=extractdata(V); end
    out{k}=struct('layer',string(T.Layer(k)),'parameter',string(T.Parameter(k)), ...
        'shape',size(V),'value',double(gather(V)));
end
end

function writeJSON(file,value)
text=jsonencode(value); % MATLAB NaN/Inf become JSON null; never zero-fill.
fid=fopen(file,'w','n','UTF-8'); assert(fid>=0);
cleanup=onCleanup(@()fclose(fid));
fprintf(fid,'%s\n',text);
end
