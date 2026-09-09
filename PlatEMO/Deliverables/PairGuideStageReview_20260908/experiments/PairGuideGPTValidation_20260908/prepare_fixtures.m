function prepare_fixtures
warning('off','all'); maxNumCompThreads(1);
root='/Users/lanai/Code/Matlab/PlatEMO/PlatEMO';
folder=fullfile(root,'Data','PairGuideGPTValidation_20260908','fixtures');
if ~isfolder(folder); mkdir(folder); end
for number=[7 8]
    for seed=1:3
        name=sprintf('LIRCMOP%d_BC',number);
        target=fullfile(folder,sprintf('%s_seed%02d.mat',name,seed));
        if isfile(target); continue; end
        source=fullfile(root,'Data','PairGuideSinglePoint_20260907','full_run', ...
            sprintf('%s_seed%02d_cgan.mat',name,seed));
        fprintf('FIXTURE_LOAD %s seed=%d\n',name,seed);
        R=load(source,'Audit','Record'); E=R.Audit.evidence;
        Model=E.lastModel; W=E.W;
        trained=E.training(cellfun(@(t)t.trained,E.training)); T=trained{end};
        Train=E.population{T.generation+1}; Data=Model.lastData;
        [present,rows]=ismember(Data.id,Train.archive.id); assert(all(present));
        Data.yF=Train.archive.yf(rows,:); Data.yI=Train.archive.yi(rows,:); Data.W=W;
        assert(isequal(Data.xF,Train.archive.xf(rows,:)) && isequal(Data.xI,Train.archive.xi(rows,:)));
        times=cellfun(@(p)p.observationFE,E.population);
        spanRows=find(times>=80000 & times<100000);
        assert(numel(spanRows)==100);
        Spans=cellfun(@(p)p.referenceScale.span,E.population(spanRows),'UniformOutput',false);
        Spans=vertcat(Spans{:});
        Prefixes=cell(2,1);
        for j=1:2
            fe=[50000 70000]; fe=fe(j);
            k=find(times==fe,1); q=find(cellfun(@(q)q.productionFE==fe,E.queries),1);
            assert(~isempty(k) && ~isempty(q));
            Prefixes{j}=struct('state',E.population{k},'query',E.queries{q}, ...
                'generation',k-1,'seed',seed,'problem',string(name));
        end
        Record=R.Record;
        save(target,'Model','Data','Train','W','Spans','Prefixes','Record','source','-v7.3');
        fprintf('FIXTURE_SAVED %s seed=%d\n',name,seed);
        clear R E Model Train Prefixes;
    end
end
fprintf('FIXTURES_COMPLETE\n');
end
