function finalize_first_metadata
% Correct early cache identities without inventing a runtime source hash.
folder=fileparts(mfilename('fullpath'));
files=dir(fullfile(folder,'first_use','LIRCMOP*_seed*_*.mat'));
original=struct('initialEpoch',200,'retrainEpoch',20,'nCritic',5, ...
    'lrG',1e-4,'lrD',1e-4,'miniBatch',32,'generatorHidden',[32 32], ...
    'criticHidden',[32 32],'trainingSigma',0.3,'sampleSigma',0.3, ...
    'gpLambda',10,'mismatchFraction',0,'zDim',6);
unrecorded=0; rows=cell(numel(files),1);
for k=1:numel(files)
    file=fullfile(files(k).folder,files(k).name);
    R=load(file,'Record','TrainingOptions'); Record=R.Record;
    ResolvedTrainingOptions=original;
    for name=string(fieldnames(R.TrainingOptions))'
        ResolvedTrainingOptions.(name)=R.TrainingOptions.(name);
    end
    exact=~isempty(regexp(char(Record.sourceHash),'^[0-9a-f]{64}$','once'));
    SourceProvenance=struct('exactRuntimeHashRecorded',exact, ...
        'runtimeSourceHash',Record.sourceHash,'snapshotManifest',"",'note',"");
    if ~exact
        if ~isfield(Record,'cacheKey'); Record.cacheKey=Record.sourceHash; end
        Record.sourceHash="";
        SourceProvenance.runtimeSourceHash="";
        SourceProvenance.snapshotManifest="first_use_source.json";
        SourceProvenance.note="Early arm identity was a cache key, not a runtime hash. Phase source snapshot retained; no retroactive exact-hash claim.";
        unrecorded=unrecorded+1;
    elseif Record.sourceHash=="85afe1987a17b1d88d0fbbd0ebf72f01aed9fb56f15d72a4febc31e87cc9889e"
        SourceProvenance.snapshotManifest="mismatch_source.json";
    end
    save(file,'Record','ResolvedTrainingOptions','SourceProvenance','-append');
    rows{k}=struct('file',string(files(k).name),'exactRuntimeHashRecorded',exact, ...
        'runtimeSourceHash',SourceProvenance.runtimeSourceHash, ...
        'snapshotManifest',SourceProvenance.snapshotManifest);
end
writetable(struct2table(vertcat(rows{:})),fullfile(folder,'first_source_provenance.csv'));
fprintf('FIRST_METADATA cases=%d earlyWithoutExactHash=%d; numerical results unchanged.\n',numel(files),unrecorded);
end
