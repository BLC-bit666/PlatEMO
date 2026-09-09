function prepare_stage_baselines
warning('off','all');maxNumCompThreads(1);
folder=fileparts(mfilename('fullpath'));root=fileparts(fileparts(folder));
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support'));addCBSPaths(root);addpath(folder);
out=fullfile(folder,'baselines');if ~isfolder(out);mkdir(out);end
files=dir(fullfile(folder,'fixtures','*.mat')); rows={};
for k=1:numel(files)
    source=fullfile(files(k).folder,files(k).name);target=fullfile(out,files(k).name);
    F=load(source);
    if isfile(target);B=load(target);R=B.R;
    else
        [R,S]=measure_stage_model(F.Model,F);save(target,'R','S');
    end
    R.problem=string(F.Meta.problem);R.searchSeed=F.Meta.searchSeed;R.stage=string(F.Meta.stage);
    R.observationFE=F.Meta.observationFE;R.trainingFE=F.Meta.trainingFE;
    R.currentArchivePairs=F.Meta.currentArchivePairs;R.currentActivePairs=F.Meta.currentActivePairs;
    R.fixedTrainingPairs=F.Meta.fixedTrainingPairs;rows{end+1}=R;
end
writetable(struct2table(vertcat(rows{:})),fullfile(folder,'stage_baselines.csv'));
fprintf('STAGE_BASELINES_READY %d cached datasets, 200 separately charged evaluations each\n',numel(rows));
end
