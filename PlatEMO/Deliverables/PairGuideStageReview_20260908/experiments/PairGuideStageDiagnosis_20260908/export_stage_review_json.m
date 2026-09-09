function export_stage_review_json
% Cached states only: no training or objective/constraint evaluations.
warning('off','all');maxNumCompThreads(1);
folder=fileparts(mfilename('fullpath'));root=fileparts(fileparts(folder));
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support'));addCBSPaths(root);
out=fullfile(root,'Deliverables','PairGuideStageReview_20260908','numeric_evidence');
assert(isfolder(out));index={};
for directory={'fixtures','baselines','runs','low_lr_runs','suffix_fixtures','suffix'}
    files=dir(fullfile(folder,directory{1},'*.mat'));
    for k=1:numel(files)
        if strcmp(files(k).name,'status.mat');continue;end
        source=fullfile(files(k).folder,files(k).name);R=load(source);
        if isfield(R,'Complete');assert(R.Complete);end
        if isfield(R,'RngEnds');R=rmfield(R,'RngEnds');end
        if isfield(R,'History');R=rmfield(R,'History');end
        if isfield(R,'Snapshots')
            for j=1:numel(R.Snapshots)
                if isfield(R.Snapshots{j},'Model') && isfield(R.Snapshots{j}.Model,'probeRecords')
                    R.Snapshots{j}.Model=rmfield(R.Snapshots{j}.Model,'probeRecords');
                end
            end
        end
        [~,stem]=fileparts(files(k).name);file=[directory{1},'__',stem,'.json'];
        saveJSON(fullfile(out,file),R);
        index{end+1}=struct('file',file,'source',source,'selection', ...
            'All saved variables except training RngEnds, repeated Model.probeRecords and suffix History. All4 model/sample checkpoints retained. RNG pair checks, step records and search trajectories remain in CSV.');
        fprintf('STAGE_REVIEW_EXPORT %s\n',file);
    end
end
saveJSON(fullfile(root,'Deliverables','PairGuideStageReview_20260908','tools','stage_numeric_sources.json'),index);
fprintf('STAGE_REVIEW_EXPORT_COMPLETE %d records; zero training and oracle calls\n',numel(index));
end

function saveJSON(path,value)
encoded=jsonencode(plain(value));
fid=fopen(path,'w','n','UTF-8'); assert(fid>=0); cleanup=onCleanup(@()fclose(fid));
fprintf(fid,'%s\n',encoded);
end

function result=plain(value)
if isa(value,'dlarray')
    array=gather(extractdata(value));
    result=struct('type','dlarray','dataClass',class(array),'size',size(array),'values',double(array));
elseif isa(value,'dlnetwork')
    result=struct('type','dlnetwork','layerNames',{string({value.Layers.Name})}, ...
        'layerClasses',{arrayfun(@(x)string(class(x)),value.Layers)}, ...
        'learnables',plain(value.Learnables),'state',plain(value.State));
elseif istable(value)
    result=plain(table2struct(value));
elseif isstruct(value)
    result=value; fields=fieldnames(value);
    for k=1:numel(value)
        for j=1:numel(fields); result(k).(fields{j})=plain(value(k).(fields{j})); end
    end
elseif iscell(value)
    result=cellfun(@plain,value,'UniformOutput',false);
elseif isnumeric(value) || islogical(value) || ischar(value) || isstring(value)
    result=value;
elseif isa(value,'function_handle')
    result=func2str(value);
else
    error('ReviewExport:UnsupportedType','Unsupported cached type: %s',class(value));
end
end
