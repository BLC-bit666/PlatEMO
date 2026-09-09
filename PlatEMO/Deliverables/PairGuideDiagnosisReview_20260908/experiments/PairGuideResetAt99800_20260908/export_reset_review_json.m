function export_reset_review_json
% Read-only export of cached fixtures, all sampled checkpoints and final models.
% No training and no objective/constraint evaluation.
warning('off','all'); maxNumCompThreads(1);
folder=fileparts(mfilename('fullpath')); root=fileparts(fileparts(folder));
addpath(fullfile(root,'Algorithms','Multi-objective optimization','CBS-CGAN','Support'));
addCBSPaths(root);
out=fullfile(folder,'review_numeric'); if ~isfolder(out); mkdir(out); end
count=0;
files=dir(fullfile(folder,'fixtures','*.mat'));
for k=1:numel(files)
    R=load(fullfile(files(k).folder,files(k).name));
    [~,stem]=fileparts(files(k).name);
    saveJSON(fullfile(out,[stem,'_fixture.json']),R); count=count+1;
end
for directory={'runs','mismatch_runs','capacity','capacity_wide'}
    files=dir(fullfile(folder,directory{1},'*.mat'));
    for k=1:numel(files)
        if strcmp(files(k).name,'status.mat'); continue; end
        R=load(fullfile(files(k).folder,files(k).name));
        value=struct(); fields=fieldnames(R);
        for j=1:numel(fields)
            key=fields{j};
            if strcmp(key,'Snapshots')
                snapshots=R.Snapshots;
                for s=1:numel(snapshots)
                    if s~=1 && s~=numel(snapshots) && isfield(snapshots{s},'Model')
                        snapshots{s}=rmfield(snapshots{s},'Model');
                    end
                end
                value.Snapshots=snapshots;
            elseif ~ismember(key,{'Events','RngEnds'})
                value.(key)=R.(key);
            end
        end
        [~,stem]=fileparts(files(k).name);
        saveJSON(fullfile(out,[directory{1},'_',stem,'.json']),value); count=count+1;
    end
end
fprintf('RESET_REVIEW_EXPORT_COMPLETE %d files; no training or oracle calls\n',count);
end

function saveJSON(path,value)
encoded=jsonencode(plain(value));
fid=fopen(path,'w','n','UTF-8'); assert(fid>=0); cleanup=onCleanup(@()fclose(fid));
fprintf(fid,'%s\n',encoded);
end

function result=plain(value)
if isa(value,'dlarray')
    array=gather(extractdata(value));
    result=struct('type','dlarray','dataClass',class(array),'size',size(array),'values',array);
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
