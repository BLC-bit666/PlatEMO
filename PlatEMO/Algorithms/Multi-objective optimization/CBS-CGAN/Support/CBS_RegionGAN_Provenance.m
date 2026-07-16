function P = CBS_RegionGAN_Provenance(repoRoot,Options,workerCount)
%CBS_REGIONGAN_PROVENANCE Build deterministic source and run provenance.

    if ~(ischar(repoRoot) || (isstring(repoRoot) && isscalar(repoRoot)))
        error('CBSRegionGAN:BadRepoRoot', ...
            'repoRoot must be a character vector or string scalar.');
    end
    repoRoot = char(repoRoot);
    try
        rootFile = java.io.File(repoRoot);
        repoRoot = char(rootFile.getCanonicalPath());
    catch
        while numel(repoRoot) > 1 && ismember(repoRoot(end),['/','\'])
            repoRoot(end) = [];
        end
    end
    if ~isfolder(repoRoot)
        error('CBSRegionGAN:MissingRepoRoot', ...
            'Repository root does not exist: %s',repoRoot);
    end
    if ~(isnumeric(workerCount) && isscalar(workerCount) && ...
            isreal(workerCount) && isfinite(workerCount))
        error('CBSRegionGAN:BadWorkerCount', ...
            'workerCount must be a finite real numeric scalar.');
    end

    Manifest = buildSourceManifest(repoRoot);
    schemaVersion = "cbs_region_wgan_igd_mainline_v1";
    if isstruct(Options) && isfield(Options,'schemaVersion') && ...
            ~isempty(Options.schemaVersion)
        schemaVersion = string(Options.schemaVersion);
    end
    P = struct( ...
        'schema_version',schemaVersion, ...
        'git_sha',gitValue(repoRoot,'rev-parse HEAD',true), ...
        'git_branch',gitValue(repoRoot,'rev-parse --abbrev-ref HEAD',false), ...
        'git_dirty',gitDirty(repoRoot), ...
        'matlab_release',string(version('-release')), ...
        'host',hostName(), ...
        'worker_count',double(workerCount), ...
        'options_json',encodeOptions(Options), ...
        'source_tree_sha256',CBS_RegionGAN_SourceManifestSHA256(Manifest), ...
        'source_manifest',Manifest);
end

function Manifest = buildSourceManifest(repoRoot)
    algorithmRoot = fullfile(repoRoot,'Algorithms', ...
        'Multi-objective optimization','CBS-CGAN');
    if ~isfolder(algorithmRoot)
        error('CBSRegionGAN:MissingAlgorithmRoot', ...
            'CBS-CGAN source directory does not exist: %s',algorithmRoot);
    end

    listing = dir(fullfile(algorithmRoot,'**','*.m'));
    absolutePaths = strings(numel(listing) + 6,1);
    for i = 1 : numel(listing)
        absolutePaths(i) = string(fullfile(listing(i).folder,listing(i).name));
    end
    problemRoot = fullfile(repoRoot,'Problems', ...
        'Multi-objective optimization','LIR-CMOP_BC');
    for problem = 5 : 10
        index = numel(listing) + problem - 4;
        absolutePaths(index) = string(fullfile(problemRoot, ...
            sprintf('LIRCMOP%d_BC.m',problem)));
        if ~isfile(absolutePaths(index))
            error('CBSRegionGAN:MissingProblemSource', ...
                'Required problem definition does not exist: %s', ...
                absolutePaths(index));
        end
    end

    prefixLength = strlength(string(repoRoot)) + 1;
    relativePaths = replace(extractAfter(absolutePaths,prefixLength),"\","/");
    [relativePaths,order] = sort(relativePaths);
    absolutePaths = absolutePaths(order);
    hashes = strings(numel(absolutePaths),1);
    byteCounts = zeros(numel(absolutePaths),1);
    for i = 1 : numel(absolutePaths)
        [hashes(i),byteCounts(i)] = fileSHA256(absolutePaths(i));
    end
    Manifest = table(relativePaths,hashes,byteCounts, ...
        'VariableNames',{'relative_path','sha256','bytes'});
end

function [value,byteCount] = fileSHA256(filePath)
    [fileID,message] = fopen(char(filePath),'rb');
    if fileID < 0
        error('CBSRegionGAN:SourceReadFailed', ...
            'Cannot read source file %s: %s',filePath,message);
    end
    cleanup = onCleanup(@()fclose(fileID));
    bytes = fread(fileID,Inf,'*uint8');
    byteCount = numel(bytes);
    value = byteSHA256(bytes);
end

function value = byteSHA256(bytes)
    digest = java.security.MessageDigest.getInstance('SHA-256');
    digest.update(bytes);
    value = lower(string(reshape(dec2hex(typecast( ...
        digest.digest(),'uint8'),2).',1,[])));
end

function value = gitValue(repoRoot,arguments,requireSHA)
    [status,output] = system(sprintf('git -C %s %s 2>/dev/null', ...
        shellQuote(repoRoot),arguments));
    value = strip(string(output));
    if status ~= 0 || strlength(value) == 0
        value = "unknown";
    elseif requireSHA && isempty(regexp(char(value), ...
            '^[0-9a-fA-F]{40}$','once'))
        value = "unknown";
    end
end

function value = gitDirty(repoRoot)
    [status,output] = system(sprintf( ...
        'git -C %s status --porcelain 2>/dev/null',shellQuote(repoRoot)));
    value = logical(status == 0 && strlength(strip(string(output))) > 0);
end

function value = shellQuote(textValue)
    escaped = replace(string(textValue),"'","'""'""'");
    value = char("'" + escaped + "'");
end

function value = encodeOptions(Options)
    try
        value = string(jsonencode(Options));
    catch exception
        identifier = string(exception.identifier);
        if strlength(identifier) == 0
            identifier = "unknown_error";
        end
        value = "jsonencode_failed:class=" + string(class(Options)) + ...
            ";identifier=" + identifier;
    end
end

function value = hostName()
    value = strip(string(getenv('HOSTNAME')));
    if strlength(value) == 0
        value = strip(string(getenv('COMPUTERNAME')));
    end
    if strlength(value) == 0
        try
            value = strip(string( ...
                java.net.InetAddress.getLocalHost().getHostName()));
        catch
            value = "unknown";
        end
    end
end
