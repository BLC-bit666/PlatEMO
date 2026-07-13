function value = CBS_RegionGAN_SourceManifestSHA256(Manifest)
%CBS_REGIONGAN_SOURCEMANIFESTSHA256 Hash a canonical source manifest.

    required = {'relative_path','sha256','bytes'};
    if ~istable(Manifest) || ...
            ~all(ismember(required,Manifest.Properties.VariableNames))
        error('CBSRegionGAN:BadSourceManifest', ...
            'Source manifest must contain relative_path, sha256, and bytes.');
    end
    paths = string(Manifest.relative_path(:));
    hashes = lower(string(Manifest.sha256(:)));
    bytes = double(Manifest.bytes(:));
    validHashes = arrayfun(@(x)~isempty(regexp(char(x), ...
        '^[0-9a-f]{64}$','once')),hashes);
    if isempty(paths) || any(strlength(paths) == 0) || ...
            numel(unique(paths)) ~= numel(paths) || ~all(validHashes) || ...
            any(~isfinite(bytes) | bytes < 0 | bytes ~= fix(bytes))
        error('CBSRegionGAN:BadSourceManifest', ...
            'Source manifest contains invalid or duplicate rows.');
    end
    [paths,order] = sort(paths);
    hashes = hashes(order);
    records = paths + char(9) + hashes + newline;
    payload = unicode2native(char(join(records,"")),'UTF-8');
    digest = java.security.MessageDigest.getInstance('SHA-256');
    digest.update(payload);
    value = lower(string(reshape(dec2hex(typecast( ...
        digest.digest(),'uint8'),2).',1,[])));
end
