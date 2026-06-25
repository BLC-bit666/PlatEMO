function [Metrics,ByVariantFE,ByProblemFE] = ...
        aggregate_CCMO_GAN_BDG_snapshot_metrics(baseOutDir)
% Aggregate AF-train vs raw-GAN snapshot diagnostics for CCMO_GAN_BDG.

    if nargin < 1 || isempty(baseOutDir)
        error('aggregate_CCMO_GAN_BDG_snapshot_metrics:MissingBaseDir', ...
            'baseOutDir is required.');
    end
    baseOutDir = char(string(baseOutDir));
    manifestFile = fullfile(baseOutDir,'af_gan_snapshots', ...
        'archive_snapshot_image_manifest.csv');
    if ~isfile(manifestFile)
        error('aggregate_CCMO_GAN_BDG_snapshot_metrics:MissingManifest', ...
            'Snapshot image manifest not found: %s',manifestFile);
    end

    Manifest = readtable(manifestFile,'TextType','string','Delimiter',',', ...
        'ReadVariableNames',true);
    Rows = repmat(emptySnapshotMetricRow(),height(Manifest),1);
    for i = 1 : height(Manifest)
        Rows(i) = snapshotMetricRow(Manifest(i,:));
    end
    Metrics = struct2table(Rows);
    outDir = fullfile(baseOutDir,'af_gan_snapshots');
    writetable(Metrics,fullfile(outDir,'snapshot_raw_gan_metrics.csv'));

    metricVars = {'AF_count','GAN_count','RawGANFeasibleRate', ...
        'GAN_to_AF_ObjNN_mean','GAN_to_AF_ObjNN_median', ...
        'GAN_to_AF_ObjNN_p90','GAN_to_AF_DecNN_mean', ...
        'GAN_to_AF_DecNN_median','GAN_to_AF_DecNN_p90', ...
        'AF_Cover_epsilon','GAN_ObjRangeRatioMean', ...
        'GAN_ObjSelfNN_median','GAN_ObjSelfNN_p10'};
    ByVariantFE = groupsummary(Metrics,{'variant','target_FE'}, ...
        'mean',metricVars);
    ByProblemFE = groupsummary(Metrics, ...
        {'variant','problem','target_FE'},'mean',metricVars);
    writetable(ByVariantFE,fullfile(outDir, ...
        'snapshot_raw_gan_metrics_by_variant_fe.csv'));
    writetable(ByProblemFE,fullfile(outDir, ...
        'snapshot_raw_gan_metrics_by_problem_fe.csv'));
end

function Row = snapshotMetricRow(ManifestRow)
    Row = emptySnapshotMetricRow();
    Row.variant = string(ManifestRow.variant);
    Row.problem = string(ManifestRow.problem);
    Row.run = double(ManifestRow.run);
    Row.target_FE = double(ManifestRow.target_FE);
    Row.actual_FE = double(ManifestRow.actual_FE);
    Row.gen = double(ManifestRow.gen);
    Row.snapshot_file = string(ManifestRow.snapshot_file);

    Snapshot = readtable(Row.snapshot_file,'TextType','string', ...
        'Delimiter',',','ReadVariableNames',true);
    role = string(Snapshot.archive_role);
    afMask = role == "AF_train";
    aiMask = role == "AI_train";
    ganMask = role == "GAN_generated";
    Row.AF_count = sum(afMask);
    Row.GAN_count = sum(ganMask);
    if Row.GAN_count <= 0 || Row.AF_count <= 0
        return;
    end

    objNames = string(Snapshot.Properties.VariableNames);
    objNames = objNames(startsWith(objNames,"obj"));
    decNames = string(Snapshot.Properties.VariableNames);
    decNames = decNames(startsWith(decNames,"dec"));
    Obj = tableMatrix(Snapshot,objNames);
    Dec = tableMatrix(Snapshot,decNames);

    ganFeasible = numericColumn(Snapshot,'intended_feasible');
    Row.RawGANFeasibleRate = mean(double(ganFeasible(ganMask) > 0));

    normBase = afMask | aiMask;
    if ~any(normBase)
        normBase = afMask;
    end
    zmin = min(Obj(normBase,:),[],1);
    zmax = max(Obj(normBase,:),[],1);
    afObj = normalizeByRange(Obj(afMask,:),zmin,zmax,false);
    ganObj = normalizeByRange(Obj(ganMask,:),zmin,zmax,false);
    objDist = min(pdist2(ganObj,afObj),[],2);
    Row.GAN_to_AF_ObjNN_mean = meanFinite(objDist);
    Row.GAN_to_AF_ObjNN_median = medianFinite(objDist);
    Row.GAN_to_AF_ObjNN_p90 = percentileFinite(objDist,90);

    coverDist = min(pdist2(afObj,ganObj),[],2);
    Row.AF_Cover_epsilon = mean(double(coverDist <= Row.AF_CoverEpsilon));

    afRange = max(afObj,[],1) - min(afObj,[],1);
    ganRange = max(ganObj,[],1) - min(ganObj,[],1);
    Row.GAN_ObjRangeRatioMean = meanFinite(ganRange ./ (afRange + 1e-12));

    if size(ganObj,1) > 1
        dSelf = pdist2(ganObj,ganObj);
        dSelf(1:size(dSelf,1)+1:end) = Inf;
        selfNN = min(dSelf,[],2);
        Row.GAN_ObjSelfNN_median = medianFinite(selfNN);
        Row.GAN_ObjSelfNN_p10 = percentileFinite(selfNN,10);
    end

    [lower,upper] = problemBounds(Row.problem,size(Dec,2));
    afDec = normalizeByRange(Dec(afMask,:),lower,upper,true);
    ganDec = normalizeByRange(Dec(ganMask,:),lower,upper,true);
    decDist = min(pdist2(ganDec,afDec),[],2);
    decDist = decDist ./ sqrt(max(1,size(ganDec,2)));
    Row.GAN_to_AF_DecNN_mean = meanFinite(decDist);
    Row.GAN_to_AF_DecNN_median = medianFinite(decDist);
    Row.GAN_to_AF_DecNN_p90 = percentileFinite(decDist,90);
end

function Row = emptySnapshotMetricRow()
    Row = struct( ...
        'variant',"", ...
        'problem',"", ...
        'run',NaN, ...
        'target_FE',NaN, ...
        'actual_FE',NaN, ...
        'gen',NaN, ...
        'AF_count',NaN, ...
        'GAN_count',NaN, ...
        'RawGANFeasibleRate',NaN, ...
        'GAN_to_AF_ObjNN_mean',NaN, ...
        'GAN_to_AF_ObjNN_median',NaN, ...
        'GAN_to_AF_ObjNN_p90',NaN, ...
        'GAN_to_AF_DecNN_mean',NaN, ...
        'GAN_to_AF_DecNN_median',NaN, ...
        'GAN_to_AF_DecNN_p90',NaN, ...
        'AF_Cover_epsilon',NaN, ...
        'AF_CoverEpsilon',0.05, ...
        'GAN_ObjRangeRatioMean',NaN, ...
        'GAN_ObjSelfNN_median',NaN, ...
        'GAN_ObjSelfNN_p10',NaN, ...
        'snapshot_file',"");
end

function M = tableMatrix(T,names)
    M = zeros(height(T),numel(names));
    for i = 1 : numel(names)
        M(:,i) = numericColumn(T,names(i));
    end
end

function x = numericColumn(T,name)
    x = T.(char(name));
    if isstring(x) || iscell(x)
        x = str2double(string(x));
    else
        x = double(x);
    end
end

function [lower,upper] = problemBounds(problemName,D)
    lower = zeros(1,D);
    upper = ones(1,D);
    try
        ctor = str2func(char(problemName));
        Problem = ctor('N',100,'maxFE',100000);
        lower = double(Problem.lower);
        upper = double(Problem.upper);
        lower = lower(1:min(D,numel(lower)));
        upper = upper(1:min(D,numel(upper)));
        if numel(lower) < D
            lower(1,D) = 0;
            upper(1,D) = 1;
        end
    catch
        lower = zeros(1,D);
        upper = ones(1,D);
    end
end

function X = normalizeByRange(X,lower,upper,clipValues)
    X = (X - lower) ./ (upper - lower + 1e-12);
    if clipValues
        X = min(max(X,0),1);
    end
end

function value = meanFinite(x)
    x = x(isfinite(x));
    if isempty(x)
        value = NaN;
    else
        value = mean(x);
    end
end

function value = medianFinite(x)
    x = x(isfinite(x));
    if isempty(x)
        value = NaN;
    else
        value = median(x);
    end
end

function value = percentileFinite(x,p)
    x = sort(x(isfinite(x)));
    if isempty(x)
        value = NaN;
        return;
    end
    p = min(max(double(p),0),100);
    value = x(max(1,ceil((p/100)*numel(x))));
end
