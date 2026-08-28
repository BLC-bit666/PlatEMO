classdef CBS_RegionWGAN_GP_Screening < CBS_RegionWGAN_GP_Core
%CBS_REGIONWGAN_GP_SCREENING Shared configuration for sequential experiments.

    methods
        function Algorithm = CBS_RegionWGAN_GP_Screening(varargin)
            Algorithm@CBS_RegionWGAN_GP_Core(varargin{:});
        end

        function Config = screeningConfigurationSnapshot(Algorithm)
        %SCREENINGCONFIGURATIONSNAPSHOT Return normalized experiment settings.
            Config = Algorithm.algorithmConfiguration();
        end
    end

    methods(Access = protected)
        function Config = algorithmConfiguration(Algorithm)
        %ALGORITHMCONFIGURATION Parse the nine screening factors.
            Config = CBS_RegionWGAN_GP_Core.mainlineDefaults();
            [keepUnpaired,pairRefCount,frontDepth,maxPerRef,gateMode, ...
                batchMode,parentMode,generationMode,mechanismAudit] = ...
                Algorithm.ParameterSet(0,5,2,5,0,0,0,0,1);

            Config.keepUnpairedAnchors = logicalFlag( ...
                keepUnpaired,'keepUnpaired');
            Config.pairNeighborRefCount = positiveIntegerOrInf( ...
                pairRefCount,'pairRefCount');
            Config.frontDepth = positiveIntegerOrInf( ...
                frontDepth,'frontDepth');
            Config.maxAnchorsPerRef = positiveIntegerOrInf( ...
                maxPerRef,'maxAnchorsPerRef');
            Config.trainGateMode = indexedMode(gateMode, ...
                ["total","split"],'trainGateMode');
            Config.batchSamplingMode = indexedMode(batchMode, ...
                ["uniform","pairflag_balanced"],'batchSamplingMode');
            Config.parentSourceMode = indexedMode(parentMode, ...
                ["population","memory_fallback"],'parentSourceMode');
            Config.guideGenerationMode = indexedMode(generationMode, ...
                ["legacy","global_critic"],'guideGenerationMode');
            Config.mechanismAudit = logicalFlag( ...
                mechanismAudit,'mechanismAudit');
            Config.guideUseMode = "local_target";
            Config.experimentArm = 100;
        end
    end
end

function value = logicalFlag(value,name)
    value = double(value);
    if ~isscalar(value) || ~isfinite(value) || ~ismember(value,[0 1])
        error('CBSRegionGAN:BadScreeningFlag', ...
            '%s must be either 0 or 1.',name);
    end
    value = logical(value);
end

function value = positiveIntegerOrInf(value,name)
    value = double(value);
    if ~isscalar(value) || isnan(value) || value < 1 || ...
            (isfinite(value) && value ~= round(value))
        error('CBSRegionGAN:BadScreeningCount', ...
            '%s must be a positive integer or Inf.',name);
    end
end

function value = indexedMode(index,modes,name)
    index = double(index);
    if ~isscalar(index) || ~isfinite(index) || index ~= round(index) || ...
            index < 0 || index >= numel(modes)
        error('CBSRegionGAN:BadScreeningMode', ...
            '%s index must be between 0 and %d.',name,numel(modes)-1);
    end
    value = modes(index+1);
end
