classdef CBS_RegionWGAN_GP_Experiment < CBS_RegionWGAN_GP_Core
%CBS_REGIONWGAN_GP_EXPERIMENT Frozen historical A0/A1/A2 configurations.
% Historical three-arm attribution for CGAN generation and utilization
% arm                 ---   2 --- 0: legacy/legacy, 1: new/legacy, 2: new/new
% ganIter             --- 100 --- Generator updates per training event
% minGANTrainCount    ---  32 --- Minimum conditioned rows for training
% rawGuideCount       --- 500 --- Raw all-reference queries in A1/A2

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference
% its original publication.
%--------------------------------------------------------------------------

    methods
        function Algorithm = CBS_RegionWGAN_GP_Experiment(varargin)
            Algorithm@CBS_RegionWGAN_GP_Core(varargin{:});
        end
    end

    methods(Access = protected)
        function Config = algorithmConfiguration(Algorithm)
        %ALGORITHMCONFIGURATION Select one frozen sequential-attribution arm.
            Config = CBS_RegionWGAN_GP_Core.mainlineDefaults();
            [arm,ganIter,minGANTrainCount,rawGuideCount] = ...
                Algorithm.ParameterSet(2,Config.ganIter, ...
                Config.minGANTrainCount,Config.rawGuideCount);
            arm = round(double(arm));
            if ~isscalar(arm) || ~isfinite(arm) || ~ismember(arm,0:2)
                error('CBSRegionGAN:BadExperimentArm', ...
                    'Experiment arm must be 0, 1, or 2.');
            end
            Config.ganIter = ganIter;
            Config.minGANTrainCount = minGANTrainCount;
            Config.rawGuideCount = rawGuideCount;
            Config.experimentArm = arm;
            if arm == 0
                Config.guideGenerationMode = "legacy";
                Config.guideUseMode = "legacy";
            elseif arm == 1
                Config.guideGenerationMode = "global_critic";
                Config.guideUseMode = "legacy";
            else
                Config.guideGenerationMode = "global_critic";
                Config.guideUseMode = "local_target";
            end
        end
    end
end
