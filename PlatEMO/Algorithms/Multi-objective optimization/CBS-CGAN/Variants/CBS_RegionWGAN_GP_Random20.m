classdef CBS_RegionWGAN_GP_Random20 < CBS_RegionWGAN_GP_Core
% <2026> <multi> <real> <constrained>
% Random-sampling ablation of the front-half CGAN offspring quota

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use PlatEMO for research.
%--------------------------------------------------------------------------

    methods
        function Algorithm = CBS_RegionWGAN_GP_Random20(varargin)
            Algorithm@CBS_RegionWGAN_GP_Core(varargin{:});
        end
    end

    methods(Access = protected)
        function Config = algorithmConfiguration(Algorithm)
        %ALGORITHMCONFIGURATION Replace only front-half CGAN offspring.
            Config = algorithmConfiguration@CBS_RegionWGAN_GP_Core(Algorithm);
            Config.experimentArm = 3;
            Config.guideGenerationMode = "random";
            Config.guideUseMode = "random";
        end
    end
end
