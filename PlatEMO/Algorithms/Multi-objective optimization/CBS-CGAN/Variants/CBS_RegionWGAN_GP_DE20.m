classdef CBS_RegionWGAN_GP_DE20 < CBS_RegionWGAN_GP_Core
% <2026> <multi> <real> <constrained>
% Replace the front-half CGAN offspring quota with ordinary DE

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use PlatEMO for research.
%--------------------------------------------------------------------------

    methods
        function Algorithm = CBS_RegionWGAN_GP_DE20(varargin)
            Algorithm@CBS_RegionWGAN_GP_Core(varargin{:});
        end
    end

    methods(Access = protected)
        function Config = algorithmConfiguration(Algorithm)
        %ALGORITHMCONFIGURATION Replace only front-half CGAN offspring.
            Config = algorithmConfiguration@CBS_RegionWGAN_GP_Core(Algorithm);
            Config.experimentArm = 4;
            Config.guideGenerationMode = "traditional_de";
            Config.guideUseMode = "traditional_de";
        end
    end
end
