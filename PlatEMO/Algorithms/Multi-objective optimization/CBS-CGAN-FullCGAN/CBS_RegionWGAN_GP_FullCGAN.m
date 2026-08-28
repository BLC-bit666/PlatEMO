classdef CBS_RegionWGAN_GP_FullCGAN < CBS_RegionWGAN_GP_Core
% <2026> <multi> <real> <constrained>
% Full-run CGAN comparison without the late boundary-target handoff

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use PlatEMO for research.
%--------------------------------------------------------------------------

    methods
        function Algorithm = CBS_RegionWGAN_GP_FullCGAN(varargin)
            Algorithm@CBS_RegionWGAN_GP_Core(varargin{:});
        end
    end

    methods(Access = protected)
        function Config = algorithmConfiguration(Algorithm)
        %ALGORITHMCONFIGURATION Keep the existing CGAN path until maxFE.
            Config = algorithmConfiguration@CBS_RegionWGAN_GP_Core(Algorithm);
            Config.experimentArm = 6;
            Config.ganStopFraction = 1.0;
        end
    end
end
