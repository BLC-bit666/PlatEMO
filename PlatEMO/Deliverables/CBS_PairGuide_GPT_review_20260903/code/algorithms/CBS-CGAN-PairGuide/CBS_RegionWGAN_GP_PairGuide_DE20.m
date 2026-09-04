classdef CBS_RegionWGAN_GP_PairGuide_DE20 < CBS_RegionWGAN_GP_PairGuide
% <2026> <multi> <real> <constrained>
% PairGuide ablation replacing its 20% CGAN-guided quota with ordinary DE

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use PlatEMO for research.
%--------------------------------------------------------------------------

    methods
        function Algorithm = CBS_RegionWGAN_GP_PairGuide_DE20(varargin)
            Algorithm@CBS_RegionWGAN_GP_PairGuide(varargin{:});
        end
    end

    methods(Access = protected)
        function Config = algorithmConfiguration(Algorithm)
        %ALGORITHMCONFIGURATION Replace only the PairGuide offspring quota.
            Config = algorithmConfiguration@CBS_RegionWGAN_GP_PairGuide( ...
                Algorithm);
            Config.experimentArm = 8;
            Config.guideGenerationMode = "traditional_de";
            Config.guideUseMode = "traditional_de";
        end
    end
end
