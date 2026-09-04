classdef CBS_RegionWGAN_GP_PairGuide_Quota30 < CBS_RegionWGAN_GP_PairGuide
% <2026> <multi> <real> <constrained>
% PairGuide experiment using a 30% constrained-population offspring quota

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use PlatEMO for research.
%--------------------------------------------------------------------------

    methods(Access = protected)
        function Config = algorithmConfiguration(Algorithm)
            Config = algorithmConfiguration@CBS_RegionWGAN_GP_PairGuide( ...
                Algorithm);
            Config.experimentArm = 30;
            Config.guideOffspringShare = 0.30;
        end
    end
end
