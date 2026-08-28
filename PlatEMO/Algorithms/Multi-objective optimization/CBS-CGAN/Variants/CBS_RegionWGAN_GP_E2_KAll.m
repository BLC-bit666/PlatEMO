classdef CBS_RegionWGAN_GP_E2_KAll < CBS_RegionWGAN_GP_Screening
% <2026> <multi> <real> <constrained>
% Pair each anchor without a reference-direction limit

    methods
        function Algorithm = CBS_RegionWGAN_GP_E2_KAll(varargin)
            Algorithm@CBS_RegionWGAN_GP_Screening( ...
                'parameter',{0,Inf,2,5,0,0,0,0,1},varargin{:});
        end
    end
end
