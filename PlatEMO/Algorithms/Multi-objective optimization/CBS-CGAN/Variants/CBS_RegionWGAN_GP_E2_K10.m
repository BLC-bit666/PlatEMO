classdef CBS_RegionWGAN_GP_E2_K10 < CBS_RegionWGAN_GP_Screening
% <2026> <multi> <real> <constrained>
% Pair each anchor within its ten nearest reference directions

    methods
        function Algorithm = CBS_RegionWGAN_GP_E2_K10(varargin)
            Algorithm@CBS_RegionWGAN_GP_Screening( ...
                'parameter',{0,10,2,5,0,0,0,0,1},varargin{:});
        end
    end
end
