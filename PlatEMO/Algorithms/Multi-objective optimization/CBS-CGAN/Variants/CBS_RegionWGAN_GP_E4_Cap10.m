classdef CBS_RegionWGAN_GP_E4_Cap10 < CBS_RegionWGAN_GP_Screening
% <2026> <multi> <real> <constrained>
% Retain at most ten feasible anchors per reference direction

    methods
        function Algorithm = CBS_RegionWGAN_GP_E4_Cap10(varargin)
            Algorithm@CBS_RegionWGAN_GP_Screening( ...
                'parameter',{0,5,2,10,0,0,0,0,1},varargin{:});
        end
    end
end
