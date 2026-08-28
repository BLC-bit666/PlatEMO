classdef CBS_RegionWGAN_GP_E4_CapAll < CBS_RegionWGAN_GP_Screening
% <2026> <multi> <real> <constrained>
% Retain feasible anchors without a per-reference capacity limit

    methods
        function Algorithm = CBS_RegionWGAN_GP_E4_CapAll(varargin)
            Algorithm@CBS_RegionWGAN_GP_Screening( ...
                'parameter',{0,5,2,Inf,0,0,0,0,1},varargin{:});
        end
    end
end
