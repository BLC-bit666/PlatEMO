classdef CBS_RegionWGAN_GP_E1_KeepAnchor < CBS_RegionWGAN_GP_Screening
% <2026> <multi> <real> <constrained>
% Keep unpaired true-feasible anchors as positive training rows

    methods
        function Algorithm = CBS_RegionWGAN_GP_E1_KeepAnchor(varargin)
            Algorithm@CBS_RegionWGAN_GP_Screening( ...
                'parameter',{1,5,2,5,0,0,0,0,1},varargin{:});
        end
    end
end
