classdef CBS_RegionWGAN_GP_E5_SplitGate < CBS_RegionWGAN_GP_Screening
% <2026> <multi> <real> <constrained>
% Require 16 positive, 8 negative, and 4 reference conditions for training

    methods
        function Algorithm = CBS_RegionWGAN_GP_E5_SplitGate(varargin)
            Algorithm@CBS_RegionWGAN_GP_Screening( ...
                'parameter',{0,5,2,5,1,0,0,0,1},varargin{:});
        end
    end
end
