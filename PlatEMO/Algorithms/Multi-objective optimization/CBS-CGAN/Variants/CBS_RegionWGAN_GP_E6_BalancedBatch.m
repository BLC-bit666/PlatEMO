classdef CBS_RegionWGAN_GP_E6_BalancedBatch < CBS_RegionWGAN_GP_Screening
% <2026> <multi> <real> <constrained>
% Draw equal positive and negative pairflag rows in each WGAN mini-batch

    methods
        function Algorithm = CBS_RegionWGAN_GP_E6_BalancedBatch(varargin)
            Algorithm@CBS_RegionWGAN_GP_Screening( ...
                'parameter',{0,5,2,5,0,1,0,0,1},varargin{:});
        end
    end
end
