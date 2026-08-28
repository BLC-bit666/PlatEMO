classdef CBS_RegionWGAN_GP_E3_AllFeasible < CBS_RegionWGAN_GP_Screening
% <2026> <multi> <real> <constrained>
% Allow every true-feasible solution to compete for anchor capacity

    methods
        function Algorithm = CBS_RegionWGAN_GP_E3_AllFeasible(varargin)
            Algorithm@CBS_RegionWGAN_GP_Screening( ...
                'parameter',{0,5,Inf,5,0,0,0,0,1},varargin{:});
        end
    end
end
