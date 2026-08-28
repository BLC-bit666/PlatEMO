classdef CBS_RegionWGAN_GP_E8_GlobalCritic < CBS_RegionWGAN_GP_Screening
% <2026> <multi> <real> <constrained>
% Query all reference directions with 500 conditions and critic filtering

    methods
        function Algorithm = CBS_RegionWGAN_GP_E8_GlobalCritic(varargin)
            Algorithm@CBS_RegionWGAN_GP_Screening( ...
                'parameter',{0,5,2,5,0,0,0,1,1},varargin{:});
        end
    end
end
