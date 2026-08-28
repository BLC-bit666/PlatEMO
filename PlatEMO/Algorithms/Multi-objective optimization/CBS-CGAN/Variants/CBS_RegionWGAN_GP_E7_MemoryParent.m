classdef CBS_RegionWGAN_GP_E7_MemoryParent < CBS_RegionWGAN_GP_Screening
% <2026> <multi> <real> <constrained>
% Supplement insufficient current feasible parents from boundary memory

    methods
        function Algorithm = CBS_RegionWGAN_GP_E7_MemoryParent(varargin)
            Algorithm@CBS_RegionWGAN_GP_Screening( ...
                'parameter',{0,5,2,5,0,0,1,0,1},varargin{:});
        end
    end
end
