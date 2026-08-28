classdef CBS_RegionWGAN_GP_A2 < CBS_RegionWGAN_GP_Experiment
% <2026> <multi> <real> <constrained>
% A2 generation comparator: new generation with local-target utilization

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use PlatEMO for research.
%--------------------------------------------------------------------------

    methods
        function Algorithm = CBS_RegionWGAN_GP_A2(varargin)
            Algorithm@CBS_RegionWGAN_GP_Experiment( ...
                varargin{:},'parameter',{2});
        end
    end
end
