classdef CBS_RegionWGAN_GP_A1 < CBS_RegionWGAN_GP_Experiment
% <2026> <multi> <real> <constrained>
% Historical A1: new generation/screening with legacy utilization

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use PlatEMO for research.
%--------------------------------------------------------------------------

    methods
        function Algorithm = CBS_RegionWGAN_GP_A1(varargin)
            Algorithm@CBS_RegionWGAN_GP_Experiment( ...
                varargin{:},'parameter',{1});
        end
    end
end
