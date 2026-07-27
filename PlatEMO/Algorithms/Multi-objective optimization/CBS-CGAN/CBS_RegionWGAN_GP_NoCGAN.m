classdef CBS_RegionWGAN_GP_NoCGAN < CBS_RegionWGAN_GP
% <2026> <multi> <real> <constrained>
% No-CGAN random-slot ablation of CBS_RegionWGAN_GP
% The CGAN module is never initialized, trained, or queried. Before 50% of
% the evaluation budget, the first population uses 40% GA, 40% ordinary DE,
% and 20% uniformly initialized random solutions. From 50% to 100% of the
% budget, it follows the mainline's CGAN-inactive reproduction exactly:
% 40% GA and 60% ordinary DE. Boundary calibration, the second population,
% environmental selection, and all real-evaluation budgets remain unchanged.

%------------------------------- Reference --------------------------------
% Ablation companion of CBS_RegionWGAN_GP; see that class for references.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function Algorithm = CBS_RegionWGAN_GP_NoCGAN(varargin)
        %CBS_REGIONWGAN_GP_NOCGAN Construct the explicit no-CGAN ablation.
            Algorithm@CBS_RegionWGAN_GP(varargin{:});
        end
    end

    methods(Static)
        function enabled = randomSlotsAtFE(currentFE,ganFELimit)
        %RANDOMSLOTSATFE Use random slots strictly before the shared cutoff.
            enabled = double(currentFE) < double(ganFELimit);
        end
    end

    methods(Access = protected)
        function enabled = cganModuleEnabled(~)
        %CGANMODULEENABLED Disable every CGAN memory/train/sample operation.
            enabled = false;
        end

        function enabled = randomGuideSlotsEnabled( ...
                ~,currentFE,ganFELimit)
        %RANDOMGUIDESLOTSENABLED Replace early guide slots by random solutions.
            enabled = CBS_RegionWGAN_GP_NoCGAN.randomSlotsAtFE( ...
                currentFE,ganFELimit);
        end
    end
end
