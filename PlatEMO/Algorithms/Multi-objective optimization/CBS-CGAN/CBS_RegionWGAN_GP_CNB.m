classdef CBS_RegionWGAN_GP_CNB < CBS_RegionWGAN_GP
% <2026> <multi> <real> <constrained>
% Learning-necessity ablation of CBS_RegionWGAN_GP (CN-B)
% Byte-identical to the mainline except for one component: the trained
% conditional WGAN-GP guide generator is replaced by the trivial
% copy-noise sampler (a uniformly drawn boundary-memory anchor row of the
% queried reference plus sigma = 0.05 Gaussian noise in the scaled
% decision space). Boundary memory, queries, guided reproduction,
% boundary calibration, budgets, and selection are all unchanged, so any
% paired difference against the mainline is attributable to the learned
% generation alone.

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
        function Algorithm = CBS_RegionWGAN_GP_CNB(varargin)
        %CBS_REGIONWGAN_GP_CNB Construct the learning-necessity ablation.
        %   The pinned switch is appended after the caller arguments, so
        %   it always wins: only the guide generator changes.
            Algorithm@CBS_RegionWGAN_GP(varargin{:}, ...
                'generatorMode','copynoise');
        end
    end
end
