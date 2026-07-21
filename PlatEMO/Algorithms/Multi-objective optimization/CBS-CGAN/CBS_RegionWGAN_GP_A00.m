classdef CBS_RegionWGAN_GP_A00 < CBS_RegionWGAN_GP
% <2026> <multi> <real> <constrained>
% Module-removal ablation of CBS_RegionWGAN_GP (A00)
% The complete boundary module is disabled: no boundary-memory training,
% no conditional generation, no guided offspring, and no boundary
% calibration. The reproduction falls back to the plain 50% SBX+PM plus
% 50% DE backbone and every generation spends exactly 200 evaluations,
% so the saved budget flows into additional generations. Everything else
% (populations, environmental selection, seeds) is byte-identical to the
% mainline machinery.

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
        function Algorithm = CBS_RegionWGAN_GP_A00(varargin)
        %CBS_REGIONWGAN_GP_A00 Construct the module-removal ablation.
        %   The pinned switches are appended after the caller arguments,
        %   so they always win: guides off, calibration off, no protected
        %   or evaluated CGAN paths, and nGen = 0 (no training events).
            Defaults = CBS_RegionWGAN_GP.mainlineDefaults();
            Algorithm@CBS_RegionWGAN_GP(varargin{:}, ...
                'guideMode','off', ...
                'boundarySearch','off', ...
                'scoutMode','off', ...
                'generatorMode','wgan', ...
                'blsWindow','late', ...
                'blsFeed','off', ...
                'parameter',{0,Defaults.zDim,Defaults.ganIter, ...
                Defaults.ganMiniBatch,Defaults.nCritic, ...
                Defaults.minGANTrainCount,Defaults.sampleSigma});
        end
    end
end
