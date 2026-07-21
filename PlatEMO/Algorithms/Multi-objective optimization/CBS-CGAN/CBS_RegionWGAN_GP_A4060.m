classdef CBS_RegionWGAN_GP_A4060 < CBS_RegionWGAN_GP
% <2026> <multi> <real> <constrained>
% Operator-ratio control of CBS_RegionWGAN_GP (A00-4060)
% Diagnostic companion of the A00 module-removal ablation. The boundary
% module stays fully disabled (no boundary memory, no training events, no
% calibration), but guideMode remains "on", so reproduction runs through
% the guided-mix path with a permanently empty guide buffer: the twenty
% guided slots fall back to plain DE every generation and the offspring
% mix is 40% SBX+PM plus 60% DE throughout. This is exactly the mainline
% composition whenever the mainline has no guides, whereas A00 uses the
% 50/50 backbone. Comparing this class against A00 isolates the pure
% operator-ratio effect that otherwise contaminates the module
% attribution in the second half of every mainline run.

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
        function Algorithm = CBS_RegionWGAN_GP_A4060(varargin)
        %CBS_REGIONWGAN_GP_A4060 Construct the operator-ratio control.
        %   The pinned switches are appended after the caller arguments,
        %   so they always win: module disabled exactly as in A00, except
        %   guideMode stays "on" to keep the 40/60 fallback composition.
            Defaults = CBS_RegionWGAN_GP.mainlineDefaults();
            Algorithm@CBS_RegionWGAN_GP(varargin{:}, ...
                'guideMode','on', ...
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
