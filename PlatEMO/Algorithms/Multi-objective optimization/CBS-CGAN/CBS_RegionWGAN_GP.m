classdef CBS_RegionWGAN_GP < CBS_RegionWGAN_GP_Core
% <2026> <multi> <real> <constrained>
% Reference vector-conditioned boundary WGAN-GP
% All-reference 500-to-200 critic generation with local A/h/T utilization
% rawGuideCount    --- 500 --- Raw all-reference CGAN queries per event
% zDim             ---   6 --- Dimension of the generator noise vector
% ganIter          --- 100 --- Generator updates per training event
% ganMiniBatch     ---  32 --- Mini-batch size of WGAN-GP training
% nCritic          ---   4 --- Critic updates per generator update
% minGANTrainCount ---  32 --- Minimum conditioned rows required for training
% sampleSigma      --- 0.3 --- Standard deviation of generator sampling noise

%------------------------------- Reference --------------------------------
% [1] Y. Tian, T. Zhang, J. Xiao, X. Zhang, and Y. Jin. A coevolutionary
% framework for constrained multi-objective optimization problems. IEEE
% Transactions on Evolutionary Computation, 2021, 25(1): 102-116.
% [2] I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. Courville.
% Improved training of Wasserstein GANs. Advances in Neural Information
% Processing Systems, 2017, 30.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use PlatEMO for research.
%--------------------------------------------------------------------------

    methods
        function Algorithm = CBS_RegionWGAN_GP(varargin)
            Algorithm@CBS_RegionWGAN_GP_Core(varargin{:});
        end
    end
end
