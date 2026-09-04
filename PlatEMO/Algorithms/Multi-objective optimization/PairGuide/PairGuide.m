classdef PairGuide < PairGuideCore
% <2026> <multi> <real> <constrained>
% Atomic boundary-pair CGAN with rho-gated midpoint guidance
% rawGuideCount    --- 500 --- Raw s=0 candidates per query event
% zDim             ---   6 --- Generator noise dimension
% ganEpoch         --- 500 --- Full pair epochs for initial training
% ganMiniBatch     ---  64 --- 32 complete pairs per mini-batch
% nCritic          ---   5 --- Critic updates per generator update
% minGANTrainCount ---  32 --- Minimum active pairs required for training
% sampleSigma      ---   1 --- Production inference noise standard deviation

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
        function Algorithm = PairGuide(varargin)
            Algorithm@PairGuideCore(varargin{:});
        end
    end

end
