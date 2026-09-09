classdef PairGuide < PairGuideCore
% <2026> <multi> <real> <constrained>
% Real boundary-pair CGAN with direct native guidance
% rawGuideCount    --- 500 --- Native paired proposals per query event
% zDim             ---   6 --- Generator noise dimension
% ganUpdates       --- 200 --- Generator updates for initial training
% ganMiniBatch     ---  32 --- At most 16 complete pairs per mini-batch
% nCritic          ---   5 --- Critic updates per generator update
% minGANTrainCount ---   8 --- Minimum active pairs required for training
% sampleSigma      --- 0.3 --- Production inference noise standard deviation

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
