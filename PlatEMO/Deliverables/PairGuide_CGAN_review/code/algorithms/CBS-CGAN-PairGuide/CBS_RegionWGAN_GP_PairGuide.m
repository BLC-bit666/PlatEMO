classdef CBS_RegionWGAN_GP_PairGuide < CBS_RegionWGAN_GP_Core
% <2026> <multi> <real> <constrained>
% Atomic boundary-pair CGAN with infeasible-side differential guidance
% rawGuideCount    --- 500 --- Raw s=0 candidates per query event
% zDim             ---   6 --- Generator noise dimension
% ganEpoch         --- 500 --- Full pair epochs for initial training
% ganMiniBatch     ---  64 --- 32 complete pairs per mini-batch
% nCritic          ---   5 --- Critic updates per generator update
% minGANTrainCount ---  32 --- Kept for shared GUI parameter compatibility
% sampleSigma      ---   1 --- Production inference noise standard deviation

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use PlatEMO for research.
%--------------------------------------------------------------------------

    methods
        function Algorithm = CBS_RegionWGAN_GP_PairGuide(varargin)
            Algorithm@CBS_RegionWGAN_GP_Core(varargin{:});
        end
    end

    methods(Access = protected)
        function Config = algorithmConfiguration(Algorithm)
        %ALGORITHMCONFIGURATION Change only the locked comparator mechanism.
            Config = algorithmConfiguration@CBS_RegionWGAN_GP_Core(Algorithm);
            Config.experimentArm = 7;
            Config.guideGenerationMode = "pair_guide";
            Config.guideUseMode = "pair_guide";
            Config.guideOffspringShare = 0.20;
            Config.ganStopFraction = inf;
            Config.rawGuideCount = 500;
            Config.zDim = 6;
            % The inherited third positional parameter is PairGuide's
            % ganEpoch; production algorithms retain ganIter semantics.
            if numel(Algorithm.parameter) >= 3 && ...
                    ~isempty(Algorithm.parameter{3})
                pairGanEpoch = Config.ganIter;
            else
                pairGanEpoch = Config.pairGanEpoch;
            end
            Config.pairGanEpoch = max(0,round(double(pairGanEpoch)));
            Config.pairInitialEpoch = Config.pairGanEpoch;
            Config.pairRetrainEpoch = 10;
            Config.ganMiniBatch = 64;
            Config.nCritic = 5;
            Config.ganLrD = 1e-4;
            Config.ganLrG = 1e-4;
            Config.gpLambda = 10;
            Config.generatorHidden = [32 32];
            Config.criticHidden = [32 32];
            Config.sampleSigma = 1;
            Config.disableOracleAudit = true;
        end
    end
end
