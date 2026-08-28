classdef CBS_RegionWGAN_GP_E0_Base < CBS_RegionWGAN_GP_Screening
% <2026> <multi> <real> <constrained>
% Sequential-screening frozen production baseline
% keepUnpaired --- 0 --- Keep unpaired true-feasible anchors
% pairRefCount  --- 5 --- Reference directions allowed for pairing
% frontDepth    --- 2 --- Feasible Pareto fronts eligible as anchors
% maxPerRef     --- 5 --- Maximum anchors per reference direction
% gateMode      --- 0 --- 0: total rows, 1: split class gate
% batchMode     --- 0 --- 0: uniform, 1: pairflag-balanced
% parentMode    --- 0 --- 0: population, 1: memory fallback
% generationMode --- 0 --- 0: legacy pool, 1: global critic pool
% mechanismAudit --- 1 --- Oracle audit without FE or RNG changes

    methods
        function Algorithm = CBS_RegionWGAN_GP_E0_Base(varargin)
            Algorithm@CBS_RegionWGAN_GP_Screening( ...
                'parameter',{0,5,2,5,0,0,0,0,1},varargin{:});
        end
    end
end
