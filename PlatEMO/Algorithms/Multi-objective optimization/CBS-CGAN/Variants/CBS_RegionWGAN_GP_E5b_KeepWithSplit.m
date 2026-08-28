classdef CBS_RegionWGAN_GP_E5b_KeepWithSplit < CBS_RegionWGAN_GP_Screening
% <2026> <multi> <real> <constrained>
% Re-test unpaired anchors after the split class gate has been selected

    methods
        function Algorithm = CBS_RegionWGAN_GP_E5b_KeepWithSplit(varargin)
            Algorithm@CBS_RegionWGAN_GP_Screening( ...
                'parameter',{1,5,2,5,1,0,0,0,1},varargin{:});
        end
    end
end
