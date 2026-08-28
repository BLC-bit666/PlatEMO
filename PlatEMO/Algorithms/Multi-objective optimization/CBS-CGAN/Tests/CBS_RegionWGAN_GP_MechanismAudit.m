classdef CBS_RegionWGAN_GP_MechanismAudit < CBS_RegionWGAN_GP_Experiment
%CBS_REGIONWGAN_GP_MECHANISMAUDIT Test-only A0/A1/A2 oracle instrumentation.

    methods
        function Algorithm = CBS_RegionWGAN_GP_MechanismAudit(varargin)
            Algorithm@CBS_RegionWGAN_GP_Experiment(varargin{:});
        end
    end

    methods(Access = protected)
        function Config = algorithmConfiguration(Algorithm)
            Config = ...
                algorithmConfiguration@CBS_RegionWGAN_GP_Experiment(Algorithm);
            Config.mechanismAudit = true;
        end
    end
end
