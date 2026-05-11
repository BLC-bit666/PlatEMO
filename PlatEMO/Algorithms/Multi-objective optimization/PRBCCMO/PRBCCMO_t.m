classdef PRBCCMO_t < ALGORITHM
% <2026> <multi> <real> <constrained>
% PRBCCMO_t
% Traced objective-boundary archive CCMO for binary unknown constraints
%
% hidden   --- 40   --- Hidden neurons of the boundary MLP
% epoch    --- 80   --- Cold-start training epochs
% lr       --- 0.05 --- Learning rate
% betaB    --- 4    --- Boundary pair budget multiplier
% etaB     --- 0.1  --- Maximum cross-pair global reshape ratio per generation
% Tretrain --- 20   --- Fixed MLP update period
% Gstart   --- 150  --- First generation allowed to train
% saveB    --- 0    --- Save boundary archive snapshots for plotting
%
% PRBCCMO_t mirrors PRBCCMO and writes core CSV diagnostics under Data/PRBCCMO_t.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    methods
        function main(Algorithm,Problem)
            Params = cell(1,8);
            [Params{:}] = Algorithm.ParameterSet(40,80,0.05,4,0.1,20,150,0);
            Params = cellfun(@double,Params);
            Observer = PRBCCMO_objective_core('initObserver',Algorithm,Problem,Params);
            Algorithm.metric.analysis_folder   = Observer.folder;
            Algorithm.metric.analysis_meta_csv = Observer.meta_file;
            Algorithm.metric.analysis_core_csv = Observer.core_file;
            if isfield(Observer,'boundary_file')
                Algorithm.metric.analysis_boundary_csv = Observer.boundary_file;
                Algorithm.metric.analysis_boundary_folder = Observer.boundary_folder;
                Algorithm.metric.analysis_boundary_manifest_csv = Observer.boundary_manifest_file;
            end
            NotTerminated = @(Population)Algorithm.NotTerminated(Population);
            PRBCCMO_objective_core('run',Algorithm,Problem,true,Params,NotTerminated,Observer);
        end
    end
end
