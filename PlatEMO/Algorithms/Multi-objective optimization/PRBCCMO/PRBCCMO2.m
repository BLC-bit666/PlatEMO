classdef PRBCCMO2 < ALGORITHM
% <2026> <multi> <real> <constrained>
% PRBCCMO
% Objective-boundary archive CCMO for binary unknown constraints
%
% hidden   --- 40   --- Hidden neurons of the boundary MLP
% epoch    --- 80   --- Cold-start training epochs
% lr       --- 0.05 --- Learning rate
% betaB    --- 4    --- Boundary pair budget multiplier
% etaB     --- 0.1  --- Maximum cross-pair global reshape ratio per generation
% Tretrain --- 20   --- Fixed MLP update period
% Gstart   --- 150  --- First generation allowed to train

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
            Params = cell(1,7);
            [Params{:}] = Algorithm.ParameterSet(40,80,0.05,4,0.1,20,150);
            Params = cellfun(@double,Params);
            NotTerminated = @(Population)Algorithm.NotTerminated(Population);
            PRBCCMO_objective_core('run',Algorithm,Problem,false,Params,NotTerminated,[]);
        end
    end
end
