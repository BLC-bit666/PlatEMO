classdef PRBCCMO < ALGORITHM
% <2026> <multi> <real> <constrained>
% PRBCCMO
% Objective-boundary archive CCMO for binary unknown constraints
%
% hidden   --- 64    --- First hidden-layer width of the boundary MLP
% epoch    --- 200   --- Cold-start Adam training epochs
% lr       --- 1e-3  --- Cold-start Adam learning rate
% betaB    --- 3    --- Boundary pair budget multiplier
% etaB     --- 0.1  --- Maximum cross-pair global reshape ratio per generation
% Tretrain --- 10   --- Fixed MLP update period
% Gstart   --- 0    --- First generation allowed to train

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
            [Params{:}] = Algorithm.ParameterSet(64,200,1e-3,3,0.1,10,0);
            Params = cellfun(@double,Params);
            NotTerminated = @(Population)Algorithm.NotTerminated(Population);
            PRBCCMO_objective_core('run',Algorithm,Problem,false,Params,NotTerminated,[]);
        end
    end
end
