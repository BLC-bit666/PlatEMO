classdef TubularColumn_BC < PROBLEM
% <2026> <multi> <real> <constrained>
% Tubular column: cost and axial shortening, with binary feasibility
%
% Source: 经典工程优化例题.pdf, section 34.2.10, pp. 16-17.
% x = [d,t]: mean diameter and wall thickness (cm), both continuous.
% The PDF cost, bounds, yield and buckling constraints are retained.
% Nominal elastic shortening is added as the second objective, including
% for infeasible designs. CalCon returns one label, not violation values.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. PlatEMO research-use and citation terms apply.
%--------------------------------------------------------------------------

    methods
        %% Default settings of the problem
        function Setting(obj)
            obj.M        = 2;
            obj.D        = 2;
            obj.lower    = [2,0.2];
            obj.upper    = [14,0.8];
            obj.encoding = ones(1,obj.D);
        end
        %% Calculate both objectives, including for infeasible designs
        function PopObj = CalObj(~,X)
            d = X(:,1); t = X(:,2);
            cost = 9.8.*d.*t + 2.*d;
            shortening = 2500*250./(0.85e6.*pi.*d.*t);
            PopObj = [cost,shortening];
        end
        %% Calculate the single overall feasibility label
        function PopCon = CalCon(~,X)
            d = X(:,1); t = X(:,2);
            P = 2500; L = 250; E = 0.85e6; sigma = 500;
            % Both residuals are already dimensionless in the PDF.
            RawCon = [P./(pi.*d.*t.*sigma)-1, ...
                8*P*L^2./(pi^3*E.*d.*t.*(d.^2+t.^2))-1];
            PopCon = double(any(RawCon>1e-12,2));
        end
        %% Conservative objective bounds for native HV, not a Pareto front
        function R = GetOptimum(obj,~)
            F = obj.CalObj(obj.upper);
            % Yield feasibility bounds shortening by sigma*L/E.
            R = [F(1),500*250/0.85e6];
        end
        %% Do not treat the HV bound as a reference Pareto front
        function score = CalMetric(obj,metName,Population)
            if strcmp(metName,'HV')
                score = CalMetric@PROBLEM(obj,metName,Population);
            else
                score = feval(metName,Population,[]);
            end
        end
    end
end
