classdef PressureVessel_BC < PROBLEM
% <2026> <multi> <real> <constrained>
% Pressure vessel: cost and capacity shortfall, with binary feasibility
%
% Source: 经典工程优化例题.pdf, section 34.2.2, pp. 3-4.
% x = [Ts,Th,R,L]. This is the PDF's continuous-thickness version; thickness
% is NOT rounded to multiples of 0.0625. All four inequalities are retained.
% Minimize [PDF cost,Vmax-V], where Vmax is fixed by the variable upper bounds.
% The fixed translation of -V keeps native HV normalization population-
% independent. It preserves the capacity objective's preference ordering.
% CalCon returns 0 iff all fixed scaled residuals are <= 1e-12, otherwise 1.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. PlatEMO research-use and citation terms apply.
%--------------------------------------------------------------------------

    methods
        %% Default settings of the problem
        function Setting(obj)
            obj.M        = 2;
            obj.D        = 4;
            obj.lower    = [0.0625,0.0625,10,10];
            obj.upper    = [99*0.0625,99*0.0625,200,200];
            obj.encoding = ones(1,obj.D);
        end
        %% Calculate both objectives, including for infeasible designs
        function PopObj = CalObj(obj,X)
            Ts = X(:,1); Th = X(:,2); R = X(:,3); L = X(:,4);
            cost = 0.6224.*Ts.*R.*L + 1.7781.*Th.*R.^2 + ...
                   3.1661.*Ts.^2.*L + 19.84.*Ts.^2.*R;
            shortfall = obj.Capacity(obj.upper)-obj.Capacity(X);
            PopObj = [cost,shortfall];
        end
        %% Calculate the single overall feasibility label
        function PopCon = CalCon(obj,X)
            RawCon = [-X(:,1)+0.0193.*X(:,3), ...
                      -X(:,2)+0.00954.*X(:,3), ...
                      1296000-obj.Capacity(X),X(:,4)-240];
            % Unit thickness scales, required capacity, and length limit.
            scale = [1,1,1296000,240];
            PopCon = double(any(RawCon./scale>1e-12,2));
        end
        %% Conservative objective bounds for native HV, not a Pareto front
        function R = GetOptimum(obj,~)
            F = obj.CalObj(obj.upper);
            R = [F(1),obj.Capacity(obj.upper)-1296000];
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
    methods(Access = private)
        function V = Capacity(~,X)
            % Shared arithmetic for the capacity objective and constraint.
            R = X(:,3); L = X(:,4);
            V = pi.*R.^2.*L + (4/3).*pi.*R.^3;
        end
    end
end
