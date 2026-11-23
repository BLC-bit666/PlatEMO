classdef FiveStageCantileverBeam_BC < PROBLEM
% <2026> <multi> <real> <constrained>
% Five-stage cantilever: volume and deflection, with binary feasibility
%
% Source: 经典工程优化例题.pdf, section 34.2.6, pp. 9-11.
% x = [b1,...,b5,h1,...,h5], from the fixed end to the loaded end.
% Reproduce the PDF's code, including its 1/(3*E) deflection coefficient:
% delta = P*l^3/(3*E) * sum([244,148,76,28,4]./(b.*h.^3)).
% CalCon returns 0 iff every scaled residual is <= 1e-12, otherwise 1.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. PlatEMO research-use and citation terms apply.
%--------------------------------------------------------------------------

    methods
        %% Default settings of the problem
        function Setting(obj)
            obj.M        = 2;
            obj.D        = 10;
            obj.lower    = [1,2.4,2.4,1,1,30,45,45,30,30];
            obj.upper    = [5,3.1,3.1,5,5,60,60,60,60,60];
            obj.encoding = ones(1,obj.D);
        end
        %% Calculate both objectives, including for infeasible designs
        function PopObj = CalObj(~,X)
            b = X(:,1:5); h = X(:,6:10);
            volume = 100.*sum(b.*h,2);
            delta  = 50000*100^3/(3*2e7).*sum([244,148,76,28,4]./(b.*h.^3),2);
            PopObj = [volume,delta];
        end
        %% Calculate the single overall feasibility label
        function PopCon = CalCon(obj,X)
            b = X(:,1:5); h = X(:,6:10);
            stress = 6*50000*100.*[5,4,3,2,1]./(b.*h.^2);
            F = obj.CalObj(X);
            RawCon = [stress-14000, h./b-20, F(:,2)-2.7];
            % Fixed stress, aspect-ratio, and deflection acceptance scales.
            scale = [14000*ones(1,5),20*ones(1,5),2.7];
            PopCon = double(any(RawCon./scale>1e-12,2));
        end
        %% Conservative objective bounds for HV, not a Pareto front
        function R = GetOptimum(obj,~)
            F = obj.CalObj(obj.upper);
            R = [F(1),2.7];
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
