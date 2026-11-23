classdef WeldedBeam_BC < PROBLEM
% <2026> <multi> <real> <constrained>
% Welded beam: manufacturing cost and deflection, with binary feasibility
%
% Source: 经典工程优化例题.pdf, section 34.2.1, pp. 1-3.
% x = [h,l,t,b]. Use the seven constraints and bounds in the PDF's code.
% Deflection is also minimized. The buckling load follows the code, without
% the extra P printed in the PDF's displayed equation. See README.md.
% CalCon returns 0 iff every scaled residual is <= 1e-12, otherwise 1.

%------------------------------- Copyright --------------------------------
% Copyright (c) 2026 BIMK Group. PlatEMO research-use and citation terms apply.
%--------------------------------------------------------------------------

    methods
        %% Default settings of the problem
        function Setting(obj)
            obj.M        = 2;
            obj.D        = 4;
            obj.lower    = [0.1,0.1,0.1,0.1];
            obj.upper    = [2,10,10,2];
            obj.encoding = ones(1,obj.D);
        end
        %% Calculate both objectives, including for infeasible designs
        function PopObj = CalObj(~,X)
            h = X(:,1); l = X(:,2); t = X(:,3); b = X(:,4);
            cost  = 1.10471.*h.^2.*l + 0.04811.*t.*b.*(14+l);
            delta = 4*6000*14^3./(3e7.*t.^3.*b);
            PopObj = [cost,delta];
        end
        %% Calculate the single overall feasibility label
        function PopCon = CalCon(obj,X)
            h = X(:,1); l = X(:,2); t = X(:,3); b = X(:,4);
            P = 6000; L = 14; E = 3e7; G = 1.2e7;
            R = sqrt(l.^2/4 + ((h+t)/2).^2);
            M = P.*(L+l/2);
            J = 2.*(sqrt(2).*h.*l.*(l.^2/12 + ((h+t)/2).^2));
            tau1 = P./(sqrt(2).*h.*l);
            tau2 = M.*R./J;
            tau  = sqrt(tau1.^2 + 2.*tau1.*tau2.*l./(2.*R) + tau2.^2);
            sigma = 6*P*L./(t.^2.*b);
            Pc = 4.013*E.*sqrt(t.^2.*b.^6/36)/L^2.* ...
                 (1-t/(2*L)*sqrt(E/(4*G)));
            F = obj.CalObj(X);
            RawCon = [tau-13600, sigma-30000, h-b, ...
                0.10471.*h.^2 + 0.04811.*t.*b.*(14+l)-5, ...
                0.125-h, F(:,2)-0.25, P-Pc];
            % Fixed engineering scales: stress limits, common h/b bound,
            % cost limit, minimum h, deflection limit, and applied load.
            scale = [13600,30000,2,5,0.125,0.25,6000];
            PopCon = double(any(RawCon./scale>1e-12,2));
        end
        %% Conservative objective bounds for HV, not a Pareto front
        function R = GetOptimum(obj,~)
            F = obj.CalObj(obj.upper);
            R = [F(1),0.25];
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
