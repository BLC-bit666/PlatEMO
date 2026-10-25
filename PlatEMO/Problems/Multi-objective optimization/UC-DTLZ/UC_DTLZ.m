classdef UC_DTLZ < PROBLEM
% <2024> <multi/many> <real> <constrained>
% Li et al. UC1 and UC2-DTLZ1 with supplementary Eq. (40) binary signals
% ucType         --- 1 --- UC family (1, or 2 with DTLZ1)
% constraintType --- 1 --- C1, C2, or C3
% dtlz           --- 1 --- DTLZ1, DTLZ2, or DTLZ3
% referenceMode  --- 1 --- 1: unfiltered NBI paper comparison; 0: feasible PF
%
% DOI: 10.1109/TEVC.2024.3425629. Supplement Appendix A, Eqs. (1)--(10),(40).
% UC2-DTLZ2/3 and UC3--UC5 remain unavailable because their printed
% definitions conflict with the figures. See REPRODUCTION_NOTES.md.

    properties(SetAccess = private)
        ucType;
        constraintType;
        dtlz;
        referenceMode;
        objectiveQueries = 0;
        constraintQueries = 0;
    end
    methods
        function obj = UC_DTLZ(varargin)
            obj@PROBLEM(varargin{:});
            % Table I supplies defaults; explicitly supplied settings win.
            column = find([2,3,5,10]==obj.M,1);
            if ~isempty(column)
                variants = [1,1,1;1,2,1;1,2,3;1,3,1;1,3,2;1,3,3;2,1,1;2,2,1;2,3,1];
                budgets = [40000,45000,180000,270000;
                           40000,45000,180000,270000;
                           50000,60000,200000,300000;
                           40000,50000,200000,300000;
                           20000,25000,150000,225000;
                           50000,60000,200000,300000;
                           45000,50000,150000,225000;
                           40000,50000,150000,225000;
                           40000,50000,150000,225000];
                row = find(all(variants==[obj.ucType,obj.constraintType,obj.dtlz],2),1);
                supplied = find(cellfun(@ischar,varargin(1:end-1)) & ...
                    ~cellfun(@isempty,varargin(2:end)));
                if ~any(strcmp(varargin(supplied),'N'))
                    obj.N = 100+50*(obj.M>=5);
                end
                if ~any(strcmp(varargin(supplied),'maxFE'))
                    obj.maxFE = budgets(row,column);
                end
            end
        end
        function Setting(obj,ucType,constraintType,dtlz)
            % Named subclasses fix the variant; the original parameterized
            % UC_DTLZ entry remains available for existing callers.
            if nargin == 1
                [ucType,constraintType,dtlz,referenceMode] = obj.ParameterSet(1,1,1,1);
            else
                referenceMode = obj.ParameterSet(1);
            end
            obj.ucType = ucType;
            obj.constraintType = constraintType;
            obj.dtlz = dtlz;
            validateattributes(referenceMode,{'numeric'},{'scalar','real','finite','integer','>=',0,'<=',1});
            obj.referenceMode = referenceMode;
            valid = [1,1;2,1;2,3;3,1;3,2;3,3];
            assert(ismember([obj.constraintType,obj.dtlz],valid,'rows'), ...
                'UC_DTLZ:UnsupportedCombination','Invalid C/DTLZ combination.');
            if ~(obj.ucType == 1 || (obj.ucType == 2 && obj.dtlz == 1))
                error('UC_DTLZ:UnresolvedDefinition', ...
                    'Only UC1 and the three UC2-DTLZ1 problems have verified definitions.');
            end
            if isempty(obj.M); obj.M = 2; end
            if isempty(obj.D)
                if obj.dtlz == 1; obj.D = obj.M+4; else; obj.D = obj.M+9; end
            end
            validateattributes(obj.M,{'numeric'},{'scalar','integer','>=',2});
            validateattributes(obj.D,{'numeric'},{'scalar','integer','>=',obj.M});
            obj.lower = zeros(1,obj.D);
            obj.upper = ones(1,obj.D);
            obj.encoding = ones(1,obj.D);
        end
        function F = CalObj(obj,X)
            obj.objectiveQueries = obj.objectiveQueries + size(X,1);
            F = Objectives(obj,X);
        end
        function C = CalCon(obj,X)
            obj.constraintQueries = obj.constraintQueries + size(X,1);
            F = Objectives(obj,X);
            base = BaseConstraints(obj,F,X(:,1));
            if obj.ucType == 2
                % Eq. (10) multiplies every UC1 constraint. Thus sum(F)>60
                % makes both constraints of C3-DTLZ1 zero, not just one.
                farRegion = double(F(:,end)/60+sum(F(:,1:end-1)/60,2)-1<=0);
                base = base.*farRegion;
            end
            % Eq. (40): retain originals and append five masked copies.
            % Local convention: resample per query/row/copy, sharing each
            % scalar p_k across the base vector. Author RNG is unspecified.
            C = base;
            for k = 1:5
                C = [C,base.*double(rand(size(X,1),1)<0.5)]; %#ok<AGROW>
            end
        end
        function [R,info] = GetOptimum(obj,N,referenceMode)
            % Main paper IV-C specifies NBI. Point count, mapping and
            % feasibility filtering below are explicit local conventions;
            % the authors' original reference files are unavailable.
            if nargin<3; referenceMode = obj.referenceMode; end
            validateattributes(referenceMode,{'numeric'},{'scalar','real','finite','integer','>=',0,'<=',1});
            W = UniformPoint(N,obj.M,'NBI');
            if obj.dtlz == 1
                % UniformPoint clips zero weights to 1e-6. Restore the
                % simplex equation instead of returning points above it.
                R = W./sum(W,2)/2;
            else
                R = W./sqrt(sum(W.^2,2));
            end
            if referenceMode == 1
                % Adopted for numerical paper comparison on 2026-10-24.
                % Agreement with table values does not verify author files.
                info = struct('method','NBI','referenceMode','unfiltered', ...
                    'requestedPoints',N,'candidateCount',size(R,1), ...
                    'coveredPatches',NaN,'totalPatches',NaN,'available',true, ...
                    'reason','user_adopted_paper_comparison');
                return;
            end
            x1 = 1-R(:,end)./sum(R,2);
            R = R(all(BaseConstraints(obj,R,x1)==0,2),:);
            info = struct('method','NBI','referenceMode','feasible', ...
                'requestedPoints',N,'candidateCount',size(R,1), ...
                'coveredPatches',1,'totalPatches',1,'available',true,'reason','');
            if obj.constraintType == 3
                if obj.dtlz == 1
                    covered = numel(unique(round(3*(1-2*R(:,end)))));
                    total = 4;
                else
                    centres = [eye(obj.M);ones(1,obj.M)/sqrt(obj.M)];
                    covered = 0;
                    for k = 1:size(centres,1)
                        covered = covered+any(sum((R-centres(k,:)).^2,2)<Radius(obj.M)^2);
                    end
                    total = obj.M+1;
                end
                info.coveredPatches = covered;
                info.totalPatches = total;
                if covered < total
                    info.available = false;
                    info.reason = 'incomplete_reference_front';
                    % Do not publish a partial front as a metric reference.
                    % Empty (0-by-0) also protects direct metric calls.
                    R = [];
                    warning('UC_DTLZ:IncompleteReferenceFront', ...
                        ['UC%d-C%d-DTLZ%d, M=%d: the NBI reference covers %d/%d feasible patches. ', ...
                         'Reference-dependent metrics are unavailable (NaN); no partial reference is returned.'], ...
                        obj.ucType,obj.constraintType,obj.dtlz,obj.M,covered,total);
                end
            end
        end
        function R = GetPF(obj)
            % A plot of the feasible front is distinct from the adopted
            % paper-comparison reference, which may include infeasible points.
            if obj.M <= 3; R = obj.GetOptimum(10000,0); else; R = []; end
        end
    end
end

function F = Objectives(obj,X)
    tail = X(:,obj.M:end)-0.5;
    if obj.dtlz == 2
        g = sum(tail.^2,2);
    else
        g = 100*(obj.D-obj.M+1+sum(tail.^2-cos(20*pi*tail),2));
    end
    if obj.dtlz == 1
        F = 0.5*(1+g).*fliplr(cumprod([ones(size(X,1),1),X(:,1:obj.M-1)],2)) ...
            .*[ones(size(X,1),1),1-X(:,obj.M-1:-1:1)];
    else
        F = (1+g).*fliplr(cumprod([ones(size(X,1),1),cos(X(:,1:obj.M-1)*pi/2)],2)) ...
            .*[ones(size(X,1),1),sin(X(:,obj.M-1:-1:1)*pi/2)];
    end
end

function C = BaseConstraints(obj,F,x1)
    if obj.constraintType == 1
        C = double(1-F(:,end)/0.6-sum(F(:,1:end-1),2)/0.5<=0);
    elseif obj.dtlz == 1
        C = double(1-sum(F,2)/0.6<=0);
        if obj.constraintType == 3
            C = [C,double(cos(6*pi*x1)-0.95<=0)];
        end
    elseif obj.constraintType == 2
        C = double(1.1^2-sum(F.^2,2)<=0);
    else
        r = Radius(obj.M);
        % Keep Eq. (9)'s square differences: expanding them can flip the
        % strict binary indicator through cancellation near the boundary.
        nearAxis = -inf(size(F,1),1);
        for i = 1:obj.M
            other = [1:i-1,i+1:obj.M];
            nearAxis = max(nearAxis,r^2-(F(:,i)-1).^2-sum(F(:,other).^2,2));
        end
        nearCentre = r^2-sum((F-1/sqrt(obj.M)).^2,2);
        C = double(max(nearAxis,nearCentre)<=0);
    end
end

function r = Radius(M)
    if M == 2; r = 0.05; else; r = 0.1; end
end
