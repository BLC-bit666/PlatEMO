function [Distance,Info] = ComputeOracleBoundaryDistance(ProblemName,Dec,SampleCount)
% Compute offline oracle boundary distance using the original continuous constraints.

    if nargin < 3 || isempty(SampleCount)
        SampleCount = 20000;
    end
    Dec = double(Dec);
    if isempty(Dec)
        Distance = zeros(0,1);
        Info = InitOracleInfo('',[],SampleCount);
        return;
    end

    persistent Cache
    if isempty(Cache)
        Cache = struct();
    end

    BaseProblem = regexprep(char(ProblemName),'_BC$','');
    Key = matlab.lang.makeValidName(BaseProblem);
    if ~isfield(Cache,Key)
        OracleProblem = feval(BaseProblem,'N',1,'maxFE',1);
        Scale = EstimateConstraintScale(OracleProblem,SampleCount,BaseProblem);
        Cache.(Key) = struct( ...
            'baseProblem',BaseProblem, ...
            'problem',OracleProblem, ...
            'scale',Scale, ...
            'sampleCount',SampleCount);
    end

    Context = Cache.(Key);
    RawCon = Context.problem.CalCon(Context.problem.CalDec(Dec));
    if isempty(RawCon)
        Distance = inf(size(Dec,1),1);
    else
        Scale = repmat(Context.scale,size(RawCon,1),1);
        Distance = min(abs(RawCon)./Scale,[],2);
    end
    Info = InitOracleInfo(Context.baseProblem,Context.scale,Context.sampleCount);
end

function Scale = EstimateConstraintScale(Problem,SampleCount,BaseProblem)
    State = rng;
    Cleaner = onCleanup(@() rng(State));
    rng(1000 + sum(double(BaseProblem)),'twister');
    Dec = rand(SampleCount,Problem.D);
    Dec = repmat(Problem.lower,SampleCount,1) + Dec.*repmat(Problem.upper-Problem.lower,SampleCount,1);
    RawCon = Problem.CalCon(Problem.CalDec(Dec));
    Scale = median(abs(RawCon),1);
    Scale(~isfinite(Scale) | Scale < 1e-12) = 1;
    Scale = reshape(Scale,1,[]);
end

function Info = InitOracleInfo(BaseProblem,Scale,SampleCount)
    Info = struct();
    Info.baseProblem = BaseProblem;
    Info.scale = Scale;
    Info.sampleCount = SampleCount;
end
