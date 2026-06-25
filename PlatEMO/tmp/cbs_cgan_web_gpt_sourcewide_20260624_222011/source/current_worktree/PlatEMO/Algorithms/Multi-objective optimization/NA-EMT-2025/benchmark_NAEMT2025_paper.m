function Results = benchmark_NAEMT2025_paper(varargin)
% Benchmark script for NAEMT2025 under the paper's parameter settings.
%
% Optional name-value pairs:
%   'Runs'       : number of independent runs, default 30
%   'Population' : population size N, default 100
%   'MaxFE'      : maximum function evaluations, default 200000
%   'Alpha'      : MLP retraining threshold, default 0.9
%   'Epsilon'    : CDPPV threshold, default 0.5
%   'N1'         : training sample size, default 1000
%   'SavePath'   : MAT file path, default benchmark_NAEMT2025_paper.mat

    Params = struct( ...
        'Runs',30, ...
        'Population',100, ...
        'MaxFE',200000, ...
        'Alpha',0.9, ...
        'Epsilon',0.5, ...
        'N1',1000, ...
        'SavePath','benchmark_NAEMT2025_paper.mat');
    Params = ParseInputs(Params,varargin{:});

    ProblemNames = arrayfun(@(i)sprintf('DASCMOP%d_BC',i),1:9,'UniformOutput',false);
    Results      = repmat(struct('Problem','','Run',0,'IGD',NaN,'HV',NaN,'FeasibleRate',NaN,'Runtime',NaN),0,1);
    Row          = 0;

    for p = 1 : numel(ProblemNames)
        ProblemName = ProblemNames{p};
        fprintf('Testing %s\n',ProblemName);
        for r = 1 : Params.Runs
            rng(r,'twister');
            Problem   = feval(ProblemName,'N',Params.Population,'maxFE',Params.MaxFE);
            Algorithm = NAEMT2025('parameter',{Params.Alpha,Params.Epsilon,Params.N1},'save',0);
            Algorithm.Solve(Problem);

            Row = Row + 1;
            Results(Row).Problem      = ProblemName;
            Results(Row).Run          = r;
            Results(Row).IGD          = Algorithm.CalMetric('IGD');
            Results(Row).HV           = Algorithm.CalMetric('HV');
            Results(Row).FeasibleRate = Algorithm.CalMetric('Feasible_rate');
            Results(Row).Runtime      = Algorithm.CalMetric('runtime');

            fprintf('  Run %02d/%02d: IGD=%.6e, HV=%.6e, FeasibleRate=%.4f, Runtime=%.2fs\n', ...
                r,Params.Runs,Results(Row).IGD(end),Results(Row).HV(end), ...
                Results(Row).FeasibleRate(end),Results(Row).Runtime(end));
        end
    end

    save(Params.SavePath,'Results','Params');
end

function Params = ParseInputs(Params,varargin)
    if mod(numel(varargin),2) ~= 0
        error('benchmark_NAEMT2025_paper:InvalidInput','Inputs must be name-value pairs.');
    end
    for i = 1 : 2 : numel(varargin)
        Name = varargin{i};
        if ~isfield(Params,Name)
            error('benchmark_NAEMT2025_paper:UnknownOption','Unknown option: %s',Name);
        end
        Params.(Name) = varargin{i+1};
    end
end
