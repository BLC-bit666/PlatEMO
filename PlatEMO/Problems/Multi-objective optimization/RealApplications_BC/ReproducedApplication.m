classdef (Abstract) ReproducedApplication < PROBLEM
%REPRODUCEDAPPLICATION Thin adapter to the pinned Python evaluators.
% Raw objectives and the single 0/1 label pass through without rescaling.
% GetOptimum supplies a frozen HV reference scale, not a Pareto front.

    properties(Access=private)
        Kind
        ReferenceBound
        Bridge = []
    end
    methods(Access=protected)
        function obj = ReproducedApplication(varargin)
            obj@PROBLEM(varargin{:});
        end
        function configure(obj,kind,dimension,upperBound)
            obj.Kind = kind;
            obj.M = 2;
            obj.D = dimension;
            obj.encoding = ones(1,dimension);
            obj.lower = zeros(1,dimension);
            obj.upper = upperBound*ones(1,dimension);
            support = fullfile(fileparts(mfilename('fullpath')),'Support');
            reference = jsondecode(fileread(fullfile(support,'reference_bounds.json')));
            obj.ReferenceBound = reference.(kind).HVNormalizationBound(:)';
        end
    end
    methods
        function Population = Evaluation(obj,varargin)
            X = obj.CalDec(varargin{1});
            Y = obj.evaluateRows(X);
            Population = SOLUTION(X,Y(:,1:2),Y(:,3),varargin{2:end});
            obj.FE = obj.FE + numel(Population);
        end
        function F = CalObj(obj,X)
            Y = obj.evaluateRows(X);
            F = Y(:,1:2);
        end
        function C = CalCon(obj,X)
            Y = obj.evaluateRows(X);
            C = Y(:,3);
        end
        function R = GetOptimum(obj,~)
            R = obj.ReferenceBound;
        end
        function score = CalMetric(obj,metName,Population)
            if strcmp(metName,'HV')
                score = CalMetric@PROBLEM(obj,metName,Population);
            else
                % Preserve native metrics; do not supply the HV bound as a PF.
                score = feval(metName,Population,[]);
            end
        end
        function n = EvaluatorCalls(obj)
            % Logical completed requests, including exact cache hits.
            n = 0;
            if ~isempty(obj.Bridge)
                n = double(obj.Bridge.calls);
            end
        end
        function stats = EvaluatorStatistics(obj)
            stats = struct('logical_calls',0);
            if ~isempty(obj.Bridge)
                module = py.importlib.import_module('json');
                stats = jsondecode(char(module.dumps(obj.Bridge.statistics())));
            end
        end
    end
    methods(Access=private)
        function Y = evaluateRows(obj,X)
            assert(size(X,2)==obj.D && all(isfinite(X),'all'), ...
                'RealApplications:BadDecision','Invalid decision matrix.');
            if isempty(obj.Bridge)
                problemDir = fileparts(mfilename('fullpath'));
                root = fileparts(fileparts(fileparts(problemDir)));
                artifact = fullfile(root,'outputs','real_world_exact_reproduction_20261119');
                python = fullfile(artifact,'.venv','bin','python');
                assert(isfile(python),'RealApplications:MissingRuntime', ...
                    'The pinned evaluator environment is missing: %s',python);
                env = pyenv;
                if env.Status == "NotLoaded"
                    setenv('DYLD_FALLBACK_LIBRARY_PATH','/opt/homebrew/lib');
                    pyenv('Version',python,'ExecutionMode','OutOfProcess');
                else
                    assert(strcmp(char(env.Executable),python), ...
                        'RealApplications:PythonMismatch','Use a fresh MATLAB session with the pinned Python runtime.');
                end
                support = fullfile(problemDir,'Support');
                sysModule = py.importlib.import_module('sys');
                pythonPath = sysModule.path;
                if int64(count(pythonPath,support))==0
                    insert(pythonPath,int32(0),support);
                end
                module = py.importlib.import_module('reproduced_evaluators');
                obj.Bridge = module.Evaluator(artifact,obj.Kind);
            end
            % The Python bridge already converts with numpy.asarray.
            Y = double(obj.Bridge.evaluate(X));
            assert(isequal(size(Y),[size(X,1),3]) && all(isfinite(Y),'all'), ...
                'RealApplications:BadResult','Evaluator must return finite raw objectives and one label per row.');
            assert(all(ismember(Y(:,3),[0,1])), ...
                'RealApplications:BadConstraint','Constraint must be a scalar 0/1 per solution.');
        end
    end
end
