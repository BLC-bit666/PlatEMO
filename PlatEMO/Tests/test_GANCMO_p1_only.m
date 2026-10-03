function test_GANCMO_p1_only
% Actual CGAN offspring can survive in P1, but cannot enter P2 directly.
    root = fileparts(fileparts(mfilename('fullpath')));
    oldPath = path; oldRng = rng;
    cleanup = onCleanup(@()restoreState(oldPath,oldRng)); %#ok<NASGU>
    addpath(fullfile(root,'Algorithms'),fullfile(root,'Algorithms','Utility functions'), ...
        fullfile(root,'Algorithms','Multi-objective optimization','GANCMO'), ...
        fullfile(root,'Problems'));
    S = struct('initialG',1,'updateG',[1 1],'criticSteps',1, ...
        'generatorHidden',[],'criticHidden',4,'minimumPairs',1,'minimumPairsContinue',1, ...
        'candidateCount',12,'batchSize',4,'noiseSigma',.02);
    % The initial endpoints share one objective direction. The first G adds
    % objective spread and would survive in both pools under the old rule.
    % These tiny training budgets test routing, not boundary performance.
    % maxFE=49 ends with a one-child batch containing G and no DE1/DE2.
    for budget = [49 72]
        for record = [true false]
            rng(123,'twister');
            P = UserProblem('N',12,'D',2,'maxFE',budget,'lower',0,'upper',1,'once',true, ...
                'initFcn',@(n)[.1*ones(n,1),.25+.5*mod((1:n)',2)], ...
                'objFcn',{@(x)x(:,1),@(x)1-x(:,1)},'conFcn',@(x)x(:,2)-.5);
            A = GANCMO('save',-3,'outputFcn',@(varargin)[]);
            A.Settings = S; A.RecordTrace = record;
            A.Solve(P);
            assert(P.FE==budget && numel(A.result{end,2})==P.N);
            if record
                assert(sum(A.Trace.rows(:,5))>0,'The check requires evaluated G.');
                assert(sum(A.Trace.rows(:,7))>0,'G must still be able to survive in P1.');
                assert(all(A.Trace.rows(:,8)==0),'CGAN directly entered P2.');
                assert(all(cellfun(@(c)~any(c.survivedP2),A.Trace.coverage)));
                tracked = A.result; trackedRng = rng;
                fprintf('PASS P1-only G maxFE=%d actualG=%d survivedP1=%d survivedP2=0\n', ...
                    budget,sum(A.Trace.rows(:,5)),sum(A.Trace.rows(:,7)));
            else
                assert(isequal(rng,trackedRng),'Recording changed the search RNG.');
                assert(isequal(cell2mat(A.result(:,1)),cell2mat(tracked(:,1))));
                for k = 1:size(tracked,1)
                    assert(isequal(A.result{k,2}.decs,tracked{k,2}.decs));
                end
                fprintf('PASS recording does not change the trajectory maxFE=%d\n',budget);
            end
        end
    end
end

function restoreState(oldPath,oldRng)
    path(oldPath); rng(oldRng);
end
