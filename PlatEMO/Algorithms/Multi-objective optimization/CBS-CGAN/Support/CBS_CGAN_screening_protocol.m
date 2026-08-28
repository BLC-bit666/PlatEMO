function Protocol = CBS_CGAN_screening_protocol(rootPath,nWorker)
%CBS_CGAN_SCREENING_PROTOCOL Frozen sequential-screening settings.

    if nargin < 1 || isempty(rootPath)
        rootPath = fileparts(which('platemo'));
    end
    if nargin < 2 || isempty(nWorker)
        nWorker = 10;
    end
    Protocol = struct();
    Protocol.rootPath = char(rootPath);
    Protocol.campaignName = 'CBS_CGAN_sequential_screening_runs1_5';
    Protocol.nWorker = double(nWorker);
    Protocol.popSize = 100;
    Protocol.maxFE = 2e5;
    Protocol.saveNum = 20;
    Protocol.runs = 1:5;
    Protocol.problems = ["DASCMOP1_BC","DASCMOP5_BC", ...
        "DASCMOP9_BC","LIRCMOP10_BC","LIRCMOP14_BC"];
    % keep, pair K, front depth, cap, gate, batch, parent, generation, audit
    Protocol.baseParameters = [0 5 2 5 0 0 0 0 1];
    Protocol.practicalWin = 0.98;
    Protocol.practicalLoss = 1.02;
    Protocol.frontTrigger = 0.20;
    Protocol.capTrigger = 0.10;
    Protocol.unsafeGateTrigger = 0.10;
    Protocol.imbalanceTrigger = 0.20;
    Protocol.parentFallbackTrigger = 0.10;
    Protocol.triggerProblemCount = 2;
end
