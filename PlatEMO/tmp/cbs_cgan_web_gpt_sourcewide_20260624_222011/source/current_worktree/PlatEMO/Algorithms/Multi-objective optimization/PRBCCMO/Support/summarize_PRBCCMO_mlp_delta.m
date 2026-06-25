function Delta = summarize_PRBCCMO_mlp_delta(inputData,outFile)
% Build paired IGD deltas: IGD(PRBCCMO-noMLPcmp) - IGD(PRBCCMO-MLPcmp).

    if nargin < 2
        outFile = "";
    end

    if istable(inputData)
        T = inputData;
    else
        T = readtable(char(string(inputData)),'TextType','string');
    end

    PRBCCMOUtils.requireColumns(T,{'problem','run','igd'});
    if ismember('variant',T.Properties.VariableNames)
        Variant = string(T.variant);
    elseif ismember('algorithm',T.Properties.VariableNames)
        Variant = string(T.algorithm);
    else
        error('summarize_PRBCCMO_mlp_delta:MissingVariant', ...
            'Input table must contain variant or algorithm.');
    end

    Problem = string(T.problem);
    Run = double(T.run);
    IGD = double(T.igd);
    Keys = unique(table(Problem,Run),'rows','stable');

    OutProblem = strings(height(Keys),1);
    OutRun = zeros(height(Keys),1);
    NoMLPIGD = zeros(height(Keys),1);
    MLPIGD = zeros(height(Keys),1);
    row = 0;
    for i = 1 : height(Keys)
        Mask = Problem == Keys.Problem(i) & Run == Keys.Run(i);
        LocalVariant = lower(Variant(Mask));
        LocalIGD = IGD(Mask);
        NoMLPMask = contains(LocalVariant,"nomlpcmp") | contains(LocalVariant,"no-mlp-cmp") | ...
            contains(LocalVariant,"no_mlp_cmp") | contains(LocalVariant,"prbccmo-nomlp");
        MLPMask = contains(LocalVariant,"mlpcmp") & ~NoMLPMask;
        if ~any(NoMLPMask) || ~any(MLPMask)
            continue;
        end
        row = row + 1;
        OutProblem(row) = Keys.Problem(i);
        OutRun(row) = Keys.Run(i);
        NoMLPIGD(row) = PRBCCMOUtils.meanFinite(LocalIGD(NoMLPMask));
        MLPIGD(row) = PRBCCMOUtils.meanFinite(LocalIGD(MLPMask));
    end
    OutProblem = OutProblem(1:row);
    OutRun = OutRun(1:row);
    NoMLPIGD = NoMLPIGD(1:row);
    MLPIGD = MLPIGD(1:row);
    DeltaIGD = NoMLPIGD - MLPIGD;
    Result = classifyIgdDelta(DeltaIGD);
    Win = double(Result == "win");
    Loss = double(Result == "loss");
    Tie = double(Result == "tie");
    Missing = double(Result == "missing");

    Delta = table(OutProblem,OutRun,NoMLPIGD,MLPIGD,DeltaIGD,Result,Win,Loss,Tie,Missing, ...
        'VariableNames',{'problem','run','igd_no_mlp_cmp','igd_mlp_cmp','delta_igd_mlp', ...
        'mlp_igd_result','mlp_igd_win','mlp_igd_loss','mlp_igd_tie','mlp_igd_missing'});
    if strlength(string(outFile)) > 0
        writetable(Delta,char(string(outFile)));
    end
end

function Result = classifyIgdDelta(DeltaIGD)
    Tol = 1e-12;
    Result = strings(numel(DeltaIGD),1);
    Result(:) = "tie";
    Result(DeltaIGD > Tol) = "win";
    Result(DeltaIGD < -Tol) = "loss";
    Result(~isfinite(DeltaIGD)) = "missing";
end
