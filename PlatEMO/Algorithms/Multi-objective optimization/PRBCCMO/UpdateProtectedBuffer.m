function [ProtectedDec,ProtectedLabel] = UpdateProtectedBuffer(ProtectedDec,ProtectedLabel,NewDec,NewLabel,MaxProtected)
% Update the protected hard-case pocket used by the boundary learner.

    if nargin < 1 || isempty(ProtectedDec)
        ProtectedDec = [];
        ProtectedLabel = [];
    end
    if nargin < 5 || MaxProtected <= 0
        ProtectedDec = [];
        ProtectedLabel = [];
        return;
    end
    if isempty(NewDec)
        if size(ProtectedDec,1) > MaxProtected
            Keep = size(ProtectedDec,1)-MaxProtected+1 : size(ProtectedDec,1);
            ProtectedDec = ProtectedDec(Keep,:);
            ProtectedLabel = ProtectedLabel(Keep);
        end
        return;
    end

    AllDec = [ProtectedDec;NewDec];
    AllLab = [ProtectedLabel(:);double(NewLabel(:))];
    Keep = KeepLatestDecisionRows(AllDec);
    AllDec = AllDec(Keep,:);
    AllLab = AllLab(Keep);

    if size(AllDec,1) > MaxProtected
        Keep = size(AllDec,1)-MaxProtected+1 : size(AllDec,1);
        AllDec = AllDec(Keep,:);
        AllLab = AllLab(Keep);
    end

    ProtectedDec = AllDec;
    ProtectedLabel = AllLab;
end
