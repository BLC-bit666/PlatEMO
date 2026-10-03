function [Entry,Seen] = GANCMOCoverage(W,Scale,P1Y,P2Y,archiveY,GY,newY,survived1,survived2,Seen)
% Observe true objective directions without influencing search or using a PF.
% Both archive endpoints are included, regardless of active training status.
    fixed = struct('minimum',zeros(1,size(W,2)),'span',ones(1,size(W,2)));
    Entry = struct('fixed',frame(W,fixed,P1Y,P2Y,archiveY,GY), ...
        'current',frame(W,Scale,P1Y,P2Y,archiveY,GY),'scale',Scale, ...
        'survivedP1',logical(survived1(:)),'survivedP2',logical(survived2(:)));
    assert(numel(survived1)==size(GY,1) && numel(survived2)==size(GY,1));
    if isempty(Seen)
        Seen = false(size(W,1),1);
        Seen(Entry.fixed.beforeCount>0) = true;
    end
    Entry.historyBefore = Seen;
    Entry.historicallyNew = ~Seen(double(Entry.fixed.ref));
    Seen(GANCMOReference(newY,W,fixed)) = true;
end

function E = frame(W,Scale,P1Y,P2Y,archiveY,GY)
    n = size(W,1);
    E = struct('p1Count',counts(P1Y,W,Scale,n),'p2Count',counts(P2Y,W,Scale,n), ...
        'archiveCount',counts(archiveY,W,Scale,n));
    E.beforeCount = E.p1Count+E.p2Count+E.archiveCount;
    E.ref = uint16(GANCMOReference(GY,W,Scale));
    E.currentlyMissing = E.beforeCount(double(E.ref))==0;
end

function C = counts(Y,W,Scale,n)
    refs = GANCMOReference(Y,W,Scale);
    C = accumarray(refs,ones(numel(refs),1),[n 1]);
end
