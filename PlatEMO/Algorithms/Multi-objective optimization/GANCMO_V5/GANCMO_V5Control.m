function [progress,quota] = GANCMO_V5Control(FE,initialFE,maxFE,n1)
% Progress alone controls P1's planned direct-CGAN share: 10% to 90%.
    if maxFE<=initialFE
        progress = 1; quota = 0; return;
    end
    progress = min(1,max(0,(FE-initialFE)/(maxFE-initialFE)));
    quota = min(n1,round(n1*(.1+.8*progress)));
end
