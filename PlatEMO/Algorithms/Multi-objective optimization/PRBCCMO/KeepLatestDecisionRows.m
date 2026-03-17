function Keep = KeepLatestDecisionRows(Dec)
% Return row indices that keep only the latest occurrence of each decision.

    if isempty(Dec)
        Keep = zeros(0,1);
        return;
    end

    [~,RevKeep] = unique(flipud(Dec),'rows','stable');
    Keep = sort(size(Dec,1)-RevKeep+1);
    Keep = Keep(:);
end
