function flag = AnyDominates(PopObj,obj)
% Return true if any objective vector in PopObj Pareto-dominates obj.

    flag = false;
    if isempty(PopObj)
        return;
    end
    LessEqual = all(PopObj<=repmat(obj,size(PopObj,1),1),2);
    Less      = any(PopObj<repmat(obj,size(PopObj,1),1),2);
    flag      = any(LessEqual & Less);
end
