function [Archive,index] = EADMMTemporaryArchive(Offspring,Population1,Population2)
% Select main-task offspring that dominate members of both populations.

    Obj1 = Population1.objs;
    Obj2 = Population2.objs;
    keep = false(1,numel(Offspring));
    for i = 1 : numel(Offspring)
        obj        = Offspring(i).obj;
        dominates1 = any(all(obj<=Obj1,2) & any(obj<Obj1,2));
        dominates2 = any(all(obj<=Obj2,2) & any(obj<Obj2,2));
        keep(i)     = dominates1 && dominates2;
    end
    index   = find(keep);
    Archive = Offspring(keep);
end
