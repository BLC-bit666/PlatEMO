function result = PairGuideCost_RC(action,varargin)
%PAIRGUIDECOST_RC Scoped real-oracle accounting, including nested CalObj rows.
%   Times are inclusive: CalCon can contain CalObj. Never sum nested times.
    persistent Target Counts
    result = [];
    switch string(action)
        case "start"
            Target = varargin{1};
            Counts = struct('CalObjBatches',0,'CalObjRows',0,'CalObjSeconds',0, ...
                'CalConBatches',0,'CalConRows',0,'CalConSeconds',0, ...
                'timing',"inclusive; nested calls counted separately", ...
                'instrumented',ismember(string(class(Target)), ...
                ["LIRCMOP5_BC","LIRCMOP6_BC","LIRCMOP7_BC","LIRCMOP8_BC","LIRCMOP9_BC", ...
                "LIRCMOP10_BC","LIRCMOP12_BC"]));
        case "stop"
            Target = [];
        case "snapshot"
            result = Counts;
        case "enter"
            if isempty(Target) || ~isequal(Target,varargin{1}); return; end
            kind = char(varargin{2});
            Counts.([kind,'Batches']) = Counts.([kind,'Batches'])+1;
            Counts.([kind,'Rows']) = Counts.([kind,'Rows'])+varargin{3};
            timer = tic;
            result = onCleanup(@()PairGuideCost_RC('elapsed',kind,toc(timer)));
        case "elapsed"
            kind = char(varargin{1});
            Counts.([kind,'Seconds']) = Counts.([kind,'Seconds'])+varargin{2};
    end
end
