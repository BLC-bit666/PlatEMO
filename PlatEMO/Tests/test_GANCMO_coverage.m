function test_GANCMO_coverage
% Real objective associations, full archive union, and historical novelty.
    root = fileparts(fileparts(mfilename('fullpath'))); oldPath = path;
    cleanup = onCleanup(@()path(oldPath));
    addpath(fullfile(root,'Algorithms','Multi-objective optimization','GANCMO'));
    W = [1 0;.5 .5;0 1]; S = struct('minimum',[0 0],'span',[1 1]);
    before = rng;
    [E,seen] = GANCMOCoverage(W,S,[3 0],zeros(0,2),[0 3],[1 1;0 4], ...
        [1 1;0 4],[true;false],[false;true],[]);
    assert(isequal(E.fixed.ref,uint16([2;3])));
    assert(isequal(E.fixed.currentlyMissing,[true;false]));
    assert(isequal(E.historicallyNew,[true;false]) && all(seen));
    assert(isequal(E.fixed.archiveCount,[0;0;1]));
    [E,seen] = GANCMOCoverage(W,S,[3 0],zeros(0,2),zeros(0,2),[1 1], ...
        [1 1],true,false,seen);
    assert(E.fixed.currentlyMissing && ~E.historicallyNew && all(seen));
    shifted = struct('minimum',[0 0],'span',[10 1]);
    E = GANCMOCoverage(W,shifted,[3 0],zeros(0,2),zeros(0,2),[1 1], ...
        [1 1],true,false,seen);
    assert(E.fixed.ref==2 && E.current.ref==3);
    E = GANCMOCoverage(W,S,[3 0],zeros(0,2),zeros(0,2),zeros(0,2), ...
        zeros(0,2),false(0,1),false(0,1),seen);
    assert(isempty(E.fixed.ref) && isempty(E.historicallyNew) && isequal(rng,before));
    fprintf('PASS true objective coverage, full archive, fixed/current scales, historical novelty, no RNG use\n');
end
