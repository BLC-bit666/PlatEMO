function [Algorithm,cleanup] = make_GANCMO_probe(varargin)
% Observe loop-boundary state in a temporary copy, never in production code.
    source=fileread(which('GANCMO'));
    source=strrep(source,'classdef GANCMO <','classdef GANCMOTestProbe <');
    source=strrep(source,'function Algorithm = GANCMO(', ...
        'function Algorithm = GANCMOTestProbe(');
    source=strrep(source,'    properties(SetAccess=private)', ...
        sprintf('    properties\n        Frames = {};\n    end\n    properties(SetAccess=private)'));
    anchor='while Algorithm.NotTerminated(P1)';
    assert(numel(strfind(source,anchor))==1,'GANCMO:ProbeAnchorChanged');
    source=strrep(source,anchor, ...
        ['while observe(Algorithm,P1,P2,Scale,W,Archive) && ', ...
        'Algorithm.NotTerminated(P1)']);
    source=[source,newline,sprintf([ ...
        'function yes = observe(A,P1,P2,Scale,W,Archive)\n', ...
        '    A.Frames{end+1}=struct(''scale'',Scale,''W'',W, ...\n', ...
        '        ''populationY'',[P1.objs;P2.objs],''archive'',Archive);\n', ...
        '    yes=true;\nend\n'])];
    previous=path; folder=tempname; mkdir(folder);
    cleanup=onCleanup(@()restore(previous,folder));
    file=fullfile(folder,'GANCMOTestProbe.m'); fid=fopen(file,'w');
    assert(fid>=0); fwrite(fid,source,'char'); fclose(fid);
    addpath(folder,'-begin');
    Algorithm=GANCMOTestProbe(varargin{:});
end

function restore(previous,folder)
    path(previous); rmdir(folder,'s');
end
