function test_CBS_platemo_compliance()
%TEST_CBS_PLATEMO_COMPLIANCE Verify the public PlatEMO algorithm contract.

    repoRoot = fileparts(fileparts(fileparts(fileparts( ...
        mfilename('fullpath')))));
    addpath(genpath(repoRoot));
    algorithmFile = which('CBS_RegionWGAN_GP');
    algorithmRoot = fileparts(algorithmFile);
    source = string(fileread(algorithmFile));

    %% Verify class metadata, references, and public parameters
    assert(~isempty(regexp(source, ...
        'classdef\s+CBS_RegionWGAN_GP\s*<\s*ALGORITHM','once')));
    assert(contains(source,'% <2026> <multi> <real> <constrained>'));
    assert(contains(source, ...
        '%------------------------------- Reference --------------------------------'));
    [names,defaultText,descriptions] = headerParameters(algorithmFile);
    expectedNames = ["nGen","zDim","ganIter","ganMiniBatch", ...
        "nCritic","minGANTrainCount","sampleSigma"];
    assert(isequal(names,expectedNames));
    assert(all(strlength(descriptions) > 0));

    Defaults = CBS_RegionWGAN_GP.mainlineDefaults();
    expectedDefaults = [Defaults.nGen,Defaults.zDim,Defaults.ganIter, ...
        Defaults.ganMiniBatch,Defaults.nCritic, ...
        Defaults.minGANTrainCount,Defaults.sampleSigma];
    assert(isequal(str2double(defaultText),expectedDefaults));
    assert(Defaults.ganStopFraction == 0.5);

    assert(contains(source,'Problem.Initialization()'));
    assert(contains(source,'Problem.Evaluation(ChildDecs)'));
    assert(contains(source,'Algorithm.NotTerminated(Population1)'));
    assert(contains(source,'Algorithm.ParameterSet('));

    %% Verify that all documented public parameters are accepted
    Parameters = {5,4,0,8,2,1,0.1};
    Algorithm = CBS_RegionWGAN_GP('parameter',Parameters,'save',0, ...
        'outputFcn',@(varargin)[]);
    assert(isa(Algorithm,'ALGORITHM') && ...
        isequal(Algorithm.parameter,Parameters));
    Problem = LIRCMOP5_BC('N',10,'D',5,'maxFE',120);
    Algorithm.Solve(Problem);
    assert(Algorithm.result{end,1} == Problem.maxFE);

    %% Verify reuse of PlatEMO utility functions after path configuration
    utilityRoot = fullfile(repoRoot,'Algorithms','Utility functions');
    utilities = ["UniformPoint","TournamentSelection","OperatorDE"];
    for i = 1 : numel(utilities)
        utilityFile = string(which(utilities(i)));
        assert(startsWith(utilityFile,string(utilityRoot) + filesep));
        assert(contains(source,utilities(i) + "("));
        assert(~isfile(fullfile(algorithmRoot,utilities(i) + ".m")));
    end

    fprintf('CBS PlatEMO compliance test passed.\n');
end

function [names,defaults,descriptions] = headerParameters(filePath)
%HEADERPARAMETERS Parse the parameter comments consumed by the PlatEMO GUI.

    lines = splitlines(string(fileread(filePath)));
    names = strings(1,0);
    defaults = strings(1,0);
    descriptions = strings(1,0);
    for i = 1 : numel(lines)
        token = regexp(lines(i), ...
            '^%\s*(\w+)\s*---\s*(.*?)\s*---\s*(.*?)\s*$', ...
            'tokens','once');
        if ~isempty(token)
            names(end+1) = string(token{1}); %#ok<AGROW>
            defaults(end+1) = string(token{2}); %#ok<AGROW>
            descriptions(end+1) = string(token{3}); %#ok<AGROW>
        end
    end
end
