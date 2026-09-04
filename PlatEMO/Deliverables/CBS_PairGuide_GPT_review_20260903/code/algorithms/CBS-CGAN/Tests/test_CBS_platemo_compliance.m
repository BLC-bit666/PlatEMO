function test_CBS_platemo_compliance()
%TEST_CBS_PLATEMO_COMPLIANCE Verify the public PlatEMO algorithm contract.

    repoRoot = fileparts(which('platemo'));
    addCBSPaths(repoRoot);
    algorithmFile = which('CBS_RegionWGAN_GP');
    algorithmRoot = fileparts(algorithmFile);
    coreFile = which('CBS_RegionWGAN_GP_Core');
    source = string(fileread(algorithmFile));
    coreSource = string(fileread(coreFile));

    %% Verify class metadata, references, and public parameters
    assert(~isempty(regexp(source, ...
        'classdef\s+CBS_RegionWGAN_GP\s*<\s*CBS_RegionWGAN_GP_Core', ...
        'once')));
    assert(contains(source,'% <2026> <multi> <real> <constrained>'));
    assert(contains(source, ...
        '%------------------------------- Reference --------------------------------'));
    [names,defaultText,descriptions] = headerParameters(algorithmFile);
    expectedNames = ["rawGuideCount","zDim","ganIter","ganMiniBatch", ...
        "nCritic","minGANTrainCount","sampleSigma"];
    assert(isequal(names,expectedNames));
    assert(all(strlength(descriptions) > 0));

    Defaults = CBS_RegionWGAN_GP.mainlineDefaults();
    expectedDefaults = [Defaults.rawGuideCount,Defaults.zDim,Defaults.ganIter, ...
        Defaults.ganMiniBatch,Defaults.nCritic, ...
        Defaults.minGANTrainCount,Defaults.sampleSigma];
    assert(isequal(str2double(defaultText),expectedDefaults));
    assert(Defaults.ganStopFraction == 0.5);

    assert(contains(coreSource,'Problem.Initialization()'));
    assert(contains(coreSource,'Problem.Evaluation(ChildDecs)'));
    assert(contains(coreSource, ...
        'Algorithm.auditNotTerminated(Population1,Problem,W)'));
    assert(contains(coreSource,'Algorithm.ParameterSet('));

    %% Verify every GUI-discoverable CBS entry through platemo(...)
    EntryAlgorithms = guiEntryAlgorithms(algorithmRoot);
    entryNames = string(cellfun(@func2str,EntryAlgorithms, ...
        'UniformOutput',false));
    assert(isequal(entryNames,"CBS_RegionWGAN_GP"));
    assert(exist('CBS_RegionWGAN_GP_PairGuide','class') == 8);
    removed = ["CBS_RegionWGAN_GP_A1","CBS_RegionWGAN_GP_A2", ...
        "CBS_RegionWGAN_GP_E0_Base","CBS_RegionWGAN_GP_DE20", ...
        "CBS_RegionWGAN_GP_FullCGAN","CBS_RegionWGAN_GP_Experiment", ...
        "CBS_RegionWGAN_GP_Random20","CBS_RegionWGAN_GP_GA20"];
    for name = removed
        assert(exist(char(name),'class') == 0);
    end
    for a = 1 : numel(EntryAlgorithms)
        [decs,objs,cons] = platemo( ...
            'algorithm',EntryAlgorithms{a}, ...
            'problem',@DASCMOP1_BC, ...
            'N',10,'D',5,'maxFE',1, ...
            'save',0,'run',1, ...
            'outputFcn',@(varargin)[]);
        assert(size(decs,1) == 10 && size(decs,2) == 5);
        assert(size(objs,1) == 10 && size(cons,1) == 10);
    end
    addCBSPaths(repoRoot);
    assert(isempty(which('CBS_RegionWGAN_GP_BT0F0')) && ...
        isempty(which('CBS_RegionWGAN_GP_NoBT')), ...
        'PlatEMO path pollution must be removed after the contract test.');

    %% Verify that all documented public parameters are accepted
    Algorithms = {@CBS_RegionWGAN_GP};
    Parameters = {5,4,0,8,2,1,0.1};
    for a = 1 : numel(Algorithms)
        Algorithm = Algorithms{a}( ...
            'parameter',Parameters,'save',0, ...
            'outputFcn',@(varargin)[]);
        assert(isa(Algorithm,'ALGORITHM') && ...
            isequal(Algorithm.parameter,Parameters));
    end
    Algorithm = CBS_RegionWGAN_GP('parameter',Parameters,'save',0, ...
        'outputFcn',@(varargin)[]);
    Problem = LIRCMOP5_BC('N',10,'D',5,'maxFE',120);
    Algorithm.Solve(Problem);
    assert(Algorithm.result{end,1} == Problem.maxFE);

    %% Verify reuse of PlatEMO utility functions after path configuration
    utilityRoot = fullfile(repoRoot,'Algorithms','Utility functions');
    utilities = ["UniformPoint","TournamentSelection","OperatorDE"];
    for i = 1 : numel(utilities)
        utilityFile = string(which(utilities(i)));
        assert(startsWith(utilityFile,string(utilityRoot) + filesep));
        assert(contains(coreSource,utilities(i) + "("));
        assert(~isfile(fullfile(algorithmRoot,utilities(i) + ".m")));
    end

    fprintf('CBS PlatEMO compliance test passed.\n');
end

function EntryAlgorithms = guiEntryAlgorithms(algorithmRoot)
%GUIENTRYALGORITHMS Mirror GUI.readList2 metadata discovery for this folder.

    knownLabels = ["none","single","multi","many","real","integer", ...
        "label","binary","permutation","large","constrained", ...
        "expensive","multimodal","sparse","dynamic","multitask", ...
        "bilevel","robust"];
    files = dir(fullfile(algorithmRoot,'**','*.m'));
    names = strings(1,0);
    for i = 1 : numel(files)
        filePath = fullfile(files(i).folder,files(i).name);
        fid = fopen(filePath,'r');
        assert(fid >= 0,'Cannot read %s.',filePath);
        firstLine = string(fgetl(fid));
        secondLine = string(fgetl(fid));
        fclose(fid);
        labels = string(regexp(secondLine,'(?<=<).*?(?=>)','match'));
        visible = false;
        for label = labels
            visible = visible || any(ismember(split(label,'/'),knownLabels));
        end
        if visible
            token = regexp(firstLine,'^\s*classdef\s+(\w+)\s*<', ...
                'tokens','once');
            assert(~isempty(token),'GUI entry must be a class: %s.',filePath);
            [~,fileName] = fileparts(filePath);
            assert(strcmp(token{1},fileName), ...
                'GUI entry class and file names differ: %s.',filePath);
            names(end+1) = string(fileName); %#ok<AGROW>
        end
    end
    names = sort(names);
    EntryAlgorithms = arrayfun(@(name)str2func(char(name)),names, ...
        'UniformOutput',false);
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
