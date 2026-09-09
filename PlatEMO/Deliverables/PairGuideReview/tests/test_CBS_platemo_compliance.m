function test_CBS_platemo_compliance()
%TEST_CBS_PLATEMO_COMPLIANCE Verify both public PlatEMO contracts.

    repoRoot = fileparts(which('platemo'));
    addCBSPaths(repoRoot);

    previousFile = which('CBS_RegionWGAN_GP');
    previousRoot = fileparts(previousFile);
    previousCoreFile = which('CBS_RegionWGAN_GP_Core');
    previousSource = string(fileread(previousFile));
    previousCoreSource = string(fileread(previousCoreFile));
    pairFile = which('PairGuide');
    pairRoot = fileparts(pairFile);
    pairCoreFile = which('PairGuideCore');
    pairSource = string(fileread(pairFile));
    pairCoreSource = string(fileread(pairCoreFile));

    %% Verify independent class metadata and parameters
    assert(~isempty(regexp(previousSource, ...
        'classdef\s+CBS_RegionWGAN_GP\s*<\s*CBS_RegionWGAN_GP_Core', ...
        'once')));
    assert(~isempty(regexp(pairSource, ...
        'classdef\s+PairGuide\s*<\s*PairGuideCore','once')));
    assert(contains(previousSource,'% <2026> <multi> <real> <constrained>'));
    assert(contains(pairSource,'% <2026> <multi> <real> <constrained>'));
    assert(contains(previousSource, ...
        '%------------------------------- Reference --------------------------------'));
    assert(contains(pairSource, ...
        '%------------------------------- Reference --------------------------------'));

    [previousNames,previousDefaultsText,previousDescriptions] = ...
        headerParameters(previousFile);
    [pairNames,pairDefaultsText,pairDescriptions] = ...
        headerParameters(pairFile);
    assert(isequal(previousNames, ...
        ["rawGuideCount","zDim","ganIter","ganMiniBatch", ...
        "nCritic","minGANTrainCount","sampleSigma"]));
    assert(isequal(pairNames, ...
        ["rawGuideCount","zDim","ganEpoch","ganMiniBatch", ...
        "nCritic","minGANTrainCount","sampleSigma"]));
    assert(all(strlength(previousDescriptions) > 0) && ...
        all(strlength(pairDescriptions) > 0));

    PreviousDefaults = CBS_RegionWGAN_GP.mainlineDefaults();
    PairDefaults = PairGuide.mainlineDefaults();
    assert(isequal(str2double(previousDefaultsText), ...
        [PreviousDefaults.rawGuideCount,PreviousDefaults.zDim, ...
        PreviousDefaults.ganIter,PreviousDefaults.ganMiniBatch, ...
        PreviousDefaults.nCritic,PreviousDefaults.minGANTrainCount, ...
        PreviousDefaults.sampleSigma]));
    assert(isequal(str2double(pairDefaultsText), ...
        [PairDefaults.rawGuideCount,PairDefaults.zDim, ...
        PairDefaults.pairGanEpoch,PairDefaults.ganMiniBatch, ...
        PairDefaults.nCritic,PairDefaults.minGANTrainCount, ...
        PairDefaults.sampleSigma]));
    assert(PreviousDefaults.ganStopFraction == 0.5 && ...
        isinf(PairDefaults.ganStopFraction) && ...
        contains(pairCoreSource,'Config.ganStopFraction = inf') && ...
        contains(pairCoreSource, ...
            'Config.guideGenerationMode = "pair_guide"'));

    assert(contains(previousCoreSource,'Problem.Initialization()') && ...
        contains(pairCoreSource,'Problem.Initialization()'));
    assert(contains(previousCoreSource,'Algorithm.ParameterSet(') && ...
        contains(pairCoreSource,'Algorithm.ParameterSet('));

    %% Verify each GUI entry by its own name
    PreviousEntries = guiEntryAlgorithms(previousRoot);
    PairEntries = guiEntryAlgorithms(pairRoot);
    assert(isequal(string(cellfun(@func2str,PreviousEntries, ...
        'UniformOutput',false)),"CBS_RegionWGAN_GP"));
    assert(isequal(string(cellfun(@func2str,PairEntries, ...
        'UniformOutput',false)),"PairGuide"));

    removed = ["CBS_RegionWGAN_GP_PairGuide", ...
        "CBS_RegionWGAN_GP_A1","CBS_RegionWGAN_GP_A2", ...
        "CBS_RegionWGAN_GP_E0_Base","CBS_RegionWGAN_GP_DE20", ...
        "CBS_RegionWGAN_GP_FullCGAN","CBS_RegionWGAN_GP_Experiment", ...
        "CBS_RegionWGAN_GP_Random20","CBS_RegionWGAN_GP_GA20", ...
        "CBS_RegionWGAN_GP_PairGuide_DE20", ...
        "CBS_RegionWGAN_GP_PairGuide_Quota30", ...
        "CBS_RegionWGAN_GP_PairGuide_Quota40", ...
        "CBS_RegionWGAN_GP_PairGuide_Quota50"];
    for name = removed
        assert(exist(char(name),'class') == 0);
    end

    Entries = [PreviousEntries,PairEntries];
    for a = 1 : numel(Entries)
        [decs,objs,cons] = platemo( ...
            'algorithm',Entries{a}, ...
            'problem',@DASCMOP1_BC, ...
            'N',10,'D',5,'maxFE',1, ...
            'save',0,'run',1, ...
            'outputFcn',@(varargin)[]);
        assert(size(decs,1) == 10 && size(decs,2) == 5);
        assert(size(objs,1) == 10 && size(cons,1) == 10);
    end
    addCBSPaths(repoRoot);

    %% Verify public parameter acceptance and normal termination
    Parameters = {5,4,0,8,2,1,0.1};
    Constructors = {@CBS_RegionWGAN_GP,@PairGuide};
    for a = 1 : numel(Constructors)
        Algorithm = Constructors{a}('parameter',Parameters,'save',0, ...
            'outputFcn',@(varargin)[]);
        assert(isa(Algorithm,'ALGORITHM') && ...
            isequal(Algorithm.parameter,Parameters));
        Problem = LIRCMOP5_BC('N',10,'D',5,'maxFE',120);
        Algorithm.Solve(Problem);
        assert(Algorithm.result{end,1} == Problem.maxFE);
    end

    %% Verify reuse of PlatEMO utilities
    utilityRoot = fullfile(repoRoot,'Algorithms','Utility functions');
    utilities = ["UniformPoint","TournamentSelection","OperatorDE"];
    combinedCore = previousCoreSource+newline+pairCoreSource;
    for i = 1 : numel(utilities)
        utilityFile = string(which(utilities(i)));
        assert(startsWith(utilityFile,string(utilityRoot) + filesep));
        assert(contains(combinedCore,utilities(i) + "("));
        assert(~isfile(fullfile(previousRoot,utilities(i) + ".m")) && ...
            ~isfile(fullfile(pairRoot,utilities(i) + ".m")));
    end

    fprintf('PairGuide and CBS PlatEMO compliance passed.\n');
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
