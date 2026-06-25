function [outDir,sheetDir,Manifest] = ...
    run_CBS_CGAN_pair6_epoch50_compare_sheets(outDir)
%RUN_CBS_CGAN_PAIR6_EPOCH50_COMPARE_SHEETS Run pair=6 and compare to pair=3.
%   The pair=3 baseline is the existing D_tau_huber_default_epoch50 contact
%   sheet from the epoch sweep. This function runs only pair=6, then writes
%   one pair3-vs-pair6 contact sheet per LIRCMOP problem.

    rootDir = fileparts(which('platemo'));
    if isempty(rootDir)
        rootDir = pwd;
    end
    addpath(genpath(rootDir));
    if nargin < 1 || isempty(outDir)
        outDir = fullfile(rootDir,'Data','CBS_CGAN', ...
            ['train_quality_pair6_vs_pair3_D_default_epoch50_', ...
            char(datetime('now','Format','yyyyMMdd_HHmmss'))]);
    end

    pair3Root = fullfile(rootDir,'Data','CBS_CGAN', ...
        'train_quality_epoch_sweep_D_default_20260624_130858');
    pair3SheetDir = fullfile(pair3Root,'contact_sheets_epoch');
    if ~isfolder(pair3SheetDir)
        error('CBSPair6Compare:MissingPair3Sheets', ...
            'Pair=3 epoch contact sheet folder not found: %s', ...
            pair3SheetDir);
    end

    Options = struct( ...
        'workerCount',10, ...
        'problemNames',["LIRCMOP5_BC";"LIRCMOP6_BC"; ...
            "LIRCMOP7_BC";"LIRCMOP8_BC";"LIRCMOP9_BC"; ...
            "LIRCMOP10_BC"], ...
        'runIds',1, ...
        'N',100, ...
        'maxFE',100000, ...
        'targets',[10000 30000 50000 70000 100000], ...
        'plotRun',1, ...
        'visualDiagnostics',true, ...
        'plotDiagnosticTrends',false, ...
        'figureKinds',"train_reconstruction", ...
        'settings',makePair6Setting());

    [Manifest,~,~,~,~] = ...
        run_CBS_CGAN_train_quality_sweep_lir_online(outDir,Options);
    sheetDir = makePairCompareSheets(pair3SheetDir,outDir,true);
end

function S = makePair6Setting()
    S = struct( ...
        'setting',"D_tau_huber_default_epoch50_pair6", ...
        'variant',"D_tau_huber_default_epoch50_pair6", ...
        'conditionMode',"ref_tau", ...
        'advWeight',1, ...
        'reconstructionWeight',1, ...
        'nGen',30, ...
        'zDim',2, ...
        'ganIter',50, ...
        'ganMiniBatch',32, ...
        'trainMode',"epoch", ...
        'trainZMode',"zero", ...
        'sampleZMode',"zero", ...
        'maxCandidatePairsPerRef',6, ...
        'networkPreset',"default", ...
        'generatorHidden',[64 64 64], ...
        'discriminatorHidden',[64 64 32]);
end

function sheetDir = makePairCompareSheets(pair3SheetDir,outDir,cleanupRaw)
    if nargin < 3 || isempty(cleanupRaw)
        cleanupRaw = false;
    end
    manifestFile = fullfile(outDir,'train_quality_sweep_figure_manifest.csv');
    if ~isfile(manifestFile)
        error('CBSPair6Compare:MissingManifest', ...
            'Pair=6 figure manifest not found: %s',manifestFile);
    end
    FigureManifest = readtable(manifestFile,'TextType','string');
    sheetDir = fullfile(outDir,'contact_sheets_pair_compare');
    if ~isfolder(sheetDir)
        mkdir(sheetDir);
    end

    problems = unique(string(FigureManifest.problem),'stable');
    targets = unique(double(FigureManifest.target_FE),'stable');
    tileH = 760;
    tileW = 1000;
    gap = 12;
    headerH = 90;
    rowLabelW = 180;
    settings = ["pair=3 epoch50";"pair=6 epoch50"];
    Rows = repmat(emptySheetRow(),numel(problems),1);

    for p = 1 : numel(problems)
        canvasH = headerH + numel(targets)*tileH + ...
            (numel(targets)+1)*gap;
        canvasW = rowLabelW + numel(settings)*tileW + ...
            (numel(settings)+1)*gap;
        canvas = uint8(255*ones(canvasH,canvasW,3));
        for c = 1 : numel(settings)
            x = rowLabelW + gap + (c-1)*(tileW+gap);
            y = gap;
            canvas = pasteImage(canvas,labelImage(settings(c),tileW, ...
                headerH-gap,20),x,y);
        end
        for r = 1 : numel(targets)
            y = headerH + gap + (r-1)*(tileH+gap);
            canvas = pasteImage(canvas,labelImage("FE="+string(targets(r)), ...
                rowLabelW-gap,tileH,20),gap,y);

            pair3Tile = cropPair3Epoch50Tile(pair3SheetDir,problems(p), ...
                r,tileH,tileW,gap,headerH,rowLabelW);
            x = rowLabelW + gap;
            canvas = pasteImage(canvas,pair3Tile,x,y);

            pair6Tile = loadPair6Tile(FigureManifest,problems(p), ...
                targets(r),tileH,tileW);
            x = rowLabelW + gap + tileW + gap;
            canvas = pasteImage(canvas,pair6Tile,x,y);
        end
        outFile = fullfile(sheetDir,sprintf( ...
            '%s_pair3_vs_pair6_epoch50.png',problems(p)));
        imwrite(canvas,outFile);
        Rows(p).problem = problems(p);
        Rows(p).sheet_file = string(outFile);
        Rows(p).pair3_source_sheet = string(fullfile(pair3SheetDir, ...
            sprintf('%s_epoch_contact_sheet.png',problems(p))));
        Rows(p).pair6_manifest_file = string(manifestFile);
    end

    SheetManifest = struct2table(Rows);
    writetable(SheetManifest,fullfile(sheetDir, ...
        'pair_compare_sheet_manifest.csv'));

    if cleanupRaw
        rawFiles = unique(string(FigureManifest.figure_file),'stable');
        for i = 1 : numel(rawFiles)
            if isfile(rawFiles(i))
                delete(rawFiles(i));
            end
        end
    end
end

function tile = cropPair3Epoch50Tile(pair3SheetDir,problem,rowIndex, ...
        tileH,tileW,gap,headerH,rowLabelW)
    sheetFile = fullfile(pair3SheetDir,sprintf( ...
        '%s_epoch_contact_sheet.png',problem));
    if ~isfile(sheetFile)
        tile = labelImage("missing pair=3",tileW,tileH,28);
        return;
    end
    img = ensureRgb(imread(sheetFile));
    epoch50Col = 2;
    x = rowLabelW + gap + (epoch50Col-1)*(tileW+gap);
    y = headerH + gap + (rowIndex-1)*(tileH+gap);
    if y+tileH-1 > size(img,1) || x+tileW-1 > size(img,2)
        tile = labelImage("bad pair=3 crop",tileW,tileH,28);
    else
        tile = img(y:y+tileH-1,x:x+tileW-1,:);
    end
end

function tile = loadPair6Tile(FigureManifest,problem,targetFE,tileH,tileW)
    mask = string(FigureManifest.problem) == problem & ...
        double(FigureManifest.target_FE) == double(targetFE);
    files = string(FigureManifest.figure_file(mask));
    if isempty(files) || ~isfile(files(1))
        tile = labelImage("missing pair=6",tileW,tileH,28);
    else
        tile = ensureRgb(imread(files(1)));
        tile = imresize(tile,[tileH tileW]);
    end
end

function Row = emptySheetRow()
    Row = struct( ...
        'problem',"", ...
        'sheet_file',"", ...
        'pair3_source_sheet',"", ...
        'pair6_manifest_file',"");
end

function img = ensureRgb(img)
    if ndims(img) == 2
        img = repmat(img,1,1,3);
    elseif size(img,3) > 3
        img = img(:,:,1:3);
    end
    if ~isa(img,'uint8')
        img = im2uint8(img);
    end
end

function canvas = pasteImage(canvas,img,x,y)
    [h,w,~] = size(img);
    canvas(y:y+h-1,x:x+w-1,:) = img;
end

function img = labelImage(txt,w,h,fontSize)
    txt = char(txt);
    f = figure('Visible','off','Color','w','Position',[100 100 w h]);
    ax = axes(f,'Position',[0 0 1 1]);
    axis(ax,'off');
    text(ax,0.5,0.5,txt,'Interpreter','none', ...
        'HorizontalAlignment','center','VerticalAlignment','middle', ...
        'FontSize',fontSize,'FontWeight','bold');
    frame = getframe(f);
    img = frame2im(frame);
    close(f);
    img = ensureRgb(img);
    if size(img,1) ~= h || size(img,2) ~= w
        img = imresize(img,[h w]);
    end
end
