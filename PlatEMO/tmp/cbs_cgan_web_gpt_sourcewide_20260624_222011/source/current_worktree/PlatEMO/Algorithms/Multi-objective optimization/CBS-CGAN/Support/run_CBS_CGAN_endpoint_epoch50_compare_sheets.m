function [outDir,sheetDir,Manifest] = ...
    run_CBS_CGAN_endpoint_epoch50_compare_sheets(outDir)
%RUN_CBS_CGAN_ENDPOINT_EPOCH50_COMPARE_SHEETS Compare thin vs endpoint targets.
%   Runs only the feasible-endpoint target variant under the current
%   D_tau_huber_default_epoch50 settings, then stitches each problem beside
%   the existing default epoch50 contact-sheet column.

    rootDir = fileparts(which('platemo'));
    if isempty(rootDir)
        rootDir = pwd;
    end
    addpath(genpath(rootDir));
    if nargin < 1 || isempty(outDir)
        outDir = fullfile(rootDir,'Data','CBS_CGAN', ...
            ['train_quality_endpoint_vs_default_D_epoch50_', ...
            char(datetime('now','Format','yyyyMMdd_HHmmss'))]);
    end

    defaultRoot = fullfile(rootDir,'Data','CBS_CGAN', ...
        'train_quality_epoch_sweep_D_default_20260624_130858');
    defaultSheetDir = fullfile(defaultRoot,'contact_sheets_epoch');
    if ~isfolder(defaultSheetDir)
        error('CBSEndpointCompare:MissingDefaultSheets', ...
            'Default epoch contact sheet folder not found: %s', ...
            defaultSheetDir);
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
        'settings',makeEndpointSetting());

    [Manifest,~,~,~,~] = ...
        run_CBS_CGAN_train_quality_sweep_lir_online(outDir,Options);
    sheetDir = makeEndpointCompareSheets(defaultSheetDir,outDir,true);
end

function S = makeEndpointSetting()
    S = struct( ...
        'setting',"D_tau_huber_default_epoch50_endpoint", ...
        'variant',"D_tau_huber_default_epoch50_endpoint", ...
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
        'maxCandidatePairsPerRef',3, ...
        'boundaryTargetMode',"feasible_endpoint", ...
        'networkPreset',"default", ...
        'generatorHidden',[64 64 64], ...
        'discriminatorHidden',[64 64 32]);
end

function sheetDir = makeEndpointCompareSheets(defaultSheetDir,outDir,cleanupRaw)
    if nargin < 3 || isempty(cleanupRaw)
        cleanupRaw = false;
    end
    manifestFile = fullfile(outDir,'train_quality_sweep_figure_manifest.csv');
    if ~isfile(manifestFile)
        error('CBSEndpointCompare:MissingManifest', ...
            'Endpoint figure manifest not found: %s',manifestFile);
    end
    FigureManifest = readtable(manifestFile,'TextType','string');
    sheetDir = fullfile(outDir,'contact_sheets_endpoint_compare');
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
    settings = ["default thin";"feasible endpoint"];
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

            defaultTile = cropDefaultEpoch50Tile(defaultSheetDir, ...
                problems(p),r,tileH,tileW,gap,headerH,rowLabelW);
            x = rowLabelW + gap;
            canvas = pasteImage(canvas,defaultTile,x,y);

            endpointTile = loadEndpointTile(FigureManifest,problems(p), ...
                targets(r),tileH,tileW);
            x = rowLabelW + gap + tileW + gap;
            canvas = pasteImage(canvas,endpointTile,x,y);
        end
        outFile = fullfile(sheetDir,sprintf( ...
            '%s_default_vs_endpoint_epoch50.png',problems(p)));
        imwrite(canvas,outFile);
        Rows(p).problem = problems(p);
        Rows(p).sheet_file = string(outFile);
        Rows(p).default_source_sheet = string(fullfile(defaultSheetDir, ...
            sprintf('%s_epoch_contact_sheet.png',problems(p))));
        Rows(p).endpoint_manifest_file = string(manifestFile);
    end

    SheetManifest = struct2table(Rows);
    writetable(SheetManifest,fullfile(sheetDir, ...
        'endpoint_compare_sheet_manifest.csv'));

    if cleanupRaw
        rawFiles = unique(string(FigureManifest.figure_file),'stable');
        for i = 1 : numel(rawFiles)
            if isfile(rawFiles(i))
                delete(rawFiles(i));
            end
        end
    end
end

function tile = cropDefaultEpoch50Tile(defaultSheetDir,problem,rowIndex, ...
        tileH,tileW,gap,headerH,rowLabelW)
    sheetFile = fullfile(defaultSheetDir,sprintf( ...
        '%s_epoch_contact_sheet.png',problem));
    if ~isfile(sheetFile)
        tile = labelImage("missing default",tileW,tileH,28);
        return;
    end
    img = ensureRgb(imread(sheetFile));
    epoch50Col = 2;
    x = rowLabelW + gap + (epoch50Col-1)*(tileW+gap);
    y = headerH + gap + (rowIndex-1)*(tileH+gap);
    if y+tileH-1 > size(img,1) || x+tileW-1 > size(img,2)
        tile = labelImage("bad default crop",tileW,tileH,28);
    else
        tile = img(y:y+tileH-1,x:x+tileW-1,:);
    end
end

function tile = loadEndpointTile(FigureManifest,problem,targetFE,tileH,tileW)
    mask = string(FigureManifest.problem) == problem & ...
        double(FigureManifest.target_FE) == double(targetFE);
    files = string(FigureManifest.figure_file(mask));
    if isempty(files) || ~isfile(files(1))
        tile = labelImage("missing endpoint",tileW,tileH,28);
    else
        tile = ensureRgb(imread(files(1)));
        tile = imresize(tile,[tileH tileW]);
    end
end

function Row = emptySheetRow()
    Row = struct( ...
        'problem',"", ...
        'sheet_file',"", ...
        'default_source_sheet',"", ...
        'endpoint_manifest_file',"");
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
