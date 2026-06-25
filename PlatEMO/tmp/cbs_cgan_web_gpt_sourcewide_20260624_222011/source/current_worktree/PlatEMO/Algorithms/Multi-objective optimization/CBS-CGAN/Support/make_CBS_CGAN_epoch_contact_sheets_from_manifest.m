function sheetDir = make_CBS_CGAN_epoch_contact_sheets_from_manifest(outDir,cleanupRaw)
%MAKE_CBS_CGAN_EPOCH_CONTACT_SHEETS_FROM_MANIFEST Stitch epoch sweep figures.

    if nargin < 2 || isempty(cleanupRaw)
        cleanupRaw = false;
    end
    manifestFile = fullfile(outDir,'train_quality_sweep_figure_manifest.csv');
    if ~isfile(manifestFile)
        error('Figure manifest not found: %s',manifestFile);
    end
    FigureManifest = readtable(manifestFile,'TextType','string');
    sheetDir = fullfile(outDir,'contact_sheets_epoch');
    if ~isfolder(sheetDir)
        mkdir(sheetDir);
    end

    problems = unique(string(FigureManifest.problem),'stable');
    settings = unique(string(FigureManifest.setting),'stable');
    targets = unique(double(FigureManifest.target_FE),'stable');
    tileH = 760;
    tileW = 1000;
    gap = 12;
    headerH = 90;
    rowLabelW = 180;
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
            for c = 1 : numel(settings)
                mask = string(FigureManifest.problem) == problems(p) & ...
                    string(FigureManifest.setting) == settings(c) & ...
                    double(FigureManifest.target_FE) == targets(r);
                files = string(FigureManifest.figure_file(mask));
                if isempty(files) || ~isfile(files(1))
                    tile = labelImage("missing",tileW,tileH,28);
                else
                    tile = imread(files(1));
                    tile = ensureRgb(tile);
                    tile = imresize(tile,[tileH tileW]);
                end
                x = rowLabelW + gap + (c-1)*(tileW+gap);
                canvas = pasteImage(canvas,tile,x,y);
            end
        end
        outFile = fullfile(sheetDir,sprintf('%s_epoch_contact_sheet.png', ...
            problems(p)));
        imwrite(canvas,outFile);
    end

    if cleanupRaw
        rawFiles = unique(string(FigureManifest.figure_file),'stable');
        for i = 1 : numel(rawFiles)
            if isfile(rawFiles(i))
                delete(rawFiles(i));
            end
        end
    end
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
