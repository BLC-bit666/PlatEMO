function [hFeasible,hInfeasible] = draw_CBS_CGAN_objective_region( ...
        Ax,Problem,problemName,C,View)
%DRAW_CBS_CGAN_OBJECTIVE_REGION Draw attainable feasible/infeasible domains.

    if nargin < 5
        View = [];
    end
    problemName = string(problemName);
    if Problem.M == 2
        [hFeasible,hInfeasible] = draw2D( ...
            Ax,Problem,problemName,C,View);
    elseif Problem.M == 3
        [hFeasible,hInfeasible] = draw3D( ...
            Ax,Problem,problemName,C,View);
    else
        error('CBSRegionGAN:ObjectiveRegionDimension', ...
            'Only two- and three-objective backgrounds are supported.');
    end
end

function [hFeasible,hInfeasible] = draw2D( ...
        Ax,Problem,problemName,C,View)

    dynamicProblems = ["LIRCMOP5_BC","LIRCMOP7_BC", ...
        "LIRCMOP8_BC","LIRCMOP10_BC","LIRCMOP12_BC"];
    if ~isempty(View) && any(problemName == dynamicProblems)
        [X,Y] = meshgrid( ...
            linspace(View.lower(1),View.upper(1),400), ...
            linspace(View.lower(2),View.upper(2),400));
        feasible = objectiveFeasible2D(problemName,X,Y);
    else
        PF = Problem.GetPF();
        if ~iscell(PF) || numel(PF) < 3 || ...
                ~isequal(size(PF{1}),size(PF{2}),size(PF{3}))
            error('CBSRegionGAN:ObjectiveSpacePF2D', ...
                'The problem does not expose a 2-D feasible-region grid.');
        end
        X = double(PF{1});
        Y = double(PF{2});
        feasible = isfinite(PF{3});
    end
    attainable = attainable2D(problemName,X,Y);
    feasible = attainable & feasible;
    infeasible = attainable & ~feasible;
    region = nan(size(X));
    region(infeasible) = 0;
    region(feasible) = 1;
    H = imagesc(Ax,X(1,:),Y(:,1),region);
    H.AlphaData = 0.34*isfinite(region);
    set(Ax,'YDir','normal');
    colormap(Ax,[C.infeasible;C.feasible]);
    hFeasible = scatter(Ax,nan,nan,90,C.feasible,'s','filled');
    hInfeasible = scatter(Ax,nan,nan,90,C.infeasible,'s','filled');
end

function feasible = objectiveFeasible2D(problemName,X,Y)
%OBJECTIVEFEASIBLE2D Apply the benchmark objective-space constraints.

    Obj = [X(:),Y(:)];
    switch problemName
        case "LIRCMOP5_BC"
            violation = ellipseViolation(Obj,[1.6,2.5], ...
                [1.6,2.5],[2,2],[4,8]);
        case {"LIRCMOP7_BC","LIRCMOP8_BC"}
            violation = ellipseViolation(Obj,[1.2,2.25,3.5], ...
                [1.2,2.25,3.5],[2,2.5,2.5],[6,12,10]);
        case {"LIRCMOP10_BC","LIRCMOP12_BC"}
            theta = -0.25*pi;
            if problemName == "LIRCMOP10_BC"
                p = 1.1;
                q = 1.2;
                a = 2;
                b = 4;
                offset = 1;
            else
                p = 1.6;
                q = 1.6;
                a = 1.5;
                b = 6;
                offset = 2.5;
            end
            u = (Obj(:,1)-p)*cos(theta)-(Obj(:,2)-q)*sin(theta);
            v = (Obj(:,1)-p)*sin(theta)+(Obj(:,2)-q)*cos(theta);
            c1 = 0.1-u.^2/a^2-v.^2/b^2;
            alpha = 0.25*pi;
            c2 = offset-Obj(:,1)*sin(alpha)-Obj(:,2)*cos(alpha)+ ...
                sin(4*pi*(Obj(:,1)*cos(alpha)-Obj(:,2)*sin(alpha)));
            violation = c1 > 0 | c2 > 0;
        otherwise
            error('CBSRegionGAN:ObjectiveConstraint2D', ...
                'No objective-space constraint is defined for %s.', ...
                problemName);
    end
    feasible = reshape(~violation,size(X));
end

function violation = ellipseViolation(Obj,p,q,a,b)
    theta = -0.25*pi;
    violation = false(size(Obj,1),1);
    for constraint = 1 : numel(p)
        u = (Obj(:,1)-p(constraint))*cos(theta)- ...
            (Obj(:,2)-q(constraint))*sin(theta);
        v = (Obj(:,1)-p(constraint))*sin(theta)+ ...
            (Obj(:,2)-q(constraint))*cos(theta);
        value = 0.1-u.^2/a(constraint)^2-v.^2/b(constraint)^2;
        violation = violation | value > 0;
    end
end

function attainable = attainable2D(problemName,X,Y)
%ATTAINABLE2D Remove the objective-space area below the ideal manifold.

    switch problemName
        case {"LIRCMOP5_BC","LIRCMOP7_BC"}
            c = 0.7057;
            a = (X-c);
            b = (Y-c);
            attainable = a >= 0 & b >= 0 & ...
                sqrt(min(1,max(0,a)))+b >= 1;
        case "LIRCMOP8_BC"
            c = 0.7057;
            a = (X-c);
            b = (Y-c);
            attainable = a >= 0 & b >= 0 & ...
                min(1,max(0,a)).^2+b >= 1;
        case "LIRCMOP10_BC"
            c = 1.7057;
            a = X/c;
            b = Y/c;
            attainable = a >= 0 & b >= 0 & ...
                sqrt(min(1,max(0,a)))+b >= 1;
        case "LIRCMOP12_BC"
            c = 1.7057;
            a = X/c;
            b = Y/c;
            attainable = a >= 0 & b >= 0 & ...
                min(1,max(0,a)).^2+b >= 1;
        otherwise
            attainable = true(size(X));
    end
end

function [hFeasible,hInfeasible] = draw3D( ...
        Ax,Problem,problemName,C,View)

    if problemName == "DASCMOP9_BC"
        PF = Problem.GetPF();
        X = double(PF{1});
        Y = double(PF{2});
        feasibleZ = double(PF{3});
        fullZ = sqrt(max(0,1-(X-0.5).^2-(Y-0.5).^2))+0.5;
        infeasibleZ = fullZ;
        infeasibleZ(isfinite(feasibleZ)) = nan;
        surf(Ax,X,Y,infeasibleZ,'FaceColor',C.infeasible, ...
            'FaceAlpha',0.34,'EdgeColor','none');
        surf(Ax,X,Y,feasibleZ,'FaceColor',C.feasible, ...
            'FaceAlpha',0.46,'EdgeColor','none');
    elseif problemName == "LIRCMOP14_BC"
        [Obj,feasible] = lircmop14Background(Problem,View);
        scatter3(Ax,Obj(~feasible,1),Obj(~feasible,2),Obj(~feasible,3), ...
            10,C.infeasible,'filled','MarkerFaceAlpha',0.14, ...
            'MarkerEdgeColor','none');
        scatter3(Ax,Obj(feasible,1),Obj(feasible,2),Obj(feasible,3), ...
            10,C.feasible,'filled','MarkerFaceAlpha',0.18, ...
            'MarkerEdgeColor','none');
    else
        error('CBSRegionGAN:ObjectiveSpacePF3D', ...
            'No 3-D background renderer is defined for %s.',problemName);
    end
    hFeasible = scatter3(Ax,nan,nan,nan,90,C.feasible,'s','filled');
    hInfeasible = scatter3(Ax,nan,nan,nan,90,C.infeasible,'s','filled');
end

function [Obj,feasible] = lircmop14Background(Problem,View)
    angles = linspace(0,1,19);
    maxRadius = 3.4;
    if ~isempty(View)
        physicalMax = 1.7057+2.5*(Problem.D-2);
        maxRadius = min(physicalMax, ...
            max(maxRadius,norm(max(0,View.upper(1:3)))));
    end
    radii = unique([linspace(1.7057,3.4,18), ...
        linspace(3.4,maxRadius,12),1.75,1.8,1.9,2,3]);
    [X1,X2,R] = ndgrid(angles,angles,radii);
    rows = numel(X1);
    Decs = 0.5*ones(rows,Problem.D);
    Decs(:,1) = X1(:);
    Decs(:,2) = X2(:);
    increment = sqrt(max(0,R(:)-1.7057)/(10*(Problem.D-2)));
    Decs(:,3:end) = 0.5+repmat(increment,1,Problem.D-2);
    Population = Problem.Evaluation(Decs);
    Obj = double(Population.objs);
    feasible = sum(max(0,double(Population.cons)),2) <= 0;
end
