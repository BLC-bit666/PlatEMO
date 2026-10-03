function [f,fit_obj_c,cv] = fit_cv(P,problem)

switch problem       

    case 1
        n = 30;
        sum_t = 0;

        for i = 3:n
            %temp = P(:,i).^2;
            temp = 100*(P(:,i) - P(:,i-1).^2).^2 + (1 - P(:,i-1)).^2;
            sum_t = sum_t+temp;
        end
        
        g = sum_t;
        f(:,1) = P(:,1);
        f(:,2) = g.*(1 - sqrt(f(:,1))./g)+2;
        % Constrains   
        thita=-0.2*pi; 
        a=0.2;  
        b=10; 
        cc=1; 
        d=0.5; 
        e=1;

        c(:,1)=-(cos(thita)*(f(:,2)-e)-sin(thita)*f(:,1)-a*(abs(sin(b*pi*(sin(thita)*(f(:,2)-e)+cos(thita)*f(:,1)).^cc))).^d);
        c(:,1) = max(0, c(:,1));
        c(:,2) = g.*(1 - sqrt(f(:,1))./g)+0.73*f(:,1) -2;%%%%
        c(:,2) = max(0, c(:,2));
        
        c_max = max(c, [], 1);
        for i = 1:size(c,2)
            if c_max(i) > 0
                c(:,i) = c(:,i) ./ c_max(i);
            end
        end
        cv = sum(c,2)/2;
        %cv = zeros(size(P,1),1);
        fit_obj_c=[f cv];           
    case 2
        n = 30;
        sum_t = 0;

        for i = 3:n
            %temp = P(:,i).^2;
            temp = 100*(P(:,i) - P(:,i-1).^2).^2 + (1 - P(:,i-1)).^2;
            sum_t = sum_t+temp;
        end
        
        g = sum_t;
        f(:,1) = P(:,1);
        f(:,2) = g.*(1 - sqrt(f(:,1))./g)+2.5;
        % Constrains   
        thita=-0.2*pi; 
        a=0.75;  
        b=10; 
        cc=1; 
        d=0.5; 
        e=1;
   
        c(:,1)=-(cos(thita)*(f(:,2)-e)-sin(thita)*f(:,1)-a*(abs(sin(b*pi*(sin(thita)*(f(:,2)-e)+cos(thita)*f(:,1)).^cc))).^d);
        c(:,1) = max(0, c(:,1));
        c(:,2) = g.*(1 - sqrt(f(:,1))./g)+0.73*f(:,1) -1.5;%%%%
        c(:,2) = max(0, c(:,2));
        
        c_max = max(c, [], 1);
        for i = 1:size(c,2)
            if c_max(i) > 0
                c(:,i) = c(:,i) ./ c_max(i);
            end
        end
        cv = sum(c,2)/2;
        %cv = zeros(size(P,1),1);
        fit_obj_c=[f cv];                
    case 3
        n = 30;
        sum_t = 0;

        for i = 3:n
            %temp = P(:,i).^2;
            temp = 100*(P(:,i) - P(:,i-1).^2).^2 + (1 - P(:,i-1)).^2;
            sum_t = sum_t+temp;
        end
        
        g = sum_t;
        f(:,1) = P(:,1);
        f(:,2) = g.*(1 - sqrt(f(:,1))./g)+4;
        % Constrains   
        thita=-0.2*pi; 
        a=2;  
        b=10; 
        cc=1; 
        d=6; 
        e=1;
        
        c(:,1)=-(cos(thita)*(f(:,2)-e)-sin(thita)*f(:,1)-a*(abs(sin(b*pi*(sin(thita)*(f(:,2)-e)+cos(thita)*f(:,1)).^cc))).^d);
        c(:,1) = max(0, c(:,1));
        c(:,2) = g.*(1 - sqrt(f(:,1))./g)+0.73*f(:,1)-2;%%%%
        c(:,2) = max(0, c(:,2));
        
        c_max = max(c, [], 1);
        for i = 1:size(c,2)
            if c_max(i) > 0
                c(:,i) = c(:,i) ./ c_max(i);
            end
        end
        cv = sum(c,2)/2;
        %cv = zeros(size(P,1),1);
        fit_obj_c=[f cv];
        
        %！！！！！！寄辛佩囃！！！！！！！！
    case 4
        n = 30;
        sum_t = 0;

        for i = 3:n
            %temp = P(:,i).^2;
            temp = 100*(P(:,i) - P(:,i-1).^2).^2 + (1 - P(:,i-1)).^2;
            sum_t = sum_t+temp;
        end
        
        g = sum_t;
        f(:,1) = P(:,1);
        f(:,2) = g.*(1 - sqrt(f(:,1))./g)+2;
        % Constrains   
        thita=-0.2*pi; 
        a=0.2;  
        b=10; 
        cc=1; 
        d=0.5; 
        e=1;

        
        c(:,1)=-(cos(thita)*(f(:,2)-e)-sin(thita)*f(:,1)-a*(abs(sin(b*pi*(sin(thita)*(f(:,2)-e)+cos(thita)*f(:,1)).^cc))).^d);
        c(:,1) = max(0, c(:,1));

        c_max = max(c, [], 1);
        for i = 1:size(c,2)
            if c_max(i) > 0
                c(:,i) = c(:,i) ./ c_max(i);
            end
        end
        cv = sum(c,2)/2;
        %cv = zeros(size(P,1),1);
        fit_obj_c=[f cv];           
    case 5
        n = 30;
        sum_t = 0;

        for i = 3:n
            %temp = P(:,i).^2;
            temp = 100*(P(:,i) - P(:,i-1).^2).^2 + (1 - P(:,i-1)).^2;
            sum_t = sum_t+temp;
        end
        
        g = sum_t;
        f(:,1) = P(:,1);
        f(:,2) = g.*(1 - sqrt(f(:,1))./g)+2.5;
        % Constrains   
        thita=-0.2*pi; 
        a=0.75;  
        b=10; 
        cc=1; 
        d=0.5; 
        e=1;

        
        c(:,1)=-(cos(thita)*(f(:,2)-e)-sin(thita)*f(:,1)-a*(abs(sin(b*pi*(sin(thita)*(f(:,2)-e)+cos(thita)*f(:,1)).^cc))).^d);
        c(:,1) = max(0, c(:,1));
        
        c_max = max(c, [], 1);
        for i = 1:size(c,2)
            if c_max(i) > 0
                c(:,i) = c(:,i) ./ c_max(i);
            end
        end
        cv = sum(c,2)/2;
        %cv = zeros(size(P,1),1);
        fit_obj_c=[f cv];                
    case 6
        n = 30;
        sum_t = 0;

        for i = 3:n
            %temp = P(:,i).^2;
            temp = 100*(P(:,i) - P(:,i-1).^2).^2 + (1 - P(:,i-1)).^2;
            sum_t = sum_t+temp;
        end
        
        g = sum_t;
        f(:,1) = P(:,1);
        f(:,2) = g.*(1 - sqrt(f(:,1))./g)+4;
        % Constrains   
        thita=-0.2*pi; 
        a=2;  
        b=10; 
        cc=1; 
        d=6; 
        e=1;
        
        c(:,1)=-(cos(thita)*(f(:,2)-e)-sin(thita)*f(:,1)-a*(abs(sin(b*pi*(sin(thita)*(f(:,2)-e)+cos(thita)*f(:,1)).^cc))).^d);
        c(:,1) = max(0, c(:,1));
        
        c_max = max(c, [], 1);
        for i = 1:size(c,2)
            if c_max(i) > 0
                c(:,i) = c(:,i) ./ c_max(i);
            end
        end
        cv = sum(c,2)/2;
        %cv = zeros(size(P,1),1);
        fit_obj_c=[f cv];
        
        %！！！！！！！！ 10略！！！！！！！！！！
    case 7
        n = 10;
        sum_t = 0;

        for i = 3:n
            %temp = P(:,i).^2;
            temp = 100*(P(:,i) - P(:,i-1).^2).^2 + (1 - P(:,i-1)).^2;
            sum_t = sum_t+temp;
        end
        
        g = sum_t;
        f(:,1) = P(:,1);
        f(:,2) = g.*(1 - sqrt(f(:,1))./g)+2;
        % Constrains   
        thita=-0.2*pi; 
        a=0.2;  
        b=10; 
        cc=1; 
        d=0.5; 
        e=1;
  
        c(:,1)=-(cos(thita)*(f(:,2)-e)-sin(thita)*f(:,1)-a*(abs(sin(b*pi*(sin(thita)*(f(:,2)-e)+cos(thita)*f(:,1)).^cc))).^d);
        c(:,1) = max(0, c(:,1));
        c(:,2) = g.*(1 - sqrt(f(:,1))./g)+0.73*f(:,1) -2;%%%%
        c(:,2) = max(0, c(:,2));
        
        c_max = max(c, [], 1);
        for i = 1:size(c,2)
            if c_max(i) > 0
                c(:,i) = c(:,i) ./ c_max(i);
            end
        end
        cv = sum(c,2)/2;
        %cv = zeros(size(P,1),1);
        fit_obj_c=[f cv];           
    case 8
        n = 10;
        sum_t = 0;

        for i = 3:n
            %temp = P(:,i).^2;
            temp = 100*(P(:,i) - P(:,i-1).^2).^2 + (1 - P(:,i-1)).^2;
            sum_t = sum_t+temp;
        end
        
        g = sum_t;
        f(:,1) = P(:,1);
        f(:,2) = g.*(1 - sqrt(f(:,1))./g)+2.5;
        % Constrains   
        thita=-0.2*pi; 
        a=0.75;  
        b=10; 
        cc=1; 
        d=0.5; 
        e=1;
        
        c(:,1)=-(cos(thita)*(f(:,2)-e)-sin(thita)*f(:,1)-a*(abs(sin(b*pi*(sin(thita)*(f(:,2)-e)+cos(thita)*f(:,1)).^cc))).^d);
        c(:,1) = max(0, c(:,1));
        c(:,2) = g.*(1 - sqrt(f(:,1))./g)+0.73*f(:,1) -1.5;%%%%
        c(:,2) = max(0, c(:,2));
        
        c_max = max(c, [], 1);
        for i = 1:size(c,2)
            if c_max(i) > 0
                c(:,i) = c(:,i) ./ c_max(i);
            end
        end
        cv = sum(c,2)/2;
        %cv = zeros(size(P,1),1);
        fit_obj_c=[f cv];                
    case 9
        n = 10;
        sum_t = 0;

        for i = 3:n
            %temp = P(:,i).^2;
            temp = 100*(P(:,i) - P(:,i-1).^2).^2 + (1 - P(:,i-1)).^2;
            sum_t = sum_t+temp;
        end
        
        g = sum_t;
        f(:,1) = P(:,1);
        f(:,2) = g.*(1 - sqrt(f(:,1))./g)+4;
        % Constrains   
        thita=-0.2*pi; 
        a=2;  
        b=10; 
        cc=1; 
        d=6; 
        e=1;
        
        c(:,1)=-(cos(thita)*(f(:,2)-e)-sin(thita)*f(:,1)-a*(abs(sin(b*pi*(sin(thita)*(f(:,2)-e)+cos(thita)*f(:,1)).^cc))).^d);
        c(:,1) = max(0, c(:,1));
        c(:,2) = g.*(1 - sqrt(f(:,1))./g)+0.73*f(:,1) -2;%%%%
        c(:,2) = max(0, c(:,2));
        
        c_max = max(c, [], 1);
        for i = 1:size(c,2)
            if c_max(i) > 0
                c(:,i) = c(:,i) ./ c_max(i);
            end
        end
        cv = sum(c,2)/2;
        %cv = zeros(size(P,1),1);
        fit_obj_c=[f cv];
           
    case 10
        n = 10;
        sum_t = 0;

        for i = 3:n
            %temp = P(:,i).^2;
            temp = 100*(P(:,i) - P(:,i-1).^2).^2 + (1 - P(:,i-1)).^2;
            sum_t = sum_t+temp;
        end
        
        g = sum_t;
        f(:,1) = P(:,1);
        f(:,2) = g.*(1 - sqrt(f(:,1))./g)+2;
        % Constrains   
        thita=-0.2*pi; 
        a=0.2;  
        b=10; 
        cc=1; 
        d=0.5; 
        e=1;
        
        c(:,1)=-(cos(thita)*(f(:,2)-e)-sin(thita)*f(:,1)-a*(abs(sin(b*pi*(sin(thita)*(f(:,2)-e)+cos(thita)*f(:,1)).^cc))).^d);
        c(:,1) = max(0, c(:,1));
        
        c_max = max(c, [], 1);
        for i = 1:size(c,2)
            if c_max(i) > 0
                c(:,i) = c(:,i) ./ c_max(i);
            end
        end
        cv = sum(c,2)/2;
        %cv = zeros(size(P,1),1);
        fit_obj_c=[f cv];           
    case 11
        n = 10;
        sum_t = 0;

        for i = 3:n
            %temp = P(:,i).^2;
            temp = 100*(P(:,i) - P(:,i-1).^2).^2 + (1 - P(:,i-1)).^2;
            sum_t = sum_t+temp;
        end
        
        g = sum_t;
        f(:,1) = P(:,1);
        f(:,2) = g.*(1 - sqrt(f(:,1))./g)+2.5;
        % Constrains   
        thita=-0.2*pi; 
        a=0.75;  
        b=10; 
        cc=1; 
        d=0.5; 
        e=1;
   
        c(:,1)=-(cos(thita)*(f(:,2)-e)-sin(thita)*f(:,1)-a*(abs(sin(b*pi*(sin(thita)*(f(:,2)-e)+cos(thita)*f(:,1)).^cc))).^d);
        c(:,1) = max(0, c(:,1));
        
        c_max = max(c, [], 1);
        for i = 1:size(c,2)
            if c_max(i) > 0
                c(:,i) = c(:,i) ./ c_max(i);
            end
        end
        cv = sum(c,2)/2;
        %cv = zeros(size(P,1),1);
        fit_obj_c=[f cv];                
    case 12
        n = 10;
        sum_t = 0;

        for i = 3:n
            %temp = P(:,i).^2;
            temp = 100*(P(:,i) - P(:,i-1).^2).^2 + (1 - P(:,i-1)).^2;
            sum_t = sum_t+temp;
        end
        
        g = sum_t;
        f(:,1) = P(:,1);
        f(:,2) = g.*(1 - sqrt(f(:,1))./g)+4;
        % Constrains   
        thita=-0.2*pi; 
        a=2;  
        b=10; 
        cc=1; 
        d=6; 
        e=1;
        
        c(:,1)=-(cos(thita)*(f(:,2)-e)-sin(thita)*f(:,1)-a*(abs(sin(b*pi*(sin(thita)*(f(:,2)-e)+cos(thita)*f(:,1)).^cc))).^d);
        c(:,1) = max(0, c(:,1));
        
        c_max = max(c, [], 1);
        for i = 1:size(c,2)
            if c_max(i) > 0
                c(:,i) = c(:,i) ./ c_max(i);
            end
        end
        cv = sum(c,2)/2;
        %cv = zeros(size(P,1),1);
        fit_obj_c=[f cv];
end
