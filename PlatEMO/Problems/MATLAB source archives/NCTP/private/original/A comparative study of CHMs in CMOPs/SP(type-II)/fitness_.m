function [fitness_val,fit_obj_c] = fitness_(P,problem,f_num)
% Calculate the modified fitness value of P
% f_num: number of objective functions
% c_num: number of constrains

popsize = size(P,1);
%f = ones(popsize,f_num);% normalized function value matrix
%c = ones(popsize,c_num);% constraint violation matrix
%v = zeros(popsize,1);% modified constraint violation matrix
cv = [];
fit_obj_c = [];

%x = zeros(popsize,f_num);% first part of the modified penalty value
%y = zeros(popsize,f_num);% first part of the modified penalty value
%d = ones(popsize,f_num);% distance value matrix

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
        f(:,2) = g.*(1 - sqrt(f(:,1))./g)+1;
        % Constrains   
        thita=-0.2*pi; 
        a=0.2;  
        b=10; 
        cc=1; 
        d=0.5; 
        e=1;

        
        c(:,1)=-(cos(thita)*(f(:,2)-e)-sin(thita)*f(:,1)-a*(abs(sin(b*pi*(sin(thita)*(f(:,2)-e)+cos(thita)*f(:,1)).^cc))).^d);
        c(:,1) = max(0, c(:,1));
        c(:,2) = g.*(1 - sqrt(f(:,1))./g)+0.73*f(:,1) - 3;%%%%
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
        f(:,2) = g.*(1 - sqrt(f(:,1))./g)+1;
        % Constrains   
        thita=-0.2*pi; 
        a=0.75;  
        b=10; 
        cc=1; 
        d=0.5; 
        e=1;
   
        c(:,1)=-(cos(thita)*(f(:,2)-e)-sin(thita)*f(:,1)-a*(abs(sin(b*pi*(sin(thita)*(f(:,2)-e)+cos(thita)*f(:,1)).^cc))).^d);
        c(:,1) = max(0, c(:,1));
        c(:,2) = g.*(1 - sqrt(f(:,1))./g)+0.73*f(:,1) - 3;%%%%
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
        f(:,2) = g.*(1 - sqrt(f(:,1))./g)+1;
        % Constrains   
        thita=-0.2*pi; 
        a=2;  
        b=10; 
        cc=1; 
        d=6; 
        e=1;
        
        c(:,1)=-(cos(thita)*(f(:,2)-e)-sin(thita)*f(:,1)-a*(abs(sin(b*pi*(sin(thita)*(f(:,2)-e)+cos(thita)*f(:,1)).^cc))).^d);
        c(:,1) = max(0, c(:,1));
        c(:,2) = g.*(1 - sqrt(f(:,1))./g)+0.73*f(:,1) - 5;%%%%
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
        
        %——————大可行域————————
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
        f(:,2) = g.*(1 - sqrt(f(:,1))./g)+1;
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
        f(:,2) = g.*(1 - sqrt(f(:,1))./g)+1;
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
        f(:,2) = g.*(1 - sqrt(f(:,1))./g)+1;
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
        
        %———————— 10维——————————
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
        f(:,2) = g.*(1 - sqrt(f(:,1))./g)+1;
        % Constrains   
        thita=-0.2*pi; 
        a=0.2;  
        b=10; 
        cc=1; 
        d=0.5; 
        e=1;
  
        c(:,1)=-(cos(thita)*(f(:,2)-e)-sin(thita)*f(:,1)-a*(abs(sin(b*pi*(sin(thita)*(f(:,2)-e)+cos(thita)*f(:,1)).^cc))).^d);
        c(:,1) = max(0, c(:,1));
        c(:,2) = g.*(1 - sqrt(f(:,1))./g)+0.73*f(:,1) - 3;%%%%
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
        f(:,2) = g.*(1 - sqrt(f(:,1))./g)+1;
        % Constrains   
        thita=-0.2*pi; 
        a=0.75;  
        b=10; 
        cc=1; 
        d=0.5; 
        e=1;
        
        c(:,1)=-(cos(thita)*(f(:,2)-e)-sin(thita)*f(:,1)-a*(abs(sin(b*pi*(sin(thita)*(f(:,2)-e)+cos(thita)*f(:,1)).^cc))).^d);
        c(:,1) = max(0, c(:,1));
        c(:,2) = g.*(1 - sqrt(f(:,1))./g)+0.73*f(:,1) - 3;%%%%
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
        f(:,2) = g.*(1 - sqrt(f(:,1))./g)+1;
        % Constrains   
        thita=-0.2*pi; 
        a=2;  
        b=10; 
        cc=1; 
        d=6; 
        e=1;
        
        c(:,1)=-(cos(thita)*(f(:,2)-e)-sin(thita)*f(:,1)-a*(abs(sin(b*pi*(sin(thita)*(f(:,2)-e)+cos(thita)*f(:,1)).^cc))).^d);
        c(:,1) = max(0, c(:,1));
        c(:,2) = g.*(1 - sqrt(f(:,1))./g)+0.73*f(:,1) - 5;%%%%
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
        f(:,2) = g.*(1 - sqrt(f(:,1))./g)+1;
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
        f(:,2) = g.*(1 - sqrt(f(:,1))./g)+1;
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
        f(:,2) = g.*(1 - sqrt(f(:,1))./g)+1;
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



% normalized object function value 
sorted_f = sort(f);
fmin = sorted_f(1,:);
fmax = sorted_f(popsize,:);
%%%%%%%%%%%%%
%fmin = min(f, [], 1);
%fmax = max(f, [], 1);
f_normalized = (f - ones(popsize,1)*fmin)./(ones(popsize,1)*(fmax - fmin)+ 1e-100);

t1 = find(cv==0);
t2 = length(t1);
rf = t2/popsize;
        
% modified constraint violation
%sorted_c = sort(c,2);
%for i = 1:popsize
%    if sorted_c(i,c_num)~=0
%       rf = rf+1;
%       v(i,:) = sum(sorted_c(i,:)/sorted_c(i,c_num))/c_num;
%    end
%end
        
% Calculate the fessible ratio


% Calculate the distance value
if rf == 0
    fitness_temp = cv* ones( 1 , f_num );
    
elseif 0 < rf && rf < 1

    d = sqrt(( cv * ones(1, f_num) ).^2 + f_normalized.^2);

    
    %找出群体中不可行解所在的位置
    temp_2 = find( cv > 0 );

    %初始化penalty的值
    penalty = zeros(popsize, f_num);
    
    %计算不可行解的penalty的值，可行解的penalty=0，不需要计算
    penalty( temp_2, : ) = (1 - rf) * cv( temp_2 ) * ones( 1 , f_num ) + rf * f_normalized( temp_2 , : );

    %将fit赋值给fit_v
    %fit_v = fit;

    %计算所有个体的最终的适应度函数值,fit_v是popsize*m的m个目标函数的矩阵,修改fit中1~m的值
    fitness_temp = d + penalty;


else

    fitness_temp = f_normalized;

end
    

fitness_val = [fitness_temp,cv];
