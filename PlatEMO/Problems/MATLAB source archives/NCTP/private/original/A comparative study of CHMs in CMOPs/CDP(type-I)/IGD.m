function average_dis = IGD(R,W)
% Calculate the IGD value of obtained pareto front 
Rsize = size(R,1);
Wsize = size(W,1);

dis_set = [];

for i = 1:Rsize
    dis = inf;
    
    for j = 1:Wsize
        current_dis = sqrt(sum((abs(R(i,:)-W(j,:))).^2));
        %current_dis = sum(abs(R(i,:)-W(j,:)));
        if current_dis<dis
            dis = current_dis;
        end
    end
    
    dis_set = [dis_set,dis];
end
    average_dis = sum(dis_set)/Rsize;