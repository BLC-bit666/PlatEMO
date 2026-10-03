function [A_B,B_A] = coverage(A,B)
% Calculate the set coverage of A and B
popsizeA = size(A,1);
popsizeB = size(B,1);
columnsize = size(A,2);
A_B = 0;
B_A = 0;
for i = 1:popsizeA
    recorderA = 0;

    for j = 1:popsizeB
        less = 0;
        equal = 0;
        more = 0;
        
        for k = 1:columnsize
            if A(i,k) < B(j,k)
                less = less+1;
            elseif A(i,k) == B(j,k)
                equal = equal+1;
            else
                more = more+1;
            end
        end
        
        if less == 0 && more ~= 0
            recorderA = 1;
        end
    end
    
    B_A = B_A+recorderA;
end

for i = 1:popsizeB
    recorderB = 0;
    
    for j = 1:popsizeA
        less = 0;
        equal = 0;
        more = 0;
        
        for k = 1:columnsize
            if B(i,k) < A(j,k)
                less = less+1;
            elseif B(i,k) == A(j,k)
                equal = equal+1;
            else
                more = more+1;
            end
        end
        
        if less == 0 && more ~= 0
            recorderB = 1;
        end
    end
    
    A_B = A_B+recorderB;
end