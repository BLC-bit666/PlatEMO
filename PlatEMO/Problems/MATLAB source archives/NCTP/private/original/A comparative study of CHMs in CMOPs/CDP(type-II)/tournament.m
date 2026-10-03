function  parent = tournament(temp_parent1, temp_parent2, n, M)

distance = n + 3;

rank= n + 2 ;

if temp_parent1(rank) < temp_parent2(rank)
    parent = temp_parent1;
elseif temp_parent1(rank) > temp_parent2(rank)
    parent = temp_parent2;
elseif temp_parent1(distance) > temp_parent2(distance)
    parent = temp_parent1;
elseif temp_parent1(distance) < temp_parent2(distance)
    parent = temp_parent2;
elseif rand <= 0.5
    parent = temp_parent1;
else
    parent = temp_parent2;
end
