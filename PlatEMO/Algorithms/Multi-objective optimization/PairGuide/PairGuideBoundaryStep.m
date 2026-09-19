function Y = PairGuideBoundaryStep(Y,Q,F,I)
% Contract actual DE children inside brackets located by unevaluated queries.
    assert(isequal(size(Y),size(Q)) && size(F,2)==size(Y,2) && isequal(size(F),size(I)));
    assert(~isempty(F) && all(isfinite([Y;Q;F;I]),'all'));
    assert(all([Y;Q;F;I]>=-1e-12 & [Y;Q;F;I]<=1+1e-12,'all'));
    V = I-F; gap2 = sum(V.^2,2);
    assert(all(gap2>0),'PairGuide:InvalidBracket','Bracket endpoints must differ.');
    tolerance = 64*eps*sqrt(size(Y,2));
    for j = 1:size(Y,1)
        t = min(1,max(0,sum((Q(j,:)-F).*V,2)./gap2));
        distance = sum((Q(j,:)-(F+t.*V)).^2,2); [~,k] = min(distance);
        c = (F(k,:)+I(k,:))/2; r = sqrt(gap2(k))/2; delta = Y(j,:)-c;
        scale = r/(r+norm(delta));
        Y(j,:) = min(1,max(0,c+scale*delta));
        assert(norm(Y(j,:)-c)<=r+tolerance && ...
            max(norm(Y(j,:)-F(k,:)),norm(Y(j,:)-I(k,:)))<=2*r+tolerance);
    end
end
