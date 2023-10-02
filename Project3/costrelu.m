function costval = costrelu(W2,W3,W4,b2,b3,b4,m,n,x1,x2,y)
costvec = zeros(m+n,1);
for i = 1:m+n
    x =[x1(i);x2(i)];
    a2 = activatenew(x,W2,b2);
    a3 = activatenew(a2,W3,b3);
    a4 = activatenew(a3,W4,b4);
    costvec(i) = norm(y(:,i) - a4,2);
end
costval = norm(costvec,2)^2;
end % of nested function