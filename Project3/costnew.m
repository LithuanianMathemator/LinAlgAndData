function costval = costnew(W2,W3,W4,W5,b2,b3,b4,b5,m,n,x1,x2,y)
costvec = zeros(m+n,1);
for i = 1:m+n
    x =[x1(i);x2(i)];
    a2 = activate(x,W2,b2);
    a3 = activate(a2,W3,b3);
    a4 = activate(a3,W4,b4);
    a5 = activate(a4,W5,b5);
    costvec(i) = norm(y(:,i) - a5,2);
end
costval = norm(costvec,2)^2;
end % of nested function