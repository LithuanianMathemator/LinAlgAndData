function y = activatenew(x,W,b)
% ACTIVATE Evaluates ReLU function.
% x is the input vector, y is the output vector
% W contains the weights, b contains the shifts
% The ith component of y is activate((Wx+b)_i)
% where activate(z) = 1/(1+exp(-z))
a = length(W*x);
y = max(W*x+b, zeros(a,1));
end % of nested function^