%% Project 3: Essentials of Deep Learning
%% Author: Jonas Lorenz
% In this project, we learn and play essential mathematical components of deep 
% learning on a simple artificial neural network by using the MATLAB script provided 
% by C. F. Higham and D. J. Higham along with their paper "Deep Learning: An Introduction 
% for Applied Mathematicians", arXiv:1801.05894v1, Jan. 2018.  Minor changes have 
% been made to the original script. 
%% Step 1: Setting up the training data

% initial training data 
m = 5; 
n = 5;
x1 = [0.1,0.3,0.1,0.6,0.4,0.6,0.5,0.9,0.4,0.7];
x2 = [0.1,0.4,0.5,0.9,0.2,0.3,0.6,0.2,0.4,0.6];
y = [ones(1,m) zeros(1,n); zeros(1,m) ones(1,n)];

% plot the training data 
figure(1) 
clf
a1 = subplot(1,1,1);
plot(x1(1:m),x2(1:m),'o','MarkerSize',12,'LineWidth',0.4,'MarkerEdgeColor','red')
hold on
plot(x1(m+1:m+n),x2(m+1:m+n),'bx','MarkerSize',12,'LineWidth',0.4) 

a1.XTick = [0 1];
a1.YTick = [0 1];
a1.FontWeight = 'Bold';
a1.FontSize = 16;
xlim([0,1])
ylim([0,1])

print -dpng pic_xy.png
tic    % begin of clock count
%% Step 2: Initialize weights and biases

%rng(5000);
rng('default') 
W2 = 0.5*randn(2,2);
W3 = 0.5*randn(3,2);
W4 = 0.5*randn(2,3);
b2 = 0.5*randn(2,1);
b3 = 0.5*randn(3,1);
b4 = 0.5*randn(2,1);
%% Step 3: set the learning rate and number of sgd iterations

eta = 0.05;  % learning rate 
Niter = 5e5; % total number of iters, adjustable to your computational budget
%% Step 4: backpropagation to train the network

savecost = zeros(Niter,1);
for counter = 1:Niter
    k = randi(m+n);  %choose one point from the sample
    x = [x1(k); x2(k)];
    % Forward pass
    a2 = activate(x,W2,b2);
    a3 = activate(a2,W3,b3);
    a4 = activate(a3,W4,b4);
    % Backward pass
    delta4 = a4.*(1-a4).*(a4-y(:,k));
    delta3 = a3.*(1-a3).*(W4'*delta4);
    delta2 = a2.*(1-a2).*(W3'*delta3);
    % Gradient step
    W2 = W2 - eta*delta2*x';
    W3 = W3 - eta*delta3*a2';
    W4 = W4 - eta*delta4*a3';
    b2 = b2 - eta*delta2;
    b3 = b3 - eta*delta3;
    b4 = b4 - eta*delta4;
    % Monitor progress
    newcost = cost(W2,W3,W4,b2,b3,b4,m,n,x1,x2,y);   % display cost to screen if you like
    savecost(counter) = newcost;
end

total_training_time = toc   % end of clock
%% Step 5: show the "convergence" of the cost function

figure(3)
clf
semilogy([1:1e4:Niter],savecost(1:1e4:Niter),'b-','LineWidth',2)
xlabel('Iteration Number')
ylabel('Value of cost function')
set(gca,'FontWeight','Bold','FontSize',18)

print -dpng pic_cost.png
%% Step 6: show the classification by displaying shaded and unshaded regions

N = 500;
Dx = 1/N;
Dy = 1/N;
xvals = [0:Dx:1];
yvals = [0:Dy:1];
for k1 = 1:N+1
    xk = xvals(k1);
    for k2 = 1:N+1
        yk = yvals(k2);
        xy = [xk;yk];
        a2 = activate(xy,W2,b2);
        a3 = activate(a2,W3,b3);
        a4 = activate(a3,W4,b4);
        Aval(k2,k1) = a4(1);%a4 is a 2 by 1 vector to indicate class
        Bval(k2,k1) = a4(2);
        %Aval(k2,k1) = norm([1,0]'-a4,2);
        %Bval(k2,k1) = norm([0,1]'-a4,2);
     end
end
[X,Y] = meshgrid(xvals,yvals);

figure(4)
clf
a2 = subplot(1,1,1);
Mval = Aval>=Bval;
%Mval = Aval<=Bval;
contourf(X,Y,Mval,[0.5 0.5]) %receiving warning!!!

hold on
colormap([1 1 1; 0.8 0.8 0.8])
plot(x1(1:m),x2(1:m),'ro','MarkerSize',12,'LineWidth',0.4)
plot(x1(m+1:m+n),x2(m+1:m+n),'bx','MarkerSize',12,'LineWidth',0.4)
a2.XTick = [0 1];
a2.YTick = [0 1];
a2.FontWeight = 'Bold';
a2.FontSize = 16;
xlim([0,1])
ylim([0,1])

print -dpng pic_bdy_bp.png
%% Task 1:
% Derive expressions of backpropagation and sgd for a network of L layers with 
% one neuron per layer.
% 
% *Solution:* Backpropagation is gradient descent, so our goal is to find the 
% gradients along which to descend. For this, we first want to get an idea of 
% what exactly we are trying to minimize here. Since our model is only one neuron 
% per layer, each step of our forward pass goes from $\mathbb{R}$ to $\mathbb{R}$. 
% Therefore our partial derivatives will just be scalars. The cost function is
% 
% $$C \colon \mathbb{R} \times \mathbb{R}^{1,2L} \to \mathbb{R}, \quad (x, \theta) 
% \mapsto \frac{1}{2}(a_L - y)^2.$$
% 
% where $x$ is our input and $\theta = (w_1, b_1, \ldots, w_L, b_K)$. We can 
% compute
% 
% $a_n = \sigma(w_na_{n-1} + b_n)$ and $a_1 = \sigma(w_1 x + b_1)$
% 
% where $\sigma$ is our activation function. We now want to adjust $b_k, w_k$ 
% such that our cost function is minimal, essentially changing it "into" the direction 
% where the negative partial derivative points to. First of we have
% 
% $\delta_L = \frac{\partial C}{\partial b_L}(x, \theta) = (a_L - y) \cdot \sigma'(w_L 
% a_{L-1} + b_L)$ and $\frac{\partial C}{\partial w_L}(x, \theta) = (a_L - y) 
% \cdot \sigma'(w_La_{L-1} + b_L) \cdot a_{L-1} = \delta_L \cdot a_{L-1}$
% 
% when simply using the chain rule to get the derivatives. Things do get a little 
% bit more complicated when finding $\delta_{L-1}$, however once we have done 
% that, the rest will somewhat fall into place. Here we first realize that we 
% essentially know how the derivative of 
% 
% $\xi \mapsto \frac{1}{2} (\sigma(\xi) - y)^2 $ since it is practically our 
% $\delta_L$. We now want to interpret $\xi$ as a function to use the chain rule 
% again, namely $b_{L-1} \mapsto w_La_{L-1} + b_L$ where $a_{L-1}$ is dependent 
% on the value of $b_{L-1}$. Using the chain rule now, we get
% 
% $\delta_{L-1} = \frac{\partial C}{\partial b_{L-1}}(x, \theta) = (a_L - y)\sigma'(w_La_{L-1} 
% + b_L) \cdot w_L \cdot \sigma'(w_{L-1}a_{L-2} + b_{L-2}) = \sigma'(w_{L-1}a_{L-2} 
% + b_{L-2}) \cdot w_{L-1} \delta_L$.
% 
% Similarly one can derive $\frac{\partial C}{\partial w_{L-1}}(x, \theta) = 
% \delta_{L-1} \cdot a_{L-2}$. One could now continue like this, but the astute 
% reader might see that our argument for $\xi$ translates onto every layer of 
% our neural network, thus giving us a compact form to write all of the derivatives, 
% namely
% 
% $\delta_{n} = \sigma'(w_n a_{n-1} + b_n) \cdot w_n \delta_{n+1}$ and $\frac{\partial 
% C}{\partial w_{n}}(x, \theta) = \delta_{n} \cdot a_{n-1}$
% 
% where $n = 1, \ldots, L-1$ and $a_0 := x$.
%% Task 2:
% (1) Add additional hidden layers and neurons to the existing network. Specifically,  
% netbpsgd is a "2-2-3-2" network, change it to a "2-5-5-5-2" network. Using a 
% learning rate of 0.05 and 3e5 number of sgd iterations to show the convergence 
% behavior of the cost function and the classification with the training data 
% set in netbpsgd.
% 
% *Solution:* Our goal is to adjust the matrices $W_k$ and vectors $b_k$ as 
% well was adding one of each.

tic    % begin of clock count
%rng(5000);
rng('default') 
W2 = 0.5*randn(5,2);
W3 = 0.5*randn(5,5);
W4 = 0.5*randn(5,5);
W5 = 0.5*randn(2,5);
b2 = 0.5*randn(5,1);
b3 = 0.5*randn(5,1);
b4 = 0.5*randn(5,1);
b5 = 0.5*randn(2,1);

eta = 0.05;  % learning rate 
Niter = 3e5; % total number of iters, adjustable to your computational budget

savecost = zeros(Niter,1);
for counter = 1:Niter
    k = randi(m+n);  %choose one point from the sample
    x = [x1(k); x2(k)];
    % Forward pass
    a2 = activate(x,W2,b2);
    a3 = activate(a2,W3,b3);
    a4 = activate(a3,W4,b4);
    a5 = activate(a4,W5,b5);
    % Backward pass
    delta5 = a5.*(1-a5).*(a5-y(:,k));
    delta4 = a4.*(1-a4).*(W5'*delta5);
    delta3 = a3.*(1-a3).*(W4'*delta4);
    delta2 = a2.*(1-a2).*(W3'*delta3);
    % Gradient step
    W2 = W2 - eta*delta2*x';
    W3 = W3 - eta*delta3*a2';
    W4 = W4 - eta*delta4*a3';
    W5 = W5 - eta*delta5*a4';

    b2 = b2 - eta*delta2;
    b3 = b3 - eta*delta3;
    b4 = b4 - eta*delta4;
    b5 = b5 - eta*delta5;
    % Monitor progress
    newcost = costnew(W2,W3,W4,W5,b2,b3,b4,b5,m,n,x1,x2,y);   % display cost to screen if you like
    savecost(counter) = newcost;
end

total_training_time = toc   % end of clock

figure(3)
clf
semilogy([1:1e4:Niter],savecost(1:1e4:Niter),'b-','LineWidth',2)
xlabel('Iteration Number')
ylabel('Value of cost function')
set(gca,'FontWeight','Bold','FontSize',18)

print -dpng pic_cost.png

N = 500;
Dx = 1/N;
Dy = 1/N;
xvals = [0:Dx:1];
yvals = [0:Dy:1];
for k1 = 1:N+1
    xk = xvals(k1);
    for k2 = 1:N+1
        yk = yvals(k2);
        xy = [xk;yk];
        a2 = activate(xy,W2,b2);
        a3 = activate(a2,W3,b3);
        a4 = activate(a3,W4,b4);
        a5 = activate(a4,W5,b5);
        Aval(k2,k1) = a5(1);%a4 is a 2 by 1 vector to indicate class
        Bval(k2,k1) = a5(2);
        %Aval(k2,k1) = norm([1,0]'-a4,2);
        %Bval(k2,k1) = norm([0,1]'-a4,2);
     end
end
[X,Y] = meshgrid(xvals,yvals);

figure(4)
clf
a2 = subplot(1,1,1);
Mval = Aval>=Bval;
%Mval = Aval<=Bval;
contourf(X,Y,Mval,[0.5 0.5]) %receiving warning!!!

hold on
colormap([1 1 1; 0.8 0.8 0.8])
plot(x1(1:m),x2(1:m),'ro','MarkerSize',12,'LineWidth',0.4)
plot(x1(m+1:m+n),x2(m+1:m+n),'bx','MarkerSize',12,'LineWidth',0.4)
a2.XTick = [0 1];
a2.YTick = [0 1];
a2.FontWeight = 'Bold';
a2.FontSize = 16;
xlim([0,1])
ylim([0,1])

print -dpng pic_bdy_bp.png
%% 
% (2) Replace the sigmoid activation function of netpbsgd with the ReLU activation 
% function. Using a learning rate of 0.0025 and a number of 3e5 sgd iterations 
% to show the convergence behaviour of the cost function and the classification 
% with the training data set in ntbpsgd.
% 
% *Solution:* Again we are doing the same as before, except that we are now 
% chaning the activation function.

tic    % begin of clock count
%rng(5000);
rng('default') 
W2 = 0.5*randn(2,2);
W3 = 0.5*randn(3,2);
W4 = 0.5*randn(2,3);
b2 = 0.5*randn(2,1);
b3 = 0.5*randn(3,1);
b4 = 0.5*randn(2,1);

eta = 0.0025;  % learning rate 
Niter = 3e5; % total number of iters, adjustable to your computational budget

savecost = zeros(Niter,1);
for counter = 1:Niter
    k = randi(m+n);  %choose one point from the sample
    x = [x1(k); x2(k)];
    % Forward pass
    a2 = activatenew(x,W2,b2);
    a3 = activatenew(a2,W3,b3);
    a4 = activatenew(a3,W4,b4);
    % Backward pass
    delta4 = (W4*a3 + b4>0).*(a4-y(:,k));
    delta3 = (W3*a2 + b3>0).*(W4'*delta4);
    delta2 = (W2*x + b2>0).*(W3'*delta3);
    % Gradient step
    W2 = W2 - eta*delta2*x';
    W3 = W3 - eta*delta3*a2';
    W4 = W4 - eta*delta4*a3';
    b2 = b2 - eta*delta2;
    b3 = b3 - eta*delta3;
    b4 = b4 - eta*delta4;
    % Monitor progress
    newcost = costrelu(W2,W3,W4,b2,b3,b4,m,n,x1,x2,y);   % display cost to screen if you like
    savecost(counter) = newcost;
end

total_training_time = toc   % end of clock

figure(3)
clf
semilogy([1:1e4:Niter],savecost(1:1e4:Niter),'b-','LineWidth',2)
xlabel('Iteration Number')
ylabel('Value of cost function')
set(gca,'FontWeight','Bold','FontSize',18)

print -dpng pic_cost.png

N = 500;
Dx = 1/N;
Dy = 1/N;
xvals = [0:Dx:1];
yvals = [0:Dy:1];
for k1 = 1:N+1
    xk = xvals(k1);
    for k2 = 1:N+1
        yk = yvals(k2);
        xy = [xk;yk];
        a2 = activatenew(xy,W2,b2);
        a3 = activatenew(a2,W3,b3);
        a4 = activatenew(a3,W4,b4);
        Aval(k2,k1) = a4(1);%a4 is a 2 by 1 vector to indicate class
        Bval(k2,k1) = a4(2);
        %Aval(k2,k1) = norm([1,0]'-a4,2);
        %Bval(k2,k1) = norm([0,1]'-a4,2);
     end
end
[X,Y] = meshgrid(xvals,yvals);

figure(4)
clf
a2 = subplot(1,1,1);
Mval = Aval>=Bval;
%Mval = Aval<=Bval;
contourf(X,Y,Mval,[0.5 0.5]) %receiving warning!!!

hold on
colormap([1 1 1; 0.8 0.8 0.8])
plot(x1(1:m),x2(1:m),'ro','MarkerSize',12,'LineWidth',0.4)
plot(x1(m+1:m+n),x2(m+1:m+n),'bx','MarkerSize',12,'LineWidth',0.4)
a2.XTick = [0 1];
a2.YTick = [0 1];
a2.FontWeight = 'Bold';
a2.FontSize = 16;
xlim([0,1])
ylim([0,1])

print -dpng pic_bdy_bp.png
%% 
% (3) Now we want to combine the previous two tasks to use ReLU with a "2-5-5-5-2" 
% network.

m = 12; 
n = 8;
x1 = [0.1,0.05,0.05,0.1,0.35,0.65,0.9,0.95,0.95,0.9,0.65,0.35,0.7,0.3,0.3,0.7,0.25,0.75,0.5,0.5];
x2 = [0.1,0.65,0.35,0.9,0.95,0.95,0.9,0.65,0.35,0.1,0.05,0.05,0.7,0.7,0.3,0.3,0.5,0.5,0.75,0.25];
y = [ones(1,m) zeros(1,n); zeros(1,m) ones(1,n)];

tic    % begin of clock count
%rng(5000);
rng('default') 
W2 = 0.5*randn(5,2);
W3 = 0.5*randn(5,5);
W4 = 0.5*randn(5,5);
W5 = 0.5*randn(2,5);
b2 = 0.5*randn(5,1);
b3 = 0.5*randn(5,1);
b4 = 0.5*randn(5,1);
b5 = 0.5*randn(2,1);

eta = 0.0025;  % learning rate 
Niter = 3e5; % total number of iters, adjustable to your computational budget

savecost = zeros(Niter,1);
for counter = 1:Niter
    k = randi(m+n);  %choose one point from the sample
    x = [x1(k); x2(k)];
    % Forward pass
    a2 = activatenew(x,W2,b2);
    a3 = activatenew(a2,W3,b3);
    a4 = activatenew(a3,W4,b4);
    a5 = activatenew(a4,W5,b5);
    % Backward pass
    delta5 = (W5*a4 + b5>0).*(a5-y(:,k));
    delta4 = (W4*a3 + b4>0).*(W5'*delta5);
    delta3 = (W3*a2 + b3>0).*(W4'*delta4);
    delta2 = (W2*x + b2>0).*(W3'*delta3);
    % Gradient step
    W2 = W2 - eta*delta2*x';
    W3 = W3 - eta*delta3*a2';
    W4 = W4 - eta*delta4*a3';
    W5 = W5 - eta*delta5*a4';

    b2 = b2 - eta*delta2;
    b3 = b3 - eta*delta3;
    b4 = b4 - eta*delta4;
    b5 = b5 - eta*delta5;
    % Monitor progress
    newcost = costnew(W2,W3,W4,W5,b2,b3,b4,b5,m,n,x1,x2,y);   % display cost to screen if you like
    savecost(counter) = newcost;
end

total_training_time = toc   % end of clock

figure(3)
clf
semilogy([1:1e4:Niter],savecost(1:1e4:Niter),'b-','LineWidth',2)
xlabel('Iteration Number')
ylabel('Value of cost function')
set(gca,'FontWeight','Bold','FontSize',18)

print -dpng pic_cost.png

N = 500;
Dx = 1/N;
Dy = 1/N;
xvals = [0:Dx:1];
yvals = [0:Dy:1];
for k1 = 1:N+1
    xk = xvals(k1);
    for k2 = 1:N+1
        yk = yvals(k2);
        xy = [xk;yk];
        a2 = activatenew(xy,W2,b2);
        a3 = activatenew(a2,W3,b3);
        a4 = activatenew(a3,W4,b4);
        a5 = activatenew(a4,W5,b5);
        Aval(k2,k1) = a5(1);%a4 is a 2 by 1 vector to indicate class
        Bval(k2,k1) = a5(2);
        %Aval(k2,k1) = norm([1,0]'-a4,2);
        %Bval(k2,k1) = norm([0,1]'-a4,2);
     end
end
[X,Y] = meshgrid(xvals,yvals);

figure(4)
clf
a2 = subplot(1,1,1);
Mval = Aval>=Bval;
%Mval = Aval<=Bval;
contourf(X,Y,Mval,[0.5 0.5]) %receiving warning!!!

hold on
colormap([1 1 1; 0.8 0.8 0.8])
plot(x1(1:m),x2(1:m),'ro','MarkerSize',12,'LineWidth',0.4)
plot(x1(m+1:m+n),x2(m+1:m+n),'bx','MarkerSize',12,'LineWidth',0.4)
a2.XTick = [0 1];
a2.YTick = [0 1];
a2.FontWeight = 'Bold';
a2.FontSize = 16;
xlim([0,1])
ylim([0,1])

print -dpng pic_bdy_bp.png
