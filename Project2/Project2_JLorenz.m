%% Project 2 – Jonas Lorenz
%% Fisher's Linear Discriminant Analysis
% Fisher's Linear Discriminant Analysis (LDA) is a dimension reduction technique 
% commonly used for supervised classification problems. The goal of LDA is to 
% project the features in higher dimension space onto a lower dimensional space, 
% such that samples from different distribution can be 'separated'. Through this 
% project, we are going to learn the basic about Fisher's LDA, and to explore 
% practical issues for a couple of datasets from applications.
%% 
% *Dataset generation*
% 
% First we want to generate two classes of data points, labelled $A$ and $B%$, 
% each drawn from a normal distribution. Each data sample $x$ is a vector of length 
% 2, collected in matrices with each column corresponding to a point.

rng(0);
% generate data points
muA = [-8;5]; muB = [-3;8];
theta = pi/4; 
T = [cos(theta),-sin(theta); sin(theta), cos(theta)];
S = T*diag([1,2])*T';
DataA = S*randn(2, 40) + muA * ones(1,40);
DataB = S*randn(2, 60) + muB * ones(1,60);

% training and testing data: each column is a sample
TrainA = DataA(:,1:end-20); 
TestA = DataA(:,end-20+1:end);
TrainB = DataB(:,1:end-20); 
TestB = DataB(:,end-20+1:end);

% show data points
figure(1)
hold on;
plot(TrainA(1,:),TrainA(2,:), '+r', 'DisplayName','ClassA');
plot(TrainB(1,:),TrainB(2,:), 'xb', 'DisplayName','ClassB');
axis equal; legend show; title('training data')
%% 
% Fisher's LDA is now centered around the optimization problem
% 
% $\max_{\|v\| \neq 0} \frac{(v^Tm_A - v^Tm_B)^2}{v^T(\Sigma_A + \Sigma_B) v}$.
% 
% *Task 1:* Show that an optimal solution is given by $v = (\Sigma_A + \Sigma_B)^{-1}(m_A 
% - m_B)$.
% 
% *Solution:* To make notation easier, we want to introduce $C := (m_A - m_B)(m_A 
% - m_B)^T$ and $D := \Sigma_A + \Sigma_B$. Both of these matrices are symmetric 
% which will come in handy later. Right now, however, we note that our optimization 
% problem can be simply rewritten to
% 
% $\max_{v \neq 0} \frac{v^TCv}{v^TDv}$.
% 
% Since scaling $v$ does not affect the maximum, we barely need to find a vector 
% "facing the right direction", allowing us to require $v^TDv$ to be equal to 
% one. This simplifies our problem to
% 
% $\begin{cases}\min -v^TCv \\v^TDv -1 = 0\end{cases}$.
% 
% Since $\frac{d}{dv} -v^TCv = 2Cv$ and $\frac{d}{dv} v^TDv - 1 = 2Dv$ (notice 
% we are using the symmetry here), we need to solve the equation
% 
% $-2Cv - 2\lambda Dv = 0$.
% 
% This is equivalent to 
% 
% $(m_A-m_B)(m_A - m_B)^Tv = -\lambda (\Sigma_A + \Sigma_B)v$,
% 
% so we can see that an optimal solution is given by
% 
% $$\mu (\Sigma + \Sigma_B)^{-1}(m_A - m_B)$$
% 
% where $\mu := \frac{-(m_A - m_B)^Tv}{\lambda}$ is simply a scalar. Recalling 
% that scaling does not affect the optimal solution, this solves the exercise 
% following that
% 
% $$v = (\Sigma + \Sigma_B)^{-1}(m_A - m_B)$$
% 
% is therefore also an optimal solution.
%% 
% The separation achieved by this vector can be seen well in the following. 
% The stars represent the mean values.

% sample mean & covariance
mA = mean(TrainA')';
mB = mean(TrainB')';
sA = cov(TrainA');
sB = cov(TrainB');

% separation vector
v = (sA+sB)\(mA-mB);
v = v/norm(v);

% thresholding constant
c = v'*(mA+mB)/2;
cv = c*v;

% show data
projA = v*v'*TrainA; projB = v*v'*TrainB;
projmA = v*v'*mA; projmB = v*v'*mB;
figure(2)
hold on;
plot([TrainA(1,:),projA(1,:)], [TrainA(2,:),projA(2,:)], '+r');
plot([TrainB(1,:),projB(1,:)], [TrainB(2,:),projB(2,:)], 'xb');
plot([mA(1),projmA(1)],[mA(2),projmA(2)],'pr','MarkerFaceColor','red','Markersize',10);
plot([mB(1),projmB(1)],[mB(2),projmB(2)],'pb','MarkerFaceColor','blue','Markersize',10);
plot(cv(1),cv(2),'pg','MarkerFaceColor','green','Markersize',10);
plot(7*[-v(1),v(1)], 7*[-v(2),v(2)], '-k');
axis equal; title('separation direction and projection for training data')
%% 
% *Task 2:* Use $v$ to build a classifier and report the success rate.
% 
% *Solution:* We want to use $c = v^T(m_A + m_B)/2$ as a thresholding constant, 
% which, judging by the green star in the plot, should be a good constant to use 
% for our classifier. We will guess $A$ for $v^Tx >c$ and $B$ otherwise. This 
% follows from the fact that $v$ is a vector "into" the third quadrant. Since 
% the projections of $A$ are almost exclusively in that quadrant while the projections 
% of $B$ are in the first, the inner product of data in $A$ with $v$ will predominantly 
% be positive, the contrary being true for $B$. Intuition for this is given by 
% the fact that the inner product between to vectors shows how big a part one 
% vector is of the other.

% first we want to classify the testing data for A
innerA = v'*TestA;
% label 1 if categorized correctly, 0 otherwise
successA = innerA > c

% same for B
innerB = v'*TestB;
successB = innerB <= c
srate = (sum(successA) + sum(successB))/(size(TestA, 2) + size(TestB, 2))
% test mean
mA = mean(TestA')';
mB = mean(TestB')';

% show data
projA = v*v'*TestA; projB = v*v'*TestB;
projmA = v*v'*mA; projmB = v*v'*mB;
figure(3)
hold on;
plot([TestA(1,:),projA(1,:)], [TestA(2,:),projA(2,:)], '+r');
plot([TestB(1,:),projB(1,:)], [TestB(2,:),projB(2,:)], 'xb');
plot([mA(1),projmA(1)],[mA(2),projmA(2)],'pr','MarkerFaceColor','red','Markersize',10);
plot([mB(1),projmB(1)],[mB(2),projmB(2)],'pb','MarkerFaceColor','blue','Markersize',10);
plot(cv(1),cv(2),'pg','MarkerFaceColor','green','Markersize',10);
plot(7*[-v(1),v(1)], 7*[-v(2),v(2)], '-k');
axis equal; title('separation direction and projection for testing data')
%% 
% We now want to look at real-life data and test the effectiveness of LDA there. 
% For this, we want to look at the sonar and ionosphere datasets.

clear;
load sonar.mat
whos('-file', 'sonar.mat')
load ionosphere.mat
whos('-file', 'ionosphere.mat')
%% 
% *Task 3:* Test and report the success rate for classification by LDA using 
% the default threshold $c$. Use 70% percent of the data to train and the rest 
% to test the classification.
% 
% *Solution:* We want to start with the sonar dataset.

A = sonar_data(sonar_label == 0, :);
B = sonar_data(sonar_label == 1, :);
TrainA = A(1:int16(0.7*length(A)), :);
TrainB = B(1:int16(0.7*length(B)), :);
TestA = A(int16(0.7*length(A))+1:end, :);
TestB = B(int16(0.7*length(B))+1:end, :);
%% 
% With the data split up, we can now train our model.

% sample mean & covariance
mA = mean(TrainA)';
mB = mean(TrainB)';
sA = cov(TrainA);
sB = cov(TrainB);

% separation vector
v = (sA + sB)\(mA - mB);
v = v/norm(v);

% thresholding constant
c = v'*(mA + mB)/2;
%% 
% To get an idea how to classify the data based on the thresholding constant, 
% we will take a quick look at how $A$ and $B$ are being projected.

test1 = v'*TrainA(1:10, :)'
test2 = v'*TrainB(1:10, :)'
%% 
% This suggests that we should classify the data as $A$ when $v^Tx > c$. Now 
% we want to test our classifier and report the success rate.

innerA = v'*TestA';
successA = innerA > c
innerB = v'*TestB';
successB = innerB <=c
srate = (sum(successA) + sum(successB))/(size(TestA, 1) + size(TestB, 1))
%% 
% Now we can test LDA on the ionosphere dataset.

A = ionosphere_data(ionosphere_label == 0, :);
B = ionosphere_data(ionosphere_label == 1, :);
TrainA = A(1:int16(0.7*length(A)), :);
TrainB = B(1:int16(0.7*length(B)), :);
TestA = A(int16(0.7*length(A))+1:end, :);
TestB = B(int16(0.7*length(B))+1:end, :);
%% 
% With the data split up, we can now train our model.

% sample mean & covariance
mA = mean(TrainA)';
mB = mean(TrainB)';
sA = cov(TrainA);
sB = cov(TrainB);

% separation vector
v = pinv(sA + sB)*(mA - mB);
v = v/norm(v);

% thresholding constant
c = v'*(mA + mB)/2;
%% 
% To get an idea how to classify the data based on the thresholding constant, 
% we will take a quick look at how $A$ and $B$ are being projected.

test1 = v'*TrainA(1:10, :)'
test2 = v'*TrainB(1:10, :)'
%% 
% Again, this suggests that we should classify the data as $A$ when $v^Tx > 
% c$. Now we want to test our classifier and report the success rate.

innerA = v'*TestA';
successA = innerA > c
innerB = v'*TestB';
successB = innerB <=c
srate = (sum(successA) + sum(successB))/(size(TestA, 1) + size(TestB, 1))
%% 
% Now that we have achieved good results with the default threshold, we want 
% to see how sensitive the classification is to the choice of $c$.
% 
% *Task 4:* Try different thresholds for $c$.
% 
% *Solution:* First we want to see how changing $c$ impacts classification on 
% the sonar training data.

A = sonar_data(sonar_label == 0, :);
B = sonar_data(sonar_label == 1, :);
TrainA = A(1:int16(0.7*length(A)), :);
TrainB = B(1:int16(0.7*length(B)), :);
TestA = A(int16(0.7*length(A))+1:end, :);
TestB = B(int16(0.7*length(B))+1:end, :);

% sample mean & covariance
mA = mean(TrainA)';
mB = mean(TrainB)';
sA = cov(TrainA);
sB = cov(TrainB);

% separation vector
v = (sA + sB)\(mA - mB);
v = v/norm(v);

% thresholding constants
c1 = v'*(mA);
c2 = v'*(mA*(1/4) + mB*(3/4));
c3 = v'*(mA*(2/5) + mB*(3/5));
c4 = v'*(mA*(14/30) + mB*(16/30));
c5 = v'*(mA*(16/30) + mB*(14/30));
c6 = v'*(mA*(3/5) + mB*(2/5));
c7 = v'*(mA*(3/4) + mB*(1/4));
c8 = v'*(mB);

% classification
innerA = v'*TestA';
successA = innerA > c1;
innerB = v'*TestB';
successB = innerB <=c1;
srate = (sum(successA) + sum(successB))/(size(TestA, 1) + size(TestB, 1))

innerA = v'*TestA';
successA = innerA > c2;
innerB = v'*TestB';
successB = innerB <=c2;
srate = (sum(successA) + sum(successB))/(size(TestA, 1) + size(TestB, 1))

innerA = v'*TestA';
successA = innerA > c3;
innerB = v'*TestB';
successB = innerB <=c3;
srate = (sum(successA) + sum(successB))/(size(TestA, 1) + size(TestB, 1))

innerA = v'*TestA';
successA = innerA > c4;
innerB = v'*TestB';
successB = innerB <=c4;
srate = (sum(successA) + sum(successB))/(size(TestA, 1) + size(TestB, 1))

innerA = v'*TestA';
successA = innerA > c5;
innerB = v'*TestB';
successB = innerB <=c5;
srate = (sum(successA) + sum(successB))/(size(TestA, 1) + size(TestB, 1))

innerA = v'*TestA';
successA = innerA > c6;
innerB = v'*TestB';
successB = innerB <=c6;
srate = (sum(successA) + sum(successB))/(size(TestA, 1) + size(TestB, 1))

innerA = v'*TestA';
successA = innerA > c7;
innerB = v'*TestB';
successB = innerB <=c7;
srate = (sum(successA) + sum(successB))/(size(TestA, 1) + size(TestB, 1))

innerA = v'*TestA';
successA = innerA > c8;
innerB = v'*TestB';
successB = innerB <=c8;
srate = (sum(successA) + sum(successB))/(size(TestA, 1) + size(TestB, 1))
%% 
% We can see that changing $c$ actually gives improved results for $c_4$ and 
% $c_6$, suggesting that the sets $A$ and $B$ are slightly intertwined.
% 
% Now we want to see how changing $c$ impacts our model for the ionosphere dataset.

A = ionosphere_data(ionosphere_label == 0, :);
B = ionosphere_data(ionosphere_label == 1, :);
TrainA = A(1:int16(0.7*length(A)), :);
TrainB = B(1:int16(0.7*length(B)), :);
TestA = A(int16(0.7*length(A))+1:end, :);
TestB = B(int16(0.7*length(B))+1:end, :);

% sample mean & covariance
mA = mean(TrainA)';
mB = mean(TrainB)';
sA = cov(TrainA);
sB = cov(TrainB);

% separation vector
v = pinv(sA + sB)*(mA - mB);
v = v/norm(v);

% thresholding constants
c1 = v'*(mA);
c2 = v'*(mA*(1/4) + mB*(3/4));
c3 = v'*(mA*(2/5) + mB*(3/5));
c4 = v'*(mA*(14/30) + mB*(16/30));
c5 = v'*(mA*(16/30) + mB*(14/30));
c6 = v'*(mA*(3/5) + mB*(2/5));
c7 = v'*(mA*(3/4) + mB*(1/4));
c8 = v'*(mB);

% classification
innerA = v'*TestA';
successA = innerA > c1;
innerB = v'*TestB';
successB = innerB <=c1;
srate = (sum(successA) + sum(successB))/(size(TestA, 1) + size(TestB, 1))

innerA = v'*TestA';
successA = innerA > c2;
innerB = v'*TestB';
successB = innerB <=c2;
srate = (sum(successA) + sum(successB))/(size(TestA, 1) + size(TestB, 1))

innerA = v'*TestA';
successA = innerA > c3;
innerB = v'*TestB';
successB = innerB <=c3;
srate = (sum(successA) + sum(successB))/(size(TestA, 1) + size(TestB, 1))

innerA = v'*TestA';
successA = innerA > c4;
innerB = v'*TestB';
successB = innerB <=c4;
srate = (sum(successA) + sum(successB))/(size(TestA, 1) + size(TestB, 1))

innerA = v'*TestA';
successA = innerA > c5;
innerB = v'*TestB';
successB = innerB <=c5;
srate = (sum(successA) + sum(successB))/(size(TestA, 1) + size(TestB, 1))

innerA = v'*TestA';
successA = innerA > c6;
innerB = v'*TestB';
successB = innerB <=c6;
srate = (sum(successA) + sum(successB))/(size(TestA, 1) + size(TestB, 1))

innerA = v'*TestA';
successA = innerA > c7;
innerB = v'*TestB';
successB = innerB <=c7;
srate = (sum(successA) + sum(successB))/(size(TestA, 1) + size(TestB, 1))

innerA = v'*TestA';
successA = innerA > c8;
innerB = v'*TestB';
successB = innerB <=c8;
srate = (sum(successA) + sum(successB))/(size(TestA, 1) + size(TestB, 1))
%% 
% We can see that here the constants $c_4$ and $c_6$ give improved results with 
% $c_3$ giving equally good results compared to the default thresholding constant.
%% 
% *Task 5:* The matrix $S_A + S_B$ for the ionosphere dataset is singular. Why? 
% How can the problem be circumvented?
% 
% *Solution:* The matrix is singular due to feature 2 of our dataset being only 
% zeros for every element. Therefore our data is not actually 34-dimensional, 
% but 33-dimensional. In the covariance matrix, this leads to zeros in both row 
% and column 2 in $s_A$ and $s_B$ each. The issue can be adressed using the Moore-Penrose 
% pseudoinverse. As the attentive reader might have noticed, that is exactly what 
% was being done in tasks 3 and 4. It is the closest we can get, norm-wise, to 
% an actual solution of a linear equation featuring a singular matrix. Looking 
% back on task 3 however, one could possibly have also considered deleting the 
% second feature from our dataset and working with the remaining 33 dimensions.

ionosphere_data
sA
sB
