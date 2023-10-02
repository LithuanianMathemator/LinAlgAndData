%{
Project Assignment 1
Author: Jonas Lorenz

In the following we will investigate different methods to classify
handwritten numbers using linear algebra.

The numbers are given in 28x28 pixel pictures. If we now take some
arbitrary handwritten number and want to classify it, we need to somehow
determine which number the written one is "closest" to. To train our model,
we will first determine the means of the respective number matrices.

How we proceed is now a question of which method we want to use.

1. The word "closest" in the introduction is no coincidence. This notion
of closeness will lead us directly to our first method. Closeness implies
some means to measure distance, which is exactly what we want to do. Given
a new handwritten number, we simply measure the Euclidean distance between
new picture and means of a given number. The number the new picture is
closest to will be our classification.

To start, we first want to compute the means of the respective numbers and
give a quick taste of our data set.
%}

clear;
load mnistdata;
% Visualize a selected train/test digit
figure(1)
n = 6;
for i = 1:n*n
    digit = train8(i,:);
    % digit = test8(i,:);
    digitImage = reshape(digit,28,28);
    subplot(n,n,i);
    image(rot90(flipud(digitImage),-1));
    colormap(gray(256));
    axis square tight off;
end
% Visualize the average train digits
T(1,:) = mean(train0);
T(2,:) = mean(train1);
T(3,:) = mean(train2);
T(4,:) = mean(train3);
T(5,:) = mean(train4);
T(6,:) = mean(train5);
T(7,:) = mean(train6);
T(8,:) = mean(train7);
T(9,:) = mean(train8);
T(10,:) = mean(train9);
for i = 1:10
    digitImage_mean(:,:,i) = reshape(T(i,:),28,28);
end
figure(2)
for i = 1:10
    subplot(2,5,i)
    image(rot90(flipud(digitImage_mean(:,:,i)),-1));
    colormap(gray(256));
    axis square tight off;
end

%{
We will now classify the test set using the function "classify" from the
file 'classify.m'. The printed table shows the successrate of each
classification for the respective numbers.

The table colums are showcasing "[number] | [correct] | [total]".
%}

successrate = zeros(10, 3);
result0 = classify(test0, T);
successrate(1,:) = [0, sum(result0(:) == 0), size(test0, 1)];
result1 = classify(test1, T);
successrate(2,:) = [1, sum(result1(:) == 1), size(test1, 1)];
result2 = classify(test2, T);
successrate(3,:) = [2, sum(result2(:) == 2), size(test2, 1)];
result3 = classify(test3, T);
successrate(4,:) = [3, sum(result3(:) == 3), size(test3, 1)];
result4 = classify(test4, T);
successrate(5,:) = [4, sum(result4(:) == 4), size(test4, 1)];
result5 = classify(test5, T);
successrate(6,:) = [5, sum(result5(:) == 5), size(test5, 1)];
result6 = classify(test6, T);
successrate(7,:) = [6, sum(result6(:) == 6), size(test6, 1)];
result7 = classify(test7, T);
successrate(8,:) = [7, sum(result7(:) == 7), size(test7, 1)];
result8 = classify(test8, T);
successrate(9,:) = [8, sum(result8(:) == 8), size(test8, 1)];
result9 = classify(test9, T);
successrate(10,:) = [9, sum(result9(:) == 9), size(test9, 1)];
disp('Succes rate of classification using Euclidean distance:')
successrate

%{
The results achieved by this are fairly good, but certainly not perfect.
While the number "1" was classified very accurately, the results from the
number "5" definitely leave us desiring better results.

We obviously have to find a different way to classify our data set.
Singular value decomposition.

2. In our second approach, we go beyond the matrix and its elements. This
time we want to interpret the picture as a linear function. The
characteristics of which we want to squeeze out of the SVD of the matrix.
We will see that by going beyond simple numbers in a grid, we can find the
hidden core of what characterizes the picture. Using the first few singular
vectors will allow us to reduce our problem to a simple least squares
problem. We want to find out how well a certain number can be represented
by some linear combination of the determined singular vectors. This will be
our new idea of the closeness mentioned in "1.".

To make a new classification, we will first clear our workspace and then
determine the first few singular vectors. How many we use can be changed
using the variable "base_len". The function used to classify using SVD is
called "svdclass" from 'svdclass.m'.
%}

% baselength: 5

clear;
load mnistdata;
basis_len = 5;
[U0,~,~] = svds(double(train0'), basis_len);
U(:,:,1) = U0;
[U1,~,~] = svds(double(train1'), basis_len);
U(:,:,2) = U1;
[U2,~,~] = svds(double(train2'), basis_len);
U(:,:,3) = U2;
[U3,~,~] = svds(double(train3'), basis_len);
U(:,:,4) = U3;
[U4,~,~] = svds(double(train4'), basis_len);
U(:,:,5) = U4;
[U5,~,~] = svds(double(train5'), basis_len);
U(:,:,6) = U5;
[U6,~,~] = svds(double(train6'), basis_len);
U(:,:,7) = U6;
[U7,~,~] = svds(double(train7'), basis_len);
U(:,:,8) = U7;
[U8,~,~] = svds(double(train8'), basis_len);
U(:,:,9) = U8;
[U9,~,~] = svds(double(train9'), basis_len);
U(:,:,10) = U9;

successrate = zeros(10, 3);
result0 = svdclass(test0, U);
successrate(1,:) = [0, sum(result0(:) == 0), size(test0, 1)];
result1 = svdclass(test1, U);
successrate(2,:) = [1, sum(result1(:) == 1), size(test1, 1)];
result2 = svdclass(test2, U);
successrate(3,:) = [2, sum(result2(:) == 2), size(test2, 1)];
result3 = svdclass(test3, U);
successrate(4,:) = [3, sum(result3(:) == 3), size(test3, 1)];
result4 = svdclass(test4, U);
successrate(5,:) = [4, sum(result4(:) == 4), size(test4, 1)];
result5 = svdclass(test5, U);
successrate(6,:) = [5, sum(result5(:) == 5), size(test5, 1)];
result6 = svdclass(test6, U);
successrate(7,:) = [6, sum(result6(:) == 6), size(test6, 1)];
result7 = svdclass(test7, U);
successrate(8,:) = [7, sum(result7(:) == 7), size(test7, 1)];
result8 = svdclass(test8, U);
successrate(9,:) = [8, sum(result8(:) == 8), size(test8, 1)];
result9 = svdclass(test9, U);
successrate(10,:) = [9, sum(result9(:) == 9), size(test9, 1)];
disp('Succes rate of classification using SVD (baselength 5):')
successrate

% baselength: 10

clear;
load mnistdata;
basis_len = 10;
[U0,~,~] = svds(double(train0'), basis_len);
U(:,:,1) = U0;
[U1,~,~] = svds(double(train1'), basis_len);
U(:,:,2) = U1;
[U2,~,~] = svds(double(train2'), basis_len);
U(:,:,3) = U2;
[U3,~,~] = svds(double(train3'), basis_len);
U(:,:,4) = U3;
[U4,~,~] = svds(double(train4'), basis_len);
U(:,:,5) = U4;
[U5,~,~] = svds(double(train5'), basis_len);
U(:,:,6) = U5;
[U6,~,~] = svds(double(train6'), basis_len);
U(:,:,7) = U6;
[U7,~,~] = svds(double(train7'), basis_len);
U(:,:,8) = U7;
[U8,~,~] = svds(double(train8'), basis_len);
U(:,:,9) = U8;
[U9,~,~] = svds(double(train9'), basis_len);
U(:,:,10) = U9;

successrate = zeros(10, 3);
result0 = svdclass(test0, U);
successrate(1,:) = [0, sum(result0(:) == 0), size(test0, 1)];
result1 = svdclass(test1, U);
successrate(2,:) = [1, sum(result1(:) == 1), size(test1, 1)];
result2 = svdclass(test2, U);
successrate(3,:) = [2, sum(result2(:) == 2), size(test2, 1)];
result3 = svdclass(test3, U);
successrate(4,:) = [3, sum(result3(:) == 3), size(test3, 1)];
result4 = svdclass(test4, U);
successrate(5,:) = [4, sum(result4(:) == 4), size(test4, 1)];
result5 = svdclass(test5, U);
successrate(6,:) = [5, sum(result5(:) == 5), size(test5, 1)];
result6 = svdclass(test6, U);
successrate(7,:) = [6, sum(result6(:) == 6), size(test6, 1)];
result7 = svdclass(test7, U);
successrate(8,:) = [7, sum(result7(:) == 7), size(test7, 1)];
result8 = svdclass(test8, U);
successrate(9,:) = [8, sum(result8(:) == 8), size(test8, 1)];
result9 = svdclass(test9, U);
successrate(10,:) = [9, sum(result9(:) == 9), size(test9, 1)];
disp('Succes rate of classification using SVD (baselength 10):')
successrate

% baselength: 20

clear;
load mnistdata;
basis_len = 20;
[U0,~,~] = svds(double(train0'), basis_len);
U(:,:,1) = U0;
[U1,~,~] = svds(double(train1'), basis_len);
U(:,:,2) = U1;
[U2,~,~] = svds(double(train2'), basis_len);
U(:,:,3) = U2;
[U3,~,~] = svds(double(train3'), basis_len);
U(:,:,4) = U3;
[U4,~,~] = svds(double(train4'), basis_len);
U(:,:,5) = U4;
[U5,~,~] = svds(double(train5'), basis_len);
U(:,:,6) = U5;
[U6,~,~] = svds(double(train6'), basis_len);
U(:,:,7) = U6;
[U7,~,~] = svds(double(train7'), basis_len);
U(:,:,8) = U7;
[U8,~,~] = svds(double(train8'), basis_len);
U(:,:,9) = U8;
[U9,~,~] = svds(double(train9'), basis_len);
U(:,:,10) = U9;

successrate = zeros(10, 3);
result0 = svdclass(test0, U);
successrate(1,:) = [0, sum(result0(:) == 0), size(test0, 1)];
result1 = svdclass(test1, U);
successrate(2,:) = [1, sum(result1(:) == 1), size(test1, 1)];
result2 = svdclass(test2, U);
successrate(3,:) = [2, sum(result2(:) == 2), size(test2, 1)];
result3 = svdclass(test3, U);
successrate(4,:) = [3, sum(result3(:) == 3), size(test3, 1)];
result4 = svdclass(test4, U);
successrate(5,:) = [4, sum(result4(:) == 4), size(test4, 1)];
result5 = svdclass(test5, U);
successrate(6,:) = [5, sum(result5(:) == 5), size(test5, 1)];
result6 = svdclass(test6, U);
successrate(7,:) = [6, sum(result6(:) == 6), size(test6, 1)];
result7 = svdclass(test7, U);
successrate(8,:) = [7, sum(result7(:) == 7), size(test7, 1)];
result8 = svdclass(test8, U);
successrate(9,:) = [8, sum(result8(:) == 8), size(test8, 1)];
result9 = svdclass(test9, U);
successrate(10,:) = [9, sum(result9(:) == 9), size(test9, 1)];
disp('Succes rate of classification using SVD (baselength 20):')
successrate

%{
By taking different baselengths, we can see that we get very good results
with a small set of base vectors. Increasing the number does not
necessarily mean great improvements in every case. Given the increasing
runtime when using more vectors it can therefore be undesirable if quick 
results are wanted or more accuracy than with a few vectors is not needed.
%}
