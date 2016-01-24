clear all; close all;
x = load('ex3x.dat');
y = load('ex3y.dat');
sigma = std(x);
m = mean(x);

transform = @(x) [(x(:,1) - m(1))/sigma(1),...
    (x(:,2) - m(2))/sigma(2)];
% adding the one column is done inside the linear regression function
x = transform(x);

theta0 = [0,0,0];
alpha = 0.05;

theta = linear_regression(theta0,x,y,alpha,true) % print value to console

% x must be properly formatted and theta be 1xd.
theta = theta';
fun = @(theta,x) sum(repmat(theta,size(x,1),1).*[ones(size(x,1),1),x],2);

fprintf(['My prediction for a living area of 1650 square feet' ...
    'and 3 bedrooms is: %.2f$\n'], fun(theta,transform([1650,3])));
