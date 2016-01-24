clear all; close all;
xdata = load('ex2x.dat');
ydata = load('ex2y.dat');
theta0 = [0,0];
alpha = 0.07;

% use debugger to see theta after the first iteration
theta = linear_regression(theta0, xdata, ydata, alpha, true) % shows theta in console

fun = @(theta, x) theta(1) + theta(2)*x;

disp(['At age 3.5 I predict a height of ' num2str(fun(theta, 3.5)) ' m']);
disp(['At age seven I predict a height of ' num2str(fun(theta, 7)) ' m']);