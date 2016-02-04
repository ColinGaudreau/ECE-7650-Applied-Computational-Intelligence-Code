clear all; close all;

X = load('ex4x.dat'); y = load('ex4y.dat');
t0 = cputime;
[theta, err, lfun] = elem_logistic_regression([0,0,0], X, y);
tf = cputime;
fprintf('Took %.4f seconds to execute', tf - t0);

theta % print to console

fun = @(x,theta) -(theta(2)/theta(3)*x + theta(1)/theta(3));
figure,
subplot(1,2,1);
plot(err,'ok');
subplot(1,2,2); hold on;
plot(X(y==0,1),X(y==0,2),'ob'); plot(X(y==1,1),X(y==1,2),'sr');
xx = [min(X(:,1)), max(X(:,1))];
plot(xx, fun(xx,theta), '-k');

prob = 1 - lfun(theta, [20,80]);
fprintf('Probability of rejection given 20%% and 80%% on first and last test: %.4f\n', prob * 100);