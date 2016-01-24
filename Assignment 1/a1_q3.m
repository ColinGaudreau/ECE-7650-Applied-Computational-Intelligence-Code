clear all; close all;
dat = load('d2noisy.txt');
x = dat(:,1:2); y = dat(:,3);
alpha = 0.02;
theta0 = randn(1,3);
theta = linear_regression(theta0, x, y, alpha, true)';

fun = @(x,theta) sum([ones(size(x,1),1),x].*repmat(theta0,size(x,1),1),2);

x1 = linspace(min(x(:,1)),max(x(:,1)),20);
x2 = linspace(min(x(:,2)),max(x(:,2)),20);
[xx1,xx2] = meshgrid(x1,x2);
xx1 = xx1(:); xx2 = xx2(:);
yy = fun([xx1,xx2],theta);
yy = reshape(yy,20,20);

figure, hold on;
plot3(x(:,1),x(:,2),y,'ok', 'linewidth', 2);
% plot3(xx1,xx2,fun([xx1,xx2],theta), 'r.', 'linewidth', 3);
surf(x1,x2,yy);
xlabel('Feature Dim. 1');
ylabel('Feature Dim. 2');
zlabel('Output');
legend('Data', 'Surface fit to data');
grid on;