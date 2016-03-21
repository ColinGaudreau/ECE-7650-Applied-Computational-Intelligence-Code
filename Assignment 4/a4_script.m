clear all; close all;

[y, X] = libsvmread('twofeature.txt');

X = full(X); % convert to full
y(y==-1) = 0;

outlier = 51; % index of outlier point
X_fixed = cat(1,X(1:outlier-1,:), X(outlier+1:end,:));
y_fixed = cat(1,y(1:outlier-1,:), y(outlier+1:end,:));

tic
model = svmtrain(y_fixed, X_fixed, '-s 0 -t 0 -c 1000'); % C=1000 corresponds to no regularization
toc

b = -model.rho;
w = model.SVs' * model.sv_coef;

bnd = @(x,theta) -theta(2)/theta(3) * x - theta(1)/theta(3);
xx = [min(X(:,1))-10, max(X(:,1))+10];

figure, hold on;
plot(X_fixed(y_fixed==1,1),X_fixed(y_fixed==1,2), 'ob');
plot(X_fixed(y_fixed~=1,1),X_fixed(y_fixed~=1,2), 'sr');
plot(xx,bnd(xx,[b;w]), '--k', 'linewidth', 3);

tic
theta = logistic_regression([0,0,0], X_fixed, y_fixed, 'grad', 100); % doesn't seem to converge using Newton's method
toc

% print result of log, regression
plot(xx,bnd(xx,theta), '--g', 'linewidth',3);

% Find centroid of both classes 
m1 = sum(X_fixed(y_fixed==1,:),1) / length(y_fixed(y_fixed==1));
m2 = sum(X_fixed(y_fixed~=1,:),1) / length(y_fixed(y_fixed~=1));
C = .5 * (m1 + m2);

w_cent = m1 - m2;
b_cent = - dot(w_cent, 0.5*(m1 + m2));

plot(xx,bnd(xx,[b_cent,w_cent]), '--y', 'linewidth',3);

legend('Class 1', 'Class 2', 'SVM', 'Logistic Regression', 'Centroid Method');

% Draw midpoint and centroids
plot(m1(1),m1(2), 'ok', 'markersize', 10, 'markerfacecolor', 'k');
plot(m2(1),m2(2), 'ok', 'markersize', 10, 'markerfacecolor', 'k');
plot(C(1),C(2), 'ok', 'markersize', 10, 'markerfacecolor', 'k');

% add new points to set
X_new = repmat([40,50], 1000, 1) + 4 * (rand(1000,2) - 0.5);
X_new = cat(1,X_new, X_fixed);
y_new = cat(1,ones(1000,1), y_fixed);

model_new = svmtrain(y_new, X_new, '-s 0 -t 0 -c 1000');

b_new = -model_new.rho;
w_new = model_new.SVs' * model_new.sv_coef;

figure, hold on;
plot(X_new(y_new==1,1),X_new(y_new==1,2), 'ob');
plot(X_new(y_new~=1,1),X_new(y_new~=1,2), 'sr');
plot(xx,bnd(xx,[b_new;w_new]), '--k', 'linewidth', 3);
plot(xx,bnd(xx,[b;w]), '--g', 'linewidth', 3);
legend('Class 1', 'Class 2', 'New SVM model', 'Old SVM model');