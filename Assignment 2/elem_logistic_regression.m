function varargout = elem_logistic_regression(theta0, xdata, ydata)

if size(theta0,1) > 1
    theta0 = theta0';
end
if size(theta0,1) > 1
    error('theta0 incorrect dimension');
end

assert(length(theta0) == size(xdata,2)+1);
assert(size(xdata,1) == size(ydata,1));

MIN_ERROR = 1e-5;
MAX_ITERATION = 1e4;
did_converge = false;
theta = theta0;

xdata = cat(2, ones(size(xdata,1),1), xdata);

err = zeros(1,MAX_ITERATION);

for i = 1:MAX_ITERATION
    new_theta = theta;
    for j = 1:length(theta)
        new_theta(j) = theta(j) - nominator(j, theta, xdata, ydata)/...
            denominator(j, theta, xdata);
    end
    
    err(i) = norm(new_theta - theta,1);
    if err(i) < MIN_ERROR
        did_converge = true;
        fprintf('%.4f\n', norm(new_theta - theta,1));
        break;
    end
    
    theta = new_theta;
end

if did_converge
    display(['Converged after ' num2str(i) ' iterations']);
else
    display('Did not converge');
end

varargout{1} = theta;
if nargout > 1
    varargout{2} = err(1:find(err,1,'last'));
end
if nargout > 2
    varargout{3} = @logistic_func;
end

    function val = nominator(ind,theta, xdata, ydata)
        theta = repmat(theta, size(xdata,1), 1);
        logistic = 1./(1 + exp(- sum(theta.*xdata, 2)));
        val = 1/size(xdata,1) * sum((logistic .* ydata).*xdata(:,ind), 1);
    end

    function val = denominator(ind, theta, xdata)
        theta = repmat(theta, size(xdata,1), 1);
        logistic = 1./(1 + exp(- sum(theta.*xdata, 2)));
        val = 0;
        for k = 1:size(xdata,2)
            val = sum(xdata(:,ind).*xdata(:,k).*logistic.*(1 - logistic), 1);
        end
    end
    
    function val = logistic_func(theta, x)
        theta = repmat(theta,size(x,1),1);
        x = cat(2, ones(size(x,1),1), x);
        val = 1./(1 + exp(-sum(x .* theta, 2)));
    end
end