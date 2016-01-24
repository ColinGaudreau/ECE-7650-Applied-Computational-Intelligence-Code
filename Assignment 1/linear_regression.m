function theta = linear_regression(theta0, xdata, ydata, alpha, varargin)
% xdata is n x d, n being the number of data points and
% d being the dimensionality of the features space.

if size(theta0,2) > 1
    theta0 = theta0';
end
if size(theta0,2) > 1
    error('theta0 incorrect dimension');
end
assert(length(theta0) == size(xdata,2)+1);
assert(size(xdata,1) == size(ydata,1));

if ~isempty(varargin)
    should_plot = varargin{1};
else
    should_plot = false;
end

MIN_ERROR = 1e-16;
MAX_ITERATION = 1e4;
did_converge = false;
theta = theta0;

xdata = cat(2, ones(size(xdata,1),1), xdata);

handle = figure, hold on;
xlabel('Iteration'); ylabel('Error');

err = zeros(1,MAX_ITERATION);

for i=1:MAX_ITERATION
    new_theta = theta - alpha * del_J(theta, xdata, ydata);
    
    if norm(new_theta - theta,1) < MIN_ERROR
        did_converge = true;
        fprintf('%.4f\n', norm(new_theta - theta,1));
        break;
    end
    
    err(i) = J(theta,xdata,ydata);
    if should_plot && mod(i,20)==0
        plot(i,err(i), '--ok', 'linewidth', 2);
        drawnow;
    end
    
    theta = new_theta;
end

if should_plot
    close(handle);
    figure,
    plot(1:i-1, err(1:i-1), '-r', 'linewidth', 3);
    xlabel('Iteration'); ylabel('Error');
end

if did_converge
    display(['Converged after ' num2str(i) ' iterations']);
else
    display('Did not converge');
end

    function val = J(theta, xdata, ydata)
        theta = repmat(theta',size(xdata,1),1);
        val = sum((sum(theta.*xdata,2) - ydata).^2, 1)/(2*size(xdata,1));
    end

    function val = del_J(theta, xdata, ydata)
        theta = repmat(theta',size(xdata,1),1);
        dif = (sum(theta.*xdata,2) - ydata);
        dif = repmat(dif, 1,size(theta,2));
        val = (sum(dif.*xdata,1)/size(xdata,1))';
    end
end