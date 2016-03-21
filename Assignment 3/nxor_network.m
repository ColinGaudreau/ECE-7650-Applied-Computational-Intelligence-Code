function [clf, weights] =  nxor_network(X,y,alpha,varargin)

%% Initialize weights
if nargin > 3
    weights = varargin{1};
else
    weights = cell(2,1);
    weights{1} = randn(3,2);
    weights{2} = randn(3,1);
end

%% Train network
MAX_ITER = 1000000;
ERR = 1e-4;
for i = 1:MAX_ITER
    ind = randperm(size(X,1));
    X = X(ind,:);
    y = y(ind);
    for j = 1:size(X,1)
        x = X(j,:)';
        activations = forward_propagate(x,weights);
        weights = backward_propagate(x,y(j),alpha,weights,activations);
    end
    
    if evaluate_error(X,y,weights) < ERR
        break;
    end
end

%% Return stuff
clf = @(x) (evaluate_nn(x,weights));

%% Functions
    function activations = forward_propagate(x,weights)
        activations = zeros(2,2);
        % compute activation for hidden layer
        newx = cat(1,x,1);
        activations(1,1) = sigmoid(dot(newx,weights{1}(:,1)));
        activations(2,1) = sigmoid(dot(newx,weights{1}(:,2)));
        
        % compute activation for output layer
        newx = cat(1,activations(:,1),1);
        activations(1,2) = sigmoid(dot(newx,weights{2}));
    end

    function weights = backward_propagate(x,y,alpha,weights,activations)
        
        newx = cat(1,x,1);
        weights{1}(:,1) = weights{1}(:,1) - alpha * newx * weights{2}(1) * ...
            activations(1,1) * (1 - activations(1,1)) * activations(1,2) * ...
            (1 - activations(1,2)) * (activations(1,2) - y);
        
        weights{1}(:,2) = weights{1}(:,2) - alpha * newx * weights{2}(2) * ...
            activations(2,1) * (1 - activations(2,1)) * activations(1,2) * ...
            (1 - activations(1,2)) * (activations(1,2) - y);
        
        newx = cat(1,activations(:,1),1);
        weights{2} = weights{2} - alpha * newx * (activations(1,2) - y) * activations(1,2) * ...
            (1 - activations(1,2));
    end

    function val = evaluate_nn(x,weights)
        if size(x,2) > 1
            x = x';
        end
        if size(x,2) > 1
            error('input vector has wrong dimensions');
        end
        activations = zeros(2,2);
        % compute activation for hidden layer
        newx = cat(1,x,1);
        activations(1,1) = sigmoid(dot(newx,weights{1}(:,1)));
        activations(2,1) = sigmoid(dot(newx,weights{1}(:,2)));
        
        % compute activation for output layer
        newx = cat(1,activations(:,1),1);
        val = sigmoid(dot(newx,weights{2}));
    end

    function val = sigmoid(x)
        val = 1./(1+exp(-x));
    end

    function val = evaluate_error(X,y,weights)
        val = 0;
        for i = 1:size(X,1)
            val = val + (evaluate_nn(X(i,:),weights) - y(i)).^2;
        end
        val = val/size(X,1);
    end
end