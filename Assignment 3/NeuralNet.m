classdef NeuralNet < handle
    
    properties
        weights;
    end
    
    methods
        
        function obj = NeuralNet(layers)
            initialize_weights(obj,layers);
        end
        
        function train(obj)
            
        end
        
        function activations = forward_propagate(obj,x)
            activations = cell(1,length(obj.weights));
            x = repmat(x,1,size(obj.weights{1},2));
            activations{1} = sigmoid(sum(x.*obj.weights{1},1)');
            for i = 2:length(activations)
                x = repmat(activations{i-1},1,size(obj.weights{i},2));
                activations{i} = sigmoid(sum(x.*obj.weights{i},1)');
            end
        end
        
        function backward_propagate(obj,x)
        end
        
        function val = sigmoid(x)
            val = 1./(1+exp(-x));
        end
        
        function initialize_weights(obj,layers)
            weights = cell(1,length(layers) - 1);
            for i = 1:length(weights)
                weights = randn(layers(i),layers(i+1));
            end
            obj.weights = weights;
        end
    end
end