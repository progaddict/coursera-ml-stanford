function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

min_error = Inf;
possible_C = [0.9 1 1.1];
possible_sigma = [0.09 0.1 0.11];
for i = 1:length(possible_C)
    current_C = possible_C(i);
    for j = 1:length(possible_sigma)
        current_sigma = possible_sigma(j);
        model= svmTrain(X, y, current_C, @(x1, x2) gaussianKernel(x1, x2, current_sigma));
        predicted_values = svmPredict(model, Xval);
        current_error = mean(double(predicted_values ~= yval));
        fprintf('%f \t %f \t %f\n', current_C, current_sigma, current_error);
        if (current_error < min_error)
            C = current_C;
            sigma = current_sigma;
            min_error = current_error;
            fprintf('updated C to %f and sigma to %f, error is %f\n', C, sigma, min_error);
        end
    end
end 
fprintf('C = %f, sigma = %f', C, sigma);

% =========================================================================

end
