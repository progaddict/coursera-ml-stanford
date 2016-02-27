function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

sigmoid_argument = zeros(m, 1);
for i=1:m,
  sigmoid_argument(i) = X(i,:) * theta;
end
estimated_probability = sigmoid(sigmoid_argument);

J = -(y' * log(estimated_probability) + (1-y)' * log(1-estimated_probability))
        + 0.5 * lambda * sum(theta(2:n).^2);
J = J / m;

diff_y = estimated_probability - y;
X_transp = X';
grad = (X_transp * diff_y + lambda * theta) ./ m;
grad(1) = (X_transp(1,:) * diff_y(:,1)) ./ m;

% =============================================================
end
