function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%
theta_reg = theta;
theta_reg(1,:)=[]; % don't regularize theta(0);

h = X*theta;
diff = h - y;
J_norm = (1/(2*m)) * diff' * diff;
grad_norm = (1/m) * (h-y)' * X;

reg = ((lambda / (2*m)) * theta_reg' * theta_reg);

J = J_norm + reg;


grad = grad_norm' + ((lambda / m ) * theta);

grad(1) = grad_norm(1,1); % don't regularize theta(0)







% =========================================================================

grad = grad(:);

end
