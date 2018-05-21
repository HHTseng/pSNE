function [J , grad] = tsne_costwithgrad(y, P)
% Calculate objective function
% Define target probability distribution
% Define distance matrix r
n = size(P, 1);
sum_y = sum(y .^ 2, 2);
num = 1 ./ (1 + bsxfun(@plus, sum_y, bsxfun(@plus, sum_y', -2 * (y * y')))); % Student-t distribution
num(1:n+1:end) = 0;                                                 % set diagonal to zero
Q = max(num ./ sum(num(:)), realmin);                               % normalize to get probabilities

% Compute the gradients (faster implementation)
L = (P - Q) .* num;

% Cost function
J = sum(P(:) .* log(P(:))) - sum(P(:) .* log(Q(:)));

if nargout > 1 % gradient required
    grad = 4 * (diag(sum(L, 1)) - L) * y;
end