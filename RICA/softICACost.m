%% Your job is to implement the RICA cost and gradient
function [cost,grad] = softICACost(theta, x, params)

% unpack weight matrix
W = reshape(theta, params.numFeatures, params.n);

% project weights to norm ball (prevents degenerate bases)
Wold = W;
W = l2rowscaled(W, 1);

%%% YOUR CODE HERE %%%
% Forward Prop
h = W*x;
r = W'*h;

% Reconstruction Loss and Back Prop
diff = (r - x);
reconstruction_cost = 0.5 * sum(sum(diff.^2));
outderv = diff;

% Sparsity Cost
% K = sqrt(params.epsilon + h.^2);
K = abs(params.epsilon + h);
sparsity_cost = params.lambda * sum(sum(K));
K = 1./K;

% compute the cost comprised of: 1) sparsity and 2) reconstruction
cost = sparsity_cost + reconstruction_cost;

% Backprop Output Layer
W2grad = outderv * h';

% Baclprop Hidden Layer
outderv = W * outderv;
outderv = outderv + params.lambda * (h .* K);

W1grad = outderv * x';
Wgrad = W1grad + W2grad';

% unproject gradient for minFunc
grad = l2rowscaledg(Wold, W, Wgrad, 1);
grad = grad(:);
