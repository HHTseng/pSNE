function ydata = tsne_p(P, labels, no_dims)
%TSNE_P Performs symmetric t-SNE on affinity matrix P
%
%   mappedX = tsne_p(P, labels, no_dims)
%
% The function performs symmetric t-SNE on pairwise similarity matrix P
% to create a low-dimensional map of no_dims dimensions (default = 2).
% The matrix P is assumed to be symmetric, sum up to 1, and have zeros
% on the diagonal.
% The labels of the data are not used by t-SNE itself, however, they
% are used to color intermediate plots. Please provide an empty labels
% matrix [] if you don't want to plot results during the optimization.
% The low-dimensional data representation is returned in mappedX.

% This file is part of the Matlab Toolbox for Dimensionality Reduction.
% The toolbox can be obtained from http://homepage.tudelft.nl/19j49
% You are free to use, change, or redistribute this code in any way you
% want for non-commercial purposes. However, it is appreciated if you
% maintain the name of the original author.
%
% (C) Laurens van der Maaten, Delft University of Technology



if ~exist('labels', 'var')
    labels = [];
end
if ~exist('no_dims', 'var') || isempty(no_dims)
    no_dims = 2;
end

% First check whether we already have an initial solution
if numel(no_dims) > 1
    initial_solution = true;
    ydata = no_dims;
    no_dims = size(ydata, 2);
else
    initial_solution = false;
end

% Initialize some variables
n = size(P, 1);                                     % number of instances
momentum = 0.5;                                     % initial momentum
final_momentum = 0.8;                               % value to which momentum is changed
mom_switch_iter = 250;                              % iteration at which momentum is changed
stop_lying_iter = 100;                              % iteration at which lying about P-values is stopped
max_iter = 1500;                                    % maximum number of iterations
epsilon = 500;                                      % initial learning rate
min_gain = .01;                                     % minimum gain for delta-bar-delta

% Make sure P-vals are set properly
P(1:n + 1:end) = 0;                                 % set diagonal to zero
P = 0.5 * (P + P');                                 % symmetrize P-values
P = max(P ./ sum(P(:)), realmin);                   % make sure P-values sum to one

% Initialize the solution
if ~initial_solution
    ydata = .0001 * randn(n, no_dims);
end

options = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true);
f = @(y)tsne_costwithgrad(y , P);
[y,fval,exitflag,output] = fminunc(f, ydata ,options)

ydata = y; cost = fval;
clear y; clear fval;

% Display scatter plot (maximally first three dimensions)
str = ['Result of t-SNE'];
if ~isempty(labels)
    if no_dims == 1
        scatter(ydata, ydata, 15, labels, 'filled'); title(str);
    elseif no_dims == 2
        scatter(ydata(:,1), ydata(:,2), 15, labels, 'filled'); title(str);
    else
        scatter3(ydata(:,1), ydata(:,2), ydata(:,3), 15, labels, 'filled'); title(str);
    end
    colormap(jet)
    colorbar
    axis equal tight
    %             axis off
    drawnow
end

end
