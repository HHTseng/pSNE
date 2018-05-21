function ydata = exit_flat_p(P, labels, no_dims, rho)
%TSNE_P Performs exit_flat on affinity matrix P
%
%   mappedX = exit_flat_p(P, labels, no_dims)
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
    if ~exist('rho', 'var') || isempty(rho)
        rho = 0.95;
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
    max_iter = 150;                                    % maximum number of iterations
    epsilon = 300;                                      % initial learning rate
    min_gain = .01;                                     % minimum gain for delta-bar-delta
    
    % Make sure P-vals are set properly
    P(1:n + 1:end) = 0;                                 % set diagonal to zero
    P = 0.5 * (P + P');                                 % symmetrize P-values
    P = max(P ./ sum(P(:)), realmin);                   % make sure P-values sum to one
    const = sum(P(:) .* log(P(:)));                     % constant in KL divergence
    g = 2;                                              % lying P-factor, default = 4
    if ~initial_solution
        P = P * g;                                      % lie about the P-vals to find better local minima
    end
    
    % Initialize the solution
    if ~initial_solution
        ydata = .0001 * randn(n, no_dims);
        ydata = bsxfun(@rdivide, ydata, sqrt(sum(ydata.^2 ,2)));
    end
    y_incs  = zeros(size(ydata));
    gains = ones(size(ydata));
    
    % Run the iterations
    for iter=1:max_iter
        
        % Compute joint probability that point i and j are neighbors
        sum_ydata = sum(ydata .^ 2, 2);
        sqnum = bsxfun(@plus, sum_ydata, bsxfun(@plus, rho^2 .* sum_ydata', -2 * rho * (ydata * ydata')));
        % sqnum_{ij} = ( ||x_i - rho * x_j ||^2 )
        
        num = ( 1 ./ sqrt(sqnum) )^(no_dims + 1) ; % (spherical) exit distribution
        % num = ( ||x_i - rho * x_j ||)^{-(m+1) }
        
        if rho==1
        num(1:n+1:end) = 0; % If rho==1, set diagonal to 0
        end
        
        Q = max(num ./ sum(num(:)), realmin);  % normalize to get probabilities
        Q = Q';  % Transposing indices; Q_ij = ( ||x_j - rho * x_i ||)^{-(m+1)} / sum(...)
        
        % Compute the gradients (faster implementation)
        L = (P - Q) ./ sqnum;
        if rho==1
        L(1:n+1:end) = 0; % If rho==1, set diagonal to 0
        end
        
        y_grads = (no_dims + 1) * ( (diag(sum(L, 2)) - rho .* L) - rho *(L' - rho * diag(sum(L,1))) ) * ydata;
            
        % Update the solution
        gains = (gains + .3) .* (sign(y_grads) ~= sign(y_incs)) ...         % note that the y_grads are actually -y_grads
              + (gains * .1) .* (sign(y_grads) == sign(y_incs));
        gains(gains < min_gain) = min_gain;
        y_incs = momentum * y_incs - epsilon * (gains .* y_grads);
        ydata = ydata + y_incs;
        ydata = bsxfun(@minus, ydata, mean(ydata, 1));
        % renomalizing data back to sphere
        % ydata = bsxfun(@rdivide, ydata, sqrt(sum(ydata.^2 ,2)));
        
        
        % find distant points & pull back
        sum_ydata = real(sqrt(sum(ydata .^ 2, 2)));
        pos=find( sum_ydata > 300);
        ydata(pos,:) = ydata(pos,:)/10;
        
        
        % Update the momentum if necessary
        if iter == mom_switch_iter
            momentum = final_momentum;
        end
        if iter == stop_lying_iter && ~initial_solution
            P = P ./ g;
        end
        
       % Print out progress
        if ~rem(iter, 50)
            cost = const - sum(P(:) .* log(Q(:)));
            disp(['Iteration ' num2str(iter) ': error is ' num2str(cost)]);
            
            % Display scatter plot (maximally first three dimensions)
            str = ['Result of R^n Exit: ', num2str(iter), ' iterations'];
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
                if ~rem(iter, 50)
                    % pause;
                end
            end
        end
    end
    