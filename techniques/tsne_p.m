function ydata = tsne_p(P, labels, no_dims, CGPU_type)
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
max_iter = 15000;                                    % maximum number of iterations
epsilon = 100;                                      % initial learning rate
min_gain = .01;                                     % minimum gain for delta-bar-delta


% Switch computation architecture
switch CGPU_type
    case {'CPU', 'cpu'}
        % Make sure P-vals are set properly
        P(1:n + 1:end) = 0;                                 % set diagonal to zero
        P = 0.5 * (P + P');                                 % symmetrize P-values
        P = max(P ./ sum(P(:)), realmin);                   % make sure P-values sum to one
        
        % disp(['max(P-P_GPU)=', num2str(max(max(P - P_GPU)))] ); % check GPU=CPU
        const = sum(P(:) .* log(P(:)));
        if ~initial_solution
            P = P * 4;                                      % lie about the P-vals to find better local minima
        end
        
        % Initialize the solution
        if ~initial_solution
            ydata = .0001 * randn(n, no_dims);
        end
        y_incs  = zeros(size(ydata));
        gains = ones(size(ydata));
        
        % Run the iterations
        for iter=1:max_iter
            % Compute joint probability that point i and j are neighbors
            sum_ydata = sum(ydata .^ 2, 2);
            num = 1 ./ (1 + bsxfun(@plus, sum_ydata, bsxfun(@plus, sum_ydata', -2 * (ydata * ydata')))); % Student-t distribution
            num(1:n+1:end) = 0;                                                 % set diagonal to zero
            
            Q = max(num ./ sum(num(:)), realmin);                               % normalize to get probabilities
            
            % Compute the gradients (faster implementation)
            L = (P - Q) .* num;
            y_grads = 4 * (diag(sum(L, 1)) - L) * ydata;
            
            % Update the solution
            gains = (gains + .2) .* (sign(y_grads) ~= sign(y_incs)) ...         % note that the y_grads are actually -y_grads
                + (gains * .8) .* (sign(y_grads) == sign(y_incs));
            gains(gains < min_gain) = min_gain;
            y_incs = momentum * y_incs - epsilon * (gains .* y_grads);
            ydata = ydata + y_incs;
            
            ydata = bsxfun(@minus, ydata, mean(ydata, 1));
            
            % find distant points & pull back
            sum_ydata = real(sqrt(sum(ydata .^ 2, 2)));
            
            % Update the momentum if necessary
            if iter == mom_switch_iter
                momentum = final_momentum;
            end
            if iter == stop_lying_iter && ~initial_solution
                P = P ./ 4;
            end
            
            % Print out progress
            if ~rem(iter, 500)
                cost = const - sum(P(:) .* log(Q(:)));
                disp(['Iteration ' num2str(iter) ': error= ' num2str(cost)]);
                
                % Display scatter plot (maximally first three dimensions)
                str = ['Result of tSNE: ', num2str(iter), ' iterations'];
                if ~isempty(labels)
                    if no_dims == 1
                        scatter(ydata, ydata, 15, labels, 'filled');  title(str);
                    elseif no_dims == 2
                        scatter(ydata(:,1), ydata(:,2), 15, labels, 'filled');  title(str);
                    else
                        scatter3(ydata(:,1), ydata(:,2), ydata(:,3), 15, labels, 'filled'); title(str);
                    end
                    colormap(jet)
                    colorbar
                    axis equal tight off
                    drawnow
                end
            end         
        end
       
        
    case {'GPU', 'gpu'}
        % Make sure P-vals are set properly
        P_GPU = gpuArray(P);                                % set diagonal to zero
        P_GPU = (P_GPU + P_GPU')/2 ;                        % symmetrize P-values
        P_GPU = max(P_GPU ./ sum(P_GPU(:)), realmin);       % make sure P-values sum to one
        
        % disp(['max(P-P_GPU)=', num2str(max(max(P - P_GPU)))] ); % check GPU=CPU
        const_GPU = sum(P_GPU(:) .* log(P_GPU(:)));         % constant in KL divergence
        
        if ~initial_solution
            P_GPU = P_GPU * 4;                              % lie about the P-vals to find better local minima
        end
        
        % Initialize the solution
        if ~initial_solution
            ydata_GPU = .0001 * randn(n, no_dims, 'gpuArray');
        end
        y_incs_GPU  = zeros(size(ydata_GPU), 'gpuArray');
        gains_GPU = ones(size(ydata_GPU), 'gpuArray');
        
        % Run the iterations
        for iter=1:max_iter
            % Compute joint probability that point i and j are neighbors
            sum_ydata_GPU = sum(ydata_GPU .^ 2, 2);
            num_GPU = 1 ./ (1 + bsxfun(@plus, sum_ydata_GPU, bsxfun(@plus, sum_ydata_GPU', -2 * (ydata_GPU * ydata_GPU')))); % Student-t distribution                                             % set diagonal to zero
            num_GPU(1:n+1:end) = 0;
            Q_GPU = max(num_GPU ./ sum(num_GPU(:)), realmin);    % normalize to get probabilities
            
            % Compute the gradients (faster implementation)
            L_GPU = (P_GPU - Q_GPU).* num_GPU;
            y_grads_GPU = 4 * (diag(sum(L_GPU, 1)) - L_GPU) * ydata_GPU;
            
            % Update the solution
            gains_GPU = (gains_GPU + .2) .* (sign(y_grads_GPU) ~= sign(y_incs_GPU)) ...         % note that the y_grads are actually -y_grads
                + (gains_GPU * .8) .* (sign(y_grads_GPU) == sign(y_incs_GPU));
            gains_GPU(gains_GPU < min_gain) = min_gain;
            
            y_incs_GPU = momentum * y_incs_GPU - epsilon * (gains_GPU .* y_grads_GPU);
            ydata_GPU = ydata_GPU + y_incs_GPU;
            ydata_GPU = bsxfun(@minus, ydata_GPU, mean(ydata_GPU, 1));
            
            % find distant points & pull back
            sum_ydata_GPU = real(sqrt(sum(ydata_GPU .^ 2, 2)));
            
            % pos = find( sum_ydata_GPU > 3000);
            % ydata_GPU(pos,:) = ydata_GPU(pos,:) / 3;
            
            % Update the momentum if necessary
            if iter == mom_switch_iter
                momentum = final_momentum;
            end
            if iter == stop_lying_iter && ~initial_solution
                P_GPU = P_GPU ./ 4;
            end
            
            % Print out progress
            if ~rem(iter, 1000)
                cost_GPU = const_GPU - sum(P_GPU(:) .* log(Q_GPU(:)));
                disp(['Iteration ' num2str(iter) ': error= ' num2str(cost_GPU)]);
                ydata = gather(ydata_GPU);
                % Display scatter plot (maximally first three dimensions)
                str_GPU = ['Result of tSNE (GPU): ', num2str(iter), ' iterations'];
                if ~isempty(labels)
                    if no_dims == 1
                        scatter(ydata, 15, labels, 'filled');  title(str_GPU);
                    elseif no_dims == 2
                        scatter(ydata(:,1), ydata(:,2), 15, labels, 'filled');  title(str_GPU);
                    else
                        scatter3(ydata(:,1), ydata(:,2), ydata(:,3), 15, labels, 'filled'); title(str_GPU);
                    end
                    colormap(jet)
                    colorbar
                    axis equal tight off
                    drawnow
                end
            end
            
        end
        
    otherwise
        error('Please select computation type: CPU or GPU');
end
end

