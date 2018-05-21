function ydata = pSNE_p(P, labels, no_dims, beta, CGPU_type)
% pSNE_P Performs dimension reduction of power-law distribution with data affinity matrix P
%
%   mappedX = pSNE_p(P, labels, no_dims)
%
% The function performs symmetric p-SNE with pairwise similarity matrix P
% to create a low-dimensional map of no_dims dimensions (default = 2).
% The matrix P is assumed to be symmetric, sum up to 1, and have zeros
% on the diagonal.
% The labels of the data are not used by p-SNE itself, however, they
% are used to color intermediate plots. Please provide an empty labels
% matrix [] if you don't want to plot results during the optimization.
% The low-dimensional data representation is returned in mappedX.
%
% Code implemented based on Laurens van der Maaten, Delft University of Technology

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
final_momentum = 0.95;                              % value to which momentum is changed
mom_switch_iter = 200;                              % iteration at which momentum is changed
stop_lying_iter = 150;                              % iteration at which lying about P-values is stopped
max_iter = 20000;                                   % maximum number of iterations
epsilon = 180;                                      % initial learning rate
min_gain = .001;                                   % minimum gain for delta-bar-delta
dgain = 0.1;                                        % increment in gain
r_crit = 0.1;                                       % critical radius of Pareto's distribution
r0 = 0.2;                                           % initial random radius
g = 4;                                              % lying P-factor, default = 4

% Save file path
save_path = 'C:\Users\HTseng\Google Drive\Dell Desktop\MNIST_KL_N3000_pSNE';

switch CGPU_type
    case {'CPU', 'cpu'}
        % Make sure P-vals are set properly
        P(1:n + 1:end) = 0;                                 % set diagonal to zero
        P = 0.5 * (P + P');                                 % symmetrize P-values
        P = max(P ./ sum(P(:)), realmin);                   % make sure P-values sum to one
        const = sum(P(:) .* log(P(:)));                     % constant in KL divergence
        if ~initial_solution
            P = P * g;                                      % lie about the P-vals to find better local minima
        end
        
        % Initialize the solution
        if ~initial_solution
            ydata = r0 * randn(n, no_dims);
        end
        y_incs  = zeros(size(ydata));
        gains = ones(size(ydata));
        
        % Run iterations for minimizing loss function
        for iter=1:max_iter
            
            % Compute joint probability that point i and j are neighbors
            sum_ydata = sum(ydata .^ 2, 2);
            
            % Definfe sqnum_{ij} = ( ||x_i - x_j ||^2 ) = r_{ij}^2
            sqnum = bsxfun(@plus, sum_ydata, bsxfun(@plus, sum_ydata', -2 * (ydata * ydata')));
            
            % For r < r_crit, take ( ||x_i - x_j ||^2 ) = r_{ij}^2 =0
            sqnum(sqnum < r_crit ) = 0;
            
            % Define num = ( ||x_i - x_j ||)^{-m} = 1/r^m
            num =  1 ./ ( (sqrt(sqnum) ).^beta ); % (spherical) exit distribution
            
            % For r < r_crit, take (1 / ||x_i - x_j ||^m) = 1 / r_{ij}^m = 0
            num(sqnum < r_crit ) = 0;  num(1:n+1:end) = 0;
            Q = max(num ./ sum(num(:)), realmin);  % normalize to get probabilities Q_{ij}
            
            % Compute the gradients (faster implementation)
            L = (P - Q) ./ sqnum ;
            
            % For r < r_crit, take L = 0
            L(sqnum < r_crit ) = 0; L(1:n+1:end) = 0;  % set diagonal to 0
            y_grads = (2 * beta) * (diag(sum(L, 2)) - L) * ydata;
            
            % Update the solution
            gains = (gains + dgain) .* (sign(y_grads) ~= sign(y_incs)) ...         % note that the y_grads are actually -y_grads
                + (gains * .8) .* (sign(y_grads) == sign(y_incs));
            
            gains(gains < min_gain) = min_gain;
            y_incs = momentum * y_incs - epsilon * (gains .* y_grads);
            ydata = ydata + y_incs;
            ydata = bsxfun(@minus, ydata, mean(ydata, 1));           
            
            % Update the momentum if necessary
            if iter == mom_switch_iter
                momentum = final_momentum;
            end
            if iter == stop_lying_iter && ~initial_solution
                P = P ./ g;
            end
            
            % Print out progress
            if ~rem(iter, 1000)
                cost = const - sum(P(:) .* log(Q(:)));
                disp(['Iteration ' num2str(iter) ': error= ' num2str(cost)]);
                
                % Display scatter plot (maximally first three dimensions)
                str = ['Result of PowerSNE: ', num2str(iter), ' iterations, \beta=', num2str(beta)];
                if ~isempty(labels)
                    if no_dims == 1
                        scatter(ydata, ydata, 15, labels, 'filled');  title(str);
                    elseif no_dims == 2
                        scatter(ydata(:,1), ydata(:,2), 15, labels, 'filled');  title(str);
                    else
                        scatter3(ydata(:,1), ydata(:,2), ydata(:,3), 15, labels, 'filled');% title(str);
                    end
                    colormap(jet)
                    colorbar
                    axis off equal tight
                    drawnow
                    
                end
            end
            
            if (iter > (max_iter/4) )
                if ~rem(iter, 200)
                    CM = sum(ydata, 1) / n ;
                    ydata_diff = ydata - repmat(CM, n, 1);
                    avg_dist = sqrt(sum(sum(ydata_diff.^2))/n ) ;
                    
                    % find distant points & pull back
                    ydata_CM_dist = real(sqrt(sum(ydata_diff .^ 2, 2)));
                    if ~isempty(find( ydata_CM_dist > (2 * avg_dist), 1))
                        pos = find( ydata_CM_dist > (2 * avg_dist) );
                        ydata(pos,:) = ydata(pos,:)/2;
                    end
                end
            end
            
            % save results
            if ~rem(iter, 1000)
                % save([save_path, '_beta=', num2str(beta), ' _iter', num2str(iter), '.mat'] , 'ydata', 'labels');
            end
        end
        
    case {'GPU', 'gpu'}
        % Make sure P-vals are set properly
        P_GPU = gpuArray(P);
        P_GPU(1:n + 1:end) = 0;                                     % set diagonal to zero
        P_GPU = 0.5 * (P_GPU + P_GPU');                             % symmetrize P-values
        P_GPU = max(P_GPU ./ sum(P_GPU(:)), realmin);               % make sure P-values sum to one
        const_GPU = sum(P_GPU(:) .* log(P_GPU(:)));                 % constant in KL divergence
        if ~initial_solution
            P_GPU = P_GPU * g;                                      % lie about the P-vals to find better local minima
        end
        
        % Initialize the solution
        if ~initial_solution
            ydata_GPU = r0 * randn(n, no_dims, 'gpuArray');
        end
        y_incs_GPU  = zeros(size(ydata_GPU), 'gpuArray');
        gains_GPU = ones(size(ydata_GPU), 'gpuArray');
        
        for iter=1:max_iter
            % Compute joint probability that point i and j are neighbors
            sum_ydata_GPU  = sum(ydata_GPU.^ 2, 2);
            % Definfe sqnum_{ij} = ( ||x_i - x_j ||^2 ) = r_{ij}^2
            sqnum_GPU = bsxfun(@plus, sum_ydata_GPU , bsxfun(@plus, sum_ydata_GPU', -2 * (ydata_GPU  * ydata_GPU')));
            
            % For r < r_crit, take ( ||x_i - x_j ||^2 ) = r_{ij}^2 =0
            sqnum_GPU (sqnum_GPU  < r_crit ) = 0;
            
            % Define num = ( ||x_i - x_j ||)^{-m} = 1/r^m
            num_GPU  =  1 ./ ( (sqrt(sqnum_GPU) ).^beta ); % (spherical) exit distribution
            
            % For r < r_crit, take (1 / ||x_i - x_j ||^m) = 1 / r_{ij}^m = 0
            num_GPU (sqnum_GPU < r_crit ) = 0;  num_GPU(1:n+1:end) = 0;
            Q_GPU  = max(num_GPU  ./ sum(num_GPU (:)), realmin);  % normalize to get probabilities Q_{ij}
            
            % Compute the gradients (faster implementation)
            L_GPU  = (P_GPU  - Q_GPU ) ./ sqnum_GPU  ;
            
            % For r < r_crit, take L = 0
            L_GPU(sqnum_GPU < r_crit ) = 0; L_GPU(1:n+1:end) = 0;  % set diagonal to 0
            y_grads_GPU = (2 * beta) * (diag(sum(L_GPU, 2)) - L_GPU) * ydata_GPU;
            
            % Update the solution
            gains_GPU = (gains_GPU + dgain) .* (sign(y_grads_GPU) ~= sign(y_incs_GPU)) ...         % note that the y_grads are actually -y_grads
                + (gains_GPU * .8) .* (sign(y_grads_GPU) == sign(y_incs_GPU));
            
            gains_GPU(gains_GPU < min_gain) = min_gain;
            y_incs_GPU = momentum * y_incs_GPU - epsilon * (gains_GPU .* y_grads_GPU);
            ydata_GPU = ydata_GPU + y_incs_GPU;
            ydata_GPU = bsxfun(@minus, ydata_GPU, mean(ydata_GPU, 1));

            % Update the momentum if necessary
            if iter == mom_switch_iter
                momentum = final_momentum;
            end
            if iter == stop_lying_iter && ~initial_solution
                P_GPU = P_GPU ./ g;
            end
            
            % Print out progress
            if ~rem(iter, 200)
                cost_GPU = const_GPU - sum(P_GPU(:) .* log(Q_GPU(:)));
                disp(['Iteration ' num2str(iter) ': error= ' num2str(cost_GPU)]);
                ydata = gather(ydata_GPU);
                % Display scatter plot (maximally first three dimensions)
                str_GPU = ['Result of PowerSNE (GPU): ', num2str(iter), ' iterations, \beta=', num2str(beta)];
                if ~isempty(labels)
                    if no_dims == 1
                        scatter(ydata, ydata, 15, labels, 'filled');  title(str_GPU);
                    elseif no_dims == 2
                        scatter(ydata(:,1), ydata(:,2), 15, labels, 'filled');  title(str_GPU);
                    else
                        scatter3(ydata(:,1), ydata(:,2), ydata(:,3), 15, labels, 'filled'); title(str_GPU);
                    end
                    colormap(jet)
                    colorbar
                    axis off equal tight
                    drawnow
                end
            end
            
            if (iter > (max_iter/4) )
                if ~rem(iter, 200)
                    CM = sum(ydata_GPU, 1) / n ;
                    ydata_diff = ydata_GPU - repmat(CM, n, 1);
                    avg_dist = sqrt(sum(sum(ydata_diff.^2))/n ) ;
                    
                    % find distant points & pull back
                    ydata_CM_dist = real(sqrt(sum(ydata_diff .^ 2, 2)));
                    
                    if ~isempty(find( ydata_CM_dist > (2 * avg_dist), 1))
                        pos = find( ydata_CM_dist > (2 * avg_dist) );
                        ydata_GPU(pos,:) = ydata_GPU(pos,:)/2;
                    end
                end
            end
            
            % save results
            if ~rem(iter, 2000)
                % ydata = gather(ydata_GPU);
                %save([save_path, '_beta=', num2str(beta), '_iter', num2str(iter), '.mat'] , 'ydata_GPU', 'labels');
            end
        end
        
    otherwise
        error('Please select computation type: CPU or GPU');
end


end
