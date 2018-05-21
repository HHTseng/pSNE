function ydata = gSNE_p(P, labels, no_dims, method, alpha, beta, CGPU_type)
%  gSNE_P Performs gSNE, arbitrary probability distribution Q with f-div, on affinity matrix P
%
%   mappedX = gSNE_p(P, labels, no_dims)
%
% The function performs gSNE, arbitrary probability distribution with f-div, on pairwise similarity matrix P
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
final_momentum = 0.98;                              % value to which momentum is changed
mom_switch_iter = 30;                              % iteration at which momentum is changed
stop_lying_iter = 10;                              % iteration at which lying about P-values is stopped
max_iter = 300000;                                  % maximum number of iterations
epsilon = 10;                                        % initial learning rate
min_gain = .00001;                                    % minimum gain for delta-bar-delta
m = method;                                         % mixture coefficient: m*pSNE + (1-m)* tSNE [psne -> 1; tsne -> 0]
r_crit = 0.1;                                       % critical radius of Pareto's distribution                                             % lying P-factor, default = 4
r0 = 0.3;                                            % initial random radius
g = 4;                                              % factor to lie about the P-vals to find better local minima
dgain = 0.05;                                        % increment in gain

save_path = '/home/thuanhsi/Dimension_reduction/Experiments/gSNE/MNIST_KL_N6000_gSNE';

switch CGPU_type
    case {'CPU', 'cpu'}
        % Make sure P-vals are set properly
        P(1:n + 1:end) = 0;                                 % set diagonal to zero
        P = 0.5 * (P + P');                                 % symmetrize P-values
        P = max(P ./ sum(P(:)), realmin);                   % make sure P-values sum to one
        
        if ~initial_solution
            P = P * g;                                      % lie about the P-vals to find better local minima
        end
        
        % Take f-div = alpha-div
        f = @(t) 4/(1-alpha^2) * ( (1- alpha)/2 + (1 + alpha)/2 * t - t.^((alpha+1)/2) );
        df = @(t) 2/(1 - alpha) *( 1 - t.^( (alpha-1)/2 ) );
        
        % Define target similarity on lower dimensions
        Q = @(x) ((x <= r_crit).* 1 + (x > r_crit).*(m ./ x.^beta + (1-m)* 1./(1+x.^2)) );  % Q = a* 1/r^s + (1-a)* t-SNE
        dQ = @(x) ((x <= r_crit).* 0 + (x > r_crit).*(- m*beta ./ x.^(beta+1) - (1-m)*2* x ./ (1+x.^2).^2)) ;   % d/dx Q(x)
        
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
            
            % sqr(sqr < r_crit ) = 0;     % For r < r_crit, take ||x_i - x_j ||^2 = r_{ij}^2 =0
            sqr = bsxfun(@plus, sum_ydata, bsxfun(@plus, sum_ydata', -2 * (ydata * ydata')));
            
            % with r_{ij} = ||y_i -  y_j || ; sqr_{ij} = ||y_i - y_j ||^2;
            r = real(sqrt(max(sqr,realmin)));
            %r(sqr < r_crit ) = 0;       % For r < r_crit, r=0
            
            Qr = Q(r);
            %Qr(sqr < r_crit ) = 0;       % For r < r_crit, Q(r)=0
            Qr(1:n+1:end) = 0;            % Q(r_ii)= 0
            
            dQr = dQ(r);
            %dQr(sqr < r_crit ) = 0;       % For r < r_crit, dQ(r)=0
            dQr(1:n+1:end) = 0;            % Q(r_ii)= 0
            
            Z = sum(Qr(:));        % Z = sum_{ab} Q(r_ab)
            q = max(Qr/ Z, realmin);    % target prob: q_ij = Q(r_ij) / Z
            
            % Compute the gradients (faster implementation)
            % dJ/dy_k = sum_i L_ki * (y_k -y_i)
            
            q_P = q./P;         % q_P(1:n+1:end) = 0;
            dfq_P = df(q./P);
            dfq_P(1:n+1:end) = 0;
            L = dQr .* ( dfq_P - (1/Z)* sum(sum(dfq_P .* Qr))  );
            % L(sqr < r_crit ) = 0;       % For r < r_crit, L(r)=0
            L(1:n+1:end) = 0;    % set diagonal to 0
            
            L_r = L./r;
            L_r(1:n+1:end) = 0;
            y_grads = 2/Z .* ( diag(sum(L_r , 1)) -  L_r) * ydata;
            %y_grads = ( diag(sum(L,2)) - L ) * ydata;
            
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
                P = P / g;
            end
            
            % Print out progress
            if ~rem(iter, 500)
                cost = P_GPU.* f(q_P);  cost(1:n+1:end) = 0;
                cost = sum(sum( cost ));
                disp(['Iteration ' num2str(iter) ': error is ' num2str(cost)]);
                
                % Display scatter plot (maximally first three dimensions)
                str = ['Result of gSNE (GPU): \alpha=', num2str(alpha), ' \beta=', num2str(beta), ', iteration: ' , num2str(iter)];
                if ~isempty(labels)
                    if no_dims == 1
                        scatter(ydata, ydata, 15, labels, 'filled'); title(str);
                    elseif no_dims == 2
                        scatter(ydata(:,1), ydata(:,2), 15, labels, 'filled'); title(str);
                    else
                        scatter3(ydata(:,1), ydata(:,2), ydata(:,3), 215, labels, 'filled'); title(str);
                    end
                    colormap(jet)
                    colorbar
                    axis equal tight
                    drawnow
                end
            end
            
            if (iter > 5000)
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
            
            % CPU save results
            if ~rem(iter, 2000)
             % save([save_path, '_m=', num2str(m), '_alpha=', num2str(alpha), '_beta=', num2str(beta), ' _iter', num2str(iter), '.mat'] , 'ydata', 'labels');
            end
        end
        
    case {'GPU', 'gpu'}
        % Make sure P-vals are set properly
        P_GPU = gpuArray(P);
        P_GPU(1:n + 1:end) = 0;                                         % set diagonal to zero
        P_GPU = 0.5 * (P_GPU + P_GPU');                                 % symmetrize P-values
        P_GPU = max(P_GPU ./ sum(P_GPU(:)), realmin);                   % make sure P-values sum to one
        % const_GPU = sum(P_GPU(:) .* log(P_GPU(:)));                     % constant in KL divergence
        if ~initial_solution
            P_GPU = P_GPU * g;                                          % lie about the P-vals to find better local minima
        end
        
        % Take f-div = alpha-div
        f = @(t) 4/(1-alpha^2) .* ( (1- alpha)/2 + (1 + alpha)/2 .* t - t.^((alpha+1)/2) );
        df  =@(t) 2/(1 - alpha) .*( 1 - t.^( (alpha-1)/2 ) );
        
        % Define target similarity on lower dimensions
        Q = @(r) ((r <= r_crit).* 1 + (r > r_crit).*(m ./ r.^beta + (1-m)* 1./(1+r.^2)) );  % Q = a* 1/r^s + (1-a)* t-SNE
        dQ = @(r) ((r <= r_crit).* 0 + (r > r_crit).*(- m*beta ./ r.^(beta+1) - (1-m)*2* r ./ (1 + r.^2).^2)) ;   % d/dx Q(x)
        
        % Q = @(r) ( exp(-r.^2) );  % Q = a* 1/r^s + (1-a)* t-SNE
        % dQ = @(r) (-2 .* r .* exp(-r.^2)) ;   % d/dx Q(x)
        
        % Initialize the solution
        if ~initial_solution
            ydata_GPU = r0 * randn(n, no_dims, 'gpuArray');
        end
        
        y_incs_GPU  = zeros(size(ydata_GPU), 'gpuArray');
        gains_GPU = ones(size(ydata_GPU), 'gpuArray');
        
        % Run iterations for minimizing loss function
        for iter=1:max_iter
            
            % Compute joint probability that point i and j are neighbors
            sum_ydata_GPU = sum(ydata_GPU .^ 2, 2);
            
            % sqr(sqr < r_crit ) = 0;     % For r < r_crit, take ||x_i - x_j ||^2 = r_{ij}^2 =0
            % sqr_GPU = bsxfun(@plus, sum_ydata_GPU, bsxfun(@plus, sum_ydata_GPU', -2 * (ydata_GPU * ydata_GPU')));
            
            % with r_{ij} = ||y_i -  y_j || ; sqr_{ij} = ||y_i - y_j ||^2;
            % r = real(sqrt(max(sqr_GPU,realmin)));
            r_GPU = sqrt(max( bsxfun(@plus, sum_ydata_GPU, bsxfun(@plus, sum_ydata_GPU', -2 * (ydata_GPU * ydata_GPU'))) , realmin));
            % where sqr_GPU = bsxfun(@plus, sum_ydata_GPU, bsxfun(@plus, sum_ydata_GPU', -2 * (ydata_GPU * ydata_GPU')));
            
            Qr = Q(r_GPU);
            % Qr( r < r_crit ) = 0;        % For r < r_crit, Q(r)=0
            Qr(1:n+1:end) = 0;             % Q(r_ii)= 0
            
            dQr = dQ(r_GPU);
            %dQr(sqr < r_crit ) = 0;       % For r < r_crit, dQ(r)=0
            dQr(1:n+1:end) = 0;            % Q(r_ii)= 0
            
            Z = sum(Qr(:));        % Z = sum_{ab} Q(r_ab)
            q_GPU = max(Qr/ Z, realmin);    % target prob: q_ij = Q(r_ij) / Z
            
            % Compute the gradients (faster implementation)
            % dJ/dy_k = sum_i L_ki * (y_k -y_i)
            
            q_P = q_GPU ./ P_GPU;   % q_P(1:n+1:end) = 0;
            dfq_P = df(q_P);  dfq_P(1:n+1:end) = 0;
            L = dQr .* ( dfq_P - (1/Z)* sum(sum(dfq_P .* Qr))  );
            % L(sqr < r_crit ) = 0;       % For r < r_crit, L(r)=0
            L(1:n+1:end) = 0;    % set diagonal to 0
            
            L_r = L./ r_GPU; L_r(1:n+1:end) = 0;
            y_grads_GPU = 2/Z .* ( diag(sum(L_r , 1)) -  L_r) * ydata_GPU;
            %y_grads = ( diag(sum(L,2)) - L ) * ydata;
            
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
            if ~rem(iter, 500)
                cost = P_GPU.* f(q_P);  cost(1:n+1:end) = 0;
                cost = sum(sum( cost ));
                disp(['Iteration ' num2str(iter) ': error is ' num2str(cost)]);
                ydata = gather(ydata_GPU);  %bring results back to CPU for plotting
                
                % Display scatter plot (maximally first three dimensions)
                str_GPU = ['Result of gSNE (GPU): \alpha=', num2str(alpha), ', \beta=', num2str(beta), ', iteration: ' , num2str(iter)];
                if ~isempty(labels)
                    if no_dims == 1
                        scatter(ydata, ydata, 15, labels, 'filled'); title(str_GPU);
                    elseif no_dims == 2
                        scatter(ydata(:,1), ydata(:,2), 15, labels, 'filled'); title(str_GPU);
                    else
                        scatter3(ydata(:,1), ydata(:,2), ydata(:,3), 15, labels, 'filled'); title(str_GPU);
                    end
                    colormap(jet)
                    colorbar
                    axis off equal tight
                    drawnow
                end
            end
            
            if (iter > (max_iter/10) )
                if ~rem(iter, 500)
                    CM = sum(ydata_GPU, 1) / n ;
                    ydata_diff = ydata_GPU - repmat(CM, n, 1);
                    avg_dist = sqrt( sum(sum(ydata_diff.^2))/n ) ;
                     
                    % find distant points & pull back
                    ydata_CM_dist = real(sqrt(sum(ydata_diff .^ 2, 2)));
                    
                    if ~isempty(find( ydata_CM_dist > (2 * avg_dist), 1))
                        pos = find( ydata_CM_dist > (2 * avg_dist) );
                        ydata_GPU(pos,:) = ydata_GPU(pos,:)/2;
                    end
                end
            end
            
            % GPU save results
            if ~rem(iter, 1000)
                % save([save_path, '_m=', num2str(m), '_alpha=', num2str(alpha), '_beta=', num2str(beta), ' _iter', num2str(iter), '.mat'] , 'ydata_GPU', 'labels');
            end
        end
        
    otherwise
        error('Please select computation type: CPU or GPU');
end

end
