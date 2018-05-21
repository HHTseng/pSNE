%% Set MNIST data location
clc; clear; close all;

% Set 3 dataset folders (may skip if you set by default)
% MNIST
MNIST_file = './Dataset/MNIST/train-images-idx3-ubyte';        % MNIST images
MNIST_label_file = './Dataset/MNIST/train-labels-idx1-ubyte';  % corresponding labels
% COIL-20
COIL20_folder = './Dataset/coil-20-proc';
% Olivetti faces
Olivetti_folder = './Dataset';

%% [Power-law SNE dimensionality reduction]

% % Load MNIST dataset
readDigits = 3000;   % load how many sample digits from MNIST
offset = 0;          % from where starting reding
[X, labels] = load_MNIST(MNIST_file, MNIST_label_file, readDigits, offset);

% Load COIL20 dataset
% [X, labels] = load_COIL20_Olivetti(COIL20_folder, 'COIL20'); % Use COIL20
% Load Olivetti-face dataset
% [X, labels] = load_COIL20_Olivetti(Olivetti_folder, 'Olivetti'); % Use Olivetti-faces

% Set pSNE parameters
no_dims = 2;         % (target) Dimension for data representation
beta = 2;            %  power-law exponent
S = 15;              %  marker size for plots
initial_dim = 50;    %  initial dim for SNE (after PCA reduction)
perplexity = 30;     %  detetermine size of data neighborhood

% Apply dimentionality reduction (output map: mappedX )
[mappedX, mapping] = compute_mapping(X, 'pSNE', no_dims, labels, initial_dim, perplexity , beta, 'GPU');

% plot results
figure
str = ['pSNE, \beta=', num2str(beta)];
if size(mappedX,2)== 2
    scatter(mappedX(:,1), mappedX(:,2), S , labels, 'filled'); title(str);
    colormap(jet); colorbar; axis off equal tight;
else
    scatter3(mappedX(:,1), mappedX(:,2), mappedX(:,3), S, labels, 'filled'); title(str);
    colormap(jet); colorbar; axis off equal tight;
end


  %% % [Generalized-SNE dimensionality reduction]

% Load MNIST dataset
readDigits = 3000;   % load how many sample digits from MNIST
offset = 0;          % from where starting reding
[X, labels] = load_MNIST(MNIST_file, MNIST_label_file, readDigits, offset);

% Load COIL20 dataset
% [X, labels] = load_COIL20_Olivetti(COIL20_folder, 'COIL20'); % Use COIL20
% % Load Olivetti-face dataset
% [X, labels] = load_COIL20_Olivetti(Olivetti_folder, 'Olivetti'); % Use Olivetti-faces

% Set gSNE parameters
no_dims = 2;        % (target) Dimension for data representation
method = 1;         % dimension reduction method = mixture coefficient: method*pSNE + (1-method)* tSNE [psne -> 1; tsne -> 0]
initial_dim = 50;   % initial dim for SNE (after PCA reduction)
perplexity = 30;    %  detetermine size of data neighborhood
alpha = -1.1;       % alpha-divergence [alpha= -1 (KL), alpha -> 0 (Hellinger) ]
beta = 2;           % power-law exponent (if choosing 'pSNE' in previous method)
S = 15;             %  marker size for plots

% Apply dimentionality reduction (output map: mappedX )
[mappedX, mapping] = compute_mapping(X, 'gSNE', no_dims, labels, initial_dim, perplexity , method, alpha, beta, 'GPU');

% plot results
figure
str = ['Result of gSNE using method=', num2str(method), 'with \alpha=', num2str(alpha), ', \beta=', num2str(beta)];
if size(mappedX,2)== 2
    scatter(mappedX(:,1), mappedX(:,2), S , labels, 'filled'); title(str);
    colormap(jet); colorbar; axis off equal tight;
else
    scatter3(mappedX(:,1), mappedX(:,2), mappedX(:,3), S, labels, 'filled'); title(str);
    colormap(jet); colorbar; axis off equal tight;
end


%% [tSNE dimensionality reduction]

% Load MNIST dataset
readDigits = 3000;   % load how many sample digits from MNIST
offset = 0;          % from where starting reding
[X, labels] = load_MNIST(MNIST_file, MNIST_label_file, readDigits, offset);

% % % select a digit
% X = X(labels==6, :); X = X(1:500, :);
% labels = labels(labels==6); labels = labels(1:500);

% Load COIL20 dataset
% [X, labels] = load_COIL20_Olivetti(COIL20_folder, 'COIL20'); % Use COIL20
% Load Olivetti-face dataset
% [X, labels] = load_COIL20_Olivetti(Olivetti_folder, 'Olivetti'); % Use Olivetti-faces

% Set tSNE parameters
no_dims = 2;        % (target) Dimension for data representation
initial_dim = 50;   % initial dim for SNE (after PCA reduction)
perplexity = 30;    %  detetermine size of data neighborhood
S = 15;             %  marker size for plots

% Apply dimentionality reduction (output map: mappedX )
[mappedX, mapping] = compute_mapping(X, 'tSNE', no_dims, labels, initial_dim, perplexity, 'GPU');

% plot results
figure
str = ['Result of tSNE'];
if size(mappedX,2)== 2
    scatter(mappedX(:,1), mappedX(:,2), S , labels, 'filled'); title(str);
    colormap(jet); colorbar; axis off equal tight;
else
    scatter3(mappedX(:,1), mappedX(:,2), mappedX(:,3), S, labels, 'filled'); title(str);
    colormap(jet); colorbar; axis off equal tight;
end


%% % [esSNE (spherical SNE) dimensionality reduction]

% Load MNIST dataset
readDigits = 2000;   % load how many sample digits from MNIST
offset = 0;          % from where starting reding
[X, labels] = load_MNIST(MNIST_file, MNIST_label_file, readDigits, offset);

% % Load COIL20 dataset
% [X, labels] = load_COIL20_Olivetti(COIL20_folder, 'COIL20'); % Use COIL20
% % Load Olivetti-face dataset
% [X, labels] = load_COIL20_Olivetti(Olivetti_folder, 'Olivetti'); % Use Olivetti-faces

% Set es-SNE parameters
no_dims = 2;        % (target) Dimension for data representation
initial_dim = 50;   %  initial dim for SNE (after PCA reduction)
perplexity = 30;    %  detetermine size of data neighborhood
rho = 0.95;         %  parameter of Exit distribution
S = 15;             %  marker size for plots

% Apply dimentionality reduction (output map: mappedX )
[mappedX, mapping] = compute_mapping(X, 'esSNE', no_dims, labels, initial_dim, perplexity , rho);

% plot results
figure
str = ['Result of eSSNE, \rho=', num2str(rho)];
if size(mappedX,2)== 2
    scatter(mappedX(:,1), mappedX(:,2), S , labels, 'filled'); title(str);
else
    scatter3(mappedX(:,1), mappedX(:,2), mappedX(:,3), S, labels, 'filled'); title(str);
    hold on
    % Attach a background sphere
    [x,y,z] = sphere;
    surf(x,y,z, 'FaceAlpha', 0.01,'FaceLighting','flat')
    hold off
end
axis equal tight
colormap(jet)
colorbar
