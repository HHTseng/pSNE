function [mappedA, mapping] = compute_mapping(A, type, no_dims, varargin)
%COMPUTE_MAPPING Performs dimensionality reduction on a dataset A
%
%   mappedA = compute_mapping(A, type)
%   mappedA = compute_mapping(A, type, no_dims)
%   mappedA = compute_mapping(A, type, no_dims, ...)
%
% Performs a technique for dimensionality reduction on the data specified
% in A, reducing data with a lower dimensionality in mappedA.
% The data on which dimensionality reduction is performed is given in A
% (rows correspond to observations, columns to dimensions). A may also be a
% (labeled or unlabeled) PRTools dataset.

% The type of dimensionality reduction used is specified by type.
% Use: 'PCA','SNE', 'SymSNE', 'tSNE', 'pSNE', 'gSNE'

% The function returns the low-dimensional representation of the data in the
% matrix mappedA. If A was a PRTools dataset, then mappedA is a PRTools
% dataset as well. For some techniques, information on the mapping is
% returned in the struct mapping.
% The variable no_dims specifies the number of dimensions in the embedded
% space (default = 2).
%
%   mappedA = compute_mapping(A, type, no_dims, parameters)
%   mappedA = compute_mapping(A, type, no_dims, parameters, eig_impl)
%
% Free parameters of the techniques can be defined as well (on the place of
% the dots). These parameters differ per technique, and are listed below.
% For techniques that perform spectral analysis of a sparse matrix, one can
% also specify in eig_impl the eigenanalysis implementation that is used.
% Possible values are 'Matlab' and 'JDQR' (default = 'Matlab'). We advice
% to use the 'Matlab' for datasets of with 10,000 or less datapoints;
% for larger problems the 'JDQR' might prove to be more fruitful.
% The free parameters for the techniques are listed below (the parameters
% should be provided in this order):
%

%   SymSNE:         - <double> perplexity -> default = 30
%   tSNE:           - <int> initial_dims -> default = min(50, size(X,2))
%   pSNE:           - <int> initial_dims -> default = min(50, size(X,2))
%   gSNE:           - <int> initial_dims -> default = min(50, size(X,2))
%
% In the parameter list above, {.., ..} indicates a list of options, and []
% indicates the default setting. The variable k indicates the number of
% nearest neighbors in a neighborhood graph. Alternatively, k may also have
% the value 'adaptive', indicating the use of adaptive neighborhood selection
% in the construction of the neighborhood graph. Note that in LTSA and
% HessianLLE, the setting 'adaptive' might cause singularities. Using the
% JDQR-solver or a fixed setting of k might resolve this problem. SPE does
% not yet support adaptive neighborhood selection.
%
% The variable sigma indicates the variance of a Gaussian kernel. The
% parameters no_analyzers and max_iterations indicate repectively the number
% of factor analyzers that is used in an MFA model and the number of
% iterations that is used in an EM algorithm.
%
% The variable lambda represents an L2-regularization parameter.

% welcome;

% Check inputs
if nargin < 2
    error('Function requires at least two inputs.');
end
if ~exist('no_dims', 'var')
    no_dims = 2;
end
if ~isempty(varargin) && strcmp(varargin{length(varargin)}, 'JDQR')
    eig_impl = 'JDQR';
    varargin(length(varargin)) = [];
elseif ~isempty(varargin) && strcmp(varargin{length(varargin)}, 'Matlab')
    eig_impl = 'Matlab';
    varargin(length(varargin)) = [];
else
    eig_impl = 'Matlab';
end
mapping = struct;

% Handle PRTools dataset
if strcmp(class(A), 'dataset')
    prtools = 1;
    AA = A;
    if ~strcmp(type, {'LDA', 'FDA', 'GDA', 'KernelLDA', 'KernelFDA', 'MCML', 'NCA', 'LMNN'})
        A = A.data;
    else
        A = [double(A.labels) A.data];
    end
else
    prtools = 0;
end

% Make sure there are no duplicates in the dataset
A = double(A);
%     if size(A, 1) ~= size(unique(A, 'rows'), 1)
%         error('Please remove duplicates from the dataset first.');
%     end

% Check whether value of no_dims is correct
if ~isnumeric(no_dims) || no_dims > size(A, 2) || ((no_dims < 1 || round(no_dims) ~= no_dims) && ~any(strcmpi(type, {'PCA', 'KLM'})))
    error('Value of no_dims should be a positive integer smaller than the original data dimensionality.');
end

% Switch case
switch type
    case {'symSNE', 'sym-SNE'}
        % Compute t-SNE mapping
        if isempty(varargin), mappedA = tsne(A, [], no_dims);
        else mappedA = symsne(A, varargin{1}, no_dims, [], varargin{2}); end
        %              symsne(A, labels, no_dims, initial_dims, perplexity );
        mapping.name = 'sym-SNE';
        
    case {'tSNE', 't-SNE'}
        % Compute t-SNE mapping
        if isempty(varargin), mappedA = tsne(A, [], no_dims);
        else mappedA = tsne(A, varargin{1}, no_dims, varargin{2}, varargin{3}); end
        %              tsne(A,  labels,    no_dims,  initial_dims, perplexity );
        mapping.name = 't-SNE';
        
    case 'esSNE'
        % Compute sSNE mapping
        if isempty(varargin), mappedA = ssne(A, [], no_dims);
        else mappedA = esSNE(A, varargin{1}, no_dims, varargin{2}, varargin{3}, varargin{4}); end
        %              ssne(X, labels, no_dims, initial_dims, perplexity, rho)
        mapping.name = 'esSNE';
        
    case 'gSNE'
        % Compute arbitrary ditribution with arbitrary f-divergence
        if isempty(varargin), mappedA = gSNE(A, [], no_dims);
        else mappedA = gSNE(A, varargin{1}, no_dims, varargin{2}, varargin{3}, varargin{4}, varargin{5}, varargin{6} ); end
        %              gSNE(X, labels, no_dims, initial_dims, perplexity, method, alpha, beta)
        mapping.name = 'gSNE';
        
    case 'pSNE'
        % Compute arbitrary ditribution with arbitrary f-divergence
        if isempty(varargin), mappedA = pSNE(A, [], no_dims);
        else mappedA = pSNE(A, varargin{1}, no_dims, varargin{2}, varargin{3}, varargin{4}); end
        %              pSNE(X, labels, no_dims, initial_dims, perplexity, beta)
        mapping.name = 'pSNE';
    otherwise
        error('Unknown dimensionality reduction technique.');
end

% JDQR makes empty figure; close it
if strcmp(eig_impl, 'JDQR')
    close(gcf);
end

% Handle PRTools dataset
if prtools == 1
    if sum(strcmp(type, {'Isomap', 'LandmarkIsomap', 'FastMVU'}))
        AA = AA(mapping.conn_comp,:);
    end
    AA.data = mappedA;
    mappedA = AA;
end
