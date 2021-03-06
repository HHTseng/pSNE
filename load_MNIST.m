function [X, labels] = load_MNIST(imgFile, labelFile, readDigits, offset)
% This function loads MNIST, with output X and labels

if ~exist('imgFile', 'var') || isempty(imgFile)
    imgFile = './Dataset/MNIST/train-images-idx3-ubyte';
end
if ~exist('labelFile', 'var') || isempty(labelFile)
    labelFile = './Dataset/MNIST/train-labels-idx1-ubyte';
end
if ~exist('readDigits', 'var') || isempty(readDigits)
    readDigits = 1000;
end
if ~exist('offset', 'var') || isempty(offset)
    offset = 0;
end

% read MNIST images
[imgs, labels] = readMNIST(imgFile, labelFile, readDigits, offset);

X = zeros(readDigits, 400);
% Transform MNIST images (3D -> 2D)
for i=1:size(imgs,3)
    X(i,:) = reshape(imgs(:,:,i),[1 400]);
end

