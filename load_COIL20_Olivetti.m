function [X, labels] = load_COIL20_Olivetti(folder_name, dataset)
% This function loads images data COIL20 or Olivetti faces (dataset = 'COIL20' or 'Olivetti' )

switch dataset
    case {'COIL20', 'coil20'}
        % check existance of file
        if ~exist('imgFile', 'var') || isempty(folder_name)
            folder_name = './Dataset/coil-20-proc';
        end
        
        labels = [];
        % loading images into X
        for i=1:20  % number of objects
            for j=1:72  % photos of the object
                filename = [folder_name, '/obj' num2str(i) '__' num2str(j-1) '.png'];
                A = imread(filename);
                X((i-1)*72+j,:) = A(:)';
            end
            lab = i * ones(72,1);
            labels = [labels; lab] ;
        end
        X = double(X);  % converting uint into double
        
        % Shuffle the dataset N=1440
        c = randperm(1440);
        X = X(c,:);
        labels = labels(c);
        
        clear filename; clear A; clear i; clear j; clear lab; clear c;
        
        
    case {'Olivetti', 'olivetti'}
        % check existance of file
        if ~exist('imgFile', 'var') || isempty(folder_name)
            load('./Dataset/olivettifaces.mat');
        end
        
        filename = [folder_name, '/olivettifaces.mat'];
        load(filename);
        
        X = faces' ;
        X = double(X);  % double in [0,255] (uint8)
        % X = double(X)/255; % convert into double in [0,1]
        
        labels =[]; % making labels
        for i=1:40
            lab = i * ones(10,1);
            labels = [labels; lab] ;
        end
        
        % Shuffle the dataset N=400
        c = randperm(400);
        X = X(c,:);
        labels = labels(c);
        
        clear faces; clear lab; clear c;
        
    otherwise
        error('Please choose either "COIL20" or "Olivetti".');
end
