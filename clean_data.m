function [ X, y, w ] = clean_data ( filename )

    % '/Users/Everwitt/Desktop/Roaming/Playground/train.csv'
    % '/Users/Everwitt/Desktop/Roaming/Playground/test.csv'
    data = csvread(filename);
    
    % generating a random w (of dimension d+1, or size(data,2))
    w = rand(size(data,2), 1)*2 - 1;
    %w(1) = 0;

    % input file will be a directory path to train / test csv
    % disp(size(data,2));
    
    ty = data(:,size(data,2));
   
    for index = 1 : size(ty,1)
        if ty(index) == 0
            ty(index) = -1;
        end
    end
    
    % disp(ty);
    
    % need to separate the last column and make replace the 0 with -1
    % need to augment the first column of X and add a column of 1s

    z = data(:,1:size(data,2)-1);
    t = ones(size(data,1), 1);
    tx = [ t, z ];
    
    X = tx;
    y = ty;

end


