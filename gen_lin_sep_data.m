function [ data ] = gen_lin_sep_data()

    % target function (2 features, 1 label):
    % if x1 + x2 >= 1 => classify as 1
    % x1 + x2 < 1 classify as 0
    % this target function has no stochastic or deterministic noise
    
    % constant for how much data we will generate
    num_data = 1000;
    
    % generate 1000 numbers between 0 and 1 twice
    x1_vector = rand(num_data, 1);
    x2_vector = rand(num_data, 1);
    
    % if the sum is less than 1 classify as 0, else classify as 1
    labels = sign(x1_vector + x2_vector - 1);
    labels(labels == -1) = 0;
    
    % return the data
    data = [ x1_vector x2_vector labels];
    

    
end