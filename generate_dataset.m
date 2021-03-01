function [ train_set, test_set ] = generate_dataset( Q_f, N_train, N_test, sigma )
%GENERATE_DATASET Generate training and test sets for the Legendre
%polynomials example
%   Inputs:
%       Q_f: order of the hypothesis
%       N_train: number of training examples
%       N_test: number of test examples
%       sigma: standard deviation of the stochastic noise
%   Outputs:
%       train_set and test_set are both 2-column matrices in which each row
%       represents an (x,y) pair

    % first column of train and test matrices
    x_train = 2*rand(N_train, 1) - 1;
    x_test = 2*rand(N_test, 1) - 1;
    
    % a(q=0 to Q+1 is generated independently from a standard normal dist)
    % epsilon_train, epsilon_test are iid standard normal variates
    epsilon_train = normrnd(0, 1, length(x_train), 1);
    epsilon_test = normrnd(0, 1, length(x_test), 1);
    a = normrnd(0, 1, Q_f+1, 1);
    
    % the normalization scalar will make the expected value of a,x of f^2
    % to be equal to 1
    normalization = 0;
    for i = 0:Q_f
        normalization = normalization + ( 1 / ( 1 + 2*i ) ); 
    end
    normalization = sqrt(normalization);
    
    % noise will be sigma * epsilon for test and train
    % f(x) will be a * z(legendre polys) * scalar(normalization)
    legendre_train = computeLegPoly(x_train, Q_f);
    legendre_test = computeLegPoly(x_test, Q_f);
    
    % a is a (Q+1) x (1)
    % leg_test/train is (length(x_train/test) x (Q+1)
    % normalization is just a scalar
    % leg_test/train * a * normalization -> (length(x_train/test) x (1)
    y_train = legendre_train * a * normalization + epsilon_train * sigma;
    y_test = legendre_test * a * normalization + epsilon_test * sigma;
    
    train_set = [ x_train, y_train ];
    test_set = [ x_test, y_test ];
    
end