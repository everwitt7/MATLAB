function [ overfit_m ] = computeOverfitMeasure( true_Q_f, N_train, N_test, var, num_expts )
%COMPUTEOVERFITMEASURE Compute how much worse H_10 is compared with H_2 in
%terms of test error. Negative number means it's better.
%   Inputs
%       true_Q_f: order of the true hypothesis
%       N_train: number of training examples
%       N_test: number of test examples
%       var: variance of the stochastic noise
%       num_expts: number of times to run the experiment
%   Output
%       overfit_m: vector of length num_expts, reporting each of the
%                  differences in error between H_10 and H_2

    overfit_m = zeros(num_expts, 1);
    
    % representing training and testing input and output:
    % x_train -> train(:,1), y_train -> train(:,2)
    % x_test -> test(:,1), y_test -> test(:,2)

    % iterate through the given number of experiments
    for i = 1:num_expts

        [ train, test ] = generate_dataset(true_Q_f, N_train, N_test, sqrt(var));
                
        % need to transform input to 2nd and 10th order polynomials to calculate glmfit
        % for both test and train data sets
        train_trans_second = computeLegPoly(train(:,1), 2);
        train_trans_tenth = computeLegPoly(train(:,1), 10);
        
        % transform the test input data
        test_trans_second = computeLegPoly(test(:,1), 2);
        test_trans_tenth = computeLegPoly(test(:,1), 10);
        
        % generating the 
        g_2 = glmfit(train_trans_second, train(:,2), 'normal', 'constant', 'off');
        g_10 = glmfit(train_trans_tenth, train(:,2), 'normal', 'constant', 'off');
        
        % below is with respect to the 2nd order polynomial and #exp=100
        % 3x1->GLM and 100x3->input => input*glm = 100x1 output prediction
        % test_trans_second * h_2 will be the prediction vector, so would
        % compute that and then compare it to the actual value vector and
        % take the squared difference to compute squared error
        prediction_2 = test_trans_second * g_2;
        prediction_10 = test_trans_tenth * g_10;
        
        % calculate respective error for 2nd and 10th order polys
        e_out_2 = ( ( prediction_2 - test(:, 2) ) .^ 2 );
        e_out_10 = ( ( prediction_10 - test(:, 2) ) .^ 2 );
        
        % calculate the difference in error for 2nd and 10th order polys
        overfit_m(i) = mean(e_out_10) - mean(e_out_2);
        
    end
    
end