function [ train_err, test_err ] = RunAdaBoost(d_train, d_test)

   % clean the data so that outputs only contains 1 and 3
%    one = 1;
   three = 3;
   five = 5;
   
   % switch between five and one
   d_train_clean = d_train(d_train(:, 1) == five | d_train(:, 1) == three, :);
   d_test_clean = d_test(d_test(:, 1) == five | d_test(:, 1) == three, :);

   % separate the testing and training data into input and output
   x_train = d_train_clean(:, 2:size(d_train_clean, 2));
   y_train = d_train_clean(:, 1);
   
   x_test = d_test_clean(:, 2:size(d_test_clean, 2));
   y_test = d_test_clean(:, 1);
   
   % transforming 1 -> -1 and 3 -> 1 for classification simplification...
   % we need to compute a sign function, so it will be easier to do this
   for i = 1:size(y_train)
       if y_train(i) == five %  can also be one
           y_train(i) = -1;
       else
           y_train(i) = 1;
       end
   end
   
   for i = 1:size(y_test)
       if y_test(i) == five % can also be one
           y_test(i) = -1;
       else
           y_test(i) = 1;
       end
   end

   % Checking to make sure the dimensions of the data are correct
%    disp(size(x_train));
%    disp(size(y_train));
%    disp(size(x_test));
%    disp(size(y_test));

    n_experiments = 50;
    n_trees = 150;
    
    % run this once to get error sums that we can add to
    [ train_err_sum, test_err_sum ] = AdaBoost(x_train, y_train, x_test, y_test, n_trees);
    
    % we already one iteration before the for loop so that we can add to it
    % here
    for i = 2:n_experiments
        [ train_err_exp, test_err_exp ] = AdaBoost(x_train, y_train, x_test, y_test, n_trees);
        train_err_sum = train_err_sum + train_err_exp;
        test_err_sum = test_err_sum + test_err_exp;
        disp(i);
    end
    
    train_err = train_err_sum / n_experiments;
    test_err = test_err_sum / n_experiments;
    
    plotAdaBoostData(train_err, test_err);
   
end
