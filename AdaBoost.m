function [ train_err, test_err ] = AdaBoost( X_tr, y_tr, X_te, y_te, n_trees )
%AdaBoost: Implement AdaBoost using decision stumps learned
%   using information gain as the weak learners.
%   X_tr: Training set
%   y_tr: Training set labels
%   X_te: Testing set
%   y_te: Testing set labels
%   n_trees: The number of trees to use

    % we want to calculate alpha, train_error, and test_error to be vectors
    % of length n_trees so that we can calculate the respective values for
    % all i = 1 to n_trees, then we can report the final element as just
    % the value of the last index in the list, and we can plot AdaBoost as
    % a function of n_trees
    alpha = zeros(n_trees, 1);
    train_err = zeros(n_trees, 1);
    test_err = zeros(n_trees, 1);
    
    % D is going to be a 1xn matrix where n is the number of training data
    % that we have, and we are going to initialize D(0) to just be equally
    % distributed across all of the inputs. We are going to update D at
    % every iteration from t=0 to n_trees, but we do not need to store
    % previous values of D because we only use D to calculate alpha(t), and 
    % we will store that value and the weak learner, h(t), because we need
    % those two pieces of information in order to calculate the output
    % prediction for the testing and training data so that we can compute
    % what the error was for both of those data sets, respectivley
    D = ones(size(X_tr, 1), 1) / size(X_tr, 1);
    
    % for t = 1,...,T in this case T = n_trees, we do the following
    % train the weak learner using distribution D(T), where D represents
    % the weight of the distribution on round t
    
    % Rather than storing the models for each t because we would then need
    % to compute H(x) = sign ( summation(t=1,T) { alpha(t) * h(t) } ), we
    % would need to compute h(1) 100 times, so it is better to treat this
    % as a dynamic programming problem and save computation by storing
    % recomputed subproblems in a matrix. We will call this matrix
    % hypothesis outputs, and this will be a nxT matrix, where n represents
    % the number of training data inputs that we have... and we will do
    % this for both the training and the testing data!!!
    dynamic_training_matrix = zeros(size(X_tr, 1), n_trees);
    dynamic_testing_matrix = zeros(size(X_te, 1), n_trees);
    
    for t = 1:n_trees
        % generate the indices from the training sample using replacement
        % that we will be using at step t in order to calculate the error
        % measure and update the weight vector D
        sample_indexes = datasample(1:size(X_tr,1), size(X_tr,1));
        
        % get the weak hyptothesis h(t): X -> { -1, +1 }
        weak_learner = fitctree(X_tr(sample_indexes, :),  y_tr(sample_indexes, :), 'SplitCriterion', 'deviance', 'Weights', D, 'MaxNumSplits', 1);

        % we need to calculate epsilon(t) in order to calculate alpha(t) so
        % that we can make a output prediction vector for testing and
        % training data... D is a (nx1) and the model is a (nx1), so taking
        % the dot product of the vectors does the same thing as running
        % through a for loop summation
        epsilon = transpose(D) * (predict(weak_learner, X_tr) ~= y_tr);
        alpha(t) = .5 * log((1 - epsilon) / epsilon); 
        
        % update the weight of inputs, D using the formula
        % D(t+1) = D(t)*exp(-alpha(t)*y(i)*h_t(i)) / Z(t)
        % Z(t) is the normalization factor and we can calculate it by 
        % Z(t) = 2*sqrt(epsilon*(1-epsilon)
        % we do not need to keep track of the previous iteration of D!
        Z = 2 * sqrt(epsilon * (1 - epsilon));

        % go through all inputs in the training data, compute the
        % prediction, multiply by the actual label, multiply by alpha,
        % divide by the normalization factor, in order to update... once
        % again this can be done as a vector instead of summation, 
        z1 = ones(size(X_tr, 1), 1) * exp(1);
        z2 = y_tr .* predict(weak_learner, X_tr);
        z3 = -1 * alpha(t) * z2;
        z4 = z1 .^ z3;
        z5 = D .* z4;
        D = z5 / Z;
        
        % compute the classification prediction for both train and test
        % output using the model, calculate the % that was incorrect, store
        % this value in train_err and test_err at index t, return a
        % complete list of train/test_err i=0,...,n_trees, plot in RunAda
        dynamic_training_matrix(:, t) = predict(weak_learner, X_tr);
        dynamic_testing_matrix(:, t) = predict(weak_learner, X_te);
        
    end
   
    % each COLUMN in dyanmic(...) represents every input in training and
    % testing data respectively, so in order to generate our prediction
    % output, we can take the dot product of our alpha vector, Tx1, and our
    % dyanimc, nxT, vector to get a nx1 prediction vector, and then we can
    % compare these predictions to our actual outputs to calculate the
    % error. We want to multiply each column(t) by element(t) in alpha
    training_matrix = transpose(alpha .* transpose(dynamic_training_matrix));
    testing_matrix = transpose(alpha .* transpose(dynamic_testing_matrix));
        
%     update the vector by summing t and t-1 from t=1 to T, so that we can
%     just take the sign at any t and that will be the prediction output
%     and we can compare this to the actual output to get the error
    for i = 2:n_trees
        training_matrix(:, i) = training_matrix(:, i) + training_matrix(:, i - 1);
        testing_matrix(:, i) = testing_matrix(:, i) + testing_matrix(:, i - 1);
    end
        
    for i = 1:n_trees
       training_output = sign(training_matrix(:, i));
       training_classification = sum(abs(training_output - y_tr)) / (2 * size(y_tr, 1));
       
       testing_output = sign(testing_matrix(:, i));
       testing_classification = sum(abs(testing_output - y_te)) / (2 * size(y_te, 1));
       
       train_err(i) =  training_classification;
       test_err(i) = testing_classification;
    end

end

