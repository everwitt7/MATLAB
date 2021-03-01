function [ ] = run_gradient_descent()

    train_file = '/Users/Everwitt/Desktop/Roaming/Playground/train.csv';
    test_file = '/Users/Everwitt/Desktop/Roaming/Playground/test.csv';
    
    [ x_train, y_train, w_train ] = clean_data(train_file);
    [ x_test, y_test, ~ ] = clean_data(test_file);
    
    % cleaning data for glmfit
    glm_x_train = x_train;
    glm_x_train(:,1) = [];
    glm_y_train = y_train;
    
    for index = 1 : size( glm_y_train, 1 )
        if glm_y_train(index) == -1
            glm_y_train(index) = 0;
        end
    end
    
    % normalizing data for logistic regression
    x_train_normalized = x_train;
    x_test_normalized = x_test;
    
    for index = 1 : size( x_train_normalized, 2 )
        x_train_normalized(:,index) = zscore(x_train_normalized(:,index));
        x_test_normalized(:,index) = zscore(x_test_normalized(:,index));
    end
    
    % iterations take values: 10000, 100000, 1000000
    max_iters = 10000;
    
    % learning rate takes random values: 0.00001
    % gradient minimized when lr = 0.005 for normalized descent
    learning_rate = .00001;
    
    % Running logistic gradient descent normally
    disp("logistic regression weight, e_in, and class_in, e_out, class_out");
    [ optimal_weights, ~ ] = logistic_reg( x_train, y_train, w_train, max_iters, learning_rate );
    [ ~ ] = find_test_error( optimal_weights, x_test, y_test );
    
    % Running the glmfit function
    disp("glmfit e_in, class_in, e_out, and class_out");
    w_glm = glmfit( glm_x_train, glm_y_train, 'binomial' );
    
    disp("E_IN");
    z1 = ( -1 * y_train ) .* ( x_train * w_glm );
    z2 = ones( size( x_train, 1 ), 1 ) * exp(1);
    e_in_vector = log( z2 .^ z1 + 1 );
    disp( sum(e_in_vector) / size( x_train, 1 ) );
    
    disp("CLASSIFICATION_ERROR_IN");
    prediction_vector = sigmf( x_train * w_glm, [1 0] );
    correct = 0;
    for i = 1:size( x_train, 1 )
       if sign( prediction_vector(i) - .5 ) == y_train(i)
           correct = correct + 1;
       end
    end
    disp( (size( x_train, 1 ) - correct) / size( x_train, 1 ) );
    
    [ ~ ] = find_test_error( w_glm, x_test, y_test );
    
    % Running logistic gradient descent after normalizing the data
    disp("normalized logistic regression weight, e_in, and class_in, e_out, class_out");
    [ normalized_weights, ~ ] = logistic_reg( x_train_normalized, y_train, w_train, max_iters, learning_rate );
    [ ~ ] = find_test_error( normalized_weights, x_test_normalized, y_test );
    
end