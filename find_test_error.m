function [ test_error ] = find_test_error( w, X, y )
%FIND_TEST_ERROR Find the test error of a linear separator
%   This function takes as inputs the weight vector representing a linear
%   separator (w), the test examples in matrix form with each row
%   representing an example (X), and the labels for the test data as a
%   column vector (y). X does not have a column of 1s as input, so that 
%   should be added. The labels are assumed to be plus or minus one. 
%   The function returns the error on the test examples as a fraction. The
%   hypothesis is assumed to be of the form (sign ( [1 x(n,:)] * w )


    % Piazza post said that we did not need E_test
    % https://piazza.com/class/jktyd725bej7gd?cid=235
    
    %disp("E_TEST");
    %z1 = (-1*y).*(X*w);
    %z2 = ones(size(X,1),1)*exp(1);
    %e_out_vector = log(z2.^z1 + 1);
    %disp(sum(e_out_vector) / size(X,1));
    
    disp("CLASSIFICATION_ERROR_OUT");
    prediction_vector = sigmf(X*w, [1 0]);
    correct = 0;
    for i = 1:size(X,1)
       if sign(prediction_vector(i) - .5) == y(i)
           correct = correct + 1;
       end
    end
    
    % percentage that the model predicted INCORRECTLY
    disp((size(X,1) - correct) / size(X,1));
    test_error = (size(X,1) - correct) / size(X,1);
    
end

