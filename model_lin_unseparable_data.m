function [ ] = model_lin_unseparable_data(x_train, y_train, x_test, y_test)

    % playing around with different kernels to classify this data
    SVMModel1 = fitcsvm(x_train, y_train,'KernelFunction','gaussian', 'ClassNames', {'0','1'});
    SVMModel2 = fitcsvm(x_train, y_train,'KernelFunction','linear', 'ClassNames', {'0','1'});
    SVMModel3 = fitcsvm(x_train, y_train,'KernelFunction','polynomial', 'PolynomialOrder', 3, 'ClassNames', {'0','1'}); 
    SVMModel4 = fitcsvm(x_train, y_train,'KernelFunction','polynomial', 'PolynomialOrder', 10, 'ClassNames', {'0','1'}); 
    
    [ label1, ~ ] = predict(SVMModel1, x_test);
    [ label2, ~ ] = predict(SVMModel2, x_test);
    [ label3, ~ ] = predict(SVMModel3, x_test);
    [ label4, ~ ] = predict(SVMModel4, x_test);
    
    gaussian_prediction = str2num(char(label1));
    linear_prediction = str2num(char(label2));
    third_polynomial_prediction = str2num(char(label3));
    tenth_polynomial_prediction = str2num(char(label4));
    random_prediction = randi([0 1], size(y_test, 1), 1);
    
    gaussian_score = 0;
    linear_score = 0;
    third_polynomial_score = 0;
    tenth_polynomial_score = 0;
    random_score = 0;
    
    for index = 1:size(y_test, 1)
        if gaussian_prediction(index) ~= y_test(index)
            gaussian_score = gaussian_score + 1;
        end
        if linear_prediction(index) ~= y_test(index)
            linear_score = linear_score + 1;
        end
        if third_polynomial_prediction(index) ~= y_test(index)
            third_polynomial_score = third_polynomial_score + 1;
        end
        if tenth_polynomial_prediction(index) ~= y_test(index)
            tenth_polynomial_score = tenth_polynomial_score + 1;
        end
        if random_prediction(index) ~= y_test(index)
            random_score = random_score + 1;
        end
    end
    
    disp('Linearly Unseparable Data Classification Error Measures');
    
    disp('Classification Gaussian Kernel')
    disp(gaussian_score / size(gaussian_prediction, 1));
    
    disp('Classification Linear Kernel')
    disp(linear_score / size(linear_prediction, 1));
    
    disp('Classification Third Order Polynimal Kernel')
    disp(third_polynomial_score / size(third_polynomial_prediction, 1));
    
    disp('Classification Tenth Order Polynimal Kernel')
    disp(tenth_polynomial_score / size(tenth_polynomial_prediction, 1));
    
    disp('Classification Random Prediction')
    disp(random_score / size(random_prediction, 1));

end