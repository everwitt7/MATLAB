function [ w, e_in ] = logistic_reg( X, y, w_init, max_its, eta )
%LOGISTIC_REG Learn logistic regression model using gradient descent
%   Inputs:
%       X : data matrix (without an initial column of 1s)
%       y : data labels (plus or minus 1)
%       w_init: initial value of the w vector (d+1 dimensional)
%       max_its: maximum number of iterations to run for
%       eta: learning rate
    
%   Outputs:
%       w : weight vector
%       e_in : in-sample error (as defined in LFD)


    % hypothesis takes the form of
    % h(x) = Theta(wT*x), where Theta is the sigmoid function (range 0-1)
    % can just use the function, probability = sigmoid(dot product)
    % when probability is >= 0.5 predict 1, < .5 predict -1

    % while loop of maximum number of iterations
    % break out of the loop if the gradient is .001
    
    % error function is cross entropy error for e_in
    % e_in = (1/n) Summation(i-n): ln(1 + exp(-y(i)wTx(i)
    % classification error -> # predicted correctly / # total inputs
    
    % calculate the gradient in order to update the weight values
    % gradient will be a vector (same dimension as the weight vector d+1)
    % gradient(e_in) = (1/n) Summation(i-n): (-y(i)x(i)) / (exp(y(i)wTx(i)) + 1)
   
    % updating the weights: w(t+1) = w(t) - eta*gradient
    
    
    % IMPORTANT NOTES FOR HOW TO TRANSLATE EQUATION TO MY IMPLEMENTATION
    % ***** !!!!!! ******
    % X is (nxd+1), y is (nx1), w is (d+1x1)
    % a ROW in X represents one INPUT (n represents the specific input)
    
    % this means to get the temp_input to put into the sigmoid function,
    % for a specific input 1<=i<=n, X(i)*w -> (1xd+1)*(d+1x1) = 1 so will
    % have to just replace "wT*x" with "x*w"
    
    % TESTING CALCULATING THE VECTOR OF E_IN BASED ON THE RANDOM W_INIT
    % WHERE e_in_vector is an (nx1) matrix
%     z1 = (-1*y).*(X*w_init);
%     z2 = ones(size(X,1),1)*exp(1);
%     e_in_vector = log(z2.^z1 + 1);
    
%     disp(sum(e_in_vector) / size(X,1));
    
    % compare to for rows in size(X,1), do x(i)*w .* y(i) and sum
%     error = 0;
%     for index = 1:size(X, 1)
%         error = error + log(exp( -1 * y(index) .* X(index,:) * w_init) + 1);
%     end
%     error = error / size(X, 1);
%     disp(error);
    
    % THESE TWO VALUES ARE EQUAL... GOOD

    % TESTING CALCULATING THE AVG GRADIENT VECTOR USING A FOR LOOP RATHER
    % THAN TRYING TO VECTORIZE THE PROCESS
    % Gradient = (1/n) * SUMMATION(1->n) { -y(i).*X(i) / exp(y(i).*X(i)W }
    % each iteration of the loop yields a vector...
    % TOP : -y(i).*X(i) -> (nx1).*(nxd+1) => (nxd+1)
    % BOTTOM : exp(y(i).*X(i)W will be a number for same reasons as the
    % above test of calculating E(in)

%     ggg = zeros(1, size(X,2));
%     for i = 1:size(X, 1)
%         top = -1 * y(i) * X(i,:);
%         bottom = exp( y(i) .* X(i,:) * w_init ) + 1;
%         t_vect = top / bottom;
%         ggg = ggg + t_vect;
%     end
%     ggg = ggg / size(X,1);
%     disp("Gradient Summation");
%     disp(ggg)
    
    % TESTING CALCULATING THE AVG GRADIENT VECTOR (BECAUSE USING BATCH
    % INSTEAD OF STOCHASTIC GRADIENT DESCENT), SO WILL ORIGINALL HAVE A
    % (nxd+1) MATRIX BUT WILL AVERAGE EACH INPUT AND GET (1xd+1)

%     z1 = (y).*(X*w_init);
%     z2 = ones(size(X,1),1)*exp(1);
%     z3 = -1 * y .* X;
%     z4 = z2.^z1 + 1;
%     z5 = z3 ./ z4;
%     z6 = mean(z5, 1);
%     disp("Gradient Vectorization");
%     disp(z6);

    % THESE TWO VALUES ARE EQUAL... GOOD
    
    % The above comments were me just messing around to improve efficiency
    % by computing E_in and Gradient_w using vectors instead of for loops
    
    it = 1;
    threshold = 0.001;
    
    while it < max_its
       
        % calculate the gradient         
        z1 = (y).*(X*w_init);
        z2 = ones(size(X,1),1)*exp(1);
        z3 = -1 * y .* X;
        z4 = z2.^z1 + 1;
        z5 = z3 ./ z4;
        gradient = mean(z5, 1);
        
        % check if the gradient is less than the threshold, if so break
        if max(abs(gradient)) < threshold
            disp("Gradient Minimized");
            disp(it);
            break;
        end
        
        % if not, update the weight vector and increase the iteration
        w_init = w_init - (eta * transpose(gradient));
        it = it + 1;
        
        if it == max_its - 1
            disp("Max Iterations");
        end
        
    end
    
%     disp("learned weight vector");
%     disp(transpose(w_init));
    
    disp("E_IN");
    z1 = (-1*y).*(X*w_init);
    z2 = ones(size(X,1),1)*exp(1);
    e_in_vector = log(z2.^z1 + 1);
    disp(sum(e_in_vector) / size(X,1));
    
    disp("CLASSIFICATION_ERROR_IN");
    prediction_vector = sigmf(X*w_init, [1 0]);
    correct = 0;
    for i = 1:size(X,1)
       if sign(prediction_vector(i) - .5) == y(i)
           correct = correct + 1;
       end
    end
    disp((size(X,1) - correct) / size(X,1));

    w = w_init;
    e_in = sum(e_in_vector) / size(X,1);
    
end

