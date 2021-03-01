 function [ num_iters, bounds] = perceptron_experiment ( N, d, num_samples )
%perceptron_experiment Code for running the perceptron experiment in HW1
%   Inputs: N is the number of training examples
%           d is the dimensionality of each example (before adding the 1)
%           num_samples is the number of times to repeat the experiment
%   Outputs: num_iters is the # of iterations PLA takes for each sample
%            bound_minus_ni is the difference between the theoretical bound
%               and the actual number of iterations
%      (both the outputs should be num_samples long)

%N->100 training examples
%d->dimension of 11, the first index is a 0
%num_samples will be 1000

% we generate the INPUT_DATA in perceptron_experiment and we generate the
% WEIGHT_VECTOR in the perceptron_learn. All we need to do is generate the
% input data of form

%   data_in takes form, input1 = (1, r, r, r, r, r, r, r, r, r, r, -1)
%                       input2 = (1, r, r, r, r, r, r, r, r, r, r, 1)
%                       input100 = (1, r, r, r, r, r, r, r, r, r, r, -1)

% so data_in will be an 100x12, where the input vector x is the first 1-11 
% indexes and label y is the 12th index and x0 is 1

% perceptron_learn(generated_data) -> call this function to get the weight
% vector and the number of iterations it took to learn the weight vector,
% num_iters will be equal to iterations returned by the function

% w ideal will be randomly generated in this function, multiply the
% randomly generated input data by that ideal weight vector to get the y
% label set, append this to the input data so that we get an 100x12 vector
% to input, then we have a randomly generated weight vector in
% perceptron_learn that we are trying to TRAIN to learn the weights of our
% randomly generated ideal weight vector

list_iters = zeros(1, num_samples);
list_bounds = zeros(1, num_samples);

for i = 1:num_samples
    % randomly generate the IDEAL weights, d+1 x 1 -> 11x1
    w_ideal = rand(d+1, 1);
    w_ideal(1) = 0;

    % randomly generate data N x d+1 -> 100x11
    random_input = -1 + 2*rand(N, d+1);

    % generate label vector of Nx1 by multiplying random_input*w_ideal
    label_input = sign(random_input*w_ideal);

    % create the data_input by combining the random_input and label_input
    % and make the first column of the input augmented by one
    data_input = [random_input label_input];
    data_input(:,1) = 1;

    % pass the data_input into perceptron_learn and receive an output of the
    % weight vector that was learned in perceptron_learn and the number of
    % iterations it took to learn that weight vector
    [weight, iterations] = perceptron_learn(data_input);
    num_iters = iterations;
    list_iters(i) = num_iters;
    
    % argmin of a Nxd+1 and d+1x1, so rho is min of Nx1 dimensions
    rho = min((random_input*w_ideal).*label_input);
    
    % need to calculate the norm of every row in the input matrix
    max_input = max(sqrt(sum(random_input.^2, 2)));
    
    % calculating the norm of the weight vector
    weight_norm = norm(w_ideal);

    % calculating the theoretical bound and then putting it into a list
    bounds = ( (max_input.^2) .* (weight_norm.^2) ) / (rho.^2);
        
    % append the difference of the theoretical bound and actual number of
    % iterations to an array to plot the histogram later
    list_bounds(i) = bounds - iterations;
end

% histogram of iterations needed to completley linearly
% separate the data
histogram(list_iters);
title("Iterations needed to update weights to Linearly Separate the Data");
xlabel("Number of Iterations Needed");
ylabel("Frequency");

figure;

% histogram of the log of the difference between the theoretical bound and
% the actual number of iterations
histogram(log(list_bounds));
title("Histogram of Log of Difference between Theoretical Bound and Actual Iterations");
xlabel("Log of Difference");
ylabel("Frequency");

end

