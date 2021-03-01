% 0 <= epsilon <= 1 -> iterate through 0 to 1
% iterate 0, 0.01, 0.02 ... 1.0 for epsilon (100 steps)
iterations = 100;

% iteration starts from zero so indexes are 1...101
bound_value_array = zeros(1, iterations + 1);
actual_prob_array = zeros(1, iterations + 1);
epsilon_index_array = zeros(1, iterations + 1);

for epsilon = 0 : 0.01 : 1
    % the actual probability and bound value for a given epsilon
    [ a, b ] = input_epsilon(epsilon); 
    % indexes need to be integers greater than zero
    index = int8(1 + epsilon * iterations);
    
    % putting the values associated with each epsilon in arrays to plot
    % after iterating through many different epsilons
    epsilon_index_array(index) = epsilon;
    actual_prob_array(index) = a;
    bound_value_array(index) = b;
end

% plotting both the epsilon(x) and bound(y1), and epsilon(x) and actual
% probability(y2)
plot(epsilon_index_array, bound_value_array, epsilon_index_array, actual_prob_array);
title("Epsilon and Actual Probability, Epsilon and Probability Bound");
xlabel("Epsilon");
ylabel("Probability and Bound");

function [ actual, bound ] = input_epsilon ( e )
    % need to solve for P(max(abs(v(i)-mu(i))) > epsilon)
    % givens 
    c = 2;
    N = 6;
    mu = 0.5;
    repetitions = 1000;

    % generating the trials 1000 times and averaging the number of times
    % that the sample error was > than the constant epsilon so that I can
    % plot this value on a graph
    sample_avg_error = zeros(1, repetitions);
    
    for avg = 1:repetitions
        sample_error = zeros(1, c);
        % generating my own binomial distribution data for a coin flip
        for coins = 1:c
            % 0 represents tails, 1 represents heads, so 50% for each as mu = 0.5
            flips = randi([0, 1], N, 1);
            heads = sum(flips);
            sample_error(coins) = (heads / N);
        end
    
        % max(abs(v(i)-mu(i)))
        max_sample_error = max(abs(sample_error - mu));
        
        % if the sample error is greater than constant epsilon then put a 1
        % into the array, otherwise put 0, so we can average the array
        % afterwards by summing it and dividing by length to approximate
        % the probability that it was greater than epsilon for a given
        % epsilon
        if max_sample_error > e
            sample_avg_error(avg) = 1;
        else
            sample_avg_error(avg) = 0;
        end
    end
    
    % returns the percent of 1000 repetitions that the max difference
    % between the sample and population error was greater than epsilon
    actual = sum(sample_avg_error) / size(sample_avg_error , 2);
    
    % bound function for P(max(abs(v(i)-mu(i))) > epsilon)
    % 4 * e ^ ( -12 * epsilon ^ 2 )
    bound = 4 * exp( -12 * ( e.^2 ));
end
