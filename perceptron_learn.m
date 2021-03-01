function [ w, iterations ] = perceptron_learn( data_in )
%perceptron_learn Run PLA on the input data
%   Inputs: data_in: Assumed to be a matrix with each row representing an
%                    (x,y) pair, with the x vector augmented with an
%                    initial 1, and the label (y) in the last column
%   Outputs: w: A weight vector (should linearly separate the data if it is
%               linearly separable)
%            iterations: The number of iterations the algorithm ran for


%   data_in takes form, input1 = (r, r, r, r, r, r, r, r, r, r, r, -1)
%                       input2 = (r, r, r, r, r, r, r, r, r, r, r, 1)
%                       input100 = (r, r, r, r, r, r, r, r, r, r, r, -1)
%   data_in has the dimensions nxm of 100x12, where the first column is 1
%   and last is the label of the input

% we will update randomly generated weights so that it fits the data
% perfectly -> there are NO errors and it is 100% linearly serpated
% correctly. We will have to iterate through the input using the error
% function update of w(t+1) = w(t) + y(t)*x(t), y is the correct label,
% and x is a training data vector in the data set

% w is a 11x1 vector that is randomly generated, and we multiply each
% training example (input1...input100) to the weights and update
% accordingly, w(1:11) = 0


% the input vector X is of 100x11
% weight vector is of 11x1
% label vector y is of 100x1
% can multiply xw to get a 100x1 and compare to y
% if doing this, and there is an uneqal d and y, how to choose which index
% to update?

% generate an intial zero weight vector to the length of the input x
weights = zeros(size(data_in,2)-1, 1);

% separate the output data y from the input_data
outputs = data_in(:, size(data_in, 2));

% separate the input vector x from the input_data
inputs = data_in(:, 1:size(data_in, 2)-1);

% while prediction is not equal to outputs iterate through the the vector
% of prediction and output (same size) and update the weight vector when
% the prediction does not equal the output label, and then go to the top
% and iterate through again until all values from prediction and output are
% equal, start with the assumption that prediction and outputs are not
% equal and that iterations to find the weight vector that generates the
% same outputs is initially zero
not_equal = 1;
iters = 0;

while not_equal
    % hypothesis prediction data
    prediction = sign(inputs * weights);
    for index = 1:size(prediction, 1)
       if(prediction(index) ~= outputs(index))
           % prediction was different then output -> run error function to
           % update the weight vector, add iter, and break out of the loop   
           weights = weights + outputs(index).*transpose(inputs(index,:));
           iters = iters + 1;
           break;
       end
       if(index == size(prediction, 1))
           % the prediction and output vectors are the same, so not_equal
           % becomes false so we can break out of the while loop
           not_equal = 0;
       end
   end
end

% set the return values of the function
w = weights;
iterations = iters;

end

