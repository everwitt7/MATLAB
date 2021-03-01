%Script that runs the set of necessary experiments

Q_f = 5:5:20; % Degree of true function
N = 40:40:120; % Number of training examples
var = 0:0.5:2; % Variance of stochastic noise

expt_data_mat_mean = zeros(length(Q_f), length(N), length(var));
expt_data_mat_med = zeros(length(Q_f), length(N), length(var));

for ii = 1:length(Q_f)
    for jj = 1:length(N)
        for kk = 1:length(var)
            overfit_measure = computeOverfitMeasure(Q_f(ii),N(jj),1000,var(kk),500);
            expt_data_mat_mean(ii,jj,kk) = mean(overfit_measure);
            expt_data_mat_med(ii,jj,kk) = median(overfit_measure);
        end
    end
    fprintf('.');
end

% plotting / graphing the data
% how does the overfit measure change as Q_f/N/var increases is the questions 
% that we want to answer -> "how does our overfit measure vary as the function
% of the complexity of the true hypothesis, number of training examples, and 
% the level of stochastic noise"
generate_plots(Q_f, N, var, expt_data_mat_mean, expt_data_mat_med);
            