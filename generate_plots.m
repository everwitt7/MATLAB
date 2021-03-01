function [ ] = generate_plots(Q_f, N, var, expt_data_mat_mean, expt_data_mat_med)

    % can hold variance to a constant and training examples to a constant
    % but can just change the values of the constant var/N to show how the
    % error measure changes with different constant var/N's

    % can just run through each constant because the dimension of the
    % expt_data_mat is only 4x3x5

    % Q_f = [ 5, 10, 15, 20 ]
    % N = [ 40, 80, 120 ]
    % var = [ 0, .5, 1, 1.5, 2 ]
    % tensor of the form (Q_f, N, var)
    
    orange = [0.9100, 0.4100, 0.1700];
    maroon = [0.6350, 0.0780, 0.1840];
    
    % Comparing OVERFIT MEASURE to Q_f, N, and var
    figure(1)
    hold on
    % N = 40, var = 0 -> y->overfit measure, x->Q_f
    plot(Q_f, expt_data_mat_mean(:,1,1), 'color', orange) 
    plot(Q_f, expt_data_mat_med(:,1,1), '--.','color', orange)
    
    % N = 80, var = 0 -> y->overfit measure, x->Q_f
    plot(Q_f, expt_data_mat_mean(:,2,1), 'r') 
    plot(Q_f, expt_data_mat_med(:,2,1), '--.r')
    
    % N = 120, var = 0 -> y->overfit measure, x->Q_f
    plot(Q_f, expt_data_mat_mean(:,3,1), 'b') 
    plot(Q_f, expt_data_mat_med(:,3,1), '--.b')
    
    % N = 40, var = 1 -> y->overfit measure, x->Q_f
    plot(Q_f, expt_data_mat_mean(:,1,3), 'g') 
    plot(Q_f, expt_data_mat_med(:,1,3), '--.g')
    
    % N = 80, var = 1 -> y->overfit measure, x->Q_f
    plot(Q_f, expt_data_mat_mean(:,2,3), 'y') 
    plot(Q_f, expt_data_mat_med(:,2,3), '--.y')
    
    % N = 120, var = 1 -> y->overfit measure, x->Q_f
    plot(Q_f, expt_data_mat_mean(:,3,3), 'c') 
    plot(Q_f, expt_data_mat_med(:,3,3), '--.c')
    
    % N = 40, var = 2 -> y->overfit measure, x->Q_f
    plot(Q_f, expt_data_mat_mean(:,1,5), 'k') 
    plot(Q_f, expt_data_mat_med(:,1,5), '--.k')
    
    % N = 80, var = 2 -> y->overfit measure, x->Q_f
    plot(Q_f, expt_data_mat_mean(:,2,5), 'm') 
    plot(Q_f, expt_data_mat_med(:,2,5), '--.m')
    
    % N = 120, var = 2 -> y->overfit measure, x->Q_f
    plot(Q_f, expt_data_mat_mean(:,3,5), 'color', maroon) 
    plot(Q_f, expt_data_mat_med(:,3,5), '--.','color', maroon)
    
    hold off

    % 3 * 6 = 18 lines! (but 15 of them are essentially the same)
    legend('Mean,N=40,var=0','Med,N=40,var=0','Mean,N=80,var=0','Med,N=80,var=0',...
        'Mean,N=120,var=0','Med,N=120,var=0','Mean,N=40,var=1','Med,N=40,var=1',...
        'Mean,N=80,var=1','Med,N=80,var=1','Mean,N=120,var=1','Med,N=120,var=1',...
        'Mean,N=40,var=2','Med,N=40,var=2','Mean,N=80,var=2','Med,N=80,var=2',...
        'Mean,N=120,var=2','Med,N=120,var=2');
    
    xlabel('Q_f, Order of Polynomial Function');
    ylabel('Overfit Error Difference');
    title('Overfit Error Measure - Dotted Lines(Median), Solid Lines(Mean)');
    % This graph really shows how important the training set size is,
    % because the tenth order polynomial would overfit the data a lot when
    % there was little data to train off of, but even when the var
    % increased the median and mean overfit measures were similar
    
    % if the stochastic noise truly affected the overfit error, than the
    % lines would not all be the same of med/mean ~ 0, instead there should
    % be distinct lines with different overfit measures that cluster around
    % the same value, but we did not see that. We normalized the value of a
    % so that we would ONLY MEASURE DETERMINISTIC and NOT STOCHASTIC noise,
    % so this is exactly what we wanted to see!
    
end