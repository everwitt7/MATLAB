function [] = var_bias_experiment()
    
    % need to generate k datasets
    k = 1000;
    
    % g_D takes in two datapoints, x1 and x2
    % and the Hypothesis g_D = ax + b
    % where a = (x1 + x2) and b = (-1)(x1)(x2)
    
    % g_bar = average of all g_D... a1+a2+...+ak / k
    learned_a = zeros(k, 1);
    learned_b = zeros(k, 1);
    
    input_x1 = zeros(k,1);
    input_x2 = zeros(k,1);
    
    for i = 1:k

        x1 = rand() * 2 - 1;
        x2 = rand() * 2 - 1;

        a = x1 + x2;
        b = -1*x1*x2;
        
        learned_a(i) = a;
        learned_b(i) = b;
        input_x1(i) = x1;
        input_x2(i) = x2;

    end
    
    g_bar = zeros(2,1);
    g_bar(1) = mean(learned_a);
    g_bar(2) = mean(learned_b);
    
    % we know the target function is f(x) = x^2   
    % generate the data of the target function 
    input = zeros(1, 201);
    output = zeros(1, 201);
    output_bar = zeros(1, 201);
    
    % can also plot points for EVERY learned hypothesis in this loop
    % or can just print out the equation for g_bar
    for i = -1 : 0.01 : 1
        index = int16((i+1)*100 + 1);
        input(index) = i;
        output(index) = i*i;
        output_bar(index) = i*g_bar(1) + g_bar(2);
    end
    
    % bias = (g_bar - target)^2
    % var = (g_D - g_bar)^2
    % calculate bias and var to calculate E_out
    % iterate through the k generated datasets and compare g_D to g_bar
    var_array = zeros(k,1);
    bias_array = zeros(k,1);
    
    for i = 1:k
        
        x = rand() * 2 - 1;

        gD_val = learned_a(i) * x + learned_b(i);
        gBar_val = g_bar(1) * x + g_bar(2);
        
        diff = gD_val - gBar_val;
        var_array(i) = diff*diff;
        
        target_val = x*x;
        other_diff = gBar_val - target_val;
        bias_array(i) = other_diff*other_diff;
        
    end
    
    disp("Experimental Var Value");
    disp(mean(var_array));
    
    disp("Experimental Bias Value");
    disp(mean(bias_array));
    
    disp("Experimental E_out Value");
    disp(mean(var_array) + mean(bias_array));
    
    plot(input, output, input, output_bar);
    title("Target Function and G Bar Function");
    xlabel("x");
    ylabel("y");

end
    