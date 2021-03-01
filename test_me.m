function [] = test_me()

    % messing around to generate a plot of non-linearly separable data
    x3 = rand(1000, 1);
    x4 = rand(1000, 1);
    
    e1 = rand(1000, 1) * .2;
    e2 = rand(1000, 1) * .2;
    
    t1 = x3 .^ 2;
    t2 = x4 .^ 2;
    
    lab = sign(x3 + x4 - 1);
    
    data = [ x3+e1 x4+e2 lab ];
    
    figure(1)
    d1 = data(data(:,3) == -1, :);
    d2 = data(data(:,3) == 1, :);
    plot(d1(:,1), d1(:,2), 'r.', 'MarkerSize', 15);
    hold on
    plot(d2(:,1), d2(:,2), 'b.', 'MarkerSize', 15);
    hold off
    legend('Classification: 0', 'Classification: 1');
    xlabel('first feature (x)');
    ylabel('second feature (y) ');
    title('Noisy Linearly Separable (x,y) pairs and labels');
    
%     data2 = [ t1 t2 lab ];
%     figure(2)
%     d3 = data2(data2(:,3) == -1, :);
%     d4 = data2(data2(:,3) == 1, :);
%     plot(d3(:,1), d3(:,2), 'r.', 'MarkerSize', 15);
%     hold on
%     plot(d4(:,1), d4(:,2), 'b.', 'MarkerSize', 15);
%     hold off
%     axis([0 10 0 10]);
%     legend('Classification: 0', 'Classification: 1');
%     xlabel('first feature (x^2)');
%     ylabel('second feature (y^2) ');
%     title('Non-Linearly Separable (x,y) pairs and labels Transformed');
end