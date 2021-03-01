function [ ] = plotAdaBoostData( train_err, test_err)
    
%     figure(1);
%     hold on
%     plot(train_err);
%     plot(test_err);
%     hold off
%     
%     legend('Training Error','Testing Error')
%     
%     xlabel('Number of Trees');
%     ylabel('Classification Error Measure');
%     title('1 vs 3 Error Measure as a Function of Number of Stumps');
    
    figure(1);
    hold on
    plot(train_err);
    plot(test_err);
    hold off
    
    legend('Training Error','Testing Error')
    
    xlabel('Number of Trees');
    ylabel('Classification Error Measure');
    title('3 vs 5 Error Measure as a Function of Number of Stumps');
    

end
