function [ ] = svm_project()

    disp('findme');

%     for i=1:1
        
    
        % read in the csv data
        filepath = '/Users/Everwitt/Downloads/SVM.csv';
        data = csvread(filepath, 1, 0);
        % randomize the data coming in 
        data = data(randsample(1:length(data),length(data)),:);

        % train / test split
        train_test_split = 0.8;
        train_thresh = train_test_split * size(data, 1);

        % split the linearly unseparable data
        x_train = data(1:train_thresh, 1:2);
        y_train = data(1:train_thresh, 3);

%         x_train = vertcat(x_train, x_train, x_train, x_train);
%         y_train = vertcat(y_train, y_train, y_train, y_train);
%         disp(size(x_train))
        
        % split the linearly unseparable data
        x_test = data(train_thresh+1:size(data,1), 1:2);
        y_test = data(train_thresh+1:size(data,1), 3);


        % display just how noisy the data is
    %     figure(1)
    %     class_one = data(data(:,3) == 0, :);
    %     class_two = data(data(:,3) == 1, :);
    %     plot(class_one(:,1), class_one(:,2), 'r.', 'MarkerSize', 15);
    %     hold on
    %     plot(class_two(:,1), class_two(:,2), 'b.', 'MarkerSize', 15);
    %     hold off
    %     legend('Classification: 0', 'Classification: 1');
    %     xlabel('first feature (x)');
    %     ylabel('second feature (y) ');
    %     title('Linearly Unseparable (x,y) pairs and labels');

        model_lin_unseparable_data(x_train, y_train, x_test, y_test);


        % now dealing with linearly separable data 
        sep_data = gen_lin_sep_data();

        sep_x_train = sep_data(1:train_thresh, 1:2);
        sep_y_train = sep_data(1:train_thresh, 3);

        sep_x_test = sep_data(train_thresh+1:size(sep_data, 1), 1:2);
        sep_y_test = sep_data(train_thresh+1:size(sep_data,1), 3);

        % display the linearly separable noiseless data
    %     figure(2)
    %     sep_class_one = sep_data(sep_data(:,3) == 0, :);
    %     sep_class_two = sep_data(sep_data(:,3) == 1, :);
    %     plot(sep_class_one(:,1), sep_class_one(:,2), 'r.', 'MarkerSize', 15);
    %     hold on
    %     plot(sep_class_two(:,1), sep_class_two(:,2), 'b.', 'MarkerSize', 15);
    %     hold off
    %     legend('Classification: 0', 'Classification: 1');
    %     xlabel('first feature (x)');
    %     ylabel('second feature (y) ');
    %     title('Linearly Separable (x,y) pairs and labels');

        LinearModel = fitclinear(sep_x_train, sep_y_train,'ClassNames', {'0','1'});
        [ label, ~ ] = predict(LinearModel, sep_x_test);
        sep_lin_predictions = str2num(char(label));
        lin_model_score = 0;

        for index = 1:size(sep_y_test, 1)
            if sep_lin_predictions(index) ~= sep_y_test(index)
                lin_model_score = lin_model_score + 1;
            end
        end

        disp('Linearly Separable Data Error Measure Scores');

        disp('Classification Separable Linear Model')
        disp(lin_model_score / size(sep_lin_predictions, 1));
%     end
    
end
