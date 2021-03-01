firstNum = 1;
secondNum = 3;

datatrain=load('zip.train');
subsample_tr = datatrain(datatrain(:,1)== firstNum | datatrain(:,1) == secondNum, :);

[n_tr, m_tr] = size(subsample_tr);
Y_tr = subsample_tr(:,1);
X_tr = subsample_tr(:,2:m_tr);
data_tr = [X_tr Y_tr];

datatest=load('zip.test');
subsample_test = datatest(datatest(:,1)== firstNum | datatest(:,1) == secondNum, :);
[n_test, m_test] = size(subsample_test);

Y_test = subsample_test(:,1);
X_test = subsample_test(:,2:m_test);
data_test = [X_test Y_test];

disp(size(Y_test, 1));

% single bag case

tree_single = fitctree(X_tr, Y_tr);
labels_single = predict(tree_single, X_test);
err_single = 0;
for i = 1:n_test
    if labels_single(i) ~= Y_test(i)
        err_single = err_single + 1;
    end
end

oob_single = err_single / n_test;
fprintf('OOB Error when trained on a single tree: %.4f\n', oob_single);


oob_multi = 0;

Y_agg = zeros(n_test, 1);
%multi bag case
% for i = 1:200
%     inds = (1:n_test)';
%     
%     sampleRows = datasample(1:n_tr, n_tr);
%     %data we will train our decision tree on
%     sampleData = X_tr(sampleRows, :);
%     sampleLabels = Y_tr(sampleRows, :);
% 
%     
%     tree_multi = fitctree(sampleData, sampleLabels);
%     
%     labels_multi = predict(tree_multi, X_test);
%     labels_multi(labels_multi == firstNum) = -1;
%     labels_multi(labels_multi == secondNum) = 1;
%     
%     Y_agg(inds) = Y_agg(inds) + labels_multi;
% 
%     labels_multi = Y_agg;
%     labels_multi(labels_multi >= 0) = secondNum;
%     labels_multi(labels_multi < 0) = firstNum;
%     err_multi = 0;
%     
%     for j = 1:n_test
%         if labels_multi(j) ~= Y_test(j)
%             err_multi = err_multi + 1;
%         end
%     end
%     
%     oob_multi = err_multi / n_test;
% end

%fprintf('OOB Error when trained on ensemble of 200 trees: %.4f\n', oob_multi);


% Get 200 trees from BaggedTrees

num_trees = 200;
num_test_exs = size(Y_test, 1);
Y_tr(Y_tr==firstNum) = -1;
Y_tr(Y_tr==secondNum) = 1;

Y_test(Y_test==firstNum) = -1;
Y_test(Y_test==secondNum) = 1;

disp("BOUTTA RUN");
[oobErr, trees] = BaggedTrees(X_tr, Y_tr, num_trees);
disp("RAN");
% T x N_TEST matrix, each row is a given round, each column is a given test
% point
test_preds = zeros(num_trees, num_test_exs);
for i = 1:num_trees
    % compute prediction for all samples in our test set using tree t
    preds = predict(trees{i}, X_test);
    test_preds(i,:) = preds;
end

% Compute final predictions by taking majority element

multi_tree_error = 0;
for i = 1:num_test_exs
    final_pred = mode(test_preds(:, i)); 
    if (final_pred ~= Y_test(i))
        multi_tree_error= multi_tree_error + 1;
    end
end

oob_err_multi = multi_tree_error / n_test;

fprintf('OOB Error when trained on ensemble of 200 trees: %.4f\n', oob_err_multi);


