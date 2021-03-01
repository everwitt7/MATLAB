% Script to load data from zip.train, filter it into datasets with only one
% and three or three and five, and compare the performance of plain
% decision trees (cross-validated) and bagged ensembles (OOB error)
load zip.train;

fprintf('Working on the one-vs-three problem...\n\n');
subsample = zip(find(zip(:,1)==1 | zip(:,1) == 3),:);
Y = subsample(:,1);
X = subsample(:,2:257);
ct = fitctree(X,Y,'CrossVal','on');
fprintf('The cross-validation error of decision trees is %.4f\n', ct.kfoldLoss);
bee = BaggedTrees(X, Y, 200);
fprintf('The OOB error of 200 bagged decision trees is %.4f\n', bee);

% oob = zeros(1,200);
% for i = 1:200
%     disp("YEE");
%     oob(i) = BaggedTrees(X,Y,i);
%     disp(oob(i));
%     disp(i);
% end

a = [0.6452, 0.3999, 0.2441, 0.16, 0.1094, 0.0752, 0.0439, 0.0313, 0.0204, 0.0156, 0.0168, 0.0072, 0.009, 0.0066, 0.0066, 0.0054, 0.0066, 0.0048, 0.0048, 0.0054, 0.006, 0.0042, 0.0036, 0.0048, 0.0042, 0.0036, 0.0036, 0.0042, 0.006, 0.003, 0.0036, 0.0036, 0.0042, 0.0042, 0.0036, 0.0054, 0.0048, 0.0048, 0.0036, 0.0036, 0.0048, 0.0042, 0.0036, 0.0042, 0.0036, 0.0036, 0.003, 0.003, 0.003, 0.0024, 0.0036, 0.0018, 0.003, 0.0024, 0.0042, 0.0036, 0.003, 0.0024, 0.003, 0.0036, 0.003, 0.0036, 0.0042, 0.003, 0.0036, 0.0036, 0.0042, 0.0036, 0.0024, 0.0036, 0.0036, 0.003, 0.003, 0.0036, 0.0024, 0.0042, 0.0024, 0.0054, 0.0024, 0.003, 0.003, 0.0042, 0.0036, 0.0036, 0.003, 0.0018, 0.003, 0.0036, 0.0036, 0.003, 0.003, 0.003, 0.003, 0.0036, 0.0042, 0.003, 0.0024, 0.0024, 0.0036, 0.0036, 0.0036, 0.0036, 0.0018, 0.003, 0.0036, 0.0036, 0.003, 0.0024, 0.0042, 0.0036, 0.003, 0.003, 0.0024, 0.003, 0.0048, 0.003, 0.0036, 0.0018, 0.0024, 0.0024, 0.003, 0.0024, 0.0024, 0.0018, 0.0036, 0.0024, 0.0024, 0.003, 0.0042, 0.003, 0.0036, 0.0036, 0.0018, 0.0042, 0.0036, 0.003, 0.0036, 0.0036, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.003, 0.0036, 0.003, 0.0024, 0.003, 0.0036, 0.0024, 0.0036, 0.003, 0.0036, 0.003, 0.0036, 0.003, 0.0042, 0.0024, 0.003, 0.0024, 0.003, 0.0036, 0.0036, 0.0024, 0.0024, 0.003, 0.003, 0.0024, 0.0024, 0.0024, 0.0042, 0.0036, 0.0042, 0.0024, 0.0024, 0.0024, 0.0024, 0.0042, 0.0024, 0.0036, 0.003, 0.003, 0.0024, 0.003, 0.003, 0.003, 0.0018, 0.0036, 0.0036, 0.003, 0.0036, 0.0036, 0.0036, 0.003, 0.003, 0.003, 0.0042, 0.0024];
figure(1)
plot((1:1:200),a);  
xlabel('Number of bagged decision trees');
ylabel('OOB error');
title('OOB error of 1 to 200 bagged decision trees on one-vs-three problem');


fprintf('\nNow working on the three-vs-five problem...\n\n');
subsample = zip(find(zip(:,1)==3 | zip(:,1) == 5),:);
Y = subsample(:,1);
X = subsample(:,2:257);
ct = fitctree(X,Y,'CrossVal','on');
fprintf('The cross-validation error of decision trees is %.4f\n', ct.kfoldLoss);
bee = BaggedTrees(X, Y, 200);
fprintf('The OOB error of 200 bagged decision trees is %.4f\n', bee);


