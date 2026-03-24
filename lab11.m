%% %problema1
load fisheriris.mat
figure;
gscatter(meas(:,1), meas(:,2), species,'rgb','osd');
xlabel('Sepal length');
ylabel('Sepal width')
%% %problema2
cv = cvpartition(species,'HoldOut',0.3);
idx = cv.test;
features = [1 2 3 4]; %se vor selecta trasaturile dorite pentru antrenare
XTrain = meas(~idx,features);
YTrain = species(~idx,:);
XTest = meas(idx,features);
YTest = species(idx,:);
%% %problema3
% creare si antrenare knn
k = 15;
knnModel = fitcknn(XTrain,YTrain,'NumNeighbors',k);
% evaluare model pe datele de test
knnPred = predict(knnModel, XTest);
knnAccuracy = sum(strcmp(knnPred, YTest)) / numel(YTest);
fprintf('Acuratețea kNN este: %.2f%% \n', knnAccuracy * 100);
%% %problema5
% crearea si antrenarea modelului Naive Bayes
nbModel = fitcnb(XTrain, YTrain);
% evaluarea modelului pe datele de test
nbPred = predict(nbModel, XTest);
nbAccuracy = sum(strcmp(nbPred, YTest)) / numel(YTest);
fprintf('Acuratețea Naive Bayes este: %.2f%%\n', nbAccuracy * 100);
plotDecisionBoundary(XTrain, YTrain, nbModel)
%% 
data = readtable("diabetes.csv");
a = data.Outcome;
cv1 = cvpartition(a,'HoldOut',0.3);
idx1 = cv1.test;
features = [3 5];
b = data.BloodPressure;
c = data.Insulin;
XTrain = meas(~idx,features);
YTrain = species(~idx,:);
XTest = meas(idx,features);
YTest = species(idx,:);