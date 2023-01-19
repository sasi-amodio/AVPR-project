%% CLASSIFICATION AND EVALUATION USING CNN AS FEATURE EXTRACTION

clear
clc
close all
addpath('FUNCTIONS/');

%% DATA IMPORT

setDir_train = fullfile('Dataset', 'TrainSet');
setDir_test = fullfile('Dataset', 'TestSet');

trainingSet = imageDatastore(setDir_train, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
testSet = imageDatastore(setDir_test, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

tbl_train = countEachLabel(trainingSet);
tbl_test = countEachLabel(testSet);

trainLabels = trainingSet.Labels;
testLabels=testSet.Labels;

numerical_training_labels = grp2idx(trainLabels');
numerical_test_labels = grp2idx(testLabels');

%% EXTRACT FEATURES

import_features = 0;

if (import_features)
    net = resnet50();
    train_data_CNN = preprocessing_CNN(trainingSet,net)';
    test_data_CNN = preprocessing_CNN(testSet,net)';
    save FEATURES/CNN/train_data_CNN 'train_data_CNN'
    save FEATURES/CNN/test_data_CNN 'test_data_CNN'

else
    load FEATURES/CNN/train_data_CNN 'train_data_CNN'
    load FEATURES/CNN/test_data_CNN 'test_data_CNN'
end

%% TRAIN THE MODEL AND OBTAIN CLASSES AND SCORES

% For RBF:
gamma = 0.015625;
sigma = sqrt(1/(2*gamma));

svm_model_CNN = fitcsvm(train_data_CNN, numerical_training_labels,'KernelFunction','rbf', 'KernelScale',sigma,'BoxConstraint',0.5,'Standardize',false);

[predicted_labels_CNN, scores_CNN] = predict(svm_model_CNN, test_data_CNN);


%% EVALUATION

[FPR,TPR,T,AUC_CNN,OPTROCPT] = perfcurve(numerical_test_labels,scores_CNN(:,1),'1');

figure();
plot(FPR,TPR);
xlabel('False Positive Rate (FPR)');
ylabel('True Positive Rate (TPR)');
title(['ROC. AUC = ' num2str(AUC_CNN)]);
hold on;
plot(OPTROCPT(1),OPTROCPT(2),'*');

[ConfusionMat,order,E_CNN] = confusionMatrix(numerical_test_labels,predicted_labels_CNN,1);