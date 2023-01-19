%% CLASSIFICATION AND EVALUATION USING ULDP

clear
clc
close all
addpath('FUNCTIONS/');

%% DATA IMPORT

setDir_train = fullfile('Dataset', 'TrainSet');
setDir_test = fullfile('Dataset', 'TestSet');

trainingSet = imageDatastore(setDir_train, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
testSet = imageDatastore(setDir_test, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

tbl_train = countEachLabel(trainingSet)
tbl_test = countEachLabel(testSet)

trainLabels = trainingSet.Labels;
testLabels=testSet.Labels;

numerical_training_labels = grp2idx(trainLabels');
numerical_test_labels = grp2idx(testLabels');

%% EXTRACT FEATURES

import_features = 0

if (import_features)
    subregions = 9;
    train_data = extractFeatures_ULDP(trainingSet,subregions);
    test_data = extractFeatures_ULDP(testSet,subregions);
    save FEATURES/ULDP/train_data_ULDP 'train_data';
    save FEATURES/ULDP/test_data_ULDP 'test_data';
else
    load FEATURES/ULDP/train_data_ULDP 'train_data';
    load FEATURES/ULDP/test_data_ULDP 'test_data';
end

%% TRAIN THE MODEL AND OBTAIN CLASSES AND SCORES

% For RBF:
gamma = 0.015625;
sigma = sqrt(1/(2*gamma));

svm_model = fitcsvm(train_data, numerical_training_labels,'KernelFunction','rbf','KernelScale',sigma,'BoxConstraint',0.5,'Standardize',false);

[predicted_labels, scores] = predict(svm_model, test_data);

%% EVALUATION

[FPR,TPR,T,AUC,OPTROCPT] = perfcurve(numerical_test_labels,scores(:,1),'1');

figure();
plot(FPR,TPR);
xlabel('False Positive Rate (FPR)');
ylabel('True Positive Rate (TPR)');
title(['ROC. AUC = ' num2str(AUC)]);
hold on;
plot(OPTROCPT(1),OPTROCPT(2),'*');

[ConfusionMat,order,E] = confusionMatrix(numerical_test_labels,predicted_labels,1);