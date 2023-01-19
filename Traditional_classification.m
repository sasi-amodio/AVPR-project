%% CLASSIFICATION AND EVALUATION USING LBP, HoG

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

import_features = 1;

if (import_features)
    subregions = 9;
    [train_data_LBP, train_data_HOG] = extractFeatures_traditional(trainingSet,subregions);
    [test_data_LBP, test_data_HOG] = extractFeatures_traditional(testSet,subregions);
    save FEATURES/HOG/train_data_HOG 'train_data_HOG';
    save FEATURES/LBP/train_data_LBP 'train_data_LBP';
    save FEATURES/HOG/test_data_HOG 'test_data_HOG';
    save FEATURES/LBP/test_data_LBP 'test_data_LBP';
else
    load FEATURES/HOG/train_data_HOG 'train_data_HOG';
    load FEATURES/LBP/train_data_LBP 'train_data_LBP';
    load FEATURES/HOG/test_data_HOG 'test_data_HOG';
    load FEATURES/LBP/test_data_LBP 'test_data_LBP';
end

%% TRAIN THE MODEL AND OBTAIN CLASSES AND SCORES

% For RBF:
gamma = 0.015625;
sigma = sqrt(1/(2*gamma));

svm_model_LBP = fitcsvm(train_data_LBP, numerical_training_labels,'KernelFunction','RBF', 'KernelScale',sigma,'BoxConstraint',0.5,'Standardize',true);
svm_model_HOG = fitcsvm(train_data_HOG, numerical_training_labels,'KernelFunction','RBF', 'KernelScale',sigma,'BoxConstraint',0.5,'Standardize',false);

[predicted_labels_LBP, scores_LBP] = predict(svm_model_LBP, test_data_LBP);
[predicted_labels_HOG, scores_HOG] = predict(svm_model_HOG, test_data_HOG);


%% EVALUATION

% LBP features

[FPR,TPR,T,AUC_LBP,OPTROCPT] = perfcurve(numerical_test_labels,scores_LBP(:,1),'1');

figure();
plot(FPR,TPR);
xlabel('False Positive Rate (FPR)');
ylabel('True Positive Rate (TPR)');
title(['ROC. AUC = ' num2str(AUC_LBP)]);
hold on;
plot(OPTROCPT(1),OPTROCPT(2),'*');

[ConfusionMat,order,E_LBP] = confusionMatrix(numerical_test_labels,predicted_labels_LBP,1);

% HOG features

[FPR,TPR,T,AUC_HOG,OPTROCPT] = perfcurve(numerical_test_labels,scores_HOG(:,1),'1');

figure();
plot(FPR,TPR);
xlabel('False Positive Rate (FPR)');
ylabel('True Positive Rate (TPR)');
title(['ROC. AUC = ' num2str(AUC_HOG)]);
hold on;
plot(OPTROCPT(1),OPTROCPT(2),'*');

[ConfusionMat,order,E_HOG] = confusionMatrix(numerical_test_labels,predicted_labels_HOG,1);