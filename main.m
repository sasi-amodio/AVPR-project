%   Project AVPR 2022 : First System 
%   Student : Salvatore Davide Amodio

%% Clear Variables, Close Current Figures, and Create Results Directory 
clc;
clear all;
close all;

%% Load Dataset and Split in Train, Validation and Test Set

dir = "dataset/all-mias/";
newdir = "dataset/all-mias-mod/";

massDir = append(dir, 'mass/');
belignDir = append(newdir, 'belign/');
malignantDir = append(newdir, 'malignant/');
massInfo = append(dir, 'mass.csv');

normalDir = append(dir, 'normal/');
newNormalDir = append(newdir, 'normal/');
normalInfo = append(dir, 'normal.csv');

massFile = readtable(massInfo);
normalFile = readtable(normalInfo);

massSet = imageDatastore(massDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
normalSet = imageDatastore(normalDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

massInfoSize = numel(massSet.Files);
normalinfosize = numel(normalSet.Files);

%% ROI Extraction

for newDim = [32,64,75,150] 

    index = 0;
    nROis = size(massFile,1);
    
    imageNameTmp = "";
    
    mkdir(belignDir);
    mkdir(malignantDir);
    
    for i = 1: nROis
        imageName = table2array(massFile(i,"col0"));
       
        type = table2array(massFile(i,"col3"));
        abn_cx = table2array(massFile(i,"col4"));
        abn_cy = table2array(massFile(i,"col5"));
        abn_radius = table2array(massFile(i,"col6"));
    
        if ~strcmp(imageNameTmp,imageName)
           index = index + 1;
           imageNameTmp = imageName;
        end
    
        img = readimage(massSet,index);
    
        if abn_cx ~= 0 
            img = img(1024- abn_cy - abn_radius : 1024- abn_cy + abn_radius , ...
                abn_cx - abn_radius : abn_cx + abn_radius);  
    
            img = imresize(img, [newDim,newDim]);
    
            if strcmp(type,"B")
                imgName = append(belignDir, 'roi_', int2str(i), '.png');
            else 
                imgName = append(malignantDir, 'roi_', int2str(i), '.png');
            end    
            imwrite(img, imgName);
        end 
    
    end 
    
    %% Load Dataset and Split in Train, Validation and Test Set
    
    dataSet = imageDatastore(newdir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
            
    p = 0.7;
    [trainingSet,testSet] = splitEachLabel(dataSet,p,'randomize');
    [trainingSet,validationSet] = splitEachLabel(trainingSet,p,'randomize');
    
    
    trainingSize = numel(trainingSet.Files);
    validationSize = numel(validationSet.Files);
    
    trainingLabels = trainingSet.Labels;
    validationLabels = validationSet.Labels;
    
    %% GridMap 
    
    bestCellSize = -1;
    bestTrainingAccuracy = [];
    bestValidationAccuracy = 1;
    bestSVM_C = [];
    bestKernel = [];
    bestMdl = [];
    
    for cellSize = 3:newDim
        
        % Initializiation of matrices that are going to contain feature vectors 
        
        disp("starts feature extraction");
    
        img = readimage(trainingSet, 1);
        feature_vector = extractLBPFeatures(img, 'cellSize', [cellSize cellSize]);
        lbpFeatureSize = length(feature_vector);
    
        trainingFeatures = zeros(trainingSize, lbpFeatureSize, 'single');
        validationFeatures = zeros(validationSize, lbpFeatureSize, 'single');
        
        % feature extractions 
    
        for i = trainingSize
            img = readimage(trainingSet, i);
            trainingFeatures(i, :) = extractLBPFeatures(img, 'CellSize', [cellSize cellSize]);
        end
        
        for i = 1 : validationSize
            img = readimage(validationSet, i);
            validationFeatures(i, :) = extractLBPFeatures(img, 'CellSize', [cellSize cellSize]);
        end
        
    
        % find the best parameters combinations  
            
        for SVM_C = [0.001, 0.05, 0.01, 0.5, 0.1, 1, 3, 5] 
            for SVM_Kernel = ["polynomial", "linear", "rbf"] 
    
                disp("starts training");
                
                SVMModel = fitcsvm(trainingFeatures,trainingLabels,'KernelFunction',SVM_Kernel,...
                           'BoxConstraint', SVM_C);
                
                CVSVMModel = crossval(SVMModel);
                classLoss = kfoldLoss(CVSVMModel);
    
                % evaluate the classifier on training data
    
                %trainingPred = predict(Mdl, trainingFeatures);
                %trainingAccuracy = mean(trainingPred == trainingLabels);
            
                % evaluate the classifier on validation data
    
                %validationPred = predict(Mdl, validationFeatures);
                %validationAccuracy = mean(validationPred == validationLabels);
                   
    
                % choose the best model by comparing the validation accuracy
                
                disp("loss : " + classLoss);
    
                if classLoss < bestValidationAccuracy 
                    bestSVM_C = SVM_C;
                    bestKernel = SVM_Kernel;
                    %bestTrainingAccuracy = trainingAccuracy;
                    bestValidationAccuracy = classLoss;  
                    bestMdl = SVMModel;
                    bestCellSize = cellSize;
    
                    disp("partial result : SVM_C = "+ SVM_C + " , Kernel = " + SVM_Kernel + ...
                        ", classLoss = " + classLoss + ", cellSize = " + cellSize);
                end 
            end
        end 
    end
        
    %% Evaluate the performance of the best classifier computed
    
    testSize = numel(testSet.Files);
    testLabels = testSet.Labels;
    
    % Initializiation of matrix that is going to contain feature vectors 
    
    img = readimage(testSet, 1);
    feature_vector = extractLBPFeatures(img, 'cellSize', [bestCellSize bestCellSize]);
    lbpFeatureSize = length(feature_vector);
    
    testFeatures = zeros(testSize, lbpFeatureSize, 'single');
    
    % feature extractions 
    
    for i = 1 : testSize
        img = readimage(testSet, i);    
        testFeatures(i, :) = extractLBPFeatures(img, 'CellSize', [bestCellSize bestCellSize]);
    end
    
    % evaluate the classifier on test data
    
    testPred = predict(bestMdl, testFeatures);
    testAccuracy = mean(testPred == testLabels);
    
    disp("final result : SVM_C = "+ bestSVM_C + " , Kernel = " + bestKernel + ...
        ", Gamma = 'auto' , trainingAccuracy = " + bestTrainingAccuracy + ...
        ", testAccuracy = " + testAccuracy + ", cellSize = " + bestCellSize);

end


%% 

% mkdir(newNormalDir);

%for i = 1: normalinfosize 

%    img = readimage(normalSet, i);
%    imgName = append(newNormalDir, 'roi_', int2str(i), '.png');

%    radius = newDim;
%    centre = size(img,1);

%    img = img(centre/2 - radius : centre/2 + radius, ...
%       centre/2 - radius : centre/2 + radius);

%    img = imresize(img, [newDim,newDim]);
%    imwrite(img, imgName);
%end 
