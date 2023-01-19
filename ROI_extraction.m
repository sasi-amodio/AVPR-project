%   Project AVPR 2022 : First System 
%   Student : Salvatore Davide Amodio

%% Clear Variables, Close Current Figures, and Create Results Directory 
clc;
clear all;
close all;

%% Load Dataset and Split in Train, Validation and Test Set

dir = "dataset-all-mias/all-mias/";
newdir = "dataset-all-mias/all-mias-mod/";

massDir = append(dir, 'mass/');
belignDir = append(newdir, 'belign/');
malignantDir = append(newdir, 'malignant/');
massInfo = append(dir, 'mass.csv');

massFile = readtable(massInfo);

massSet = imageDatastore(massDir, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

massInfoSize = numel(massSet.Files);

%% ROIs Extraction (without resizing the extracted ROIs)

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

trainingSize = numel(trainingSet.Files);
testSize = numel(testSet.Files);

trainingLabels = trainingSet.Labels;
testLabeles = testSet.Labels;
    

