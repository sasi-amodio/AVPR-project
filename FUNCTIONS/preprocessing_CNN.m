function features = preprocessing_CNN(set,net)
    
    imageSize = net.Layers(1).InputSize;
    augmentedSet = augmentedImageDatastore(imageSize, set, 'ColorPreprocessing', 'gray2rgb');

    featureLayer = 'fc1000';
    features = activations(net, augmentedSet, featureLayer, ...
    'MiniBatchSize', 32, 'OutputAs', 'columns');

end