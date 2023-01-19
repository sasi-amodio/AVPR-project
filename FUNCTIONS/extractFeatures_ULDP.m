function data = extractFeatures_ULDP(set,subregions)
    for i = 1:size(set.Files,1)

        % Read image
        img = readimage(set,i);
        % Smooth image
        %img = imgaussfilt(img,0.5);
        % Resize image
        img = imresize(img, [75 75]);
        % Apply Histogram Equalization
        %img = histeq(img,50);
        % Extract ULDP features
        uldp_features = extractULDPFeatures(img,subregions);

        % Store in a matrix for training
        data(i,:) = uldp_features;

       
    end
end