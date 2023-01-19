function [LBPfeatures,HOGfeatures] = extractFeatures_traditional(set,subregions)
    for i = 1:size(set.Files,1)
        % Read image
        img = readimage(set,i);
        % Smooth image
        img = imgaussfilt(img,0.5);
        % Resize image
        img = imresize(img, [75 75]);
        % Apply Histogram Equalization
        img = histeq(img,50);

        %% LBP FEATURES

        [N,M] = size(img);
        subN = int16(N/sqrt(subregions));
        subM = int16(M/sqrt(subregions));
        
        lbp_features = [];

        for j = 1:3
            for k = 1:3
                subreg = img(subN*(j-1)+1:subN*j,subM*(k-1)+1:subN*k);
                lbp_features = [lbp_features extractLBPFeatures(subreg)];
                % plot the histogram subregion:
                %bar(extractLBPFeatures(subreg))
                %title(['LBP feature histogram for subregion ' num2str((j-1)*3+k)])
            end
        end
      
        % LBP features before L2-norm
        % bar(lbp_features)
        % title('LBP features histogram before L2-norm')

        % L2 normalization for LBP
        s = 0;
        for ind = 1:size(lbp_features,2)
            s = s + lbp_features(ind)^2;
        end

        L2norm = sqrt(s);
        lbp_features = lbp_features / L2norm;
        
        % LBP features before L2-norm
        % bar(lbp_features)
        % title('LBP features histogram before L2-norm')

        % Store in a matrix 
        LBPfeatures(i,:) = lbp_features;

        %% HOG FEATURES
        hog_features = extractHOGFeatures(img,'CellSize',[3 3],'BlockSize',[8 8]);

        % L2 normalization for HoG
        s = 0;
        for ind = 1:size(hog_features,2)
            s = s + hog_features(ind)^2;
        end

        L2norm = sqrt(s);
        hog_features = hog_features / L2norm;
   
        % Store in a matrix 
        HOGfeatures(i,:) = hog_features;
    end
end