function [ConfusionMat,order,E] = confusionMatrix(y_real,y_predict,flag_plotMatrix)

    [ConfusionMat,order] = confusionmat(y_real,y_predict);
    E=100*(ConfusionMat(1,2)+ConfusionMat(2,1))/(sum(sum(ConfusionMat)));
    
    if (flag_plotMatrix==1)
        figure();
        confusionchart(ConfusionMat,order);
    end
end
