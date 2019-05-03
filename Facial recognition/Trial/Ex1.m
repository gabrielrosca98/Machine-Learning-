load ORLfacedata

dataSubject = data([1:10, 291:300],:);
labelSubject = labels([1:10,291:300]);
for index = 1:50
    [Xtr, Xte, Ytr, Yte] = PartitionData(dataSubject,labelSubject,3);
    dataStructure(index).Xtr = Xtr;
    dataStructure(index).Xte = Xte;
    dataStructure(index).Ytr = Ytr;
    dataStructure(index).Yte = Yte;
end


wrongTesting = 0;
for index3 = 1:6
    for index = 1: 50
        for index2 = 1:14
            classTesting(index3).Class(index, index2) = knearest(index3, dataStructure(index).Xte(index2,:), dataStructure(index).Xtr, dataStructure(index).Ytr);
            if classTesting(index3).Class(index, index2) ~= dataStructure(index).Yte(index2)
                wrongTesting = wrongTesting + 1;
            end
        end
        errorRateTesting(index3, index) = (wrongTesting / 14) * 100;
        accuracyTesting(index3, index) = 100 - errorRateTesting(index3, index);
        wrongTesting = 0;
    end
end

disp('----------------------')
wrongTraining = 0;
for index3 = 1:6
    for index1 = 1:50
        for index2 = 1:6
            classTraining(index3).Class(index1, index2) = knearest(index3, dataStructure(index1).Xtr(index2,:), dataStructure(index1).Xtr, dataStructure(index1).Ytr);
            if classTraining(index3).Class(index1, index2) ~= dataStructure(index1).Ytr(index2)
                wrongTraining = wrongTraining + 1;
            end
        end
        errorRateTraining(index3, index1) = (wrongTraining / 6) * 100;
        accuracyTraining(index3, index1) = 100 - errorRateTraining(index3, index1);
        wrongTraining = 0;
    end
end

disp('Testing')
for index = 1:6
    for index2 = 1:50
        %fprintf('For k = %d and sample number = %d: Error rate:%f\tAccuracy Testing: %f \n', index, index2,errorRateTesting(index, index2), accuracyTesting(index, index2))
    end
    meanAccuracyTesting(index) = mean(accuracyTesting(index,:));
    fprintf('For k = %d we have mean average of accuracy of %f\n', index, meanAccuracyTesting(index))
end

disp('Training')
for index = 1:6
    for index2 = 1:50
      %fprintf('For k = %d: Error rate:%f\tAccuracy Training: %f \n', index,errorRateTraining(index, index2), accuracyTraining(index, index2))
    end
    meanAccuracyTraining(index) = mean(accuracyTraining(index,:));
    fprintf('For k = %d we have mean average of accuracy of %f\n', index, meanAccuracyTraining(index))
end

standardDeviationTraining = std(transpose(accuracyTraining));
standardDeviationTesting = std(transpose(accuracyTesting));

k = 1:6;

figure('Name','Training');
errorbar(k, meanAccuracyTraining, standardDeviationTraining)

figure('Name','Testing');
errorbar(k, meanAccuracyTesting, standardDeviationTesting)

figure('Name','Show result');
ShowResult(Xte, Yte, classTesting(1).Class(50,:),4)
