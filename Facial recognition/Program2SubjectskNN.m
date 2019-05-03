clear
clc
load ORLfacedata
%Store the 50 data sets in a structure
dataSubject = data([1:10, 291:300],:);
labelSubject = labels([1:10,291:300]);
for indexDataSet = 1:50
    [Xtr, Xte, Ytr, Yte] = PartitionData(dataSubject,labelSubject,3);
    dataStructure1(indexDataSet).Xtr = Xtr;
    dataStructure1(indexDataSet).Xte = Xte;
    dataStructure1(indexDataSet).Ytr = Ytr;
    dataStructure1(indexDataSet).Yte = Yte;
end
tic;
%For each number of neighbours(from 1 to 6) and each data set calculate the
%accuracy of the classifier for testing.
wrongTesting = 0;
for indexNumberNeighbour = 1:6
    for indexDataSet = 1: 50
        for indexSample = 1:14
            %Get the predicted class
            classTesting(indexNumberNeighbour).Class(indexDataSet, indexSample) = knearest(indexNumberNeighbour, dataStructure1(indexDataSet).Xte(indexSample,:), dataStructure1(indexDataSet).Xtr, dataStructure1(indexDataSet).Ytr);
            %Check whether the result is correct or not
            if classTesting(indexNumberNeighbour).Class(indexDataSet, indexSample) ~= dataStructure1(indexDataSet).Yte(indexSample)
                wrongTesting = wrongTesting + 1;
            end
        end
        %Calculate the accuracy for each data set while using k neighbours
        errorRateTesting(indexNumberNeighbour, indexDataSet) = (wrongTesting / 14) * 100;
        accuracyTesting(indexNumberNeighbour, indexDataSet) = 100 - errorRateTesting(indexNumberNeighbour, indexDataSet);
        %Reset the value used to count the mistakes
        wrongTesting = 0;
    end
end
toc;
disp('----------------------')
%For each number of neighbours(from 1 to 6) and each data set calculate the
%accuracy of the classifier for training.
wrongTraining = 0;
for indexNumberNeighbour = 1:6
    for indexDataSet = 1:50
        for indexSample = 1:6
            %Get the predicted class
            classTraining(indexNumberNeighbour).Class(indexDataSet, indexSample) = knearest(indexNumberNeighbour, dataStructure1(indexDataSet).Xtr(indexSample,:), dataStructure1(indexDataSet).Xtr, dataStructure1(indexDataSet).Ytr);
            %Check whether the result is correct or not
            if classTraining(indexNumberNeighbour).Class(indexDataSet, indexSample) ~= dataStructure1(indexDataSet).Ytr(indexSample)
                wrongTraining = wrongTraining + 1;
            end
        end
        %Calculate the accuracy for each data set while using k neighbours
        errorRateTraining(indexNumberNeighbour, indexDataSet) = (wrongTraining / 6) * 100;
        accuracyTraining(indexNumberNeighbour, indexDataSet) = 100 - errorRateTraining(indexNumberNeighbour, indexDataSet);
        %Reset the value used to count the mistakes
        wrongTraining = 0;
    end
end

%Print the mean accuracy for each neighbour number for testing samples
disp('Testing')
for indexNumberNeighbour = 1:6
    for indexDataSet = 1:50
        %fprintf('For k = %d and sample number = %d: Error rate:%f\tAccuracy Testing: %f \n', indexNumberNeighbour, indexDataSet,errorRateTesting(indexNumberNeighbour, indexDataSet), accuracyTesting(indexNumberNeighbour, indexDataSet))
    end
    %calculate the mean accuracy for each neighbour
    meanAccuracyTesting(indexNumberNeighbour) = mean(accuracyTesting(indexNumberNeighbour,:));
    fprintf('For k = %d we have mean average of accuracy of %f\n', indexNumberNeighbour, meanAccuracyTesting(indexNumberNeighbour))
end


%Print the mean accuracy for each neighbour number for training samples
disp('Training')
for indexNumberNeighbour = 1:6
    for indexDataSet = 1:50
      %fprintf('For k = %d: Error rate:%f\tAccuracy Training: %f \n', indexNumberNeighbour,errorRateTraining(indexNumberNeighbour, indexDataSet), accuracyTraining(indexNumberNeighbour, indexDataSet))
    end
    %calculate the mean accuracy for each neighbour
    meanAccuracyTraining(indexNumberNeighbour) = mean(accuracyTraining(indexNumberNeighbour,:));
    fprintf('For k = %d we have mean average of accuracy of %f\n', indexNumberNeighbour, meanAccuracyTraining(indexNumberNeighbour))
end

%Calculate the standard deviation for testing/training
standardDeviationTraining = std(transpose(accuracyTraining));
standardDeviationTesting = std(transpose(accuracyTesting));

%Index of neighbour
indexNumberNeighbour = 1:6;

%Error bar for the training set
figure('Name','Training');
errorbar(indexNumberNeighbour, meanAccuracyTraining, standardDeviationTraining)
xlabel('Number neighbours')
ylabel('Training accuracy')

%Error bar for the testing set
figure('Name','Testing');
errorbar(indexNumberNeighbour, meanAccuracyTesting, standardDeviationTesting)
xlabel('Number neighbours')
ylabel('Testing accuracy')

%Show result for the selected data set
figure('Name','Show result');
ShowResult(Xte, Yte, classTesting(1).Class(50,:),4)

%Print the mean accuracy
meanKNN = mean(meanAccuracyTesting);
fprintf('The accuracy of kNN classifier is %f\n', meanKNN)
toc;
