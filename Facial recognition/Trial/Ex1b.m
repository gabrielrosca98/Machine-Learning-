load ORLfacedata
%Store the 50 data sets in a structure
for index = 1:50
    [Xtr, Xte, Ytr, Yte] = PartitionData(data,labels,5);
    dataStructure(index).Xtr = Xtr;
    dataStructure(index).Xte = Xte;
    dataStructure(index).Ytr = Ytr;
    dataStructure(index).Yte = Yte;
end

%Choose a random data set to calculate the error for each k(number of neighbours)
indexDataSet = randi([1  50]);
for indexNeighbour = 1:199
  wrongClass = 0;
    for indexSample = 1:200
      Xtraining = dataStructure(indexDataSet).Xtr([1:indexSample - 1,indexSample + 1:200],:);
      Ytraining = dataStructure(indexDataSet).Ytr([1:indexSample - 1, indexSample + 1:200],:);
      predictedClass(indexSample) = knearest(indexNeighbour, dataStructure(indexDataSet).Xtr(indexSample,:), Xtraining, Ytraining);
      if predictedClass(indexSample) ~= dataStructure(indexDataSet).Ytr(indexSample)
        wrongClass = wrongClass + 1;
      end
    end
  errorRate(indexNeighbour) = (wrongClass / 200) * 100;
  %fprintf('The errror rate for %d is %f\n',indexNeighbour, errorRate(indexNeighbour))
end

%Choose the minimal error rate from all the error rates
minError = min(errorRate);

%Find the number of neighbours coresponding to this error
numberNeighbour = find(errorRate <= minError, 1);

fprintf('We choose the hyperparameter value of %d for k with the minimal error rate: %f and the testing accuracy: %f\n', numberNeighbour, minError, 100 - minError)

%Calculate the testing accuracy for the all 50 data sets using the found hyperparameter
%value of number neighbours
for indexDataSet = 1:50
  wrongResult = 0;
  for indexSample = 1:200
    predicClassTest(indexDataSet, indexSample) = knearest(numberNeighbour, dataStructure(indexDataSet).Xte(indexSample,:), dataStructure(indexDataSet).Xtr, dataStructure(indexDataSet).Ytr);
    if predicClassTest(indexDataSet, indexSample) ~= dataStructure(indexDataSet).Ytr(indexSample)
      wrongResult = wrongResult + 1;
    end
  end
  errorRateTesting(indexDataSet) = (wrongResult / 200) * 100;
  accuracyTesting(indexDataSet) = 100 - errorRateTesting(indexDataSet);
  %fprintf('The accuracy for %d neighbours is %.2f in data set number %d\n', numberNeighbour,accuracyTesting(indexDataSet), indexDataSet)
end

%Calculate the mean accuracy
meanAccuracyTesting = mean(accuracyTesting);

%Calculate the standard deviatin
standardDeviationTesting = std(accuracyTesting);
fprintf('\n')
fprintf('The mean accuracy for testing is %f and the standard deviation for testing is %f\n', meanAccuracyTesting, standardDeviationTesting)
%fprintf('\nThe accuracy rates for each subject using the found hyperparameter')
fprintf('\n')

%Calculate the accuracy rate for each subject
for indexSubject = 1:40
  wrong = 0;
  for indexSample = 1:10
    %Select the training data/label except the current sample to be tested
    Xtraining = data([1:(indexSubject - 1) * 10 + indexSample - 1, (indexSubject - 1) * 10 + indexSample + 1:400],:);
    Ytraining = labels([1:(indexSubject - 1) * 10 + indexSample - 1, (indexSubject - 1) * 10 + indexSample + 1:400],:);
    predictedClass = knearest(numberNeighbour, data((indexSubject - 1) * 10 + indexSample, :), Xtraining, Ytraining);
    if predictedClass ~= labels((indexSubject - 1) * 10 + indexSample)
      wrong = wrong + 1;
    end
  end
  errorRateSubject(indexSubject) = (wrong / 10) * 100;
  accuracyRateSubject(indexSubject) = 100 - errorRateSubject(indexSubject);
  fprintf('The accuracy rate for subject %d is:\t %.2f\tusing\t%d neighbours\n', indexSubject, accuracyRateSubject(indexSubject), numberNeighbour)
end

fprintf('Accuracy rate using the training sets from data set 1.\n');

for indexSubject = 1:40
  wrong = 0;
  for indexSample = 1:5
    predictedClassTs = knearest(numberNeighbour, dataStructure(1).Xte([(indexSubject - 1) * 5 + indexSample], :), dataStructure(1).Xtr, dataStructure(1).Ytr);
    if predictedClassTs ~= dataStructure(1).Yte((indexSubject - 1) * 5 + indexSample)
      wrong = wrong + 1;
    end
  end
  accRateSub(indexSubject) = 100 - (wrong / 100) * 100;
  fprintf('The accuracy rate for subject %d is:\t %.2f\tusing\t%d neighbours\n', indexSubject, accRateSub(indexSubject), numberNeighbour)
end

figure('Name', 'Accuracy Rate Subjectss')
indexSubject = 1:40;
plot(indexSubject, accRateSub);
xlabel('Subject number');
ylabel('Accuracy rate');



figure('Name', 'Accuracy rate for each subject')
indexSubject = 1:40;
plot(indexSubject, accuracyRateSubject)
xlabel('Subject number');
ylabel('Accuracy rate');

figure('Name', 'Accuracy rate for each data set using the chosen hyperparameter')
indexDataSet = 1:50;
plot(indexDataSet, accuracyTesting)
xlabel('Data set number')
ylabel('Accuracy rate')
