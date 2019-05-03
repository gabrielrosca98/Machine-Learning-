Exercise1b
clc
%Initialise the YTraining matrix with 0s
yClass = zeros(200, 40);

%Create the YTraining matrix used to calculate the W coefficient matrix
for index = 1:40
  %Calculate the indexes for each subject photos in the data matrix
  startIndex = (index - 1) * 5 + 1;
  finishIndex = (index - 1) * 5 + 5;
  %The element is 1 if it is from the indexSubj class
  for indexSubj = startIndex:finishIndex
    yClass(indexSubj, dataStructure2(10).Ytr(indexSubj)) = 1;
  end
end

%Create a new Xtilda from the training set
XtildaTraining = [ones(200, 1), dataStructure2(10).Xtr];
%Calculate the coefficient matrix
w = pinv(XtildaTraining) * yClass;
%Transpose the coefficient matrix for easier calculation
w = transpose(w);

%Calculate the acc rate for each data set
for indexDataSet = 1:50
  %Memorise how many wrong classifications we have for the current data set
  wrongClass = 0;
  %Iterate through all the 200 sample photos for the testing set
  for indexSample = 1:200
    %Initialise the maximum value with the lowest possible int value
    maximumValue = intmin;
    %Initialise the final class for the prediction with 0
    finalClass = 0;
    %For each classifier calculate the y result
    for indexClassifier = 1:40
      %Create a new Xtilda
      Xtesting = [1, dataStructure2(indexDataSet).Xte(indexSample,:)];
      %Calculate the y result and memorise it
      yResultDataSet = Xtesting * transpose(w(indexClassifier,:));
      %Check whether is maximumValue and assign the final class to the current classifier index
      if yResultDataSet > maximumValue
        maximumValue = yResultDataSet;
        finalClass = indexClassifier;
      end
    end
    %Count the mistakes
    if finalClass ~= dataStructure2(indexDataSet).Yte(indexSample)
      wrongClass = wrongClass + 1;
    end
  end
  %Memorise the acc
  accRateLinear(indexDataSet) = 100 - (wrongClass / 200) * 100;
  fprintf('The accuracy for data set %d is %f\n', indexDataSet, accRateLinear(indexDataSet))
end

%Classify each photo from the data sets(testing) for each subject
%Calculate the acc. rate for each subject
for indexDataSet = 1:20
  %For each subject
  for indexSubject = 1:40
    %Memorise how many wrong classifications we have for the current data set
    wrongSubject = 0;
    %Initialise the maximum value with the lowest possible int value
    maximumValue = intmin;
    %Initialise the final class for the prediction with 0
    finalClass = 0;
    for indexSample = 1:5
      %Create new Xtilda
      Xtesting = [1, dataStructure2(indexDataSet).Xte((indexSubject - 1) * 5 + indexSample,:)];
      for indexClassifier = 1:40
        %Calculate the y and memorise it
        ySubjTesting =  Xtesting * transpose(w(indexClassifier, :));
        %Check whether is maximumValue and assign the final class to the current classifier index
        if ySubjTesting > maximumValue
          maximumValue = ySubjTesting;
          finalClass = indexClassifier;
        end
      end
      %Check whether it was a missclassification
      if finalClass ~= dataStructure2(indexDataSet).Yte((indexSubject - 1) * 5 + indexSample)
        wrongSubject = wrongSubject + 1;
      end
    end
    %Memorise the accuracy
    accSubjLinear(indexDataSet, indexSubject) = 100 - (wrongSubject/ 5) * 100;
  end
end

%Plot the comparison between the kNN and Linear multi-class classifier
indexDataSet = 1:50;
figure('Name', 'Linear multi-class vs kNN')
plot(indexDataSet, accRateLinear, indexDataSet, accuracyTesting);
xlabel('Index data set');
ylabel('Accuracy testing');

%Plot the accuracy for each subject using the multi-class classification
indexSubject = 1:40;
figure('Name', 'Accuracy for each subject using linear multi-class classification')
meanSubject = mean(accSubjLinear);
plot(indexSubject, meanSubject)
xlabel('Index subject')
ylabel('Mean accuracy')
