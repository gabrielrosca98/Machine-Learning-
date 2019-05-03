Exercise1a
clc
load ORLfacedata

tic;
%Create the X tilda and calculate the req. W coefficient vector
iDataSet = randi([1  50]);
XtildaTraining = [ones(size(dataStructure1(iDataSet).Xtr, 1), 1), dataStructure1(iDataSet).Xtr];
W = pinv(XtildaTraining) * dataStructure1(iDataSet).Ytr;

%Calculate the y result and check the result
for indexDataSet = 1:50
  %Memorise how many wrong classifications we have for the current data set
  wrongTesting = 0;
  for indexSample = 1:14
    %Create a new Xtilda for the current photo/sample
    XtildaTesting = [1, dataStructure1(indexDataSet).Xte([indexSample],:)];
    %Save the result of the multiplication
    yResult(indexDataSet, indexSample) = XtildaTesting * W;
    %If the yResult is bigger than 15 assign it to class 30
    if XtildaTesting * W > 15
      predictedClassTesting(indexDataSet, indexSample) = 30;
      %Count the mistake if missclassified
      if predictedClassTesting(indexDataSet, indexSample) ~= dataStructure1(indexDataSet).Yte(indexSample)
        wrongTesting = wrongTesting + 1;
      end
    else
      %Assign to class 1
      predictedClassTesting(indexDataSet, indexSample) = 1;
      %Count the mistake if missclassified
      if predictedClassTesting(indexDataSet, indexSample) ~= dataStructure1(indexDataSet).Yte(indexSample)
        wrongTesting = wrongTesting + 1;
      end
    end
  end
  %Calculate the error/accuracy rate
  errorRateTesting(indexDataSet) = (wrongTesting / 14) * 100;
  accuracyRateTesting(indexDataSet) = 100 - errorRateTesting(indexDataSet);
  fprintf('For data set number %d the classifier has a %f accuracy rate.\n', indexDataSet, accuracyRateTesting(indexDataSet))
end
toc;

%For each data set calculate the accuracy for each subject
for indexDataSet = 1:50
  %Memorise the number of wrong classificatons for subject 1 and 30
  wrong1 = 0;
  wrong2 = 0;
  %For the subject 1 see the classification result
  for indexSample = 1:7
    %Create a new Xtilda for the current photo/sample
    XtildaTesting = [1, dataStructure1(indexDataSet).Xte([indexSample],:)];
    %Save the result of the multiplication in a matrix
    yResultSubj(indexDataSet, indexSample) = XtildaTesting * W;
    if yResultSubj(indexDataSet, indexSample) > 15
      %Count the mistake if missclassified
      wrong1 = wrong1 + 1;
    end
  end
  %For the subject 30 see the classification result
  for indexSample = 8:14
    XtildaTesting = [1, dataStructure1(indexDataSet).Xte([indexSample],:)];
    yResultSubj(indexDataSet, indexSample) = XtildaTesting * W;
    if yResultSubj(indexDataSet, indexSample) <= 15
      %Count the mistake if missclassified
      wrong2 = wrong2 + 1;
    end
  end
  %Calculate the accuracy for each subject and memorise them in a matrix
  accRSubject1(indexDataSet) = 100 - (wrong1 / 7) * 100;
  accRSubject2(indexDataSet) = 100 - (wrong2 / 7) * 100;
end

%Plot the testing accuracies for subject 1 and subject 30
figure('Name', 'Testing accuracies for subject 1 and 30');
indexDataSet = 1:50;
f1 = accRSubject1;
f2 = accRSubject2;
plot(indexDataSet, f1, indexDataSet,f2);
xlabel('Data set number');
ylabel('Accuracy testing')

%Calculate the mean accuracy for testing
accMean = mean(accuracyRateTesting);

%Print the mean accuracy for the linear binary classifier
fprintf('The mean accuracy for the linear binary classifier is %f\n',  accMean)

%Plot the testing accuracies for the 50 data sets
x = 1:50;
y = accuracyRateTesting;
figure('Name','Testing graph');
plot(x, y)
xlabel('Data sample');
ylabel('Testing accuracy');
title('Testing accuracies');

%Show results for a random times
iNumberOfSample = randi([1  5]);
for indexSampleToShow = 1: iNumberOfSample
  %Choose a random sample
  iSample = randi([1  50]);
  figure('Name', 'Classification result on a random data set')
  %Show result of the classification
  ShowResult(dataStructure1(iSample).Xte, dataStructure1(iSample).Yte, predictedClassTesting(iSample,:),4)
end
