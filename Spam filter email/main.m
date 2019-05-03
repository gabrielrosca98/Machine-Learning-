close all;
clear all;
fname = input('Enter a filename to load data for training/testing: ','s');
if fname(1) ~= 's'
  load(fname);
else
  data = importdata(fname);
end
charDecider = fname(3);
%For discrete values for features and non spam data
if charDecider ~= 'c' & fname(1) ~= 's'
  % Put your NB training function below
  [probabilityModel, uniqueValuesParam] = NBTrain(AttributeSet, LabelSet); % NB training
  % Put your NB test function below
  [predictLabel, accuracy] = NBTest(probabilityModel , testAttributeSet, validLabel, uniqueValuesParam, fname); % NB test
%For continuous values for features and non spam data
else if fname(3) == 'c' & fname(1) ~= 's'
  [valuesFeatures, uniqueValuesParam] = NBTrainC(AttributeSet, LabelSet); % NB training
  [predictLabel, accuracy] = NBTestC(valuesFeatures , testAttributeSet, validLabel, uniqueValuesParam, fname); % NB test
  else
    %Do the 10 cross-fold validation
    for indexCross = 1:10
      trainData1 = data([(indexCross - 1) * 181 + 1:(indexCross - 1) * 181 + 181], 1 : size(data, 2) - 1);
      labelTrainData1 = data([(indexCross - 1) * 181 + 1:(indexCross - 1) * 181 + 181], size(data, 2));
      testData1 = data([(indexCross) * 181 + 1:(indexCross - 1) * 278 + 1813], 1 : size(data, 2) - 1);
      labelTestData1 = data([(indexCross) * 181 + 1:(indexCross - 1) * 278 + 1813], size(data, 2));
      trainData2 = data([(indexCross - 1) * 278 + 1814:(indexCross) * 278 + 1814], 1 : size(data, 2) - 1);
      labelTrainData2 = data([(indexCross - 1) * 278 + 1814:(indexCross) * 278 + 1814], size(data, 2));
      testData2 = data([(indexCross) * 278 + 1814 + 1: end], 1 : size(data, 2) - 1);
      labelTestData2 = data([(indexCross) * 278 + 1814 + 1: end], size(data, 2));
      trainData = [trainData1; trainData2];
      testData = [testData1; testData2];
      labelTrainData = [labelTrainData1;labelTrainData2];
      labelTestData = [labelTestData1;labelTestData2];
      [valuesFeatures, uniqueValuesParam] = NBTrainC(trainData, labelTrainData);
      [predictLabel, accuracy] = NBTestC(valuesFeatures, testData,labelTestData, uniqueValuesParam, fname);
      accuracyFold(indexCross) = accuracy;
    end
    %Calculate mean accuracy and standard deviation
    meanAccuracy = mean(accuracyFold);
    standardDeviation  = std(accuracyFold);
    fprintf('The mean accuracy is %f and the standard deviation is %f\n', meanAccuracy, standardDeviation);
    figure('Name', 'Errorbar')
    indexCross = 1:10;
    standardDv = repmat(standardDeviation, 1, 10);
    errorbar(accuracyFold, standardDv);
    xlabel('Index cross');
    ylabel('Testing accuracy');
  end
end
