close all;
clear all;
fname = input('Enter a filename to load data for training/testing: ','s');
load(fname);

% Put your NB training function below
%[probabilityModel] = NBTrain(AttributeSet, LabelSet); % NB training
% Put your NB test function below
%[predictLabel, accuracy] = NBTest( probabilityModel , testAttributeSet, validLabel); % NB test

  numberClasses = size(unique(LabelSet), 1);
  uniqueClasses = unique(LabelSet);
  uniqueClasses = [uniqueClasses, zeros(size(uniqueClasses, 1) ,1)];

  %Determine how many unique values for the features we have
  for indexFeature = 1:size(AttributeSet, 2)
    uniqueValuesParam(indexFeature).Matrix = unique(AttributeSet(:, indexFeature));
    uniqueValuesParam(indexFeature).Matrix = [uniqueValuesParam(indexFeature).Matrix, zeros(size(uniqueValuesParam(indexFeature).Matrix, 1), 1)];
    uniqueValuesParam(indexFeature).ClassesFrequency = [uniqueClasses, zeros(size(uniqueClasses, 1), size(uniqueValuesParam(indexFeature).Matrix, 1))];
  end
  %a = [uniqueClasses,zeros(3, size(uniqueValuesParam(1).Matrix, 1))];
  for indexSample = 1:size(AttributeSet, 1)
    for indexFeature = 1:size(AttributeSet, 2)
      featureRowIndex = find(uniqueValuesParam(indexFeature).Matrix(:, 1) == AttributeSet(indexSample,indexFeature));
      uniqueValuesParam(indexFeature).Matrix(featureRowIndex,2) = uniqueValuesParam(indexFeature).Matrix(featureRowIndex,2) + 1;
      for indexClass = 1:numberClasses
        uniqueValuesParam(indexFeature).ClassesFrequency(indexClass, featureRowIndex + 1) = uniqueValuesParam(indexFeature).ClassesFrequency(indexClass, featureRowIndex + 1) + sum(uniqueValuesParam(indexFeature).ClassesFrequency(indexClass, 1) ==  LabelSet(indexSample) & AttributeSet(indexSample, indexFeature) == uniqueValuesParam(indexFeature).Matrix(featureRowIndex, 1));
      end
    end
  end
  for indexFeature = 1:size(AttributeSet, 2)
    for indexClass = 1:numberClasses
      suma(indexFeature, indexClass) = sum(LabelSet == uniqueClasses(indexClass));
      for indexUniqueValueFeature = 1:size(uniqueValuesParam(indexFeature).Matrix, 1)
        probabilityModel(indexFeature, indexClass).ValueProbability(indexUniqueValueFeature, 1) = (uniqueValuesParam(indexFeature).ClassesFrequency(indexClass, indexUniqueValueFeature + 1) / suma(indexFeature, indexClass)) * 100;
      end
    end
  end

  for indexClass = 1: numberClasses
    uniqueClasses(indexClass, 2) = sum(uniqueValuesParam(1).ClassesFrequency(indexClass, :));
  end

  confusionMatrix = zeros(numberClasses, numberClasses);

  wrongResult = 0;
  for indexSample = 1:size(testAttributeSet, 1)
    maximum = realmin;
    for indexClass = 1: numberClasses
      probabilityClassTest(indexSample, indexClass) = 1;
      for indexFeature = 1: size(testAttributeSet, 2)
        indexOfFeatureValue = find(testAttributeSet(indexSample, indexFeature) == uniqueValuesParam(indexFeature).Matrix(:,1));
        if probabilityModel(indexFeature, indexClass).ValueProbability(indexOfFeatureValue, 1) == 0
          frequencyClass = sum(uniqueValuesParam(indexFeature).ClassesFrequency(indexClass, :));
          priorEstimate = 1 / size(uniqueValuesParam(indexFeature).Matrix, 1);
          m = 1;
          probabilityModel(indexFeature, indexClass).ValueProbability(indexClass, indexOfFeatureValue + 1) = (0 + m * priorEstimate)/(frequencyClass + 1);
        end
        probabilityClassTest(indexSample, indexClass) = probabilityClassTest(indexSample, indexClass) * probabilityModel(indexFeature, indexClass).ValueProbability(indexOfFeatureValue, 1);
      end
      probabilityClassTest(indexSample, indexClass) = probabilityClassTest(indexSample, indexClass) * (uniqueClasses(indexClass, 2) / sum(uniqueClasses(:, 2)));
      if probabilityClassTest(indexSample, indexClass) > maximum
        maximum = probabilityClassTest(indexSample, indexClass);
        finalClass = uniqueClasses(indexClass, 1);
      end
    end
    if finalClass ~= validLabel(indexSample, 1)
      wrongResult = wrongResult + 1;
    end
    idxClassFinal = find(uniqueClasses(:,1) == finalClass);
    idxClassValid = find(uniqueClasses(:,1) == validLabel(indexSample, 1));
    confusionMatrix(idxClassValid, idxClassFinal) = confusionMatrix(idxClassValid, idxClassFinal) + 1;
  end
  accuracy = 100 - (wrongResult / size(testAttributeSet, 1)) * 100;



fprintf('********************************************** \n');
fprintf('Overall Accuracy on Dataset %s: %f \n', fname, accuracy);
fprintf('********************************************** \n');

fprintf('\n');
fprintf('********************************************** \n');
fprintf('\t \tConfusion matrix \n');
fprintf('\t \tPredicted class \n');
fprintf('\t\t');
for indexClass = 1:numberClasses
  fprintf('%d\t', uniqueClasses(indexClass, 1));
end
fprintf('\n');
for indexClass = 1:numberClasses
  fprintf('Actual class:%d\t', uniqueClasses(indexClass, 1));
  for indexClass2 = 1:numberClasses
    fprintf('%d\t', confusionMatrix(indexClass, indexClass2));
  end
  fprintf('\n');
end
fprintf('********************************************** \n');
