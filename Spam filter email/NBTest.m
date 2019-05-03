function [predictedLabel, accuracy] = NBTest(probabilityModel, testAttributeSet, validLabel, uniqueValuesParam, fname)
  %Get the unique values for classes
  numberClasses = size(unique(validLabel), 1);
  uniqueClasses = unique(validLabel);
  uniqueClasses = [uniqueClasses, zeros(size(uniqueClasses, 1) ,1)];
  %Calculate the frequency for each class
  for indexClass = 1: numberClasses
    uniqueClasses(indexClass, 2) = sum(uniqueValuesParam(1).ClassesFrequency(indexClass, [2:end]));
  end
  %Confusion matrix
  confusionMatrix = zeros(numberClasses, numberClasses);
  wrongResult = 0;
  %For each sample calculate the probability for each class
  for indexSample = 1:size(testAttributeSet, 1)
    maximum = realmin;
    for indexClass = 1: numberClasses
      probabilityClassTest(indexSample, indexClass) = 1;
      for indexFeature = 1: size(testAttributeSet, 2)
        indexOfFeatureValue = find(testAttributeSet(indexSample, indexFeature) == uniqueValuesParam(indexFeature).Matrix(:,1));
        %Zero conditional probability
        if probabilityModel(indexFeature, indexClass).ValueProbability(indexOfFeatureValue, 1) == 0
          frequencyClass = sum(uniqueValuesParam(indexFeature).ClassesFrequency(indexClass, :));
          priorEstimate = 1 / size(uniqueValuesParam(indexFeature).Matrix, 1);
          m = 1;
          probabilityModel(indexFeature, indexClass).ValueProbability(indexClass, indexOfFeatureValue + 1) = (0 + m * priorEstimate)/(frequencyClass + 1);
        end
        probabilityClassTest(indexSample, indexClass) = probabilityClassTest(indexSample, indexClass) * probabilityModel(indexFeature, indexClass).ValueProbability(indexOfFeatureValue, 1);
      end
      probabilityClassTest(indexSample, indexClass) = probabilityClassTest(indexSample, indexClass) * (uniqueClasses(indexClass, 2) / sum(uniqueClasses(:, 2)));
      %Apply MAP rule
      if probabilityClassTest(indexSample, indexClass) > maximum
        maximum = probabilityClassTest(indexSample, indexClass);
        finalClass = uniqueClasses(indexClass, 1);
      end
    end
    %Final class for the current sample
    predictedLabel(indexSample,1) = finalClass;
    %Count if wrong
    if finalClass ~= validLabel(indexSample, 1)
      wrongResult = wrongResult + 1;
    end
    %Confusion matrix
    idxClassFinal = find(uniqueClasses(:,1) == finalClass);
    idxClassValid = find(uniqueClasses(:,1) == validLabel(indexSample, 1));
    confusionMatrix(idxClassValid, idxClassFinal) = confusionMatrix(idxClassValid, idxClassFinal) + 1;
  end
  %Print confusion matrix and calculate accuracy
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
end
