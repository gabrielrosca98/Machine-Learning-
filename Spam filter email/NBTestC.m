function [predictedLabel, accuracy] = NBTest(valuesFeatures, testAttributeSet, validLabel, uniqueValuesParam, fname)
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
        %Calculate probability for continuous values
        probCurrentFeature = (1/(sqrt(2 * pi * valuesFeatures(indexFeature, indexClass).Std))) * exp(-(testAttributeSet(indexSample, indexFeature) - (valuesFeatures(indexFeature, indexClass).Mean).^2)/(2 * (valuesFeatures(indexFeature, indexClass).Std).^2));
        if probCurrentFeature > 0
          probabilityClassTest(indexSample, indexClass) = probabilityClassTest(indexSample, indexClass) * probCurrentFeature * 10;
        else
          %Zero conditional probability
          frequencyClass = sum(uniqueValuesParam(indexFeature).ClassesFrequency(indexClass, :));
          priorEstimate = 1 / size(uniqueValuesParam(indexFeature).Matrix, 1);
          m = 4;
          probCurrentFeature = (0 + m * priorEstimate)/(frequencyClass + 1);
          probabilityClassTest(indexSample, indexClass) = probabilityClassTest(indexSample, indexClass) * probCurrentFeature * 10;
        end
      end
      probabilityClassTest(indexSample, indexClass) = probabilityClassTest(indexSample, indexClass) * (uniqueClasses(indexClass, 2) / sum(uniqueClasses(:, 2)));
      %Apply MAP rule
      if probabilityClassTest(indexSample, indexClass) > maximum
        maximum = probabilityClassTest(indexSample, indexClass);
        finalClass = uniqueClasses(indexClass, 1);
      end
    end
    %Decide final class and count if wrong
    predictedLabel(indexSample,1) = finalClass;
    if finalClass ~= validLabel(indexSample, 1)
      wrongResult = wrongResult + 1;
    end
    %Calculate confusion matrix
    idxClassFinal = find(uniqueClasses(:,1) == finalClass);
    idxClassValid = find(uniqueClasses(:,1) == validLabel(indexSample, 1));
    confusionMatrix(idxClassValid, idxClassFinal) = confusionMatrix(idxClassValid, idxClassFinal) + 1;
  end
  %Calculate accuracy
  accuracy = 100 - (wrongResult / size(testAttributeSet, 1)) * 100;
  %Confusion matrix
  if(fname(1) ~= 's')
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
end
